"""One-shot D labels from the shared final CL mask and TRAIN-only OOF probabilities."""

from pathlib import Path

import numpy as np

from .disk import DiskArray, atomic_npy
from .labels import OriginalLabels, decode_mask
from .protocol import fingerprint, read_json, seal, sha256, verified, write_csv, write_json

POLICY = 'one_shot_oof_argmax_on_shared_final_issues'


def auto_correct_tile(original_mask, p_defect, issue_mask, valid_mask, encoding, *, p_normal=None):
    target, valid = decode_mask(original_mask, encoding, valid_mask)
    p_normal = 1-p_defect if p_normal is None else p_normal
    if p_defect.shape != target.shape or p_normal.shape != target.shape or issue_mask.shape != target.shape:
        raise ValueError('Correction arrays must align and have finite valid OOF probabilities.')
    if (not np.isfinite(p_defect[valid]).all() or not np.isfinite(p_normal[valid]).all()
            or np.any(p_defect[valid]<0) or np.any(p_defect[valid]>1)
            or np.any(p_normal[valid]<0) or np.any(p_normal[valid]>1)
            or not np.allclose((p_defect+p_normal)[valid],1,atol=1e-5)):
        raise ValueError('Correction arrays must align and have finite valid OOF probabilities.')
    predicted = (p_defect > p_normal).astype(np.uint8)  # Exact binary argmax; class 0 wins ties.
    changed = valid & issue_mask.astype(bool) & (predicted != target)
    removed, added = changed & (target == 1), changed & (target == 0)
    corrected = original_mask.copy()
    corrected[removed] = encoding.get('normal_value', 0)
    corrected[added] = encoding.get('defect_value', 1)
    if not np.array_equal(corrected[~changed], original_mask[~changed]):
        raise ValueError('Correction modified a non-candidate or invalid pixel.')
    return corrected, removed, added


def generate_auto(args, output, rows, monitor=None):
    destination = output/'auto'
    if verified(destination, ('manifest.json', 'tiles.csv', 'summary.json')):
        return
    if not verified(output/'issues', ('ignore.npy', 'scoring.json')):
        raise ValueError('Generate the shared final CL issues first.')
    destination.mkdir(parents=True, exist_ok=True)
    source = OriginalLabels(args.data)
    surface = read_json(output/'oof/surface.json')
    probability_path = output/'oof'/surface['probability_file']
    probability, issues = DiskArray(probability_path), DiskArray(output/'issues/ignore.npy')
    hashes = {'original_manifest_sha256': sha256(args.data/'manifest.json'),
              'oof_probabilities_sha256': sha256(probability_path),
              'oof_provenance_sha256': sha256(output/'oof/provenance.json'),
              'final_issue_mask_sha256': sha256(output/'issues/ignore.npy'),
              'cl_policy_sha256': fingerprint(read_json(output/'issues/scoring.json'))}
    entries, records = [], []
    for index, row in enumerate(rows):
        raw, before, valid = source.load(row)
        corrected, removed, added = auto_correct_tile(raw, probability[index, 1], issues[index], valid,
                                                       source.encoding, p_normal=probability[index, 0])
        after, _ = decode_mask(corrected, source.encoding, valid)
        counts = {'changed_1_to_0': int(removed.sum()), 'changed_0_to_1': int(added.sum()),
                  'defect_before': int((before.astype(bool) & valid).sum()),
                  'defect_after': int((after.astype(bool) & valid).sum()), 'valid_pixels': int(valid.sum())}
        if counts['defect_after'] != counts['defect_before'] - counts['changed_1_to_0'] + counts['changed_0_to_1']:
            raise ValueError('Auto-correction pixel accounting failed.')
        relative = f'tiles/tile_{index:05d}'
        paths = {'mask': relative+'_auto.npy', 'changed_1_to_0_mask': relative+'_1_to_0.npy',
                 'changed_0_to_1_mask': relative+'_0_to_1.npy', 'valid_mask': relative+'_valid.npy'}
        for key, array in zip(paths, (corrected, removed, added, valid)):
            atomic_npy(destination/paths[key], array)
        entries.append({'index': index, 'image': row['image'], 'original_mask': row['mask'],
                        'original_mask_sha256': sha256(args.data/row['mask']), **paths,
                        'auto_mask_sha256': sha256(destination/paths['mask'])})
        records.append({'tile_id': row['image'], 'case_id': row['group'], 'task_id': row['task_id'],
                        'xyxy_exclusive': ','.join(map(str,row['xyxy'])),
                        'cl_issue_pixels': int((issues[index] & valid).sum()), **counts,
                        'changed_total': counts['changed_1_to_0']+counts['changed_0_to_1'],
                        'defect_ratio_before': counts['defect_before']/counts['valid_pixels'] if counts['valid_pixels'] else 0.,
                        'defect_ratio_after': counts['defect_after']/counts['valid_pixels'] if counts['valid_pixels'] else 0.,
                        'auto_correction_policy': POLICY, **hashes})
        if monitor:
            monitor.record('auto_masks', tiles=index+1, pixels=(index+1)*65536, chunk=65536)
    manifest = {'label_kind': 'auto-corrected training label', 'policy': POLICY, 'one_shot': True,
                'additional_confidence_threshold': None, 'morphology': None, 'minimum_area': None,
                'encoding': source.encoding, 'valid_encoding': 'boolean', 'change_mask_encoding': 'boolean',
                'classes': {'0': 'Normal (background)', '1': 'Defect (existing Exposure foreground)'},
                'overlap_policy': surface['policy'], 'hashes': hashes, 'tiles': entries}
    write_json(destination/'manifest.json', manifest)
    write_csv(destination/'tiles.csv', records)
    write_json(destination/'summary.json', {'label_kind': manifest['label_kind'], 'policy': POLICY,
        'tiles': len(rows), **{key:sum(r[key] for r in records) for key in
        ('valid_pixels','cl_issue_pixels','changed_1_to_0','changed_0_to_1','changed_total','defect_before','defect_after')}})
    seal(destination, sorted(str(p.relative_to(destination)) for p in destination.rglob('*')
                             if p.is_file() and p.name not in ('complete.json','memory_log.csv','error.json') and p.suffix!='.tmp'))
