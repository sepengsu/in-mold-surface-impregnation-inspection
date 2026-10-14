"""Explicit mask encodings, original validity, and approved human revisions."""

from pathlib import Path

import numpy as np

from src.training.data import read_gray
from .disk import mapped
from .protocol import read_json, sha256


def read_mask(path):
    path = Path(path)
    if path.suffix == '.npy':
        with mapped(path) as array:
            return np.array(array, copy=True)
    return read_gray(path)


def decode_mask(raw, encoding, valid=None):
    if raw.dtype != np.uint8 or raw.ndim != 2:
        raise ValueError('Mask must be a two-dimensional uint8 array.')
    normal, defect, ignore = (encoding.get(k) for k in ('normal_value', 'defect_value', 'ignore_value'))
    if ignore is not None and ignore in (normal, defect):
        raise ValueError('Ignore and Normal/Defect encodings must be distinct.')
    effective = np.ones(raw.shape, bool) if valid is None else np.asarray(valid, dtype=bool).copy()
    if effective.shape != raw.shape:
        raise ValueError('Mask and validity shapes differ.')
    if ignore is not None:
        effective &= raw != ignore
    if encoding.get('rule') == 'positive_is_defect':
        binary = raw > 0
    else:
        if normal is None or defect is None or normal == defect:
            raise ValueError('Explicit normal_value and defect_value are required.')
        if np.any(effective & (raw != normal) & (raw != defect)):
            raise ValueError('Unexpected mask value in a valid pixel.')
        binary = raw == defect
    return binary.astype(np.uint8), effective


class OriginalLabels:
    def __init__(self, root):
        self.root = Path(root).resolve()
        manifest = read_json(self.root / 'manifest.json')
        self.tiles = {tile['image']: tile for source in manifest['images'] for tile in source['tiles']}
        self.encoding = manifest['summary'].get('mask_encoding')
        if self.encoding is None:
            # Legacy masks encode alpha>0. Infer storage values with one tile
            # at a time and 256 counters; never infer an Ignore value.
            observed = np.zeros(256, dtype=bool)
            for tile in self.tiles.values():
                raw = read_mask(self.root/tile['mask'])
                observed[np.unique(raw)] = True
            foreground = np.flatnonzero(observed)[np.flatnonzero(observed)>0]
            defect = int(foreground[-1]) if foreground.size else 1
            self.encoding = {
                'rule':'positive_is_defect', 'normal_value':0, 'defect_value':defect, 'ignore_value':None,
                'observed_storage_values':np.flatnonzero(observed).tolist(),
                'new_foreground_storage':'Maximum observed positive alpha; 1 for an entirely empty dataset.',
                'meaning':'Existing alpha > 0 foreground; 255 is Defect, not Ignore unless explicitly declared.',
            }

    def load(self, row):
        raw = read_mask(self.root / row['mask'])
        tile = self.tiles.get(row['image'], {})
        valid_path = row.get('valid_mask') or tile.get('valid_mask')
        valid = read_mask(self.root / valid_path) > 0 if valid_path else None
        binary, valid = decode_mask(raw, self.encoding, valid)
        return raw, binary, valid

    def validity_fingerprint(self, rows):
        files = {}
        for row in rows:
            value = row.get('valid_mask') or self.tiles.get(row['image'], {}).get('valid_mask')
            if value:
                path = (self.root / value).resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError('Validity path escapes the dataset.')
                files[value] = sha256(path)
        return {'encoding': self.encoding, 'valid_mask_sha256': files}


def approved_revision(path, rows, manifest_sha256, *, require_all=False):
    """Only explicitly approved files are eligible; inputs are read-only."""
    if path is None:
        return None
    path = Path(path).resolve()
    raw = read_json(path)
    if raw.get('approved') is not True or raw.get('dataset_manifest_sha256') != manifest_sha256:
        raise ValueError('Revision needs approved=true and the matching dataset_manifest_sha256.')
    allowed = {r['image'] for r in rows}
    files, index = {}, {}
    for tile in raw.get('tiles', []):
        image = tile['image']
        if image not in allowed or image in index or tile.get('approved', True) is not True:
            raise ValueError('Revision contains unapproved, duplicate or non-target-split images.')
        target = (path.parent / tile['mask']).resolve()
        digest = sha256(target)
        if digest != tile.get('sha256'):
            raise ValueError(f'Approved revision hash mismatch: {image}')
        files[str(target)], index[image] = digest, str(target)
    if not index or require_all and set(index) != allowed:
        raise ValueError('Approved revision does not cover the required images.')
    return {'manifest': str(path), 'manifest_sha256': sha256(path), 'files': files, 'masks': index,
            'encoding': raw.get('mask_encoding', {'normal_value': 0, 'defect_value': 1, 'ignore_value': None}),
            'reviewer': raw.get('reviewer'), 'unreviewed_policy': 'Original labels retained.'}


def verify_revision(revision):
    if revision is None:
        return
    if sha256(revision['manifest']) != revision['manifest_sha256']:
        raise ValueError('Approved revision manifest changed.')
    for path, digest in revision['files'].items():
        if sha256(path) != digest:
            raise ValueError(f'Approved revision mask changed: {path}')


def revised_label(original, row, revision):
    raw, binary, valid = original.load(row)
    if revision is not None and row['image'] in revision['masks']:
        reviewed = read_mask(revision['masks'][row['image']])
        target, reviewed_valid = decode_mask(reviewed, revision['encoding'])
        if target.shape != binary.shape or np.any(valid & ~reviewed_valid):
            raise ValueError('Reviewed labels must preserve all original valid evaluation pixels.')
        binary[valid] = target[valid]
    return raw, binary, valid
