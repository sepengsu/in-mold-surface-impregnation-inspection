"""Exact global cleanlab selection from batch-returned scores stored on disk."""

import importlib.metadata
from pathlib import Path

import numpy as np

from .disk import DiskArray, create_npy, write_slice
from .protocol import read_json, write_json

AUDIT_BATCH_IMAGES = 8


def _pixel_batches(labels, probabilities, valid, chunk_pixels):
    pixels_per_tile = int(np.prod(labels.shape[1:]))
    classes = probabilities.shape[1]
    for start in range(0, labels.size, chunk_pixels):
        end = min(start + chunk_pixels, labels.size)
        first, last = start // pixels_per_tile, (end - 1) // pixels_per_tile + 1
        offset = start - first * pixels_per_tile
        targets = np.asarray(labels[first:last]).reshape(-1)[offset:offset + end - start].astype(int)
        predicted = np.moveaxis(probabilities[first:last], 0, 1).reshape(classes, -1).T[offset:offset + end - start]
        usable = (np.ones(end - start, bool) if valid is None else
                  np.asarray(valid[first:last], dtype=bool).reshape(-1)[offset:offset + end - start].copy())
        yield start, end, targets, predicted, usable


def _score_cutoff(scores, top, chunk_pixels, monitor=None):
    """Exact IEEE float radix selection: four passes for float32, eight for float64.

    NaN marks invalid pixels. Only a chunk and 256 counters are kept in RAM.
    Byte histograms select exact floating-point bit patterns, without quantizing.
    """
    if scores.dtype not in (np.dtype('float32'), np.dtype('float64')) or not 0 < top <= scores.size or chunk_pixels < 1:
        raise ValueError('Selection requires floating scores, a valid rank and positive chunk size.')
    flat = scores.reshape(-1)
    bits = scores.dtype.itemsize * 8
    uint = np.dtype('uint32' if bits == 32 else 'uint64')
    prefix, prefix_mask, rank = 0, 0, top - 1
    for shift in range(bits - 8, -1, -8):
        histogram = np.zeros(256, dtype=np.int64)
        for start in range(0, flat.size, chunk_pixels):
            values = flat[start:start + chunk_pixels]
            values = values[np.isfinite(values)]
            keys = values.view(uint) & np.array((1 << (bits - 1)) - 1, dtype=uint)
            if prefix_mask:
                keys = keys[(keys & np.array(prefix_mask, dtype=uint)) == np.array(prefix, dtype=uint)]
            bins = ((keys >> shift) & 255).astype(np.intp)
            histogram += np.bincount(bins, minlength=256)
            if monitor:
                monitor.record(f'cutoff_byte_{shift}', pixels=min(start + chunk_pixels, flat.size), chunk=chunk_pixels)
        cumulative = histogram.cumsum()
        if rank >= int(cumulative[-1]):
            raise ValueError('Rank exceeds the number of valid scores.')
        digit = int(np.searchsorted(cumulative, rank + 1))
        rank -= int(cumulative[digit - 1]) if digit else 0
        prefix |= digit << shift
        prefix_mask |= 255 << shift
    cutoff = np.array([prefix], dtype=uint).view(scores.dtype)[0]
    return cutoff, rank + 1


def find_label_issues_disk(labels, probabilities, destination, images, *,
                           batch_images=AUDIT_BATCH_IMAGES, pixel_chunk_size=None,
                           sort_chunk_size=None, valid=None, monitor=None, verbose=True):
    """A single Inspector sees all valid TRAIN pixels; no result-list APIs are used."""
    from cleanlab.experimental.label_issues_batched import LabelInspector

    if (labels.ndim != 3 or probabilities.ndim != 4 or probabilities.shape[0] != len(labels)
            or probabilities.shape[2:] != labels.shape[1:] or probabilities.shape[1] != 2 or labels.size == 0):
        raise ValueError('Expected aligned nonempty (N,H,W) labels and (N,2,H,W) probabilities.')
    if probabilities.dtype not in (np.dtype('float32'), np.dtype('float64')) or labels.dtype.kind not in 'biu':
        raise ValueError('Expected integer labels and float32/float64 probabilities.')
    pixels_per_tile = int(np.prod(labels.shape[1:]))
    chunk = batch_images * pixels_per_tile if pixel_chunk_size is None else pixel_chunk_size
    sort_chunk = chunk if sort_chunk_size is None else sort_chunk_size
    if (batch_images < 1 or chunk < 1 or sort_chunk < 1 or len(images) != len(labels)
            or len(set(images)) != len(images) or valid is not None and valid.shape != labels.shape):
        raise ValueError('Positive chunk size, aligned validity and unique image names in TRAIN order are required.')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'complete.json').exists():
        raise ValueError('A completed audit is immutable; use a new directory.')
    # Do not use segmentation.find_label_issues: its Inspector stores scores.
    inspector = LabelInspector(num_class=2, store_results=False, n_jobs=1, verbose=False)
    total_valid = 0
    if verbose:
        print(f'Global cleanlab thresholds: {len(labels)} tiles; at most {chunk:,} pixels per chunk', flush=True)
    for start, end, targets, predicted, usable in _pixel_batches(labels, probabilities, valid, chunk):
        targets, predicted = targets[usable], predicted[usable]
        if (np.any(targets < 0) or np.any(targets >= 2) or not np.isfinite(predicted).all()
                or np.any(predicted < 0) or np.any(predicted > 1)
                or not np.allclose(predicted.sum(axis=1), 1, atol=1e-5)):
            raise ValueError('Invalid class labels or probability distributions.')
        if len(targets):
            inspector.update_confident_thresholds(targets, predicted)
            total_valid += len(targets)
        if monitor:
            monitor.record('thresholds', tiles=(end + pixels_per_tile - 1) // pixels_per_tile,
                           pixels=total_valid, chunk=end - start)

    paths = {name: destination / (name + '.tmp') for name in
             ('label_quality.npy', 'top_selected.npy', 'ignore.npy', 'valid.npy')}
    for name, path in paths.items():
        create_npy(path, labels.shape, probabilities.dtype if name == 'label_quality.npy' else np.bool_)
    if verbose:
        print('Writing score_label_quality() batch returns directly to disk (store_results=False)...', flush=True)
    valid_done = 0
    for start, end, targets, predicted, usable in _pixel_batches(labels, probabilities, valid, chunk):
        score = np.full(end - start, np.nan, dtype=probabilities.dtype)
        if usable.any():
            returned = inspector.score_label_quality(targets[usable], predicted[usable], update_num_issues=True)
            if returned.dtype != probabilities.dtype or not np.isfinite(returned).all() or np.any(returned < 0) or np.any(returned > 1):
                raise ValueError('Invalid cleanlab score precision or values.')
            score[usable] = returned
        write_slice(paths['label_quality.npy'], slice(start, end), score, flat=True)
        write_slice(paths['valid.npy'], slice(start, end), usable, flat=True)
        valid_done += int(usable.sum())
        if monitor:
            monitor.record('scoring', tiles=(end + pixels_per_tile - 1) // pixels_per_tile,
                           pixels=valid_done, chunk=end - start)
    top = int(inspector.get_num_issues(silent=True)) if total_valid else 0
    if not 0 <= top <= total_valid:
        raise ValueError('Estimated issue count is outside [0, valid pixel count].')
    scores = DiskArray(paths['label_quality.npy'])
    if verbose:
        print(f'Selecting global lowest {top:,} of {total_valid:,} valid scores without RAM sorting...', flush=True)
    cutoff, ties_left = _score_cutoff(scores, top, sort_chunk, monitor) if top else (None, 0)
    cutoff_ties, cutoff_id, selected_count, flagged_count = ties_left, None, 0, 0
    flat_scores = scores.reshape(-1)
    for start, end, targets, predicted, usable in _pixel_batches(labels, probabilities, valid, chunk):
        quality = flat_scores[start:end]
        selected = np.zeros(end - start, bool) if cutoff is None else usable & (quality < cutoff)
        if ties_left:
            tied = np.flatnonzero(usable & (quality == cutoff))
            take = min(ties_left, len(tied))
            if take:
                selected[tied[:take]] = True
                cutoff_id = start + int(tied[take - 1])
                ties_left -= take
        selected_count += int(selected.sum())
        # Match segmentation downsample=1: remove agreement AFTER top-T; no refill.
        mask = selected & (predicted.argmax(axis=1) != targets)
        write_slice(paths['top_selected.npy'], slice(start, end), selected, flat=True)
        write_slice(paths['ignore.npy'], slice(start, end), mask, flat=True)
        flagged_count += int(mask.sum())
        if monitor:
            monitor.record('final_issues', tiles=(end + pixels_per_tile - 1) // pixels_per_tile,
                           pixels=end, chunk=end - start)
    if selected_count != top or ties_left:
        raise ValueError('Global selection did not cover exactly T pixels.')
    for name, temporary in paths.items():
        temporary.replace(destination / name)
    metadata = {
        'status': 'complete' if total_valid else 'no_valid_pixels',
        'method': 'cleanlab.experimental.label_issues_batched.LabelInspector',
        'cleanlab_version': importlib.metadata.version('cleanlab'),
        'store_results': False, 'score_method': 'self_confidence', 'estimation_method': 'off_diagonal',
        'score_definition': 'OOF probability of the original given label; lower means more suspect.',
        'score_file': 'label_quality.npy', 'mask_file': 'ignore.npy', 'valid_file': 'valid.npy',
        'pre_filter_file': 'top_selected.npy', 'score_dtype': str(probabilities.dtype), 'invalid_score': 'NaN',
        'shape': list(labels.shape), 'images': list(images), 'valid_pixels': total_valid,
        'global_pixel_id': 'image_index * H * W + y * W + x; invalid IDs are retained but never ranked.',
        'batch_images': (chunk + pixels_per_tile - 1) // pixels_per_tile,
        'batch_size': chunk * 2, 'pixel_chunk_size': chunk, 'sort_chunk_size': sort_chunk,
        'downsample': 1, 'n_jobs': 1,
        'confident_thresholds': inspector.get_confident_thresholds(silent=True).tolist() if total_valid else None,
        'estimated_issues_before_prediction_filter': top, 'flagged_pixels': flagged_count,
        'score_cutoff': float(cutoff) if cutoff is not None else None, 'cutoff_global_pixel_id': cutoff_id,
        'selected_ties_at_cutoff': cutoff_ties,
        'selection': 'Global lowest T scores, then remove prediction/label agreements without replenishment.',
        'selection_algorithm': f'{probabilities.dtype.itemsize} exact byte-histogram passes; short-lived file mappings.',
        'tie_break': 'Ascending TRAIN image index, then row-major pixel index.',
        'reference': 'cleanlab.segmentation.filter.find_label_issues(downsample=1), deterministic cutoff ties; not all generic filter_by modes.',
        'restart_policy': 'Incomplete passes restart from a new Inspector and freshly truncated .tmp arrays; no statistics are resumed.',
    }
    write_json(destination / 'scoring.json', metadata)
    return metadata


def load_image_audit(directory, image):
    directory = Path(directory)
    metadata = read_json(directory / 'scoring.json')
    try:
        index = metadata['images'].index(image)
    except ValueError as error:
        raise ValueError(f'Image is not in this TRAIN audit: {image}') from error
    scores, issues = DiskArray(directory / 'label_quality.npy'), DiskArray(directory / 'ignore.npy')
    shape = tuple(metadata['shape'])
    if scores.shape != shape or issues.shape != shape or str(scores.dtype) != metadata['score_dtype'] or issues.dtype != np.bool_:
        raise ValueError('Saved score/mask shape or dtype does not match scoring metadata.')
    return scores[index], issues[index]
