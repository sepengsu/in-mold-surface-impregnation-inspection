"""One deterministic owner per source surface pixel, shared by B and D."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from .disk import DiskArray, create_npy, write_slice
from .labels import OriginalLabels
from .protocol import write_json


def intersection(left, right):
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    if x0 >= x1 or y0 >= y1:
        return None
    return ((slice(y0-left[1], y1-left[1]), slice(x0-left[0], x1-left[0])),
            (slice(y0-right[1], y1-right[1]), slice(x0-right[0], x1-right[0])))


def earlier_overlaps(rows):
    groups, overlaps = defaultdict(list), [[] for _ in rows]
    for index, row in enumerate(rows):
        for previous in groups[row['task_id']]:
            slices = intersection(row['xyxy'], rows[previous]['xyxy'])
            if slices is not None:
                overlaps[index].append((previous, slices))
        groups[row['task_id']].append(index)
    return overlaps


def prepare_surface(data, rows, oof_directory, *, monitor=None):
    directory = Path(oof_directory)
    original = OriginalLabels(data)
    overlaps = earlier_overlaps(rows)
    shape = (len(rows), 256, 256)
    valid_path, unique_path = directory/'original_valid.npy.tmp', directory/'ownership_valid.npy.tmp'
    create_npy(valid_path, shape, bool)
    create_npy(unique_path, shape, bool)
    has_overlap = any(overlaps)
    probability = DiskArray(directory/'probabilities.npy')
    canonical_path = directory/'canonical_probabilities.npy.tmp'
    if has_overlap:
        create_npy(canonical_path, probability.shape, probability.dtype)
    unique_count = 0
    for index, row in enumerate(rows):
        _, target, valid = original.load(row)
        unique = valid.copy()
        predicted = probability[index] if has_overlap else None
        for previous, (current_slice, previous_slice) in overlaps[index]:
            _, previous_target, previous_valid = original.load(rows[previous])
            shared = valid[current_slice] & previous_valid[previous_slice]
            if np.any(shared & (target[current_slice] != previous_target[previous_slice])):
                raise ValueError('Overlapping tiles disagree on original valid labels; reconcile source GT first.')
            owned_here = unique[current_slice] & shared
            if has_overlap:
                source = probability[previous]
                for k in (0, 1):
                    predicted[k][current_slice][owned_here] = source[k][previous_slice][owned_here]
            unique[current_slice] &= ~shared
        write_slice(valid_path, index, valid)
        write_slice(unique_path, index, unique)
        if has_overlap:
            write_slice(canonical_path, index, predicted)
        unique_count += int(unique.sum())
        if monitor:
            monitor.record('surface_validity', tiles=index+1, pixels=unique_count, chunk=65536)
    valid_path.replace(directory/'original_valid.npy')
    unique_path.replace(directory/'ownership_valid.npy')
    if has_overlap:
        canonical_path.replace(directory/'canonical_probabilities.npy')
    metadata = {'overlap_detected': has_overlap, 'unique_valid_pixels': unique_count,
                'policy': 'First valid TRAIN tile owns each original source pixel; owner OOF probabilities and final candidates are broadcast to overlapping tiles.',
                'probability_file': 'canonical_probabilities.npy' if has_overlap else 'probabilities.npy',
                'original_valid_file': 'original_valid.npy', 'ranking_valid_file': 'ownership_valid.npy',
                'encoding': original.encoding,
                'tiles': [{'index': i, 'image': r['image'], 'task_id': r['task_id'], 'case': r['group'],
                           'xyxy_exclusive': r['xyxy'], 'global_pixel_offset': i*65536} for i,r in enumerate(rows)]}
    write_json(directory/'surface.json', metadata)
    return metadata


def broadcast_shared_issues(rows, oof_directory, issues_directory):
    """Keep ranking unique, but use identical owner candidates on every occurrence."""
    source, destination = Path(oof_directory), Path(issues_directory)
    overlaps = earlier_overlaps(rows)
    for filename in ('ignore.npy', 'label_quality.npy'):
        array = DiskArray(destination/filename)
        for index, neighbors in enumerate(overlaps):
            if not neighbors:
                continue
            current = array[index]
            assigned = np.zeros(current.shape, bool)
            current_valid = DiskArray(source/'original_valid.npy')[index]
            for previous, (here, there) in neighbors:
                owner_valid = DiskArray(source/'original_valid.npy')[previous][there]
                usable = current_valid[here] & owner_valid & ~assigned[here]
                prior = array[previous][there]
                current[here][usable] = prior[usable]
                assigned[here] |= usable
            write_slice(destination/filename, index, current)
