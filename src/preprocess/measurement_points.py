"""Export three spaced ROI-midline measurement tiles per annotated image."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .roi_centered_tiling import plan_roi_centered_tiling
from .tile_dataset import _write_png
from .visualize_roi_points import read_gray
from ..segmentation.label_studio import task_brush_alpha

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT/'data/raw/rectangle_roi_tiles_256'
DEFAULT_OUTPUT = PROJECT/'data/label_studio_exports/2026-10-01/rectangle_measurement_points_547_256'


def export(source, output, spacing=547):
    source, output = Path(source).resolve(), Path(output).resolve()
    raw = (source/'manifest.json').read_bytes()
    previous = json.loads(raw)
    if previous['summary']['settings']['alignment'] != 'roi_centerline':
        raise ValueError('A source manifest with saved ROI geometry is required.')
    matches = json.loads(Path(previous['summary']['source_matches']).read_text('utf-8'))
    label_raw = Path(previous['summary']['source_export']).read_bytes()
    if hashlib.sha256(label_raw).hexdigest() != matches['source_export_sha256']:
        raise ValueError('Label export hash changed.')
    tasks = {r['id']: r for r in json.loads(label_raw)}
    matched = {r['task_id']: r for r in matches['matches']}
    if len(tasks) != len(json.loads(label_raw)) or len(matched) != len(matches['matches']):
        raise ValueError('Duplicate task IDs in source data.')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('images', 'masks'):
        (output/name).mkdir()
    (output/'RUNNING.json').write_text(json.dumps({'source': str(source), 'spacing': spacing}), 'utf-8')
    images = []
    for record in previous['images']:
        task = record['task_id']
        path = Path(record['source_image'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['source_sha256']:
            raise ValueError(f'Source ROI changed: {task}')
        image = read_gray(path)
        raw_path = Path(matched[task]['local_original_image'])
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != matched[task]['original_sha256']:
            raise ValueError(f'Raw source image changed: {task}')
        original = read_gray(raw_path)
        alpha = task_brush_alpha(tasks[task])
        if alpha.shape != image.shape or original.shape != image.shape:
            raise ValueError(f'Source/mask shape differs: {task}')
        geometry = record['roi_geometry']
        plan = plan_roi_centered_tiling(image, geometry['top_line'], geometry['bottom_line'],
                                       selection='central_three', point_spacing=spacing)
        if len(plan.positions) != 3:
            raise ValueError(f'Task {task}: fixed measurement points invalid: {plan.rejected}')
        tiles = []
        for x, y in sorted(plan.positions):
            point = plan.central_points[(x, y)]
            patch, mask = image[y:y+256, x:x+256], alpha[y:y+256, x:x+256]
            np.testing.assert_array_equal(patch, original[y:y+256, x:x+256])
            if patch.shape != (256, 256) or (patch == 0).any():
                raise ValueError(f'Invalid pixels at task {task}, {point}')
            filename = f'task_{task}_{point}_x{x:04d}_y{y:04d}.png'
            _write_png(output/'images'/filename, patch)
            _write_png(output/'masks'/filename, mask)
            np.testing.assert_array_equal(read_gray(output/'images'/filename), patch)
            np.testing.assert_array_equal(read_gray(output/'masks'/filename), mask)
            tiles.append({'xyxy_exclusive': [x, y, x+256, y+256], 'image': f'images/{filename}',
                          'mask': f'masks/{filename}', 'label_pixels': int(np.count_nonzero(mask)),
                          'grid_index': list(plan.grid_indices[(x, y)]), 'measurement_point': point})
        assert [t['measurement_point'] for t in tiles] == ['P1', 'P2', 'P3']
        assert all(tiles[i+1]['xyxy_exclusive'][0]-tiles[i]['xyxy_exclusive'][0] == spacing for i in (0, 1))
        images.append({**record, 'tiles': tiles, 'candidate_tiles': 3, 'accepted_tiles': 3,
                       'rejected': plan.rejected, 'grid_bounds_xyxy_exclusive': plan.grid_bounds})
    summary = {**previous['summary'], 'settings': {**previous['summary']['settings'],
        'selection': 'measurement_three', 'purpose': 'measurement', 'point_spacing_px': spacing,
        'tile_gap_px': spacing-256, 'origin': 'P2 at horizontal image center; vertical ROI midline',
        'layout_basis': 'User-approved 2/5/8 template, nominal 32.5 mm spacing',
        'estimated_mm_per_pixel': .0055*(300-25.4)/25.4, 'empirically_calibrated': False,
        'image_point_order': 'P1 left, P2 center, P3 right for both directions',
        'physical_point_ids_for_000_template': {'P1': 2, 'P2': 5, 'P3': 8},
        'physical_registration_verified': False,
        'orientation_note': '180 images retain image-left/right labels; physical point IDs are not registered.'},
        'source_images': len(images), 'candidate_tiles': len(images)*3, 'accepted_tiles': len(images)*3,
        'rejected_tiles': 0, 'rejection_reasons': {},
        'tiles_by_position': {p: sum(r['position'] == p for r in images)*3 for p in ('000', '180')},
        'tiles_per_image_min': 3, 'tiles_per_image_max': 3, 'images_without_tiles': 0,
        'image_png_count': len(images)*3, 'mask_png_count': len(images)*3,
        'count_only': False, 'central_three_tiles': len(images)*3,
        'central_three_000_tiles': sum(r['position'] == '000' for r in images)*3,
        'source_geometry_manifest': str(source/'manifest.json'),
        'source_geometry_manifest_sha256': hashlib.sha256(raw).hexdigest()}
    manifest = {'summary': summary, 'images': images}
    for name, value in [('manifest.json', manifest), ('summary.json', summary), ('verification.json', {
        'verified_images': len(images), 'verified_pairs': len(images)*3,
        'raw_roi_tile_pixels_equal': True, 'decoded_label_pixels_equal': True,
        'all_three_points_valid': True, 'black_pixels': 0, 'horizontal_center_spacing_px': spacing,
        'physical_registration_verified': False})]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')
    (output/'RUNNING.json').unlink()
    print(json.dumps({'output': str(output), 'sources': len(images), 'tiles': len(images)*3,
                      'center_spacing_px': spacing}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--spacing', type=int, default=547)
    args = parser.parse_args(argv)
    export(args.source, args.output, args.spacing)


if __name__ == '__main__':
    main()
