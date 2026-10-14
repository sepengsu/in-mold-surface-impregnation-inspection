"""Append saved Project 12 GT for Case16 specimen02 to a separate dataset version."""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import html
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from src.Namrc.protocol import (load_frozen_splits, read_json, select_central_splits,
                                    sha256, validate_splits, write_json)
from src.inference.build_point_viewer import build
from src.inference.compare_label_revision import csv_file, font
from src.preprocess.rectangle_dataset import plan_rectangle
from src.segmentation.label_studio import task_brush_alpha
from src.training.data import describe_split, load_tiles, read_gray

ROOT = Path(__file__).resolve().parents[2]
TASK, PROJECT, SHOT = 800, 12, 1155


def snapshot(directory):
    return {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*') if p.is_file()}


def rebase(value, parent, output):
    if isinstance(value, dict):
        return {key: rebase(item, parent, output) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase(item, parent, output) for item in value]
    if isinstance(value, str) and (value == str(parent) or value.startswith(str(parent) + '\\')
                                  or value.startswith(str(parent) + '/')):
        return str(output) + value[len(str(parent)):]
    return value


def png(path, pixels):
    Image.fromarray(pixels).save(path)
    if not np.array_equal(read_gray(path), pixels):
        raise ValueError(f'Saved PNG pixels differ: {path}')


def render_capture(output, source, points):
    raw = read_gray(source['original_image'])
    canvas = Image.new('RGB', (1780, 1300), '#f5f7fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 24), 'Case16 · 고정측 · 시편02 · 샷 #1155 · Project12 / Task800',
              fill='#203146', font=font(32, True))
    draw.text((40, 76), 'GT 함침율 · 원촬영 000 그대로 · P1 왼쪽 / P2 중앙 / P3 오른쪽',
              fill='#4b5f76', font=font(23))
    left, top, size = 70, 166, 1024
    scale = size / raw.shape[1]
    canvas.paste(Image.fromarray(raw).convert('RGB').resize((size, size)), (left, top))
    rx, ry, rx1, ry1 = source['roi_geometry']['xyxy_exclusive']
    draw.rectangle((left+rx*scale, top+ry*scale, left+(rx1-1)*scale, top+(ry1-1)*scale),
                   outline='#60d092', width=2)
    native_panels = []
    for index, (row, color) in enumerate(zip(points, ('#ffcd59', '#40d7e5', '#b58cff'))):
        x, y, x1, y1 = row['xyxy']
        label = f"{row['point']}  GT {row['gt_impregnation_pct']:.4f}%"
        draw.rectangle((left+x*scale, top+y*scale, left+(x1-1)*scale, top+(y1-1)*scale),
                       outline=color, width=4)
        draw.text((left+x*scale, top+y*scale-32), label, fill=color, font=font(20, True),
                  stroke_width=1, stroke_fill='#182a3c')
        row_top, panel_y = 154+338*index, 217+338*index
        draw.text((1162, row_top), f'{label}  (x={x}, y={y})', fill='#203146', font=font(22, True))
        for px, heading, pixels in ((1162, '원본 크롭', read_gray(output/row['image'])),
                                   (1450, 'Exposure GT', (read_gray(output/row['gt_mask']) > 0).astype(np.uint8)*255)):
            draw.text((px, row_top+33), heading, fill='#4b5f76', font=font(18))
            panel = np.asarray(Image.fromarray(pixels).convert('RGB'))
            canvas.paste(Image.fromarray(panel), (px, panel_y))
            native_panels.append((px, panel_y, panel))
    draw.text((70, 1236), '초록 테두리: 기존 직사각형 ROI · 측정 간격 547 px · 각 256×256 px',
              fill='#4b5f76', font=font(21))
    draw.text((70, 1270), 'GT 함침율(%) = 100 × (1 − alpha>0 픽셀 수 / 65,536)',
              fill='#4b5f76', font=font(20))
    path = output/'analysis/task_800.png'
    canvas.save(path)
    with Image.open(path) as saved:
        pixels = np.asarray(saved)
        assert all(np.array_equal(pixels[y:y+256, x:x+256], panel) for x, y, panel in native_panels)
    return {'task_id': TASK, 'case': 'case16', 'specimen': '02', 'direction': '000',
            'split': 'train', 'rotation_deg': 0, 'title': 'Case16 · 고정측 · 시편02 · 샷 #1155',
            'figure': path.name, 'original_image': source['original_image'],
            'original_sha256': source['original_sha256'], 'raw': 'raw/task_800.png',
            'mask': 'mask/task_800.png', 'mask_alpha': 'mask_alpha/task_800.png',
            'points': [{k: row[k] for k in ('point', 'xyxy', 'gt_impregnation_pct')} for row in points]}


def measurement_page(output, rows):
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in
        (row['case_number'], row['side'], row['specimen_number'], row['shot_number'],
         row['project_id'], row['task_id'], row['split'],
         *(f"{row[f'{point}_gt_pct']:.4f}" for point in ('P1', 'P2', 'P3')),
         f"{row['mean_gt_pct']:.4f}")) + '</tr>' for row in rows)
    headings = ('Case', '측', '시편', '샷', 'Project', 'Task', '분할', 'P1 GT (%)', 'P2 GT (%)', 'P3 GT (%)', '평균 (%)')
    page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>changed_v3 GT 함침율</title>'
            '<style>body{font:15px "Malgun Gothic";margin:24px;color:#203146}table{border-collapse:collapse}'
            'td,th{border:1px solid #d4deea;padding:8px 12px;text-align:right}</style>'
            '<h1>changed_v3 · 72장 GT 함침율</h1><p><a href="index.html">원본과 P1/P2/P3 위치</a> · '
            '<a href="task_800.png">Case16 시편02 측정 그림</a> · '
            '<a href="../specimen_measurements.csv">측정값 CSV</a></p>'
            '<p>GT 함침율(%) = 100 × (1 − Exposure(alpha&gt;0) 픽셀 수 / 65,536). '
            '현재 영상의 왼쪽 P1, 중앙 P2, 오른쪽 P3.</p><table><thead><tr>'
            + ''.join(f'<th>{heading}</th>' for heading in headings) + '</tr></thead><tbody>' + body + '</tbody></table></html>')
    (output/'analysis/measurements.html').write_text(page, encoding='utf-8')


def append_capture(parent, output, download, geometry):
    parent, output, download, geometry = (Path(p).resolve() for p in (parent, output, download, geometry))
    if output.exists() or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('Choose a new output outside the parent dataset; existing data is never overwritten.')
    full, _ = load_frozen_splits(parent, parent/'splits.json')
    parent_manifest = read_json(parent/'manifest.json')
    if len(parent_manifest['images']) != 71 or {k: len(v) for k, v in full.items()} != {'train': 1035, 'val': 187, 'test': 251}:
        raise ValueError('Expected the complete frozen 71-image changed_v2 parent.')
    if any(s['task_id'] == TASK for s in parent_manifest['images']):
        raise ValueError('Task800 already exists in the parent dataset.')
    membership = {r['group']: name for name, rows in full.items() for r in rows}
    if membership.get('case16') != 'train':
        raise ValueError('Expected frozen Case16 TRAIN membership.')
    meta, exported, geo = read_json(download/'metadata.json'), read_json(download/'annotation.json'), read_json(geometry)
    if (meta['project_id'], meta['task_id'], meta['label'], exported['task_id']) != (PROJECT, TASK, 'Exposure', TASK):
        raise ValueError('Expected Project12 / Task800 Exposure annotation.')
    annotation = exported['annotation']
    if (annotation['id'] != meta['annotation_id'] or annotation.get('was_cancelled', False)
            or annotation['updated_at'] != meta['annotation_updated_at']):
        raise ValueError('Saved annotation identity/version differs from metadata.')
    if (geo['case_number'], geo['side'], geo['specimen'], geo['shot'], geo['position']) != (16, '고정측', 2, SHOT, '000'):
        raise ValueError('Expected Case16 fixed-side specimen02 / shot1155 / original000 geometry.')
    original = Path(geo['source_image']).resolve()
    if original.name + '_roi.png' not in meta['image_reference'] or '/02/000/10000us/' not in meta['image_reference']:
        raise ValueError('Annotation image reference does not match the original capture.')
    if sha256(original) != geo['source_sha256']:
        raise ValueError('Original image changed since the fixed measurement geometry.')
    raw = read_gray(original)
    with Image.open(download/'image.png') as image:
        visible = np.asarray(image.convert('L'))
    alpha = task_brush_alpha({'id': TASK, 'annotations': [annotation]})
    if raw.shape != visible.shape or alpha.shape != raw.shape:
        raise ValueError('Original, annotation image and alpha dimensions differ.')
    support = visible > 0
    if not np.array_equal(raw[support], visible[support]):
        raise ValueError('Annotated visible image pixels differ from the original photograph.')
    if not np.array_equal(alpha, read_gray(download/'mask_alpha.png')) or int(np.count_nonzero(alpha)) != meta['exposure_pixels']:
        raise ValueError('Decoded annotation differs from the saved alpha/count.')
    if not np.array_equal(read_gray(download/'mask.png') > 0, alpha > 0) or np.any((alpha > 0) & ~support):
        raise ValueError('Saved binary mask differs, or Exposure extends outside the visible annotation domain.')
    positions, rejected, columns, cy = plan_rectangle(raw, geo['rectangle'], support, 16)
    previous_points = {p['point']: p for p in geo['tiles']}
    for p in positions:
        if p['measurement_point']:
            prior = previous_points[p['measurement_point']]
            x, y = p['x'], p['y']
            if [x, y, x+256, y+256] != prior['xyxy_exclusive'] or sha256(prior['image']) != prior['image_sha256']:
                raise ValueError('Fixed P1/P2/P3 geometry or old crop changed.')
            if not np.array_equal(raw[y:y+256, x:x+256], read_gray(prior['image'])):
                raise ValueError('Fixed measurement crop does not match the original.')
    parent_before = snapshot(parent)
    protected = {str(p): sha256(p) for p in [original, geometry, *[p for p in download.iterdir() if p.is_file()]]}
    print(f'Verified Project12/Task800/Annotation{annotation["id"]}; {len(positions)} tiles. Copying parent...', flush=True)
    output.mkdir(parents=True)
    write_json(output/'RUNNING.json', {'stage': 'copy_and_append', 'parent': str(parent)})
    shutil.copytree(parent, output, dirs_exist_ok=True)
    for name in ('metadata.json', 'annotation.json', 'image.png', 'mask_alpha.png', 'mask.png'):
        destination = output/'provenance/project12_task800'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(download/name, destination)
    shutil.copy2(parent/'manifest.json', output/'provenance/changed_v2_manifest.json')
    shutil.copy2(parent/'splits.json', output/'provenance/changed_v2_splits.json')
    shutil.copy2(geometry, output/'provenance/case16_fixed_geometry.json')
    for relative in ('raw/task_800.png', 'raw_original/task_800.png'):
        shutil.copy2(original, output/relative)
    shutil.copy2(download/'mask_alpha.png', output/'mask_alpha/task_800.png')
    png(output/'mask/task_800.png', (alpha > 0).astype(np.uint8)*255)
    shutil.copy2(download/'image.png', output/'annotation_images/task_800.png')
    roi = np.zeros_like(raw)
    rx, ry, rx1, ry1 = geo['rectangle']
    roi[ry:ry1, rx:rx1] = raw[ry:ry1, rx:rx1]
    png(output/'roi_sources/task_800.png', roi)
    template = next(s['specimen_match'] for s in parent_manifest['images'] if s['case'] == 'case16')
    match = {**copy.deepcopy(template), 'project_id': PROJECT, 'task_id': TASK, 'specimen_number': 2,
             'shot_number': SHOT, 'excel_row': geo['row'], 'excel_P123_cells': f"G{geo['row']}:I{geo['row']}",
             'mapping_original_shot': SHOT, 'matching_sheet_row': None, 'matching_weight_g': None,
             'excel_weight_g': 75.62, 'weight_difference_g': None, 'image_reference': meta['image_reference'],
             'mapping_method': 'user_requested_Project12_and_exact_original_image_pixels',
             'mapping_note': 'Case16 fixed-side specimen02/shot1155; saved annotation image matches the original.',
             'measurement_coordinates_status': 'existing_fixed_256px_spacing547',
             'local_original_image': original.relative_to(ROOT).as_posix(), 'local_original_sha256': sha256(original)}
    source = {'project_id': PROJECT, 'task_id': TASK, 'case': 'case16', 'capture': '02', 'position': '000',
              'capture_kind': 'specimen', 'specimen_number': 2, 'shot_number': SHOT, 'specimen_match': match,
              'split': 'train', 'original_image': str(output/'raw/task_800.png'), 'original_sha256': sha256(original),
              'unrotated_original_image': str(output/'raw_original/task_800.png'), 'unrotated_original_sha256': sha256(original),
              'parent_original_image': str(original), 'parent_original_sha256': sha256(original),
              'annotation_source_image': str(output/'annotation_images/task_800.png'),
              'annotation_source_sha256': sha256(download/'image.png'), 'annotation_ids': [annotation['id']],
              'source_image': str(output/'roi_sources/task_800.png'), 'source_sha256': sha256(output/'roi_sources/task_800.png'),
              'label_source': 'mask_alpha/task_800.png', 'binary_mask': 'mask/task_800.png',
              'image_size_wh': list(raw.shape[::-1]), 'accepted_tiles': len(positions), 'rejected': rejected,
              'horizontal_grid_x0': columns, 'orientation_rotation_deg': 0, 'orientation_selected_using_gt': False,
              'orientation_policy': 'Preserve original000 and previously fixed Case16 specimen02 measurement geometry.',
              'orientation_review_method': 'No additional rotation requested for Task800.',
              'position_semantics': 'Original acquisition000; no additional rotation.',
              'roi_geometry': {'xyxy_exclusive': geo['rectangle'], 'top_line': geo['top_line'],
                               'bottom_line': geo['bottom_line'], 'centerline': geo['centerline'],
                               'center_y_at_image_center': cy, 'needs_review': False, 'manually_verified': False,
                               'quality_status': 'VISUALLY_REVIEWED', 'visual_review': 'Existing reviewed rectangle; same fixed point coordinates.',
                               'parameters': {'threshold_fraction': .10, 'x_trim_fraction': .075, 'profile_smoothing_sigma_px': 12.276},
                               'coordinate_policy': 'Reuse exact prior ROI/P1/P2/P3; no GT-conditioned repositioning.'}, 'tiles': []}
    point_rows, new_mappings = [], []
    for position in positions:
        x, y, point = position['x'], position['y'], position['measurement_point']
        box, name = [x, y, x+256, y+256], f'task_800_x{x:04d}_y{y:04d}.png'
        image_name, mask_name = 'images/'+name, 'masks/'+name
        crop, mask = raw[y:y+256, x:x+256], alpha[y:y+256, x:x+256]
        png(output/image_name, crop)
        png(output/mask_name, mask)
        count = int(np.count_nonzero(mask))
        source['tiles'].append({'xyxy_exclusive': box, 'image': image_name, 'mask': mask_name, 'label_pixels': count,
                                'measurement_point': point, 'grid_index': position['grid_index'], 'annotation_domain_fraction': 1.0,
                                'parent_xyxy_exclusive': box, 'parent_measurement_point': point})
        new_mappings.append({'project_id': PROJECT, 'task_id': TASK, 'split': 'train',
                             'rotation_deg': 0, 'source_image': None, 'source_mask': None, 'image': image_name, 'mask': mask_name,
                             'source_xyxy': box, 'xyxy': box, 'source_point': point, 'point': point, 'label_pixels': count})
        if point:
            point_rows.append({'project_id': PROJECT, 'task_id': TASK, 'case': 'case16', 'specimen': '02',
                               'specimen_number': 2, 'shot_number': SHOT, 'capture_kind': 'specimen', 'direction': '000',
                               'rotation_deg': 0, 'split': 'train', 'point': point, 'source_point': point, 'xyxy': box,
                               'image': image_name, 'gt_mask': mask_name, 'exposure_pixels': count,
                               'gt_impregnation_pct': 100*(1-count/65536)})
            for directory, pixels in (('raw', crop), ('mask_alpha', mask), ('mask', (mask > 0).astype(np.uint8)*255)):
                png(output/f'P1_P2_P3/{directory}/task_800_{point}.png', pixels)
    point_rows.sort(key=lambda row: row['point'])
    rates = {row['point']: row['gt_impregnation_pct'] for row in point_rows}
    measurement = {'project_id': PROJECT, 'task_id': TASK, 'case_number': 16, 'side': '고정측', 'specimen_number': 2,
                   'shot_number': SHOT, 'excel_row': geo['row'], 'rotation_deg': 0, 'split': 'train',
                   **{f'{point}_gt_pct': rates[point] for point in rates}, 'mean_gt_pct': sum(rates.values())/3,
                   **{f'excel_same_position_{point}_gt_pct': rates[point] for point in rates}}
    manifest = rebase(copy.deepcopy(parent_manifest), parent, output)
    manifest['images'].append(source)
    summary = manifest['summary']
    summary.update(created_at_utc=datetime.now(timezone.utc).isoformat(), parent_dataset=str(parent),
                   parent_manifest_sha256=sha256(parent/'manifest.json'), source_images=72,
                   accepted_tiles=1473+len(positions), image_png_count=1473+len(positions), mask_png_count=1473+len(positions),
                   tiles_by_position={'000': 1473+len(positions)}, central_three_tiles=216, central_three_000_tiles=216,
                   visual_reviewed_images=72, source_images_by_project={'9': 68, '10': 3, '12': 1},
                   source_gt=str(output/'mask_alpha'), source_gt_provenance=str(output/'provenance/label_sources.json'),
                   supervision_version=parent_manifest['summary']['supervision_version']+' + Project12 Task800 Annotation450',
                   new_case16_tiles=len(positions), new_case16_shot=SHOT)
    summary['new_task_shots'][str(TASK)] = SHOT
    summary['orientation_normalization']['rotation_counts'] = {'0': 61, '180': 11}
    summary['orientation_normalization']['criterion'] += ' Task800 preserves original000 orientation and fixed measurement points.'
    write_json(output/'manifest.json', manifest)
    write_json(output/'summary.json', summary)
    collections = {name: rebase(read_json(parent/f'{name}.json'), parent, output) for name in
                   ('point_measurements', 'specimen_measurements', 'orientation_audit', 'tile_mapping')}
    collections['point_measurements'].extend(point_rows)
    collections['specimen_measurements'].append(measurement)
    collections['specimen_measurements'].sort(key=lambda row: row['excel_row'])
    collections['tile_mapping'].extend(new_mappings)
    collections['orientation_audit'].append({'project_id': PROJECT, 'task_id': TASK, 'case': 'case16', 'capture': '02',
        'capture_kind': 'specimen', 'shot_number': SHOT, 'position': '000', 'rotation_deg': 0, 'split': 'train',
        'selection_method': 'Preserve original000 and fixed measurement points.', 'total_exposure_pixels': int(np.count_nonzero(alpha)),
        'source_original_sha256': sha256(original), 'saved_original_sha256': sha256(output/'raw/task_800.png'),
        'source_alpha_sha256': sha256(download/'mask_alpha.png'), 'saved_alpha_sha256': sha256(output/'mask_alpha/task_800.png'),
        **{f'{prefix}_{point}_gt_pct': rates[point] for prefix in ('before', 'after') for point in rates}})
    for name, rows in collections.items():
        write_json(output/f'{name}.json', rows)
        csv_file(output/f'{name}.csv', rows)
    write_json(output/'P1_P2_P3/manifest.json', {'points': collections['point_measurements']})
    registry = read_json(parent/'provenance/task_excel_mapping.json')
    registry['rows'].append(match)
    dataset_matches = {(s['project_id'], s['task_id']): s['specimen_match'] for s in manifest['images']}
    registry['rows'] = [copy.deepcopy(dataset_matches.get((r['project_id'], r['task_id']), r)) for r in registry['rows']]
    registry['without_downloaded_gt'] = [r for r in registry.get('without_downloaded_gt', [])
                                         if (r['case_number'], r['side'], r['shot_number']) != (16, '고정측', SHOT)]
    registry['case16_project12_added_at_utc'] = datetime.now(timezone.utc).isoformat()
    registry['generated_at_utc'] = registry['case16_project12_added_at_utc']
    registry['summary'].update(downloaded_tasks=len(registry['rows']),
        excel_eligible_gt_rows=sum(bool(r.get('excel_eligible')) for r in registry['rows']),
        without_downloaded_gt_rows=len(registry['without_downloaded_gt']), project10_pending_measurement_tasks=[])
    registry['sources']['dataset_manifest'] = str(output/'manifest.json')
    registry['sources']['sha256'][str(output/'manifest.json')] = sha256(output/'manifest.json')
    write_json(output/'provenance/task_excel_mapping.json', registry)
    write_json(output/'provenance/label_sources.json', [{'project_id': s['project_id'], 'task_id': s['task_id'],
        'annotation_ids': s['annotation_ids'], 'dataset_alpha': s['label_source'],
        'saved_download': str(download) if s['task_id'] == TASK else str(ROOT/f"data/label_studio_downloads/project_{s['project_id']}/task_{s['task_id']}"),
        'alpha_sha256': sha256(output/s['label_source']), 'rotation_deg': s['orientation_rotation_deg']}
        for s in manifest['images']])
    gallery = rebase(read_json(parent/'analysis/gallery.json'), parent, output)
    gallery.append(render_capture(output, source, point_rows))
    write_json(output/'analysis/gallery.json', gallery)
    measurement_page(output, collections['specimen_measurements'])
    result = {'project_id': PROJECT, 'task_id': TASK, 'annotation_id': annotation['id'],
              'annotation_updated_at': meta['annotation_updated_at'], 'downloaded_at_utc': meta['downloaded_at_utc'],
              'case_number': 16, 'side': '고정측', 'specimen_number': 2, 'shot_number': SHOT, 'rotation_deg': 0,
              'formula': '100 * (1 - count(alpha > 0) / 65536)', 'points': point_rows,
              'mean_gt_impregnation_pct': sum(rates.values())/3, 'alpha_values': np.unique(alpha).tolist(),
              'original_image': str(original), 'original_sha256': sha256(original), 'rectangle': geo['rectangle'],
              'new_tile_pairs': len(positions), 'source': 'Saved Label Studio Project12 annotation; GT area measurement.'}
    write_json(output/'case16_specimen02_measurement.json', result)
    csv_file(output/'analysis/case16_specimen02.csv', point_rows)
    write_json(output/'provenance/case16_input_sha256.json', protected)
    (output/'RUNNING.json').unlink()
    try:
        records, manifest_hash = load_tiles(output)
        splits = {name: sorted((r for r in records if membership[r['group']] == name), key=lambda r: r['image'])
                  for name in full}
        validate_splits(splits)
        assert all([r for r in splits[name] if r['task_id'] != TASK] == full[name] for name in full)
        write_json(output/'splits.json', {'manifest_sha256': manifest_hash, 'seed': 42, 'summary': describe_split(splits),
            'splits': splits, 'parent_split_sha256': sha256(parent/'splits.json'), 'orientation_selection_uses_gt': False,
            'new_sources_split_policy': 'Case16 specimen02 remains TRAIN with all other Case16 captures.'})
        build(output)
        assert snapshot(parent) == parent_before
        assert all(sha256(path) == expected for path, expected in protected.items())
        for relative in parent_before:
            if relative.startswith(('raw/', 'raw_original/', 'mask/', 'mask_alpha/', 'annotation_images/', 'roi_sources/',
                                    'images/', 'masks/', 'P1_P2_P3/raw/', 'P1_P2_P3/mask/', 'P1_P2_P3/mask_alpha/')):
                assert sha256(output/relative) == parent_before[relative], relative
        for tile in source['tiles']:
            x, y, x1, y1 = tile['xyxy_exclusive']
            assert np.array_equal(read_gray(output/tile['image']), raw[y:y1, x:x1])
            assert np.array_equal(read_gray(output/tile['mask']), alpha[y:y1, x:x1])
            assert support[y:y1, x:x1].all()
        central = select_central_splits(splits, True)
        verification = {'status': 'passed', 'source_images': len(manifest['images']), 'tile_pairs': len(records),
            'central_point_pairs': len(collections['point_measurements']), 'new_case16_tile_pairs': len(positions),
            'split_tiles': {k: len(v) for k, v in splits.items()}, 'central_split_tiles': {k: len(v) for k, v in central.items()},
            'existing_case_membership_and_row_order_preserved': True, 'case16_specimen02_in_train_only': True,
            'parent_entire_dataset_unchanged': True, 'saved_downloads_and_original_unchanged': True,
            'old_image_and_alpha_files_byte_identical': True, 'annotation_RLE_equals_alpha_png': True,
            'alpha_intermediate_values_preserved': np.unique(alpha).tolist(), 'new_tiles_equal_original_and_alpha_crops': True,
            'new_tiles_inside_known_annotation_domain': True, 'P123_equal_previous_fixed_coordinates': True,
            'new_capture_additional_rotation': 0, 'training_or_inference_run': False}
        write_json(output/'verification.json', verification)
        (output/'README.md').write_text(
            '# changed_v3\n\nchanged_v2에 Project12 Task800/Annotation450의 Case16 고정측 시편02(샷1155)를 추가했습니다.\n'
            '기존 71장과 기존 회전/ROI/좌표/GT를 보존하고, 새 시편은 원촬영000 방향과 기존 P1/P2/P3 좌표를 그대로 사용합니다.\n'
            f'전체 72장, {len(records)}타일, 중앙 216포인트. Train {len(splits["train"])} / Val {len(splits["val"])} / Test {len(splits["test"])}.\n'
            f'CentralOnly: Train {len(central["train"])} / Val {len(central["val"])} / Test {len(central["test"])}. Case16 시편02는 TRAIN입니다.\n'
            'Project9 68장 + Project10 3장 + Project12 1장. 추가 회전 0장; 기존 11장 회전은 유지합니다.\n'
            f'Project12 다운로드 시각(UTC): {meta["downloaded_at_utc"]}. Annotation 수정 시각(UTC): {meta["annotation_updated_at"]}.\n'
            'Exposure는 alpha>0이며 64/127/191/255를 모두 포함합니다. 원래 alpha는 masks/와 mask_alpha/에 보존합니다.\n'
            f'P1 {rates["P1"]:.8f}% / P2 {rates["P2"]:.8f}% / P3 {rates["P3"]:.8f}% / 평균 {sum(rates.values())/3:.8f}%.\n'
            '원본 DOE/매칭 엑셀과 changed_v2, 기존 실험 결과는 변경하지 않았습니다. 실제 학습/모델 추론은 실행하지 않았습니다.\n'
            'analysis/index.html: 72장 원본·GT·P1/P2/P3. analysis/task_800.png: 새 시편 측정 그림.\n'
            'case16_specimen02_measurement.json: 픽셀 수·좌표·함침율. verification.json: 원본/마스크/분할 검증.\n\n'
            'PowerShell 실행 예시(새 실험 출력):\n\n```powershell\nSet-Location D:\\Code\\vision_data\n'
            '.\\src\\confident\\run_confident.ps1 -Data .\\data\\changed_v3 -CentralOnly -Stage all '
            '-Arms A,B,D -Output ".\\analysis\\confident\\changed_v3_central_$(Get-Date -Format yyyyMMdd_HHmmss)"\n```\n',
            encoding='utf-8')
    except Exception:
        write_json(output/'RUNNING.json', {'stage': 'validation_failed'})
        raise
    print({'output': str(output), 'rates': rates, 'mean': result['mean_gt_impregnation_pct'], **verification}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT/'data/changed_v2')
    parser.add_argument('--output', type=Path, default=ROOT/'data/changed_v3')
    parser.add_argument('--download', type=Path, default=ROOT/'data/label_studio_downloads/project_12/task_800')
    parser.add_argument('--geometry', type=Path, default=ROOT/'outputs/measurements/rectangle_roi_20261001/case16_provenance.json')
    args = parser.parse_args()
    append_capture(args.parent, args.output, args.download, args.geometry)


if __name__ == '__main__':
    main()
