"""Export native rectangular P1/P2/P3 crops from frozen changed_v3 GT.

Uses saved specimen ROI centers, not Exposure centroids. No training or Gold
labels are used. The output is a new dataset; existing directories are refused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import psutil

ROOT = Path(__file__).resolve().parents[2]
POINTS = ('P1', 'P2', 'P3')
COLORS = ('#ffca50', '#36d6ef', '#be9aff')
TEMPLATE = Path(__file__).with_name('changed_v3_tall_points_template.html')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def gray(path, allow_rgb_grayscale=False):
    with Image.open(path) as image:
        value = np.array(image)
    if (allow_rgb_grayscale and value.ndim == 3 and value.shape[2] == 3 and
            np.array_equal(value[:, :, 0], value[:, :, 1]) and
            np.array_equal(value[:, :, 0], value[:, :, 2])):
        value = value[:, :, 0].copy()
    if value.ndim != 2 or value.dtype != np.uint8:
        raise ValueError(f'Expected native uint8 grayscale: {path}')
    return value


def write_png(path, pixels):
    Image.fromarray(pixels).save(path)
    with Image.open(path) as image:
        saved = np.array(image)
    if not np.array_equal(saved, pixels):
        raise ValueError(f'Saved PNG differs from native source pixels: {path}')


def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf' if bold else
                              'C:/Windows/Fonts/malgun.ttf', size)


def identity(source):
    match = source['specimen_match']
    target = (f"샷{match['shot_number']}" if match['specimen_number'] is None else
              f"시편{match['specimen_number']:02d}")
    return f"Case{match['case_number']:02d} · {match['side']} · {target}"


def crop_box(old_box, roi_box, image_wh, width, height, override=None, offset_y=0):
    """Round an ROI pixel center to the nearest feasible pixel crop origin.

    Coordinates are zero-based and right/bottom exclusive. An even-size crop
    has a half-integer pixel center; nearest rounding can differ by 0.5 px.
    Overrides supply pixel-center coordinates, either x/y or only one axis.
    """
    old_x0, _, old_x1, _ = old_box
    requested_x = (old_x0 + old_x1 - 1) / 2
    requested_y = (roi_box[1] + roi_box[3] - 1) / 2 + offset_y
    if override:
        requested_x = float(override.get('x', requested_x))
        requested_y = float(override.get('y', requested_y))
    if not all(math.isfinite(v) for v in (requested_x, requested_y)):
        raise ValueError('Center coordinates must be finite.')
    x0 = math.floor(requested_x - (width - 1) / 2 + 0.5)
    y0 = math.floor(requested_y - (height - 1) / 2 + 0.5)
    box = [x0, y0, x0 + width, y0 + height]
    if not (0 <= x0 < box[2] <= image_wh[0] and
            0 <= y0 < box[3] <= image_wh[1]):
        raise ValueError(f'Crop outside original image: {box}')
    return box, [requested_x, requested_y]


def overlay(raw, alpha):
    value = np.repeat(raw[:, :, None], 3, axis=2)
    active = alpha > 0
    value[active] = (value[active].astype(np.float32) * 0.60 +
                     np.array([255, 55, 85]) * 0.40).astype(np.uint8)
    return value


def dashed_rectangle(draw, box, color, width=2):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    for x in range(x0, x1, 14):
        draw.line((x, y0, min(x + 7, x1), y0), fill=color, width=width)
        draw.line((x, y1, min(x + 7, x1), y1), fill=color, width=width)
    for y in range(y0, y1, 14):
        draw.line((x0, y, x0, min(y + 7, y1)), fill=color, width=width)
        draw.line((x1, y, x1, min(y + 7, y1)), fill=color, width=width)


def render_figure(output, source, raw, alpha, points, width, height):
    canvas = Image.new('RGB', (1940, 1500), '#f3f6fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 22), identity(source), fill='#17304b', font=font(32, True))
    draw.text((32, 70), f"Project {source['project_id']} / Task {source['task_id']} · "
              f"{source['split']} · 기존 회전 {source['orientation_rotation_deg']}° · "
              f"가로 {width} × 세로 {height} 원본 픽셀", fill='#425872', font=font(23))
    left, top, side = 32, 154, 870
    base = Image.fromarray(overlay(raw, alpha)).resize((side, side), Image.Resampling.LANCZOS)
    canvas.paste(base, (left, top))
    sx, sy = side / raw.shape[1], side / raw.shape[0]
    roi = source['roi_geometry']['xyxy_exclusive']
    def projected(box):
        return (left + box[0] * sx, top + box[1] * sy,
                left + (box[2] - 1) * sx, top + (box[3] - 1) * sy)
    draw.rectangle(projected(roi), outline='#52c38a', width=2)
    cy = (roi[1] + roi[3] - 1) / 2
    draw.line((left, top + cy * sy, left + side - 1, top + cy * sy),
              fill='#52c38a', width=1)
    panel_width = 300
    panel_scale = min(256 / width, 512 / height)
    panel_wh = (round(width * panel_scale), round(height * panel_scale))
    for index, point in enumerate(points):
        color = COLORS[index]
        box = point['xyxy_exclusive']
        draw.rectangle(projected(box), outline=color, width=4)
        dashed_rectangle(draw, projected(point['parent_xyxy_exclusive']), '#ffffff')
        px, py, _, _ = projected(box)
        draw.text((px, py - 28), point['measurement_point'], fill=color,
                  font=font(21, True), stroke_width=1, stroke_fill='#17304b')
        col = 962 + index * panel_width
        label = f"{point['measurement_point']}  GT {point['gt_impregnation_pct']:.3f}%"
        draw.text((col, 119), label, fill='#17304b', font=font(23, True))
        draw.text((col, 157), '원본 크롭', fill='#425872', font=font(18))
        x0, y0, x1, y1 = box
        patch = raw[y0:y1, x0:x1]
        active = alpha[y0:y1, x0:x1]
        original = Image.fromarray(patch).convert('RGB')
        mask = Image.fromarray((active > 0).astype(np.uint8) * 255).convert('RGB')
        combined = Image.fromarray(overlay(patch, active))
        for image, yy in ((original, 194), (mask, 759)):
            if image.size != panel_wh:
                image = image.resize(panel_wh, Image.Resampling.NEAREST)
            canvas.paste(image, (col, yy))
        draw.text((col, 720), 'GT 미함침 · 흰색 = alpha>0', fill='#425872', font=font(18))
        draw.text((col, 1290), f"x [{x0}, {x1})", fill='#17304b', font=font(18))
        draw.text((col, 1320), f"y [{y0}, {y1})", fill='#17304b', font=font(18))
        draw.text((col, 1350), f"중심 Δy {point['center_shift_xy'][1]:+.1f}px", fill='#425872', font=font(18))
        write_png(output / point['overlay'], np.array(combined))
    draw.text((32, 1060), '초록: 저장된 시편 ROI와 중심선', fill='#17304b', font=font(23))
    draw.text((32, 1102), '색상 실선: 새 P1/P2/P3 256×512 영역', fill='#17304b', font=font(23))
    draw.text((32, 1144), '흰 점선: 기존 256×256 영역 · 붉은색: GT 미함침', fill='#17304b', font=font(22))
    draw.text((32, 1206), 'GT 함침율 = 100 × (1 − Exposure 픽셀 수 / 전체 픽셀 수)',
              fill='#425872', font=font(20))
    draw.text((32, 1244), f'각 {width * height:,}픽셀 사용 · 리사이즈/패딩 없음',
              fill='#425872', font=font(20))
    draw.text((32, 1320), 'P1 왼쪽 · P2 중앙 · P3 오른쪽 · 가로 간격 547px',
              fill='#425872', font=font(20))
    draw.text((32, 1360), 'GT 출처: changed_v3 저장 원본 alpha · Gold 라벨 사용 없음',
              fill='#425872', font=font(20))
    target = output / 'analysis' / f"task_{source['task_id']}.png"
    canvas.save(target)
    return target


def verify_parent(parent, manifest, splits):
    if splits['manifest_sha256'] != digest(parent / 'manifest.json'):
        raise ValueError('Parent split/manifest hash mismatch.')
    counts = {key: len(rows) for key, rows in splits['splits'].items()}
    if counts != {'train': 1056, 'val': 187, 'test': 251} or len(manifest['images']) != 72:
        raise ValueError('Expected frozen changed_v3 72 captures and 1,494 pairs.')
    by_image = {}
    case_splits = {}
    for split, rows in splits['splits'].items():
        for row in rows:
            if row['image'] in by_image:
                raise ValueError('Duplicate parent tile.')
            by_image[row['image']] = (split, row)
            previous = case_splits.setdefault(row['case'], split)
            if previous != split:
                raise ValueError('Parent Case appears in multiple splits.')
            for kind in ('image', 'mask'):
                if digest(parent / row[kind]) != row[kind + '_sha256']:
                    raise ValueError(f'Parent tile hash mismatch: {row[kind]}')
    tiles = [tile for source in manifest['images'] for tile in source['tiles']]
    if len(tiles) != 1494 or {t['image'] for t in tiles} != set(by_image):
        raise ValueError('Complete parent manifest/split coverage mismatch.')
    for source in manifest['images']:
        if source['split'] != case_splits[source['case']]:
            raise ValueError('Parent source split mismatch.')
        for tile in source['tiles']:
            split, row = by_image[tile['image']]
            if (split != source['split'] or row['mask'] != tile['mask'] or
                    row['xyxy'] != tile['xyxy_exclusive'] or
                    row['task_id'] != source['task_id'] or row['case'] != source['case'] or
                    row['label_pixels'] != tile['label_pixels']):
                raise ValueError('Parent manifest tile differs from frozen split record.')
    return counts


def export(parent, output, width=256, height=512, centers_path=None, offset_y=0,
           memory_budget_mib=4096):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if (output.exists() or output.is_relative_to(parent) or parent.is_relative_to(output)):
        raise ValueError('Choose a new output outside changed_v3; existing data is never overwritten.')
    if width != 256 or height < 1 or memory_budget_mib <= 0 or not math.isfinite(offset_y):
        raise ValueError('Width must be 256; height/budget must be positive and offset finite.')
    manifest = read_json(parent / 'manifest.json')
    splits = read_json(parent / 'splits.json')
    protected = {str(p.resolve()): digest(p) for p in parent.rglob('*') if p.is_file()}
    peak_mib = 0.0
    process = psutil.Process()
    def guard():
        nonlocal peak_mib
        current = process.memory_info().rss / 1024**2
        peak_mib = max(peak_mib, current)
        if current > memory_budget_mib:
            raise MemoryError(f'RAM {current:.1f} MiB exceeds {memory_budget_mib} MiB guard.')
    guard()
    counts = verify_parent(parent, manifest, splits)
    centers = read_json(centers_path) if centers_path else {}
    source_ids = {str(source['task_id']) for source in manifest['images']}
    if set(centers) - source_ids:
        raise ValueError('Center overrides contain unknown Task IDs.')
    for values in centers.values():
        if not isinstance(values, dict) or set(values) - set(POINTS):
            raise ValueError('Center overrides must map Task ID to P1/P2/P3.')
    output.mkdir(parents=True)
    for directory in ('images', 'masks', 'binary_masks', 'overlays', 'analysis', 'provenance'):
        (output / directory).mkdir()
    write_json(output / 'RUNNING.json', {'source': str(parent), 'status': 'exporting'})
    shutil.copy2(parent / 'manifest.json', output / 'provenance/parent_manifest.json')
    shutil.copy2(parent / 'splits.json', output / 'provenance/parent_splits.json')
    shutil.copy2(__file__, output / 'provenance/export_script.py')
    shutil.copy2(TEMPLATE, output / 'provenance/viewer_template.html')
    write_json(output / 'provenance/source_file_sha256.json', protected)
    if centers_path:
        shutil.copy2(centers_path, output / 'provenance/centers.json')
    label_sources = {s['task_id']: s for s in read_json(parent / 'provenance/label_sources.json')}
    records, table, viewer = [], [], []
    exported_splits = {name: [] for name in ('train', 'val', 'test')}
    ordered = sorted(manifest['images'], key=lambda s: (s['specimen_match']['case_number'],
                      s['specimen_match']['side'] == '이동측',
                      s['specimen_number'] or s['shot_number']))
    for ordinal, source in enumerate(ordered, 1):
        guard()
        task = source['task_id']
        source_alpha = parent / source['label_source']
        provenance = label_sources[task]
        paths = [(Path(source['original_image']), source['original_sha256']),
                 (Path(source['annotation_source_image']), source['annotation_source_sha256']),
                 (source_alpha, provenance['alpha_sha256']),
                 (Path(source['unrotated_original_image']), source['unrotated_original_sha256'])]
        for path, expected in paths:
            if digest(path) != expected:
                raise ValueError(f'Source provenance hash mismatch: {path}')
        if (provenance['annotation_ids'] != source['annotation_ids'] or
                provenance['project_id'] != source['project_id'] or
                provenance['rotation_deg'] != source['orientation_rotation_deg']):
            raise ValueError('Source annotation identity/orientation mismatch.')
        raw, alpha = gray(source['original_image']), gray(source_alpha)
        annotation = gray(source['annotation_source_image'], allow_rgb_grayscale=True)
        unrotated = gray(source['unrotated_original_image'])
        rotation = source['orientation_rotation_deg']
        if rotation not in (0, 180) or not np.array_equal(raw, np.flip(unrotated, (0, 1)) if rotation else unrotated):
            raise ValueError('Source orientation differs from frozen changed_v3.')
        del unrotated
        if not (raw.shape == alpha.shape == annotation.shape and
                [raw.shape[1], raw.shape[0]] == source['image_size_wh']):
            raise ValueError('Source image/GT/annotation shapes mismatch.')
        support = annotation > 0
        if not np.array_equal(annotation[support], raw[support]):
            raise ValueError('Annotated visible source pixels differ from original.')
        if not np.array_equal(gray(parent / source['binary_mask']) > 0, alpha > 0):
            raise ValueError('Full source binary GT differs from alpha>0.')
        roi = source['roi_geometry']['xyxy_exclusive']
        old_points = sorted((t for t in source['tiles'] if t['measurement_point']),
                            key=lambda t: t['measurement_point'])
        if [t['measurement_point'] for t in old_points] != list(POINTS):
            raise ValueError('Each source must have exactly P1/P2/P3.')
        points = []
        for old in old_points:
            point = old['measurement_point']
            ox, oy, ox1, oy1 = old['xyxy_exclusive']
            if not (np.array_equal(gray(parent / old['image']), raw[oy:oy1, ox:ox1]) and
                    np.array_equal(gray(parent / old['mask']), alpha[oy:oy1, ox:ox1])):
                raise ValueError('Parent central crop differs from full original/GT.')
            override = centers.get(str(task), {}).get(point)
            box, requested = crop_box(old['xyxy_exclusive'], roi, source['image_size_wh'],
                                      width, height, override, offset_y)
            x, y, x1, y1 = box
            if not (roi[0] + 16 <= x and x1 <= roi[2] - 16 and
                    roi[1] + 16 <= y and y1 <= roi[3] - 16):
                raise ValueError(f'Expanded {task}/{point} crosses saved ROI + 16px margin.')
            patch, mask = raw[y:y1, x:x1], alpha[y:y1, x:x1]
            if (patch.shape != (height, width) or not (patch > 0).all() or
                    not support[y:y1, x:x1].all()):
                raise ValueError(f'{task}/{point} is outside known annotation domain.')
            name = f'task_{task}_{point}_x{x:04d}_y{y:04d}.png'
            image_file, alpha_file, binary_file = (output / d / name for d in ('images', 'masks', 'binary_masks'))
            write_png(image_file, patch)
            write_png(alpha_file, mask)
            write_png(binary_file, (mask > 0).astype(np.uint8) * 255)
            n = int(np.count_nonzero(mask > 0))
            old_n = int(np.count_nonzero(alpha[oy:oy1, ox:ox1] > 0))
            center = [(x + x1 - 1) / 2, (y + y1 - 1) / 2]
            previous_center = [(ox + ox1 - 1) / 2, (oy + oy1 - 1) / 2]
            tile = {'image': f'images/{name}', 'mask': f'masks/{name}',
                    'binary_mask': f'binary_masks/{name}', 'overlay': f'overlays/{name}',
                    'xyxy_exclusive': box, 'measurement_point': point,
                    'parent_image': old['image'], 'parent_mask': old['mask'],
                    'parent_xyxy_exclusive': old['xyxy_exclusive'],
                    'requested_center_xy': requested, 'center_pixel_coordinates': center,
                    'center_shift_xy': [a - b for a, b in zip(center, previous_center)],
                    'center_policy': 'manual_override' if override else 'saved_specimen_roi_midpoint',
                    'label_pixels': n, 'valid_pixels': width * height,
                    'gt_impregnation_pct': 100 * (1 - n / (width * height)),
                    'parent_gt_impregnation_pct': 100 * (1 - old_n / 65536),
                    'annotation_domain_fraction': 1.0,
                    'image_sha256': digest(image_file), 'mask_sha256': digest(alpha_file),
                    'binary_mask_sha256': digest(binary_file)}
            tile['gt_change_pp'] = tile['gt_impregnation_pct'] - tile['parent_gt_impregnation_pct']
            points.append(tile)
            match = source['specimen_match']
            table.append({'task_id': task, 'project_id': source['project_id'],
                          'annotation_ids': ','.join(map(str, source['annotation_ids'])),
                          'case': source['case'], 'side': match['side'],
                          'specimen_number': match['specimen_number'], 'shot_number': match['shot_number'],
                          'split': source['split'], 'rotation_deg': rotation, 'point': point,
                          'x0': x, 'y0': y, 'x1_exclusive': x1, 'y1_exclusive': y1,
                          'center_x': center[0], 'center_y': center[1],
                          'center_shift_y': tile['center_shift_xy'][1],
                          'exposure_pixels': n, 'valid_pixels': width * height,
                          'gt_impregnation_pct': tile['gt_impregnation_pct'],
                          'previous_256x256_gt_pct': tile['parent_gt_impregnation_pct'],
                          'gt_change_pp': tile['gt_change_pp'],
                          'image': tile['image'], 'mask': tile['mask'],
                          'image_sha256': tile['image_sha256'], 'mask_sha256': tile['mask_sha256'],
                          'source_image_sha256': source['original_sha256'],
                          'source_alpha_sha256': provenance['alpha_sha256']})
            exported_splits[source['split']].append({**tile, 'task_id': task,
                'case': source['case'], 'group': f"case{source['specimen_match']['case_number']:02d}", 'capture': source['capture'],
                'position': source['position'], 'xyxy': box})
        result = {k: copy.deepcopy(source[k]) for k in ('task_id', 'project_id', 'case', 'capture',
                   'position', 'image_size_wh', 'roi_geometry', 'specimen_match', 'specimen_number',
                   'shot_number', 'annotation_ids', 'orientation_rotation_deg', 'split',
                   'original_image', 'original_sha256', 'annotation_source_image',
                   'annotation_source_sha256')}
        result.update(label_source=str(source_alpha), label_source_sha256=provenance['alpha_sha256'],
                      accepted_tiles=3, tiles=points)
        records.append(result)
        figure = render_figure(output, source, raw, alpha, points, width, height)
        for tile in points:
            tile['overlay_sha256'] = digest(output / tile['overlay'])
        viewer.append({'task_id': task, 'label': identity(source), 'case': source['case'],
                       'split': source['split'], 'rotation': rotation,
                       'annotation_ids': source['annotation_ids'], 'project_id': source['project_id'],
                       'figure': figure.name, 'points': points})
        guard()
        del raw, alpha, annotation, support, patch, mask
        print(f'Exported {ordinal}/72 captures; {ordinal * 3}/216 crops', flush=True)
    summary = {'dataset': output.name, 'source_dataset': str(parent),
               'source_manifest_sha256': digest(parent / 'manifest.json'),
               'source_splits_sha256': digest(parent / 'splits.json'),
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'source_images': 72, 'accepted_tiles': 216, 'central_three_tiles': 216,
               'image_png_count': 216, 'mask_png_count': 216, 'count_only': False,
               'settings': {'width': width, 'height': height, 'patch_size_wh': [width, height],
                            'alignment': 'saved_specimen_roi_midpoint', 'point_spacing_px': 547,
                            'center_offset_y': offset_y, 'edge_margin': 16,
                            'resized': False, 'padding': False, 'additional_rotation': False,
                            'coordinates': 'zero-based [x0,y0,x1,y1), pixel center (x0+x1-1)/2',
                            'rounding': 'nearest origin; ties towards positive coordinate'},
               'mask_policy': 'Original alpha preserved; alpha>0 is Exposure; 255 is foreground.',
               'split_policy': 'Inherit each original Task and Case split; only P1/P2/P3 exported.',
               'mask_encoding': {'rule': 'positive_is_defect', 'ignore_value': None},
               'gold_labels_used': False, 'model_training_or_inference_run': False,
               'code_sha256': digest(__file__), 'center_overrides_sha256': digest(centers_path) if centers_path else None}
    full_manifest = {'summary': summary, 'images': records}
    write_json(output / 'manifest.json', full_manifest)
    split_summary = {name: {'tiles': len(rows), 'cases': sorted({r['group'] for r in rows}),
                           'source_images': len({r['task_id'] for r in rows}),
                           'exposure_fraction': sum(r['label_pixels'] for r in rows) / (len(rows) * width * height)}
                     for name, rows in exported_splits.items()}
    if {name: len(rows) for name, rows in exported_splits.items()} != {'train': 153, 'val': 27, 'test': 36}:
        raise ValueError('Unexpected inherited central split counts.')
    write_json(output / 'splits.json', {'manifest_sha256': digest(output / 'manifest.json'),
               'seed': splits['seed'], 'summary': split_summary, 'splits': exported_splits})
    with (output / 'point_measurements.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(output / 'point_measurements.json', table)
    viewer_payload = json.dumps({'records': viewer, 'width': width, 'height': height},
                                ensure_ascii=False).replace('<', '\\u003c')
    page = TEMPLATE.read_text(encoding='utf-8').replace('__DATA__', viewer_payload)
    (output / 'analysis/index.html').write_text(page, encoding='utf-8')
    write_json(output / 'centers.example.json', {'616': {'P1': {'y': 1185.0},
                                                      'P2': {'y': 1185.0}, 'P3': {'y': 1185.0}}})
    report = {'completed': True, 'source_pairs_verified': sum(counts.values()),
              'exported_pairs': 216, 'source_captures': 72, 'source_unchanged': True,
              'all_native_image_alpha_binary_crops_pixel_equal': True,
              'all_crops_fully_inside_saved_roi_and_annotation_domain': True,
              'all_rotations_and_case_splits_preserved': True,
              'image_shape_hw': [height, width], 'mask_shape_hw': [height, width],
              'split_summary': split_summary,
              'center_shift_y_counts': dict(Counter(str(t['center_shift_xy'][1]) for r in records for t in r['tiles'])),
              'gt_impregnation_pct_mean_per_crop': float(np.mean([r['gt_impregnation_pct'] for r in table])),
              'memory_budget_mib': memory_budget_mib, 'observed_peak_rss_mib': round(peak_mib, 2),
              'gold_labels_used': False, 'training_run': False}
    readme = f'''# {output.name}

changed_v3의 저장 GT로 72촬영본 × P1/P2/P3 = 216쌍을 생성했습니다.
가로 {width} × 세로 {height}; 원본 픽셀 직접 crop, 리사이즈/패딩 없음.
좌우 위치와 547px 간격, 원래 0/180° 회전, Case 분할을 유지합니다.
중심: 저장 시편 ROI의 상·하단 픽셀 중점. Exposure 중심을 사용하지 않습니다.
짝수 크기의 실제 픽셀 중심은 반정수이므로 ROI 중심에서 최대 0.5px 반올림 오차가 있습니다.

Train 153 / Validation 27 / 기존 Test 36 (원촬영본 51 / 9 / 12).
이 버전은 중앙 포인트만 포함합니다. 기존 1,494쌍의 전체 타일 버전을 대체하지 않습니다.
masks/: 원래 uint8 alpha, binary_masks/: alpha>0이면255. 255는 미함침이며 Ignore 아님.
GT 함침율 = 100 × (1 − Exposure 픽셀 수 / {width * height:,}). 전체 크롭 픽셀 사용.
point_measurements.csv/json: 기존 256×256과 새 영역의 좌표·픽셀 수·GT 함침율·차이(pp)·해시.
manifest.json / splits.json: 새 영역, 원래 분할, 원본 Task/Annotation, GT·코드 provenance.
analysis/index.html: 72개 원본의 자른 위치와 P1/P2/P3 결과. overlays/: 각 포인트 GT 오버레이.

중심 재조정(새 출력에 재생성):
```powershell
.\\.venv\\Scripts\\python.exe -m src.preprocess.export_changed_v3_tall_points --output data/changed_v3_p123_256x512_custom --centers-json data/{output.name}/centers.example.json
```
centers JSON은 원본 Task ID → P1/P2/P3 → 픽셀 중심 x/y이며, 생략 축은 기본 중심입니다.
기본 ROI에서 전부 위/아래로 이동하려면 --center-offset-y 숫자를 지정합니다.
ROI 가장자리 16px/알려진 annotation 영역을 벗어나면 오류로 중단하며 임의 패딩하지 않습니다.

GT는 changed_v3의 고정 저장본입니다. Project13 Gold GT는 이 버전에 사용하지 않았습니다.
Gold72는 원래 256×256 평가를 유지하므로 새 256×512 측정값과 별도입니다.
학습/추론은 실행하지 않았습니다. 현재 src/training/data.py 로더는 256×256을 강제하므로
이 직사각형 데이터로 학습하려면 별도 크기 대응이 필요합니다.
verification.json: 216쌍 native 픽셀·alpha·annotation domain·분할 및 원본 전체 파일 보존 검증.
'''
    (output / 'README.md').write_text(readme, encoding='utf-8')
    now = {str(p.resolve()): digest(p) for p in parent.rglob('*') if p.is_file()}
    if protected != now:
        raise ValueError('Frozen changed_v3 source files changed during export.')
    write_json(output / 'verification.json', report)
    (output / 'RUNNING.json').unlink()
    print(json.dumps({'output': str(output), **report}, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT / 'data/changed_v3')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/changed_v3_p123_256x512')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--centers-json', type=Path)
    parser.add_argument('--center-offset-y', type=float, default=0)
    parser.add_argument('--memory-budget-mib', type=int, default=4096)
    args = parser.parse_args()
    export(args.parent, args.output, args.width, args.height, args.centers_json,
           args.center_offset_y, args.memory_budget_mib)


if __name__ == '__main__':
    main()
