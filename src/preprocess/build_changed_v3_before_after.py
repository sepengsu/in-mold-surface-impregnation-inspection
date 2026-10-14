"""Add a Before/After viewer without modifying frozen training pixels or GT."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import psutil

from .export_changed_v3_tall_points import (COLORS, ROOT, digest, font, gray,
                                          identity, overlay, read_json, write_json, write_png)

TEMPLATE = Path(__file__).with_name('changed_v3_before_after_template.html')


def overview(raw, alpha, roi, boxes):
    panel = Image.fromarray(overlay(raw, alpha))
    panel.thumbnail((870, 870), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    sx, sy = panel.width / raw.shape[1], panel.height / raw.shape[0]
    def projection(box):
        return (box[0] * sx, box[1] * sy, (box[2] - 1) * sx, (box[3] - 1) * sy)
    draw.rectangle(projection(roi), outline='#52c38a', width=2)
    cy = (roi[1] + roi[3] - 1) / 2
    draw.line((0, cy * sy, panel.width - 1, cy * sy), fill='#52c38a', width=1)
    for index, (point, box) in enumerate(boxes):
        color = COLORS[index]
        draw.rectangle(projection(box), outline=color, width=4)
        x, y, _, _ = projection(box)
        draw.text((x, max(0, y - 26)), point, fill=color, font=font(21, True),
                  stroke_width=1, stroke_fill='#17304b')
    return panel


def preview(root, record, before, after):
    canvas = Image.new('RGB', (1840, 1770), '#f2f5f9')
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 24), record['label'] + ' · Before / After', fill='#17304b', font=font(30, True))
    draw.text((32, 76), 'Before 256×256', fill='#17304b', font=font(26, True))
    draw.text((938, 76), 'After 256×512 · 시편 ROI 중심', fill='#17304b', font=font(26, True))
    canvas.paste(before, (32, 125))
    canvas.paste(after, (938, 125))
    for index, point in enumerate(record['points']):
        left = 32 + index * 604
        draw.text((left, 1030), f"{point['point']}   Δ {point['change_pp']:+.3f}pp", fill='#17304b', font=font(23, True))
        for version, x, pct in (('before', left, point['before_pct']),
                                ('after', left + 278, point['after_pct'])):
            title = f"{version.title()}  {pct:.3f}%"
            draw.text((x, 1075), title, fill='#425872', font=font(21))
            draw.rectangle((x, 1125, x + 255, 1636), fill='#e0e7ee')
            image_path = root / 'analysis' / point[version]['overlay']
            with Image.open(image_path) as image:
                native = image.copy()
            y = 1125 + (point['before_offset_y'] if version == 'before' else 0)
            canvas.paste(native, (x, y))
        draw.text((left, 1660), f"Before {point['before_box']}", fill='#425872', font=font(17))
        draw.text((left, 1690), f"After  {point['after_box']}", fill='#425872', font=font(17))
    draw.text((32, 1730), '동일 배율 · 회색: 비교용 표시 여백 · 빨강: GT 미함침 · 차이 단위 pp',
              fill='#425872', font=font(21))
    canvas.save(root / 'analysis/before_after/preview_task_616.png')


def build(dataset, parent, memory_budget_mib=4096):
    dataset, parent = Path(dataset).resolve(), Path(parent).resolve()
    if memory_budget_mib <= 0:
        raise ValueError('Memory guard budget must be positive.')
    manifest = read_json(dataset / 'manifest.json')
    parent_manifest = read_json(parent / 'manifest.json')
    if manifest['summary']['settings']['patch_size_wh'] != [256, 512]:
        raise ValueError('Expected native 256x512 data.')
    if digest(parent / 'manifest.json') != manifest['summary']['source_manifest_sha256']:
        raise ValueError('Frozen changed_v3 parent manifest differs.')
    if len(manifest['images']) != 72:
        raise ValueError('Expected 72 sources.')
    original = {s['task_id']: s for s in parent_manifest['images']}
    protected = {}
    for root in (parent, dataset):
        for path in root.rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if root == dataset and (relative == 'analysis/index.html' or relative.startswith('analysis/before_after/')):
                continue
            protected[str(path)] = digest(path)
    directory = dataset / 'analysis/before_after'
    directory.mkdir(exist_ok=True)
    for name in ('images', 'binary_masks', 'overlays', 'overviews', 'provenance'):
        (directory / name).mkdir(exist_ok=True)
    backup = directory / 'provenance/index_before.html'
    if not backup.exists():
        shutil.copy2(dataset / 'analysis/index.html', backup)
    write_json(directory / 'RUNNING.json', {'status': 'creating_before_after_assets'})
    records, pixel_checks = [], 0
    process = psutil.Process()
    observed_mib = 0.0
    def guard():
        nonlocal observed_mib
        current = process.memory_info().rss / 1024**2
        observed_mib = max(observed_mib, current)
        if current > memory_budget_mib:
            raise MemoryError('Before/After viewer exceeded memory guard.')
    for ordinal, source in enumerate(manifest['images'], 1):
        guard()
        task = source['task_id']
        old = original[task]
        raw, alpha = gray(old['original_image']), gray(parent / old['label_source'])
        if digest(old['original_image']) != old['original_sha256']:
            raise ValueError('Frozen original image hash differs.')
        old_points = {t['measurement_point']: t for t in old['tiles'] if t['measurement_point']}
        record = {'task_id': task, 'project_id': source['project_id'], 'label': identity(source),
                  'annotation_ids': source['annotation_ids'], 'split': source['split'],
                  'rotation': source['orientation_rotation_deg'],
                  'before_overview': f'before_after/overviews/task_{task}_before.png',
                  'after_overview': f'before_after/overviews/task_{task}_after.png', 'points': []}
        for point in source['tiles']:
            name = point['measurement_point']
            previous = old_points[name]
            ox, oy, ox1, oy1 = previous['xyxy_exclusive']
            x, y, x1, y1 = point['xyxy_exclusive']
            old_image, old_alpha = gray(parent / previous['image']), gray(parent / previous['mask'])
            if not (np.array_equal(old_image, raw[oy:oy1, ox:ox1]) and
                    np.array_equal(old_alpha, alpha[oy:oy1, ox:ox1]) and
                    np.array_equal(gray(dataset / point['image']), raw[y:y1, x:x1]) and
                    np.array_equal(gray(dataset / point['mask']), alpha[y:y1, x:x1])):
                raise ValueError('Before/After pixels differ from frozen source.')
            if not (ox == x and ox1 == x1 and 0 <= oy - y <= 256 and oy1 <= y1):
                raise ValueError('Before crop must fit in the aligned After comparison stage.')
            old_count = int(np.count_nonzero(old_alpha > 0))
            before_pct = 100 * (1 - old_count / 65536)
            after_pct = 100 * (1 - np.count_nonzero(alpha[y:y1, x:x1] > 0) / 131072)
            if not (abs(before_pct - point['parent_gt_impregnation_pct']) < 1e-10 and
                    abs(after_pct - point['gt_impregnation_pct']) < 1e-10):
                raise ValueError('Before/After GT measurement mismatch.')
            filename = f'task_{task}_{name}.png'
            shutil.copy2(parent / previous['image'], directory / 'images' / filename)
            write_png(directory / 'binary_masks' / filename, (old_alpha > 0).astype(np.uint8) * 255)
            write_png(directory / 'overlays' / filename, overlay(old_image, old_alpha))
            record['points'].append({'point': name, 'before_box': previous['xyxy_exclusive'],
                'after_box': point['xyxy_exclusive'], 'before_offset_y': oy - y,
                'center_shift_y': point['center_shift_xy'][1],
                'before_pct': before_pct, 'after_pct': float(after_pct),
                'change_pp': float(after_pct - before_pct),
                'before': {'image': f'before_after/images/{filename}',
                           'mask': f'before_after/binary_masks/{filename}',
                           'overlay': f'before_after/overlays/{filename}'},
                'after': {'image': '../' + point['image'], 'mask': '../' + point['binary_mask'],
                          'overlay': '../' + point['overlay']}})
            pixel_checks += 1
        before = overview(raw, alpha, source['roi_geometry']['xyxy_exclusive'],
                          [(p['point'], p['before_box']) for p in record['points']])
        after = overview(raw, alpha, source['roi_geometry']['xyxy_exclusive'],
                         [(p['point'], p['after_box']) for p in record['points']])
        before.save(dataset / 'analysis' / record['before_overview'])
        after.save(dataset / 'analysis' / record['after_overview'])
        if task == 616:
            preview(dataset, record, before, after)
        records.append(record)
        guard()
        del raw, alpha, old_image, old_alpha
        if ordinal % 12 == 0:
            print(f'Before/After: {ordinal}/72 captures', flush=True)
    payload = {'records': records, 'before_wh': [256, 256], 'after_wh': [256, 512],
               'native_scale_alignment': True}
    write_json(directory / 'comparison.json', payload)
    serialized = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    page = TEMPLATE.read_text(encoding='utf-8').replace('__DATA__', serialized)
    if (dataset / 'analysis/case_point_before_after/index.html').is_file():
        page = page.replace('</h1>', '</h1>\n<p class="links"><a href="case_point_before_after/index.html">Case별 · 포인트별 Before / After 수치 보기</a></p>', 1)
    for path, expected in protected.items():
        if digest(path) != expected:
            raise ValueError(f'Protected dataset file changed: {path}')
    for record in records:
        for key in ('before_overview', 'after_overview'):
            if not (dataset / 'analysis' / record[key]).is_file():
                raise ValueError('Missing overview.')
        for point in record['points']:
            for version in ('before', 'after'):
                for value in point[version].values():
                    if not (dataset / 'analysis' / value).is_file():
                        raise ValueError('Missing comparison asset.')
    if pixel_checks != 216:
        raise ValueError('Incomplete Before/After viewer.')
    (dataset / 'analysis/index.html').write_text(page, encoding='utf-8')
    shutil.copy2(__file__, directory / 'provenance/build_viewer.py')
    shutil.copy2(TEMPLATE, directory / 'provenance/viewer_template.html')
    verification = {'completed': True, 'native_before_after_pairs_verified': pixel_checks,
                    'captures': len(records), 'all_training_pixels_masks_and_splits_unchanged': True,
                    'all_comparison_assets_exist': True, 'before_wh': [256, 256], 'after_wh': [256, 512],
                    'same_display_scale': True, 'gray_before_margin_is_display_only': True,
                    'memory_budget_mib': memory_budget_mib, 'observed_peak_rss_mib': round(observed_mib, 2),
                    'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'builder_sha256': digest(__file__), 'template_sha256': digest(TEMPLATE),
                    'parent_manifest_sha256': digest(parent / 'manifest.json'),
                    'dataset_manifest_sha256': digest(dataset / 'manifest.json'),
                    'comparison_json_sha256': digest(directory / 'comparison.json')}
    write_json(directory / 'verification.json', verification)
    (directory / 'RUNNING.json').unlink()
    print(json.dumps({'entry': str(dataset / 'analysis/index.html'), **verification}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/changed_v3_p123_256x512')
    parser.add_argument('--parent', type=Path, default=ROOT / 'data/changed_v3')
    parser.add_argument('--memory-budget-mib', type=int, default=4096)
    args = parser.parse_args()
    build(args.dataset, args.parent, args.memory_budget_mib)
