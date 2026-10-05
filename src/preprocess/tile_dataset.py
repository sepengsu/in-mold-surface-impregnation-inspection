"""Export interior image/mask pairs from a verified Label Studio image manifest.

Example: python -m src.preprocess.tile_dataset --matches matches.json --output new_dir
Use --count-only to inspect counts without decoding labels or writing PNG files.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ._validation import grayscale
from .interior_tiling import plan_interior_tiling
from .roi_centered_tiling import plan_roi_centered_tiling
from ..segmentation.label_studio import task_brush_alpha

LOGGER = logging.getLogger(__name__)


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _verified_bytes(path: Path, expected_hash: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return raw


def _write_png(path: Path, pixels: np.ndarray) -> None:
    ok, encoded = cv2.imencode('.png', pixels)
    if not ok:
        raise OSError(f'Cannot encode {path}')
    with path.open('xb') as stream:
        stream.write(encoded.tobytes())


def _write_json(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def export_tiles(
    matches_path: Path, output: Path | None = None, *, export_path: Path | None = None,
    patch_size: int = 256, stride: int = 256, edge_margin: int = 0,
    black_threshold: int = 0, border_tiles: int = 0,
    position: str = 'all', count_only: bool = False,
    alignment: str = 'image_center', roi_results: Path | None = None,
    selection: str = 'all', roi_review_policy: str = 'include',
) -> dict:
    """Process source tasks once, keeping geometry independent of annotations.

    An output directory must be new. Interrupted exports remain inspectable via
    RUNNING.json; summary.json is written only after a successful full batch.
    Input/export hashes are checked, and duplicate task IDs are rejected before
    writing. Masks retain alpha values; use >0 for binary Exposure membership.
    """
    if position not in ('all', '000', '180'):
        raise ValueError('position must be all, 000 or 180')
    if alignment not in ('image_center', 'roi_centerline'):
        raise ValueError('alignment must be image_center or roi_centerline')
    if selection not in ('all', 'central_three') or roi_review_policy not in ('include','exclude','error'):
        raise ValueError('Invalid tile selection or ROI review policy')
    if alignment == 'image_center' and (roi_results is not None or selection != 'all'):
        raise ValueError('roi_results and central_three require roi_centerline alignment')
    if alignment == 'roi_centerline' and (roi_results is None or border_tiles or stride != patch_size):
        raise ValueError('roi_centerline requires --roi-results, stride == patch-size and border-tiles == 0')
    # Validate parameters even when a direction has no matching sources.
    plan_interior_tiling(np.ones((1, 1), dtype=np.uint8), patch_size, stride,
                         edge_margin=edge_margin, black_threshold=black_threshold,
                         border_tiles=border_tiles)
    matches_path = Path(matches_path).resolve()
    source_manifest = json.loads(matches_path.read_text(encoding='utf-8'))
    rows = [r for r in source_manifest['matches'] if position == 'all' or r['position'] == position]
    roi_rows, skipped_roi_tasks, roi_hash = {}, [], None
    if alignment == 'roi_centerline':
        roi_results = Path(roi_results).resolve()
        roi_hash = hashlib.sha256(roi_results.read_bytes()).hexdigest()
        with roi_results.open(encoding='utf-8-sig',newline='') as stream:
            for boundary in csv.DictReader(stream):
                if not boundary.get('output_path'):
                    continue
                source = _path(boundary['output_path'],roi_results.parent)
                if source in roi_rows:
                    raise ValueError(f'Duplicate ROI boundary record: {source}')
                roi_rows[source] = boundary
        retained = []
        for row in rows:
            source = _path(row['local_roi_image'],matches_path.parent)
            if source not in roi_rows:
                raise ValueError(f'No matching ROI boundaries for {source}')
            boundary = roi_rows[source]
            if boundary.get('found') != '1' or boundary.get('image_output') != 'masked':
                raise ValueError(f'ROI was not detected or source coordinates were changed: {source}')
            if boundary.get('needs_review') == '1':
                if roi_review_policy == 'error':
                    raise ValueError(f'ROI requires review: {source}')
                if roi_review_policy == 'exclude':
                    skipped_roi_tasks.append(row['task_id'])
                    continue
            retained.append(row)
        rows = retained
    if not rows:
        raise ValueError(f'No source images for position {position}')
    task_ids = [r['task_id'] for r in rows]
    if any(type(i) is not int or i < 0 for i in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError('Task IDs must be unique nonnegative integers')
    tasks = {}
    if not count_only:
        if output is None:
            raise ValueError('output is required when exporting tiles')
        export_path = (Path(export_path).resolve() if export_path is not None else
                       _path(source_manifest['source_export'], matches_path.parent))
        export_data = json.loads(_verified_bytes(export_path, source_manifest['source_export_sha256']))
        for task in export_data:
            if task['id'] in tasks:
                raise ValueError(f"Duplicate export task {task['id']}")
            tasks[task['id']] = task
        missing = set(task_ids) - tasks.keys()
        if missing:
            raise ValueError(f'Missing annotations for tasks: {sorted(missing)}')
    if output is not None:
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / 'RUNNING.json', {'matches': str(matches_path), 'count_only': count_only})
        if not count_only:
            (output / 'images').mkdir()
            (output / 'masks').mkdir()
    settings = dict(patch_size=patch_size, stride=stride, edge_margin=edge_margin,
                    black_threshold=black_threshold, border_tiles=border_tiles,
                    position=position, alignment=alignment, selection=selection, padding=False, resized=False)
    if roi_results is not None:
        settings.update(roi_results=str(roi_results),roi_results_sha256=roi_hash,
                        roi_review_policy=roi_review_policy,
                        center_rule='y0=floor((top(cx)+bottom(cx))/2-size/2); cx=x0+(size-1)/2',
                        horizontal_extent='camera field of view; no inferred specimen endpoints')
    records, reasons, per_position = [], Counter(), Counter()
    for index, row in enumerate(rows, 1):
        source = _path(row['local_roi_image'], matches_path.parent)
        raw = _verified_bytes(source, row['roi_sha256'])
        pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        grayscale(pixels)
        if pixels.shape != (row['height'], row['width']):
            raise ValueError(f'Source dimensions changed for task {row["task_id"]}')
        roi_metadata = None
        if alignment == 'roi_centerline':
            boundary = roi_rows[source]
            top = (float(boundary['top_slope']),float(boundary['top_intercept']))
            bottom = (float(boundary['bottom_slope']),float(boundary['bottom_intercept']))
            layout = plan_roi_centered_tiling(pixels,top,bottom,patch_size,stride,
                edge_margin=edge_margin,black_threshold=black_threshold,selection=selection)
            central_kept = [xy for xy in layout.positions if xy in layout.central_points]
            if len(central_kept) != 3:
                raise ValueError(f'Task {row["task_id"]}: not all three fixed ROI-center points are valid')
            roi_metadata = {'top_line':top,'bottom_line':bottom,'centerline':layout.centerline,
                'center_y_at_image_center':layout.centerline[0]*((pixels.shape[1]-1)/2)+layout.centerline[1],
                'needs_review':boundary.get('needs_review')=='1',
                'review_reason':boundary.get('needs_review_reason') or boundary.get('reason',''),
                'quality_status':boundary.get('quality_status') or boundary.get('status'),
                'manually_verified':False}
        else:
            layout = plan_interior_tiling(pixels, patch_size, stride, edge_margin=edge_margin,
                                          black_threshold=black_threshold, border_tiles=border_tiles)
        alpha = None
        if not count_only:
            alpha = task_brush_alpha(tasks[row['task_id']])
            if alpha.shape != pixels.shape:
                raise ValueError(f'Image/mask dimensions differ for task {row["task_id"]}')
        tiles = []
        for x, y in layout.positions:
            filename = f"task_{row['task_id']}_x{x:04d}_y{y:04d}.png"
            tile = {'xyxy_exclusive': [x, y, x + patch_size, y + patch_size],
                    'image': None, 'mask': None, 'label_pixels': None}
            if roi_metadata is not None:
                tile.update(grid_index=list(layout.grid_indices[(x,y)]),
                            measurement_point=layout.central_points.get((x,y)))
            if output is not None and not count_only:
                patch = pixels[y:y + patch_size, x:x + patch_size]
                mask = alpha[y:y + patch_size, x:x + patch_size]
                _write_png(output / 'images' / filename, patch)
                _write_png(output / 'masks' / filename, mask)
                tile.update(image=f'images/{filename}', mask=f'masks/{filename}',
                            label_pixels=int(np.count_nonzero(mask)))
            tiles.append(tile)
        reasons.update(layout.rejected)
        per_position[row['position']] += len(tiles)
        records.append({'task_id': row['task_id'], 'case': row['case'], 'capture': row['capture'],
                        'position': row['position'], 'source_image': str(source),
                        'source_sha256': row['roi_sha256'], 'annotation_ids': row.get('annotation_ids', []),
                        'image_size_wh': [pixels.shape[1], pixels.shape[0]],
                        'grid_bounds_xyxy_exclusive': layout.grid_bounds,
                        'candidate_tiles': len(layout.candidate_positions),
                        'accepted_tiles': len(tiles), 'rejected': layout.rejected, 'tiles': tiles,
                        **({'roi_geometry':roi_metadata} if roi_metadata is not None else {})})
        if index % 10 == 0 or index == len(rows):
            LOGGER.info('Processed %d/%d images; kept %d tiles', index, len(rows), sum(per_position.values()))
    counts = [r['accepted_tiles'] for r in records]
    summary = {'source_matches': str(matches_path), 'source_export': str(export_path) if export_path else None,
               'settings': settings, 'source_images': len(rows),
               'candidate_tiles': sum(r['candidate_tiles'] for r in records),
               'accepted_tiles': sum(counts), 'rejected_tiles': sum(reasons.values()),
               'rejection_reasons': dict(reasons), 'tiles_by_position': dict(per_position),
               'tiles_per_image_min': min(counts), 'tiles_per_image_max': max(counts),
               'images_without_tiles': sum(n == 0 for n in counts),
               'image_png_count': 0 if count_only else sum(counts),
               'mask_png_count': 0 if count_only else sum(counts),
               'count_only': count_only, 'mask_policy': 'original alpha; Exposure membership alpha > 0'}
    if alignment == 'roi_centerline':
        summary.update(roi_review_flagged_images=sum(r['roi_geometry']['needs_review'] for r in records),
                       roi_review_excluded_tasks=skipped_roi_tasks,
                       central_three_tiles=sum(t['measurement_point'] is not None for r in records for t in r['tiles']),
                       central_three_000_tiles=sum(t['measurement_point'] is not None for r in records
                                                   if r['position']=='000' for t in r['tiles']))
    if output is not None:
        _write_json(output / 'manifest.json', {'summary': summary, 'images': records})
        _write_json(output / 'summary.json', summary)
        (output / 'RUNNING.json').unlink()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', type=Path, required=True, help='Verified image-match JSON')
    parser.add_argument('--export', type=Path, dest='export_path', help='Override the source Label Studio JSON path')
    parser.add_argument('--output', type=Path, help='New output directory (required unless --count-only)')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--edge-margin', type=int, default=0, help='Extra valid pixels required on every side')
    parser.add_argument('--black-threshold', type=int, default=0, help='Pixels <= this value are excluded')
    parser.add_argument('--border-tiles', type=int, default=0, help='Drop this many outer rings of the full centered grid')
    parser.add_argument('--position', choices=('all', '000', '180'), default='all')
    parser.add_argument('--count-only', action='store_true', help='Count without decoding/saving masks or tile PNGs')
    parser.add_argument('--alignment',choices=('image_center','roi_centerline'),default='image_center')
    parser.add_argument('--roi-results',type=Path,help='ROI boundary CSV belonging to the source masked images')
    parser.add_argument('--selection',choices=('all','central_three'),default='all')
    parser.add_argument('--roi-review-policy',choices=('include','exclude','error'),default='include',
                        help='Preserve review flags by default; never treat these boundaries as manually verified')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    try:
        summary = export_tiles(args.matches, args.output, export_path=args.export_path,
                               patch_size=args.patch_size, stride=args.stride, edge_margin=args.edge_margin,
                               black_threshold=args.black_threshold, border_tiles=args.border_tiles,
                               position=args.position, count_only=args.count_only, alignment=args.alignment,
                               roi_results=args.roi_results,selection=args.selection,roi_review_policy=args.roi_review_policy)
    except (OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
        LOGGER.error('%s', exc)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
