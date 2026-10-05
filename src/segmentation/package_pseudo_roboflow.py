"""Package reviewed or quarantined pseudo labels as a Roboflow YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import yaml


PACKAGEABLE_STATUSES = {'draft', 'empty_draft', 'roi_review', 'empty_roi_review'}
REVIEW_STATUSES = {'roi_review', 'empty_roi_review'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
MAPPING_FIELDS = (
    'source', 'roi_image', 'packaged_image', 'packaged_label', 'status', 'roi_status',
    'label_review_status', 'candidate_flags', 'candidate_method', 'region_count',
    'threshold_native', 'low_threshold_native', 'period_x', 'period_y',
    'period_x_correlation', 'period_y_correlation', 'consensus_exposures_used',
    'consensus_min', 'boundary_guard_px',
)


def safe_relative(value: str, description: str) -> Path:
    relative = Path(value)
    windows = PureWindowsPath(value)
    if (not value or relative.is_absolute() or windows.is_absolute() or windows.drive
            or '..' in relative.parts or '..' in windows.parts):
        raise ValueError(f'{description} is not a safe relative path: {value}')
    return relative


def resolve_inside(root: Path, value: str, description: str) -> tuple[Path, Path]:
    relative = safe_relative(value, description)
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'{description} escapes its root: {value}')
    return path, relative


def packaged_stem(relative: Path) -> str:
    readable = '__'.join(relative.with_suffix('').parts)
    readable = re.sub(r'[^A-Za-z0-9._-]+', '-', readable).strip('-_.') or 'image'
    digest = hashlib.sha256(relative.as_posix().encode('utf-8')).hexdigest()[:10]
    return f'{readable}__{digest}'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'source', 'status', 'label_path', 'exposure', 'region_count'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('pseudo_results.csv is missing required columns')
        return list(reader)


def read_class_name(pseudo_root: Path, requested: str | None) -> tuple[str, dict]:
    manifest_path = pseudo_root / 'run_manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'pseudo manifest not found: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    names = manifest.get('class_names') or []
    class_name = requested or (names[0] if len(names) == 1 else None)
    if not class_name or not isinstance(class_name, str) or any(c in class_name for c in '\r\n\0'):
        raise ValueError('one valid class name is required')
    return class_name, manifest


def read_roi_images(report: Path, mode: str, exposure: str) -> dict[str, Path]:
    if not report.is_file():
        raise ValueError(f'ROI report not found: {report}')
    with report.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'source', 'mode', 'exposure', 'image_output', 'output_path'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('ROI CSV is missing masked-image columns')
        rows = [row for row in reader if row['mode'] == mode
                and row['exposure'].casefold() == exposure.casefold()]
    images = {}
    for row in rows:
        if row['image_output'] != 'masked':
            raise ValueError('ROI report must contain original-size masked output')
        source = safe_relative(row['source'], 'ROI source').as_posix()
        path, _ = resolve_inside(report.parent, row['output_path'], 'ROI output_path')
        if source in images:
            raise ValueError(f'duplicate ROI image row: {source}')
        if not path.is_file():
            raise ValueError(f'masked ROI image not found: {path}')
        images[source] = path
    if not images:
        raise ValueError(f'no ROI images for mode={mode}, exposure={exposure}')
    return images


def validate_yolo_label(path: Path) -> int:
    objects = 0
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] != '0' or len(tokens) < 7 or (len(tokens) - 1) % 2:
                raise ValueError(f'invalid YOLO segmentation at {path}:{line_number}')
            try:
                coordinates = [float(value) for value in tokens[1:]]
            except ValueError as exc:
                raise ValueError(f'nonnumeric YOLO coordinate at {path}:{line_number}') from exc
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
                raise ValueError(f'YOLO coordinate outside [0, 1] at {path}:{line_number}')
            objects += 1
    return objects


def create_zip(source: Path, destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f'zip output already exists: {destination}')
    if destination.is_relative_to(source.resolve()):
        raise ValueError('zip output must not be inside the dataset directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', type=Path, default=Path('data/0918'))
    parser.add_argument('--pseudo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New or empty Roboflow dataset directory')
    parser.add_argument('--zip', dest='zip_path', type=Path, help='Optional ZIP for manual upload')
    parser.add_argument('--exposure', default='10000us')
    parser.add_argument('--expected-count', type=int, default=0, help='Fail unless this many images are packaged')
    parser.add_argument('--class-name')
    parser.add_argument('--image-source', choices=('masked-roi', 'original'), default='masked-roi')
    parser.add_argument('--roi-report', type=Path, help='Override the ROI CSV recorded in the pseudo manifest')
    parser.add_argument('--include-review', action='store_true', help='Include quarantined roi_review rows')
    parser.add_argument('--require-preserve-small', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate every image and label without writing files')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_count < 0:
        parser.error('--expected-count must be nonnegative')
    if not re.fullmatch(r'\d+us', args.exposure, re.IGNORECASE):
        parser.error('--exposure must be a folder such as 10000us')
    args.source, args.pseudo_root, args.output = (
        path.resolve() for path in (args.source, args.pseudo_root, args.output)
    )
    if not args.source.is_dir():
        parser.error(f'source directory not found: {args.source}')
    if not args.pseudo_root.is_dir():
        parser.error(f'pseudo root not found: {args.pseudo_root}')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('output must be a new or empty directory')

    try:
        class_name, pseudo_manifest = read_class_name(args.pseudo_root, args.class_name)
        if args.require_preserve_small and not pseudo_manifest.get('parameters', {}).get('preserve_small'):
            raise ValueError('pseudo labels were not generated with --preserve-small')
        report_path = args.pseudo_root / 'pseudo_results.csv'
        rows = read_csv(report_path)
        roi_images = None
        roi_report = None
        if args.image_source == 'masked-roi':
            manifest_parameters = pseudo_manifest.get('parameters', {})
            roi_report_value = args.roi_report or manifest_parameters.get('roi_report')
            if not roi_report_value:
                raise ValueError('masked-roi packaging requires --roi-report or manifest parameters.roi_report')
            roi_report = Path(roi_report_value).resolve()
            expected_hash = pseudo_manifest.get('roi_report_sha256')
            actual_hash = hashlib.sha256(roi_report.read_bytes()).hexdigest()
            if expected_hash and actual_hash != expected_hash:
                raise ValueError('ROI report does not match the report used to create the pseudo labels')
            mode = manifest_parameters.get('mode')
            if not mode:
                raise ValueError('pseudo manifest is missing parameters.mode')
            roi_images = read_roi_images(roi_report, mode, args.exposure)
        entries = []
        output_names: set[str] = set()
        status_counts = Counter()
        total_regions = 0
        for row in rows:
            if row['exposure'].casefold() != args.exposure.casefold():
                continue
            status = row['status']
            if status not in PACKAGEABLE_STATUSES:
                raise ValueError(f"unpackageable row status={status}: {row['source']}")
            if status in REVIEW_STATUSES and not args.include_review:
                continue
            original_path, source_relative = resolve_inside(args.source, row['source'], 'source')
            label_path, _ = resolve_inside(args.pseudo_root, row['label_path'], 'label_path')
            if not original_path.is_file() or original_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f'original image not found or unsupported: {original_path}')
            if source_relative.parent.name.casefold() != args.exposure.casefold():
                raise ValueError(f'exposure disagrees with source path: {source_relative}')
            image_path = roi_images[source_relative.as_posix()] if roi_images is not None else original_path
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f'training image has an unsupported extension: {image_path}')
            if not label_path.is_file():
                raise ValueError(f'label file not found: {label_path}')
            objects = validate_yolo_label(label_path)
            try:
                recorded_regions = int(row['region_count'])
            except ValueError as exc:
                raise ValueError(f"invalid region_count for {row['source']}") from exc
            if objects != recorded_regions:
                raise ValueError(
                    f"label/CSV region count mismatch for {row['source']}: {objects} != {recorded_regions}"
                )
            stem = packaged_stem(source_relative)
            output_key = stem.casefold()
            if output_key in output_names:
                raise ValueError(f'packaged filename collision: {source_relative}')
            output_names.add(output_key)
            entries.append((row, image_path, label_path, stem, source_relative))
            status_counts[status] += 1
            total_regions += objects
        if args.expected_count and len(entries) != args.expected_count:
            raise ValueError(f'expected {args.expected_count} images, found {len(entries)}')
        if not entries:
            raise ValueError('no matching pseudo labels to package')
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(f'images={len(entries)}')
        print(f'labels={len(entries)}')
        print(f'regions={total_regions}')
        print('statuses=' + json.dumps(dict(status_counts), ensure_ascii=False))
        print('dry_run=1')
        return 0

    image_root = args.output / 'train' / 'images'
    label_root = args.output / 'train' / 'labels'
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    mapping_rows = []
    for row, image_path, label_path, stem, source_relative in entries:
        image_output = image_root / f'{stem}{image_path.suffix.lower()}'
        label_output = label_root / f'{stem}.txt'
        shutil.copy2(image_path, image_output)
        shutil.copy2(label_path, label_output)
        mapping_rows.append({
            key: value for key, value in {
                **row,
                'roi_image': str(image_path) if args.image_source == 'masked-roi' else '',
                'packaged_image': image_output.relative_to(args.output).as_posix(),
                'packaged_label': label_output.relative_to(args.output).as_posix(),
            }.items() if key in MAPPING_FIELDS
        })

    with (args.output / 'source_mapping.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_FIELDS)
        writer.writeheader()
        writer.writerows(mapping_rows)
    (args.output / 'classes.txt').write_text(class_name + '\n', encoding='utf-8')
    data_yaml = {'path': '.', 'train': 'train/images', 'names': {0: class_name}}
    with (args.output / 'data.yaml').open('x', encoding='utf-8') as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False, allow_unicode=True)
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'format': 'YOLO segmentation',
        'source': str(args.source),
        'pseudo_root': str(args.pseudo_root),
        'pseudo_results_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'exposure': args.exposure,
        'image_source': args.image_source,
        'roi_report': str(roi_report) if roi_report is not None else None,
        'image_count': len(entries),
        'label_count': len(entries),
        'region_count': total_regions,
        'status_counts': dict(status_counts),
        'class_names': [class_name],
        'candidate_method': pseudo_manifest.get('parameters', {}).get('candidate_method', 'gaussian_otsu'),
        'hysteresis_low_ratio': pseudo_manifest.get('parameters', {}).get('hysteresis_low_ratio'),
        'hysteresis_close_kernel': pseudo_manifest.get('parameters', {}).get('hysteresis_close_kernel'),
        'periodic_consensus': {
            key: pseudo_manifest.get('parameters', {}).get(key)
            for key in (
                'consensus_exposures', 'consensus_min', 'period_min', 'period_max',
                'period_fallback_x', 'period_fallback_y', 'period_min_correlation',
                'illumination_sigma', 'seed_z', 'support_z', 'growth_steps',
                'boundary_guard',
            )
        },
        'includes_roi_review': any(row['status'] in REVIEW_STATUSES for row, *_ in entries),
        'label_review_status': 'pending',
        'warning': 'Pseudo labels are candidate annotations and require human review before GT use.',
    }
    with (args.output / 'package_manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    if args.zip_path:
        try:
            create_zip(args.output, args.zip_path)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    print(f'images={len(entries)}')
    print(f'labels={len(entries)}')
    print(f'regions={total_regions}')
    print('statuses=' + json.dumps(dict(status_counts), ensure_ascii=False))
    print(f'dataset={args.output}')
    if args.zip_path:
        print(f'zip={args.zip_path.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
