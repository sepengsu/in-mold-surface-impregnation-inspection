"""Balance the fixed GT subset between two Roboflow projects and upload masks."""

import argparse
import base64
from collections import Counter
import contextlib
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import sys
import zlib

import cv2
import numpy as np

from src.preprocess.export_split_folders import digest
from src.training.data import read_gray

PROJECT = Path(__file__).resolve().parents[2]
PROJECT_IDS = ('namrc_gold', 'namrc_gold_v2')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def partition(rows, seed=42):
    """Integer constraints balance case, specimen, direction and central points."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    if len(rows) != 100:
        raise ValueError('This upload requires exactly the selected 100 TEST images.')
    conditions, lower, upper = [], [], []
    def add(indices, lo=None, hi=None):
        vector = np.zeros(len(rows))
        vector[indices] = 1
        conditions.append(vector)
        lower.append(len(indices) // 2 if lo is None else lo)
        upper.append((len(indices) + 1) // 2 if hi is None else hi)
    add(list(range(100)), 50, 50)
    cases = sorted({r['group'] for r in rows})
    if cases != ['case05', 'case10', 'case14']:
        raise ValueError('Unexpected TEST case membership.')
    target = {'case05': 17, 'case10': 17, 'case14': 16}
    for case in cases:
        add([i for i, r in enumerate(rows) if r['group'] == case], target[case], target[case])
    for keys in [('case', 'capture'), ('case', 'capture', 'position'), ('group', 'position')]:
        for values in sorted({tuple(r[k] for k in keys) for r in rows}):
            add([i for i, r in enumerate(rows) if tuple(r[k] for k in keys) == values])
    central = [i for i, r in enumerate(rows) if r['position'] == '000' and r.get('measurement_point') in ('P1', 'P2', 'P3')]
    add(central)
    for keys in [('group',), ('case', 'capture'), ('measurement_point',)]:
        for values in sorted({tuple(rows[i][k] for k in keys) for i in central}):
            add([i for i in central if tuple(rows[i][k] for k in keys) == values])
    matrix = np.array(conditions)
    result = milp(np.random.default_rng(seed).uniform(-1, 1, len(rows)), integrality=np.ones(len(rows)),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, lower, upper),
                  options={'time_limit': 30})
    if not result.success:
        raise ValueError(f'Cannot balance two groups: {result.message}')
    chosen = np.rint(result.x).astype(int)
    values = matrix @ chosen
    if not (np.all(values >= np.array(lower)) and np.all(values <= np.array(upper))):
        raise ValueError('Balanced assignment failed verification.')
    return {name: [r for i, r in enumerate(rows) if chosen[i] == value]
            for name, value in zip(PROJECT_IDS, (1, 0))}


def coco_annotation(image_name, binary):
    """Encode every 8-connected region with exact COCO RLE, including tiny ones."""
    from pycocotools import mask as coco_mask

    n, components, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    annotations, restored = [], np.zeros_like(binary, dtype=bool)
    for label in range(1, n):
        component = np.asfortranarray(components == label, dtype=np.uint8)
        rle = coco_mask.encode(component)
        restored |= coco_mask.decode(rle).astype(bool)
        x, y, w, h, area = map(int, stats[label])
        annotations.append({'id': label, 'image_id': 1, 'category_id': 1,
                            'segmentation': {'size': rle['size'], 'counts': rle['counts'].decode('ascii')},
                            'area': area, 'bbox': [x, y, w, h], 'iscrowd': 0})
    if not np.array_equal(restored, binary):
        raise ValueError('COCO RLE encoding changed mask pixels.')
    return {'images': [{'id': 1, 'file_name': image_name, 'height': binary.shape[0], 'width': binary.shape[1]}],
            'categories': [{'id': 1, 'name': 'Exposure', 'supercategory': 'Exposure'}],
            'annotations': annotations}


def prepare(source, output):
    selection_path = source / 'selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    rows = sorted(selection['records'], key=lambda r: r['image'])
    if len({r['image'] for r in rows}) != 100 or selection['source_split'] != 'test':
        raise ValueError('Source must be the verified 100-image TEST subset.')
    if output.exists():
        saved = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
        if saved['source_selection_sha256'] != digest(selection_path):
            raise ValueError('The GT selection changed; use a new output directory.')
        for r in saved['records']:
            for kind in ('image', 'mask'):
                if digest(output / r[kind]) != r[f'{kind}_sha256'] or digest(source / r[f'source_{kind}']) != r[f'{kind}_sha256']:
                    raise ValueError('A prepared file or GT source changed; preserving upload provenance.')
            if digest(output / r['annotation']) != r['annotation_sha256']:
                raise ValueError('Prepared annotation changed.')
        return saved
    groups = partition(rows)
    output.mkdir(parents=True)
    records, summaries = [], {}
    for project_id, selected in groups.items():
        for folder in ('images', 'masks', 'annotations'):
            (output / project_id / folder).mkdir(parents=True)
        for row in selected:
            copied = {**row, 'project_id': project_id, 'source_image': row['image'], 'source_mask': row['mask']}
            for kind, folder in (('image', 'images'), ('mask', 'masks')):
                original = (source / row[kind]).resolve()
                if not original.is_relative_to(source) or not original.is_file():
                    raise ValueError('Missing source or unsafe path.')
                target = Path(project_id) / folder / original.name
                shutil.copy2(original, output / target)
                copied[kind] = target.as_posix()
                copied[f'{kind}_sha256'] = digest(original)
                if digest(output / target) != copied[f'{kind}_sha256']:
                    raise ValueError('Copied file differs from source.')
            image = read_gray(output / copied['image'])
            binary = read_gray(output / copied['mask']) > 0
            if image.shape != (256, 256) or binary.shape != image.shape:
                raise ValueError('Expected aligned 256 x 256 images and masks.')
            annotation = coco_annotation(Path(copied['image']).name, binary)
            annotation_path = Path(project_id) / 'annotations' / (Path(copied['image']).stem + '.coco.json')
            write_json(output / annotation_path, annotation)
            copied.update({'annotation': annotation_path.as_posix(), 'annotation_sha256': digest(output / annotation_path),
                           'exposure_pixels': int(binary.sum()), 'regions': len(annotation['annotations'])})
            records.append(copied)
        summaries[project_id] = {'images': len(selected), 'cases': dict(sorted(Counter(r['group'] for r in selected).items())),
                                'specimens': dict(sorted(Counter(f"{r['case']}/{r['capture']}" for r in selected).items())),
                                'directions': dict(sorted(Counter(r['position'] for r in selected).items())),
                                'central_000_points': sum(r['position'] == '000' and r.get('measurement_point') in ('P1', 'P2', 'P3') for r in selected)}
    plan = {'source': str(source), 'source_selection_sha256': digest(selection_path), 'seed': 42,
            'split': 'test', 'annotation_format': 'COCO RLE, 8-connected components, no pixel filtering',
            'projects': summaries, 'records': records}
    write_json(output / 'plan.json', plan)
    fields = ['project_id', 'image', 'mask', 'annotation', 'group', 'capture', 'position', 'measurement_point', 'exposure_pixels', 'regions']
    with (output / 'files.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: r.get(key, '') for key in fields} for r in records)
    print(json.dumps(summaries, indent=2), flush=True)
    return plan


def connect(config_path):
    import yaml
    import roboflow

    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config = config.get('roboflow_api_upload', config)
    key = os.environ.get(config.get('api_key_env', 'ROBOFLOW_API_KEY')) or config.get('api_key')
    if not key:
        raise ValueError('No API key in the existing config/environment.')
    with contextlib.redirect_stdout(io.StringIO()):
        rf = roboflow.Roboflow(api_key=key)
        workspace = rf.workspace(config.get('workspace')) if config.get('workspace') else rf.workspace()
        available = {p['id'].split('/')[-1]: p for p in workspace.project_list}
        for name in PROJECT_IDS:
            if name not in available or available[name]['type'] != 'instance-segmentation':
                raise ValueError(f'Expected existing Instance Segmentation project: {name}')
        projects = {name: workspace.project(name) for name in PROJECT_IDS}
    return key, workspace.url, projects


def upload(plan, output, config_path, limit=None):
    try:
        key, workspace, projects = connect(config_path)
    except Exception as error:
        # Authentication errors can include a URL containing the API key.
        print(f'Roboflow connection failed ({type(error).__name__}). Check the existing private config.', file=sys.stderr)
        return 1
    receipt_path = output / 'receipts.json'
    receipts = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {'workspace': workspace, 'records': {}}
    if receipts['workspace'] != workspace:
        raise ValueError('Workspace differs from the saved receipts.')
    completed_now = 0
    try:
        for row in plan['records']:
            record_key = row['image']
            state = receipts['records'].get(record_key, {})
            if state.get('annotation_success'):
                continue
            if limit is not None and completed_now >= limit:
                break
            project = projects[row['project_id']]
            if not state.get('image_id'):
                response, _, _ = project.upload_image(str(output / row['image']), split='test',
                    num_retry_uploads=0, batch_name='gold100-balanced-20261008',
                    tag_names=['gold100-balanced-20261008', row['group'], f"specimen-{row['capture']}", f"direction-{row['position']}"],
                    metadata={'case': row['group'], 'specimen': row['capture'], 'direction': row['position'],
                              'measurement_point': row.get('measurement_point') or '', 'source_sha256': row['image_sha256']})
                if not response.get('id') or not (response.get('success') or response.get('duplicate')):
                    raise ValueError('Image upload did not return a successful receipt.')
                state.update({'image_id': response['id'], 'project_id': row['project_id'],
                              'duplicate': bool(response.get('duplicate')), 'image_sha256': row['image_sha256']})
                receipts['records'][record_key] = state
                write_json(receipt_path, receipts)
            annotation, _, _ = project.save_annotation(str(output / row['annotation']), image_id=state['image_id'],
                job_name='gold100-balanced-20261008', is_prediction=False, annotation_overwrite=False, num_retry_uploads=0)
            if not annotation.get('success'):
                raise ValueError(f'Annotation was not confirmed: {annotation}')
            state.update({'annotation_success': True, 'annotation_response': annotation,
                          'annotation_sha256': row['annotation_sha256']})
            write_json(receipt_path, receipts)
            completed_now += 1
            print(f"{row['project_id']}: {sum(r.get('annotation_success', False) for r in receipts['records'].values())}/100 | {Path(row['image']).name}", flush=True)
    except Exception as error:
        print(f'{type(error).__name__}: {str(error).replace(str(key), "[REDACTED]")}', file=sys.stderr, flush=True)
        return 1
    return 0


def verify(plan, output, config_path):
    """Read every remote annotation back and compare its pixels with the source."""
    from pycocotools import mask as coco_mask

    try:
        key, workspace, projects = connect(config_path)
    except Exception as error:
        print(f'Roboflow connection failed ({type(error).__name__}).', file=sys.stderr)
        return 1
    report = {'workspace': workspace, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'complete': False, 'projects': {}, 'records': []}
    report_path = output / 'verification.json'
    try:
        receipts = json.loads((output / 'receipts.json').read_text(encoding='utf-8'))
        if receipts['workspace'] != workspace:
            raise ValueError('Workspace differs from the upload receipts.')
        for name, project in projects.items():
            expected = [r for r in plan['records'] if r['project_id'] == name]
            states = [receipts['records'][r['image']] for r in expected]
            if not all(s.get('annotation_success') and s['project_id'] == name for s in states):
                raise ValueError(f'{name}: some annotations have not been uploaded.')
            ids = {s['image_id'] for s in states}
            remote = project.search(limit=100, fields=['id', 'name', 'split'])
            if len(ids) != 50 or {r['id'] for r in remote} != ids:
                raise ValueError(f'{name}: remote image membership differs from the 50-image plan.')
            report['projects'][name] = {**plan['projects'][name], 'remote_images': len(remote),
                                       'url': f'https://app.roboflow.com/{workspace}/{name}'}
            for row, state in zip(expected, states):
                detail = project.image(state['image_id'])
                if detail['split'] != 'test' or detail['name'] != Path(row['image']).name:
                    raise ValueError('Remote filename or TEST split differs.')
                gt = read_gray(output / row['mask']) > 0
                restored = np.zeros_like(gt)
                boxes = (detail.get('annotation') or {}).get('boxes', [])
                for box in boxes:
                    if box['type'] != 'mask' or box['label'] != 'Exposure':
                        raise ValueError('Unexpected remote annotation type/class.')
                    # Roboflow returns the COCO counts compressed with zlib/base64.
                    counts = zlib.decompress(base64.b64decode(box['mask']))
                    restored |= coco_mask.decode({'size': list(gt.shape), 'counts': counts}).astype(bool)
                difference = int(np.count_nonzero(restored != gt))
                result = {'image': row['image'], 'project_id': name, 'image_id': state['image_id'],
                          'split': detail['split'], 'source_exposure_pixels': int(gt.sum()),
                          'remote_exposure_pixels': int(restored.sum()), 'mismatch_pixels': difference,
                          'source_regions': row['regions'], 'remote_regions': len(boxes)}
                report['records'].append(result)
                write_json(report_path, report)
                if difference or len(boxes) != row['regions']:
                    raise ValueError(f"Remote mask differs: {row['image']}")
                if len(report['records']) % 10 == 0:
                    print(f"Verified {len(report['records'])}/100 remote masks: pixel differences = 0", flush=True)
        report['complete'] = len(report['records']) == 100
        report['total_mismatch_pixels'] = sum(r['mismatch_pixels'] for r in report['records'])
        write_json(report_path, report)
        write_report(plan, report, output)
    except Exception as error:
        print(f'{type(error).__name__}: {str(error).replace(str(key), "[REDACTED]")}', file=sys.stderr, flush=True)
        return 1
    return 0


def write_report(plan, report, output):
    lines = ['# Roboflow Gold 100 업로드', '',
             '- 원본: `data/cropped/test` 252장 중 기존에 균등 선정한 `data/gt` 100장.',
             '- 기존 API Key와 Workspace를 사용하고 프로젝트만 지정했습니다.',
             '- 이미지와 기존 Exposure 마스크를 함께 업로드했습니다. 모든 이미지는 `test`입니다.',
             '- 두 프로젝트 간 이미지 중복은 없으며, 원본 데이터셋과 Train/Validation은 변경하지 않았습니다.',
             '- Case·시편·촬영 방향·중앙 P1/P2/P3를 균형 배정했습니다. 예측값이나 오차는 선정에 사용하지 않았습니다.',
             '', '| 프로젝트 | Case 05 | Case 10 | Case 14 | 합계 | 000 / 180 | 중앙 000 P1·P2·P3 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, p in report['projects'].items():
        c, d = p['cases'], p['directions']
        lines.append(f"| [{name}]({p['url']}) | {c['case05']} | {c['case10']} | {c['case14']} | {p['remote_images']} | {d.get('000', 0)} / {d.get('180', 0)} | {p['central_000_points']} |")
    lines += ['', '## 마스크 보존 검증', '',
              '마스크의 양수 픽셀을 Exposure로 보고, 8방향 연결 영역별 COCO RLE로 변환했습니다. '
              '1픽셀 영역과 내부 구멍도 보존하며, 면적 필터나 다각형 근사화를 적용하지 않았습니다.', '',
              f"서버의 100개 마스크를 다시 읽고 복원한 결과 원본과 다른 픽셀은 **{report['total_mismatch_pixels']}개**입니다. "
              '영역 수, 파일명, Test 분할, 프로젝트별 50개 이미지 ID도 확인했습니다.', '',
              f"검증 시각(UTC): {report['checked_at_utc']}", '',
              '## 파일', '', '- `plan.json`, `files.csv`: 프로젝트별 배정과 원본 추적 정보.',
              '- `receipts.json`: 이미지 ID 및 업로드 성공 기록.',
              '- `verification.json`: 이미지별 서버 마스크 비교 결과.',
              '- 각 프로젝트의 `images/`, `masks/`, `annotations/`: 업로드 이미지·원본 마스크·COCO RLE.', '',
              '## PowerShell', '', '```powershell', 'Set-Location D:\\Code\\vision_data',
              '# 업로드 없이 서버 결과 재검증',
              '.\\.venv\\Scripts\\python.exe -m src.segmentation.upload_gold_projects --verify',
              '# 중단된 업로드 재개 후 검증: 완료된 항목은 receipts.json 기준으로 건너뜀',
              '.\\.venv\\Scripts\\python.exe -m src.segmentation.upload_gold_projects --upload --verify',
              '```', '',
              '이 마스크들은 기존 라벨이며, 정밀 수정을 완료한 Gold 정답을 뜻하지 않습니다. '
              '수정 후에는 Test 평가에 사용하고, Test 라벨로 학습하거나 임계값을 고르지 않습니다.', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=PROJECT / 'data/gt')
    p.add_argument('--output', type=Path, default=PROJECT / 'data/upload/namrc_gold_20261008')
    p.add_argument('--config', type=Path, default=PROJECT / 'config/roboflow_api_upload.yaml')
    p.add_argument('--upload', action='store_true')
    p.add_argument('--verify', action='store_true', help='Read all remote masks and verify exact pixel equality.')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    plan = prepare(args.source.resolve(), args.output.resolve())
    if args.upload:
        status = upload(plan, args.output.resolve(), args.config.resolve(), args.limit)
        if status:
            return status
    if args.verify:
        return verify(plan, args.output.resolve(), args.config.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
