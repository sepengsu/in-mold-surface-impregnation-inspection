"""Evaluate a revised Label Studio export against frozen A-model TEST predictions."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.segmentation.label_studio import task_brush_alpha
from src.training.data import read_gray
from src.inference.report_signed_error import overlap_pixels

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/raw/rectangle_roi_tiles_256'
BASE = ROOT / 'analysis/9point'
CENTRAL = ROOT / 'analysis/p123_inference_A_20261001_090710/tiles_all'
RUN = ROOT / 'outputs/training/unet_rectangle_A_20261001_090710_311'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_file(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latest_annotation(task):
    candidates = [a for a in task.get('annotations', []) if not a.get('was_cancelled', False)]
    if not candidates:
        raise ValueError(f"Task {task['id']} has no completed annotation.")
    # Never union older and newer annotation versions: that would undo erasures.
    def order(a):
        timestamp = a.get('updated_at') or a.get('created_at')
        if not timestamp:
            raise ValueError('Annotation timestamp is required for revision selection.')
        return datetime.fromisoformat(timestamp.replace('Z', '+00:00')), a['id']
    selected = max(candidates, key=order)
    return selected, [a['id'] for a in candidates if a is not selected]


def measure(truth, predicted):
    if truth.shape != (256, 256) or predicted.shape != truth.shape:
        raise ValueError('Expected aligned native 256 x 256 masks.')
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    truth_pct, pred_pct = 100 * (1 - float(truth.mean())), 100 * (1 - float(predicted.mean()))
    signed = truth_pct - pred_pct
    assert signed == 100 * (fp - fn) / 65536
    return dict(truth_pct=truth_pct, pred_pct=pred_pct, signed_error_pp=signed,
                absolute_error_pp=abs(signed), tp=tp, fp=fp, fn=fn,
                iou=tp / (tp + fp + fn) if tp + fp + fn else 1.,
                dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.)


def aggregate(rows, version):
    tp, fp, fn = (sum(r[f'{version}_{key}'] for r in rows) for key in ('tp', 'fp', 'fn'))
    return dict(tiles=len(rows), mean_absolute_error_pp=float(np.mean([r[f'{version}_absolute_error_pp'] for r in rows])),
                mean_signed_error_pp=float(np.mean([r[f'{version}_signed_error_pp'] for r in rows])),
                micro_iou=tp / (tp + fp + fn) if tp + fp + fn else 1.,
                micro_dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                macro_iou=float(np.mean([r[f'{version}_iou'] for r in rows])),
                macro_dice=float(np.mean([r[f'{version}_dice'] for r in rows])))


def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf' if bold else 'C:/Windows/Fonts/malgun.ttf', size)


def montage(output, rows, name, title):
    """Six native-size panels: raw, two GT versions, fixed prediction, two overlaps."""
    width, height = 1662, 1150
    canvas = Image.new('RGB', (width, height), '#f6f8fb')
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), title, fill='#192b3f', font=font(27, True))
    titles = ['원본 타일', '이전 정답', '새 정답', '기존 A 모델 예측', '이전 정답 vs 예측', '새 정답 vs 예측']
    panels_for_check = []
    for i, row in enumerate(rows):
        folder = output / 'tiles' / Path(row['image']).stem
        raw, old, new, pred = (read_gray(folder / f'{key}.png') for key in ('raw', 'old', 'new', 'pred'))
        y0 = 56 + i * 346
        text = (f"{row['point']}  함침율: 이전 {row['old_truth_pct']:.2f}% → 새 정답 {row['new_truth_pct']:.2f}% / 예측 {row['new_pred_pct']:.2f}%"
                f"     절대차 {row['old_absolute_error_pp']:.2f} → {row['new_absolute_error_pp']:.2f}%p"
                f"     IoU {row['old_iou']:.3f} → {row['new_iou']:.3f}")
        draw.text((18, y0), text, fill='#192b3f', font=font(20))
        panels = [raw, old, new, pred, overlap_pixels(old > 0, pred > 0, raw), overlap_pixels(new > 0, pred > 0, raw)]
        for col, (label, pixels) in enumerate(zip(titles, panels)):
            x, y = 18 + col * 274, y0 + 66
            draw.text((x, y0 + 35), label, fill='#192b3f', font=font(18))
            panel = Image.fromarray(pixels).convert('RGB')
            canvas.paste(panel, (x, y))
            panels_for_check.append((x, y, np.array(panel)))
    draw.text((18, 1102), '각 패널은 256×256 전체 타일 · 초록: 일치 / 파랑: 정답만 있음 / 빨강: 예측만 있음 · 모델·임계값·위치 동일', fill='#192b3f', font=font(19))
    canvas.save(output / name)
    saved = np.array(Image.open(output / name))
    assert all(np.array_equal(saved[y:y+256, x:x+256], pixels) for x, y, pixels in panels_for_check)


def reports(output, result, rows):
    scopes = result['scopes']
    lines = ['# 새 Label Studio 정답으로 A 모델 재평가', '',
             '2026-10-09 제공 JSON을 정답으로 사용했습니다. 모델을 재학습하지 않고 저장된 이진 예측을 재사용했습니다.', '',
             '노출(Exposure)은 미함침 영역이며, 브러시 alpha > 0을 포함합니다. 함침율 = 100 × (1 − Exposure 픽셀 수 / 65,536).', '',
             '절대차 = |정답 함침율 − 예측 함침율| (%p). 평균은 각 지점의 절대차를 먼저 계산한 뒤 취합니다. '
             'IoU·Dice는 Exposure 클래스이며 아래 표는 픽셀 TP·FP·FN을 합산한 micro 지표입니다.', '',
             '| 범위 | 타일 수 | 이전 평균 절대차 (%p) | 새 평균 절대차 (%p) | 이전 IoU | 새 IoU | 이전 Dice | 새 Dice |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    names = {'test_all_252': 'Test 전체 (000·180)', 'test_central_000_27': 'Test 중앙 P1·P2·P3 (000)',
             'case05_central_000_9': 'Case 05 중앙 9점 (000)', 'case05_specimen03_central_000_3': 'Case 05 시편 03 중앙 3점 (000)'}
    for key, label in names.items():
        old, new = scopes[key]['old'], scopes[key]['new']
        lines.append(f"| {label} | {new['tiles']} | {old['mean_absolute_error_pp']:.4f} | {new['mean_absolute_error_pp']:.4f} | {old['micro_iou']:.4f} | {new['micro_iou']:.4f} | {old['micro_dice']:.4f} | {new['micro_dice']:.4f} |")
    lines += ['', '## Case 05 시편 03 / Task 641', '',
              '| 지점 | 이전 정답 (%) | 새 정답 (%) | 예측 (%) | 이전 절대차 (%p) | 새 절대차 (%p) | 새 정답 − 예측 (%p) | 새 IoU | 새 Dice |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['task_id'] == 641 and r['point']:
            lines.append(f"| {r['point']} | {r['old_truth_pct']:.2f} | {r['new_truth_pct']:.2f} | {r['new_pred_pct']:.2f} | {r['old_absolute_error_pp']:.2f} | {r['new_absolute_error_pp']:.2f} | {r['new_signed_error_pp']:+.2f} | {r['new_iou']:.4f} | {r['new_dice']:.4f} |")
    audit = result['audit']
    lines += ['', '## 비교 조건과 추적 정보', '',
              f"- 내보내기 작업 {audit['tasks']}개. 기존 유효 타일 영역에서 라벨이 바뀐 촬영본 {audit['changed_tasks']}개, 타일 {audit['changed_tiles']}개.",
              f"- Test에서 라벨이 바뀐 촬영본 {audit['changed_test_tasks']}개, 타일 {audit['changed_test_tiles']}개.",
              '- 복수 Annotation은 취소되지 않은 최신 updated_at을 선택했습니다. 이전 버전과 합치지 않습니다.',
              '- 기존 이미지·학습 데이터·Roboflow 업로드 라벨은 수정하지 않았습니다. 새 정답은 이 평가 폴더 안에 보존합니다.',
              f"- 체크포인트 SHA-256: `{result['checkpoint_sha256']}` / epoch 59 / threshold 0.5.",
              f"- 새 JSON SHA-256: `{result['export_sha256']}`.",
              '- 모든 Test 이미지/이전 마스크의 SHA-256, 저장 예측의 픽셀 수와 TP·FP·FN, 중앙 27개 예측의 이전 P1/P2/P3 결과 일치를 확인했습니다.',
              '- `annotation_audit.csv`: 작업별 선택된 Annotation과 변경 픽셀. `test_tiles.csv`: 252개 전후 비교.',
              '- `central_27.csv`, `case_summary.csv`, `specimen_summary.csv`: 중앙 지점·Case·시편별 비교.',
              '- `input/`: 제공 JSON 사본. `revised_alpha/`: 86개 촬영본의 새 브러시 마스크.',
              '- `tiles/`: Test 252개의 원본·이전 정답·새 정답·고정 예측 PNG.', '',
              '이 결과는 제공한 수정 라벨을 정답으로 삼았을 때의 평가입니다. Test 정답만 바꿨으므로 지표 변화가 모델 자체의 개선을 뜻하지는 않습니다.', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    body = ['<h1>새 정답 라벨로 기존 A 모델 재평가</h1><p>모델·임계값·타일 위치 고정 · Test 전체 252개 / 중앙 27개 · 차이 단위 %p</p>',
            '<p><a href="README.md">상세 보고서</a> · <a href="test_tiles.csv">252개 결과 CSV</a> · <a href="central_27.csv">중앙 27개 CSV</a></p>',
            '<table><tr><th>범위</th><th>이전 평균 절대차</th><th>새 평균 절대차</th><th>이전 IoU</th><th>새 IoU</th></tr>']
    for key, label in names.items():
        old, new = scopes[key]['old'], scopes[key]['new']
        body.append(f"<tr><td>{label}</td><td>{old['mean_absolute_error_pp']:.3f}</td><td>{new['mean_absolute_error_pp']:.3f}</td><td>{old['micro_iou']:.4f}</td><td>{new['micro_iou']:.4f}</td></tr>")
    body.append('</table>')
    for capture in ('03', '02', '01'):
        filename = f'case05_specimen{capture}_P123.png'
        body.append(f'<h2>Case 05 시편 {capture} · 000 방향</h2><a href="{filename}"><img src="{filename}" alt="시편 {capture} P1 P2 P3 이전 정답 새 정답 예측 비교"></a>')
    (output / 'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>정답 수정 전후 비교</title><style>body{font-family:"Malgun Gothic",sans-serif;margin:28px;background:#f6f8fb;color:#192b3f}img{max-width:100%}table{border-collapse:collapse;background:white}td,th{padding:12px;border:1px solid #ccd7e2}a{color:#17639b}</style>'+''.join(body)+'</html>', encoding='utf-8')


def run(export, output):
    tasks_list, manifest, baseline, split = read(export), read(DATA / 'manifest.json'), read(BASE / 'summary.json'), read(RUN / 'splits.json')
    tasks = {t['id']: t for t in tasks_list}
    if len(tasks) != len(tasks_list) or set(tasks) != {s['task_id'] for s in manifest['images']}:
        raise ValueError('Export tasks must match the original 86 source images exactly.')
    if sha(DATA / 'manifest.json') != baseline['manifest_sha256'] or sha(RUN / 'best.pt') != baseline['checkpoint_sha256']:
        raise ValueError('The frozen dataset or A checkpoint changed.')
    if baseline['checkpoint_epoch'] != 59 or baseline['threshold'] != 0.5 or split['manifest_sha256'] != baseline['manifest_sha256']:
        raise ValueError('Threshold or split provenance differs.')
    frozen = {r['image']: r for r in baseline['all_tiles']}
    if len(frozen) != 252 or set(frozen) != {r['image'] for r in split['splits']['test']}:
        raise ValueError('Expected the original fixed TEST 252 tiles.')
    split_by_task = {r['task_id']: name for name, rows in split['splits'].items() for r in rows}
    central_predictions = {r['image']: r for r in read(CENTRAL / 'predictions.json')}
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('input', 'revised_alpha', 'tiles'):
        (output / folder).mkdir()
    shutil.copy2(export, output / 'input' / export.name)
    save(output / 'RUNNING.json', {'input': str(export), 'input_sha256': sha(export)})
    rows, audit = [], []
    for index, source in enumerate(manifest['images'], 1):
        task_id, task = source['task_id'], tasks[source['task_id']]
        image_url = unquote(task['data']['image'])
        expected_name = Path(source['annotation_source_image']).name
        expected_suffix = '/'.join([source['case'], source['capture'], source['position'], '10000us', expected_name])
        if not image_url.endswith(expected_suffix):
            raise ValueError(f'Task {task_id}: source image identity differs.')
        annotation, excluded = latest_annotation(task)
        alpha = task_brush_alpha({**task, 'annotations': [annotation]})
        if list(alpha.shape[::-1]) != source['image_size_wh']:
            raise ValueError(f'Task {task_id}: mask dimensions differ.')
        Image.fromarray(alpha).save(output / 'revised_alpha' / f'task_{task_id}.png')
        changed, different, added, removed, alpha_changes = 0, 0, 0, 0, 0
        for tile in source['tiles']:
            old_alpha = read_gray(DATA / tile['mask'])
            x, y, x1, y1 = tile['xyxy_exclusive']
            new_alpha = alpha[y:y1, x:x1]
            old, new = old_alpha > 0, new_alpha > 0
            if old.shape != (256, 256) or new.shape != old.shape or int(old.sum()) != tile['label_pixels']:
                raise ValueError('Mask crop or old manifest count differs.')
            count = int((old != new).sum())
            changed += int(count > 0)
            different += count
            added += int((new & ~old).sum())
            removed += int((old & ~new).sum())
            alpha_changes += int((old_alpha != new_alpha).sum())
            if split_by_task[task_id] != 'test':
                continue
            original = frozen[tile['image']]
            if sha(DATA / tile['image']) != original['image_sha256'] or sha(DATA / tile['mask']) != original['mask_sha256']:
                raise ValueError('Frozen image or label bytes changed.')
            if original['xyxy'] != tile['xyxy_exclusive']:
                raise ValueError('Saved prediction crop coordinates differ.')
            raw = read_gray(DATA / tile['image'])
            pred_path = BASE / original['assets']['pred']
            pred = read_gray(pred_path) > 0
            old_metrics, new_metrics = measure(old, pred), measure(new, pred)
            for key in ('tp', 'fp', 'fn', 'iou', 'dice'):
                if abs(old_metrics[key] - original[key]) > 1e-12:
                    raise ValueError(f'Saved baseline metrics differ for {tile["image"]}: {key}')
            if int(pred.sum()) != original['pred_exposure_pixels']:
                raise ValueError('Saved prediction pixel count differs.')
            if tile.get('measurement_point'):
                cp = central_predictions[tile['image']]
                if not np.array_equal(pred, read_gray(CENTRAL / cp['pred_mask']) > 0):
                    raise ValueError('Central-point saved predictions disagree.')
            row = dict(image=tile['image'], task_id=task_id, annotation_id=annotation['id'],
                       case=source['case'], specimen=source['capture'], direction=source['position'],
                       point=tile.get('measurement_point'), xyxy=tile['xyxy_exclusive'],
                       changed_pixels=count, added_exposure_pixels=int((new & ~old).sum()), removed_exposure_pixels=int((old & ~new).sum()),
                       prediction_sha256=sha(pred_path), new_alpha_sha256=hashlib.sha256(new_alpha.tobytes()).hexdigest(),
                       **{f'old_{k}': v for k, v in old_metrics.items()}, **{f'new_{k}': v for k, v in new_metrics.items()})
            rows.append(row)
            folder = output / 'tiles' / Path(tile['image']).stem
            folder.mkdir()
            for key, pixels in [('raw', raw), ('old', old.astype(np.uint8)*255), ('new', new.astype(np.uint8)*255),
                                ('new_alpha', new_alpha), ('pred', pred.astype(np.uint8)*255)]:
                Image.fromarray(pixels).save(folder / f'{key}.png')
        audit.append(dict(task_id=task_id, case=source['case'], specimen=source['capture'], direction=source['position'],
                          split=split_by_task[task_id], old_annotation_ids=source['annotation_ids'], annotation_id=annotation['id'],
                          updated_at=annotation['updated_at'], excluded_older_annotations=excluded,
                          tiles=len(source['tiles']), changed_tiles=changed, changed_pixels=different,
                          added_exposure_pixels=added, removed_exposure_pixels=removed, changed_alpha_pixels=alpha_changes))
        if index % 10 == 0 or index == len(tasks):
            print(f'Decoded and audited {index}/{len(tasks)} sources; TEST compared {len(rows)}/252', flush=True)
    rows.sort(key=lambda r: r['image'])
    central = [r for r in rows if r['direction'] == '000' and r['point']]
    if len(rows) != 252 or len(central) != 27:
        raise ValueError('Incomplete TEST comparison.')
    groups = {'test_all_252': rows, 'test_central_000_27': central,
              'case05_central_000_9': [r for r in central if r['case'] == 'case05'],
              'case05_specimen03_central_000_3': [r for r in central if r['task_id'] == 641]}
    result = dict(status='completed', checkpoint_sha256=baseline['checkpoint_sha256'], checkpoint_epoch=59,
                  threshold=0.5, retrained=False, predictions_reused=252, export_sha256=sha(export),
                  annotation_policy='Latest non-cancelled annotation by updated_at; never union different revisions.',
                  scopes={key: {version: aggregate(values, version) for version in ('old', 'new')} for key, values in groups.items()},
                  audit=dict(tasks=len(audit), changed_tasks=sum(a['changed_tiles'] > 0 for a in audit),
                             changed_tiles=sum(a['changed_tiles'] for a in audit),
                             changed_test_tasks=sum(a['changed_tiles'] > 0 for a in audit if a['split'] == 'test'),
                             changed_test_tiles=sum(a['changed_tiles'] for a in audit if a['split'] == 'test')))
    case_rows, specimen_rows = [], []
    for case in sorted({r['case'] for r in central}):
        case_values = [r for r in central if r['case'] == case]
        case_rows.append(dict(case=case, **{f'{version}_{k}': v for version in ('old', 'new') for k, v in aggregate(case_values, version).items()}))
        for specimen in sorted({r['specimen'] for r in case_values}):
            values = [r for r in case_values if r['specimen'] == specimen]
            specimen_rows.append(dict(case=case, specimen=specimen, task_id=values[0]['task_id'],
                                      **{f'{version}_{k}': v for version in ('old', 'new') for k, v in aggregate(values, version).items()}))
    for name, values in [('annotation_audit', audit), ('test_tiles', rows), ('central_27', central), ('case_summary', case_rows), ('specimen_summary', specimen_rows)]:
        csv_file(output / f'{name}.csv', values)
        save(output / f'{name}.json', values)
    for specimen in ('01', '02', '03'):
        values = sorted((r for r in central if r['case'] == 'case05' and r['specimen'] == specimen), key=lambda r: r['point'])
        montage(output, values, f'case05_specimen{specimen}_P123.png', f'Case 05 · 시편 {specimen} · 000 방향 · 정답 수정 전후 비교')
    reports(output, result, rows)
    save(output / 'summary.json', result)
    save(output / 'verification.json', dict(status='passed', test_tiles=252, central_000_points=27,
        image_and_old_mask_hashes_verified=True, saved_prediction_metrics_verified=True,
        central_prediction_masks_equal=True, task_paths_and_dimensions_verified=True,
        copied_export_hash_equal=sha(export) == sha(output / 'input' / export.name),
        native_256px_figure_panels_verified=54))
    (output / 'RUNNING.json').unlink()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/label_revision_20261009_A')
    args = parser.parse_args()
    run(args.export.resolve(), args.output.resolve())
