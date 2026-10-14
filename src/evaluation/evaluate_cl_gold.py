"""Evaluate a completed full-tile A/B/D experiment on one frozen Gold72 snapshot."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gc
import html
from pathlib import Path

import numpy as np
from PIL import Image

from src.Namrc.oof import validate_oof
from src.Namrc.protocol import load_frozen_splits, read_json, sha256, verified, write_csv, write_json
from src.Namrc.resources import MemoryMonitor
from src.inference.compare_gold_p3_masks import (
    KST, file_info, freeze_inputs, read_gray, require, scores, verify_files,
)
from src.segmentation.label_studio import task_brush_alpha

ROOT = Path(__file__).resolve().parents[2]
ARMS = {'A': 'baseline', 'B': 'cl_filter', 'D': 'cl_auto'}
LABELS = {'A': '원본 라벨', 'B': 'CL 의심 픽셀 제외', 'D': 'CL 자동 수정 라벨'}


def check_gold(row):
    files = row['gold_files']
    gray = read_gray(files['image.png']['path'])
    gt = np.load(files['mask.npy']['path'], allow_pickle=False)
    alpha = np.load(files['mask_alpha.npy']['path'], allow_pickle=False)
    require(gray.shape == gt.shape == alpha.shape == (256, 256), 'Gold native size differs.')
    require(gt.dtype == alpha.dtype == np.uint8 and set(np.unique(gt)) <= {0, 1}, 'Gold encoding differs.')
    require(np.array_equal(gt, alpha > 0), 'Gold alpha and binary differ.')
    require(np.array_equal(read_gray(files['mask.png']['path']), gt * 255), 'Gold PNG and NPY differ.')
    require(np.array_equal(read_gray(files['mask_alpha.png']['path']), alpha), 'Gold alpha PNG differs.')
    annotation = read_json(files['annotation.json']['path'])
    if annotation['annotations'][0]['result']:
        require(np.array_equal(task_brush_alpha(annotation), alpha), 'Gold annotation RLE differs.')
    else:
        require(not alpha.any(), 'Empty Gold annotation has positive pixels.')
    require(int(gt.sum()) == row['expected_gold_exposure_pixels'], 'Gold count differs.')
    x0, y0, x1, y1 = row['p3_xyxy_exclusive']
    raw = read_gray(row['original_files']['raw_image']['path'])
    require(np.array_equal(gray, raw[y0:y1, x0:x1]), 'Gold is not the recorded native P3 crop.')
    return gray, gt.astype(bool)


def prepare(args, monitor):
    experiment = args.experiment
    config = read_json(experiment / 'experiment.json')
    require(not config['settings'].get('central_only', False), 'Central-only model excluded.')
    require(not config['settings'].get('smoke_test', False), 'Smoke model excluded.')
    model_data = Path(config['settings']['data'])
    splits, digest = load_frozen_splits(model_data, experiment / 'splits.json')
    require(digest == config['manifest_sha256'], 'Experiment data differs.')
    require(any(r.get('measurement_point') not in ('P1', 'P2', 'P3') for r in splits['train']), 'Only P1/P2/P3 trained.')
    mapping_splits, _ = load_frozen_splits(args.data, args.data / 'splits.json')
    require({k: len(v) for k, v in mapping_splits.items()} == {'train': 1056, 'val': 187, 'test': 251}, 'changed_v3 split counts differ.')
    print('Validating Train-only Case-held-out OOF and immutable training provenance...', flush=True)
    _, oof = validate_oof(experiment, splits['train'], model_data)
    monitor.record('oof_validated', force=True)
    models = {}
    model_files = [file_info(experiment / n) for n in ('experiment.json', 'splits.json', 'folds.json', 'oof/provenance.json', 'issues/summary.json', 'auto/manifest.json')]
    common = None
    for arm, name in ARMS.items():
        directory = experiment / name
        require(verified(directory, ('best.pt', 'result.json', 'splits.json', 'config.json')), f'{arm} fit is incomplete or changed.')
        fit = read_json(directory / 'config.json')
        result = read_json(directory / 'result.json')
        require(read_json(directory / 'splits.json')['splits'] == splits, f'{arm} train/selection split differs.')
        require(fit['confident']['arm'] == arm and result['status'] == 'completed', 'Arm provenance differs.')
        require(fit['confident']['experiment_fingerprint'] == config['fingerprint'], 'Training fingerprint differs.')
        require(fit['threshold'] == .5 and fit['manifest_sha256'] == digest and fit['workers'] == 0, 'Fit policy differs.')
        if arm in ('B', 'D'):
            paired = fit['confident']
            pair = (paired['issues_sha256'], paired['shared_oof_sha256'])
            require(common is None or pair == common, 'B/D use different CL issues/probabilities.')
            common = pair
        checkpoint = file_info(directory / 'best.pt')
        models[arm] = {'name': name, 'treatment': LABELS[arm], 'checkpoint': checkpoint,
                       'config': file_info(directory / 'config.json'), 'result': file_info(directory / 'result.json'),
                       'splits': file_info(directory / 'splits.json'), 'best_epoch': result['best_epoch'],
                       'best_validation_dice': result['best_val_dice'], 'confident': fit['confident'],
                       'training_data': str(model_data), 'training_manifest_sha256': digest,
                       'split_counts': {k: len(v) for k, v in splits.items()},
                       'split_cases': {k: sorted({r['group'] for r in v}) for k, v in splits.items()}}
        model_files.extend(models[arm][k] for k in ('checkpoint', 'config', 'result', 'splits'))
    auto = read_json(experiment / 'auto/manifest.json')
    surface = read_json(experiment / 'oof/surface.json')
    issues_path = experiment / 'issues/ignore.npy'
    require(sha256(issues_path) == common[0] == auto['hashes']['final_issue_mask_sha256'], 'Shared CL mask hash differs.')
    probability_path = experiment / 'oof' / surface['probability_file']
    require(sha256(probability_path) == common[1] == auto['hashes']['oof_probabilities_sha256'], 'Shared OOF probabilities differ.')
    require(models['D']['confident']['auto_manifest_sha256'] == sha256(experiment / 'auto/manifest.json'), 'D label manifest differs.')
    frozen = freeze_inputs(args.data, args.gold, args.output)
    source_members = {k: {r['task_id'] for r in rows} for k, rows in splits.items()}
    case_members = {k: {r['group'] for r in rows} for k, rows in splits.items()}
    for row in frozen['records']:
        check_gold(row)
        task = row['source_task_id']
        case = f"case{row['case_number']:02d}"
        actual = next((k for k, members in source_members.items() if task in members), None)
        case_actual = next((k for k, members in case_members.items() if case in members), None)
        row.update(model_source_split=actual or 'unseen_capture_train_case', model_case_split=case_actual,
                   capture=row['source_filename'].split('_capture', 1)[1].split('_task', 1)[0],
                   checkpoint_sha256={arm: info['checkpoint']['sha256'] for arm, info in models.items()},
                   model_provenance={arm: {'training_data': info['training_data'], 'best_epoch': info['best_epoch'],
                                          'training_manifest_sha256': digest, 'treatment': info['treatment']}
                                     for arm, info in models.items()},
                   training_image_overlap=actual == 'train', validation_image_overlap=actual == 'val',
                   training_case_overlap=case_actual == 'train', validation_case_overlap=case_actual == 'val',
                   train_oof_processed=task in source_members['train'],
                   independent_case_test=actual == 'test' and case_actual == 'test')
        require(actual is None or actual == row['source_split'], 'Model/source split mapping changed.')
        for rows in splits.values():
            matches = [r for r in rows if r['task_id'] == task and r.get('measurement_point') == 'P3']
            if matches:
                require(len(matches) == 1 and matches[0]['image_sha256'] == row['gold_files']['image.png']['sha256'], 'Training P3 image identity differs.')
                require(matches[0]['xyxy'] == row['p3_xyxy_exclusive'], 'Training P3 coordinates differ.')
    require(Counter(r['source_split'] for r in frozen['records']) == {'train': 51, 'val': 9, 'test': 12}, 'Gold source split counts differ.')
    frozen.update(scope='Full-tile CL A/B/D checkpoints evaluated against saved Gold P3 72; no training.',
                  models=models, model_files=model_files, oof_provenance=oof,
                  dataset_splits=file_info(args.data / 'splits.json'),
                  evaluation_policy={'threshold': .5, 'augmentation': 'none', 'normalization': 'uint8 / 255',
                                     'native_size': [256, 256], 'valid_pixels': 65536, 'batch_size': 1,
                                     'workers': 0, 'persistent_workers': False, 'pin_memory': False,
                                     'amp': args.device == 'cuda', 'ram_budget_mib': 4096, 'vram_budget_mib': 4096},
                  source_split_counts=dict(Counter(r['source_split'] for r in frozen['records'])),
                  actual_model_membership_counts=dict(Counter(r['model_source_split'] for r in frozen['records'])),
                  cl_issue_summary=read_json(experiment / 'issues/summary.json'),
                  shared_cl_inputs={'issues_path': str(issues_path), 'issues_sha256': common[0],
                                    'oof_probabilities_path': str(probability_path), 'oof_probabilities_sha256': common[1]},
                  source_code=file_info(Path(__file__)))
    write_json(args.output / 'gold_manifest.json', frozen)
    print(f"Gold72 frozen; actual model membership: {frozen['actual_model_membership_counts']}", flush=True)
    return frozen


def verify_snapshot(frozen):
    verify_files(frozen)
    for info in [*frozen['model_files'], frozen['dataset_splits'], frozen['source_code']]:
        require(sha256(info['path']) == info['sha256'], f"Frozen input changed: {info['path']}")


def aggregate(rows):
    n = len(rows)
    result = {'images': n, 'total_pixels': n * 65536,
              **scores(*(sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn', 'tn')))}
    for name in ('dice', 'iou', 'precision', 'recall'):
        valid = [r[name] for r in rows if r[name] is not None]
        result[name + '_image_mean'] = sum(valid) / len(valid) if valid else None
        result[name + '_defined_images'] = len(valid)
    for name in ('gt_exposure_pixels', 'pred_exposure_pixels'):
        result[name] = sum(r[name] for r in rows)
    for name in ('gt_impregnation_pct', 'pred_impregnation_pct', 'abs_error_pp', 'signed_error_pp'):
        result[name + '_image_mean'] = sum(r[name] for r in rows) / n
    result['mae_pp'] = result['abs_error_pp_image_mean']
    return result


def overlay(gray, mask, color):
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    rgb[mask] = .55 * rgb[mask] + .45 * np.array(color)
    return rgb.astype(np.uint8)


def evaluate_arm(args, frozen, arm, monitor):
    import torch
    from src.inference.predict import load_checkpoint
    from src.training.memory import CudaMemoryGuard

    device = torch.device(args.device)
    destination = args.output / arm
    destination.mkdir()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    guard = CudaMemoryGuard(device, destination, budget_mib=4096)
    provenance = frozen['models'][arm]
    model, metadata = load_checkpoint(Path(provenance['checkpoint']['path']), device)
    require(metadata['sha256'] == provenance['checkpoint']['sha256'] and metadata['epoch'] == provenance['best_epoch']
            and metadata['threshold'] == .5 and not metadata['smoke_test'], 'Checkpoint metadata differs.')
    monitor.record(arm + '_loaded', force=True)
    results = []
    with torch.inference_mode():
        for index, snapshot in enumerate(frozen['records'], 1):
            gray, gt = check_gold(snapshot)
            guard.check(arm + '_before')
            tensor = torch.from_numpy(gray.astype(np.float32) / 255)[None, None].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(tensor)
            probability = logits.float().sigmoid().cpu().numpy()[0, 0]
            require(probability.shape == (256, 256) and np.isfinite(probability).all(), 'Invalid prediction.')
            pred = probability >= .5
            counts = (int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum()), int((~pred & ~gt).sum()))
            gt_count, pred_count = int(gt.sum()), int(pred.sum())
            gt_pct, pred_pct = 100 * (1 - gt_count / 65536), 100 * (1 - pred_count / 65536)
            task = snapshot['gold_task_id']
            tile = destination / f'task_{task}'
            tile.mkdir()
            np.save(tile / 'probability.npy', probability, allow_pickle=False)
            np.save(tile / 'prediction.npy', pred.astype(np.uint8), allow_pickle=False)
            Image.fromarray(gt.astype(np.uint8) * 255).save(tile / 'gt.png')
            Image.fromarray(pred.astype(np.uint8) * 255).save(tile / 'prediction.png')
            Image.fromarray(overlay(gray, gt, (40, 204, 142))).save(tile / 'gt_overlay.png')
            Image.fromarray(overlay(gray, pred, (58, 152, 255))).save(tile / 'prediction_overlay.png')
            error = np.repeat(gray[..., None], 3, axis=2)
            error[pred & gt] = (40, 204, 142)
            error[pred & ~gt] = (245, 158, 11)
            error[~pred & gt] = (239, 70, 108)
            Image.fromarray(error).save(tile / 'error_overlay.png')
            row = {k: snapshot[k] for k in ('case_number', 'side', 'specimen_number', 'target', 'shot_number', 'capture',
                   'source_project_id', 'source_task_id', 'gold_task_id', 'gold_annotation_id', 'source_split',
                   'model_source_split', 'model_case_split', 'training_image_overlap', 'validation_image_overlap',
                   'training_case_overlap', 'validation_case_overlap', 'train_oof_processed', 'independent_case_test')}
            row.update(arm=arm, checkpoint_sha256=metadata['sha256'], image_sha256=snapshot['gold_files']['image.png']['sha256'],
                       gt_sha256=snapshot['gold_files']['mask.npy']['sha256'], threshold=.5, total_pixels=65536,
                       **scores(*counts), gt_exposure_pixels=gt_count, pred_exposure_pixels=pred_count,
                       gt_impregnation_pct=gt_pct, pred_impregnation_pct=pred_pct,
                       signed_error_pp=pred_pct - gt_pct, abs_error_pp=abs(pred_pct - gt_pct),
                       prediction_path=str(tile / 'prediction.png'), gt_path=str(tile / 'gt.png'),
                       overlay_path=str(tile / 'error_overlay.png'))
            results.append(row)
            del logits, tensor
            guard.check(arm + '_after')
            monitor.record(arm + '_predict', tiles=index, pixels=index * 65536)
            if index % 24 == 0:
                print(f'{arm}: {index}/72 evaluated', flush=True)
    write_csv(destination / 'per_image.csv', results)
    write_json(destination / 'per_image.json', results)
    write_json(destination / 'resources.json', {'host_peak_rss_mib': monitor.maximum, 'gpu_peak_reserved_mib': guard.peak_reserved_mib})
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    monitor.record(arm + '_released', force=True)
    return results


def build_reports(args, frozen, rows):
    groups = []
    for arm in ARMS:
        chosen = [r for r in rows if r['arm'] == arm]
        buckets = {'all72': chosen}
        buckets.update({f'source_{s}': [r for r in chosen if r['source_split'] == s] for s in ('train', 'val', 'test')})
        buckets.update({f'actual_{s}': [r for r in chosen if r['model_source_split'] == s]
                        for s in sorted({r['model_source_split'] for r in chosen})})
        for subset, selected in buckets.items():
            groups.append({'arm': arm, 'subset': subset, **aggregate(selected)})
    case_groups = []
    for fields, scope in [(('arm', 'case_number'), 'case'), (('arm', 'case_number', 'side'), 'case_side'),
                          (('arm', 'case_number', 'side', 'target', 'shot_number'), 'specimen_or_shot')]:
        buckets = defaultdict(list)
        for row in rows:
            buckets[tuple(row[k] for k in fields)].append(row)
        for key, selected in buckets.items():
            case_groups.append({'scope': scope, **dict(zip(fields, key)), **aggregate(selected)})
    write_csv(args.output / 'per_image.csv', rows)
    write_json(args.output / 'per_image.json', rows)
    write_csv(args.output / 'by_subset.csv', groups)
    write_json(args.output / 'by_case_side_specimen.json', case_groups)
    write_csv(args.output / 'by_case_side_specimen.csv', [{k: g.get(k) for k in
        ('scope', 'arm', 'case_number', 'side', 'target', 'shot_number', *aggregate(rows[:1]).keys())} for g in case_groups])
    changes = []
    for subset in ('all72', 'source_train', 'source_val', 'source_test'):
        a = next(g for g in groups if g['arm'] == 'A' and g['subset'] == subset)
        for arm in ('B', 'D'):
            b = next(g for g in groups if g['arm'] == arm and g['subset'] == subset)
            changes.append({'arm': arm, 'subset': subset, 'dice_change': b['dice'] - a['dice'],
                            'iou_change': b['iou'] - a['iou'], 'mae_improvement_pp': a['mae_pp'] - b['mae_pp']})
    summary = {'created_at': datetime.now(KST).isoformat(), 'groups': groups, 'changes_vs_A': changes,
               'models': frozen['models'], 'gold_manifest_sha256': sha256(args.output / 'gold_manifest.json'),
               'source_membership': frozen['source_split_counts'], 'actual_membership': frozen['actual_model_membership_counts'],
               'metric_policy': 'Pixel pooled Dice/IoU/Precision/Recall; image means reported separately; MAE in percentage points.',
               'empty_mask_policy': 'Dice/IoU=1 if both empty; undefined precision/recall=null and excluded from corresponding image mean.',
               'interpretation': 'Gold72 includes training/selection overlap; source_test12 is Case-held-out for these checkpoints.',
               'unseen_capture_note': 'Task800 is absent from changed_v2 training; Case16 is a training/OOF Case. It is not Case-held-out.',
               'selection_policy': 'Saved validation-selected checkpoints; fixed threshold 0.5; Gold used only for evaluation.',
               'cl_issues': frozen['cl_issue_summary'], 'training_executed': False}
    write_json(args.output / 'summary.json', summary)
    headings = {'all72': 'Gold 전체 72장', 'source_train': '원래 Train 소속 51장',
                'source_val': '원래 Validation 소속 9장', 'source_test': 'Case held-out Test 12장'}
    text = ['# 전체 타일 CL 모델 · Gold P3 72장 평가', '',
            'changed_v2 Train 1,035타일로 학습한 A/B/D를 동일한 최신 저장 Gold로 평가했다. 체크포인트는 기존 Validation Dice로 선택했으며 이진화 임계값은 0.5다.', '',
            'Gold72는 원래 Train 51 / Validation 9 / Test 12로 구성된다. 실제 v2 모델은 Train 촬영본 50장, Validation 9장을 사용했다. 추가 Task800은 학습에 없지만 Case16은 학습과 CL OOF에 포함되므로 독립 Test가 아니다.', '',
            '| 집합 | 모델 | Dice | IoU | Precision | Recall | 함침율 MAE (pp) |',
            '|---|---|---:|---:|---:|---:|---:|']
    for subset, label in headings.items():
        for arm in ARMS:
            g = next(g for g in groups if g['arm'] == arm and g['subset'] == subset)
            text.append(f"| {label} | {arm} {LABELS[arm]} | {g['dice']:.5f} | {g['iou']:.5f} | {g['precision']:.5f} | {g['recall']:.5f} | {g['mae_pp']:.4f} |")
    text += ['', 'Dice/IoU/Precision/Recall 표는 전체 픽셀 집계다. 이미지별 평균은 by_subset.csv에 따로 있다. GT/예측/오버레이와 TP/FP/FN, Exposure 픽셀 수, 이미지별 함침율, Task/Annotation 및 체크포인트 SHA256은 per_image.csv/json과 A/B/D 폴더에 저장했다.', '',
             'Gold 전체 점수는 독립적인 미학습 Test 성능을 뜻하지 않는다. D 자동 수정 라벨은 사람 검수 정답이 아니다. 새 학습이나 Gold 기반 모델·임계값 선택은 수행하지 않았다.']
    (args.output / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    table = ''.join('<tr>' + ''.join(f'<td>{html.escape(x.strip())}</td>' for x in line.strip('|').split('|')) + '</tr>'
                    for line in text if line.startswith('|') and not line.startswith('|---'))
    cards = []
    for source in frozen['records']:
        task = source['gold_task_id']
        title = f"Case{source['case_number']:02d} {source['side']} {source['target']} · 샷{source['shot_number']} · Gold {task} / 원본 {source['source_task_id']}"
        parts = [f'<article data-split="{source["source_split"]}"><h2>{html.escape(title)}</h2>',
                 f'<p>원래 분할: {source["source_split"]} · 실제 모델 소속: {source["model_source_split"]}</p><div class="images">',
                 f'<figure><figcaption>Gold GT</figcaption><img loading="lazy" src="A/task_{task}/gt_overlay.png"></figure>']
        for arm in ARMS:
            row = next(r for r in rows if r['arm'] == arm and r['gold_task_id'] == task)
            parts.append(f'<figure><figcaption>{arm} {LABELS[arm]}<br>함침율 {row["pred_impregnation_pct"]:.3f}% · 오차 {row["signed_error_pp"]:+.3f} pp<br>Dice {row["dice"]:.4f}</figcaption><img loading="lazy" class="pred" src="{arm}/task_{task}/prediction_overlay.png" data-pred="{arm}/task_{task}/prediction_overlay.png" data-error="{arm}/task_{task}/error_overlay.png" data-mask="{arm}/task_{task}/prediction.png"></figure>')
        parts.append(f'</div><p>GT 함침율 {row["gt_impregnation_pct"]:.3f}% · GT Exposure {row["gt_exposure_pixels"]:,} px · 평가 65,536 px</p></article>')
        cards.append(''.join(parts))
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CL A/B/D Gold72 평가</title><style>
body{font:15px 'Malgun Gothic',sans-serif;line-height:1.65;color:#193149;background:#f2f5f8;margin:25px auto;max-width:1300px;padding:0 18px}h1{font-size:28px}h2{font-size:18px}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}td{padding:8px;border-bottom:1px solid #dde3e9}tr:first-child{font-weight:bold;background:#e8f0f8}article{background:white;margin:22px 0;padding:18px;border:1px solid #dce4eb;border-radius:10px}.images{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{width:256px;height:256px;max-width:100%;object-fit:contain;image-rendering:pixelated}figcaption{min-height:80px}button,select{padding:9px;margin-right:8px}.note{background:#fff0d6;padding:15px;border-radius:8px}@media(max-width:850px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><h1>전체 타일 CL 모델 · Gold P3 72장 평가</h1>'''
    page += '<p>changed_v2 Train 1,035타일 · A 원본 / B 의심 픽셀 제외 / D 자동 수정 · 고정 임계값 0.5 · native 256×256</p>'
    page += '<p class="note">Gold72: 원래 Train 51 / Validation 9 / Test 12. 실제 학습 촬영본은 50장이며 Task800은 학습에 없는 같은 Train Case의 새 촬영본입니다. 전체 점수에는 학습·모델 선택 중복이 포함됩니다. Case held-out 결과는 Test12에서 확인하세요.</p>'
    page += '<p><a href="per_image.csv">이미지별 CSV</a> · <a href="by_subset.csv">집합별 CSV</a> · <a href="by_case_side_specimen.csv">Case·측·시편/샷 CSV</a> · <a href="summary.json">결과 JSON</a> · <a href="gold_manifest.json">고정 Gold 및 모델 출처</a></p><div style="overflow:auto"><table>' + table + '</table></div>'
    page += '<p>표: 전체 픽셀 집계 / 함침율 MAE: 이미지별 절대오차 평균(pp). 오차 = 예측 − GT. 오류 지도: 녹색 TP / 주황 FP / 분홍 FN.</p><p><select id="split"><option value="all">전체 72장</option><option value="train">Train 51장</option><option value="val">Validation 9장</option><option value="test">Test 12장</option></select><button data-mode="pred">예측 오버레이</button><button data-mode="error">TP / FP / FN</button><button data-mode="mask">예측 마스크</button></p>'
    page += ''.join(cards) + '''<script>document.querySelector('#split').onchange=e=>document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.split!==e.target.value);document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>document.querySelectorAll('img.pred').forEach(i=>i.src=i.dataset[b.dataset.mode]));</script></html>'''
    (args.output / 'index.html').write_text(page, encoding='utf-8')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data/changed_v3')
    p.add_argument('--gold', type=Path, default=ROOT / 'data/gold')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args(argv)
    for name in ('experiment', 'data', 'gold', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    require(not (args.output / 'gold_manifest.json').exists(), 'Use a new output; existing frozen evaluation is immutable.')
    args.output.mkdir(parents=True, exist_ok=True)
    monitor = MemoryMonitor(args.output, 4096, 1024)
    monitor.record('prepare', force=True)
    try:
        frozen = prepare(args, monitor)
        verify_snapshot(frozen)
        manifest_sha = sha256(args.output / 'gold_manifest.json')
        rows = []
        for arm in ARMS:
            verify_snapshot(frozen)
            rows.extend(evaluate_arm(args, frozen, arm, monitor))
        verify_snapshot(frozen)
        require(sha256(args.output / 'gold_manifest.json') == manifest_sha, 'Gold manifest changed during evaluation.')
        summary = build_reports(args, frozen, rows)
        monitor.record('complete', force=True)
        write_json(args.output / 'complete.json', {'gold_manifest_sha256': manifest_sha, 'summary_sha256': sha256(args.output / 'summary.json'),
                                                 'images_per_arm': 72, 'arms': list(ARMS), 'host_peak_rss_mib': monitor.maximum})
        for g in summary['groups']:
            if g['subset'] in ('all72', 'source_test'):
                print(f"{g['arm']} {g['subset']}: Dice {g['dice']:.6f}, IoU {g['iou']:.6f}, MAE {g['mae_pp']:.4f} pp", flush=True)
    except BaseException as error:
        write_json(args.output / 'error.json', {'exception': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
