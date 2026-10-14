"""Audit P1/P2/P3 with one existing checkpoint; inference and disk CL only."""

import argparse
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np

from src.Namrc.disk import DiskArray, create_npy, write_slice
from src.Namrc.geometry import broadcast_shared_issues, prepare_surface
from src.Namrc.labels import OriginalLabels
from src.Namrc.protocol import (load_frozen_splits, read_json, seal, select_central_splits,
                                    sha256, verified, write_csv, write_json)
from src.Namrc.reports import page, save_panels, table
from src.Namrc.resources import MemoryMonitor, capacity_plan
from src.Namrc.run import predict_to_disk
from src.Namrc.scoring import find_label_issues_disk
from src.training.data import read_gray

PROJECT = Path(__file__).resolve().parents[2]


def aggregate(rows):
    """Pixel-weighted CL cleanliness; invalid and repeated surface pixels excluded."""
    valid = sum(r['unique_valid_pixels'] for r in rows)
    issues = sum(r['unique_issue_pixels'] for r in rows)
    normal = sum(r['unique_pred_normal_pixels'] for r in rows)
    if not 0 <= issues <= valid or not 0 <= normal <= valid:
        raise ValueError('Invalid pixel accounting.')
    return {'tiles': len(rows), 'captures': len({r['task_id'] for r in rows}),
            'valid_pixels': valid, 'issue_pixels': issues, 'clean_pixels': valid-issues,
            'cl_clean_pct': 100*(valid-issues)/valid if valid else None,
            'cl_issue_pct': 100*issues/valid if valid else None,
            'pred_impregnation_pct': 100*normal/valid if valid else None,
            'tiles_without_issues': sum(r['issue_pixels'] == 0 for r in rows)}


def summarize(rows):
    by_point = [{'point': point, **aggregate([r for r in rows if r['point'] == point])}
                for point in ('P1', 'P2', 'P3')]
    by_split = [{'split': split, 'point': point,
                 **aggregate([r for r in rows if r['split'] == split and r['point'] == point])}
                for split in ('train', 'val', 'test') for point in ('P1', 'P2', 'P3')]
    return by_point, by_split, aggregate(rows)


def run(args):
    args.data, args.checkpoint = args.data.resolve(), args.checkpoint.resolve()
    if min(args.batch_size, args.pixel_chunk_size, args.sort_chunk_size) < 1:
        raise ValueError('Batch and chunk sizes must be positive.')
    if args.memory_budget_mib <= 0 or args.min_free_ram_mib < 0:
        raise ValueError('Invalid memory budget.')
    run_dir = args.checkpoint.parent
    required = ('best.pt', 'config.json', 'result.json', 'splits.json')
    if args.checkpoint.name != 'best.pt' or not all((run_dir/name).is_file() for name in required):
        raise ValueError('Use a completed training run best.pt with config/result/splits.json.')
    fit_config, fit_result = read_json(run_dir/'config.json'), read_json(run_dir/'result.json')
    if fit_config.get('smoke_test') or fit_result.get('status') != 'completed':
        raise ValueError('Checkpoint training must be completed and non-smoke.')
    if (run_dir/'complete.json').exists() and not verified(run_dir, required):
        raise ValueError('Training artifact integrity check failed.')
    fit_splits = read_json(run_dir/'splits.json')['splits']
    fit_images = {r['image']: r for r in fit_splits['train']}
    fit_cases = {r['group'] for r in fit_splits['train']}
    splits, digest = load_frozen_splits(args.data, args.data/'splits.json')
    splits = select_central_splits(splits, True)
    rows = [r for name in ('train', 'val', 'test') for r in splits[name]]
    partition = {r['image']: name for name, subset in splits.items() for r in subset}
    hashes = {str(run_dir/name): sha256(run_dir/name) for name in required}
    hashes.update({str(args.data/'manifest.json'): digest,
                   str(args.data/'splits.json'): sha256(args.data/'splits.json')})
    output = (args.output or PROJECT/'analysis/confident'/
              f"{args.data.name}_A_p123_clean_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    if (output.is_relative_to(args.data) or args.data.is_relative_to(output)
            or output.is_relative_to(run_dir) or run_dir.is_relative_to(output)):
        raise ValueError('Use a new output directory outside the dataset and training run.')
    args.smoke_test, args.workers = False, 0
    plan = capacity_plan(len(rows)*65536, args.pixel_chunk_size, args.sort_chunk_size)
    config = {'protocol': 'single_checkpoint_all_p123_cl_audit_v1', 'training_executed': False,
              'checkpoint': str(args.checkpoint), 'checkpoint_best_epoch': fit_result['best_epoch'],
              'data': str(args.data), 'scope': 'All frozen Train/Val/Test P1/P2/P3, jointly ranked.',
              'is_oof': False, 'source_hashes': hashes,
              'clean_definition': '100 * (unique valid pixels - final CL issue pixels) / unique valid pixels',
              'interpretation': 'Model-dependent label candidates, not verified ground-truth correctness.',
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'},
              'split_tiles': {k: len(v) for k, v in splits.items()}, 'capacity_plan': plan}
    if output.exists():
        if (not args.resume or not (output/'RUNNING.json').is_file()
                or read_json(output/'experiment.json') != config or read_json(output/'tiles.json') != rows):
            raise ValueError('Resume requires the same incomplete audit, inputs and settings.')
    elif args.resume:
        raise ValueError('Cannot resume a nonexistent audit.')
    else:
        output.mkdir(parents=True)
    monitor = MemoryMonitor(output, args.memory_budget_mib, args.min_free_ram_mib)
    monitor.record('prepare', force=True)
    if plan['estimated_scratch_mib'] > args.memory_budget_mib/2:
        raise MemoryError('Reduce pixel chunks for this memory budget.')
    monitor.check_disk(sum(plan[k] for k in ('estimated_audit_disk_bytes', 'estimated_oof_disk_bytes',
                                            'temporary_extra_disk_bytes')))
    write_json(output/'RUNNING.json', config)
    write_json(output/'experiment.json', config)
    write_json(output/'tiles.json', rows)
    print(f'Existing checkpoint inference only: {len(rows)} P1/P2/P3 tiles\nOutput: {output}', flush=True)
    probability = predict_to_disk(args, output, 'A', args.checkpoint, rows)
    inputs = output/'inputs'
    inputs.mkdir(exist_ok=True)
    create_npy(inputs/'probabilities.npy', (len(rows), 2, 256, 256), np.float32)
    create_npy(inputs/'labels.npy', (len(rows), 256, 256), np.uint8)
    original = OriginalLabels(args.data)
    for index, row in enumerate(rows):
        p = probability[index]
        write_slice(inputs/'probabilities.npy', index, np.stack([1-p, p]))
        write_slice(inputs/'labels.npy', index, original.load(row)[1])
        monitor.record('labels_to_disk', tiles=index+1, chunk=65536)
    surface = prepare_surface(args.data, rows, inputs, monitor=monitor)
    surface['policy'] = 'First valid selected tile owns a surface pixel; shared A-model probabilities and candidates.'
    write_json(inputs/'surface.json', surface)
    seal(inputs, [p.name for p in inputs.iterdir() if p.is_file() and p.name != 'complete.json'])
    destination = output/'issues'
    labels = DiskArray(inputs/'labels.npy')
    probabilities = DiskArray(inputs/surface['probability_file'])
    unique = DiskArray(inputs/'ownership_valid.npy')
    valid = DiskArray(inputs/'original_valid.npy')
    scoring = find_label_issues_disk(labels, probabilities, destination, [r['image'] for r in rows],
                                    pixel_chunk_size=args.pixel_chunk_size, sort_chunk_size=args.sort_chunk_size,
                                    valid=unique, monitor=monitor)
    broadcast_shared_issues(rows, inputs, destination)
    (destination/'valid.npy').replace(destination/'ranking_valid.npy')
    shutil.copyfile(inputs/'original_valid.npy', destination/'valid.npy')
    scoring.update(is_oof=False, checkpoint=str(args.checkpoint), checkpoint_sha256=hashes[str(args.checkpoint)],
                   scope=config['scope'], score_definition='A-model probability of the original given label.',
                   tie_break='Selected image index, then row-major pixel index.',
                   overlap_policy=surface['policy'], ranking_valid_file='ranking_valid.npy')
    write_json(destination/'scoring.json', scoring)
    issues, quality = DiskArray(destination/'ignore.npy'), DiskArray(destination/'label_quality.npy')
    records = []
    for index, row in enumerate(rows):
        usable, owned, flagged, q = valid[index], unique[index], issues[index], quality[index]
        predicted_normal = probabilities[index, 1] < .5
        seen = fit_images.get(row['image'])
        records.append({'index': index, 'image': row['image'], 'case': row['group'],
                        'task_id': row['task_id'], 'point': row['measurement_point'], 'split': partition[row['image']],
                        'same_image_used_for_fit': bool(seen and seen['image_sha256'] == row['image_sha256']),
                        'case_used_for_fit': row['group'] in fit_cases,
                        'valid_pixels': int(usable.sum()), 'issue_pixels': int((flagged & usable).sum()),
                        'unique_valid_pixels': int(owned.sum()), 'unique_issue_pixels': int((flagged & owned).sum()),
                        'unique_pred_normal_pixels': int((predicted_normal & owned).sum()),
                        'cl_clean_pct': 100*(1-(flagged & usable).sum()/usable.sum()) if usable.any() else None,
                        'mean_label_quality': float(q[usable].mean()) if usable.any() else None})
    by_point, by_split, overall = summarize(records)
    summary = {**config, 'status': 'completed', 'by_point': by_point, 'by_split': by_split, 'overall': overall,
               'store_results': False, 'thresholds': scoring['confident_thresholds'],
               'checkpoint_seen_images': sum(r['same_image_used_for_fit'] for r in records),
               'peak_observed_rss_sum_mib': monitor.maximum}
    write_csv(output/'per_image.csv', records)
    write_csv(output/'by_point.csv', by_point)
    write_csv(output/'by_split.csv', by_split)
    previews = output/'previews'
    previews.mkdir(exist_ok=True)
    body = ('<p>기존 전체 Train 학습 A 모델의 추론으로 계산했습니다. 재학습 없음. OOF 확률이 아닙니다.</p>'
            '<p>CL Clean = 의심 픽셀로 선택되지 않은 유효 픽셀 비율. 라벨의 실제 정확도나 함침율과는 다른 값입니다. '
            'Train/Val/Test 전체의 임계값과 후보 개수를 한 번에 계산했습니다.</p>'
            '<p>pred_impregnation_pct는 모델이 Exposure 이외로 예측한 면적 비율입니다.</p>'
            + table([{'point': '전체', **overall}, *by_point]) + '<h2>Split별</h2>' + table(by_split)
            + '<p><a href="per_image.csv">이미지별 CSV</a> · <a href="issues/scoring.json">CL 설정</a></p>')
    for point in ('P1', 'P2', 'P3'):
        selected = max((r for r in records if r['point'] == point), key=lambda r: r['issue_pixels'])
        index = selected['index']
        name = f'{point}_most_issues.png'
        save_panels(previews/name,
                    [read_gray(args.data/rows[index]['image']), labels[index]*255,
                     np.round(probabilities[index, 1]*255), issues[index].astype(np.uint8)*255,
                     np.round(np.nan_to_num(quality[index])*255)],
                    ['Original', 'Label: Exposure', 'A-model probability', 'CL issue pixels', 'Label quality'])
        body += f'<p>{point} 최대 의심 픽셀: {selected["issue_pixels"]:,} · Task {selected["task_id"]}</p><img src="previews/{name}">'
    page(output/'index.html', 'A model: P1/P2/P3 CL Clean 비율', body)
    for path, digest_before in hashes.items():
        if sha256(path) != digest_before:
            raise ValueError(f'Input changed during inference/audit: {path}')
    # Re-read image/mask hashes so completed results are tied to unchanged inputs.
    fresh, _ = load_frozen_splits(args.data, args.data/'splits.json')
    if select_central_splits(fresh, True) != splits:
        raise ValueError('Dataset changed during audit.')
    monitor.record('completed', force=True)
    summary['peak_observed_rss_sum_mib'] = monitor.maximum
    write_json(output/'summary.json', summary)
    seal(destination, sorted(str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()))
    (output/'RUNNING.json').unlink()
    seal(output, ['experiment.json', 'summary.json', 'tiles.json', 'by_point.csv', 'by_split.csv',
                  'per_image.csv', 'index.html'])
    print(table(by_point), flush=True)
    print(f'Report: {output / "index.html"}', flush=True)
    return output, summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, default=PROJECT/'data/changed_v3')
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='cuda')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--pixel-chunk-size', type=int, default=262144)
    p.add_argument('--sort-chunk-size', type=int, default=262144)
    p.add_argument('--memory-budget-mib', type=float, default=4096)
    p.add_argument('--min-free-ram-mib', type=float, default=1024)
    return p


def main(argv=None):
    run(parser().parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
