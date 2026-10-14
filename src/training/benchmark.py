"""Compare ordinary segmentation architectures on the complete frozen changed_v2 split."""

import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import sys

from .architectures import MODEL_IDS, model_spec
from .data import describe_split, load_tiles
from .resources import MemoryMonitor, run_command

PROJECT = Path(__file__).resolve().parents[2]

# The only original source versions eligible for this explicitly requested repair.
REPAIR_BASELINE = {
    'src/training/resources.py': 'd640a1b8155950046ec1dfab4b866393a735da54e38bb9b2392e3b7057ec5e4a',
    'src/training/train.py': 'f53f828e92bc57f53d83d02ad022f63f7d4d64c6565925c9374ca70d193f7f48',
    'src/training/benchmark_worker.py': 'd5f3a3797aa6ed2c0d37315226882226a710a2cca9d88ce4622a525ae8c451b5',
    'src/training/benchmark.py': '1499356667b6b8c879d1ac1ff1b501b7709a656077e5876777d9aa01927c65ed',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seal(directory, names):
    write_json(Path(directory)/'verified_files.json', {name: sha256(Path(directory)/name) for name in names})


def verify(directory):
    directory = Path(directory)
    saved = read_json(directory/'verified_files.json')
    for name, digest in saved.items():
        if Path(name).name != name or sha256(directory/name) != digest:
            raise ValueError(f'Changed or invalid completed artifact: {directory/name}')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data', type=Path, default=PROJECT/'data/changed_v2')
    result.add_argument('--output', type=Path)
    result.add_argument('--mode', choices=('models', 'prepare', 'repair-resume', 'memory-check', 'all'), default='all')
    result.add_argument('--models', choices=MODEL_IDS, nargs='+', default=list(MODEL_IDS))
    result.add_argument('--seeds', type=int, nargs='+', default=[42])
    result.add_argument('--epochs', type=int, default=500)
    result.add_argument('--patience', type=int, default=20)
    result.add_argument('--learning-rate', type=float, default=.001)
    result.add_argument('--micro-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--effective-batch-size', type=int, default=16)
    result.add_argument('--memory-budget-mib', type=float, default=4096)
    result.add_argument('--min-free-ram-mib', type=float, default=2048)
    result.add_argument('--vram-budget-mib', type=float, default=8192)
    result.add_argument('--min-free-vram-mib', type=float, default=1024)
    result.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--smoke-test', action='store_true', help='Partial one-epoch integration check, not a performance result.')
    return result


def frozen_splits(data):
    """No resplitting, no filtering, no label modification; verify every source file."""
    records, digest = load_tiles(data)
    manifest = read_json(data/'manifest.json')
    if manifest['summary'].get('mask_encoding', {}).get('ignore_value') is not None:
        raise ValueError('This benchmark uses original alpha > 0 labels, not ignore-valued revisions.')
    if any('valid_mask' in row for row in records):
        raise ValueError('This benchmark requires complete original tiles without external validity masks.')
    saved = read_json(data/'splits.json')
    if saved['manifest_sha256'] != digest:
        raise ValueError('Frozen split does not match the data manifest.')
    current = {row['image']: row for row in records}
    splits, used, groups = {}, set(), set()
    if set(saved['splits']) != {'train', 'val', 'test'}:
        raise ValueError('Expected exactly train/val/test.')
    for name in ('train', 'val', 'test'):
        rows = saved['splits'][name]
        if not rows:
            raise ValueError(f'Empty frozen split: {name}')
        split_groups = set()
        splits[name] = []
        for row in rows:
            actual = current.get(row['image'])
            if actual is None or any(row.get(key) != value for key, value in actual.items()):
                raise ValueError(f'Frozen source/label/geometry changed: {row["image"]}')
            if row['image'] in used:
                raise ValueError('Repeated tile across frozen splits.')
            used.add(row['image'])
            split_groups.add(row['group'])
            splits[name].append(actual)
        if split_groups & groups:
            raise ValueError('Case leakage between Train/Val/Test.')
        groups.update(split_groups)
    if used != set(current):
        raise ValueError('Frozen split does not include the entire dataset.')
    return splits, digest, saved.get('seed', 42)


def settings(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            if key not in ('mode', 'output', 'resume')}


def verify_resume_plan(output, plan, *, repair=False):
    """Keep the original plan, record a bounded logging/continuation repair separately."""
    original = read_json(output/'plan.json')
    if original == plan:
        return
    if {key: value for key, value in original.items() if key != 'sources'} != {key: value for key, value in plan.items() if key != 'sources'}:
        raise ValueError('Resume settings or dataset changed. Use a new Output directory.')
    compatibility_path = output/'resume_compatibility.json'
    if compatibility_path.exists():
        compatibility = read_json(compatibility_path)
        if (compatibility['original_plan_sha256'] == sha256(output/'plan.json')
                and compatibility['from_sources'] == original['sources']
                and compatibility['to_sources'] == plan['sources']):
            return
    changed = {name for name in set(original['sources']) | set(plan['sources'])
               if original['sources'].get(name) != plan['sources'].get(name)}
    if not repair:
        raise ValueError('Resume source changed. For the known temporary-checkpoint monitor fix, run -Mode repair-resume -Resume first.')
    if not changed or not changed <= REPAIR_BASELINE.keys() or any(original['sources'].get(name) != REPAIR_BASELINE[name] for name in changed):
        raise ValueError('This repair cannot bypass unrelated source/architecture/loss changes.')
    for path in (output/'runs').glob('*/seed_*/*/attempt_*/verified_files.json'):
        verify(path.parent)
    for path in (output/'memory_preflight').glob('micro_*/*/verified_files.json'):
        verify(path.parent)
    write_json(compatibility_path, dict(original_plan_sha256=sha256(output/'plan.json'),
               from_sources=original['sources'], to_sources=plan['sources'], changed=sorted(changed),
               reason='Fix transient .tmp stat race and continue interrupted epoch checkpoints; original labels, architecture, loss and completed artifacts preserved.'))
    print('Recorded compatible monitor/continuation repair; original plan and completed artifacts preserved.', flush=True)


def prepare(args):
    if min(args.epochs, args.patience, args.effective_batch_size) < 1:
        raise ValueError('Epochs, patience and effective batch must be positive.')
    if args.effective_batch_size % args.micro_batch_size:
        raise ValueError('Effective batch must be divisible by micro batch.')
    for key in ('learning_rate', 'memory_budget_mib', 'vram_budget_mib'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    for key in ('min_free_ram_mib', 'min_free_vram_mib'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Repeated model/seed.')
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError('Seeds must be in [0, 2**32).')
    args.data = args.data.resolve()
    if args.resume and args.output is None:
        raise ValueError('Resume requires the original output directory.')
    output = (args.output or PROJECT/'outputs/training'/f'segmentation_benchmark_{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    if output.is_relative_to(args.data) or args.data.is_relative_to(output):
        raise ValueError('Output and dataset directories must not contain each other.')
    splits, digest, split_seed = frozen_splits(args.data)
    files = sorted((PROJECT/'src/training').glob('*.py')) + [PROJECT/'src/inference/predict.py', PROJECT/'pyproject.toml', PROJECT/'uv.lock']
    plan = dict(protocol='ordinary_segmentation_architecture_comparison_v1', settings=settings(args),
                model_specs=[model_spec(model_id) for model_id in args.models],
                manifest_sha256=digest, frozen_split_sha256=sha256(args.data/'splits.json'),
                split_seed=split_seed, split_summary=describe_split(splits),
                sources={path.relative_to(PROJECT).as_posix(): sha256(path) for path in files},
                training='All original Train tiles, alpha > 0, basic augmentation, BCE + Dice, uniform sampling, from scratch.',
                selection='Mean best Validation Dice across requested seeds; Validation MAE then model id break ties. Test deferred until all fits complete.',
                memory='One child process per model/seed; no tensor cache; workers=0; AMP; checkpointing; common effective batch; measured RAM and bounded CUDA allocator.',
                metric_unit='Full 256x256 tiles, including overlap; P1/P2/P3 use their complete original tiles.')
    if args.resume:
        verify_resume_plan(output, plan, repair=args.mode == 'repair-resume')
        if read_json(output/'splits.json')['splits'] != splits:
            raise ValueError('Saved benchmark split changed.')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'plan.json', plan)
        write_json(output/'splits.json', dict(manifest_sha256=digest, seed=split_seed, splits=splits))
    monitor = MemoryMonitor(output, args.memory_budget_mib, args.min_free_ram_mib)
    monitor.record('prepared', force=True)
    # Conservative room for last+best Adam checkpoints and interrupted writes, plus imports.
    monitor.check_disk(len(args.models)*len(args.seeds)*3*1024**3 + 512*1024**2)
    print(f'Output: {output}\nFULL frozen split: {plan["split_summary"]}', flush=True)
    return output, splits, plan, monitor


def run_job(args, output, monitor, job, stage):
    job.update(data=str(args.data), memory_budget_mib=args.memory_budget_mib,
               min_free_ram_mib=args.min_free_ram_mib, vram_budget_mib=args.vram_budget_mib,
               min_free_vram_mib=args.min_free_vram_mib, device=args.device)
    path = output/'jobs'/f'{stage}.json'
    write_json(path, job)
    run_command([sys.executable, '-X', 'utf8', '-u', '-m', 'src.training.benchmark_worker', str(path)], monitor, stage)


def memory_checks(args, output, splits, monitor):
    if (output/'batch_plan.json').exists():
        saved = read_json(output/'batch_plan.json')
        if read_json(output/'batch_plan_verified.json')['sha256'] != sha256(output/'batch_plan.json'):
            raise ValueError('Saved batch plan changed.')
        for model_id, expected in zip(args.models, saved['preflights']):
            directory = output/'memory_preflight'/f'micro_{saved["micro_batch_size"]}'/model_id
            verify(directory)
            if read_json(directory/'result.json') != expected:
                raise ValueError('Saved preflight changed.')
        if len(saved['preflights']) != len(args.models):
            raise ValueError('Saved preflight model count changed.')
        monitor.record('reuse_fixed_batch_plan', force=True)
        return saved
    candidates = [args.micro_batch_size] + ([1] if args.micro_batch_size == 2 else [])
    last_error = None
    for micro in candidates:
        passed, retryable = [], False
        for model_id in args.models:
            directory = output/'memory_preflight'/f'micro_{micro}'/model_id
            if (directory/'verified_files.json').exists():
                verify(directory)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                job = dict(mode='preflight', output=str(directory), model_spec=model_spec(model_id),
                           rows=splits['train'][:2*args.effective_batch_size], batch_size=micro,
                           effective_batch_size=args.effective_batch_size, seed=args.seeds[0])
                try:
                    run_job(args, output, monitor, job, f'preflight_{model_id}_micro{micro}')
                except (RuntimeError, MemoryError) as error:
                    last_error = error
                    failure = read_json(directory/'error.json') if (directory/'error.json').exists() else {}
                    retryable = isinstance(error, MemoryError) or failure.get('memory_failure', False)
                    if not retryable:
                        raise
                    break
            passed.append(read_json(directory/'result.json'))
        if len(passed) == len(args.models):
            batch = dict(micro_batch_size=micro, gradient_accumulation_steps=args.effective_batch_size//micro,
                         effective_batch_size=args.effective_batch_size, preflights=passed)
            write_json(output/'batch_plan.json', batch)
            write_json(output/'batch_plan_verified.json', dict(sha256=sha256(output/'batch_plan.json')))
            return batch
        if retryable and micro != candidates[-1]:
            print('Memory limit reached; recheck EVERY model at common micro batch 1.', flush=True)
    raise MemoryError(f'Could not pass memory checks at micro batch 1: {last_error}')


def completed_attempt(root, mode):
    for attempt in sorted(root.glob('attempt_*'), reverse=True):
        if (attempt/'verified_files.json').is_file():
            verify(attempt)
            result = read_json(attempt/'result.json')
            if result.get('mode') == mode:
                return attempt
    return None


def new_attempt(root):
    root.mkdir(parents=True, exist_ok=True)
    index = max((int(path.name.split('_')[-1]) for path in root.glob('attempt_*')), default=0)+1
    directory = root/f'attempt_{index:03d}'
    directory.mkdir(exist_ok=False)
    return directory


def interrupted_checkpoint(root):
    for attempt in sorted(root.glob('attempt_*'), reverse=True):
        if not (attempt/'verified_files.json').exists() and all((attempt/name).is_file() for name in
                ('last.pt', 'best.pt', 'history.csv', 'config.json', 'splits.json')):
            return attempt/'last.pt'
    return None


def select_model(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['model'], []).append(row)
    candidates = [dict(model=model_id, mean_val_dice=sum(row['val_dice'] for row in group)/len(group),
                       mean_val_impregnation_mae_pp=sum(row['val_impregnation_mae_pp'] for row in group)/len(group))
                  for model_id, group in grouped.items()]
    return min(candidates, key=lambda row: (-row['mean_val_dice'], row['mean_val_impregnation_mae_pp'], row['model']))


def summarize(output, rows, selected=None, status='in_progress'):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['model'], []).append(row)
    aggregates = []
    for model_id, group in grouped.items():
        aggregate = dict(model=model_id, runs=len(group), parameters=group[0]['parameters'])
        for key in ('val_dice', 'val_iou', 'val_impregnation_mae_pp', 'test_dice', 'test_iou',
                    'test_recall', 'test_impregnation_mae_pp'):
            values = [row[key] for row in group if key in row]
            if len(values) != len(group):
                continue
            average = sum(values)/len(values)
            aggregate[key+'_mean'] = average
            aggregate[key+'_std'] = (sum((value-average)**2 for value in values)/(len(values)-1))**.5 if len(values)>1 else None
        aggregates.append(aggregate)
    write_json(output/'comparison.json', dict(status=status, selected=selected, runs=rows,
                                            model_aggregates=aggregates,
                                            smoke_test=read_json(output/'plan.json')['settings']['smoke_test']))
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (output/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    labels = ('model', 'seed', 'parameters', 'best_epoch', 'val_dice', 'test_dice', 'test_iou',
              'test_recall', 'test_impregnation_mae_pp', 'gpu_peak_reserved_mib', 'host_peak_rss_sum_mib')
    content = []
    for row in rows:
        values = []
        for key in labels:
            value = row.get(key, '')
            values.append(f'<td>{html.escape(f"{value:.4f}" if isinstance(value, float) else str(value))}</td>')
        content.append('<tr>'+''.join(values)+'</tr>')
    warning = 'Smoke test: partial integration check; these are NOT performance results.' if read_json(output/'plan.json')['settings']['smoke_test'] else 'All original labels; from scratch; fixed Case split. Higher Dice/IoU/Recall, lower MAE are better.'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Segmentation architecture comparison</title>'
        '<style>body{font:15px system-ui;margin:24px;background:#f7f9fc}table{border-collapse:collapse;background:white}'
        'td,th{padding:10px;border:1px solid #ddd}th{background:#e9eff7}td{text-align:right}</style>'
        f'<h1>Segmentation architecture comparison</h1><p>{html.escape(status)} — {html.escape(warning)}</p>'
        '<p>Metrics are fractions. The Test results never select the model. P1/P2/P3 metrics are in each Test result.json.</p>'
        f'<p>Validation-selected: {html.escape(str(selected))}</p><table><tr>'
        + ''.join(f'<th>{key}</th>' for key in labels)+'</tr>'+''.join(content)+'</table>', encoding='utf-8')


def execute(args, output, splits, plan, monitor, batch):
    rows = []
    for model_id in args.models:
        for seed in args.seeds:
            root = output/'runs'/model_id/f'seed_{seed}'
            directory = completed_attempt(root/'fit', 'fit')
            if directory is None:
                continuation = interrupted_checkpoint(root/'fit')
                directory = new_attempt(root/'fit')
                job = dict(mode='fit', output=str(directory), splits=splits, split_seed=plan['split_seed'],
                           manifest_sha256=plan['manifest_sha256'], model_spec=model_spec(model_id),
                           batch_size=batch['micro_batch_size'], effective_batch_size=args.effective_batch_size,
                           seed=seed, epochs=args.epochs, patience=args.patience, learning_rate=args.learning_rate,
                           smoke_test=args.smoke_test)
                if continuation is not None:
                    job['resume_checkpoint'] = str(continuation)
                    print(f'Continuing interrupted checkpoint: {continuation}', flush=True)
                run_job(args, output, monitor, job, f'fit_{model_id}_seed{seed}_{directory.name}')
            result, config = read_json(directory/'result.json'), read_json(directory/'config.json')
            expected = 'smoke_test_completed' if args.smoke_test else 'completed'
            if result['status'] != expected or result['test_status'] != 'skipped':
                raise ValueError('Expected completed validation-only ordinary training.')
            val = result['best_val']
            rows.append(dict(model=model_id, seed=seed, parameters=config['parameters'], best_epoch=result['best_epoch'],
                             epochs_completed=result['epochs_completed'], val_dice=val['dice'], val_iou=val['iou'],
                             val_impregnation_mae_pp=val['impregnation_mae_pp'], val_recall=val['recall'],
                             seconds=result['seconds'], gpu_peak_reserved_mib=result['gpu_peak_reserved_mib'],
                             host_peak_rss_sum_mib=result['host_peak_rss_sum_mib'],
                             micro_batch_size=batch['micro_batch_size'], effective_batch_size=args.effective_batch_size,
                             checkpoint=str(directory/'best.pt')))
            summarize(output, rows)
    selected = select_model(rows)
    write_json(output/'selected_model.json', selected)
    # Freeze selection after all fits, before opening any Test results.
    for row in rows:
        root = output/'runs'/row['model']/f'seed_{row["seed"]}'/'test'
        directory = completed_attempt(root, 'evaluate')
        if directory is None:
            directory = new_attempt(root)
            job = dict(mode='evaluate', output=str(directory), checkpoint=row['checkpoint'], rows=splits['test'],
                       batch_size=batch['micro_batch_size'], smoke_test=args.smoke_test)
            run_job(args, output, monitor, job, f'test_{row["model"]}_seed{row["seed"]}_{directory.name}')
        result = read_json(directory/'result.json')
        for key in ('dice', 'iou', 'precision', 'recall', 'impregnation_mae_pp', 'samples'):
            row['test_'+key] = result['all_tiles'][key]
        row['test_output'] = str(directory)
        summarize(output, rows, selected)
    status = 'smoke_test_completed' if args.smoke_test else 'completed'
    summarize(output, rows, selected, status)
    monitor.record('completed', force=True)
    return selected


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == 'models':
        for model_id in args.models:
            print(json.dumps(model_spec(model_id), ensure_ascii=False), flush=True)
        return 0
    output, splits, plan, monitor = prepare(args)
    try:
        if args.mode in ('prepare', 'repair-resume'):
            if args.mode == 'prepare':
                summarize(output, [], status='prepared')
        else:
            batch = memory_checks(args, output, splits, monitor)
            if args.mode == 'memory-check':
                summarize(output, [], status='memory_checked')
            else:
                execute(args, output, splits, plan, monitor, batch)
        write_json(output/('resume_repair_verification.json' if args.mode == 'repair-resume' else 'verification.json'), dict(status='passed', mode=args.mode,
                    host_peak_rss_sum_mib=monitor.maximum, split_summary=plan['split_summary'],
                    original_manifest_unchanged=sha256(args.data/'manifest.json') == plan['manifest_sha256'],
                    original_split_unchanged=sha256(args.data/'splits.json') == plan['frozen_split_sha256']))
    except BaseException as error:
        write_json(output/'error.json', dict(error=str(error), exception=type(error).__name__))
        raise
    print(f'Saved ordinary segmentation comparison: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
