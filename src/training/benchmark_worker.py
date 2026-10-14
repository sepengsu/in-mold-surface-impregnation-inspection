"""Short-lived ordinary training, memory probe or Test-evaluation worker; no CL."""

import argparse
import csv
from pathlib import Path
import random

import numpy as np

from .architectures import build_model, parameter_count
from .benchmark import read_json, write_json, seal
from .data import ExposureDataset
from .resources import MemoryMonitor


def setup(job):
    import torch
    from .memory import CudaMemoryGuard
    torch.set_num_threads(4)
    random.seed(job.get('seed', 42))
    np.random.seed(job.get('seed', 42))
    torch.manual_seed(job.get('seed', 42))
    device = torch.device(job['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no automatic CPU fallback.')
        torch.cuda.manual_seed_all(job.get('seed', 42))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
    guard = CudaMemoryGuard(device, job['output'], job['vram_budget_mib'], job['min_free_vram_mib'])
    return device, guard


def preflight(job, monitor):
    import torch
    from torch.utils.data import DataLoader
    from .train import run_epoch
    from src.inference.predict import load_checkpoint
    device, guard = setup(job)
    amp = device.type == 'cuda'
    model = build_model(job['model_spec'], True).to(device)
    parameters = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    loader = DataLoader(ExposureDataset(job['data'], job['rows'], augment=True, seed=job['seed']),
                        batch_size=job['batch_size'], num_workers=0, pin_memory=False)
    accumulation = job['effective_batch_size']//job['batch_size']
    updates = 0
    for attempt in range(5):
        metrics = run_epoch(model, loader, device, .5, amp, optimizer, scaler,
                            gradient_accumulation_steps=accumulation, vram_guard=guard,
                            progress_callback=monitor.record)
        updates = min((int(state['step']) for state in optimizer.state.values()), default=0)
        if updates >= 2:
            break
    if updates < 2:
        raise RuntimeError('Could not finish two finite AdamW updates; check numerical stability.')
    path = Path(job['output'])/'checkpoint_probe.pt'
    spec = job['model_spec']
    torch.save(dict(architecture='grayscale_unet' if spec['family'] == 'unet' else 'grayscale_segmentation_benchmark',
                    model_spec=spec, base_channels=spec.get('base_channels', 32), model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), epoch=0, threshold=.5, smoke_test=True,
                    label='Exposure', mask_rule='alpha > 0', normalization='uint8 / 255'), path)
    monitor.record('checkpoint_saved', force=True)
    del model, optimizer, scaler
    if amp:
        torch.cuda.empty_cache()
    model, _ = load_checkpoint(path, device)
    x = next(iter(loader))[0].to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        prediction = model(x)
    if prediction.shape != x.shape or not bool(torch.isfinite(prediction).all()):
        raise RuntimeError('Expected finite, aligned, one-logit-per-pixel output.')
    guard.check('inference', force=True)
    monitor.record('inference', force=True)
    path.unlink()  # Only the throwaway checkpoint produced by this job.
    write_json(Path(job['output'])/'result.json', dict(status='passed', mode='preflight', model_spec=spec,
               parameters=parameters, tiles_tested=len(job['rows']), optimizer_updates=updates,
               micro_batch_size=job['batch_size'], gradient_accumulation_steps=accumulation,
               effective_batch_size=job['effective_batch_size'], training_metrics=metrics,
               gpu_peak_reserved_mib=guard.peak_reserved_mib if amp else None,
               gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if amp else None,
               host_peak_rss_sum_mib=monitor.maximum, note='Memory check only; no performance conclusion.'))
    seal(job['output'], ('result.json',))
    print(f"Memory check passed: {spec['id']} | {parameters:,} parameters | reserved {guard.peak_reserved_mib:.0f} MiB", flush=True)


def fit(job, monitor):
    from .train import parser, train
    from .augmentation import describe_augmentation
    from .losses import describe_loss
    from .sampling import build_sampling_plan
    args = parser().parse_args([])
    args.data, args.output = Path(job['data']), Path(job['output'])
    for key in ('epochs', 'patience', 'learning_rate', 'seed', 'split_seed', 'device', 'smoke_test',
                'vram_budget_mib', 'min_free_vram_mib'):
        setattr(args, key, job[key])
    args.batch_size = job['batch_size']
    args.base_channels = job['model_spec'].get('base_channels', 32)
    args.gradient_accumulation_steps = job['effective_batch_size']//args.batch_size
    args.activation_checkpointing, args.skip_test, args.pin_memory = True, True, False
    args.workers = 0
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model=job['model_spec']['id'], model_spec=job['model_spec'], manifest_sha256=job['manifest_sha256'],
                  target_rule='alpha > 0', label='Exposure', experiment='ordinary_supervised_segmentation',
                  loss_config=describe_loss('bce_dice'), augmentation_config=describe_augmentation(args.augmentation))
    weights, sampling = build_sampling_plan(args.data, job['splits']['train'], 'uniform', .05, 3)
    config['sampling_config'] = sampling['summary']
    write_json(args.output/'splits.json', dict(manifest_sha256=job['manifest_sha256'], seed=args.split_seed, splits=job['splits']))
    train(args, job['splits'], args.output, config, weights, model_spec=job['model_spec'], progress_callback=monitor.record,
          resume_checkpoint=job.get('resume_checkpoint'))
    monitor.record('fit_finished', force=True)
    result = read_json(args.output/'result.json')
    result.update(mode='fit', host_peak_rss_sum_mib=monitor.maximum)
    write_json(args.output/'result.json', result)
    seal(args.output, ('best.pt', 'result.json', 'config.json', 'splits.json', 'history.csv'))


def evaluate(job, monitor):
    import torch
    from torch.utils.data import DataLoader
    from .model import SegmentationMetrics, segmentation_loss
    from src.inference.predict import load_checkpoint
    device, guard = setup(job)
    amp = device.type == 'cuda'
    model, metadata = load_checkpoint(Path(job['checkpoint']), device)
    loader = DataLoader(ExposureDataset(job['data'], job['rows']), batch_size=job['batch_size'],
                        num_workers=0, pin_memory=False)
    metrics = {'all_tiles': SegmentationMetrics(), **{point: SegmentationMetrics() for point in ('P1', 'P2', 'P3')}}
    columns = ('image', 'case', 'capture', 'position', 'task_id', 'measurement_point', 'true_exposure_pixels',
               'pred_exposure_pixels', 'true_impregnation_pct', 'pred_impregnation_pct', 'error_pp', 'tp', 'fp', 'fn')
    output = Path(job['output'])
    with (output/'test_predictions.csv').open('w', encoding='utf-8-sig', newline='') as all_stream, \
         (output/'central_points.csv').open('w', encoding='utf-8-sig', newline='') as point_stream:
        writers = [csv.DictWriter(stream, fieldnames=columns) for stream in (all_stream, point_stream)]
        for writer in writers:
            writer.writeheader()
        with torch.inference_mode():
            for batch_index, (x, y, indices) in enumerate(loader):
                if job['smoke_test'] and batch_index >= 2:
                    break
                guard.check('test_before')
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(x)
                    loss = segmentation_loss(logits, y)
                metrics['all_tiles'].update(logits.float(), y, loss.item())
                predictions, targets = logits.float().sigmoid() >= .5, y > .5
                for local, index in enumerate(indices.tolist()):
                    row = job['rows'][index]
                    p, t = predictions[local], targets[local]
                    truth, predicted = int(t.sum()), int(p.sum())
                    point = row.get('measurement_point')
                    record = {key: row.get(key, '') for key in columns[:6]}
                    record.update(true_exposure_pixels=truth, pred_exposure_pixels=predicted,
                                  true_impregnation_pct=100*(1-truth/65536), pred_impregnation_pct=100*(1-predicted/65536),
                                  error_pp=100*abs(truth-predicted)/65536, tp=int((p&t).sum()),
                                  fp=int((p&~t).sum()), fn=int((~p&t).sum()))
                    writers[0].writerow(record)
                    if point in ('P1', 'P2', 'P3'):
                        logits_one, target_one = logits[local:local+1].float(), y[local:local+1]
                        metrics[point].update(logits_one, target_one, segmentation_loss(logits_one, target_one).item())
                        writers[1].writerow(record)
                monitor.record('test', metrics['all_tiles'].samples)
                guard.check('test_after')
                del x, y, logits, loss, predictions, targets
    monitor.record('test_finished', force=True)
    write_json(output/'result.json', dict(status='completed', mode='evaluate', smoke_test=job['smoke_test'],
               test_is_partial=job['smoke_test'], checkpoint_sha256=metadata['sha256'],
               **{key: value.compute() if value.samples else None for key, value in metrics.items()},
               gpu_peak_reserved_mib=guard.peak_reserved_mib if amp else None, host_peak_rss_sum_mib=monitor.maximum,
               measurement_policy='All original complete 256x256 tiles; P1/P2/P3 never use overlap-cropped areas.'))
    seal(output, ('result.json', 'test_predictions.csv', 'central_points.csv'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    job = read_json(parser.parse_args(argv).job)
    monitor = MemoryMonitor(job['output'], job['memory_budget_mib'], job['min_free_ram_mib'])
    try:
        monitor.record(job['mode'], force=True)
        {'preflight': preflight, 'fit': fit, 'evaluate': evaluate}[job['mode']](job, monitor)
    except BaseException as error:
        import torch
        memory_failure = isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError))
        write_json(Path(job['output'])/'error.json', dict(error=str(error), exception=type(error).__name__,
                                                      mode=job['mode'], memory_failure=memory_failure))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
