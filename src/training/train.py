"""Train Exposure segmentation: python -m src.training.train --help."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

from .data import ExposureDataset, describe_split, load_tiles, split_by_case
from .augmentation import MODES, describe_augmentation, save_preview
from .losses import LOSS_MODES, describe_loss
from .sampling import SAMPLING_MODES, build_sampling_plan, make_sampler, validate_sampling

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT / "data/raw/rectangle_roi_tiles_256"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def capture_continuation(loaders, device, stale, train_metrics):
    import torch
    numpy_state = np.random.get_state()
    state = dict(stale=stale, train_metrics=train_metrics, torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                 python_rng=random.getstate(),
                 numpy_rng=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]], loaders={})
    for name, loader in loaders.items():
        dataset = loader.dataset
        generators = getattr(dataset, '_generators', None)
        state['loaders'][name] = dict(shuffle=loader.generator.get_state(),
            augmentation=[generator.get_state() for generator in generators] if generators else None)
    return state


def restore_continuation(state, loaders, device):
    import torch
    for name, saved in state['loaders'].items():
        loaders[name].generator.set_state(saved['shuffle'])
        if saved['augmentation']:
            for generator, value in zip(loaders[name].dataset.generators(), saved['augmentation']):
                generator.set_state(value)
    random.setstate(state['python_rng'])
    numpy_state = state['numpy_rng']
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch_rng'])
    if device.type == 'cuda' and state['cuda_rng'] is not None:
        torch.cuda.set_rng_state(state['cuda_rng'], device)


def restore_fit(checkpoint_path, model, optimizer, scheduler, scaler, loaders, device, args, config, output):
    """Read the last complete epoch into a NEW attempt; never modify the source run."""
    import torch
    source = Path(checkpoint_path).resolve()
    if source.parent == output.resolve():
        raise ValueError('Checkpoint continuation requires a new attempt directory.')
    prior_config = json.loads((source.parent/'config.json').read_text(encoding='utf-8'))
    keys = ('data', 'model_spec', 'seed', 'split_seed', 'augmentation', 'loss_config', 'sampling_config',
            'batch_size', 'gradient_accumulation_steps', 'learning_rate', 'smoke_test', 'epochs', 'patience',
            'workers', 'pin_memory', 'activation_checkpointing', 'threshold', 'manifest_sha256', 'device', 'no_amp')
    if any(prior_config.get(key) != config.get(key) for key in keys):
        raise ValueError('Interrupted fit has different data, model or training settings.')
    if json.loads((source.parent/'splits.json').read_text(encoding='utf-8')) != json.loads((output/'splits.json').read_text(encoding='utf-8')):
        raise ValueError('Interrupted fit split differs.')
    saved = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('model_spec') != config.get('model_spec') or saved['seed'] != args.seed:
        raise ValueError('Checkpoint model/seed differs.')
    epoch = int(saved['epoch'])
    if not 1 <= epoch <= args.epochs:
        raise ValueError('Invalid last checkpoint epoch.')
    previous_best = source.parent/'best.pt'
    best = torch.load(previous_best, map_location='cpu', weights_only=True, mmap=True)
    if best['epoch'] > epoch or best['val_metrics']['dice'] > saved['best_val_dice']:
        raise ValueError('Last/best checkpoints are inconsistent.')
    if best['val_metrics']['dice'] == saved['best_val_dice']:
        best_path, best_epoch = previous_best, int(best['epoch'])
    elif saved['val_metrics']['dice'] == saved['best_val_dice']:
        best_path, best_epoch = source, epoch
    else:
        raise ValueError('The selected best checkpoint is missing.')
    del best
    with (source.parent/'history.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        history = [row for row in reader if row.get('epoch') and int(row['epoch']) <= epoch]
    if [int(row['epoch']) for row in history] not in (list(range(1, epoch+1)), list(range(1, epoch))):
        raise ValueError('Interrupted history is incomplete beyond the latest checkpoint row.')
    state = saved.get('continuation_state')
    if not history or int(history[-1]['epoch']) != epoch:
        recovered = dict(epoch=epoch, learning_rate=saved['optimizer_state_dict']['param_groups'][0]['lr'], seconds=0,
                         **{f'val_{key}': value for key, value in saved['val_metrics'].items()},
                         **({f'train_{key}': value for key, value in state['train_metrics'].items()} if state else {}))
        history.append(recovered)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    scheduler.load_state_dict(saved['scheduler_state_dict'])
    scaler.load_state_dict(saved['scaler_state_dict'])
    stale = int(state['stale']) if state else epoch-best_epoch
    if state:
        restore_continuation(state, loaders, device)
    else:
        print('Legacy checkpoint: RNG states were not saved. Weights/Adam/scheduler/AMP/early-stopping resume; random streams restart.', flush=True)
    best_dice = float(saved['best_val_dice'])
    del saved
    # Streaming copies preserve the source snapshots and bound additional RAM.
    for original, target in ((best_path, output/'best.pt'), (source, output/'last.pt')):
        temporary = target.with_suffix('.tmp')
        shutil.copyfile(original, temporary)
        temporary.replace(target)
    with (output/'history.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    previous_seconds = sum(float(row.get('seconds') or 0) for row in history)
    info = dict(checkpoint=str(source), restored_epoch=epoch, next_epoch=epoch+1, best_epoch=best_epoch,
                best_val_dice=best_dice, stale=stale, rng_restored=state is not None,
                history_rows=len(history), source_attempt_preserved=True)
    write_json(output/'resume.json', info)
    config['continuation'] = info
    write_json(output/'config.json', config)
    print(f'Resuming completed epoch {epoch}; next epoch {epoch+1}; best epoch {best_epoch}.', flush=True)
    return epoch, best_dice, best_epoch, stale, previous_seconds


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=DEFAULT_DATA)
    result.add_argument("--output", type=Path)
    result.add_argument("--position", choices=("all", "000", "180"), default="all")
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--base-channels", type=int, default=32)
    result.add_argument('--gradient-accumulation-steps', type=int, default=1)
    result.add_argument('--activation-checkpointing', action='store_true')
    result.add_argument('--vram-budget-mib', type=float, default=0)
    result.add_argument('--min-free-vram-mib', type=float, default=0)
    result.add_argument("--augmentation", choices=MODES, default="basic")
    result.add_argument("--loss", dest="loss_mode", choices=LOSS_MODES, default="bce_dice")
    result.add_argument("--tversky-alpha", type=float, default=0.3, help="False-positive penalty.")
    result.add_argument("--tversky-beta", type=float, default=0.7, help="False-negative penalty; alpha + beta = 1.")
    result.add_argument("--sampling", choices=SAMPLING_MODES, default="uniform")
    result.add_argument("--large-region-fraction", type=float, default=0.05)
    result.add_argument("--large-region-weight", type=float, default=3.0)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--split-seed", type=int, help="Case split seed; defaults to --seed for compatibility.")
    result.add_argument("--val-fraction", type=float, default=0.15)
    result.add_argument("--test-fraction", type=float, default=0.15)
    result.add_argument("--patience", type=int, default=20)
    result.add_argument("--threshold", type=float, default=0.5)
    result.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    result.add_argument("--no-amp", action="store_true")
    result.add_argument("--skip-test", action="store_true", help="Choose models using validation only; defer test evaluation.")
    result.add_argument("--prepare-only", action="store_true", help="Validate PNGs and save case splits; no torch required.")
    result.add_argument("--smoke-test", action="store_true", help="One epoch, two batches per split; not a performance evaluation.")
    return result


def run_epoch(model, loader, device, threshold, amp, optimizer=None, scaler=None, max_batches=None,
              loss_options=None, loss_function=None, progress_callback=None,
              gradient_accumulation_steps=1, vram_guard=None):
    import torch
    from .model import SegmentationMetrics, segmentation_loss

    training = optimizer is not None
    if gradient_accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be positive.')
    model.train(training)
    metrics = SegmentationMetrics(threshold)
    pending_batches, pending_samples = 0, 0
    normalizer = max(1, (getattr(loader, 'batch_size', None) or 1) * gradient_accumulation_steps)
    if training:
        optimizer.zero_grad(set_to_none=True)

    def update_parameters():
        scaler.unscale_(optimizer)
        # Average over supervised images, including a short final accumulation group.
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(normalizer / pending_samples)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, targets = batch[:2]
        if vram_guard:
            vram_guard.check('train_before' if training else 'val_before')
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        valid = batch[3].to(device, non_blocking=True) if len(batch) == 4 else None
        if valid is not None and not bool(valid.any()):
            if training:
                pending_batches += 1
                if pending_batches == gradient_accumulation_steps:
                    if pending_samples:
                        update_parameters()
                    pending_batches, pending_samples = 0, 0
            continue  # No supervision: also skip the AdamW weight-decay step.
        supervised = int(valid.flatten(1).any(1).sum()) if valid is not None else images.shape[0]
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = (loss_function(logits, targets, valid) if loss_function is not None
                        else segmentation_loss(logits, targets, **(loss_options or {})))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite loss; check images and learning rate.")
            if training:
                scaler.scale(loss * (supervised / normalizer)).backward()
                pending_batches += 1
                pending_samples += supervised
                if pending_batches == gradient_accumulation_steps:
                    update_parameters()
                    pending_batches, pending_samples = 0, 0
        metrics.update(logits.detach().float(), targets, loss.detach().item(), valid=valid)
        if progress_callback is not None:
            progress_callback('train' if training else 'val', metrics.samples, metrics.samples*65536,
                              images.shape[0]*65536)
        if vram_guard:
            vram_guard.check('train_after' if training else 'val_after')
        del logits, loss, images, targets, valid
    if training and pending_samples:
        update_parameters()
    if vram_guard:
        vram_guard.check('epoch_finished', force=True)
    return metrics.compute()


def save_test_predictions(model, loader, rows, output, device, threshold, amp, max_batches=None):
    """Save one CSV row per evaluated test tile, with both impregnation rates."""
    import torch

    columns = ("image", "case", "capture", "position", "task_id", "true_exposure_pixels",
               "pred_exposure_pixels", "true_impregnation_pct", "pred_impregnation_pct",
               "measurement_point", "alignment", "roi_needs_review")
    with (output / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        model.eval()
        with torch.inference_mode():
            for batch_index, (images, targets, indices) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(images.to(device))
                predicted = (logits.float().sigmoid() >= threshold).sum((1, 2, 3)).cpu().tolist()
                actual = targets.sum((1, 2, 3)).tolist()
                for index, pred, truth in zip(indices.tolist(), predicted, actual):
                    source = rows[index]
                    writer.writerow({
                        **{key: source[key] for key in columns[:5]},
                        "true_exposure_pixels": int(truth), "pred_exposure_pixels": int(pred),
                        "true_impregnation_pct": 100 * (1 - truth / 65536),
                        "pred_impregnation_pct": 100 * (1 - pred / 65536),
                        **{key: source.get(key, "") for key in columns[9:]},
                    })


def train(args, splits, output, config, sampling_weights, *, train_dataset=None, train_loss=None,
          evaluation_datasets=None, progress_callback=None, model_spec=None, resume_checkpoint=None):
    import torch
    from torch.utils.data import DataLoader
    from .model import UNet

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install with run_train.ps1 -InstallDependencies or use -Device cpu.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    from .memory import CudaMemoryGuard
    guard = CudaMemoryGuard(device, output, getattr(args, 'vram_budget_mib', 0),
                            getattr(args, 'min_free_vram_mib', 0))
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
    config["environment"] = {
        "python": sys.version, "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp": amp,
    }
    save_preview(args.data, splits["train"], args.augmentation, output / "augmentation_preview.png", args.seed)
    if model_spec is None:
        model = UNet(args.base_channels, getattr(args, 'activation_checkpointing', False))
    else:
        from .architectures import build_model
        model = build_model(model_spec, getattr(args, 'activation_checkpointing', False))
        config['model_spec'] = model_spec
    model = model.to(device)
    config["parameters"] = sum(p.numel() for p in model.parameters())
    write_json(output / "config.json", config)
    print(f"Device: {device} | {config['environment']['gpu']} | AMP: {amp}", flush=True)
    loaders = {}
    loss_options = {"mode": args.loss_mode, "alpha": args.tversky_alpha, "beta": args.tversky_beta}
    for index, (name, rows) in enumerate(splits.items()):
        generator = torch.Generator().manual_seed(args.seed + index)
        sampler = make_sampler(sampling_weights, args.sampling, args.seed) if name == "train" else None
        loaders[name] = DataLoader(
            (train_dataset if name == "train" and train_dataset is not None else
             (evaluation_datasets[name] if evaluation_datasets and name in evaluation_datasets else
              ExposureDataset(args.data, rows, augment=name == "train", augmentation=args.augmentation, seed=args.seed))), batch_size=args.batch_size,
            shuffle=name == "train" and sampler is None, num_workers=args.workers,
            pin_memory=getattr(args, 'pin_memory', device.type == "cuda"), persistent_workers=False,
            generator=generator, sampler=sampler,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001,
                                  **({'foreach': False} if getattr(args, 'vram_budget_mib', 0) else {}))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epochs, max_batches = (1, 2) if args.smoke_test else (args.epochs, None)
    best_dice, best_epoch, stale = -1.0, 0, 0
    resumed_epoch, previous_seconds, checkpoint = 0, 0., None
    if resume_checkpoint is not None:
        if args.workers != 0:
            raise ValueError('Checkpoint continuation requires DataLoader workers=0.')
        resumed_epoch, best_dice, best_epoch, stale, previous_seconds = restore_fit(
            resume_checkpoint, model, optimizer, scheduler, scaler, loaders, device, args, config, output)
    epoch = resumed_epoch
    started = time.perf_counter()
    metric_names = ("loss", "dice", "iou", "precision", "recall", "impregnation_mae_pp", "samples")
    fields = ["epoch", "learning_rate", "seconds"] + [f"{s}_{k}" for s in ("train", "val") for k in metric_names]
    with (output / "history.csv").open("a" if resumed_epoch else "w", encoding="utf-8", newline="") as history:
        writer = csv.DictWriter(history, fieldnames=fields)
        if not resumed_epoch:
            writer.writeheader()
        remaining_epochs = range(resumed_epoch + 1, epochs + 1) if stale < args.patience else ()
        for epoch in remaining_epochs:
            tick = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_metrics = run_epoch(model, loaders["train"], device, args.threshold, amp,
                                      optimizer, scaler, max_batches, loss_options=loss_options,
                                      loss_function=train_loss, progress_callback=progress_callback,
                                      gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
                                      vram_guard=guard)
            val_metrics = run_epoch(model, loaders["val"], device, args.threshold, amp,
                                    max_batches=max_batches, loss_options=loss_options,
                                    loss_function=train_loss if evaluation_datasets else None,
                                    progress_callback=progress_callback, vram_guard=guard)
            scheduler.step(val_metrics["loss"])
            improved = val_metrics["dice"] > best_dice
            if improved:
                best_dice, best_epoch, stale = val_metrics["dice"], epoch, 0
            else:
                stale += 1
            checkpoint = {
                "architecture": "grayscale_unet", "base_channels": args.base_channels,
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "threshold": args.threshold, "label": "Exposure", "mask_rule": "alpha > 0",
                "normalization": "uint8 / 255", "best_val_dice": best_dice,
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(), "smoke_test": args.smoke_test,
                "val_metrics": val_metrics, "augmentation": describe_augmentation(args.augmentation),
                "loss_config": config["loss_config"], "sampling_config": config["sampling_config"],
                "seed": args.seed, "split_seed": args.split_seed,
                **({"confident": config["confident"]} if "confident" in config else {}),
            }
            if model_spec is not None:
                checkpoint['model_spec'] = model_spec
                checkpoint['continuation_state'] = capture_continuation(loaders, device, stale, train_metrics)
                if model_spec['family'] != 'unet':
                    checkpoint['architecture'] = 'grayscale_segmentation_benchmark'
            # Write to a temporary path so interrupted writes leave the prior checkpoint intact.
            torch.save(checkpoint, output / "last.tmp")
            (output / "last.tmp").replace(output / "last.pt")
            if improved:
                torch.save(checkpoint, output / "best.tmp")
                (output / "best.tmp").replace(output / "best.pt")
            guard.check('checkpoint_saved', force=True)
            row = {"epoch": epoch, "learning_rate": lr, "seconds": time.perf_counter() - tick,
                   **{f"train_{key}": value for key, value in train_metrics.items()},
                   **{f"val_{key}": value for key, value in val_metrics.items()}}
            writer.writerow(row)
            history.flush()
            print(f"Epoch {epoch}/{epochs} | train loss {train_metrics['loss']:.4f} | "
                  f"val Dice {val_metrics['dice']:.4f}, IoU {val_metrics['iou']:.4f}, "
                  f"impregnation MAE {val_metrics['impregnation_mae_pp']:.3f} pp | "
                  f"{row['seconds']:.1f}s", flush=True)
            if stale >= args.patience:
                print(f"Early stopping: no validation Dice improvement for {stale} epochs.", flush=True)
                break
    # Test never participates in checkpoint selection or the learning-rate schedule.
    # Release Adam state before loading the selected model; never load optimizer state onto CUDA.
    del checkpoint, optimizer, scheduler, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    best = torch.load(output / "best.pt", map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(best["model_state_dict"])
    best_val_metrics = best['val_metrics']
    del best
    test_metrics = None
    if not args.skip_test:
        test_metrics = run_epoch(model, loaders["test"], device, args.threshold, amp,
                                 max_batches=max_batches, loss_options=loss_options)
        save_test_predictions(model, loaders["test"], splits["test"], output, device, args.threshold, amp, max_batches)
    write_json(output / "result.json", {
        "status": "smoke_test_completed" if args.smoke_test else "completed",
        "best_epoch": best_epoch, "best_val_dice": best_dice, "epochs_completed": epoch,
        "best_val": best_val_metrics, "test": test_metrics,
        "test_is_partial": args.smoke_test and not args.skip_test,
        "test_status": "skipped" if args.skip_test else "partial" if args.smoke_test else "completed",
        "seconds": previous_seconds + time.perf_counter() - started,
        "resumed_from_epoch": resumed_epoch,
        "gpu_peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
        'gpu_peak_reserved_mib': guard.peak_reserved_mib if device.type == 'cuda' else None,
        'micro_batch_size': args.batch_size,
        'gradient_accumulation_steps': getattr(args, 'gradient_accumulation_steps', 1),
        'effective_batch_size': args.batch_size * getattr(args, 'gradient_accumulation_steps', 1),
        "checkpoint": "best.pt", "split_summary": describe_split(splits),
    })
    print(f"Saved: {output}", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    loss_config = describe_loss(args.loss_mode, args.tversky_alpha, args.tversky_beta)
    validate_sampling(args.sampling, args.large_region_fraction, args.large_region_weight)
    if args.prepare_only and args.smoke_test:
        raise ValueError("Choose either --prepare-only or --smoke-test.")
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not 0 < args.threshold < 1 or args.seed < 0 or args.seed >= 2**32
            or args.split_seed < 0 or args.split_seed >= 2**32
            or args.base_channels < 8 or args.base_channels % 8
            or args.gradient_accumulation_steps < 1 or min(args.vram_budget_mib, args.min_free_vram_mib) < 0):
        raise ValueError("Invalid training parameters. Use positive epochs/batch/lr/patience and base channels divisible by 8.")
    args.data = args.data.resolve()
    print("Validating paired tiles and case split...", flush=True)
    records, manifest_hash = load_tiles(args.data, args.position)
    splits = split_by_case(records, args.split_seed, args.val_fraction, args.test_fraction)
    summary = describe_split(splits)
    sampling_weights, sampling_plan = build_sampling_plan(
        args.data, splits["train"], args.sampling, args.large_region_fraction, args.large_region_weight)
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "prepare" if args.prepare_only else "smoke" if args.smoke_test else "unet"
    output = (args.output or PROJECT / "outputs/training" / f"{prefix}_{stamp}").resolve()
    if output.is_relative_to(args.data):
        raise ValueError("Training output must be outside the tile dataset.")
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"output": str(output), "manifest_sha256": manifest_hash, "split_unit": "case number (fixed/moving together)",
                   "model": "UNet from scratch", "loss": "0.5 BCE + 0.5 " + loss_config["overlap"], "label": "Exposure",
                   "target_rule": "alpha > 0", "split_summary": summary})
    config["loss_config"] = loss_config
    config["sampling_config"] = sampling_plan["summary"]
    config["augmentation_config"] = describe_augmentation(args.augmentation)
    config["augmentation_rng"] = "independent geometry and photometric generators; seed independent of model width"
    write_json(output / "config.json", config)
    write_json(output / "sampling.json", sampling_plan)
    write_json(output / "splits.json", {"manifest_sha256": manifest_hash, "seed": args.split_seed,
                                         "summary": summary, "splits": splits})
    if args.prepare_only:
        write_json(output / "result.json", {"status": "prepared", "split_summary": summary})
        print(f"Prepared: {output}", flush=True)
        return 0
    try:
        train(args, splits, output, config, sampling_weights)
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "result.json", {"status": "failed", "error": str(error), "exception": type(error).__name__})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
