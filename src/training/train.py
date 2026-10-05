"""Train Exposure segmentation: python -m src.training.train --help."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
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


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=DEFAULT_DATA)
    result.add_argument("--output", type=Path)
    result.add_argument("--position", choices=("all", "000", "180"), default="all")
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--base-channels", type=int, default=32)
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
              loss_options=None):
    import torch
    from .model import SegmentationMetrics, segmentation_loss

    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(threshold)
    for batch_index, (images, targets, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = segmentation_loss(logits, targets, **(loss_options or {}))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite loss; check images and learning rate.")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        metrics.update(logits.detach().float(), targets, loss.detach().item())
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


def train(args, splits, output, config, sampling_weights):
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
    model = UNet(args.base_channels).to(device)
    config["parameters"] = sum(p.numel() for p in model.parameters())
    write_json(output / "config.json", config)
    print(f"Device: {device} | {config['environment']['gpu']} | AMP: {amp}", flush=True)
    loaders = {}
    loss_options = {"mode": args.loss_mode, "alpha": args.tversky_alpha, "beta": args.tversky_beta}
    for index, (name, rows) in enumerate(splits.items()):
        generator = torch.Generator().manual_seed(args.seed + index)
        sampler = make_sampler(sampling_weights, args.sampling, args.seed) if name == "train" else None
        loaders[name] = DataLoader(
            ExposureDataset(args.data, rows, augment=name == "train", augmentation=args.augmentation, seed=args.seed), batch_size=args.batch_size,
            shuffle=name == "train" and sampler is None, num_workers=args.workers, pin_memory=device.type == "cuda",
            generator=generator, sampler=sampler,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epochs, max_batches = (1, 2) if args.smoke_test else (args.epochs, None)
    best_dice, best_epoch, stale = -1.0, 0, 0
    started = time.perf_counter()
    metric_names = ("loss", "dice", "iou", "precision", "recall", "impregnation_mae_pp", "samples")
    fields = ["epoch", "learning_rate", "seconds"] + [f"{s}_{k}" for s in ("train", "val") for k in metric_names]
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as history:
        writer = csv.DictWriter(history, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            tick = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_metrics = run_epoch(model, loaders["train"], device, args.threshold, amp,
                                      optimizer, scaler, max_batches, loss_options=loss_options)
            val_metrics = run_epoch(model, loaders["val"], device, args.threshold, amp,
                                    max_batches=max_batches, loss_options=loss_options)
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
            }
            # Write to a temporary path so interrupted writes leave the prior checkpoint intact.
            torch.save(checkpoint, output / "last.tmp")
            (output / "last.tmp").replace(output / "last.pt")
            if improved:
                torch.save(checkpoint, output / "best.tmp")
                (output / "best.tmp").replace(output / "best.pt")
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
    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = None
    if not args.skip_test:
        test_metrics = run_epoch(model, loaders["test"], device, args.threshold, amp,
                                 max_batches=max_batches, loss_options=loss_options)
        save_test_predictions(model, loaders["test"], splits["test"], output, device, args.threshold, amp, max_batches)
    write_json(output / "result.json", {
        "status": "smoke_test_completed" if args.smoke_test else "completed",
        "best_epoch": best_epoch, "best_val_dice": best_dice, "epochs_completed": epoch,
        "best_val": best["val_metrics"], "test": test_metrics,
        "test_is_partial": args.smoke_test and not args.skip_test,
        "test_status": "skipped" if args.skip_test else "partial" if args.smoke_test else "completed",
        "seconds": time.perf_counter() - started,
        "gpu_peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
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
            or args.base_channels < 8 or args.base_channels % 8):
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
