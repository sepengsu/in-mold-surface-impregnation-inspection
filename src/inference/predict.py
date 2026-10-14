"""Predict Exposure tiles and save comparison PNGs plus a local HTML gallery."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

from src.training.data import load_tiles, read_gray, relative_file
from src.training.model import UNet
from .visualize import comparison, save_png, write_gallery, write_overview

PROJECT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def latest_checkpoint(training_root):
    """Use a completed single run or the validation-selected comparison winner."""
    eligible = []
    for run in Path(training_root).glob("unet_*"):
        result, config, checkpoint = run / "result.json", run / "config.json", run / "best.pt"
        if not all(p.is_file() for p in (result, config, checkpoint)):
            continue
        metadata, settings = read_json(result), read_json(config)
        if metadata.get("status") == "completed" and not settings.get("smoke_test", False):
            eligible.append((result.stat().st_mtime_ns, checkpoint))
    for run in Path(training_root).glob("comparison_*"):
        result, selection = run / "comparison.json", run / "selected_model.json"
        if not result.is_file() or not selection.is_file() or (run / "RUNNING.json").exists():
            continue
        metadata, selected = read_json(result), read_json(selection)
        if metadata.get("status") != "completed" or metadata.get("smoke_test") or selected.get("smoke_test"):
            continue
        checkpoint = Path(selected["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(run.resolve()) or not checkpoint.is_file():
            raise ValueError(f"Invalid selected checkpoint: {selection}")
        if read_json(checkpoint.parent / "result.json")["status"] != "completed":
            continue
        eligible.append((result.stat().st_mtime_ns, checkpoint))
    if not eligible:
        raise ValueError("No completed training/comparison run found. Wait for training or specify --checkpoint / -Checkpoint.")
    return max(eligible, key=lambda item: (item[0], str(item[1])))[1].resolve()


def load_checkpoint(path, device):
    # This loader accepts the architecture and preprocessing saved by our trainer.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = {"label": "Exposure", "mask_rule": "alpha > 0",
                "normalization": "uint8 / 255"}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Unsupported checkpoint {key}: {checkpoint.get(key)!r}")
    if checkpoint.get('architecture') == 'grayscale_segmentation_benchmark':
        from src.training.architectures import build_model
        model = build_model(checkpoint['model_spec'])
    elif checkpoint.get('architecture') == 'grayscale_unet':
        model = UNet(checkpoint['base_channels'])
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint.get('architecture')!r}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # Release optimizer buffers: inference does not need them on CPU or GPU.
    metadata = {key: checkpoint[key] for key in ("epoch", "threshold", "base_channels", "smoke_test")}
    if 'model_spec' in checkpoint:
        metadata['model_spec'] = checkpoint['model_spec']
    del checkpoint
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    metadata["sha256"] = digest.hexdigest()
    model.to(device).eval()
    return model, metadata


def select_records(checkpoint, data=None, split="test", position="all", images=None, limit=0, central_three=False):
    if images is not None:
        if central_three:
            raise ValueError('Central-point selection requires a labeled ROI tiling manifest.')
        source = Path(images).resolve()
        if source.is_file():
            paths, root = [source], source.parent
        elif source.is_dir():
            paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".png")
            root = source
        else:
            raise ValueError(f"Image path does not exist: {source}")
        if not paths:
            raise ValueError("No PNG tiles found in --images.")
        if position != "all":
            raise ValueError("Direction filtering requires a tiling manifest, not --images.")
        records = [{"image": p.relative_to(root).as_posix(), "mask": None,
                    "image_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "case": "", "capture": "", "position": "", "task_id": None, "xyxy": None}
                   for p in paths]
        total = len(records)
        records = records[:limit] if limit else records
        for row in records:
            if Path(row["image"]).suffix.lower() != ".png":
                raise ValueError("Unlabeled inputs must be PNG tiles.")
            if read_gray(root / row["image"]).shape != (256, 256):
                raise ValueError(f"Expected a 256 x 256 tile (no automatic resizing): {row['image']}")
        return root, records, {"split": "unlabeled", "available_tiles": total, "manifest_sha256": None}

    config_path = checkpoint.parent / "config.json"
    if data is None:
        if not config_path.is_file():
            raise ValueError("Specify --data when config.json is not next to the checkpoint.")
        data = Path(read_json(config_path)["data"])
    root = Path(data).resolve()
    records, digest = load_tiles(root, position)
    if split != "all":
        path = checkpoint.parent / "splits.json"
        if not path.is_file():
            raise ValueError("splits.json is required for train/val/test selection. Use --split all for a new dataset.")
        saved = read_json(path)
        if digest != saved["manifest_sha256"]:
            raise ValueError("Dataset manifest differs from the training split. Use --split all for a new dataset.")
        selected = saved["splits"][split]
        if position != "all":
            selected = [r for r in selected if r["position"] == position]
        available = {r["image"]: r for r in records}
        for saved_row in selected:
            current = available.get(saved_row["image"])
            if current is None or any(current[key] != saved_row[key] for key in ("image_sha256", "mask_sha256", "mask")):
                raise ValueError(f"Tile changed since training: {saved_row['image']}")
        records = [available[row["image"]] for row in selected]
    if central_three:
        records = [row for row in records if row.get('measurement_point') in ('P1','P2','P3')]
    if not records:
        raise ValueError("No tiles match the selected split/direction.")
    records.sort(key=lambda row: row["image"])
    total = len(records)
    return root, records[:limit] if limit else records, {"split": split, "available_tiles": total, "manifest_sha256": digest}


def pixel_metrics(predicted, actual):
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    denominator, union = 2 * tp + fp + fn, tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "dice": 2 * tp / denominator if denominator else 1.0,
            "iou": tp / union if union else 1.0}


def summarize_metrics(rows):
    labeled = [row for row in rows if row["true_exposure_pixels"] is not None]
    if not labeled:
        return None
    tp, fp, fn = (sum(row[key] for row in labeled) for key in ("tp", "fp", "fn"))
    denominator, union = 2 * tp + fp + fn, tp + fp + fn
    true_rates = [100 * (1-row['true_exposure_pixels']/65536) for row in labeled]
    zero_truth = sum(rate == 0 for rate in true_rates)
    mape = (None if zero_truth else sum(100*row['impregnation_abs_error_pp']/rate
            for row,rate in zip(labeled,true_rates))/len(labeled))
    return {"labeled_tiles": len(labeled), "dice": 2 * tp / denominator if denominator else 1.0,
            "iou": tp / union if union else 1.0,
            "impregnation_mae_pp": sum(row["impregnation_abs_error_pp"] for row in labeled) / len(labeled),
            "impregnation_mape_pct": mape, "mape_zero_truth_tiles": zero_truth,
            "tp": tp, "fp": fp, "fn": fn}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, help="Defaults to the latest completed training run or selected comparison winner.")
    source = result.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, help="Paired tile dataset with manifest.json; defaults to training config.")
    source.add_argument("--images", type=Path, help="Unlabeled 256 x 256 grayscale PNG or directory.")
    result.add_argument("--output", type=Path)
    result.add_argument("--split", choices=("test", "val", "train", "all"), default="test")
    result.add_argument("--position", choices=("all", "000", "180"), default="all")
    result.add_argument('--central-three',action='store_true',help='Only manifest-marked ROI center P1/P2/P3 tiles.')
    result.add_argument("--threshold", type=float, help="Defaults to the value saved in the checkpoint.")
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    result.add_argument("--no-amp", action="store_true")
    result.add_argument("--limit", type=int, default=0, help="0 = all selected tiles; positive = first N tiles.")
    return result


def predict(args):
    if args.batch_size < 1 or args.limit < 0:
        raise ValueError("batch-size must be positive; limit must be non-negative.")
    checkpoint = (args.checkpoint or latest_checkpoint(PROJECT / "outputs/training")).resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or install the training dependencies.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    amp = device.type == "cuda" and not args.no_amp
    model, metadata = load_checkpoint(checkpoint, device)
    threshold = args.threshold if args.threshold is not None else metadata["threshold"]
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1.")
    print(f"Checkpoint: {checkpoint} | epoch {metadata['epoch']}", flush=True)
    print(f"Device: {device} | AMP: {amp} | threshold: {threshold}", flush=True)
    root, records, selection = select_records(checkpoint, args.data, args.split, args.position, args.images, args.limit,
                                               central_three=args.central_three)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or PROJECT / "outputs/inference" / f"{checkpoint.parent.name}_{selection['split']}_{stamp}").resolve()
    single_image = args.images is not None and Path(args.images).is_file()
    if output.is_relative_to(root) and not single_image:
        raise ValueError("Output must be outside the input image directory.")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("comparisons", "pred_masks", "probabilities"):
        (output / folder).mkdir()
    write_json(output / "RUNNING.json", {"checkpoint": str(checkpoint), "selected_tiles": len(records)})
    started = time.perf_counter()
    rows = []
    checkpoint_digest = metadata["sha256"]
    try:
        with torch.inference_mode():
            for start in range(0, len(records), args.batch_size):
                batch = records[start:start + args.batch_size]
                images = [read_gray(relative_file(root, row["image"])) for row in batch]
                tensors = np.ascontiguousarray(np.stack(images)[:, None], dtype=np.float32) / 255.0
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(torch.from_numpy(tensors).to(device))
                probabilities = logits.float().sigmoid().cpu().numpy()[:, 0]
                if not np.isfinite(probabilities).all():
                    raise RuntimeError("Non-finite model probabilities.")
                for offset, (source, image, probability) in enumerate(zip(batch, images, probabilities)):
                    predicted = probability >= threshold
                    actual = read_gray(relative_file(root, source["mask"])) > 0 if source["mask"] else None
                    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source["image"]).stem).strip("._")[:96] or "tile"
                    tile_id = f"{start + offset + 1:06d}_{safe_stem}"
                    prediction_count = int(np.count_nonzero(predicted))
                    true_count = int(np.count_nonzero(actual)) if actual is not None else None
                    pred_rate = 100 * (1 - prediction_count / 65536)
                    true_rate = 100 * (1 - true_count / 65536) if true_count is not None else None
                    row = {
                        **source, "id": tile_id, "threshold": threshold,
                        "pred_exposure_pixels": prediction_count, "true_exposure_pixels": true_count,
                        "pred_impregnation_pct": pred_rate, "true_impregnation_pct": true_rate,
                        "impregnation_abs_error_pp": abs(pred_rate - true_rate) if true_rate is not None else None,
                        **(pixel_metrics(predicted, actual) if actual is not None else
                           {"tp": None, "fp": None, "fn": None, "dice": None, "iou": None}),
                        "comparison": f"comparisons/{tile_id}.png", "pred_mask": f"pred_masks/{tile_id}.png",
                        "probability_png": f"probabilities/{tile_id}.png",
                    }
                    save_png(output / row["pred_mask"], predicted.astype(np.uint8) * 255)
                    # 16-bit grayscale stores probability = pixel / 65535, before thresholding.
                    save_png(output / row["probability_png"], np.round(probability * 65535).astype(np.uint16))
                    save_png(output / row["comparison"], comparison(image, probability, actual, row))
                    rows.append(row)
                print(f"Predicted {min(start + len(batch), len(records))}/{len(records)}", flush=True)
        overview_ids = write_overview(output, rows)
        fields = ("id", "image", "mask", "case", "capture", "position", "task_id", "threshold",
                  "pred_exposure_pixels", "true_exposure_pixels", "pred_impregnation_pct", "true_impregnation_pct",
                  "impregnation_abs_error_pp", "dice", "iou", "tp", "fp", "fn", "comparison", "pred_mask", "probability_png",
                  "measurement_point", "alignment", "roi_needs_review")
        with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "status": "completed", "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_digest,
            "checkpoint_epoch": metadata["epoch"], "smoke_test_checkpoint": metadata["smoke_test"],
            "data": str(root), "output": str(output), **selection, "position": args.position,
            "selected_tiles": len(rows), "limited": len(rows) < selection["available_tiles"],
            "central_three_only": args.central_three,
            "threshold": threshold, "label": "Exposure = unimpregnated", "mask_rule": "alpha > 0",
            "probability_png_encoding": "uint16 / 65535", "pred_mask_encoding": "0 or 255",
            "device": str(device), "torch": str(torch.__version__), "amp": amp,
            "metrics": summarize_metrics(rows), "overview_selection": "uniformly spaced in input order",
            "overview_ids": overview_ids, "seconds": time.perf_counter() - started,
        }
        write_gallery(output, rows, summary)
        write_json(output / "summary.json", summary)
        write_json(output / "predictions.json", rows)
        (output / "RUNNING.json").unlink()
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "summary.json", {"status": "failed", "error": str(error), "completed_tiles": len(rows)})
        raise
    print(json.dumps(summary["metrics"], ensure_ascii=True), flush=True)
    print(f"Gallery: {output / 'index.html'}", flush=True)
    return output, summary


def main(argv=None):
    predict(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
