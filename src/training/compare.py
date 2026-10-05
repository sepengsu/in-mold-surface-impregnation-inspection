"""Run controlled training ablations and test the validation-selected model."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .train import DEFAULT_DATA, PROJECT, write_json
from .diagnostics import grouped_metrics
from .losses import describe_loss
from .sampling import validate_sampling

EXPERIMENTS = {
    "A": {"augmentation": "basic", "base_channels": 32},
    "B": {"augmentation": "strong", "base_channels": 32},
    "C": {"augmentation": "basic", "base_channels": 48},
    "D": {"augmentation": "strong", "base_channels": 48},
}

IMPROVEMENT_EXPERIMENTS = {
    "A": {"name": "baseline", "sampling": "uniform", "loss": "bce_dice"},
    "B": {"name": "sampling", "sampling": "large_region", "loss": "bce_dice"},
    "C": {"name": "tversky", "sampling": "uniform", "loss": "bce_tversky"},
    "D": {"name": "combined", "sampling": "large_region", "loss": "bce_tversky"},
}


def experiment_settings(suite):
    if suite == "augmentation":
        return EXPERIMENTS
    if suite == "improvement":
        return {key: {"augmentation": "basic", "base_channels": 32, **settings}
                for key, settings in IMPROVEMENT_EXPERIMENTS.items()}
    raise ValueError(f"Unknown experiment suite: {suite}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_winner(rows):
    """Test scores never enter selection; MAE breaks exact validation Dice ties."""
    if not rows:
        raise ValueError("No completed experiments to select.")
    return min(rows, key=lambda row: (-row["val_dice"], row["val_impregnation_mae_pp"], row["experiment"]))


def invoke(command, log_path):
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=environment)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f"Command exited with {code}; see {log_path}")


def collect_run(run, experiment, expected_split_hash=None, smoke_test=False):
    config, result = read_json(run / "config.json"), read_json(run / "result.json")
    status = "smoke_test_completed" if smoke_test else "completed"
    if result["status"] != status or result.get("test_status") != "skipped":
        raise ValueError(f"Expected completed validation-only training: {run}")
    split_hash = hashlib.sha256((run / "splits.json").read_bytes()).hexdigest()
    if expected_split_hash is not None and split_hash != expected_split_hash:
        raise ValueError("Experiment splits differ; refusing to compare.")
    metrics = result["best_val"]
    return {
        "experiment": experiment, "augmentation": config["augmentation"], "base_channels": config["base_channels"],
        "loss_mode": config.get("loss_mode", "bce_dice"), "sampling": config.get("sampling", "uniform"),
        "seed": config.get("seed"), "split_seed": config.get("split_seed", config.get("seed")),
        "parameters": config["parameters"], "best_epoch": result["best_epoch"],
        "epochs_completed": result["epochs_completed"], "val_dice": metrics["dice"], "val_iou": metrics["iou"],
        "val_impregnation_mae_pp": metrics["impregnation_mae_pp"], "val_samples": metrics["samples"],
        "val_precision": metrics.get("precision"), "val_recall": metrics.get("recall"),
        "gpu_peak_memory_mib": result["gpu_peak_memory_mib"], "seconds": result["seconds"],
        "run": str(run), "checkpoint": str(run / "best.pt"), "split_sha256": split_hash,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=DEFAULT_DATA)
    result.add_argument("--output", type=Path)
    result.add_argument("--suite", choices=("augmentation", "improvement"), default="augmentation")
    result.add_argument("--experiments", choices=("all", "bd"), default="all")
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--patience", type=int, default=20)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--split-seed", type=int)
    result.add_argument("--large-region-fraction", type=float, default=0.05)
    result.add_argument("--large-region-weight", type=float, default=3.0)
    result.add_argument("--tversky-alpha", type=float, default=0.3)
    result.add_argument("--tversky-beta", type=float, default=0.7)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--position", choices=("all", "000", "180"), default="all")
    result.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    result.add_argument("--no-amp", action="store_true")
    result.add_argument("--smoke-test", action="store_true")
    return result


def infer(args, checkpoint, output, split, log_path):
    command = [sys.executable, "-u", "-m", "src.inference.predict", "--checkpoint", str(checkpoint),
               "--data", str(args.data), "--output", str(output), "--split", split,
               "--batch-size", str(args.batch_size), "--device", args.device]
    if args.no_amp:
        command.append("--no-amp")
    if args.smoke_test:
        command.extend(["--limit", str(args.batch_size * 2)])
    invoke(command, log_path)
    return read_json(output / "summary.json")


def main(argv=None):
    args = parser().parse_args(argv)
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    describe_loss("bce_tversky", args.tversky_alpha, args.tversky_beta)
    validate_sampling("large_region", args.large_region_fraction, args.large_region_weight)
    if args.suite == "improvement" and args.experiments != "all":
        raise ValueError("The improvement suite requires --experiments all for its four-way comparison.")
    experiments = experiment_settings(args.suite)
    args.data = args.data.resolve()
    if not (args.data / "manifest.json").is_file():
        raise ValueError(f"Tile manifest not found: {args.data}")
    prefix = "comparison_improvement" if args.suite == "improvement" else "comparison"
    output = (args.output or PROJECT / "outputs/training" / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    if output.is_relative_to(args.data):
        raise ValueError("Comparison output must be outside the dataset.")
    output.mkdir(parents=True, exist_ok=False)
    selected_ids = list(experiments) if args.experiments == "all" else ["B", "D"]
    selection_rule = "highest validation Dice; lowest validation impregnation MAE on exact ties"
    write_json(output / "RUNNING.json", {"suite": args.suite, "experiments": selected_ids, "selection_rule": selection_rule})
    rows = []
    try:
        for experiment in selected_ids:
            settings = experiments[experiment]
            suffix = settings.get("name", f"{settings['augmentation']}_{settings['base_channels']}")
            run = output / f"{experiment}_{suffix}"
            command = [sys.executable, "-u", "-m", "src.training.train", "--data", str(args.data), "--output", str(run),
                       "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                       "--learning-rate", str(args.learning_rate), "--patience", str(args.patience),
                       "--seed", str(args.seed), "--split-seed", str(args.split_seed),
                       "--workers", str(args.workers), "--position", args.position,
                       "--device", args.device, "--augmentation", settings["augmentation"],
                       "--base-channels", str(settings["base_channels"]), "--skip-test"]
            if args.suite == "improvement":
                command.extend(["--loss", settings["loss"], "--sampling", settings["sampling"],
                                "--tversky-alpha", str(args.tversky_alpha), "--tversky-beta", str(args.tversky_beta),
                                "--large-region-fraction", str(args.large_region_fraction),
                                "--large-region-weight", str(args.large_region_weight)])
            if args.smoke_test:
                command.append("--smoke-test")
            if args.no_amp:
                command.append("--no-amp")
            print(f"\nExperiment {experiment}: {settings}", flush=True)
            invoke(command, output / f"{experiment}_train.log")
            rows.append(collect_run(run, experiment, rows[0]["split_sha256"] if rows else None, args.smoke_test))
            if args.suite == "improvement":
                val_output = run / "val_inference"
                infer(args, run / "best.pt", val_output, "val", output / f"{experiment}_val_inference.log")
                diagnostics = grouped_metrics(args.data, read_json(val_output / "predictions.json"), args.large_region_fraction)
                diagnostics["partial"] = args.smoke_test
                write_json(run / "validation_groups.json", diagnostics)
                rows[-1].update({
                    "val_large_tiles": diagnostics["large_region_tiles"]["tiles"],
                    "val_large_dice": diagnostics["large_region_tiles"]["dice"],
                    "val_large_recall": diagnostics["large_region_tiles"]["recall"],
                    "val_p95_error_pp": diagnostics["all"]["impregnation_p95_pp"],
                    "val_case_macro_mae_pp": diagnostics["case_macro_mae_pp"],
                })
            write_json(output / "progress.json", {"completed": rows, "remaining": selected_ids[len(rows):]})
        winner = select_winner(rows)
        with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["selected"] + list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({"selected": row is winner, **row})
        print(f"Validation-selected model: {winner['experiment']} | Dice {winner['val_dice']:.4f}", flush=True)
        inference_output = output / "test_inference"
        test_summary = infer(args, winner["checkpoint"], inference_output, "test", output / "test_inference.log")
        if args.suite == "improvement":
            diagnostics = grouped_metrics(args.data, read_json(inference_output / "predictions.json"), args.large_region_fraction)
            diagnostics["partial"] = test_summary["limited"]
            write_json(output / "test_groups.json", diagnostics)
        result = {
            "status": "smoke_test_completed" if args.smoke_test else "completed",
            "smoke_test": args.smoke_test, "selection_rule": selection_rule, "suite": args.suite,
            "experiments": rows, "selected_experiment": winner["experiment"],
            "checkpoint": winner["checkpoint"], "split_sha256": winner["split_sha256"],
            "test": test_summary["metrics"], "test_is_partial": test_summary["limited"],
            "gallery": str(inference_output / "index.html"),
        }
        write_json(output / "selected_model.json", {
            "experiment": winner["experiment"], "checkpoint": winner["checkpoint"], "selection_rule": selection_rule,
            "validation": {key: value for key, value in winner.items() if key.startswith("val_")},
            "smoke_test": args.smoke_test,
        })
        write_json(output / "comparison.json", result)
        (output / "RUNNING.json").unlink()
        print(f"Comparison: {output / 'comparison.csv'}\nGallery: {result['gallery']}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "comparison.json", {"status": "failed", "error": str(error), "experiments": rows})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
