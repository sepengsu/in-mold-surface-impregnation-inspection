"""Validate and optionally upload a prepared YOLO dataset to Roboflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("roboflow_api_upload", data)


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def count_dataset(dataset: Path) -> dict:
    counts: dict[str, dict[str, int]] = {}
    missing_labels: list[str] = []
    for split in ("train", "valid", "test"):
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        images = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS] if image_dir.exists() else []
        labels = [p for p in label_dir.rglob("*.txt")
                  if p.name.casefold() not in {"classes.txt", "labelmap.txt", "_darknet.labels"}] if label_dir.exists() else []
        label_stems = {p.relative_to(label_dir).with_suffix("").as_posix().casefold() for p in labels}
        for image in images:
            key = image.relative_to(image_dir).with_suffix("").as_posix().casefold()
            if key not in label_stems:
                missing_labels.append((image.relative_to(dataset)).as_posix())
        counts[split] = {"images": len(images), "labels": len(labels)}
    return {"counts": counts, "missing_labels": missing_labels}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Upload a prepared YOLO dataset to Roboflow from YAML config.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--workspace")
    parser.add_argument("--project")
    parser.add_argument("--batch-name")
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--api-key-env", default="ROBOFLOW_API_KEY")
    parser.add_argument("--zip-upload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print what would upload.")
    args = parser.parse_args(argv)

    project_root = Path.cwd()
    config = load_config(args.config)
    dataset_value = args.dataset or config.get("dataset")
    if not dataset_value:
        raise SystemExit("dataset is required via --dataset or config")
    dataset = resolve_path(project_root, dataset_value).resolve()
    workspace = args.workspace if args.workspace is not None else config.get("workspace")
    project = args.project if args.project is not None else config.get("project")
    batch_name = args.batch_name if args.batch_name is not None else config.get("batch_name")
    api_key_env = args.api_key_env or config.get("api_key_env", "ROBOFLOW_API_KEY")
    api_key = os.environ.get(api_key_env) or config.get("api_key")
    zip_upload = args.zip_upload if args.zip_upload is not None else bool(config.get("zip_upload", False))

    if not dataset.exists():
        raise SystemExit(f"dataset folder not found: {dataset}")

    summary = count_dataset(dataset)
    total_images = sum(item["images"] for item in summary["counts"].values())
    total_labels = sum(item["labels"] for item in summary["counts"].values())
    if summary["missing_labels"]:
        raise SystemExit("missing labels:\n" + "\n".join(summary["missing_labels"][:50]))
    if total_images == 0:
        raise SystemExit(f"no images found in {dataset}")
    if total_images != total_labels:
        raise SystemExit(f"image/label count mismatch: images={total_images}, labels={total_labels}")
    if args.expected_count and total_images != args.expected_count:
        raise SystemExit(f"expected {args.expected_count} images, found {total_images}")

    print(f"dataset={dataset}")
    print(f"workspace={workspace}")
    print(f"project={project}")
    print(f"batch_name={batch_name}")
    print(f"zip_upload={zip_upload}")
    print(f"api_key_env={api_key_env}")
    print(f"counts={summary['counts']}")

    if args.dry_run:
        print("dry_run=1")
        return

    if not api_key or str(api_key).startswith("YOUR_"):
        raise SystemExit(f"set the {api_key_env} environment variable")
    if not project or project.startswith("YOUR_"):
        raise SystemExit("set --project or fill project in the config")

    import roboflow

    rf = roboflow.Roboflow(api_key=api_key)
    ws = rf.workspace(workspace) if workspace else rf.workspace()
    result = ws.upload_dataset(
        dataset_path=str(dataset),
        project_name=project,
        num_workers=int(config.get("concurrency", 4)),
        project_license=config.get("project_license", "Private"),
        project_type=config.get("project_type", "instance-segmentation"),
        batch_name=batch_name,
        num_retries=int(config.get("retries", 0)),
        is_prediction=bool(config.get("is_prediction", False)),
        use_zip_upload=zip_upload,
        annotation_overwrite=config.get("annotation_overwrite"),
        split=config.get("split"),
        tags=config.get("tags"),
        wait=bool(config.get("wait", True)),
    )
    print(f"upload_result={result}")


if __name__ == "__main__":
    main()
