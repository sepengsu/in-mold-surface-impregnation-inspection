"""Prepare an existing YOLO segmentation dataset for Roboflow upload."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SKIP_LABEL_FILES = {"classes.txt", "labelmap.txt"}


def slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-z0-9_.-]+", "-", value)
    return value.strip("-") or "item"


def parse_ratio(value: str) -> tuple[float, float, float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ratio must be train,valid,test")
    if any(part < 0 for part in parts) or sum(parts) <= 0:
        raise argparse.ArgumentTypeError("ratio values must be non-negative and not all zero")
    total = sum(parts)
    return tuple(part / total for part in parts)


def split_for_group(group: str, ratio: tuple[float, float, float]) -> str:
    digest = hashlib.sha256(group.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    train, valid, _test = ratio
    if value < train:
        return "train"
    if value < train + valid:
        return "valid"
    return "test"


def group_key(relative_label: Path, levels: int) -> str:
    if levels <= 0:
        return relative_label.as_posix()
    parts = relative_label.with_suffix("").parts
    return "/".join(parts[: min(levels, len(parts))])


def find_image(source_root: Path, relative_label: Path, extensions: set[str]) -> Path | None:
    stem = relative_label.with_suffix("")
    for extension in sorted(extensions):
        candidate = source_root / stem.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def flat_name(relative_path: Path) -> str:
    return slug("_".join(relative_path.parts))


def read_label_rows(label_file: Path) -> list[str]:
    rows = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row]


def valid_yolo_segmentation(label_file: Path, allow_empty: bool) -> bool:
    rows = read_label_rows(label_file)
    if not rows:
        return allow_empty
    for row in rows:
        parts = row.split()
        if len(parts) < 7 or len(parts[1:]) % 2:
            return False
        int(parts[0])
        coords = [float(part) for part in parts[1:]]
        if any(coord < 0 or coord > 1 for coord in coords):
            return False
    return True


def iter_label_pairs(
    source_root: Path,
    label_roots: list[Path],
    extensions: set[str],
    allow_empty: bool,
) -> tuple[list[tuple[Path, Path, Path]], list[str]]:
    pairs: list[tuple[Path, Path, Path]] = []
    warnings: list[str] = []
    seen: set[Path] = set()
    for label_root in label_roots:
        for label_file in sorted(label_root.rglob("*.txt")):
            if label_file.name.lower() in SKIP_LABEL_FILES:
                continue
            relative_label = label_file.relative_to(label_root)
            if relative_label in seen:
                warnings.append(f"duplicate label skipped: {relative_label.as_posix()}")
                continue
            seen.add(relative_label)
            image_file = find_image(source_root, relative_label, extensions)
            if image_file is None:
                warnings.append(f"missing image for label: {relative_label.as_posix()}")
                continue
            try:
                ok = valid_yolo_segmentation(label_file, allow_empty)
            except Exception as exc:
                warnings.append(f"invalid label skipped: {relative_label.as_posix()} ({exc})")
                continue
            if not ok:
                warnings.append(f"invalid label skipped: {relative_label.as_posix()}")
                continue
            pairs.append((image_file, label_file, relative_label))
    return pairs, warnings


def write_data_yaml(output: Path, class_names: list[str]) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    content = (
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names}\n"
    )
    (output / "data.yaml").write_text(content, encoding="utf-8")


def zip_dir(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path != destination:
                archive.write(path, path.relative_to(source).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Roboflow-ready YOLO segmentation zip from original images and label folders."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Original image root.")
    parser.add_argument(
        "--label-root",
        type=Path,
        action="append",
        required=True,
        help="YOLO label root. Repeat this option to include draft and roi_review labels.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output dataset folder.")
    parser.add_argument("--zip", dest="zip_path", type=Path, help="Optional zip output path.")
    parser.add_argument("--class-name", default="Exposure")
    parser.add_argument("--ratio", type=parse_ratio, default=parse_ratio("0.8,0.2,0"))
    parser.add_argument(
        "--group-levels",
        type=int,
        default=1,
        help="Keep labels with the same first N relative folders in the same split. Use 1 for case-level split.",
    )
    parser.add_argument("--allow-empty", action="store_true", help="Keep empty label files as negative samples.")
    parser.add_argument("--seed", type=int, default=0, help="Only affects output ordering for equal paths.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    label_roots = [path.resolve() for path in args.label_root]
    output = args.output.resolve()
    zip_path = args.zip_path.resolve() if args.zip_path else None

    if not source_root.is_dir():
        raise SystemExit(f"source root does not exist: {source_root}")
    for label_root in label_roots:
        if not label_root.is_dir():
            raise SystemExit(f"label root does not exist: {label_root}")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists. Use --overwrite or choose a new folder: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    pairs, warnings = iter_label_pairs(source_root, label_roots, IMAGE_EXTENSIONS, args.allow_empty)
    if not pairs:
        raise SystemExit("no image/label pairs found")

    random.Random(args.seed).shuffle(pairs)
    counts = {"train": 0, "valid": 0, "test": 0}
    used_names: set[str] = set()
    manifest_rows = []

    for image_file, label_file, relative_label in pairs:
        split = split_for_group(group_key(relative_label, args.group_levels), args.ratio)
        image_name = flat_name(relative_label.with_suffix(image_file.suffix.lower()))
        label_name = flat_name(relative_label)
        if image_name in used_names or label_name in used_names:
            raise SystemExit(f"filename collision after flattening: {relative_label.as_posix()}")
        used_names.add(image_name)
        used_names.add(label_name)

        image_out = output / split / "images" / image_name
        label_out = output / split / "labels" / label_name
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_file, image_out)
        shutil.copy2(label_file, label_out)
        counts[split] += 1
        manifest_rows.append(
            f"{split},{image_file.relative_to(source_root).as_posix()},{label_file},{image_out.relative_to(output).as_posix()},{label_out.relative_to(output).as_posix()}"
        )

    write_data_yaml(output, [args.class_name])
    (output / "classes.txt").write_text(args.class_name + "\n", encoding="utf-8")
    (output / "manifest.csv").write_text(
        "split,source_image,source_label,output_image,output_label\n" + "\n".join(manifest_rows) + "\n",
        encoding="utf-8-sig",
    )
    if warnings:
        (output / "warnings.txt").write_text("\n".join(warnings) + "\n", encoding="utf-8")
    if zip_path:
        zip_dir(output, zip_path)

    print(f"pairs={len(pairs)}")
    print(f"counts={counts}")
    print(f"output={output}")
    if zip_path:
        print(f"zip={zip_path}")
    if warnings:
        print(f"warnings={len(warnings)}")


if __name__ == "__main__":
    main()
