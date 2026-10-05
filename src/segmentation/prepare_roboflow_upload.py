"""Select source images and prepare a Roboflow upload archive."""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".heic", ".heif"}


def slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-z0-9_.-]+", "-", value)
    return value.strip("-")


def iter_images(
    source: Path,
    prefix: str,
    exposure: str | None,
    extensions: set[str],
    exclude_prefixes: list[str],
):
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if prefix and not path.name.startswith(prefix):
            continue
        if any(path.name.startswith(exclude_prefix) for exclude_prefix in exclude_prefixes):
            continue
        if exposure and path.parent.name != exposure:
            continue
        yield path


def build_output_name(source: Path, image_path: Path) -> str:
    relative_parent = image_path.parent.relative_to(source)
    parent_slug = slug("_".join(relative_parent.parts))
    file_slug = slug(image_path.name)
    if parent_slug:
        return f"{parent_slug}_{file_slug}"
    return file_slug


def load_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise SystemExit(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    return config.get("roboflow_upload", config)


def create_zip(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in sorted(source_dir.iterdir()):
            if image_path.is_file():
                archive.write(image_path, arcname=image_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy only the images that should be uploaded to Roboflow."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--exposure", help="Use an empty string to include every exposure.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    select_config = config.get("select", {})
    exclude_config = config.get("exclude", {})

    source = args.source or Path(config.get("source", "data/0915_data"))
    output = args.output or Path(config.get("output", "data/roboflow_upload/0915_mono_10000us"))
    zip_path = args.zip_path or (Path(config["zip"]) if config.get("zip") else None)
    prefix = args.prefix if args.prefix is not None else select_config.get("filename_prefix", "mono_")
    exposure = args.exposure if args.exposure is not None else select_config.get("exposure", "10000us")
    exposure = exposure or None
    extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in select_config.get("extensions", IMAGE_EXTENSIONS)
    }
    exclude_prefixes = list(exclude_config.get("filename_prefixes", []))

    if not source.exists():
        raise SystemExit(f"Source folder does not exist: {source}")

    images = sorted(iter_images(source, prefix, exposure, extensions, exclude_prefixes))
    if not images:
        raise SystemExit("No matching images found.")

    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in images:
        output_name = build_output_name(source, image_path)
        destination = output / output_name
        if not args.dry_run:
            shutil.copy2(image_path, destination)
        copied += 1

    print(f"matched={len(images)}")
    print(f"copied={copied}")
    print(f"output={output}")

    if zip_path:
        if not args.dry_run:
            create_zip(output, zip_path)
        print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
