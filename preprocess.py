"""Command-line entry point for fixed-ROI segmentation preprocessing."""

import argparse
from collections import Counter
import hashlib
import logging
from pathlib import Path

import cv2

from src.preprocess import load_config, preprocess_image
from src.preprocess.config import Config
from src.preprocess.pipeline import IMAGE_EXTENSIONS, PreprocessResult

LOGGER = logging.getLogger("preprocess")


def discover_images(source: Path, output_root: Path) -> list[Path]:
    """Find supported images recursively, excluding generated output directories."""
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input does not exist: {source}")
    excluded = [(output_root / name).resolve() for name in ("roi", "clahe", "tiles")]
    images = [path for path in source.rglob("*")
              if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
              and not any(path.resolve().is_relative_to(root) for root in excluded)]
    if not images:
        raise ValueError(f"No supported images found in: {source}")
    return sorted(images, key=lambda path: path.as_posix())


def image_ids(images: list[Path], source: Path) -> list[str]:
    """Disambiguate duplicate stems with stable relative-path hashes."""
    counts = Counter(path.stem.casefold() for path in images)
    identities = []
    for path in images:
        image_id = path.stem
        if counts[image_id.casefold()] > 1:
            relative = path.relative_to(source).as_posix()
            suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
            image_id = f"{image_id}__{suffix}"
        identities.append(image_id)
    if len({value.casefold() for value in identities}) != len(identities):
        raise ValueError("Output image IDs collide; rename the conflicting input images")
    return identities


def log_result(path: Path, result: PreprocessResult, config: Config) -> None:
    """Report dimensions as width x height and the actual tile count."""
    height, width = result["original_shape"]
    roi_height, roi_width = result["roi_shape"]
    clahe, tiling = config["clahe"], config["tiling"]
    grid_x, grid_y = clahe["tile_grid_size"]
    description = (f"clip={clahe['clip_limit']}, grid={grid_x}x{grid_y}"
                   if clahe["enabled"] else "disabled (raw ROI)")
    LOGGER.info(
        "%s\nOriginal : %d x %d\nROI      : %d x %d\nCLAHE    : %s\n"
        "Patch    : %d x %d\nStride   : %d\nTiles    : %d",
        path, width, height, roi_width, roi_height, description,
        tiling["patch_size"], tiling["patch_size"], tiling["stride"], len(result["patches"]),
    )


def main(argv: list[str] | None = None) -> int:
    """Process images independently; return nonzero if setup or any image fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Image or directory (recursive)")
    parser.add_argument("--config", type=Path, default=Path("config/preprocess.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        config = load_config(args.config)
        source = args.input.expanduser().resolve()
        images = discover_images(source, config["output"]["root_dir"])
        identities = image_ids(images, source)
    except (OSError, ValueError, TypeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    processed = failed = total_tiles = 0
    for path, image_id in zip(images, identities, strict=True):
        try:
            result = preprocess_image(path, config, image_id=image_id)
            log_result(path, result, config)
            processed += 1
            total_tiles += len(result["patches"])
            del result  # Keep batch memory bounded to one image's results.
        except (OSError, ValueError, TypeError, cv2.error) as exc:
            LOGGER.error("%s: %s", path, exc)
            failed += 1
    LOGGER.info("Processed images: %d\nFailed images: %d\nTotal tiles: %d",
                processed, failed, total_tiles)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
