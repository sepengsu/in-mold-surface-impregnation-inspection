"""Validate tiling manifests and split complete experimental cases."""

import hashlib
import json
from pathlib import Path
import random
import re

import cv2
import numpy as np


def read_gray(path):
    path = Path(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit grayscale PNG: {path}")
    return image


def relative_file(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing file or path outside dataset: {value}")
    return path


def load_tiles(dataset, position="all"):
    """Read and verify every pair; alpha > 0 is Exposure, including alpha=1."""
    if position not in {"all", "000", "180"}:
        raise ValueError("position must be all, 000, or 180")
    root = Path(dataset).resolve()
    if (root / "RUNNING.json").exists():
        raise ValueError("Tiling export is incomplete (RUNNING.json exists).")
    manifest_path = root / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest["summary"].get("count_only"):
        raise ValueError("A count-only manifest has no training PNGs.")
    records, seen_images, seen_masks = [], set(), set()
    for source in manifest["images"]:
        if position != "all" and source["position"] != position:
            continue
        # Keep fixed/moving sides, captures and 000/180 of one case together.
        match = re.fullmatch(r"case(\d+)(?:-.*)?", source["case"], re.IGNORECASE)
        if not match:
            raise ValueError(f"Unrecognized case name: {source['case']}")
        group = f"case{int(match.group(1)):02d}"
        for tile in source["tiles"]:
            image_path = relative_file(root, tile["image"])
            mask_path = relative_file(root, tile["mask"])
            if image_path in seen_images or mask_path in seen_masks:
                raise ValueError(f"Duplicate image/mask pair: {image_path}")
            seen_images.add(image_path)
            seen_masks.add(mask_path)
            image, alpha = read_gray(image_path), read_gray(mask_path)
            if image.shape != (256, 256) or alpha.shape != image.shape:
                raise ValueError(f"Expected aligned 256 x 256 tiles: {image_path}")
            encoding = manifest['summary'].get('mask_encoding', {})
            valid_value = tile.get('valid_mask')
            valid = (read_gray(relative_file(root, valid_value)) > 0 if valid_value else
                     np.ones(alpha.shape, dtype=bool))
            if valid.shape != alpha.shape:
                raise ValueError('Tile validity shape mismatch.')
            ignore = encoding.get('ignore_value')
            if ignore is not None:
                valid &= alpha != ignore
            if np.any((image == 0) & valid):
                raise ValueError(f"Black pixels found in interior tile: {image_path}")
            foreground_mask = (alpha == encoding['defect_value'] if encoding.get('defect_value') is not None
                               and encoding.get('rule') != 'positive_is_defect' else alpha > 0)
            foreground = int(np.count_nonzero(foreground_mask & valid))
            if foreground != tile["label_pixels"]:
                raise ValueError(f"Mask pixel count differs from manifest: {mask_path}")
            records.append({
                "image": tile["image"], "mask": tile["mask"],
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                "group": group, "case": source["case"],
                "capture": source["capture"], "position": source["position"],
                "task_id": source["task_id"], "xyxy": tile["xyxy_exclusive"],
                "label_pixels": foreground,
                **({'valid_mask': valid_value, 'valid_mask_sha256': hashlib.sha256(relative_file(root,valid_value).read_bytes()).hexdigest()}
                   if valid_value else {}),
                **({"measurement_point": tile.get("measurement_point"),
                    "grid_index": tile.get("grid_index"),
                    "alignment": manifest["summary"]["settings"]["alignment"],
                    "roi_needs_review": source["roi_geometry"]["needs_review"]}
                   if "roi_geometry" in source else {}),
            })
    if not records:
        raise ValueError("No image/mask pairs selected.")
    return records, hashlib.sha256(raw_manifest).hexdigest()


def split_by_case(records, seed=42, val_fraction=0.15, test_fraction=0.15):
    """Ratios apply to case counts; tile counts differ by ROI area."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to < 1.")
    groups = sorted({row["group"] for row in records})
    if len(groups) < 3:
        raise ValueError("At least three different cases are needed for train/val/test.")
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_fraction))
    n_test = max(1, round(len(groups) * test_fraction))
    if n_val + n_test >= len(groups):
        raise ValueError("Split ratios leave no training cases.")
    membership = {
        "val": set(groups[:n_val]),
        "test": set(groups[n_val:n_val + n_test]),
        "train": set(groups[n_val + n_test:]),
    }
    return {name: sorted((r for r in records if r["group"] in members),
                         key=lambda r: r["image"])
            for name, members in membership.items()}


def describe_split(splits):
    return {name: {
        "tiles": len(rows), "cases": sorted({r["group"] for r in rows}),
        "source_images": len({r["task_id"] for r in rows}),
        "exposure_fraction": sum(r["label_pixels"] for r in rows) / (len(rows) * 65536),
    } for name, rows in splits.items()}


class ExposureDataset:
    """PyTorch-compatible map dataset; import torch only when reading samples."""

    def __init__(self, root, records, augment=False, augmentation="basic", seed=42):
        from .augmentation import MODES
        if augmentation not in MODES:
            raise ValueError(f"Unknown augmentation mode: {augmentation}")
        self.root = Path(root)
        self.records = records
        self.augment = augment
        self.augmentation = augmentation if augment else "none"
        self.seed = seed
        self._generators = None

    def generators(self):
        import torch
        if self._generators is None:
            worker = torch.utils.data.get_worker_info()
            seed = worker.seed if worker is not None else self.seed
            # Independent streams keep geometry equal between basic/strong modes,
            # and independent of how much randomness model initialization consumes.
            self._generators = (
                torch.Generator().manual_seed(seed),
                torch.Generator().manual_seed((seed + 100003) % (2**63)),
            )
        return self._generators

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        from .augmentation import augment_pair

        row = self.records[index]
        image = read_gray(self.root / row["image"])
        mask = read_gray(self.root / row["mask"]) > 0
        generators = self.generators() if self.augmentation != "none" else (None, None)
        x, y = augment_pair(image, mask, self.augmentation, *generators)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None]), index
