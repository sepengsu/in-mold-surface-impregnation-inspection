"""Training-only sampling weights based on connected Exposure regions."""

from collections import defaultdict
import math
from pathlib import Path

import cv2
import numpy as np

from .data import read_gray, relative_file

SAMPLING_MODES = ("uniform", "large_region")


def largest_region_pixels(alpha):
    binary = (alpha > 0).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0


def validate_sampling(mode, min_fraction, weight):
    if mode not in SAMPLING_MODES:
        raise ValueError(f"Unknown sampling mode: {mode}")
    if not math.isfinite(min_fraction) or not 0 < min_fraction <= 1:
        raise ValueError("Large-region fraction must be in (0, 1].")
    if not math.isfinite(weight) or weight < 1:
        raise ValueError("Large-region weight must be finite and at least 1.")


def build_sampling_plan(root, train_rows, mode="uniform", min_fraction=0.05, weight=3.0):
    """Only the supplied training rows determine weights; never scan other splits."""
    validate_sampling(mode, min_fraction, weight)
    if not train_rows:
        raise ValueError("Cannot sample an empty training split.")
    root = Path(root).resolve()
    details = []
    for row in train_rows:
        alpha = read_gray(relative_file(root, row["mask"]))
        largest = largest_region_pixels(alpha)
        eligible = largest / alpha.size >= min_fraction
        details.append({
            "image": row["image"], "group": row["group"], "case": row["case"],
            "largest_region_pixels": largest, "large_region": eligible,
            "weight": float(weight if mode == "large_region" and eligible else 1),
        })
    weights = np.array([r["weight"] for r in details], dtype=np.float64)
    total_weight = float(weights.sum())
    cases = defaultdict(lambda: {"tiles": 0, "large_tiles": 0, "weight_sum": 0.0})
    for row in details:
        case = cases[row["group"]]
        case["tiles"] += 1
        case["large_tiles"] += int(row["large_region"])
        case["weight_sum"] += row["weight"]
    for case in cases.values():
        case["expected_draw_fraction"] = case["weight_sum"] / total_weight
    summary = {
        "mode": mode, "split": "train only", "connectivity": 8, "mask_rule": "alpha > 0",
        "min_fraction": min_fraction, "large_region_weight": weight,
        "tiles": len(details), "large_tiles": sum(r["large_region"] for r in details),
        "samples_per_epoch": len(details), "replacement": mode == "large_region",
        "expected_large_fraction": sum(r["weight"] for r in details if r["large_region"]) / total_weight,
        "cases": dict(sorted(cases.items())),
    }
    return weights, {"summary": summary, "tiles": details}


def make_sampler(weights, mode, seed):
    """Uniform keeps the original DataLoader shuffle; weighted RNG is independent."""
    if mode not in SAMPLING_MODES:
        raise ValueError(f"Unknown sampling mode: {mode}")
    if mode == "uniform":
        return None
    import torch
    from torch.utils.data import WeightedRandomSampler

    generator = torch.Generator().manual_seed(seed + 200003)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
