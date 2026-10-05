"""Describe saved inference errors by case and label morphology, without tuning."""

from pathlib import Path

import numpy as np

from .data import read_gray, relative_file
from .sampling import largest_region_pixels


def summarize(rows):
    if not rows:
        return {"tiles": 0, "dice": None, "iou": None, "precision": None, "recall": None,
                "impregnation_mae_pp": None, "impregnation_p95_pp": None,
                "impregnation_max_pp": None, "above_5pp_fraction": None}
    tp, fp, fn = (sum(r[key] for r in rows) for key in ("tp", "fp", "fn"))
    errors = np.array([r["impregnation_abs_error_pp"] for r in rows], dtype=np.float64)
    return {
        "tiles": len(rows), "dice": 2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.0,
        "iou": tp/(tp+fp+fn) if tp+fp+fn else 1.0,
        "precision": tp/(tp+fp) if tp+fp else 1.0, "recall": tp/(tp+fn) if tp+fn else 1.0,
        "impregnation_mae_pp": float(errors.mean()), "impregnation_p95_pp": float(np.quantile(errors, .95)),
        "impregnation_max_pp": float(errors.max()), "above_5pp_fraction": float((errors > 5).mean()),
    }


def grouped_metrics(root, rows, min_fraction=0.05):
    """Large-region metrics cover entire qualifying tiles, not just components."""
    root = Path(root).resolve()
    large = []
    for row in rows:
        alpha = read_gray(relative_file(root, row["mask"]))
        if largest_region_pixels(alpha) / alpha.size >= min_fraction:
            large.append(row)
    cases = {group: summarize([r for r in rows if r["group"] == group])
             for group in sorted({r["group"] for r in rows})}
    return {
        "all": summarize(rows), "cases": cases,
        "case_macro_dice": float(np.mean([r["dice"] for r in cases.values()])) if cases else None,
        "case_macro_mae_pp": float(np.mean([r["impregnation_mae_pp"] for r in cases.values()])) if cases else None,
        "large_region_rule": {"min_fraction": min_fraction, "connectivity": 8, "mask_rule": "alpha > 0",
                              "metric_scope": "all pixels of qualifying tiles"},
        "large_region_tiles": summarize(large),
    }
