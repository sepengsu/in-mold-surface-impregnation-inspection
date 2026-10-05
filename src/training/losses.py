"""Binary Exposure objectives; metadata is usable without importing PyTorch."""

import math

LOSS_MODES = ("bce_dice", "bce_tversky")


def describe_loss(mode="bce_dice", alpha=0.3, beta=0.7):
    if mode not in LOSS_MODES:
        raise ValueError(f"Unknown loss: {mode}")
    if (not math.isfinite(alpha) or not math.isfinite(beta) or alpha <= 0 or beta <= 0
            or not math.isclose(alpha + beta, 1.0, abs_tol=1e-6)):
        raise ValueError("Tversky alpha and beta must be positive, finite and sum to 1.")
    return {
        "mode": mode, "bce_weight": 0.5, "overlap_weight": 0.5,
        "overlap": "per-image soft Dice" if mode == "bce_dice" else "per-image soft Tversky",
        "alpha_fp": alpha if mode == "bce_tversky" else None,
        "beta_fn": beta if mode == "bce_tversky" else None,
        "smoothing": 1.0 if mode == "bce_dice" else 0.5,
        "reduction": "mean over images", "precision": "float32",
    }


def segmentation_loss(logits, targets, mode="bce_dice", alpha=0.3, beta=0.7):
    """Preserve the baseline loss exactly; Tversky alpha penalizes FP, beta FN.

    Smoothing 0.5 makes alpha=beta=0.5 equivalent to the original Dice with
    smoothing 1. All sums use float32, including when logits come from AMP.
    """
    from torch.nn import functional as F

    describe_loss(mode, alpha, beta)
    logits, targets = logits.float(), targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = logits.sigmoid()
    dims = (1, 2, 3)
    intersection = (probabilities * targets).sum(dims)
    if mode == "bce_dice":
        overlap = (2 * intersection + 1) / (probabilities.sum(dims) + targets.sum(dims) + 1)
    else:
        fp = (probabilities * (1 - targets)).sum(dims)
        fn = ((1 - probabilities) * targets).sum(dims)
        overlap = (intersection + 0.5) / (intersection + alpha * fp + beta * fn + 0.5)
    return 0.5 * bce + 0.5 * (1 - overlap.mean())
