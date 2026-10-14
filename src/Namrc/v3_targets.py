"""Versioned v3 targets: C is soft weighting, independent of legacy reviewed C."""

import numpy as np

from src.training.data import ExposureDataset, read_gray
from .disk import DiskArray
from .labels import OriginalLabels

ARMS = {'A': 'original', 'B': 'ignore', 'C': 'soft_weight', 'D': 'correct'}


def arm_target_weight(original, candidate, proposal, arm):
    if arm not in ARMS or original.shape != candidate.shape or original.shape != proposal.shape:
        raise ValueError('Aligned target/candidate/proposal arrays and a v3 arm are required.')
    if candidate.dtype != np.bool_ or not np.isin(original, [0, 1]).all() or not np.isin(proposal, [0, 1]).all():
        raise ValueError('Use binary targets/proposals and a separate boolean candidate mask.')
    if np.any(candidate & (original == proposal)):
        raise ValueError('Every common candidate must disagree with the original label.')
    target = original.copy()
    weight = np.ones(original.shape, np.float32)
    if arm == 'B':
        weight[candidate] = 0
    elif arm == 'C':
        weight[candidate] = .5
    elif arm == 'D':
        target[candidate] = proposal[candidate]
    return target, weight


class V3Dataset(ExposureDataset):
    def __init__(self, root, records, arm='A', *, candidates=None, proposals=None, **kwargs):
        super().__init__(root, records, **kwargs)
        self.arm = arm
        self.original = OriginalLabels(root)
        self.candidates = DiskArray(candidates) if candidates else None
        self.proposals = DiskArray(proposals) if proposals else None
        if arm != 'A' and (self.candidates is None or self.proposals is None):
            raise ValueError('B/C/D require the same frozen common candidates and OOF proposals.')
        if self.candidates is not None:
            shape = (len(records), 256, 256)
            if self.candidates.shape != shape or self.proposals.shape != shape:
                raise ValueError('Common arrays must be in the exact frozen TRAIN order.')

    def __getitem__(self, index):
        import torch
        from src.training.augmentation import augment_pair
        row = self.records[index]
        image = read_gray(self.root / row['image'])
        _, original, valid = self.original.load(row)
        if not valid.all() or image.shape != original.shape or image.shape != (256, 256):
            raise ValueError('The v3 protocol requires native 256x256, fully valid alpha > 0 tiles.')
        candidate = self.candidates[index] if self.candidates is not None else np.zeros_like(valid)
        proposal = self.proposals[index] if self.proposals is not None else original
        target, weight = arm_target_weight(original, candidate, proposal, self.arm)
        # Integer packing preserves half weights through the existing exact
        # rot90/flip geometry, using exactly the baseline RNG calls.
        packed = target.astype(np.uint8) + 2 * (weight * 2).astype(np.uint8)
        generators = self.generators() if self.augmentation != 'none' else (None, None)
        x, packed = augment_pair(image, packed, self.augmentation, *generators)
        return (torch.from_numpy(x[None]), torch.from_numpy((packed % 2)[None]), index,
                torch.from_numpy((np.floor(packed / 2) / 2)[None]))


def weighted_bce_dice(logits, targets, weights):
    """Per-image weight-normalized BCE and Dice; use weight exactly once.

    All-zero images contribute no loss or gradient. The shared trainer also
    skips an all-zero batch, including AdamW weight decay. Validation has ones.
    """
    import torch
    from torch.nn import functional as F
    if weights is None or logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError('Logits, targets and weights must have identical shapes.')
    logits, targets, weights = logits.float(), targets.float(), weights.float()
    if (not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())
            or bool((weights > 1).any())):
        raise ValueError('Loss weights must be finite and in [0, 1].')
    dims = tuple(range(1, logits.ndim))
    mass = weights.sum(dims)
    usable = mass > 0
    if not bool(usable.any()):
        return logits.sum() * 0
    bce = (weights * F.binary_cross_entropy_with_logits(logits, targets, reduction='none')).sum(dims)
    bce = bce / mass.clamp_min(torch.finfo(torch.float32).tiny)
    p = logits.sigmoid()
    dice = (2 * (weights * p * targets).sum(dims) + 1) / (
        (weights * p).sum(dims) + (weights * targets).sum(dims) + 1)
    return (.5 * bce[usable] + .5 * (1 - dice[usable])).mean()
