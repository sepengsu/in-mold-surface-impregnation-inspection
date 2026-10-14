"""Keep original targets; ignore suspect TRAIN pixels only in the loss."""

from pathlib import Path

import numpy as np

from src.training.data import ExposureDataset, read_gray
from .disk import DiskArray
from .labels import OriginalLabels, decode_mask, read_mask, revised_label
from .protocol import read_json


class IgnorePixelDataset(ExposureDataset):
    def __init__(self, root, records, issues_path, **kwargs):
        super().__init__(root, records, **kwargs)
        self.issues_path = str(Path(issues_path).resolve())
        self._issues = None  # Each Windows DataLoader worker opens its own memmap.

    def __getitem__(self, index):
        import torch
        from src.training.augmentation import augment_pair

        if self._issues is None:
            self._issues = np.load(self.issues_path, mmap_mode="r")
            if self._issues.shape != (len(self.records), 256, 256) or self._issues.dtype != np.bool_:
                raise ValueError("Issue masks must be boolean N x 256 x 256 in training-record order.")
        row = self.records[index]
        image = read_gray(self.root / row["image"])
        target = (read_gray(self.root / row["mask"]) > 0).astype(np.uint8)
        valid = (~self._issues[index]).astype(np.uint8)
        # Pack two binary maps so the existing rotation/flip transforms both
        # identically without interpolation or changing the baseline RNG stream.
        packed = target + 2 * valid
        generators = self.generators() if self.augmentation != "none" else (None, None)
        x, packed = augment_pair(image, packed, self.augmentation, *generators)
        y, valid = packed % 2, packed >= 2
        return (torch.from_numpy(x[None]), torch.from_numpy(y[None]), index,
                torch.from_numpy(valid[None]))


class ArmDataset(ExposureDataset):
    """No image cache; read one original/derived tile and validity per sample."""

    def __init__(self, root, records, arm='A', *, issues=None, auto=None, revision=None, **kwargs):
        super().__init__(root, records, **kwargs)
        self.arm, self.revision = arm, revision
        self.original = OriginalLabels(root)
        self.issues = DiskArray(issues) if issues else None
        self.auto_directory = Path(auto) if auto else None
        self.auto = read_json(self.auto_directory/'manifest.json') if auto else None
        if arm == 'B' and self.issues is None or arm == 'D' and self.auto is None:
            raise ValueError('B requires the shared issues; D requires saved one-shot auto labels.')
        if self.auto and [t['image'] for t in self.auto['tiles']] != [r['image'] for r in records]:
            raise ValueError('D labels are not in TRAIN order.')

    def __getitem__(self, index):
        import torch
        from src.training.augmentation import augment_pair
        row = self.records[index]
        image = read_gray(self.root/row['image'])
        _, target, original_valid = revised_label(self.original, row, self.revision)
        valid = original_valid.copy()
        if self.arm == 'B':
            valid &= ~self.issues[index]
        elif self.arm == 'D':
            entry = self.auto['tiles'][index]
            raw = read_mask(self.auto_directory/entry['mask'])
            target, saved_valid = decode_mask(raw, self.auto['encoding'], read_mask(self.auto_directory/entry['valid_mask']))
            if not np.array_equal(saved_valid, original_valid):
                raise ValueError('D changed the original validity mask.')
            valid = original_valid  # Corrected candidate pixels remain supervised.
        packed = target.astype(np.uint8) + 2*valid.astype(np.uint8)
        generators = self.generators() if self.augmentation != 'none' else (None,None)
        x, packed = augment_pair(image, packed, self.augmentation, *generators)
        return (torch.from_numpy(x[None]), torch.from_numpy((packed%2)[None]), index,
                torch.from_numpy((packed>=2)[None]))


def masked_bce_dice(logits, targets, valid):
    """Per-image valid-pixel BCE + Dice; all-invalid images have zero weight."""
    import torch
    from torch.nn import functional as F

    if valid is None or logits.shape != targets.shape or logits.shape != valid.shape:
        raise ValueError("Logits, targets and validity masks must have identical shapes.")
    logits, targets, valid = logits.float(), targets.float(), valid.float()
    dims = (1, 2, 3)
    count = valid.sum(dims)
    usable = count > 0
    if not bool(usable.any()):
        return logits.sum() * 0
    bce = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * valid).sum(dims)
    bce = bce / count.clamp_min(1)
    probability = logits.sigmoid() * valid
    target = targets * valid
    dice = (2 * (probability * target).sum(dims) + 1) / (probability.sum(dims) + target.sum(dims) + 1)
    return (0.5 * bce[usable] + 0.5 * (1 - dice[usable])).mean()
