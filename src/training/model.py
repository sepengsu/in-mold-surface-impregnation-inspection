"""Small grayscale U-Net; no pretrained weights or online downloads."""

import torch
from torch import nn
from torch.nn import functional as F

from .losses import segmentation_loss  # Backwards-compatible public import.


class DoubleConv(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, outgoing), outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, outgoing), outgoing), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8.")
        widths = [base_channels * 2**i for i in range(5)]
        self.encoders = nn.ModuleList(
            DoubleConv(1 if i == 0 else widths[i - 1], width)
            for i, width in enumerate(widths)
        )
        self.decoders = nn.ModuleList(
            DoubleConv(widths[i + 1] + widths[i], widths[i])
            for i in range(3, -1, -1)
        )
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))
        return self.head(x)


class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.samples = 0
        self.loss_sum = self.mae_sum = 0.0

    @torch.no_grad()
    def update(self, logits, targets, loss):
        predictions = logits.sigmoid() >= self.threshold
        actual = targets > 0.5
        self.tp += int((predictions & actual).sum().item())
        self.fp += int((predictions & ~actual).sum().item())
        self.fn += int((~predictions & actual).sum().item())
        batch = targets.shape[0]
        self.samples += batch
        self.loss_sum += float(loss) * batch
        # 100*(1-Exposure fraction): absolute difference in percentage points.
        pred_rate = 100 * (1 - predictions.float().mean((1, 2, 3)))
        true_rate = 100 * (1 - actual.float().mean((1, 2, 3)))
        self.mae_sum += (pred_rate - true_rate).abs().sum().item()

    def compute(self):
        if not self.samples:
            raise ValueError("No samples were evaluated.")
        dice_denominator = 2 * self.tp + self.fp + self.fn
        union = self.tp + self.fp + self.fn
        return {
            "loss": self.loss_sum / self.samples,
            "dice": 2 * self.tp / dice_denominator if dice_denominator else 1.0,
            "iou": self.tp / union if union else 1.0,
            "precision": self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0,
            "recall": self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0,
            "impregnation_mae_pp": self.mae_sum / self.samples,
            "samples": self.samples,
        }
