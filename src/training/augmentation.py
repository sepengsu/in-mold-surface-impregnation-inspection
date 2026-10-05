"""Mild grayscale augmentation; geometry preserves every binary label pixel."""

from copy import deepcopy

import cv2
import numpy as np

MODES = ("none", "basic", "strong")
PHOTO_CONFIG = {
    "brightness_contrast": {"probability": 0.5, "brightness_factor": [0.9, 1.1], "contrast_factor": [0.9, 1.1]},
    "gamma": {"probability": 0.3, "exponent": [0.9, 1.1]},
    "blur": {"probability": 0.15, "kernel": 3, "sigma": [0.2, 0.6]},
    "noise": {"probability": 0.2, "std_on_0_1": [0.002, 0.008]},
}


def describe_augmentation(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown augmentation mode: {mode}")
    return {
        "mode": mode, "geometry": "random 0/90/180/270 rotation + horizontal flip p=0.5" if mode != "none" else None,
        "photometric": deepcopy(PHOTO_CONFIG) if mode == "strong" else None,
        "application": "training only", "mask_interpolation": "none", "resize_or_padding": False,
        "normalization": "uint8 / 255", "output_range": [0, 1],
    }


def photometric(image, generator):
    """Brightness/contrast -> gamma -> mild blur -> noise, all image-only."""
    import torch

    def uniform(bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand((), generator=generator).item()

    def enabled(name):
        return torch.rand((), generator=generator).item() < PHOTO_CONFIG[name]["probability"]

    x = np.array(image, dtype=np.float32, copy=True)
    if enabled("brightness_contrast"):
        settings = PHOTO_CONFIG["brightness_contrast"]
        brightness, contrast = uniform(settings["brightness_factor"]), uniform(settings["contrast_factor"])
        mean = float(x.mean())
        x = np.clip(((x - mean) * contrast + mean) * brightness, 0, 1)
    if enabled("gamma"):
        x = np.power(x, uniform(PHOTO_CONFIG["gamma"]["exponent"]))
    if enabled("blur"):
        size = PHOTO_CONFIG["blur"]["kernel"]
        x = cv2.GaussianBlur(x, (size, size), uniform(PHOTO_CONFIG["blur"]["sigma"]), borderType=cv2.BORDER_REFLECT_101)
    if enabled("noise"):
        std = uniform(PHOTO_CONFIG["noise"]["std_on_0_1"])
        x = x + torch.randn(x.shape, generator=generator).numpy() * std
    return np.ascontiguousarray(np.clip(x, 0, 1), dtype=np.float32)


def augment_pair(image, mask, mode, geometry_generator=None, photo_generator=None):
    """Return normalized float32 image and binary mask with aligned geometry."""
    import torch

    if mode not in MODES:
        raise ValueError(f"Unknown augmentation mode: {mode}")
    if mode != "none":
        turns = int(torch.randint(4, (1,), generator=geometry_generator).item())
        image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        if bool(torch.rand((), generator=geometry_generator) < 0.5):
            image, mask = np.fliplr(image), np.fliplr(mask)
    x = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    if mode == "strong":
        x = photometric(x, photo_generator)
    return x, np.ascontiguousarray(mask, dtype=np.float32)


def save_preview(root, records, mode, output, seed):
    """Show two training tiles, three variants each, with their aligned masks."""
    import torch
    from .data import read_gray

    indices = sorted({0, len(records) // 2})
    size, gap, row_height = 256, 16, 584
    canvas = np.full((row_height * len(indices), 4 * (size + gap) + gap, 3), 22, np.uint8)
    geometry = torch.Generator().manual_seed(seed)
    photo = torch.Generator().manual_seed(seed + 100003)
    for row_index, index in enumerate(indices):
        record = records[index]
        image = read_gray(root / record["image"])
        mask = read_gray(root / record["mask"]) > 0
        pairs = [(image.astype(np.float32) / 255, mask)]
        pairs += [augment_pair(image, mask, mode, geometry, photo) for _ in range(3)]
        y = row_index * row_height
        title = f"TRAIN task {record['task_id']} | {mode} | Exposure pixels: {int(mask.sum())}"
        cv2.putText(canvas, title, (gap, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        for column, (x, target) in enumerate(pairs):
            left = gap + column * (size + gap)
            name = "Original" if column == 0 else f"Variant {column}"
            cv2.putText(canvas, name, (left, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
            rgb = cv2.cvtColor(np.round(x * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            canvas[y + 58:y + 314, left:left + size] = rgb
            overlay = rgb.copy()
            selection = target > 0
            overlay[selection] = np.round(0.5 * overlay[selection] + 0.5 * np.array([60, 220, 60])).astype(np.uint8)
            canvas[y + 324:y + 580, left:left + size] = overlay
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise OSError("Unable to encode augmentation preview")
    encoded.tofile(output)
