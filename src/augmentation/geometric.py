"""Exact pixel permutations and explicit integer crops, without interpolation."""

import cv2
import numpy as np


def component_signature(mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sorted(stats[1:, cv2.CC_STAT_AREA].tolist())


def integer_crop(image, mask, x, y, size=256):
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (x,y,size)):
        raise ValueError('Crop coordinates must be integers.')
    if mask.shape != image.shape or size < 1 or min(x,y) < 0 or x+size>image.shape[1] or y+size>image.shape[0]:
        raise ValueError('Crop must stay inside aligned source image and GT.')
    return image[y:y+size,x:x+size], mask[y:y+size,x:x+size]


def geometric(image, mask, settings, generator):
    import torch
    operations = []
    for name, operation in (('horizontal_flip',np.fliplr),('vertical_flip',np.flipud),
                            ('rotate180',lambda a:np.rot90(a,2))):
        if settings.get(name) and torch.rand((),generator=generator).item()<.5:
            image, mask = operation(image), operation(mask)
            operations.append(name)
    return image, mask, operations
