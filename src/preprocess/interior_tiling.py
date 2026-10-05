"""Centered native-resolution tiles that stay entirely inside a nonblack ROI.

Unlike the coverage-oriented tiler, this module drops partial boundary tiles,
never pads or shifts the final tile, and rejects black pixels before enhancement.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ._validation import GrayImage, grayscale, integer
from .tiling import validate_tiling


@dataclass(frozen=True)
class InteriorLayout:
    """Row-major retained coordinates, with disjoint rejection categories."""

    positions: list[tuple[int, int]]
    candidate_positions: list[tuple[int, int]]
    rejected: dict[str, int]
    grid_bounds: tuple[int, int, int, int] | None
    patch_size: int
    stride: int
    edge_margin: int
    black_threshold: int
    border_tiles: int


def _centered_axis(length: int, size: int, stride: int) -> list[int]:
    if length < size:
        return []
    center_start = (length - size) // 2
    start = center_start - center_start // stride * stride
    return list(range(start, length - size + 1, stride))


def plan_interior_tiling(
    image: GrayImage, patch_size: int = 256, stride: int = 256, *,
    edge_margin: int = 0, black_threshold: int = 0,
    valid_mask: np.ndarray | None = None, border_tiles: int = 0,
) -> InteriorLayout:
    """Keep only full tiles whose entire footprint and margin are valid.

    ``image`` must be the raw masked uint8 mono image (before CLAHE). A pixel
    <= black_threshold is invalid. Optional boolean valid_mask additionally
    restricts the ROI; it never makes black pixels valid. Exposure/label masks
    must not be used as the ROI mask.

    edge_margin requires this many valid pixels on every side, including at
    diagonal corners (a square margin). border_tiles drops rings of the full
    image-centered candidate grid, not rings of the irregular ROI. At odd image
    dimensions, the central tile origin is rounded down by at most half a pixel.
    """
    grayscale(image)
    patch_size, stride = validate_tiling(patch_size, stride)
    edge_margin = integer(edge_margin, 'edge_margin', 0)
    border_tiles = integer(border_tiles, 'border_tiles', 0)
    black_threshold = integer(black_threshold, 'black_threshold', 0)
    if black_threshold > 255:
        raise ValueError('black_threshold must be <= 255')
    if valid_mask is not None:
        if not isinstance(valid_mask, np.ndarray) or valid_mask.dtype != np.bool_:
            raise TypeError('valid_mask must be a boolean NumPy array')
        if valid_mask.shape != image.shape:
            raise ValueError('valid_mask shape must equal image shape')
    height, width = image.shape
    xs, ys = (_centered_axis(n, patch_size, stride) for n in (width, height))
    candidates = [(x, y) for y in ys for x in xs]
    bounds = (xs[0], ys[0], xs[-1] + patch_size, ys[-1] + patch_size) if candidates else None
    rejected = dict(black=0, roi_boundary=0, edge_margin=0, grid_border=0)
    valid = image > black_threshold
    if valid_mask is not None:
        valid &= valid_mask
    # Integral sums make every tile/margin check constant time.
    integral = cv2.integral(valid.astype(np.uint8), sdepth=cv2.CV_64F)

    def valid_pixels(x0: int, y0: int, x1: int, y1: int) -> int:
        return int(integral[y1, x1] - integral[y0, x1]
                   - integral[y1, x0] + integral[y0, x0])

    positions = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            x1, y1 = x + patch_size, y + patch_size
            count = valid_pixels(x, y, x1, y1)
            if count == 0:
                rejected['black'] += 1
            elif count != patch_size ** 2:
                rejected['roi_boundary'] += 1
            elif (row < border_tiles or row >= len(ys) - border_tiles
                  or col < border_tiles or col >= len(xs) - border_tiles):
                rejected['grid_border'] += 1
            elif edge_margin:
                a, b, c, d = x - edge_margin, y - edge_margin, x1 + edge_margin, y1 + edge_margin
                if (a < 0 or b < 0 or c > width or d > height
                        or valid_pixels(a, b, c, d) != (c - a) * (d - b)):
                    rejected['edge_margin'] += 1
                else:
                    positions.append((x, y))
            else:
                positions.append((x, y))
    return InteriorLayout(positions, candidates, rejected, bounds, patch_size,
                          stride, edge_margin, black_threshold, border_tiles)


def tile_interior_image(
    image: GrayImage, patch_size: int = 256, stride: int = 256, *,
    edge_margin: int = 0, black_threshold: int = 0,
    valid_mask: np.ndarray | None = None, border_tiles: int = 0,
) -> tuple[list[GrayImage], InteriorLayout]:
    """Return independent copies of raw tiles plus filtering/coordinate metadata."""
    layout = plan_interior_tiling(image, patch_size, stride, edge_margin=edge_margin,
                                  black_threshold=black_threshold, valid_mask=valid_mask,
                                  border_tiles=border_tiles)
    patches = [image[y:y + patch_size, x:x + patch_size].copy() for x, y in layout.positions]
    return patches, layout
