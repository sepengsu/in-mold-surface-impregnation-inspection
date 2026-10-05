"""Native square tiles anchored to a geometric ROI's fitted midline.

The ROI is bounded by y = slope*x + intercept. Its horizontal extent is the
camera field of view, not an inferred full specimen width. Each column's row
zero follows the ROI midline. No rotation or resampling occurs.
"""

from dataclasses import dataclass, field
import math
import cv2
import numpy as np
from ._validation import grayscale, integer
from .interior_tiling import InteriorLayout, _centered_axis
from .tiling import validate_tiling


@dataclass(frozen=True)
class ROICenteredLayout(InteriorLayout):
    grid_indices: dict = field(default_factory=dict)
    central_points: dict = field(default_factory=dict)
    centerline: tuple[float, float] = (0.0, 0.0)


def plan_roi_centered_tiling(
    image: np.ndarray, top_line: tuple[float, float], bottom_line: tuple[float, float],
    patch_size: int = 256, stride: int = 256, *, edge_margin: int = 0,
    black_threshold: int = 0, selection: str = 'all', point_spacing: int | None = None,
) -> ROICenteredLayout:
    """Keep full tiles inside both the saved image and fitted ROI boundaries.

    Boundaries use pixel-center x coordinates (0..width-1). Row-zero y start is
    floor(midline(x + (size-1)/2) - size/2). Central points are row zero, columns
    -1, 0, +1. For central_three, point_spacing can separate measurement centers
    independently of tile size. Invalid points are rejected, never shifted.
    """
    grayscale(image)
    patch_size, stride = validate_tiling(patch_size, stride)
    if stride != patch_size:
        raise ValueError('ROI-centered measurement requires stride == patch_size (no overlap).')
    edge_margin = integer(edge_margin, 'edge_margin', 0)
    black_threshold = integer(black_threshold, 'black_threshold', 0)
    if black_threshold > 255:
        raise ValueError('black_threshold must be <= 255')
    if selection not in ('all', 'central_three'):
        raise ValueError('selection must be all or central_three')
    if point_spacing is not None and selection != 'central_three':
        raise ValueError('point_spacing is only supported for central_three')
    spacing = stride if point_spacing is None else integer(point_spacing, 'point_spacing', patch_size)
    top_line, bottom_line = tuple(map(float, top_line)), tuple(map(float, bottom_line))
    if len(top_line) != 2 or len(bottom_line) != 2 or not all(map(math.isfinite, (*top_line, *bottom_line))):
        raise ValueError('ROI boundaries must be finite (slope, intercept) pairs')
    h, w = image.shape
    xs = np.arange(w)
    top = top_line[0] * xs + top_line[1]
    bottom = bottom_line[0] * xs + bottom_line[1]
    if np.any(top >= bottom):
        raise ValueError('ROI top and bottom boundaries cross or have zero height')
    midline = ((top_line[0]+bottom_line[0])/2, (top_line[1]+bottom_line[1])/2)
    geometry = (np.arange(h)[:, None] >= top) & (np.arange(h)[:, None] < bottom)
    valid = geometry & (image > black_threshold)
    integral = cv2.integral(valid.astype(np.uint8), sdepth=cv2.CV_64F)

    def count(x0, y0, x1, y1):
        return int(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])

    center_x0 = (w-patch_size)//2
    columns = ([center_x0 + i*spacing for i in (-1,0,1)] if selection == 'central_three'
               else _centered_axis(w,patch_size,stride))
    candidates, indices, central = [], {}, {}
    for x in columns:
        col = (x-center_x0)//(spacing if selection == 'central_three' else stride)
        cy = midline[0]*(x+(patch_size-1)/2)+midline[1]
        anchor_y = math.floor(cy-patch_size/2)
        rows = ([0] if selection == 'central_three' else
                range(math.ceil(-anchor_y/stride), math.floor((h-patch_size-anchor_y)/stride)+1))
        for row in rows:
            xy = (x,anchor_y+row*stride)
            candidates.append(xy)
            indices[xy] = (row,col)
            if row == 0 and col in (-1,0,1):
                central[xy] = f'P{col+2}'
    candidates.sort(key=lambda xy: indices[xy])
    rejected = dict(outside_image=0, black=0, roi_boundary=0, edge_margin=0)
    positions = []
    for x,y in candidates:
        x1,y1 = x+patch_size,y+patch_size
        if x<0 or y<0 or x1>w or y1>h:
            rejected['outside_image'] += 1
        elif not np.any(image[y:y1,x:x1] > black_threshold):
            rejected['black'] += 1
        elif count(x,y,x1,y1) != patch_size**2:
            rejected['roi_boundary'] += 1
        elif edge_margin and (x<edge_margin or y<edge_margin or x1+edge_margin>w or y1+edge_margin>h
                or count(x-edge_margin,y-edge_margin,x1+edge_margin,y1+edge_margin)
                != (patch_size+2*edge_margin)**2):
            rejected['edge_margin'] += 1
        else:
            positions.append((x,y))
    bounds = (min(x for x,y in candidates),min(y for x,y in candidates),
              max(x+patch_size for x,y in candidates),max(y+patch_size for x,y in candidates)) if candidates else None
    return ROICenteredLayout(positions,candidates,rejected,bounds,patch_size,stride,
                             edge_margin,black_threshold,0,indices,central,midline)
