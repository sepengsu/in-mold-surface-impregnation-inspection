"""Horizontal rectangle ROI based on a robust texture envelope.

Promoted unchanged from the user-approved single-image experiment.
No labels or previously fitted ROI boundaries are used for detection.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_closing
from .roi_boundary import features
from .roi_boundary_quality import load_config

def segments(flags: np.ndarray) -> list[tuple[int, int]]:
    differences = np.diff(np.r_[False, flags, False].astype(np.int8))
    return list(zip(np.flatnonzero(differences == 1).tolist(),
                    np.flatnonzero(differences == -1).tolist()))


def select_band(profile: np.ndarray, background: float, high: float, fraction: float) -> tuple[int, int, float]:
    threshold = background + fraction * (high - background)
    flags = profile >= threshold
    # Join brief weak-texture rows, never bridge a large background interval.
    gap = max(3, int(round(len(profile) * .02)))
    flags = binary_closing(flags, structure=np.ones(gap, dtype=bool))
    candidates = [(a, b) for a, b in segments(flags) if b - a >= .15 * len(profile)]
    if not candidates:
        raise ValueError('No substantial continuous textured band was detected.')
    start, stop = max(candidates, key=lambda ab: float(np.maximum(profile[ab[0]:ab[1]]-threshold, 0).sum()))
    if start <= gap or stop >= len(profile)-gap:
        raise ValueError('Vertical boundary reaches the image edge; cannot estimate both edges.')
    return start, stop, float(threshold)


def estimate(image: np.ndarray, fraction: float, trim: float, inset: int) -> tuple[dict, dict]:
    if not 0 < fraction < 1 or not 0 <= trim < .4 or inset < 0:
        raise ValueError('fraction must be (0,1), x-trim [0,.4), and inset >= 0.')
    h, w = image.shape[:2]
    feature = features(image, load_config())
    keep = (feature['x'] >= w*trim) & (feature['x'] < w*(1-trim))
    texture = feature['texture'][keep].astype(np.float64)
    if len(texture) < 4:
        raise ValueError('Too few image columns for a robust profile.')
    # Equalize bin strength so bright exposed fibers do not dominate the band.
    scales = np.percentile(texture, 95, axis=1, keepdims=True)
    normalized = np.clip(texture/np.maximum(scales, 1e-6), 0, 1.5)
    profile = gaussian_filter1d(np.median(normalized, axis=0), sigma=max(2., h*.006))
    edge_rows = max(1, int(h*.2))
    background = float(np.median(np.r_[profile[:edge_rows], profile[-edge_rows:]]))
    high = float(np.percentile(profile, 95))
    if high-background < .05:
        raise ValueError('Insufficient texture contrast; review image manually.')
    start, stop, threshold = select_band(profile, background, high, fraction)
    y0, y1 = start+inset, stop-inset
    if y1 <= y0:
        raise ValueError('Inset removes the whole ROI.')
    sensitivity = []
    for value in sorted(set([max(.01, fraction-.02), fraction, min(.99, fraction+.02)])):
        a, b, _ = select_band(profile, background, high, value)
        sensitivity.append({'fraction': value, 'y0': a, 'y1_exclusive': b})
    report = {
        'method': 'horizontal rectangle from robust full-width texture envelope',
        'status': 'EXPERIMENTAL_REVIEW_REQUIRED',
        'source_size_wh': [w, h], 'xyxy_exclusive': [0, y0, w, y1],
        'roi_size_wh': [w, y1-y0], 'detected_band_before_inset': [start, stop],
        'center_y_pixel_coordinate': (y0+y1-1)/2,
        'horizontal_boundary_assumption': True,
        'x_extent': 'full image field of view; material endpoints not detected',
        'roi_label_used': False, 'saved_roi_lines_used': False,
        'resized': False, 'rotated': False, 'inset_px': inset,
        'parameters': {'threshold_fraction': fraction, 'x_trim_fraction': trim,
                       'profile_smoothing_sigma_px': max(2., h*.006)},
        'background_profile': background, 'high_texture_profile': high,
        'threshold': threshold, 'threshold_sensitivity': sensitivity,
    }
    return report, {'profile': profile, 'normalized': normalized}

