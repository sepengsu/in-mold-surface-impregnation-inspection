"""Create reviewable pseudo-segmentation labels inside extracted textile ROIs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from src.roi import auto_roi_local_std as core


ALGORITHM = 'roi_pseudo_segmentation_v3'
FIELDS = (
    'source', 'roi_source', 'mode', 'exposure', 'roi_status', 'roi_needs_review_reason',
    'status', 'reason', 'label_review_status', 'candidate_flags', 'input_height', 'input_width',
    'dtype', 'roi_pixels', 'candidate_method', 'threshold_native', 'low_threshold_native',
    'period_x', 'period_y', 'period_x_correlation', 'period_y_correlation',
    'consensus_exposures_used', 'consensus_min', 'boundary_guard_px',
    'region_count', 'foreground_pixels',
    'foreground_roi_fraction', 'raw_foreground_pixels', 'expanded_small_regions',
    'touches_roi_boundary', 'raw_mask_path', 'mask_path', 'label_path', 'overlay_path',
)


def source_path(root: Path, name: str) -> tuple[Path, Path]:
    relative = Path(name)
    windows = PureWindowsPath(name)
    if (not name or relative.is_absolute() or windows.is_absolute() or windows.drive
            or '..' in relative.parts or '..' in windows.parts):
        raise ValueError(f'not a source-relative path: {name}')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'source path escapes input directory: {name}')
    return path, relative


def read_rows(report: Path, source: Path, mode: str, exposure: str) -> list[dict]:
    with report.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'source', 'mode', 'status', 'exposure', 'roi_detected', 'needs_review',
                    'input_height', 'input_width', 'image_output', 'top_slope', 'top_intercept',
                    'bottom_slope', 'bottom_intercept', 'margin'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('ROI CSV is missing required columns')
        rows = [row for row in reader if row['mode'] == mode
                and (exposure == 'all' or row['exposure'].casefold() == exposure.casefold())]
    if not rows:
        raise ValueError(f'no ROI rows for mode={mode}, exposure={exposure}')
    seen = set()
    for row in rows:
        _, relative = source_path(source, row['source'])
        # PNG masks and TXT labels use the same stem, even for JPEG/TIFF sources.
        key = relative.with_suffix('.txt').as_posix().casefold()
        if key in seen:
            raise ValueError(f'duplicate source or colliding label filename: {relative}')
        seen.add(key)
        if relative.parent.name.casefold() != row['exposure'].casefold():
            raise ValueError(f'exposure disagrees with source path: {relative}')
        if row['image_output'] != 'masked':
            raise ValueError('use an original-size masked ROI report, not crop/rectified output')
    return rows


def finite_number(row: dict, key: str, default=None) -> float:
    text = row.get(key)
    if text is None or text == '':
        if default is None:
            raise ValueError(f'missing ROI value: {key}')
        text = default
    value = float(text)
    if not np.isfinite(value):
        raise ValueError(f'nonfinite ROI value: {key}')
    return value


def roi_mask(row: dict, shape: tuple) -> np.ndarray:
    h, w = shape[:2]
    if [int(row['input_height']), int(row['input_width'])] != [h, w]:
        raise ValueError('ROI report and original image dimensions differ')
    xs = np.arange(w, dtype=float)
    top = finite_number(row, 'top_slope') * xs + finite_number(row, 'top_intercept')
    bottom = finite_number(row, 'bottom_slope') * xs + finite_number(row, 'bottom_intercept')
    if not np.isfinite(np.r_[top, bottom]).all() or np.any(bottom <= top):
        raise ValueError('invalid or crossing ROI boundaries')
    if row['mode'] == 'gabor_boundary':
        if row.get('mask_rounding') != 'nearest_inclusive':
            raise ValueError('missing or unsupported gabor_boundary mask rounding')
        mt, mb = (finite_number(row, 'roi_margin_' + side) for side in ('top', 'bottom'))
        if mt < 0 or mb < 0 or int(mt) != mt or int(mb) != mb:
            raise ValueError('ROI margins must be nonnegative integers')
        top, bottom = np.rint(top) - mt, np.rint(bottom) + mb + 1
    else:
        margin = finite_number(row, 'margin')
        if margin < 0:
            raise ValueError('ROI margin must be nonnegative')
        top, bottom = np.floor(top - margin), np.ceil(bottom + margin)
    top, bottom = np.clip(top, 0, h), np.clip(bottom, 0, h)
    ys = np.arange(h)[:, None]
    mask = (ys >= top) & (ys < bottom)
    if not mask.any():
        raise ValueError('ROI has no retained pixels')
    return mask


def gaussian_difference(image: np.ndarray, roi: np.ndarray, kernel: int, sigma: float,
                        threshold: float | None) -> tuple[np.ndarray, float]:
    if image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError(f'unsupported image dtype: {image.dtype}; expected uint8/uint16')
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError('expected mono, BGR or BGRA image')
    if roi.shape != image.shape[:2] or roi.dtype != bool or not roi.any():
        raise ValueError('expected a nonempty boolean ROI at source resolution')
    maximum = np.iinfo(image.dtype).max
    if threshold is not None and threshold > maximum:
        raise ValueError(f'threshold exceeds {image.dtype} intensity range')
    # Blur the untouched original, never its black-masked export. Float subtraction
    # also avoids the upstream int16 overflow on uint16 inputs.
    gray = core.gray_float(image)
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), sigma)
    difference = np.abs(gray - blurred)
    difference = np.clip(np.rint(difference), 0, maximum).astype(image.dtype)
    if threshold is None:
        threshold, _ = cv2.threshold(difference[roi].reshape(-1, 1), 0, maximum,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return difference, float(threshold)


def gaussian_candidates(image: np.ndarray, roi: np.ndarray, kernel: int, sigma: float,
                        threshold: float | None) -> tuple[np.ndarray, float]:
    difference, threshold = gaussian_difference(image, roi, kernel, sigma, threshold)
    raw = ((difference > threshold) & roi).astype(np.uint8) * 255
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    filled[~roi] = 0
    return filled, float(threshold)


def hysteresis_regions(difference: np.ndarray, roi: np.ndarray, high: float,
                       low_ratio: float, close_kernel: int) -> tuple[np.ndarray, float]:
    """Grow strong residual seeds through connected lower-contrast support."""
    low = float(high * low_ratio)
    seed = (difference > high) & roi
    support = ((difference > low) & roi).astype(np.uint8)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, kernel)
        support[~roi] = 0
    if not seed.any() or not support.any():
        return np.zeros(roi.shape, np.uint8), low
    count, components = cv2.connectedComponents(support, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(components[seed])] = True
    keep[0] = False
    return keep[components].astype(np.uint8) * 255, low


def gaussian_hysteresis_candidates(
        image: np.ndarray, roi: np.ndarray, kernel: int, sigma: float,
        threshold: float | None, low_ratio: float, close_kernel: int,
) -> tuple[np.ndarray, float, float]:
    difference, high = gaussian_difference(image, roi, kernel, sigma, threshold)
    candidate, low = hysteresis_regions(difference, roi, high, low_ratio, close_kernel)
    return candidate, high, low


def rectification(row: dict, shape: tuple) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Return native-to-rectified homographies and output size for one ROI."""
    h, w = shape[:2]
    xs = np.array([0., w - 1.], dtype=np.float64)
    top = finite_number(row, 'top_slope') * xs + finite_number(row, 'top_intercept')
    bottom = finite_number(row, 'bottom_slope') * xs + finite_number(row, 'bottom_intercept')
    if row['mode'] == 'gabor_boundary':
        top -= finite_number(row, 'roi_margin_top')
        bottom += finite_number(row, 'roi_margin_bottom') + 1
    else:
        margin = finite_number(row, 'margin')
        top -= margin
        bottom += margin
    top, bottom = np.clip(top, 0, h - 1), np.clip(bottom, 0, h - 1)
    if np.any(bottom - top < 2):
        raise ValueError('ROI is too short to rectify')
    output_height = max(2, int(round(float(np.mean(bottom - top)))))
    source_quad = np.float32([[0, top[0]], [w - 1, top[1]],
                              [w - 1, bottom[1]], [0, bottom[0]]])
    target_quad = np.float32([[0, 0], [w - 1, 0],
                              [w - 1, output_height - 1], [0, output_height - 1]])
    forward = cv2.getPerspectiveTransform(source_quad, target_quad)
    inverse = cv2.getPerspectiveTransform(target_quad, source_quad)
    return forward, inverse, (w, output_height)


def robust_flat_field(gray: np.ndarray, sigma: float) -> np.ndarray:
    background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma,
                                  borderType=cv2.BORDER_REFLECT)
    centered = gray.astype(np.float32) - background
    median = float(np.median(centered))
    mad = float(np.median(np.abs(centered - median)))
    scale = max(1.4826 * mad, float(np.std(centered)) * .05, 1e-6)
    return ((centered - median) / scale).astype(np.float32)


def autocorrelation_period(signal: np.ndarray, minimum: int, maximum: int,
                           fallback: int, min_correlation: float) -> tuple[int, float]:
    values = np.asarray(signal, np.float64)
    values -= np.mean(values)
    if not np.isfinite(values).all() or np.std(values) < 1e-9:
        return fallback, 0.
    size = 1
    while size < 2 * len(values):
        size *= 2
    spectrum = np.fft.rfft(values, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:len(values)]
    correlation /= np.maximum(np.arange(len(values), 0, -1), 1)
    correlation /= max(float(correlation[0]), 1e-9)
    upper = min(maximum, len(correlation) - 1)
    if upper < minimum:
        return fallback, 0.
    period = minimum + int(np.argmax(correlation[minimum:upper + 1]))
    score = float(correlation[period])
    return (period, score) if score >= min_correlation else (fallback, score)


def estimate_periods(normalized: np.ndarray, args) -> tuple[int, int, float, float]:
    activity = np.abs(normalized)
    period_x, correlation_x = autocorrelation_period(
        activity.mean(axis=0), args.period_min, args.period_max,
        args.period_fallback_x, args.period_min_correlation)
    period_y, correlation_y = autocorrelation_period(
        activity.mean(axis=1), args.period_min, args.period_max,
        args.period_fallback_y, args.period_min_correlation)
    return period_x, period_y, correlation_x, correlation_y


def periodic_residual(normalized: np.ndarray, period_x: int,
                      period_y: int, boundary_guard: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = ((0, -period_x), (0, period_x), (-period_y, 0), (period_y, 0))
    jitter = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    differences = []
    for dy, dx in offsets:
        best = np.full(normalized.shape, np.inf, np.float32)
        for jy, jx in jitter:
            shifted = np.roll(normalized, shift=(dy + jy, dx + jx), axis=(0, 1))
            np.minimum(best, np.abs(normalized - shifted), out=best)
        differences.append(best)
    residual = np.median(np.stack(differences), axis=0).astype(np.float32)
    guard_x = max(boundary_guard, period_x + 1)
    guard_y = max(boundary_guard, period_y + 1)
    valid = np.zeros(normalized.shape, bool)
    if normalized.shape[0] > 2 * guard_y and normalized.shape[1] > 2 * guard_x:
        valid[guard_y:-guard_y, guard_x:-guard_x] = True
    if not valid.any():
        raise ValueError('rectified ROI is too small for the period and boundary guard')
    residual[~valid] = 0
    return residual, valid


def robust_zscore(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    samples = values[valid]
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    scale = max(1.4826 * mad, float(np.std(samples)) * .05, 1e-6)
    score = (values - median) / scale
    score[~valid] = 0
    return score.astype(np.float32)


def limited_region_grow(seed: np.ndarray, support: np.ndarray, steps: int,
                        close_kernel: int) -> np.ndarray:
    support_u8 = support.astype(np.uint8)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        support_u8 = cv2.morphologyEx(support_u8, cv2.MORPH_CLOSE, kernel)
    grown = seed.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        updated = cv2.dilate(grown, kernel) & support_u8
        if np.array_equal(updated, grown):
            break
        grown = updated
    return grown * 255


def exposure_relative(relative: Path, exposure: str) -> Path:
    parts = list(relative.parts)
    if len(parts) < 2 or not re.fullmatch(r'\d+us', parts[-2], re.IGNORECASE):
        raise ValueError(f'source does not end in an exposure folder: {relative}')
    parts[-2] = exposure
    return Path(*parts)


def periodic_consensus_candidates(row: dict, relative: Path, image: np.ndarray,
                                  roi: np.ndarray, args) -> tuple[np.ndarray, dict]:
    forward, inverse, output_size = rectification(row, image.shape)
    reference_gray = core.gray_float(image)
    reference_rectified = cv2.warpPerspective(
        reference_gray, forward, output_size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT)
    reference_normalized = robust_flat_field(reference_rectified, args.illumination_sigma)
    period_x, period_y, correlation_x, correlation_y = estimate_periods(reference_normalized, args)
    score_maps, used = [], []
    for exposure in args.consensus_exposures:
        current_relative = exposure_relative(relative, exposure)
        current_path, _ = source_path(args.source, current_relative.as_posix())
        current = image if exposure.casefold() == row['exposure'].casefold() else core.imread(current_path)
        if current is None:
            continue
        if current.shape[:2] != image.shape[:2]:
            raise ValueError(f'consensus exposure size mismatch: {current_relative}')
        gray = core.gray_float(current)
        rectified = cv2.warpPerspective(gray, forward, output_size, flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
        normalized = robust_flat_field(rectified, args.illumination_sigma)
        residual, valid = periodic_residual(normalized, period_x, period_y, args.boundary_guard)
        score_maps.append(robust_zscore(residual, valid))
        used.append(exposure)
    if len(score_maps) < args.consensus_min:
        raise ValueError(f'only {len(score_maps)} consensus exposures are readable; need {args.consensus_min}')
    scores = np.stack(score_maps)
    seed = np.count_nonzero(scores >= args.seed_z, axis=0) >= args.consensus_min
    support = np.count_nonzero(scores >= args.support_z, axis=0) >= args.consensus_min
    rectified_mask = limited_region_grow(
        seed, support, args.growth_steps, args.hysteresis_close_kernel)
    native = cv2.warpPerspective(rectified_mask, inverse, (image.shape[1], image.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    native[~roi] = 0
    details = {
        'threshold': args.seed_z, 'low_threshold': args.support_z,
        'period_x': period_x, 'period_y': period_y,
        'period_x_correlation': correlation_x, 'period_y_correlation': correlation_y,
        'consensus_exposures_used': ';'.join(used), 'consensus_min': args.consensus_min,
        'boundary_guard_px': max(args.boundary_guard, period_x + 1, period_y + 1),
    }
    return native, details


def expand_degenerate_contour(contour: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
    """Turn a point/thin-line component into the smallest useful area polygon."""
    h, w = roi.shape
    x, y, width, height = cv2.boundingRect(contour)
    x0, y0 = max(0, x - 1), max(0, y - 1)
    x1, y1 = min(w, x + width + 1), min(h, y + height + 1)
    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    offset = np.array([[[x0, y0]]], dtype=contour.dtype)
    cv2.drawContours(local, [contour - offset], -1, 255, cv2.FILLED)
    local = cv2.dilate(local, np.ones((3, 3), np.uint8))
    local[~roi[y0:y1, x0:x1]] = 0
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    expanded = max(contours, key=cv2.contourArea) + offset
    return expanded if len(expanded) >= 3 and cv2.contourArea(expanded) > 0 else None


def export_polygons(mask: np.ndarray, roi: np.ndarray, min_area: float,
                    approximation: float, preserve_small: bool = False
                    ) -> tuple[np.ndarray, list[str], list[np.ndarray], int]:
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    final = np.zeros_like(mask)
    polygons, labels, expanded_count = [], [], 0

    def raster(poly: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
        x, y, width, height = cv2.boundingRect(poly)
        local = np.zeros((height, width), np.uint8)
        offset = np.array([[[x, y]]], dtype=poly.dtype)
        cv2.fillPoly(local, [poly - offset], 255)
        return x, y, width, height, local

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        polygon = (contour if preserve_small else
                   cv2.approxPolyDP(contour, approximation * cv2.arcLength(contour, True), True))
        if len(polygon) < 3 or cv2.contourArea(polygon) <= 0:
            polygon = contour
            if len(polygon) < 3 or cv2.contourArea(polygon) <= 0:
                if not preserve_small:
                    continue
                polygon = expand_degenerate_contour(contour, roi)
                if polygon is None:
                    continue
                expanded_count += 1
        # Simplification may bridge a staircase in a tilted ROI. Fall back to its
        # exact contour if needed, and keep the PNG identical to the polygon raster.
        x, y, width, height, local = raster(polygon)
        valid = roi[y:y + height, x:x + width]
        if np.any(local[~valid]):
            polygon = contour
            x, y, width, height, local = raster(polygon)
            valid = roi[y:y + height, x:x + width]
            if np.any(local[~valid]):
                raise ValueError('contour fill leaves ROI; cannot export a faithful YOLO polygon')
        target = final[y:y + height, x:x + width]
        np.maximum(target, local, out=target)
        coordinates = polygon.reshape(-1, 2).astype(float) / np.array([w, h])
        labels.append('0 ' + ' '.join(f'{v:.8f}' for v in coordinates.ravel()))
        polygons.append(polygon)
    return final, labels, polygons, expanded_count


def overlay_image(image: np.ndarray, roi: np.ndarray, mask: np.ndarray,
                  polygons: list[np.ndarray]) -> np.ndarray:
    overlay = core.display_bgr(image).copy()
    selected = mask != 0
    overlay[selected] = (overlay[selected] * .55 + np.array([0, 0, 255]) * .45).astype(np.uint8)
    boundaries, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, boundaries, -1, (0, 255, 0), 2)
    if polygons:
        cv2.drawContours(overlay, polygons, -1, (0, 255, 255), 1)
    return overlay


def process_row(row: dict, args) -> dict:
    status = row.get('quality_status') or row['status']
    result = {k: row.get(k, '') for k in ('source', 'roi_source', 'mode', 'exposure')}
    result.update(roi_status=status, roi_needs_review_reason=row.get('needs_review_reason') or row.get('reason', ''),
                  label_review_status='pending', status='skipped')
    if row['status'] == 'skipped':
        result['reason'] = 'roi_skipped:' + row.get('reason', '')
        return result
    if row.get('roi_detected') != '1':
        result['reason'] = 'no_localized_roi'
        return result
    accepted = status in {'PASS', 'RECOVERED', 'MANUAL', 'manual', 'auto', 'auto_fallback', 'fitted', 'recovered'}
    review = row.get('needs_review') != '0' or not accepted
    if review and args.review_policy == 'skip':
        result['reason'] = 'roi_requires_review'
        return result
    path, relative = source_path(args.source, row['source'])
    image = core.imread(path)
    if image is None:
        raise ValueError('unreadable original image')
    roi = roi_mask(row, image.shape)
    diagnostics = {}
    if args.candidate_method == 'periodic_consensus':
        candidate, diagnostics = periodic_consensus_candidates(row, relative, image, roi, args)
        threshold = diagnostics['threshold']
        low_threshold = diagnostics['low_threshold']
    elif args.candidate_method == 'gaussian_hysteresis':
        candidate, threshold, low_threshold = gaussian_hysteresis_candidates(
            image, roi, args.gaussian_kernel, args.gaussian_sigma, args.mask_threshold,
            args.hysteresis_low_ratio, args.hysteresis_close_kernel)
    else:
        candidate, threshold = gaussian_candidates(
            image, roi, args.gaussian_kernel, args.gaussian_sigma, args.mask_threshold)
        low_threshold = None
    minimum_area = 0. if args.preserve_small else args.min_contour_area
    mask, labels, polygons, expanded = export_polygons(
        candidate, roi, minimum_area, args.contour_approximation, args.preserve_small)
    edge = roi & ~cv2.erode(roi.astype(np.uint8), np.ones((3, 3), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    touches = bool(np.any((mask != 0) & edge))
    bucket = Path('roi_review') if review else Path('.')
    mask_path = bucket / 'masks' / relative.with_suffix('.png')
    raw_mask_path = bucket / 'raw_masks' / relative.with_suffix('.png') if args.preserve_small else None
    label_path = bucket / 'labels' / relative.with_suffix('.txt')
    overlay_path = bucket / 'overlays' / relative.with_suffix('.png') if args.save_overlays else None
    core.imwrite(args.output / mask_path, mask)
    if raw_mask_path is not None:
        core.imwrite(args.output / raw_mask_path, candidate)
    label_file = args.output / label_path
    label_file.parent.mkdir(parents=True, exist_ok=True)
    with label_file.open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(labels) + ('\n' if labels else ''))
    if overlay_path is not None:
        core.imwrite(args.output / overlay_path, overlay_image(image, roi, mask, polygons))
    flags = []
    if review:
        flags.append('roi_requires_review')
    if touches:
        flags.append('touches_roi_boundary')
    if not labels:
        flags.append('no_candidates_not_confirmed_normal')
    if args.preserve_small:
        flags.append('small_components_preserved')
    if args.candidate_method == 'gaussian_hysteresis':
        flags.append('hysteresis_region_growing')
    if args.candidate_method == 'periodic_consensus':
        flags.extend(('periodic_template_suppression', 'multi_exposure_consensus',
                      'limited_region_growing', 'rectified_analysis'))
    state = 'roi_review' if review else 'draft'
    if not labels:
        state = 'empty_' + state
    result.update(status=state, reason='', candidate_flags=';'.join(flags),
                  input_height=image.shape[0], input_width=image.shape[1], dtype=str(image.dtype),
                  roi_pixels=int(roi.sum()), candidate_method=args.candidate_method,
                  threshold_native=threshold,
                  low_threshold_native=low_threshold if low_threshold is not None else '',
                  period_x=diagnostics.get('period_x', ''), period_y=diagnostics.get('period_y', ''),
                  period_x_correlation=diagnostics.get('period_x_correlation', ''),
                  period_y_correlation=diagnostics.get('period_y_correlation', ''),
                  consensus_exposures_used=diagnostics.get('consensus_exposures_used', ''),
                  consensus_min=diagnostics.get('consensus_min', ''),
                  boundary_guard_px=diagnostics.get('boundary_guard_px', ''),
                  region_count=len(labels),
                  foreground_pixels=int(np.count_nonzero(mask)),
                  foreground_roi_fraction=float(np.count_nonzero(mask) / roi.sum()),
                  raw_foreground_pixels=int(np.count_nonzero(candidate)), expanded_small_regions=expanded,
                  touches_roi_boundary=int(touches), mask_path=mask_path.as_posix(), label_path=label_path.as_posix(),
                  raw_mask_path=raw_mask_path.as_posix() if raw_mask_path is not None else '',
                  overlay_path=overlay_path.as_posix() if overlay_path is not None else '')
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', type=Path, default=Path('data/0918'), help='Original, NOT black-masked images')
    parser.add_argument('--roi-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory')
    parser.add_argument('--mode', choices=(*core.MODES, 'auto', 'gabor_boundary'), default='gabor_boundary')
    parser.add_argument('--exposure', default='10000us', help='Exposure folder, or all; each image is processed independently')
    parser.add_argument('--gaussian-kernel', type=int, default=101)
    parser.add_argument('--gaussian-sigma', type=float, default=10.)
    parser.add_argument('--mask-threshold', type=float, help='Fixed threshold in source intensity units; omit for ROI-only Otsu')
    parser.add_argument('--candidate-method',
                        choices=('gaussian_otsu', 'gaussian_hysteresis', 'periodic_consensus'),
                        default='gaussian_otsu')
    parser.add_argument('--hysteresis-low-ratio', type=float, default=.4,
                        help='Low threshold as a fraction of the Otsu/fixed high threshold')
    parser.add_argument('--hysteresis-close-kernel', type=int, default=3,
                        help='Odd elliptical closing kernel; 0 or 1 disables closing')
    parser.add_argument('--consensus-exposures', default='8000us,10000us,12000us')
    parser.add_argument('--consensus-min', type=int, default=2)
    parser.add_argument('--period-min', type=int, default=24)
    parser.add_argument('--period-max', type=int, default=48)
    parser.add_argument('--period-fallback-x', type=int, default=35)
    parser.add_argument('--period-fallback-y', type=int, default=33)
    parser.add_argument('--period-min-correlation', type=float, default=.15)
    parser.add_argument('--illumination-sigma', type=float, default=80.)
    parser.add_argument('--seed-z', type=float, default=25.)
    parser.add_argument('--support-z', type=float, default=12.)
    parser.add_argument('--growth-steps', type=int, default=2)
    parser.add_argument('--boundary-guard', type=int, default=64)
    parser.add_argument('--min-contour-area', type=float, default=50., help='Minimum contour area in native square pixels')
    parser.add_argument('--contour-approximation', type=float, default=.002, help='Fraction of contour perimeter')
    parser.add_argument('--preserve-small', action='store_true',
                        help='Keep every Otsu component; expand zero-area points/lines only for YOLO polygons')
    parser.add_argument('--class-name', default='Exposure', help='Single proposed class, always class ID 0')
    parser.add_argument('--review-policy', choices=('quarantine', 'skip'), default='quarantine')
    parser.add_argument('--save-overlays', action='store_true')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gaussian_kernel <= 0 or args.gaussian_kernel % 2 != 1:
        parser.error('--gaussian-kernel must be a positive odd integer')
    for key in ('gaussian_sigma', 'min_contour_area', 'contour_approximation', 'mask_threshold'):
        value = getattr(args, key)
        if value is not None and (not np.isfinite(value) or value < 0):
            parser.error(f'--{key.replace("_", "-")} must be finite and nonnegative')
    if args.contour_approximation > .1:
        parser.error('--contour-approximation must be <= 0.1')
    if args.mask_threshold is not None and args.mask_threshold > 65535:
        parser.error('--mask-threshold must be <= 65535')
    if not np.isfinite(args.hysteresis_low_ratio) or not 0 < args.hysteresis_low_ratio <= 1:
        parser.error('--hysteresis-low-ratio must be in (0, 1]')
    if args.hysteresis_close_kernel < 0 or (args.hysteresis_close_kernel > 1
                                            and args.hysteresis_close_kernel % 2 != 1):
        parser.error('--hysteresis-close-kernel must be 0, 1, or a positive odd integer')
    args.consensus_exposures = tuple(
        value.strip() for value in args.consensus_exposures.split(',') if value.strip())
    if (not args.consensus_exposures or len(set(map(str.casefold, args.consensus_exposures)))
            != len(args.consensus_exposures)
            or any(not re.fullmatch(r'\d+us', value, re.IGNORECASE)
                   for value in args.consensus_exposures)):
        parser.error('--consensus-exposures must be unique comma-separated exposure folders')
    if args.consensus_min < 1 or args.consensus_min > len(args.consensus_exposures):
        parser.error('--consensus-min must be between 1 and the number of consensus exposures')
    if args.period_min < 2 or args.period_max < args.period_min:
        parser.error('--period-min/--period-max must satisfy 2 <= min <= max')
    if not all(args.period_min <= value <= args.period_max
               for value in (args.period_fallback_x, args.period_fallback_y)):
        parser.error('period fallbacks must be inside the configured period range')
    if not np.isfinite(args.period_min_correlation) or not 0 <= args.period_min_correlation <= 1:
        parser.error('--period-min-correlation must be in [0, 1]')
    if (not np.isfinite(args.illumination_sigma) or args.illumination_sigma <= 0
            or not np.isfinite(args.seed_z) or not np.isfinite(args.support_z)
            or not 0 <= args.support_z < args.seed_z):
        parser.error('illumination sigma must be positive and z thresholds require 0 <= support < seed')
    if args.growth_steps < 0 or args.boundary_guard < 0:
        parser.error('--growth-steps and --boundary-guard must be nonnegative')
    if not args.class_name.strip() or any(c in args.class_name for c in '\r\n\0'):
        parser.error('--class-name must be a nonempty single line')
    if args.exposure != 'all' and not re.fullmatch(r'\d+us', args.exposure, re.IGNORECASE):
        parser.error('--exposure must be a folder such as 10000us, or all')
    if args.candidate_method == 'periodic_consensus':
        if args.exposure == 'all':
            parser.error('periodic_consensus requires one reference --exposure')
        if args.exposure.casefold() not in {value.casefold() for value in args.consensus_exposures}:
            parser.error('the reference --exposure must be included in --consensus-exposures')
    args.source, args.output, args.roi_report = (p.resolve() for p in (args.source, args.output, args.roi_report))
    if not args.source.is_dir():
        parser.error('original source directory does not exist')
    if args.source.is_relative_to(args.output) or args.output.is_relative_to(args.source):
        parser.error('source and output trees must be separate, not nested')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('output must be a new or empty directory; old results are never overwritten')
    try:
        rows = read_rows(args.roi_report, args.source, args.mode, args.exposure)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with (args.output / 'pseudo_results.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            try:
                result = process_row(row, args)
            except (OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
                result = {k: row.get(k, '') for k in ('source', 'roi_source', 'mode', 'exposure')}
                result.update(status='error', reason=str(exc), label_review_status='pending')
            writer.writerow(result)
            handle.flush()
            stats[result['status']] += 1
            stats['regions'] += result.get('region_count', 0)
            print(f"pseudo {i}/{len(rows)} {row['source']} {result['status']} "
                  f"regions={result.get('region_count', 0)} {result.get('reason', '')}", flush=True)
    for bucket in (Path('.'), Path('roi_review')):
        label_root = args.output / bucket / 'labels'
        if label_root.is_dir():
            for filename in ('classes.txt', 'labelmap.txt'):
                with (label_root / filename).open('x', encoding='utf-8') as handle:
                    handle.write(args.class_name.strip() + '\n')
    manifest = {
        'algorithm': ALGORITHM, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'roi_report_sha256': hashlib.sha256(args.roi_report.read_bytes()).hexdigest(),
        'selected_images': len(rows), 'counts': dict(stats), 'class_names': [args.class_name.strip()],
        'label_review_status': 'pending', 'empty_label_is_confirmed_normal': False,
        'small_component_policy': ('raw mask kept exactly; zero-area components expanded by one pixel for YOLO'
                                   if args.preserve_small else f'contours below {args.min_contour_area} px^2 removed'),
        'upstream_algorithm': 'https://github.com/yutakseo/pseudo_labeller/tree/5b9cea5',
        'note': 'Candidate labels, not GT. Review normal texture, glare, scratches, and boundary artifacts.',
    }
    with (args.output / 'run_manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print('pseudo_summary=' + json.dumps(dict(stats)), flush=True)
    print(f'output={args.output}', flush=True)
    return 1 if stats['error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
