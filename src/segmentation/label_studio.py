"""Decode Label Studio brush RLE into its original uint8 alpha channel."""

import numpy as np


def encode_brush_alpha(alpha: np.ndarray) -> list[int]:
    """Encode uint8 alpha as Label Studio brush RLE without thresholding.

    The stream holds four repeated channel values per pixel, matching
    Label Studio mask2rle. Repeated runs use 3/4/8/16-bit length fields.
    """
    if not isinstance(alpha, np.ndarray) or alpha.dtype != np.uint8 or alpha.ndim != 2 or not alpha.size:
        raise ValueError('Expected a nonempty two-dimensional uint8 alpha array')
    if alpha.size * 4 >= 2**32:
        raise ValueError('Brush stream is too large for its 32-bit length header')
    values = np.repeat(alpha.reshape(-1), 4)
    boundaries = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1, [values.size]))
    packed, accumulator, pending = bytearray(), 0, 0

    def write(value, width):
        nonlocal accumulator, pending
        accumulator = (accumulator << width) | int(value)
        pending += width
        while pending >= 8:
            pending -= 8
            packed.append((accumulator >> pending) & 255)
        accumulator &= (1 << pending) - 1

    sizes = (3, 4, 8, 16)
    write(values.size, 32)
    write(7, 5)  # Eight-bit channel values.
    for width in sizes:
        write(width - 1, 4)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        remaining = int(end - start)
        while remaining:
            count = min(remaining, 65536)
            size_index = next(i for i, width in enumerate(sizes) if count <= 1 << width)
            write(1, 1)  # Repeated value, including one-value runs.
            write(size_index, 2)
            write(count - 1, sizes[size_index])
            write(values[start], 8)
            remaining -= count
    if pending:
        packed.append(accumulator << (8 - pending))
    return list(packed)


def decode_brush_alpha(result: dict) -> np.ndarray:
    """Decode the RLE format used by this project's Label Studio brush exports.

    Preserve intermediate alpha values; binary membership is alpha > 0.
    Reject rotated, truncated or incorrectly sized masks instead of misaligning
    their coordinates with source images.
    """
    width, height = result['original_width'], result['original_height']
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0):
        raise ValueError('Brush dimensions must be positive integers')
    if result.get('image_rotation', 0) != 0:
        raise ValueError('Rotated brush results are unsupported')
    if result['value']['format'] != 'rle':
        raise ValueError('Expected brush RLE format')
    packed = bytes(result['value']['rle'])
    bits = ''.join(format(v, '08b') for v in packed)
    cursor = 0

    def read(count: int) -> int:
        nonlocal cursor
        end = cursor + count
        if end > len(bits):
            raise ValueError('Truncated brush RLE input')
        value = int(bits[cursor:end], 2)
        cursor = end
        return value

    count, word_size = read(32), read(5) + 1
    lengths = [read(4) + 1 for _ in range(4)]
    if count != width * height * 4 or word_size != 8:
        raise ValueError('Brush RLE dimensions or word size do not match RGBA')
    output = np.empty(count, dtype=np.uint8)
    index = 0
    while index < count:
        repeated = read(1)
        length = read(lengths[read(2)]) + 1
        end = index + length
        if end > count:
            raise ValueError('Brush RLE run exceeds image dimensions')
        if repeated:
            output[index:end] = read(word_size)
        else:
            for i in range(index, end):
                output[i] = read(word_size)
        index = end
    return output.reshape(height, width, 4)[:, :, 3].copy()


def task_brush_alpha(task: dict, label: str = 'Exposure') -> np.ndarray:
    """Combine this task's completed brush regions for a single class by max."""
    masks = [decode_brush_alpha(r)
             for a in task.get('annotations', []) if not a.get('was_cancelled', False)
             for r in a.get('result', [])
             if r.get('type') == 'brushlabels' and label in r.get('value', {}).get('brushlabels', [])]
    if not masks:
        raise ValueError(f"Task {task.get('id')} has no completed {label!r} brush label")
    if any(m.shape != masks[0].shape for m in masks):
        raise ValueError(f"Task {task.get('id')} contains different brush dimensions")
    return np.maximum.reduce(masks)
