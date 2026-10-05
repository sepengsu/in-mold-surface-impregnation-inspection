"""Fixed-coordinate cropping at the original pixel resolution."""

from ._validation import GrayImage, grayscale, integer


def crop_roi(image: GrayImage, x: int, y: int, width: int, height: int) -> GrayImage:
    """Copy a valid (x, y, width, height) ROI without changing its pixels."""
    grayscale(image)
    x, y = integer(x, "roi.x", 0), integer(y, "roi.y", 0)
    width = integer(width, "roi.width")
    height = integer(height, "roi.height")
    image_height, image_width = image.shape
    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI (x={x}, y={y}, width={width}, height={height}) exceeds "
            f"image bounds (width={image_width}, height={image_height})"
        )
    return image[y:y + height, x:x + width].copy()
