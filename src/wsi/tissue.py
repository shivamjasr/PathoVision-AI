from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def make_tissue_mask(
    thumbnail: Image.Image,
    saturation_threshold: int = 20,
    brightness_threshold: int = 245,
    morph_kernel_size: int = 7,
    min_component_area: int = 64,
) -> np.ndarray:
    """Estimate tissue regions from a thumbnail.

    The method is intentionally simple and interpretable:
    colored tissue generally differs from bright white slide background.
    It is a preprocessing heuristic, not a tumor classifier.
    """

    if morph_kernel_size < 3 or morph_kernel_size % 2 == 0:
        raise ValueError("morph_kernel_size must be an odd integer >= 3.")

    rgb = np.asarray(thumbnail.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    saturation = hsv[..., 1]
    brightness = hsv[..., 2]

    # Keep pixels that are sufficiently colored OR sufficiently dark.
    # This is deliberately permissive so tissue is not accidentally removed.
    mask = (
        (saturation >= saturation_threshold)
        | (brightness <= brightness_threshold)
    ).astype(np.uint8) * 255

    kernel = np.ones(
        (morph_kernel_size, morph_kernel_size),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            cleaned[labels == label_id] = 255

    return cleaned > 0


def tissue_fraction(
    mask: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> float:
    """Return fraction of tissue pixels inside a mask rectangle."""
    h, w = mask.shape

    x0 = max(0, min(int(x0), w))
    x1 = max(0, min(int(x1), w))
    y0 = max(0, min(int(y0), h))
    y1 = max(0, min(int(y1), h))

    if x1 <= x0 or y1 <= y0:
        return 0.0

    region = mask[y0:y1, x0:x1]
    return float(region.mean())


def save_mask(mask: np.ndarray, path):
    image = Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L",
    )
    image.save(path)
