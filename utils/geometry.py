from __future__ import annotations
import numpy as np


def mask_area(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask))


def mask_perimeter(mask: np.ndarray) -> float:
    """
    Simple 4-neighbour perimeter approximation.
    """
    mask = mask.astype(np.uint8)
    up = np.roll(mask, -1, axis=0)
    down = np.roll(mask, 1, axis=0)
    left = np.roll(mask, -1, axis=1)
    right = np.roll(mask, 1, axis=1)

    boundary = (
        (mask != up) |
        (mask != down) |
        (mask != left) |
        (mask != right)
    ) & (mask == 1)

    return float(np.count_nonzero(boundary))


def compactness(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def bbox_width_height(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return float(max(0, x1 - x0 + 1)), float(max(0, y1 - y0 + 1))


def aspect_ratio_from_bbox(bbox: tuple[int, int, int, int]) -> float:
    w, h = bbox_width_height(bbox)
    if h <= 0:
        return 0.0
    return float(w / h)