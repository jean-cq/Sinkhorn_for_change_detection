from __future__ import annotations
import numpy as np

# def object_zero_fraction(image: np.ndarray, mask: np.ndarray) -> float:
#     pixels = image[mask]
#     if len(pixels) == 0:
#         return 1.0
#     zero_rows = np.all(pixels == 0, axis=1)
#     return float(zero_rows.mean())

def object_zero_fraction(image: np.ndarray, mask: np.ndarray, offset: tuple[int, int]) -> float:
    y0, x0 = offset
    h, w = mask.shape
    patch = image[y0:y0+h, x0:x0+w]

    pixels = patch[mask]
    if len(pixels) == 0:
        return 1.0

    pixels = np.nan_to_num(pixels, nan=0.0, posinf=255.0, neginf=0.0)
    zero_rows = np.all(pixels == 0, axis=1)
    return float(zero_rows.mean())

def filter_objects(
    image: np.ndarray,
    objects: list[dict],
    min_area: float = 100.0,
    max_zero_fraction: float = 0.6,
    min_sam_score: float = 0.0,
) -> list[dict]:
    """
    Remove invalid or weak objects.
    """
    kept = []
    for obj in objects:
        area = obj.get("area", 0.0)
        if area < min_area:
            continue

        if obj.get("sam_score", 0.0) < min_sam_score:
            continue

        # zero_frac = object_zero_fraction(image, obj["mask"])
        zero_frac = object_zero_fraction(image, obj["mask"], obj.get("offset", (0, 0)))
        if zero_frac > max_zero_fraction:
            continue

        kept.append(obj)

    return kept
# def filter_objects(
#     image: np.ndarray,
#     objects: list[dict],
#     min_area: float = 100.0,
#     max_area_ratio: float = 0.95,
#     max_zero_fraction: float = 0.6,
#     min_sam_score: float = 0.0,
# ) -> list[dict]:
#     H, W = image.shape[:2]
#     image_area = H * W
#
#     kept = []
#     for obj in objects:
#         area = obj.get("area", 0.0)
#         if area < min_area:
#             continue
#         if obj.get("sam_score", 0.0) < min_sam_score:
#             continue
#
#         zero_frac = object_zero_fraction(image, obj["mask"])
#         if zero_frac > max_zero_fraction:
#             continue
#
#         kept.append(obj)
#
#     return kept