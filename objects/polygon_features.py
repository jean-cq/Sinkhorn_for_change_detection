# objects/polygon_features.py
from __future__ import annotations
import numpy as np
from utils.geometry import mask_area, mask_perimeter, compactness, aspect_ratio_from_bbox


# def masked_rgb_stats(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Compute mean/std RGB within a mask, ignoring NaN and inf values.
#     """
#     pixels = image[mask]
#
#     if len(pixels) == 0:
#         mean = np.zeros(3, dtype=np.float32)
#         std = np.zeros(3, dtype=np.float32)
#         return mean, std
#
#     pixels = pixels.astype(np.float32)
#     pixels = np.nan_to_num(pixels, nan=0.0, posinf=255.0, neginf=0.0)
#
#     mean = pixels.mean(axis=0).astype(np.float32)
#     std = pixels.std(axis=0).astype(np.float32)
#
#     return mean, std

def masked_rgb_stats(image: np.ndarray, mask: np.ndarray, offset: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    y0, x0 = offset
    h, w = mask.shape
    patch = image[y0:y0+h, x0:x0+w]

    pixels = patch[mask]

    if len(pixels) == 0:
        mean = np.zeros(3, dtype=np.float32)
        std = np.zeros(3, dtype=np.float32)
        return mean, std

    pixels = pixels.astype(np.float32)
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=255.0, neginf=0.0)

    mean = pixels.mean(axis=0).astype(np.float32)
    std = pixels.std(axis=0).astype(np.float32)
    return mean, std


def extract_shape_features(obj: dict) -> np.ndarray:
    area = mask_area(obj["mask"])
    perimeter = mask_perimeter(obj["mask"])
    compact = compactness(area, perimeter)
    aspect = aspect_ratio_from_bbox(obj["bbox"])

    return np.array([
        area,
        perimeter,
        compact,
        aspect,
        obj.get("sam_score", 0.0),
        obj.get("stability_score", 0.0),
    ], dtype=np.float32)


def extract_appearance_features_basic(
    image: np.ndarray,
    obj: dict,
) -> np.ndarray:
    # mean_rgb, std_rgb = masked_rgb_stats(image, obj["mask"])
    mean_rgb, std_rgb = masked_rgb_stats(image, obj["mask"], obj.get("offset", (0, 0)))
    feat = np.concatenate([mean_rgb, std_rgb], axis=0).astype(np.float32)
    return feat

def attach_object_features(
    image: np.ndarray,
    objects: list[dict],
) -> list[dict]:
    """
    Attach shape and appearance features to each object.
    """
    out = []
    for obj in objects:
        obj = dict(obj)
        obj["shape_features"] = extract_shape_features(obj)
        obj["appearance_features"] = extract_appearance_features_basic(image, obj)
        out.append(obj)
    return out


def stack_object_arrays(objects: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack object features into arrays.

    Returns:
        XY: (N, 2) centroid coordinates
        F:  (N, d_feat) appearance feature matrix
        S:  (N, d_shape) shape/meta feature matrix
    """
    XY = np.stack([obj["centroid"] for obj in objects], axis=0).astype(np.float32)
    F = np.stack([obj["appearance_features"] for obj in objects], axis=0).astype(np.float32)
    S = np.stack([obj["shape_features"] for obj in objects], axis=0).astype(np.float32)
    return XY, F, S