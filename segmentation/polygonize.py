from __future__ import annotations
import numpy as np


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def mask_to_centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    cx = xs.mean()
    cy = ys.mean()
    return np.array([cx, cy], dtype=np.float32)


def mask_to_bbox_global(mask: np.ndarray, y0: int, x0: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return (
        int(xs.min() + x0),
        int(ys.min() + y0),
        int(xs.max() + x0),
        int(ys.max() + y0),
    )


def mask_to_centroid_global(mask: np.ndarray, y0: int, x0: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    cx = xs.mean() + x0
    cy = ys.mean() + y0
    return np.array([cx, cy], dtype=np.float32)


def masks_to_objects(masks: list[dict]) -> list[dict]:
    """
    Convert SAM mask dictionaries into object dictionaries.

    Supports tile-local masks with global offsets.
    """
    objects = []

    for i, m in enumerate(masks):
        seg = m["segmentation"].astype(bool)
        y0, x0 = m.get("offset", (0, 0))

        obj = {
            "id": i,
            "mask": seg,  # still local mask
            "offset": (y0, x0),
            "image_shape": m.get("image_shape", None),
            "bbox": m.get("bbox", mask_to_bbox_global(seg, y0, x0)),
            "centroid": mask_to_centroid_global(seg, y0, x0),
            "sam_score": float(m.get("predicted_iou", 0.0)),
            "stability_score": float(m.get("stability_score", 0.0)),
            "area": float(np.count_nonzero(seg)),
        }
        objects.append(obj)

    return objects