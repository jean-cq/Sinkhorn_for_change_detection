from __future__ import annotations
import numpy as np


def rasterize_object_scores(
    objects: list[dict],
    scores: np.ndarray,
    image_shape: tuple[int, int],
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Paint each object's score onto its mask.
    """
    H, W = image_shape
    out = np.full((H, W), fill_value, dtype=np.float32)

    for obj, s in zip(objects, scores):
        mask = obj["mask"]
        # out[mask] = float(s)
        y0, x0 = obj.get("offset", (0, 0))
        h, w = mask.shape

        sub = out[y0:y0 + h, x0:x0 + w]
        sub[mask] = float(s)
        out[y0:y0 + h, x0:x0 + w] = sub
    return out