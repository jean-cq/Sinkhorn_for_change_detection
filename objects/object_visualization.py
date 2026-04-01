from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


# def normalize_img(img: np.ndarray) -> np.ndarray:
#     img = img.astype(np.float32)
#     if img.max() > 1.0:
#         img = img / 255.0
#     return np.clip(img, 0.0, 1.0)
#
#
# def show_object_score_overlay(
#     image: np.ndarray,
#     score_map: np.ndarray,
#     title: str = "Object-based change map",
#     cmap: str = "hot",
#     alpha: float = 0.55,
# ):
#     img = normalize_img(image)
#
#     plt.figure(figsize=(8, 8))
#     plt.imshow(img)
#     plt.imshow(score_map, cmap=cmap, alpha=alpha)
#     plt.title(title)
#     plt.axis("off")
#     plt.show()

def normalize_rgb_for_display(
    image: np.ndarray,
    lower_pct: float = 2.0,
    upper_pct: float = 98.0,
) -> np.ndarray:
    """
    Convert an RGB GeoTIFF image into a display-ready uint8 image.

    Args:
        image: (H, W, 3) RGB image, possibly float with NaN values.
        lower_pct: Lower percentile for contrast stretching.
        upper_pct: Upper percentile for contrast stretching.

    Returns:
        (H, W, 3) uint8 image suitable for matplotlib display.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")

    img = image.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.zeros_like(img, dtype=np.float32)

    for c in range(3):
        band = img[..., c]
        lo = np.percentile(band, lower_pct)
        hi = np.percentile(band, upper_pct)

        if hi <= lo:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)

    return out


def show_object_score_overlay(
    image: np.ndarray,
    score_map: np.ndarray,
    title: str = "Object-based Change Overlay",
    cmap: str = "hot",
    alpha: float = 0.55,
):
    """
    Show object-based change scores overlaid on the original RGB image.
    """
    base_img = normalize_rgb_for_display(image)

    plt.figure(figsize=(8, 8))

    # Show the base satellite image first
    plt.imshow(base_img)

    # Mask background so only scored object regions are overlaid
    masked_scores = np.ma.masked_invalid(score_map)

    if np.isfinite(score_map).any():
        valid = score_map[np.isfinite(score_map)]
        vmin = np.percentile(valid, 2)
        vmax = np.percentile(valid, 98)
        if vmax <= vmin:
            vmin = valid.min()
            vmax = valid.max() + 1e-6
    else:
        vmin, vmax = 0.0, 1.0

    plt.imshow(
        masked_scores,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )

    plt.title(title)
    plt.axis("off")
    plt.show()