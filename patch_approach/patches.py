import numpy as np
def to_patch_grid(values: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """
    Reshape (N,) vector into (grid_h, grid_w) in row-major patch order.
    Assumes you enumerated patches row-by-row.
    """
    values = np.asarray(values)
    if values.size != grid_h * grid_w:
        raise ValueError(f"values has size {values.size}, expected {grid_h*grid_w} (grid_h*grid_w).")
    return values.reshape(grid_h, grid_w)


def extract_patches_nonoverlap(img: np.ndarray, patch: int):
    """
    img: (H,W,C) numpy
    returns:
      patches: (N, patch, patch, C)
      centers: (N,2) patch centers in pixel coords (x,y)
      grid_hw: (grid_h, grid_w)
    """
    H, W, C = img.shape
    grid_h, grid_w = H // patch, W // patch
    Hc, Wc = grid_h * patch, grid_w * patch
    img = img[:Hc, :Wc, :]

    patches = []
    centers = []
    for r in range(grid_h):
        for c in range(grid_w):
            r0, r1 = r * patch, (r + 1) * patch
            c0, c1 = c * patch, (c + 1) * patch
            patches.append(img[r0:r1, c0:c1, :])
            centers.append([c0 + patch / 2.0, r0 + patch / 2.0])  # (x,y)
    return np.stack(patches, axis=0), np.asarray(centers), (grid_h, grid_w)

def filter_bad_patches(
    patches1: np.ndarray,
    XY1: np.ndarray,
    patches2: np.ndarray,
    XY2: np.ndarray,
    *,
    zero_threshold: float = 0.6,
):
    """
    Remove patch pairs where either date has too much zero-valued area.
    Assumes zero pixels mainly come from masked / missing regions.

    zero_threshold:
        fraction of pixels allowed to be zero before dropping the patch.
    """
    # fraction of pixels that are exactly zero across all channels
    frac_zero_1 = np.mean(np.all(patches1 == 0, axis=-1), axis=(1, 2))
    frac_zero_2 = np.mean(np.all(patches2 == 0, axis=-1), axis=(1, 2))

    keep = (frac_zero_1 < zero_threshold) & (frac_zero_2 < zero_threshold)

    return (
        patches1[keep],
        XY1[keep],
        patches2[keep],
        XY2[keep],
        keep
    )
def restore_patch_grid(filtered_values, keep_mask, grid_h, grid_w, fill_value=np.nan):
    full = np.full(grid_h * grid_w, fill_value, dtype=float)
    full[keep_mask] = filtered_values
    return full.reshape(grid_h, grid_w)