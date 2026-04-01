import numpy as np
import matplotlib.pyplot as plt
import rasterio


def raster_to_patch_grid(arr: np.ndarray, patch_size: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    H, W = arr.shape
    gh, gw = H // patch_size, W // patch_size
    Hc, Wc = gh * patch_size, gw * patch_size
    arr = arr[:Hc, :Wc]

    out = np.zeros((gh, gw), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            patch = arr[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size
            ]
            out[i, j] = np.nanmean(patch)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def robust_normalize(x: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.nanpercentile(x, low)
    hi = np.nanpercentile(x, high)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)


def main():
    path = "data/output/geoai_ref/probability_mask.tif"
    patch_size = 32

    with rasterio.open(path) as src:
        prob = src.read(1).astype(np.float32)

    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob_patch = raster_to_patch_grid(prob, patch_size)
    prob_patch = robust_normalize(prob_patch)

    plt.figure(figsize=(6, 6))
    plt.imshow(prob_patch, cmap="hot")
    plt.colorbar()
    plt.title("GeoAI Probability Patch Heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()