from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling


OUT_DIR = Path("../data/output/geoai_ref")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb_for_display(image_path: str) -> np.ndarray:
    with rasterio.open(image_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)

    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.nan_to_num(rgb, nan=0.0)

    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return rgb


def load_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_raster_resampled(path: str, out_hw: tuple[int, int], nearest: bool = False) -> np.ndarray:
    out_h, out_w = out_hw
    method = Resampling.nearest if nearest else Resampling.bilinear
    with rasterio.open(path) as src:
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=method,
        ).astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


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


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=np.float32)

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmax - xmin < 1e-12:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - xmin) / (xmax - xmin)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_geoai_references(
    rgb_image_path: str = "../data/raw/NUS_S2_RGB_2025_MayJul_small.tif",
    binary_mask_path: str = "../data/output/geoai_ref/binary_mask.tif",
    probability_mask_path: str = "../data/output/geoai_ref/probability_mask.tif",
    instance_masks_path: str = "../data/output/geoai_ref/instance_masks.tif",
    patch_size: int = 32,
):
    rgb = load_rgb_for_display(rgb_image_path)
    H, W = rgb.shape[:2]

    # patch-based references
    binary_mask = load_raster_resampled(binary_mask_path, (H, W), nearest=True)
    prob_mask = load_raster_resampled(probability_mask_path, (H, W), nearest=False)
    instance_mask = load_raster_resampled(instance_masks_path, (H, W), nearest=True)

    # For object comparison, keep the instance raster itself
    np.save(OUT_DIR / "geoai_instance_mask.npy", instance_mask.astype(np.float32))

    # Patch references
    binary_patch = raster_to_patch_grid(binary_mask, patch_size)
    prob_patch = raster_to_patch_grid(prob_mask, patch_size)

    # Instance-based patch ref: binary-ize nonzero IDs first
    instance_binary = (instance_mask > 0).astype(np.float32)
    instance_patch = raster_to_patch_grid(instance_binary, patch_size)

    np.save(OUT_DIR / "geoai_binary_patchref.npy", binary_patch.astype(np.float32))
    np.save(OUT_DIR / "geoai_prob_patchref.npy", prob_patch.astype(np.float32))
    np.save(OUT_DIR / "geoai_instance_patchref.npy", instance_patch.astype(np.float32))

    # Sanity-check overlay
    overlay = minmax_normalize(instance_binary)
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.imshow(overlay, cmap="hot", alpha=0.7)
    plt.axis("off")
    plt.title("GeoAI Instance Overlay")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "geoai_instance_overlay.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("[INFO] Saved:")
    print(" - geoai_instance_mask.npy")
    print(" - geoai_binary_patchref.npy")
    print(" - geoai_prob_patchref.npy")
    print(" - geoai_instance_patchref.npy")
    print(" - geoai_instance_overlay.png")


if __name__ == "__main__":
    prepare_geoai_references()