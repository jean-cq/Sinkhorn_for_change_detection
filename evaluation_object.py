from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


OBJECT_OT_RASTER_PATH = "data/output/object_score_map_Tengah_2020_MayJul__VS__Tengah_2025_MayJul.npy"
GEOAI_PATCH_REF_PATH = "data/output/geoai_ref_Tengah_2020_MayJul__VS__Tengah_2025_MayJul/geoai_prob_patchref.npy"

# OBJECT_OT_RASTER_PATH = "data/output/object_score_map_ideal_image_1__VS__ideal_image_2.npy"
# GEOAI_PATCH_REF_PATH = "data/output/geoai_ref_ideal_image_1__VS__ideal_image_2/geoai_prob_patchref.npy"

# satellite patch
PATCH_SIZE = 16
# ideal patch
# PATCH_SIZE = 8
OUT_DIR = Path("data/output/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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

    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=np.float32)

    lo = np.nanpercentile(x, low)
    hi = np.nanpercentile(x, high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def align_two(a: np.ndarray, b: np.ndarray):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def threshold_binary(x: np.ndarray, q: float = 0.9) -> np.ndarray:
    t = np.quantile(x, q)
    return (x >= t).astype(np.uint8)


def iou_score(pred: np.ndarray, ref: np.ndarray) -> float:
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return 0.0 if union == 0 else float(inter / union)


def f1_score(pred: np.ndarray, ref: np.ndarray) -> float:
    tp = np.logical_and(pred == 1, ref == 1).sum()
    fp = np.logical_and(pred == 1, ref == 0).sum()
    fn = np.logical_and(pred == 0, ref == 1).sum()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float((2 * tp) / denom)


def main():
    object_ot = np.load(OBJECT_OT_RASTER_PATH).astype(np.float32)
    object_ot = np.nan_to_num(object_ot, nan=0.0, posinf=0.0, neginf=0.0)

    # stabilize heavy-tail
    # object_ot = np.log1p(np.maximum(object_ot, 0.0))

    # convert object OT raster to patch grid
    object_patch = raster_to_patch_grid(object_ot, PATCH_SIZE)
    print(object_patch.min(), object_patch.max())

    # use SAME patch reference as patch-based method
    geoai_patch_ref = np.load(GEOAI_PATCH_REF_PATH).astype(np.float32)
    geoai_patch_ref = np.nan_to_num(geoai_patch_ref, nan=0.0, posinf=0.0, neginf=0.0)

    object_patch, geoai_patch_ref = align_two(object_patch, geoai_patch_ref)

    object_norm = robust_normalize(object_patch)
    geoai_norm = robust_normalize(geoai_patch_ref)

    corr = pearson_corr(object_norm, geoai_norm)
    print(f"Object OT vs GeoAI patch ref | Pearson: {corr:.4f}")

    for q in [0.80, 0.85, 0.90, 0.95]:
        p = threshold_binary(object_norm, q=q)
        g = threshold_binary(geoai_norm, q=q)
        print(f"q={q:.2f} | IoU={iou_score(p,g):.4f} | F1={f1_score(p,g):.4f}")

    np.save(OUT_DIR / "object_ot_patchgrid_norm.npy", object_norm)
    np.save(OUT_DIR / "geoai_patch_ref_for_object_norm.npy", geoai_norm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(object_norm, cmap="YlOrRd")
    axes[0].set_title("Object OT (patch-grid)")
    axes[0].axis("off")

    axes[1].imshow(geoai_norm, cmap="YlOrRd")
    axes[1].set_title("GeoAI Patch Ref")
    axes[1].axis("off")

    plt.tight_layout()
    # change the name of the output if you wanna try other graphs/output another comparison
    plt.savefig(OUT_DIR / "object_vs_geoai_patchref.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()