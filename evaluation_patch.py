from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# PATCH_OT_PATH = "data/output/heatmap_Tengah_2020_MayJul__VS__Tengah_2020_MayJul.npy"
# GEOAI_PATCH_REF_PATH = "data/output/geoai_ref_Tengah_2020_MayJul__VS__Tengah_2025_MayJul/geoai_prob_patchref.npy"

PATCH_OT_PATH = "data/output/heatmap_ideal_image_1__VS__ideal_image_2.npy"
GEOAI_PATCH_REF_PATH = "data/output/geoai_ref_ideal_image_1__VS__ideal_image_2/geoai_prob_patchref.npy"

OUT_DIR = Path("data/output/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def downsample_patch_grid(arr: np.ndarray, factor: int) -> np.ndarray:
    H, W = arr.shape
    if (H % factor == 0 and W % factor == 0):
        arr = arr.reshape(H // factor, factor, W // factor, factor)
        return arr.mean(axis=(1, 3))
    else:
        return arr

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
    patch_ot = np.load(PATCH_OT_PATH).astype(np.float32)
    patch_ot = np.nan_to_num(patch_ot, nan=0.0, posinf=0.0, neginf=0.0)

    geoai_ref = np.load(GEOAI_PATCH_REF_PATH).astype(np.float32)
    geoai_ref = np.nan_to_num(geoai_ref, nan=0.0, posinf=0.0, neginf=0.0)
    if patch_ot.shape != geoai_ref.shape:
        factor_h = patch_ot.shape[0] // geoai_ref.shape[0]
        factor_w = patch_ot.shape[1] // geoai_ref.shape[1]
        assert factor_h == factor_w, "Non-square downsampling factor"
        patch_ot = downsample_patch_grid(patch_ot, factor_h)
    patch_ot, geoai_ref = align_two(patch_ot, geoai_ref)

    print("patch_ot shape:", patch_ot.shape)
    print("patch_ot min/max:", patch_ot.min(), patch_ot.max())
    print("geoai_ref shape:", geoai_ref.shape)
    print("geoai_ref min/max:", geoai_ref.min(), geoai_ref.max())

    patch_norm = robust_normalize(patch_ot)
    geoai_norm = robust_normalize(geoai_ref)

    corr = pearson_corr(patch_norm, geoai_norm)
    print(f"Patch OT vs GeoAI patch ref | Pearson: {corr:.4f}")

    for q in [0.80, 0.85, 0.90, 0.95]:
        p = threshold_binary(patch_norm, q=q)
        g = threshold_binary(geoai_norm, q=q)
        print(f"q={q:.2f} | IoU={iou_score(p,g):.4f} | F1={f1_score(p,g):.4f}")

    np.save(OUT_DIR / "patch_ot_norm.npy", patch_norm)
    np.save(OUT_DIR / "geoai_patch_ref_norm.npy", geoai_norm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(patch_norm, cmap="YlOrRd")
    axes[0].set_title("Patch OT")
    axes[0].axis("off")

    axes[1].imshow(geoai_norm, cmap="YlOrRd")
    axes[1].set_title("GeoAI Patch Ref")
    axes[1].axis("off")

    plt.tight_layout()
    # change the name of the output if you wanna try other graphs/output another comparison
    plt.savefig(OUT_DIR / "patch_vs_geoai_patchref.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()