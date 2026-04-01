from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from geoai_evaluation.geoai_rasterize import raster_to_patch_grid


PATCH_HEATMAP_PATH = "data/output/heatmap_small.npy"
OBJECT_RASTER_PATH = "data/output/object_score_map_small.npy"

# from your GeoAI object-reference pipeline
GEOAI_PATCH_REF_PATH = "data/output/geoai_ref/geoai_patch_reference.npy"
GEOAI_VECTOR_RASTER_PATH = "data/output/geoai_ref/geoai_vector_raster.npy"

PATCH_SIZE = 32
OUT_DIR = Path("data/output/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=np.float32)

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmax - xmin < 1e-12:
        return np.zeros_like(x, dtype=np.float32)

    out = (x - xmin) / (xmax - xmin)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def robust_normalize(x: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=np.float32)

    lo = np.percentile(x, low)
    hi = np.percentile(x, high)
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def align_three(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    h = min(a.shape[0], b.shape[0], c.shape[0])
    w = min(a.shape[1], b.shape[1], c.shape[1])
    return a[:h, :w], b[:h, :w], c[:h, :w]


def pearson_corr_2d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()

    if a.size == 0 or b.size == 0:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))

    x_sorted = x[sorter]
    ranks = np.zeros(len(x), dtype=np.float64)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x_sorted[j] == x_sorted[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j

    return ranks[inv]


def spearman_corr_2d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()

    if a.size == 0 or b.size == 0:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0

    ra = rankdata_average(a)
    rb = rankdata_average(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0

    return float(np.corrcoef(ra, rb)[0, 1])


def threshold_binary(x: np.ndarray, q: float = 0.9) -> np.ndarray:
    t = np.quantile(x, q)
    return (x >= t).astype(np.uint8)


def iou_score(pred: np.ndarray, ref: np.ndarray) -> float:
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def f1_score_binary(pred: np.ndarray, ref: np.ndarray) -> float:
    tp = np.logical_and(pred == 1, ref == 1).sum()
    fp = np.logical_and(pred == 1, ref == 0).sum()
    fn = np.logical_and(pred == 0, ref == 1).sum()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def evaluate_against_reference(
    name: str,
    pred_map: np.ndarray,
    ref_map: np.ndarray,
    quantiles=(0.80, 0.85, 0.90, 0.95),
):
    pearson = pearson_corr_2d(pred_map, ref_map)
    spearman = spearman_corr_2d(pred_map, ref_map)

    rows = []
    for q in quantiles:
        pred_bin = threshold_binary(pred_map, q=q)
        ref_bin = threshold_binary(ref_map, q=q)
        rows.append({
            "name": name,
            "q": float(q),
            "iou": iou_score(pred_bin, ref_bin),
            "f1": f1_score_binary(pred_bin, ref_bin),
        })

    return pearson, spearman, rows


def save_metrics_csv(rows, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("name,q,iou,f1\n")
        for r in rows:
            f.write(f"{r['name']},{r['q']:.2f},{r['iou']:.6f},{r['f1']:.6f}\n")


def print_metric_table(rows):
    print("\n=== Threshold sweep ===")
    print(f"{'Method':<16} {'q':>6} {'IoU':>10} {'F1':>10}")
    for r in rows:
        print(f"{r['name']:<16} {r['q']:>6.2f} {r['iou']:>10.4f} {r['f1']:>10.4f}")


def main():
    # patch OT is already patch-level
    patch_heatmap = np.load(PATCH_HEATMAP_PATH).astype(np.float32)

    # object OT is pixel-level raster -> convert to patch-level for visualization
    object_raster = np.load(OBJECT_RASTER_PATH).astype(np.float32)
    object_raster = np.nan_to_num(object_raster, nan=0.0, posinf=0.0, neginf=0.0)

    # optional stabilization for heavy-tail object OT
    object_raster_log = np.log1p(np.maximum(object_raster, 0.0))
    object_patch = raster_to_patch_grid(object_raster_log, PATCH_SIZE)

    # GeoAI references
    geoai_patch_ref = np.load(GEOAI_PATCH_REF_PATH).astype(np.float32)
    geoai_vector_raster = np.load(GEOAI_VECTOR_RASTER_PATH).astype(np.float32)
    geoai_vector_raster = np.nan_to_num(geoai_vector_raster, nan=0.0, posinf=0.0, neginf=0.0)
    geoai_vector_patch = raster_to_patch_grid(geoai_vector_raster, PATCH_SIZE)

    # ---------- Patch OT vs GeoAI patch reference ----------
    patch_map, geoai_patch_map = patch_heatmap, geoai_patch_ref
    patch_map, geoai_patch_map, _dummy = align_three(
        patch_map, geoai_patch_map, geoai_patch_map
    )
    patch_norm = robust_normalize(patch_map)
    geoai_patch_norm = robust_normalize(geoai_patch_map)

    patch_pearson, patch_spearman, patch_rows = evaluate_against_reference(
        "Patch OT", patch_norm, geoai_patch_norm
    )

    # ---------- Object OT vs GeoAI vector-raster reference ----------
    object_map, geoai_obj_map = object_patch, geoai_vector_patch
    object_map, geoai_obj_map, _dummy = align_three(
        object_map, geoai_obj_map, geoai_obj_map
    )
    object_norm = robust_normalize(object_map)
    geoai_obj_norm = robust_normalize(geoai_obj_map)

    object_pearson, object_spearman, object_rows = evaluate_against_reference(
        "Object OT", object_norm, geoai_obj_norm
    )

    # diagnostics
    print("=== Shapes ===")
    print(f"Patch OT:          {patch_norm.shape}")
    print(f"Object OT:         {object_norm.shape}")
    print(f"GeoAI patch ref:   {geoai_patch_norm.shape}")
    print(f"GeoAI object ref:  {geoai_obj_norm.shape}")

    print("\n=== Correlations ===")
    print(f"Patch OT  vs GeoAI patch   | Pearson: {patch_pearson:.4f} | Spearman: {patch_spearman:.4f}")
    print(f"Object OT vs GeoAI objects | Pearson: {object_pearson:.4f} | Spearman: {object_spearman:.4f}")

    all_rows = patch_rows + object_rows
    print_metric_table(all_rows)
    save_metrics_csv(all_rows, OUT_DIR / "metrics_threshold_sweep.csv")

    # save normalized arrays for later reuse
    np.save(OUT_DIR / "patch_ot_patchgrid.npy", patch_norm)
    np.save(OUT_DIR / "object_ot_patchgrid.npy", object_norm)
    np.save(OUT_DIR / "geoai_patch_reference.npy", geoai_patch_norm)
    np.save(OUT_DIR / "geoai_object_reference.npy", geoai_obj_norm)

    # summary json
    summary = {
        "patch_ot_vs_geoai_patch": {
            "pearson": patch_pearson,
            "spearman": patch_spearman,
        },
        "object_ot_vs_geoai_objects": {
            "pearson": object_pearson,
            "spearman": object_spearman,
        },
    }
    with open(OUT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # figure 1: method vs its intended GeoAI reference
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(patch_norm, cmap="hot")
    axes[0].set_title("Patch OT")
    axes[0].axis("off")

    axes[1].imshow(geoai_patch_norm, cmap="hot")
    axes[1].set_title("GeoAI Patch Ref")
    axes[1].axis("off")

    axes[2].imshow(object_norm, cmap="hot")
    axes[2].set_title("Object OT")
    axes[2].axis("off")

    axes[3].imshow(geoai_obj_norm, cmap="hot")
    axes[3].set_title("GeoAI Object Ref")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison_four_panel.png", dpi=200, bbox_inches="tight")
    plt.show()

    # figure 2: binary sweep
    quantiles = [0.80, 0.85, 0.90, 0.95]
    fig, axes = plt.subplots(len(quantiles), 4, figsize=(12, 3 * len(quantiles)))

    for r, q in enumerate(quantiles):
        axes[r, 0].imshow(threshold_binary(patch_norm, q=q), cmap="gray")
        axes[r, 0].set_title(f"Patch q={q:.2f}")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(threshold_binary(geoai_patch_norm, q=q), cmap="gray")
        axes[r, 1].set_title(f"GeoAI Patch q={q:.2f}")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(threshold_binary(object_norm, q=q), cmap="gray")
        axes[r, 2].set_title(f"Object q={q:.2f}")
        axes[r, 2].axis("off")

        axes[r, 3].imshow(threshold_binary(geoai_obj_norm, q=q), cmap="gray")
        axes[r, 3].set_title(f"GeoAI Obj q={q:.2f}")
        axes[r, 3].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "threshold_sweep_four_panel.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()