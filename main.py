import time
import numpy as np
import matplotlib.pyplot as plt

from geotiff_processing import read_geotiff_rgb

from segmentation.sam_segment import run_sam_segmentation_tiled, run_sam_segmentation_tiled_joint
from segmentation.mask_postprocess import postprocess_sam_masks
from segmentation.polygonize import masks_to_objects

from objects.polygon_features import attach_object_features, stack_object_arrays
from objects.object_filtering import filter_objects
from objects.object_visualization import show_object_score_overlay
from utils.auto_params import choose_params
from utils.path_lib import build_run_names

from utils.rasterize import rasterize_object_scores
from utils.cache_io import save_npy, save_pickle, load_pickle

from sinkhorn import sinkhorn_object_change
import torch
from transformers import pipeline


USE_CACHE = True

# PATH1 = "data/ideal/ideal_image_1.tif"
# PATH2 = "data/ideal/ideal_image_2.tif"
PATH1 = "data/raw/Tengah_2020_MayJul.tif"
PATH2 = "data/raw/Tengah_2025_MayJul.tif"

paths = build_run_names(PATH1, PATH2)

MASK_CACHE_1 = paths["mask_cache_1"]
MASK_CACHE_2 = paths["mask_cache_2"]
OBJ_CACHE_1 = paths["obj_cache_1"]
OBJ_CACHE_2 = paths["obj_cache_2"]
SCORE_MAP_PATH = paths["object_score_map"]

def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = np.min(x)
    hi = np.max(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def build_sam_generator():
    """
    Create and return SAM automatic mask generator.
    """

    if torch.cuda.is_available():
        device = 0
    else:
        device = -1

    generator = pipeline(
        task="mask-generation",
        model="facebook/sam-vit-base",
        device=device,
    )
    return generator

def main():
    t0 = time.perf_counter()

    # 1. Read images
    img1, meta1 = read_geotiff_rgb(PATH1)
    img2, meta2 = read_geotiff_rgb(PATH2)

    img1 = np.nan_to_num(img1, nan=0.0)
    img2 = np.nan_to_num(img2, nan=0.0)

    params = choose_params(img1)

    print("[INFO] Auto mode:", params["mode"])
    print("[INFO] Auto params:", params)

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    t1 = time.perf_counter()
    print(f"[TIME] Read GeoTIFFs: {t1 - t0:.3f}s")
    print(f"[INFO] Image shape: {img1.shape}")

    # 2. Build SAM generator
    sam_generator = build_sam_generator()

    # 3. Run SAM
    raw_masks1, raw_masks2 = run_sam_segmentation_tiled_joint(
        img1,
        img2,
        sam_generator,
        tile_size=params["tile_size"],
        overlap=params["overlap"],
        points_per_batch=params["points_per_batch"],
        pred_iou_thresh=params["pred_iou_thresh"],
        stability_score_thresh=params["stability_score_thresh"],
        zero_threshold=params["zero_threshold"],
        window_size=params["window_size"],
    )
    t2 = time.perf_counter()
    print(f"[TIME] SAM segmentation: {t2 - t1:.3f}s")
    print(f"[INFO] Raw mask count T1: {len(raw_masks1)}")
    print(f"[INFO] Raw mask count T2: {len(raw_masks2)}")

    # 4. Postprocess masks
    masks1 = postprocess_sam_masks(
        raw_masks1,
        image_shape=img1.shape[:2],
        min_area=params["min_area"],
        max_area_ratio=params["max_area_ratio"],
        iou_threshold=params["iou_threshold"],
        bbox_iou_threshold=params["bbox_iou_threshold"],
        max_masks_after_sort=params["max_masks_after_sort"],
    )

    masks2 = postprocess_sam_masks(
        raw_masks2,
        image_shape=img1.shape[:2],
        min_area=params["min_area"],
        max_area_ratio=params["max_area_ratio"],
        iou_threshold=params["iou_threshold"],
        bbox_iou_threshold=params["bbox_iou_threshold"],
        max_masks_after_sort=params["max_masks_after_sort"],
    )

    t3 = time.perf_counter()
    print(f"[TIME] Mask postprocessing: {t3 - t2:.3f}s")
    print(f"[INFO] Kept mask count T1: {len(masks1)}")
    print(f"[INFO] Kept mask count T2: {len(masks2)}")

    # 5. Convert masks to objects
    if USE_CACHE:
        try:
            objects1 = load_pickle(OBJ_CACHE_1)
            objects2 = load_pickle(OBJ_CACHE_2)
            print("[INFO] Loaded cached objects.")
        except FileNotFoundError:
            objects1 = masks_to_objects(masks1)
            objects2 = masks_to_objects(masks2)
            save_pickle(OBJ_CACHE_1, objects1)
            save_pickle(OBJ_CACHE_2, objects2)
            print("[INFO] Built and saved objects.")
    else:
        objects1 = masks_to_objects(masks1)
        objects2 = masks_to_objects(masks2)

    t4 = time.perf_counter()
    print(f"[TIME] Object conversion: {t4 - t3:.3f}s")
    print(f"[INFO] Object count T1: {len(objects1)}")
    print(f"[INFO] Object count T2: {len(objects2)}")

    # 6. Attach object features
    objects1 = attach_object_features(img1, objects1)
    objects2 = attach_object_features(img2, objects2)

    t5 = time.perf_counter()
    print(f"[TIME] Feature extraction: {t5 - t4:.3f}s")

    # 7. Filter objects
    objects1 = filter_objects(
        img1,
        objects1,
        min_area=0.0,
        max_zero_fraction=params["max_zero_fraction"],
        min_sam_score=0.0,
    )
    objects2 = filter_objects(
        img2,
        objects2,
        min_area=0.0,
        max_zero_fraction=params["max_zero_fraction"],
        min_sam_score=0.0,
    )

    if len(objects1) == 0 or len(objects2) == 0:
        print("[ERROR] No valid objects remain after filtering.")
        print(f"[ERROR] objects1={len(objects1)}, objects2={len(objects2)}")
        return

    t6 = time.perf_counter()
    print(f"[TIME] Object filtering: {t6 - t5:.3f}s")
    print(f"[INFO] Final object count T1: {len(objects1)}")
    print(f"[INFO] Final object count T2: {len(objects2)}")

    if len(objects1) == 0 or len(objects2) == 0:
        raise ValueError("No valid objects remain after filtering.")

    # 8. Stack arrays
    XY1, F1, S1 = stack_object_arrays(objects1)
    XY2, F2, S2 = stack_object_arrays(objects2)

    print(f"[INFO] XY1 shape: {XY1.shape}, F1 shape: {F1.shape}, S1 shape: {S1.shape}")
    print(f"[INFO] XY2 shape: {XY2.shape}, F2 shape: {F2.shape}, S2 shape: {S2.shape}")

    if np.isnan(XY1).any() or np.isnan(F1).any() or np.isnan(S1).any():
        raise ValueError("NaN detected in source object features.")
    if np.isnan(XY2).any() or np.isnan(F2).any() or np.isnan(S2).any():
        raise ValueError("NaN detected in target object features.")
    # 9. Run object-based Sinkhorn OT directly
    result = sinkhorn_object_change(
        XY1, F1, S1,
        XY2, F2, S2,
        alpha=1,
        beta=1,
        gamma=1,
        gate_radius=0.15,
        gate_cost=1e6,
        # for ideal image test use eps=0.0001
        # eps=0.0001,
        eps=0.005,
        tau_a=0.5,
        tau_b=0.5,
        n_iters=500,
        tol=1e-6,
    )
    t7 = time.perf_counter()
    print(f"[TIME] Sinkhorn OT: {t7 - t6:.3f}s")
    print(f"[INFO] Total OT cost: {result['ot_cost']:.6f}")
    print(f"[INFO] Cost matrix shape: {result['C'].shape}")

    # 10. Rasterize scores
    exp_src = np.nan_to_num(result["score_expected_cost_src"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    unm_src = np.nan_to_num(result["score_unmatched_src"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    exp_tgt = np.nan_to_num(result["score_expected_cost_tgt"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    unm_tgt = np.nan_to_num(result["score_unmatched_tgt"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    exp_src_n = normalize01(exp_src)
    unm_src_n = normalize01(unm_src)
    exp_tgt_n = normalize01(exp_tgt)
    unm_tgt_n = normalize01(unm_tgt)

    scores_for_map_src = np.maximum(exp_src_n, unm_src_n)
    scores_for_map_tgt = np.maximum(exp_tgt_n, unm_tgt_n)
    score_map_src = rasterize_object_scores(
        objects=objects1,
        scores=scores_for_map_src,
        image_shape=img1.shape[:2],
        fill_value=params["fill_value"],
    )
    score_map_tgt = rasterize_object_scores(
        objects=objects2,
        scores=scores_for_map_tgt,
        image_shape=img1.shape[:2],
        fill_value=params["fill_value"],
    )
    score_map = np.fmax(score_map_src, score_map_tgt)
    save_npy(SCORE_MAP_PATH, score_map)

    t8 = time.perf_counter()
    print(f"[TIME] Rasterize + save: {t8 - t7:.3f}s")
    print(f"[TIME] Total runtime: {t8 - t0:.3f}s")

    # 11. Visualize
    plt.figure(figsize=(8, 8))
    plt.imshow(score_map, cmap="YlOrRd")
    plt.colorbar()
    plt.title("Object-based Change Heatmap")
    plt.axis("off")
    plt.show()

    show_object_score_overlay(
        image=img1,
        score_map=score_map,
        title="Object-based Change Overlay",
        cmap="YlOrRd",
        alpha=0.55,
    )


if __name__ == "__main__":
    main()