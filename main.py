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

from utils.rasterize import rasterize_object_scores
from utils.cache_io import save_npy, save_pickle, load_pickle

from sinkhorn import sinkhorn_object_change
import torch
from transformers import pipeline


USE_CACHE = True

PATH1 = "data/raw/NUS_S2_RGB_2020_MayJul_small.tif"
PATH2 = "data/raw/NUS_S2_RGB_2025_MayJul_small.tif"

MASK_CACHE_1 = "data/cache/raw_masks_2020_small.pkl"
MASK_CACHE_2 = "data/cache/raw_masks_2025_small.pkl"
OBJ_CACHE_1 = "data/cache/objects_2020_small.pkl"
OBJ_CACHE_2 = "data/cache/objects_2025_small.pkl"
SCORE_MAP_PATH = "data/output/object_score_map_small_betahalf.npy"


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

#
# def get_masks(image: np.ndarray, cache_path: str, sam_generator):
#     if USE_CACHE:
#         try:
#             masks = load_pickle(cache_path)
#             print(f"[INFO] Loaded cached masks from {cache_path}")
#             return masks
#         except FileNotFoundError:
#             pass
#
#     masks = run_sam_segmentation(image, sam_generator)
#
#     if USE_CACHE:
#         save_pickle(cache_path, masks)
#         print(f"[INFO] Saved masks to {cache_path}")
#
#     return masks

def get_masks_tiled(image: np.ndarray, cache_path: str, sam_generator):
    if USE_CACHE:
        try:
            masks = load_pickle(cache_path)
            print(f"[INFO] Loaded cached masks from {cache_path}")
            return masks
        except FileNotFoundError:
            pass

    masks = run_sam_segmentation_tiled(
        image,
        sam_generator,
        tile_size=512,
        overlap=64,
        points_per_batch=32,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.85,
        mask_threshold=0.0,
    )

    if USE_CACHE:
        save_pickle(cache_path, masks)
        print(f"[INFO] Saved masks to {cache_path}")

    return masks
def get_masks_tiled_joint(
    img1,
    img2,
    cache1,
    cache2,
    sam_generator,
    use_cache=True,
):
    """
    Joint tiled SAM with shared preprocessing + caching.
    """

    if use_cache:
        try:
            import pickle
            with open(cache1, "rb") as f:
                masks1 = pickle.load(f)
            with open(cache2, "rb") as f:
                masks2 = pickle.load(f)

            print(f"[INFO] Loaded cached masks from {cache1}, {cache2}")
            return masks1, masks2

        except FileNotFoundError:
            pass

    # 🔥 run joint SAM
    masks1, masks2 = run_sam_segmentation_tiled_joint(
        img1,
        img2,
        sam_generator,
        tile_size=512,
        overlap=64,
        points_per_batch=32,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.85,
        zero_threshold=0.6,
        window_size=16,
    )

    save_pickle(cache1, masks1)
    save_pickle(cache2, masks2)

    print(f"[INFO] Saved masks to {cache1}, {cache2}")

    return masks1, masks2

def main():
    t0 = time.perf_counter()

    # 1. Read images
    img1, meta1 = read_geotiff_rgb(PATH1)
    img2, meta2 = read_geotiff_rgb(PATH2)

    img1 = np.nan_to_num(img1, nan=0.0)
    img2 = np.nan_to_num(img2, nan=0.0)

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    t1 = time.perf_counter()
    print(f"[TIME] Read GeoTIFFs: {t1 - t0:.3f}s")
    print(f"[INFO] Image shape: {img1.shape}")

    # 2. Build SAM generator
    sam_generator = build_sam_generator()

    # 3. Run SAM
    # raw_masks1 = get_masks_tiled(img1, MASK_CACHE_1, sam_generator)
    # raw_masks2 = get_masks_tiled(img2, MASK_CACHE_2, sam_generator)
    raw_masks1, raw_masks2 = get_masks_tiled_joint(
        img1,
        img2,
        MASK_CACHE_1,
        MASK_CACHE_2,
        sam_generator,
    )
    t2 = time.perf_counter()
    print(f"[TIME] SAM segmentation: {t2 - t1:.3f}s")
    print(f"[INFO] Raw mask count T1: {len(raw_masks1)}")
    print(f"[INFO] Raw mask count T2: {len(raw_masks2)}")

    # 4. Postprocess masks
    masks1 = postprocess_sam_masks(
        raw_masks1,
        image_shape=img1.shape[:2],
        min_area=80,
        max_area_ratio=0.10,
        iou_threshold=0.85,
        bbox_iou_threshold=0.2,
        max_masks_after_sort=5000,
    )

    masks2 = postprocess_sam_masks(
        raw_masks2,
        image_shape=img2.shape[:2],
        min_area=80,
        max_area_ratio=0.10,
        iou_threshold=0.85,
        bbox_iou_threshold=0.2,
        max_masks_after_sort=5000,
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
        min_area=100.0,
        max_zero_fraction=0.6,
        min_sam_score=0.0,
    )
    objects2 = filter_objects(
        img2,
        objects2,
        min_area=100.0,
        max_zero_fraction=0.6,
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
    print("[DEBUG] Any NaN in XY1?", np.isnan(XY1).any())
    print("[DEBUG] Any NaN in F1?", np.isnan(F1).any())
    print("[DEBUG] Any NaN in S1?", np.isnan(S1).any())
    print("[DEBUG] Any NaN in XY2?", np.isnan(XY2).any())
    print("[DEBUG] Any NaN in F2?", np.isnan(F2).any())
    print("[DEBUG] Any NaN in S2?", np.isnan(S2).any())
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
        eps=0.0001,
        tau_a=0.5,
        tau_b=0.5,
        n_iters=500,
        tol=1e-6,
    )
    print("[DEBUG] Any NaN in cost matrix?", np.isnan(result["C"]).any())
    print("[DEBUG] score_expected_cost:", result["score_expected_cost"])

    t7 = time.perf_counter()
    print(f"[TIME] Sinkhorn OT: {t7 - t6:.3f}s")
    print(f"[INFO] Total OT cost: {result['ot_cost']:.6f}")
    print(f"[INFO] Cost matrix shape: {result['C'].shape}")

    # 10. Rasterize scores
    # score_expected_cost is row-wise, so it corresponds to source objects1
    raw_scores = result["score_expected_cost"].astype(np.float32)
    scores_for_map = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
    # log compression
    # scores_for_map = np.log1p(np.maximum(raw_scores, 0.0))

    # sqrt compression
    # scores_for_map = np.sqrt(np.maximum(raw_scores, 0.0))
    # areas = np.array([obj["area"] for obj in objects1], dtype=np.float32)

    # scores_for_map = raw_scores / np.sqrt(np.maximum(areas, 1.0))

    score_map = rasterize_object_scores(
        objects=objects1,
        scores=scores_for_map,
        image_shape=img1.shape[:2],
        fill_value=np.nan,
    )

    save_npy(SCORE_MAP_PATH, score_map)

    t8 = time.perf_counter()
    print(f"[TIME] Rasterize + save: {t8 - t7:.3f}s")
    print(f"[TIME] Total runtime: {t8 - t0:.3f}s")

    # 11. Visualize
    plt.figure(figsize=(8, 8))
    plt.imshow(score_map, cmap="hot")
    plt.colorbar()
    plt.title("Object-based Change Heatmap")
    plt.axis("off")
    plt.show()

    show_object_score_overlay(
        image=img1,
        score_map=score_map,
        title="Object-based Change Overlay",
        cmap="hot",
        alpha=0.55,
    )


if __name__ == "__main__":
    main()