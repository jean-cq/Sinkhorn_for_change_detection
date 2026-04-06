import numpy as np

def choose_patch_size(H, W, target_patches=1024, allowed_sizes=(4, 8, 16, 32, 64, 128)):
    total_pixels = H * W
    patch_area = total_pixels / target_patches
    ideal_patch = np.sqrt(patch_area)

    best = allowed_sizes[0]
    for p in allowed_sizes:
        if abs(p - ideal_patch) < abs(best - ideal_patch):
            best = p
    return best

def detect_image_mode(img: np.ndarray) -> str:
    frac_zero = np.mean(np.all(img == 0, axis=-1))
    if frac_zero > 0.7:
        return "ideal"
    return "satellite"

def choose_params(img: np.ndarray):
    H, W = img.shape[:2]
    mode = detect_image_mode(img)

    if mode == "ideal":
        return {
            "mode": "ideal",
            "tile_size": None,
            "overlap": 0,
            "points_per_batch": 16,
            "pred_iou_thresh": 0.70,
            "stability_score_thresh": 0.70,
            "zero_threshold": 1.01,
            "window_size": 1,
            "min_area": max(4, int(0.0003 * H * W)),
            "max_area_ratio": 0.10,
            "iou_threshold": 0.90,
            "bbox_iou_threshold": 0.50,
            "max_masks_after_sort": 200,
            "max_zero_fraction": 0.2,
            "patch_size": choose_patch_size(H, W),
            "fill_value": 0,
        }

    # satellite
    tile_size = 512 if max(H, W) > 1024 else None
    overlap = 64 if tile_size is not None else 0

    return {
        "mode": "satellite",
        "tile_size": tile_size,
        "overlap": overlap,
        "points_per_batch": 32,
        "pred_iou_thresh": 0.85,
        "stability_score_thresh": 0.85,
        "zero_threshold": 0.60,
        "window_size": 16,
        "min_area": max(80, int(0.00005 * H * W)),
        "max_area_ratio": 0.10,
        "iou_threshold": 0.85,
        "bbox_iou_threshold": 0.20,
        "max_masks_after_sort": 5000,
        "max_zero_fraction": 0.60,
        "patch_size": choose_patch_size(H, W),
        "fill_value":np.nan,
    }