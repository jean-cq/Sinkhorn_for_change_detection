from __future__ import annotations
import numpy as np
from PIL import Image

def apply_joint_zero_filter_to_tiles(
    tile1: np.ndarray,
    tile2: np.ndarray,
    *,
    zero_threshold: float = 0.6,
    window_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply patch-style joint zero filtering to a pair of tiles before SAM.

    For each pixel location, compute the local fraction of all-zero RGB pixels
    in both tiles, and keep the location only if both are below threshold.
    """
    if tile1.shape != tile2.shape:
        raise ValueError(f"Tile shape mismatch: {tile1.shape} vs {tile2.shape}")

    import cv2

    z1 = np.all(tile1 == 0, axis=-1).astype(np.float32)
    z2 = np.all(tile2 == 0, axis=-1).astype(np.float32)

    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)

    frac_zero_1 = cv2.filter2D(z1, -1, kernel, borderType=cv2.BORDER_REFLECT)
    frac_zero_2 = cv2.filter2D(z2, -1, kernel, borderType=cv2.BORDER_REFLECT)

    keep = (frac_zero_1 < zero_threshold) & (frac_zero_2 < zero_threshold)

    tile1c = tile1.copy()
    tile2c = tile2.copy()

    tile1c[~keep] = 0
    tile2c[~keep] = 0

    return tile1c, tile2c

def normalize_rgb_for_sam(
    image: np.ndarray,
    lower_pct: float = 2.0,
    upper_pct: float = 98.0,
) -> np.ndarray:
    """
    Normalize an RGB image into uint8 display-style format for SAM.

    Args:
        image: (H, W, 3) image array, possibly float with NaN values.
        lower_pct: Lower percentile for contrast stretching.
        upper_pct: Upper percentile for contrast stretching.

    Returns:
        (H, W, 3) uint8 image in [0, 255].
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

    out = (255.0 * out).astype(np.uint8)
    return out


def generate_tiles(
    image_shape: tuple[int, int],
    tile_size: int = 512,
    overlap: int = 64,
) -> list[tuple[int, int, int, int]]:
    """
    Generate overlapping tile windows.

    Args:
        image_shape: (H, W)
        tile_size: Tile side length in pixels.
        overlap: Overlap between adjacent tiles.

    Returns:
        List of windows (y0, y1, x0, x1).
    """
    H, W = image_shape
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_size must be larger than overlap")

    windows = []

    y_starts = list(range(0, max(H - tile_size + 1, 1), stride))
    x_starts = list(range(0, max(W - tile_size + 1, 1), stride))

    if not y_starts or y_starts[-1] != max(H - tile_size, 0):
        y_starts.append(max(H - tile_size, 0))
    if not x_starts or x_starts[-1] != max(W - tile_size, 0):
        x_starts.append(max(W - tile_size, 0))

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            windows.append((y0, y1, x0, x1))

    return windows


def run_sam_on_tile(
    tile_rgb: np.ndarray,
    sam_generator,
    points_per_batch: int = 32,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.7,
    mask_threshold: float = 0.0,
) -> list[dict]:
    """
    Run SAM on one RGB tile.

    Args:
        tile_rgb: (h, w, 3) uint8 RGB tile.
        sam_generator: Hugging Face SAM mask-generation pipeline.
        points_per_batch: SAM inference parameter.
        pred_iou_thresh: Minimum predicted IoU threshold.
        stability_score_thresh: Minimum stability threshold.
        mask_threshold: Threshold for binarizing predicted masks.

    Returns:
        List of raw mask records in tile-local coordinates.
    """
    pil_img = Image.fromarray(tile_rgb)

    outputs = sam_generator(
        pil_img,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        mask_threshold=mask_threshold,
    )

    masks = outputs["masks"]
    scores = outputs.get("scores", None)
    bboxes = outputs.get("bounding_boxes", None)

    records = []
    for i, mask in enumerate(masks):
        seg = np.asarray(mask, dtype=bool)

        rec = {
            "segmentation": seg,
            "area": int(seg.sum()),
        }

        if scores is not None:
            rec["predicted_iou"] = float(scores[i])

        if bboxes is not None:
            rec["bbox"] = bboxes[i]

        records.append(rec)

    return records


def shift_mask_to_global(
    local_mask: np.ndarray,
    y0: int,
    x0: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Place a tile-local mask back into full-image coordinates.

    Args:
        local_mask: (h, w) boolean mask in tile coordinates.
        y0, x0: Top-left tile origin in the full image.
        image_shape: (H, W) of the full image.

    Returns:
        (H, W) boolean mask in global image coordinates.
    """
    H, W = image_shape
    h, w = local_mask.shape

    global_mask = np.zeros((H, W), dtype=bool)
    global_mask[y0:y0 + h, x0:x0 + w] = local_mask
    return global_mask


def shift_bbox_to_global(bbox, y0: int, x0: int):
    """
    Shift a tile-local bbox into full-image coordinates.

    Assumes bbox format [x_min, y_min, x_max, y_max].
    """
    if bbox is None:
        return None

    x_min, y_min, x_max, y_max = bbox
    return [x_min + x0, y_min + y0, x_max + x0, y_max + y0]
def run_sam_segmentation_tiled_joint(
    image1: np.ndarray,
    image2: np.ndarray,
    sam_generator,
    tile_size: int = 512,
    overlap: int = 64,
    points_per_batch: int = 32,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.7,
    mask_threshold: float = 0.0,
    zero_threshold: float = 0.6,
    window_size: int = 16,
) -> tuple[list[dict], list[dict]]:
    """
    Run SAM tile-by-tile on two images.

    Pipeline:
      1. crop raw tiles from both dates
      2. apply joint zero-based filtering on raw tiles
      3. normalize the cleaned tiles for SAM
      4. run SAM on each cleaned tile

    Returns:
        raw_masks1, raw_masks2
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Shape mismatch: {image1.shape} vs {image2.shape}")

    # make sure raw images are cleaned but NOT normalized yet
    image1 = np.nan_to_num(image1, nan=0.0, posinf=0.0, neginf=0.0)
    image2 = np.nan_to_num(image2, nan=0.0, posinf=0.0, neginf=0.0)

    H, W = image1.shape[:2]
    windows = generate_tiles((H, W), tile_size=tile_size, overlap=overlap)

    all_records1 = []
    all_records2 = []

    for tile_id, (y0, y1, x0, x1) in enumerate(windows):
        # 1) crop RAW tiles first
        raw_tile1 = image1[y0:y1, x0:x1].copy()
        raw_tile2 = image2[y0:y1, x0:x1].copy()

        # 2) apply joint filtering on RAW tiles
        raw_tile1, raw_tile2 = apply_joint_zero_filter_to_tiles(
            raw_tile1,
            raw_tile2,
            zero_threshold=zero_threshold,
            window_size=window_size,
        )

        # 3) only now normalize for SAM
        tile1 = normalize_rgb_for_sam(raw_tile1)
        tile2 = normalize_rgb_for_sam(raw_tile2)

        # optional: skip tiles that are mostly invalid after filtering
        if np.all(tile1 == 0) and np.all(tile2 == 0):
            continue

        # 4) run SAM
        tile_masks1 = run_sam_on_tile(
            tile1,
            sam_generator,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            mask_threshold=mask_threshold,
        )

        tile_masks2 = run_sam_on_tile(
            tile2,
            sam_generator,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            mask_threshold=mask_threshold,
        )

        for rec in tile_masks1:
            local_mask = rec["segmentation"]
            new_rec = {
                "segmentation": local_mask,
                "area": int(local_mask.sum()),
                "tile_id": tile_id,
                "tile_window": (y0, y1, x0, x1),
                "offset": (y0, x0),
                "image_shape": (H, W),
            }
            if "predicted_iou" in rec:
                new_rec["predicted_iou"] = rec["predicted_iou"]
            if "bbox" in rec:
                new_rec["bbox"] = shift_bbox_to_global(rec["bbox"], y0, x0)
            all_records1.append(new_rec)

        for rec in tile_masks2:
            local_mask = rec["segmentation"]
            new_rec = {
                "segmentation": local_mask,
                "area": int(local_mask.sum()),
                "tile_id": tile_id,
                "tile_window": (y0, y1, x0, x1),
                "offset": (y0, x0),
                "image_shape": (H, W),
            }
            if "predicted_iou" in rec:
                new_rec["predicted_iou"] = rec["predicted_iou"]
            if "bbox" in rec:
                new_rec["bbox"] = shift_bbox_to_global(rec["bbox"], y0, x0)
            all_records2.append(new_rec)

    return all_records1, all_records2

def run_sam_segmentation_tiled(
    image: np.ndarray,
    sam_generator,
    tile_size: int = 512,
    overlap: int = 64,
    points_per_batch: int = 32,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.7,
    mask_threshold: float = 0.0,
) -> list[dict]:
    """
    Run SAM automatic mask generation tile-by-tile on an RGB image.

    Returns:
        List of raw SAM mask dictionaries stored in tile-local coordinates.
    """
    img_uint8 = normalize_rgb_for_sam(image)
    H, W = img_uint8.shape[:2]

    windows = generate_tiles((H, W), tile_size=tile_size, overlap=overlap)
    all_records = []

    for tile_id, (y0, y1, x0, x1) in enumerate(windows):
        tile = img_uint8[y0:y1, x0:x1]

        tile_masks = run_sam_on_tile(
            tile,
            sam_generator,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            mask_threshold=mask_threshold,
        )

        for rec in tile_masks:
            local_mask = rec["segmentation"]

            new_rec = {
                "segmentation": local_mask,   # keep local, not global
                "area": int(local_mask.sum()),
                "tile_id": tile_id,
                "tile_window": (y0, y1, x0, x1),
                "offset": (y0, x0),
                "image_shape": (H, W),
            }

            if "predicted_iou" in rec:
                new_rec["predicted_iou"] = rec["predicted_iou"]

            if "bbox" in rec:
                new_rec["bbox"] = shift_bbox_to_global(rec["bbox"], y0, x0)

            all_records.append(new_rec)

    return all_records

# def run_sam_segmentation(
#     image: np.ndarray,
#     sam_generator,
#     points_per_batch: int = 32,
#     pred_iou_thresh: float = 0.70,
#     stability_score_thresh: float = 0.70,
#     mask_threshold: float = 0.0,
# ):
#     """
#     Run SAM automatic mask generation on an RGB image using a Hugging Face
#     mask-generation pipeline.
#
#     Args:
#         image: (H, W, 3) numpy array representing an RGB image.
#         sam_generator: Hugging Face SAM mask-generation pipeline object.
#         points_per_batch: Number of point prompts processed per batch.
#         pred_iou_thresh: Minimum predicted IoU score required to keep a mask.
#         stability_score_thresh: Minimum stability score required to keep a mask.
#         mask_threshold: Threshold used to binarize predicted masks.
#
#     Returns:
#         List of raw SAM mask dictionaries in pixel coordinates.
#     """
#     if image.ndim != 3 or image.shape[2] != 3:
#         raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
#
#     img = image.astype(np.float32)
#     img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
#
#     if image.dtype != np.uint8:
#         if img.max() <= 1.0:
#             img = (255.0 * img).clip(0, 255).astype(np.uint8)
#         else:
#             img = img.clip(0, 255).astype(np.uint8)
#     else:
#         img = img.astype(np.uint8)
#
#     pil_img = Image.fromarray(img)
#
#     outputs = sam_generator(
#         pil_img,
#         points_per_batch=points_per_batch,
#         pred_iou_thresh=pred_iou_thresh,
#         stability_score_thresh=stability_score_thresh,
#         mask_threshold=mask_threshold,
#     )
#
#     masks = outputs["masks"]
#     scores = outputs.get("scores", None)
#     bboxes = outputs.get("bounding_boxes", None)
#
#     records = []
#     for i, mask in enumerate(masks):
#         seg = np.asarray(mask, dtype=bool)
#         rec = {
#             "segmentation": seg,
#             "area": int(seg.sum()),
#         }
#         if scores is not None:
#             rec["predicted_iou"] = float(scores[i])
#         if bboxes is not None:
#             rec["bbox"] = bboxes[i]
#         records.append(rec)
#
#     return records