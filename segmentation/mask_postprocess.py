from __future__ import annotations
import numpy as np

# def mask_area(mask: np.ndarray) -> int:
#     return int(np.count_nonzero(mask))
#
#
# def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
#     inter = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
#     return float(inter / union) if union > 0 else 0.0
#
#
# def filter_masks_by_area(
#     masks: list[dict],
#     image_shape: tuple[int, int],
#     min_area: int = 100,
#     max_area_ratio: float = 0.5,
# ) -> list[dict]:
#     """
#     Remove masks that are too small or too large.
#     """
#     H, W = image_shape
#     max_area = H * W * max_area_ratio
#
#     kept = []
#     for m in masks:
#         seg = m["segmentation"]
#         area = mask_area(seg)
#         if area < min_area:
#             continue
#         if area > max_area:
#             continue
#         kept.append(m)
#     return kept
#
#
# def remove_duplicate_masks(
#     masks: list[dict],
#     iou_threshold: float = 0.9,
# ) -> list[dict]:
#     """
#     Remove near-duplicate masks using IoU.
#     Keep the larger one first if masks are sorted by area descending.
#     """
#     if not masks:
#         return []
#
#     masks_sorted = sorted(
#         masks,
#         key=lambda x: x.get("area", np.count_nonzero(x["segmentation"])),
#         reverse=True,
#     )
#
#     kept = []
#     for m in masks_sorted:
#         seg = m["segmentation"]
#         duplicate = False
#         for km in kept:
#             iou = compute_iou(seg, km["segmentation"])
#             if iou >= iou_threshold:
#                 duplicate = True
#                 break
#         if not duplicate:
#             kept.append(m)
#
#     return kept
#
#
# def postprocess_sam_masks(
#     masks: list[dict],
#     image_shape: tuple[int, int],
#     min_area: int = 100,
#     max_area_ratio: float = 0.5,
#     iou_threshold: float = 0.9,
# ) -> list[dict]:
#     """
#     Full postprocessing for SAM masks.
#     """
#     masks = filter_masks_by_area(
#         masks,
#         image_shape=image_shape,
#         min_area=min_area,
#         max_area_ratio=max_area_ratio,
#     )
#     masks = remove_duplicate_masks(masks, iou_threshold=iou_threshold)
#     return masks

def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def bbox_iou(box1, box2) -> float:
    """
    IoU for boxes in [x_min, y_min, x_max, y_max].
    """
    if box1 is None or box2 is None:
        return 0.0

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    iw = max(0, inter_xmax - inter_xmin + 1)
    ih = max(0, inter_ymax - inter_ymin + 1)
    inter = iw * ih

    area1 = max(0, x1_max - x1_min + 1) * max(0, y1_max - y1_min + 1)
    area2 = max(0, x2_max - x2_min + 1) * max(0, y2_max - y2_min + 1)

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_mask_iou_local(m1: dict, m2: dict) -> float:
    """
    IoU for local masks with offsets.
    Only computes over the overlapping region in global coordinates.
    """
    mask1 = m1["segmentation"]
    mask2 = m2["segmentation"]

    y01, x01 = m1.get("offset", (0, 0))
    y02, x02 = m2.get("offset", (0, 0))

    h1, w1 = mask1.shape
    h2, w2 = mask2.shape

    y11, x11 = y01 + h1, x01 + w1
    y12, x12 = y02 + h2, x02 + w2

    oy0 = max(y01, y02)
    ox0 = max(x01, x02)
    oy1 = min(y11, y12)
    ox1 = min(x11, x12)

    if oy1 <= oy0 or ox1 <= ox0:
        return 0.0

    a1 = mask1[oy0 - y01 : oy1 - y01, ox0 - x01 : ox1 - x01]
    a2 = mask2[oy0 - y02 : oy1 - y02, ox0 - x02 : ox1 - x02]

    inter = np.logical_and(a1, a2).sum()

    area1 = m1.get("area", int(mask1.sum()))
    area2 = m2.get("area", int(mask2.sum()))
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def filter_masks_by_area(
    masks: list[dict],
    image_shape: tuple[int, int],
    min_area: int = 50,
    max_area_ratio: float = 0.25,
) -> list[dict]:
    H, W = image_shape
    max_area = H * W * max_area_ratio

    kept = []
    for m in masks:
        area = m.get("area", np.count_nonzero(m["segmentation"]))
        if area < min_area:
            continue
        if area > max_area:
            continue
        kept.append(m)
    return kept


def remove_duplicate_masks_fast(
    masks: list[dict],
    iou_threshold: float = 0.85,
    bbox_iou_threshold: float = 0.2,
    max_masks_after_sort: int = 1500,
) -> list[dict]:
    """
    Faster duplicate removal:
    - sort by score and area
    - keep only top masks
    - compare bbox first
    - compute real IoU only when bbox overlap is meaningful
    """
    if not masks:
        return []

    masks_sorted = sorted(
        masks,
        key=lambda x: (x.get("predicted_iou", 0.0), x.get("area", 0)),
        reverse=True,
    )

    if max_masks_after_sort is not None:
        masks_sorted = masks_sorted[:max_masks_after_sort]

    kept = []

    for m in masks_sorted:
        duplicate = False
        box_m = m.get("bbox", None)

        for km in kept:
            box_k = km.get("bbox", None)

            # cheap reject
            if bbox_iou(box_m, box_k) < bbox_iou_threshold:
                continue

            # expensive check only if bbox overlap is nontrivial
            iou = compute_mask_iou_local(m, km)
            if iou >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(m)

    return kept


def postprocess_sam_masks(
    masks: list[dict],
    image_shape: tuple[int, int],
    min_area: int = 50,
    max_area_ratio: float = 0.25,
    iou_threshold: float = 0.85,
    bbox_iou_threshold: float = 0.2,
    max_masks_after_sort: int = 1500,
) -> list[dict]:
    masks = filter_masks_by_area(
        masks,
        image_shape=image_shape,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
    )

    masks = remove_duplicate_masks_fast(
        masks,
        iou_threshold=iou_threshold,
        bbox_iou_threshold=bbox_iou_threshold,
        max_masks_after_sort=max_masks_after_sort,
    )

    return masks