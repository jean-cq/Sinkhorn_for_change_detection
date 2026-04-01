import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from patch_approach.patches import extract_patches_nonoverlap, filter_bad_patches, restore_patch_grid
from geotiff_processing import read_geotiff_rgb
from embeddings import encode_patches_torchgeo
from sinkhorn import sinkhorn_patch_change
def show_rgb(img, title="RGB"):
    # robust scaling using percentile
    p_low, p_high = np.percentile(img, (2, 98))
    img_vis = np.clip((img - p_low) / (p_high - p_low + 1e-6), 0, 1)

    plt.figure(figsize=(6,6))
    plt.imshow(img_vis)
    plt.title(title)
    plt.axis("off")
    plt.show()
def overlay_heatmap_on_rgb(img, heatmap, alpha=0.5, show=True):
    H, W, _ = img.shape

    # resize heatmap
    heat_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_NEAREST)

    # normalize (handle NaN safely)
    hmin = np.nanmin(heat_resized)
    hmax = np.nanmax(heat_resized)
    heat_norm = (heat_resized - hmin) / (hmax - hmin + 1e-6)

    heat_norm = np.nan_to_num(heat_norm, nan=0.0)
    # highlight the top changes
    threshold = np.percentile(heat_norm, 90)
    heat_norm[heat_norm < threshold] = 0

    # colormap
    cmap = plt.get_cmap("hot")
    heat_color = cmap(heat_norm)[..., :3]

    # normalize RGB
    p_low, p_high = np.percentile(img, (2, 98))
    img_vis = np.clip((img - p_low) / (p_high - p_low + 1e-6), 0, 1)

    # overlay
    overlay = (1 - alpha) * img_vis + alpha * heat_color
    overlay = np.clip(overlay, 0, 1)

    if show:
        plt.figure(figsize=(6,6))
        plt.imshow(overlay)
        plt.title("Overlay: Change on RGB")
        plt.axis("off")
        plt.show()

    return overlay

if __name__ == "__main__":
    t0 = time.perf_counter()
    img1, meta1 = read_geotiff_rgb("../data/raw/NUS_S2_RGB_2020_MayJul_small.tif")
    img2, meta2 = read_geotiff_rgb("../data/raw/NUS_S2_RGB_2025_MayJul_small.tif")
    img1 = np.nan_to_num(img1, nan=0.0)
    img2 = np.nan_to_num(img2, nan=0.0)
    #
    # print("After nan fix:")
    # print(img1.shape,img1.min(), img1.max())
    # print(img2.shape,img2.min(), img2.max())

    # show_rgb(img1, "2020")
    # show_rgb(img2, "2025")
    # Must match for patching
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    t1 = time.perf_counter()
    print(f"[TIME] Read + nan fix: {t1 - t0:.3f}s")
    PATCH_SIZE = 16
    patches1, XY1, (gh, gw) = extract_patches_nonoverlap(img1, PATCH_SIZE)
    patches2, XY2, _ = extract_patches_nonoverlap(img2, PATCH_SIZE)
    t2 = time.perf_counter()
    print(f"[TIME] Patch extraction: {t2 - t1:.3f}s")
    print(f"[INFO] Original patch count: {len(patches1)}")

    patches1, XY1, patches2, XY2, keep_mask = filter_bad_patches(
        patches1, XY1, patches2, XY2, zero_threshold=0.6
    )
    t3 = time.perf_counter()
    print(f"[TIME] Patch filtering: {t3 - t2:.3f}s")
    print(f"[INFO] Kept patch count: {len(patches1)}")

    t4 = time.perf_counter()
    USE_CACHE = True

    if USE_CACHE:
        try:
            F1 = np.load("../data/cache/F1_small.npy")
            F2 = np.load("../data/cache/F2_small.npy")
            XY1 = np.load("../data/cache/XY1_small.npy")
            XY2 = np.load("../data/cache/XY2_small.npy")
            print("[INFO] Loaded cached embeddings and coordinates.")
        except FileNotFoundError:
            # F1 = encode_patches(patches1)
            # F2 = encode_patches(patches2)
            F1 = encode_patches_torchgeo(patches1)
            F2 = encode_patches_torchgeo(patches2)

            # whole-image TorchGeo embeddings
            # F1_all = encode_image_to_patch_embeddings_torchgeo(img1, patch_size=PATCH_SIZE)
            # F2_all = encode_image_to_patch_embeddings_torchgeo(img2, patch_size=PATCH_SIZE)
            # # keep only valid patches
            # F1 = F1_all[keep_mask]
            # F2 = F2_all[keep_mask]
            np.save("../data/cache/F1.npy", F1)
            np.save("../data/cache/F2.npy", F2)
            np.save("../data/cache/XY1.npy", XY1)
            np.save("../data/cache/XY2.npy", XY2)
            print("[INFO] Computed and saved embeddings.")
    else:
        # F1 = encode_patches(patches1)
        # F2 = encode_patches(patches2)
        F1 = encode_patches_torchgeo(patches1)
        F2 = encode_patches_torchgeo(patches2)
        # F1_all = encode_image_to_patch_embeddings_torchgeo(img1, patch_size=PATCH_SIZE)
        # F2_all = encode_image_to_patch_embeddings_torchgeo(img2, patch_size=PATCH_SIZE)
        #
        # F1 = F1_all[keep_mask]
        # F2 = F2_all[keep_mask]


    result = sinkhorn_patch_change(XY1, F1, XY2, F2,
        gate_radius=0.03,
        eps=0.0005,
        tau_a=0.5,
        tau_b=0.5,
        n_iters=500,
        tol=1e-6,
    )
    t5 = time.perf_counter()
    print(f"[TIME] Sinkhorn: {t5 - t4:.3f}s")

    # heatmap = to_patch_grid(result["score_expected_cost"], gh, gw)
    heatmap = restore_patch_grid(
        result["score_expected_cost"],
        keep_mask,
        gh,
        gw,
        fill_value=np.nan
    )
    np.save("../data/output/heatmap_small_0005.npy", heatmap)

    t6 = time.perf_counter()
    print(f"[TIME] Heatmap restore/save: {t6 - t5:.3f}s")
    print(f"[TIME] Total: {t6 - t0:.3f}s")


    plt.imshow(heatmap, cmap="hot")
    plt.colorbar()
    plt.title("Patch Change Heatmap")
    plt.show()
    overlay_heatmap_on_rgb(img2, heatmap)