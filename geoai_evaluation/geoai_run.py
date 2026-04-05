from pathlib import Path
import numpy as np
import rasterio
from geoai.change_detection import ChangeDetection
from geoai_evaluation.geoai_prepare_refs import prepare_geoai_references
from utils.auto_params import choose_patch_size


def run_geoai_change_detection(
    image_t1: str,
    image_t2: str,
    out_dir: str = None,
    sam_model_type: str = "vit_h",
    patch_size: int | None = None,
    target_patches: int = 1024,
    prepare_refs: bool = True,
):
    name1 = Path(image_t1).stem
    name2 = Path(image_t2).stem
    pair_name = f"{name1}__VS__{name2}"

    if out_dir is None:
        out_dir = f"../data/output/geoai_ref_{pair_name}"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # auto choose patch size if not provided
    if patch_size is None:
        with rasterio.open(image_t1) as src:
            H, W = src.height, src.width
        patch_size = choose_patch_size(H, W, target_patches=target_patches)
        print(f"[INFO] Auto-selected patch_size={patch_size} for image size ({H}, {W})")
    else:
        with rasterio.open(image_t1) as src:
            H, W = src.height, src.width
        print(f"[INFO] Using provided patch_size={patch_size} for image size ({H}, {W})")

    detector = ChangeDetection(sam_model_type=sam_model_type)

    detector.set_hyperparameters(
        change_confidence_threshold=170,
        use_normalized_feature=True,
        bitemporal_match=True,
    )
    detector.set_mask_generator_params(
        points_per_side=64,
        stability_score_thresh=0.97,
    )

    binary_mask_path = out_dir / "binary_mask.tif"
    probability_mask_path = out_dir / "probability_mask.tif"
    instance_masks_path = out_dir / "instance_masks.tif"

    detector.detect_changes(
        image_t1,
        image_t2,
        output_path=str(binary_mask_path),
        export_probability=True,
        probability_output_path=str(probability_mask_path),
        export_instance_masks=True,
        instance_masks_output_path=str(instance_masks_path),
        return_detailed_results=False,
        return_results=False,
    )

    with rasterio.open(probability_mask_path) as src:
        prob = src.read(1).astype(np.float32)
    np.save(out_dir / "geoai_probability.npy", prob)

    if prepare_refs:
        prepare_geoai_references(
            rgb_image_path=image_t1,
            binary_mask_path=str(binary_mask_path),
            probability_mask_path=str(probability_mask_path),
            instance_masks_path=str(instance_masks_path),
            patch_size=patch_size,
            out_dir=str(out_dir),
        )

    return {
        "binary_mask": str(binary_mask_path),
        "probability_mask": str(probability_mask_path),
        "instance_masks": str(instance_masks_path),
        "geoai_probability_npy": str(out_dir / "geoai_probability.npy"),
        "out_dir": str(out_dir),
        "patch_size": patch_size,
    }


PATH1 = "../data/ideal/ideal_image_1.tif"
PATH2 = "../data/ideal/ideal_image_2.tif"
# PATH1 = "../data/raw/NUS_S2_RGB_2020_MayJul_small.tif"
# PATH2 = "../data/raw/NUS_S2_RGB_2025_MayJul_small.tif"

if __name__ == "__main__":
    outputs = run_geoai_change_detection(
        image_t1=PATH1,
        image_t2=PATH2,
        patch_size=None,      # auto choose
        target_patches=1024,
        prepare_refs=True,
    )
    print(outputs)