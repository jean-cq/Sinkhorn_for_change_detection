from pathlib import Path
import numpy as np
import rasterio
from geoai.change_detection import ChangeDetection


def run_geoai_change_detection(
    image_t1: str,
    image_t2: str,
    out_dir: str = "../data/output/geoai_ref",
    sam_model_type: str = "vit_h",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    detector.detect_changes(
        image_t1,
        image_t2,
        output_path=str(out_dir / "binary_mask.tif"),
        export_probability=True,
        probability_output_path=str(out_dir / "probability_mask.tif"),
        export_instance_masks=True,
        instance_masks_output_path=str(out_dir / "instance_masks.tif"),
        return_detailed_results=False,
        return_results=False,
    )

    with rasterio.open(out_dir / "probability_mask.tif") as src:
        prob = src.read(1).astype(np.float32)
    np.save(out_dir / "geoai_probability.npy", prob)

    return {
        "binary_mask": str(out_dir / "binary_mask.tif"),
        "probability_mask": str(out_dir / "probability_mask.tif"),
        "instance_masks": str(out_dir / "instance_masks.tif"),
        "geoai_probability_npy": str(out_dir / "geoai_probability.npy"),
    }


if __name__ == "__main__":
    outputs = run_geoai_change_detection(
        image_t1="../data/raw/NUS_S2_RGB_2020_MayJul_small.tif",
        image_t2="../data/raw/NUS_S2_RGB_2025_MayJul_small.tif",
    )
    print(outputs)