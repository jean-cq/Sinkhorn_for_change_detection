# Sinkhorn for Satellite Image Change Detection

This project investigates Sinkhorn-based optimal transport for satellite image change detection using Sentinel-2 imagery. Two representations are implemented under the same OT framework: a patch-based approach, where fixed image patches are compared using spatial coordinates and deep visual features, and an object-based approach, where SAM-generated segmentation masks are converted into polygons and compared using object-level descriptors. The repository also includes evaluation scripts for comparing both OT-based approaches against a GeoAI baseline.

## Installation

Install the required packages from the project root:

```bash
pip install -r requirements.txt
```

## Data setup

Place your input pair of GeoTIFF images in the `data/raw/` folder, or use the ideal test pair in `data/ideal/` if available.

The current version of the code uses Sentinel-2 images of the Tengah area in Singapore, obtained through Google Earth Engine using the following script:

`https://code.earthengine.google.com/?accept_repo=users/cqq67/MA4198`

If you use a different image pair, update the image paths inside the relevant Python files before running the scripts:

- `main.py` for the object-based approach
- `patch_approach/main_patch.py` for the patch-based approach
- `geoai_evaluation/geoai_run.py` for the GeoAI baseline
- `evaluation_patch.py` and `evaluation_object.py` for evaluation against the GeoAI baseline

If you use the ideal test images instead of the satellite images, some additional parameter changes are required in each approach. Please check the relevant sections below and adjust them accordingly.

---

## Run the patch-based approach

### Update input paths

In `patch_approach/main_patch.py`, change the input paths as needed.

Example:

```python
# PATH1 = "../data/ideal/ideal_image_1.tif"
# PATH2 = "../data/ideal/ideal_image_2.tif"

PATH1 = "../data/raw/Tengah_2020_MayJul.tif"
PATH2 = "../data/raw/Tengah_2025_MayJul.tif"
```

### Additional change for ideal images

If you are using the ideal test images, disable the bad-patch filtering step and keep all patches instead.

For example, replace the filtered mask logic with:

```python
keep_mask = np.ones(len(patches1), dtype=bool)
```

and comment out the `filter_bad_patches(...)` operation.

### Run

From the project root:

```bash
python patch_approach/main_patch.py
```

This runs the patch-based Sinkhorn pipeline and saves the patch-level heatmap output to `data/output/`.

The output naming convention follows:

```python
pair_name = f"{name1}__VS__{name2}"
```

For example:

```python
"patch_heatmap": f"data/output/heatmap_{pair_name}.npy"
```

---

## Run the object-based approach

### Update input paths

In `main.py`, change the input paths as needed.

Example:

```python
# PATH1 = "data/ideal/ideal_image_1.tif"
# PATH2 = "data/ideal/ideal_image_2.tif"

PATH1 = "data/raw/Tengah_2020_MayJul.tif"
PATH2 = "data/raw/Tengah_2025_MayJul.tif"
```

### Additional change for ideal images

If you are using the ideal test images, also change the entropic regularization parameter `eps` accordingly.

Example:

```python
# for ideal image test use eps=0.0001
# eps=0.0001,
eps=0.005,
```

Use the value appropriate for your input setting.

### Run

From the project root:

```bash
python main.py
```

This runs the object-based Sinkhorn pipeline with SAM-based segmentation and saves the object score map to `data/output/`.

The output naming convention follows:

```python
pair_name = f"{name1}__VS__{name2}"
```

For example:

```python
"object_score_map": f"data/output/object_score_map_{pair_name}.npy
```
---

## Run the GeoAI baseline and prepare reference outputs

Before running, update the image paths in `geoai_evaluation/geoai_run.py`.

Example:

```python
# PATH1 = "../data/ideal/ideal_image_1.tif"
# PATH2 = "../data/ideal/ideal_image_2.tif"

PATH1 = "../data/raw/Tengah_2020_MayJul.tif"
PATH2 = "../data/raw/Tengah_2025_MayJul.tif"
```

From the project root:

```bash
python geoai_evaluation/geoai_run.py
```

This generates GeoAI change detection outputs and prepares patch-level reference arrays for later evaluation.

---

## Run evaluation for the patch-based approach

Before running `evaluation_patch.py`, update the file paths so that they point to:

- the corresponding patch heatmap produced by the patch-based OT approach
- the matching `geoai_prob_patchref.npy` file in the relevant `geoai_ref` folder

Example:

```python
PATCH_OT_PATH = "data/output/heatmap_Tengah_2020_MayJul__VS__Tengah_2025_MayJul.npy"
GEOAI_PATCH_REF_PATH = "data/output/geoai_ref/geoai_prob_patchref.npy"

# PATCH_OT_PATH = "data/output/heatmap_ideal_image_1__VS__ideal_image_2.npy"
# GEOAI_PATCH_REF_PATH = "data/output/geoai_ref_ideal/geoai_prob_patchref.npy"
```

From the project root:

```bash
python evaluation_patch.py
```

This compares the patch-based OT output against the prepared GeoAI patch reference and reports Pearson correlation, IoU, and F1 scores.

---

## Run evaluation for the object-based approach

Before running `evaluation_object.py`, update the file paths so that they point to:

- the corresponding `object_score_map`
- the matching `geoai_prob_patchref.npy` file in the relevant `geoai_ref` folder

Example:

```python
OBJECT_OT_RASTER_PATH = "data/output/object_score_map_Tengah_2020_MayJul__VS__Tengah_2025_MayJul.tif"
GEOAI_PATCH_REF_PATH = "data/output/geoai_ref_Tengah_2020_MayJul__VS__Tengah_2025_MayJul/geoai_prob_patchref.npy"

# OBJECT_OT_RASTER_PATH = "data/output/object_score_map_ideal_image_1__VS__ideal_image_2.tif"
# GEOAI_PATCH_REF_PATH = "data/output/geoai_ref_ideal_image_1__VS__ideal_image_2/geoai_prob_patchref.npy"
```

Also make sure the patch size matches the input setting:

```python
# satellite patch
PATCH_SIZE = 16

# ideal patch
# PATCH_SIZE = 8
```

From the project root:

```bash
python evaluation_object.py
```

This converts the object-based raster output to patch level and compares it against the GeoAI patch reference using Pearson correlation, IoU, and F1 scores.

---

## Main files

- `main.py` – object-based Sinkhorn change detection
- `patch_approach/main_patch.py` – patch-based Sinkhorn change detection
- `evaluation_patch.py` – evaluation for patch-based OT
- `evaluation_object.py` – evaluation for object-based OT
- `geoai_evaluation/geoai_run.py` – run GeoAI baseline
- `geoai_evaluation/geoai_prepare_refs.py` – prepare GeoAI reference arrays

## Notes

- Some scripts use hardcoded file paths. Please update them before running the pipeline on a new image pair.
- The object-based approach uses SAM-based segmentation and may run much faster with GPU support.
- Output files are saved under `data/output/`.

