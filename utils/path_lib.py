from pathlib import Path

def stem_no_ext(path: str) -> str:
    return Path(path).stem

def build_run_names(path1: str, path2: str):
    name1 = stem_no_ext(path1)
    name2 = stem_no_ext(path2)
    pair_name = f"{name1}__VS__{name2}"

    return {
        "name1": name1,
        "name2": name2,
        "pair_name": pair_name,

        "mask_cache_1": f"data/cache/raw_masks_{name1}.pkl",
        "mask_cache_2": f"data/cache/raw_masks_{name2}.pkl",
        "obj_cache_1": f"data/cache/objects_{name1}.pkl",
        "obj_cache_2": f"data/cache/objects_{name2}.pkl",

        "object_score_map": f"data/output/object_score_map_{pair_name}.npy",

        "patch_F1": f"data/cache/F1_{name1}.npy",
        "patch_F2": f"data/cache/F2_{name2}.npy",
        "patch_XY1": f"data/cache/XY1_{name1}.npy",
        "patch_XY2": f"data/cache/XY2_{name2}.npy",
        "patch_heatmap": f"data/output/heatmap_{pair_name}.npy",

        "geoai_out_dir": f"data/output/geoai_ref_{pair_name}",
    }