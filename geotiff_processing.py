import numpy as np
import rasterio

def read_geotiff_rgb(path: str) -> tuple[np.ndarray, dict]:
    """
    Returns:
      img: (H, W, 3) float32
      meta: rasterio metadata (includes transform, crs, etc.)
    Assumes bands are RGB (3 bands).
    """
    with rasterio.open(path) as src:
        # src.read() gives (bands, H, W)
        arr = src.read(out_dtype=np.float32)
        meta = src.meta.copy()

    if arr.shape[0] < 3:
        raise ValueError(f"Expected >=3 bands, got {arr.shape[0]}")

    # Take first 3 bands as RGB
    arr = arr[:3]                 # (3,H,W)
    img = np.transpose(arr, (1, 2, 0))  # (H,W,3)

    return img, meta