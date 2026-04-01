import rasterio
import matplotlib.pyplot as plt
import numpy as np

tif_path = "data/raw/NUS_S2_RGB_2025_MayJul.tif"

with rasterio.open(tif_path) as src:
    # Read first 3 bands
    img = src.read([1, 2, 3])  # shape: (3, H, W)

# Convert to (H, W, 3)
img = np.transpose(img, (1, 2, 0))

# Normalize for display
img = img.astype(float)
img = (img - img.min()) / (img.max() - img.min())

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title("RGB GeoTIFF")
plt.axis("off")
plt.show()