import numpy as np
import rasterio
from rasterio.transform import from_origin

H, W = 256, 256

def make_blank():
    # shape: (bands, height, width)
    return np.zeros((3, H, W), dtype=np.uint8)

def draw_rect(img, top, left, height, width, color):
    r, g, b = color
    img[0, top:top+height, left:left+width] = r
    img[1, top:top+height, left:left+width] = g
    img[2, top:top+height, left:left+width] = b

# --- Image 1 ---
img1 = make_blank()
draw_rect(img1, 40, 40, 50, 50, (255, 0, 0))      # red square
draw_rect(img1, 50, 140, 60, 60, (0, 255, 0))     # green square
draw_rect(img1, 160, 100, 50, 50, (0, 0, 255))    # blue square

# --- Image 2 ---
img2 = make_blank()
draw_rect(img2, 40, 55, 50, 50, (255, 0, 0))      # red shifted right
# green removed
draw_rect(img2, 160, 100, 50, 50, (0, 255, 255))  # blue -> cyan
draw_rect(img2, 180, 20, 40, 40, (255, 255, 0))   # new yellow object

transform = from_origin(0, 256, 1, 1)  # dummy georeferencing

profile = {
    "driver": "GTiff",
    "height": H,
    "width": W,
    "count": 3,
    "dtype": "uint8",
    "crs": "EPSG:4326",
    "transform": transform
}

with rasterio.open("data/ideal/ideal_image_1.tif", "w", **profile) as dst:
    dst.write(img1)

with rasterio.open("data/ideal/ideal_image_2.tif", "w", **profile) as dst:
    dst.write(img2)

print("Saved ideal_image_1.tif and ideal_image_2.tif")