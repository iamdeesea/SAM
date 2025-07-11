import os
import torch
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import box, mapping
from torchvision.transforms import Resize
from PIL import Image

# === CONFIG ===
ndvi_path = "juliflora_ndvi_bboxfix.tif"
polygon_path = "juliflora_polygons.geojson"
output_dir = "sam2_training_data"
os.makedirs(output_dir, exist_ok=True)

# === LOAD NDVI ===
with rasterio.open(ndvi_path) as src:
    ndvi_data = src.read(1)
    ndvi_transform = src.transform
    ndvi_crs = src.crs
    ndvi_bounds = src.bounds
    ndvi_meta = src.meta

# Normalize NDVI (0-255 uint8)
ndvi_scaled = ((ndvi_data - np.nanmin(ndvi_data)) / (np.nanmax(ndvi_data) - np.nanmin(ndvi_data)) * 255).astype(np.uint8)
image = Image.fromarray(ndvi_scaled)

# === LOAD POLYGONS ===
gdf = gpd.read_file(polygon_path)
gdf = gdf.to_crs(ndvi_crs)

# === CREATE BINARY MASK ===
mask = np.zeros(ndvi_scaled.shape, dtype=np.uint8)
shapes = [(geom, 1) for geom in gdf.geometry if geom.is_valid]

from rasterio.features import rasterize
mask = rasterize(shapes=shapes, out_shape=mask.shape, transform=ndvi_transform, fill=0, dtype=np.uint8)

# === EXTRACT CENTER POINTS (as prompts) ===
centroids = gdf.centroid

training_samples = []

for i, point in enumerate(centroids):
    row, col = ~ndvi_transform * (point.x, point.y)
    row, col = int(row), int(col)

    if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
        sample = {
            "image": ndvi_scaled,
            "mask": mask,
            "input_point": torch.tensor([[col, row]]),
            "input_label": torch.tensor([1]),
        }
        torch.save(sample, os.path.join(output_dir, f"sample_{i:03d}.pt"))

print(f"✅ Saved {len(centroids)} training samples to {output_dir}")
