import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import box, Point
import random
import matplotlib.pyplot as plt

# === Load NDVI raster ===
ndvi_path = 'juliflora_ndvi_bboxfix.tif'
raster = rasterio.open(ndvi_path)
ndvi_array = raster.read(1)
raster_crs = raster.crs
raster_transform = raster.transform

# === Load and reproject GeoJSON ===
geojson_path = 'juliflora_polygons.geojson'
gdf = gpd.read_file(geojson_path).to_crs(raster_crs)

# === Generate binary mask ===
mask = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=raster.shape,
    transform=raster_transform,
    fill=0,
    dtype='uint8'
)

# === Sample prompt points ===

def sample_points_from_mask(mask, num_points, value=1):
    ys, xs = np.where(mask == value)
    indices = list(zip(xs, ys))
    if len(indices) == 0:
        return []
    sampled = random.sample(indices, min(num_points, len(indices)))
    # Convert pixel coords to world coords
    return [raster.transform * (x, y) for x, y in sampled]

# Sample inside polygons (positive prompts)
positive_prompts = sample_points_from_mask(mask, num_points=10, value=1)

# Sample outside (negative prompts) where NDVI is valid and not in mask
outside_mask = (mask == 0) & np.isfinite(ndvi_array)
negative_prompts = sample_points_from_mask(outside_mask.astype(np.uint8), num_points=10, value=1)

# === Visualize ===
plt.figure(figsize=(10, 10))
plt.imshow(ndvi_array, cmap='gray')
plt.imshow(mask, cmap='Reds', alpha=0.4)

# Plot prompts (now in image coordinates, y-down)
if positive_prompts:
    xs, ys = zip(*[raster.index(*pt) for pt in positive_prompts])
    plt.scatter(xs, ys, c='lime', label='Positive Prompts', edgecolors='black')

if negative_prompts:
    xs, ys = zip(*[raster.index(*pt) for pt in negative_prompts])
    plt.scatter(xs, ys, c='blue', label='Negative Prompts', edgecolors='white')

plt.gca().invert_yaxis()  # <- IMPORTANT: Aligns with raster (y-down)


plt.title('Binary Mask and SAM Prompts')
plt.legend()
plt.show()
