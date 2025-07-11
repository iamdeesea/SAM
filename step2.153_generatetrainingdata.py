import os
import json
import numpy as np
import cv2
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm

# Paths
ndvi_path = "juliflora_ndvi_bboxfix.tif"
geojson_path = "juliflora_polygons.geojson"
output_dir = "sam2_training_data"
os.makedirs(output_dir, exist_ok=True)

# Load NDVI
with rasterio.open(ndvi_path) as src:
    ndvi = src.read(1)
    ndvi_transform = src.transform
    ndvi_crs = src.crs
    ndvi_bounds = src.bounds

# Load and clean GeoJSON
gdf = gpd.read_file(geojson_path)
gdf = gdf[gdf.geometry.notnull()]
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

# Fix geometries
gdf["geometry"] = gdf["geometry"].buffer(0)

# Reproject and clip to NDVI bounds
gdf = gdf.to_crs(ndvi_crs)
bbox_geom = box(*ndvi_bounds)
gdf = gdf[gdf.intersects(bbox_geom)]

if gdf.empty:
    raise ValueError("❌ No valid geometries found after filtering and clipping.")

# Rasterize
mask = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=ndvi.shape,
    transform=ndvi_transform,
    fill=0,
    dtype=np.uint8
)

# Extract training patches
tile_size = 256
stride = 256
count = 0

for y in tqdm(range(0, ndvi.shape[0] - tile_size + 1, stride)):
    for x in range(0, ndvi.shape[1] - tile_size + 1, stride):
        ndvi_tile = ndvi[y:y+tile_size, x:x+tile_size]
        mask_tile = mask[y:y+tile_size, x:x+tile_size]

        if np.sum(mask_tile) == 0:
            continue  # skip empty masks

        image_rgb = cv2.normalize(ndvi_tile, None, 0, 255, cv2.NORM_MINMAX)
        image_rgb = cv2.merge([image_rgb]*3).astype(np.uint8)

        sample_path = os.path.join(output_dir, f"sample_{count:04d}")
        os.makedirs(sample_path, exist_ok=True)

        cv2.imwrite(os.path.join(sample_path, "image.png"), image_rgb)
        cv2.imwrite(os.path.join(sample_path, "mask.png"), mask_tile * 255)

        # Generate positive and negative prompts
        pos_yx = np.argwhere(mask_tile > 0)
        neg_yx = np.argwhere(mask_tile == 0)

        prompts = {
            "positive_points": [[int(p[1]), int(p[0])] for p in pos_yx[::max(1, len(pos_yx)//5)]],
            "negative_points": [[int(p[1]), int(p[0])] for p in neg_yx[::max(1, len(neg_yx)//5)]]
        }

        with open(os.path.join(sample_path, "prompts.json"), "w") as f:
            json.dump(prompts, f)

        count += 1

print(f"✅ Done. Generated {count} training samples in {output_dir}")
