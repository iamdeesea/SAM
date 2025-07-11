import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import numpy as np
import gcsfs

# ---------- CONFIG ----------
GCS_IMAGE_PATH = "gs://your-bucket/ndvi_feature_babul_kacch_179.tif"
POLYGON_PATH = "juliflora_179.geojson"  # Or .shp
MASK_OUTPUT_PATH = "masks/juliflora_mask_179.tif"
# ----------------------------

# Load image from GCS
gcs = gcsfs.GCSFileSystem()
with gcs.open(GCS_IMAGE_PATH, 'rb') as f:
    with rasterio.open(f) as src:
        profile = src.profile
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs

# Load polygon file
gdf = gpd.read_file(POLYGON_PATH)
if gdf.crs != crs:
    gdf = gdf.to_crs(crs)

# Rasterize polygon
mask = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=out_shape,
    transform=transform,
    fill=0,
    dtype='uint8'
)

# Save the binary mask
profile.update({
    'count': 1,
    'dtype': 'uint8'
})
os.makedirs(os.path.dirname(MASK_OUTPUT_PATH), exist_ok=True)
with rasterio.open(MASK_OUTPUT_PATH, 'w', **profile) as dst:
    dst.write(mask, 1)

print(f"Saved binary mask to: {MASK_OUTPUT_PATH}")
