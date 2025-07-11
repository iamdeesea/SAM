import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from pathlib import Path

ndvi_path   = Path("juliflora_ndvi_bboxfix.tif")              # ← adjust
poly_path   = Path("juliflora_polygons.geojson")

# --- NDVI ---
with rasterio.open(ndvi_path) as src:
    ndvi      = src.read(1)                      # 2‑D array (H, W)
    profile   = src.profile                      # keep for saving later
    transform = src.transform
    crs       = src.crs
    height, width = src.height, src.width

# --- polygons ---
polys = gpd.read_file(poly_path)
if polys.crs != crs:
    polys = polys.to_crs(crs)
# A value of 1 where Prosopis is present, 0 elsewhere
mask = rasterize(
    [(geom, 1) for geom in polys.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.uint8
)
mask_path = ndvi_path.with_name("prosopis_mask.tif")
with rasterio.open(mask_path, "w", **profile, count=1, dtype=np.uint8) as dst:
    dst.write(mask, 1)
