# prepare_patches_single.py
from pathlib import Path
import numpy as np, rasterio
from rasterio.features import rasterize
import geopandas as gpd, random, os

# ---------------- user paths -----------------
NDVI_TIF      = Path("juliflora_ndvi_bboxfix.tif")     # your single NDVI file
POLYGON_GEOJS = Path("juliflora_polygons.geojson")     # your GeoJSON
PATCH         = 1024                                   # size of square patch
VAL_SPLIT     = 0.2                                    # 20 % held‑out val
OUT_DIR       = Path("patches")
OUT_DIR.mkdir(exist_ok=True)
# ---------------------------------------------

# 1. Load raster
with rasterio.open(NDVI_TIF) as src:
    ndvi       = src.read(1)                           # (H, W)
    raster_crs = src.crs
    transform  = src.transform
H, W = ndvi.shape
print(f"NDVI size: {H} × {W} | CRS: {raster_crs}")

# 2. Load polygons and re‑project
gdf = gpd.read_file(POLYGON_GEOJS)
if gdf.empty:
    raise ValueError("GeoJSON has no geometries!")
gdf = gdf.to_crs(raster_crs)                           # align CRS

# 3. Rasterise polygons → binary mask
mask = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=(H, W),
    transform=transform,
    dtype=np.uint8
)
print(f"Mask positive pixels: {int(mask.sum())}")

# 4. Extract patches
imgs, msks = [], []
for y in range(0, H - PATCH + 1, PATCH):
    for x in range(0, W - PATCH + 1, PATCH):
        m_patch = mask[y:y+PATCH, x:x+PATCH]
        keep = m_patch.sum() > 50 or random.random() < 0.2  # <‑‑ main filter
        if keep:
            img_patch = ndvi[y:y+PATCH, x:x+PATCH]
            img_rgb   = np.stack([img_patch]*3, axis=0).astype(np.float32) / 255.0
            imgs.append(img_rgb)
            msks.append(m_patch)

print(f"Total kept patches: {len(imgs)}")
if len(imgs) == 0:
    raise RuntimeError("No patches passed the filter. Try lowering the pixel threshold.")

# 5. Train / val split (optional)
idx = np.arange(len(imgs))
np.random.shuffle(idx)
cut = int(len(idx) * (1 - VAL_SPLIT))
train_idx, val_idx = idx[:cut], idx[cut:]

np.savez_compressed(OUT_DIR/"train_imgs.npz", imgs=np.stack(imgs)[train_idx])
np.savez_compressed(OUT_DIR/"train_msks.npz", msks=np.stack(msks)[train_idx])
np.savez_compressed(OUT_DIR/"val_imgs.npz",   imgs=np.stack(imgs)[val_idx])
np.savez_compressed(OUT_DIR/"val_msks.npz",   msks=np.stack(msks)[val_idx])
print("✓ NPZ files written to", OUT_DIR.resolve())
