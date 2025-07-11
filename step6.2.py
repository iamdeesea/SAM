import os
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
from tqdm import tqdm
import albumentations as A
import cv2
import random
from sklearn.model_selection import train_test_split

# ------------------ CONFIG ------------------
NDVI_PATH = "juliflora_ndvi_bboxfix.tif"
GEOJSON_PATH = "juliflora_polygons.geojson"
PATCH_SIZE = 512
STRIDE = PATCH_SIZE // 2
MIN_POSITIVE_PIXELS = 5
NEGATIVE_SAMPLE_PROB = 0.6
AUG_MULT = 3
VAL_SPLIT = 0.2
OUT_DIR = Path("patches")
OUT_DIR.mkdir(exist_ok=True)

AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
])
# --------------------------------------------

print(f"\n✓ Loading NDVI: {NDVI_PATH}")
with rasterio.open(NDVI_PATH) as src:
    ndvi = src.read(1)
    transform = src.transform
    crs = src.crs
    meta = src.meta
    width, height = src.width, src.height

print(f"NDVI size: {height} × {width} | CRS: {crs}")

print(f"✓ Loading polygons: {GEOJSON_PATH}")
gdf = gpd.read_file(GEOJSON_PATH)

if gdf.empty:
    print("⚠️  No polygons found in GeoJSON.")
    exit()

if gdf.crs.to_epsg() != crs.to_epsg():
    print(f"⤴ Reprojecting GeoJSON from {gdf.crs} → {crs}")
    gdf = gdf.to_crs(crs)

mask = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=ndvi.shape,
    transform=transform,
    fill=0,
    dtype=np.uint8
)

print(f"Mask positive pixels: {mask.sum()}")

imgs, msks = [], []
for y in range(0, height - PATCH_SIZE + 1, STRIDE):
    for x in range(0, width - PATCH_SIZE + 1, STRIDE):
        img_patch = ndvi[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

        if np.isnan(img_patch).any():
            continue

        img_rgb = np.repeat(np.expand_dims(img_patch, axis=-1), 3, axis=-1)
        img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.ptp() + 1e-6) * 255).astype(np.uint8)

        keep = mask_patch.sum() > MIN_POSITIVE_PIXELS or random.random() < NEGATIVE_SAMPLE_PROB
        if keep:
            for _ in range(AUG_MULT):
                aug = AUG(image=img_rgb, mask=mask_patch)
                img_aug = aug["image"] / 255.0
                mask_aug = aug["mask"]

                imgs.append(img_aug.transpose(2, 0, 1))
                msks.append(mask_aug)

print(f"Total kept patches: {len(imgs)}")
if len(imgs) == 0:
    print("❌ No patches saved. Try lowering MIN_POSITIVE_PIXELS or increasing NEGATIVE_SAMPLE_PROB.")
    exit()

X_train, X_val, y_train, y_val = train_test_split(imgs, msks, test_size=VAL_SPLIT, random_state=42)

np.savez_compressed(OUT_DIR / "train.npz", images=np.stack(X_train), masks=np.stack(y_train))
np.savez_compressed(OUT_DIR / "val.npz", images=np.stack(X_val), masks=np.stack(y_val))
print(f"✓ Saved {len(X_train)} training and {len(X_val)} validation patches to {OUT_DIR.resolve()}")
