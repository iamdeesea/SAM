import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
import torch
import os
from albumentations import Resize
# --------------------
# CONFIG
# --------------------
NDVI_PATH = "juliflora_ndvi_bboxfix.tif"
GEOJSON_PATH = "juliflora_polygons.geojson"
MASK_PATH = "prosopis_mask.tif"
IMG_SIZE = 256
BATCH_SIZE = 2

# --------------------
# DATASET CLASS
# --------------------
class NDVIDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.stack([self.images[idx]] * 3, axis=-1)  # Convert (H, W) → (H, W, 3)
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


# --------------------
# MAIN FUNCTION
# --------------------
def main():
    # 1. Load NDVI raster
    with rasterio.open(NDVI_PATH) as src:
        ndvi = src.read(1)
        profile = src.profile
        transform = src.transform

    print("✓ NDVI shape :", ndvi.shape)

   # 2. Load polygons and match CRS
    gdf = gpd.read_file(GEOJSON_PATH)
    print("✓ Polygons   :", len(gdf))

    # Open raster to get CRS
    with rasterio.open(NDVI_PATH) as src:
        raster_crs = src.crs

    #    Reproject polygons to match NDVI CRS
    if gdf.crs != raster_crs:
        print(f"⤴ Reprojecting polygons from {gdf.crs} to {raster_crs}")
    gdf = gdf.to_crs(raster_crs)



    # 3. Rasterize polygons
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=ndvi.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # Save mask (optional)
    profile.update(count=1, dtype=np.uint8)
    with rasterio.open(MASK_PATH, "w", **profile) as dst:
        dst.write(mask, 1)
    print("✓ Mask saved to", MASK_PATH)

    # 4. Extract image-mask patches
    imgs, msks = [], []
    H, W = ndvi.shape

    for i in range(0, H - IMG_SIZE, IMG_SIZE):
        for j in range(0, W - IMG_SIZE, IMG_SIZE):
            img_patch = ndvi[i:i + IMG_SIZE, j:j + IMG_SIZE]
            msk_patch = mask[i:i + IMG_SIZE, j:j + IMG_SIZE]

            # Check if mask patch contains any foreground pixels
            if msk_patch.sum() > 0:
                imgs.append(img_patch.astype(np.float32) / 255.0)
                msks.append(msk_patch.astype(np.uint8))

    print("✓ Total valid patches:", len(imgs))

    # 5. Train/val split
    train_imgs, val_imgs, train_msks, val_msks = train_test_split(
        imgs, msks, test_size=0.25, random_state=42
    )
    print("✓ Train samples:", len(train_imgs), "| Val samples:", len(val_imgs))

    # 6. Define transforms
    transforms = Compose([
    Resize(1024, 1024),                    # <--- REQUIRED FOR SAM
    Normalize(mean=0.0, std=1.0),
    ToTensorV2()
    ])

    train_ds = NDVIDataset(train_imgs, train_msks, transform=transforms)
    val_ds = NDVIDataset(val_imgs, val_msks, transform=transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 7. Try getting one batch
    imgs, msks = next(iter(train_loader))
    print("✓ Batch shapes :", imgs.shape, msks.shape)

# Make these available for import
    global_train_loader = train_loader
    global_val_loader = val_loader
    return train_loader, val_loader  # ADD THIS at the end of main()


# --------------------
# WINDOWS FIX FOR MULTIPROCESSING
# --------------------
# Optional: run step6 for testing, or import just the loaders
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
# ------------------------------------------------------------------
# --------------------
# WINDOWS FIX FOR MULTIPROCESSING
# --------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_loader, val_loader = main()  # get return values
    imgs, msks = next(iter(train_loader))
    print("✓ Batch shapes :", imgs.shape, msks.shape)

# When imported: build loaders and expose them
else:
    train_loader, val_loader = main()



