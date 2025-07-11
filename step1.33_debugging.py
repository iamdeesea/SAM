import rasterio
import numpy as np

with rasterio.open("juliflora_mask_kachchh.tif") as src:
    mask = src.read(1)

unique_values = np.unique(mask)
print("Unique pixel values in mask:", unique_values)
