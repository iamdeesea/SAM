import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

# ---- Step 1: Load the screenshot ----
screenshot_path = 'juliflora_screenshot.png'  # <-- your screenshot file
image = cv2.imread(screenshot_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---- Step 2: Create binary mask from red/pink highlights ----
# You may need to adjust these HSV values depending on actual highlight color
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([160, 50, 50])  # light reddish-pink
upper = np.array([180, 255, 255])  # deep red
mask = cv2.inRange(hsv, lower, upper)
binary_mask = (mask > 0).astype(np.uint8)

# ---- Step 3: Align with exported NDVI GeoTIFF ----
ndvi_path = 'juliflora_ndvi_kachchh.tif'  # exported from Earth Engine
with rasterio.open(ndvi_path) as src:
    profile = src.profile
    transform = src.transform
    crs = src.crs
    width, height = src.width, src.height

# Resize binary mask to match NDVI raster shape
binary_mask_resized = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)

# ---- Step 4: Save georeferenced binary mask as GeoTIFF ----
mask_output_path = 'juliflora_mask_kachchh.tif'
profile.update({
    'count': 1,
    'dtype': 'uint8'
})

with rasterio.open(mask_output_path, 'w', **profile) as dst:
    dst.write(binary_mask_resized, 1)

print(f'✅ Saved georeferenced binary mask to: {mask_output_path}')

# ---- Step 5: Convert binary mask to GeoJSON polygons ----
shapes_gen = shapes(binary_mask_resized, mask=binary_mask_resized > 0, transform=transform)
geoms = [shape(geom) for geom, value in shapes_gen if value == 1]

gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
gdf = gdf[gdf.area > 10]  # Optional: filter small noise

geojson_output_path = 'juliflora_clusters_kachchh.geojson'
gdf.to_file(geojson_output_path, driver='GeoJSON')

print(f'✅ Saved *Prosopis juliflora* polygons to: {geojson_output_path}')
