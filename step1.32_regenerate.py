import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import shape
from rasterio.features import shapes

# Step 1: Load the binary mask
with rasterio.open("juliflora_mask_kachchh.tif") as src:
    mask = src.read(1)  # first band
    transform = src.transform

# Step 2: Extract shapes where pixel == 1
shapes_gen = shapes(mask, mask=mask == 1, transform=transform)

# Step 3: Convert to polygons
polygons = []
for geom, value in shapes_gen:
    if value == 1:
        polygons.append(shape(geom))

# Step 4: Save to GeoJSON
gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
gdf.to_file("juliflora_clusters_kachchh.geojson", driver="GeoJSON")

print(f"✅ Saved {len(gdf)} polygons to juliflora_clusters_kachchh.geojson")
