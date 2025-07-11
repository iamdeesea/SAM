import geopandas as gpd
import rasterio
import numpy as np
import folium
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from matplotlib.colors import Normalize
import os

# === Load GeoJSON ===
gdf = gpd.read_file("juliflora_polygons.geojson")
print("✅ Loaded GeoJSON")
print("  - CRS:", gdf.crs)
print("  - Bounds:", gdf.total_bounds)
print("  - Empty?:", gdf.empty)

# === Load NDVI ===
with rasterio.open("juliflora_ndvi_kachchh.tif") as src:
    ndvi = src.read(1)
    ndvi_transform = src.transform
    ndvi_crs = src.crs
    ndvi_bounds = src.bounds

print("✅ Loaded NDVI")
print("  - CRS:", ndvi_crs)
print("  - Bounds:", ndvi_bounds)

# === Reproject GeoJSON to NDVI CRS ===
if gdf.crs != ndvi_crs:
    gdf = gdf.to_crs(ndvi_crs)

print("✅ Reprojected GeoJSON to NDVI CRS")
print("  - New Bounds:", gdf.total_bounds)

# === Normalize NDVI and color ===
ndvi = np.clip(ndvi, -1, 1)
norm = Normalize(vmin=-1, vmax=1)
ndvi_rgb = plt.colormaps.get_cmap("YlGn")(norm(ndvi))[:, :, :3]  # RGB only
ndvi_rgb = (ndvi_rgb * 255).astype(np.uint8)

# === Save RGB NDVI as image tile ===
output_tile = "ndvi_tile.png"
plt.imsave(output_tile, ndvi_rgb)

# === Get centroid for map center ===
centroid = gdf.geometry.union_all().centroid
centroid_latlon = gpd.GeoSeries([centroid], crs=ndvi_crs).to_crs("EPSG:4326").geometry[0]
lat, lon = centroid_latlon.y, centroid_latlon.x

# === Create Folium map ===
m = folium.Map(location=[lat, lon], zoom_start=12, tiles="OpenStreetMap")

# Overlay NDVI tile
folium.raster_layers.ImageOverlay(
    name="NDVI",
    image=output_tile,
    bounds=[[ndvi_bounds.bottom, ndvi_bounds.left], [ndvi_bounds.top, ndvi_bounds.right]],
    opacity=0.6,
).add_to(m)

# Add GeoJSON
geojson_path = "juliflora_polygons.geojson"
folium.GeoJson(geojson_path, name="GeoJSON Overlay", style_function=lambda x: {
    "fillColor": "blue",
    "color": "black",
    "weight": 1,
    "fillOpacity": 0.3
}).add_to(m)

folium.LayerControl().add_to(m)
m.save("juliflora_ndvi_map.html")
print("✅ Map saved as juliflora_ndvi_map.html")
