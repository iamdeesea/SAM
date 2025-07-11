import geopandas as gpd
import folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from folium.raster_layers import ImageOverlay
from rasterio.warp import transform_bounds
from shapely.geometry import mapping

# Load GeoJSON and convert to WGS84
gdf = gpd.read_file("juliflora_polygons.geojson").to_crs(epsg=4326)

# Get center for the map
centroid = gdf.geometry.union_all().centroid
map_center = [centroid.y, centroid.x]

# Base map
m = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB.Positron")

# Add polygons
folium.GeoJson(
    gdf,
    name="Juliflora Polygons",
    style_function=lambda x: {
        "color": "blue",
        "weight": 1,
        "fillOpacity": 0.2
    }
).add_to(m)

# Load NDVI raster
with rasterio.open("juliflora_ndvi_kachchh.tif") as src:
    ndvi = src.read(1).astype(float)
    ndvi_bounds = src.bounds
    ndvi_crs = src.crs

    # Reproject bounds to EPSG:4326 if necessary
    if ndvi_crs.to_string() != "EPSG:4326":
        ndvi_bounds = transform_bounds(ndvi_crs, "EPSG:4326", *ndvi_bounds)

# Normalize NDVI
ndvi = np.nan_to_num(ndvi, nan=0)
ndvi_min, ndvi_max = ndvi.min(), ndvi.max()
ndvi_scaled = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min) * 255).astype(np.uint8)

# Flip Y-axis for correct orientation in folium
ndvi_scaled_flipped = np.flipud(ndvi_scaled)

# Make RGB grayscale
ndvi_rgb = np.stack([ndvi_scaled_flipped] * 3, axis=-1)

# Save image
plt.imsave("ndvi_overlay.png", ndvi_rgb)

# Fix bounds order: [southwest, northeast]
image_overlay = ImageOverlay(
    name="NDVI",
    image="ndvi_overlay.png",
    bounds=[[ndvi_bounds[1], ndvi_bounds[0]], [ndvi_bounds[3], ndvi_bounds[2]]],
    opacity=0.6,
    interactive=True,
    cross_origin=False
)
image_overlay.add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save map
m.save("juliflora_ndvi_map.html")
print("✅ Fixed and saved: juliflora_ndvi_map.html")
