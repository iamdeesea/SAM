import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
from shapely.geometry import box

# Load NDVI and get its bounds
with rasterio.open("juliflora_ndvi_bboxfix.tif") as src:
    ndvi_bounds = src.bounds
    ndvi_crs = src.crs
    print("NDVI bounds (UTM):", ndvi_bounds)

# Transform NDVI bounds to lat/lon for comparison
ndvi_bounds_wgs84 = transform_bounds(ndvi_crs, "EPSG:4326", *ndvi_bounds)
print("NDVI bounds in EPSG:4326:", ndvi_bounds_wgs84)

# Load polygons
gdf = gpd.read_file("juliflora_polygons.geojson")
print("GeoJSON bounds:", gdf.total_bounds)

# Plot both in EPSG:4326 for easy check
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the polygon GeoJSON
gdf.plot(ax=ax, color='none', edgecolor='red', linewidth=1, label="GeoJSON Polygons")

# Create and plot NDVI extent as a rectangle
ndvi_box_wgs84 = box(*ndvi_bounds_wgs84)
gpd.GeoSeries([ndvi_box_wgs84], crs="EPSG:4326").plot(
    ax=ax, edgecolor='blue', facecolor='none', linewidth=2, label="NDVI Extent"
)

plt.legend()
plt.title("NDVI Raster vs GeoJSON Polygons (EPSG:4326)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()
