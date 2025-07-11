import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import shape, Polygon
import geopandas as gpd
from skimage import measure

# Step 1: Load binary mask (white = Prosopis)
mask_path = "detected_mask.png"  # Replace with your actual path
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = (mask > 127).astype(np.uint8)  # Ensure binary (0 or 1)

# Step 2: Define bounds from known area (Kachchh bounding box)
# Example: approximate lat/lon bounds for your tile
minx, miny = 68.5, 23.0
maxx, maxy = 70.0, 24.5
height, width = mask.shape
transform = from_bounds(minx, miny, maxx, maxy, width, height)

# Step 3: Convert mask to polygons
polygons = []
for region in measure.regionprops(measure.label(mask)):
    if region.area >= 100:  # Skip tiny noise
        coords = region.coords[:, [1, 0]]  # (x, y)
        poly_coords = [(c[0], c[1]) for c in coords]
        poly = Polygon(poly_coords)
        if poly.is_valid:
            polygons.append(poly)

# Step 4: Create GeoDataFrame
gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
gdf.to_file("juliflora_clusters_kachchh.geojson", driver="GeoJSON")

print(f"✅ Saved {len(polygons)} polygons to juliflora_clusters_kachchh.geojson")
