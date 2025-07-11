import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

# Load the binary mask
mask = cv2.imread("detected_mask.png", cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError("Could not load 'detected_mask.png'.")

# Threshold to ensure binary mask
_, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert contours to polygons
polygons = []
for cnt in contours:
    if len(cnt) >= 3:  # valid polygon
        coords = cnt.squeeze().tolist()
        poly = Polygon(coords)
        if poly.is_valid and poly.area > 10:  # optional: filter very small blobs
            polygons.append(poly)

# Save to GeoJSON
gdf = gpd.GeoDataFrame(geometry=polygons)
gdf.to_file("juliflora_clusters_kachchh.geojson", driver='GeoJSON')

print(f"✅ Saved {len(polygons)} polygons to 'juliflora_clusters_kachchh.geojson'")
