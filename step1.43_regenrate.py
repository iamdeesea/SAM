import geopandas as gpd
import json
from shapely.geometry import Point

# Load GeoJSON file with polygons
gdf = gpd.read_file("juliflora_clusters_kachchh.geojson")

# Ensure the geometry is valid
gdf = gdf[gdf.is_valid & gdf.geometry.notnull()]

# Compute centroid for each polygon
gdf['centroid'] = gdf.geometry.centroid

# Reproject to match image CRS if needed (optional)
# gdf = gdf.to_crs("EPSG:32643")  # UTM zone for Kachchh region, adjust as needed

# Convert centroids to prompt format
prompts = []
for centroid in gdf['centroid']:
    if isinstance(centroid, Point):
        x, y = centroid.x, centroid.y
        prompts.append({
            "point": [[x, y]],
            "label": [1]
        })

# Save to sam2_prompts.json
with open("sam2_prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print(f"✅ Successfully created {len(prompts)} prompts with valid centroids.")
