import geopandas as gpd
import json

# Load your polygon GeoJSON
gdf = gpd.read_file("juliflora_clusters_kachchh.geojson")

# Create a list of prompts (centroids as (x, y) in image pixel space or lat/lon if georeferenced)
prompts = []
for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    prompts.append({
        "id": idx,
        "type": "point",
        "coordinates": [centroid.x, centroid.y],
        "label": 1  # foreground
    })

# Save prompts as JSON
with open("sam2_prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print(f"✅ Extracted {len(prompts)} SAM2 point prompts and saved to sam2_prompts.json")
