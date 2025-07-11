import geopandas as gpd
import json

# Load the GeoJSON
gdf = gpd.read_file("juliflora_clusters_kachchh.geojson")

# Compute centroids
gdf['centroid'] = gdf.geometry.centroid
prompt_points = [[pt.x, pt.y] for pt in gdf['centroid']]

# Optionally save as JSON file
with open('sam2_prompts.json', 'w') as f:
    json.dump(prompt_points, f, indent=2)

print(f'✅ Extracted {len(prompt_points)} SAM2 point prompts and saved to sam2_prompts.json')
