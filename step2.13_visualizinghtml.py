import geopandas as gpd
import folium

# Load your polygons
gdf = gpd.read_file("juliflora_polygons.geojson")

# Reproject to WGS84 for web maps
gdf = gdf.to_crs(epsg=4326)

# Create a folium map centered on your data
center = gdf.geometry.unary_union.centroid
m = folium.Map(location=[center.y, center.x], zoom_start=12)

# Add polygons to the map
folium.GeoJson(gdf).add_to(m)

# Save or display the map
m.save("juliflora_map.html")
print("✅ Map saved as juliflora_map.html. Open it in a browser.")
