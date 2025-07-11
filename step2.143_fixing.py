import geopandas as gpd

gdf = gpd.read_file("juliflora_polygons.geojson")
print(gdf.total_bounds)
