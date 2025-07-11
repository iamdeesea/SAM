import geopandas as gpd

gdf = gpd.read_file("juliflora_clusters_kachchh.geojson")
print("Geometry types:", gdf.geom_type.value_counts())
print("Total features:", len(gdf))
