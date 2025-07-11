import rasterio
from rasterio.features import shapes
import numpy as np
import json
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

# Load georeferenced mask
mask_path = "juliflora_mask_kachchh.tif"
with rasterio.open(mask_path) as mask_src:
    mask = mask_src.read(1)  # single band mask
    transform = mask_src.transform
    crs = mask_src.crs

# Convert mask to polygons
mask = mask.astype(np.uint8)
results = (
    {"properties": {"value": v}, "geometry": s}
    for s, v in shapes(mask, transform=transform)
    if v == 1  # only juliflora pixels
)

# Optionally: reproject to WGS84 for mapping
transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

features = []
for result in results:
    geom = shape(result["geometry"])
    reprojected_geom = shapely_transform(transformer.transform, geom)
    features.append({
        "type": "Feature",
        "geometry": mapping(reprojected_geom),
        "properties": result["properties"]
    })

# Save to GeoJSON
geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open("juliflora_polygons.geojson", "w") as f:
    json.dump(geojson, f)

print(f"✅ Saved {len(features)} polygons to juliflora_polygons.geojson")
