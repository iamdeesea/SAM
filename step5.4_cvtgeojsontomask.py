import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

def geojson_to_mask(geojson_path, ref_tif, output_tif):
    gdf = gpd.read_file(geojson_path)
    with rasterio.open(ref_tif) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='uint8', nodata=0)
        transform = src.transform
        shape = src.shape
        
        with rasterio.open(output_tif, 'w', **meta) as dst:
            raster = rasterize(
                [(geom, 1) for geom in gdf.geometry],
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype='uint8'
            )
            dst.write(raster, 1)

# Example usage
base_dir = "data/train"
for img_file in os.listdir(os.path.join(base_dir, "images")):
    if img_file.endswith(".tif"):
        img_path = os.path.join(base_dir, "images", img_file)
        geojson_path = os.path.join(base_dir, "geojson", img_file.replace(".tif", ".geojson"))
        mask_path = os.path.join(base_dir, "masks", f"mask_{img_file}")
        
        if os.path.exists(geojson_path):
            geojson_to_mask(geojson_path, img_path, mask_path)
            print(f"Created mask for {img_file}")