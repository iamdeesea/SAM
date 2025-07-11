import rasterio

with rasterio.open("juliflora_ndvi_kachchh.tif") as src:
    print("CRS:", src.crs)
    print("Transform:", src.transform)
