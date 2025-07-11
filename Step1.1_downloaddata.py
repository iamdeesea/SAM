import ee
ee.Authenticate()
ee.Initialize(project='forgettingsatellitedata')
print("Earth Engine initialized.")
roi = ee.Geometry.BBox(69.1, 23.1, 70.0, 23.9)
# NDVI from Sentinel-2 Surface Reflectance
def get_ndvi(roi):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(roi) \
            .filterDate('2024-11-01', '2025-03-31') \
            .sort('CLOUDY_PIXEL_PERCENTAGE') \
            .first()
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ndvi.clip(roi)

# DEM from NASA SRTM
def get_dem(roi):
    dem = ee.Image('USGS/SRTMGL1_003')
    return dem.clip(roi)
def export_image(image, description, folder, region, scale=10):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region.getInfo()['coordinates'],
        scale=scale,
        maxPixels=1e13
    )
    task.start()
    print(f"✅ Export started: {description}")
ndvi = get_ndvi(roi)
dem = get_dem(roi)

export_image(ndvi, 'juliflora_ndvi_kachchh', 'GEE_exports', roi)
export_image(dem, 'juliflora_dem_kachchh', 'GEE_exports', roi)
