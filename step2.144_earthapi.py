import ee
ee.Initialize(project='forgettingsatellitedata')  # Use your GCP project ID

# Define bounding box from your GeoJSON
region = ee.Geometry.BBox(68.7, 24.3407255, 68.9433585, 24.5)

# Load Sentinel-2 and calculate NDVI
s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
    .filterDate("2022-01-01", "2022-12-31") \
    .filterBounds(region) \
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)) \
    .median()

ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")

# Export to Google Drive
export_task = ee.batch.Export.image.toDrive(
    image=ndvi,
    description='juliflora_ndvi_kachchh_bboxfix',
    folder='GEE_exports',
    fileNamePrefix='juliflora_ndvi_bboxfix',
    region=region,
    scale=10,               # 10m resolution
    crs='EPSG:32642',       # Match your NDVI image CRS
    maxPixels=1e13
)

export_task.start()
print("✅ Export started to Google Drive")
