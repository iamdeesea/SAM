#!/usr/bin/env python
"""
Accurate Geospatial Inference Pipeline
"""
import numpy as np, rasterio, torch, os, folium, geopandas as gpd, cv2
from shapely.geometry import shape
from shapely.ops import unary_union
from rasterio.windows import Window
from rasterio.features import shapes
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from step7 import SAMBinaryHead

# ----------------------------------------------------------------------
# 1. CONFIG ------------------------------------------------------------------
NDVI_PATH = "juliflora_ndvi_bboxfix.tif"
CKPT_PATH = "sam_binary_epoch10.pth"
SAM_WEIGHTS = "sam_vit_h.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 1024
OVERLAP = 256
THRESH = 0.25
MASK_TIF = "prosopis_pred_mask.tif"
POLY_GEOJS = "prosopis_pred_polygons.geojson"
HTML_OUT = "prosopis_map.html"
SIMPLIFY_TOLERANCE = 5.0
LOCATION = [23.25, 69.67]  # Kutch, Gujarat coordinates

# ----------------------------------------------------------------------
# 2. LOAD MODEL -------------------------------------------------------------
sam = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHTS).to(DEVICE)
predictor = SamPredictor(sam)
model = SAMBinaryHead(predictor).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# ----------------------------------------------------------------------
# 3. VERIFY RASTER GEOREFERENCING -------------------------------------------
with rasterio.open(NDVI_PATH) as src:
    print("\n=== RASTER METADATA ===")
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    print(f"Width: {src.width}, Height: {src.height}")
    print(f"Bounds: {src.bounds}")
    print(f"Driver: {src.driver}")
    print(f"Count: {src.count}")
    print(f"Res: {src.res}\n")
    
    profile = src.profile.copy()
    H, W = src.height, src.width
    out_arr = np.zeros((H, W), dtype=np.float32)
    count_arr = np.zeros((H, W), dtype=np.uint8)
    
    stride = PATCH - OVERLAP
    
    for row in tqdm(range(0, H, stride), desc="Predicting"):
        for col in range(0, W, stride):
            # Get window dimensions
            win_row = min(PATCH, H - row)
            win_col = min(PATCH, W - col)
            
            # Read NDVI patch
            window = Window(col, row, win_col, win_row)
            ndvi = src.read(1, window=window)
            ndvi = np.nan_to_num(ndvi, nan=0.0)
            
            # Normalize to [0, 1] range
            ndvi = np.clip(ndvi, -1, 1)
            ndvi = (ndvi + 1) / 2
            
            # Pad to 1024x1024 if needed
            pad_bottom = PATCH - win_row
            pad_right = PATCH - win_col
            padded_ndvi = np.pad(ndvi, ((0, pad_bottom), (0, pad_right)), 
                                mode='constant', constant_values=0)
            
            # Create 3-channel "RGB"
            ndvi_rgb = np.stack([padded_ndvi, padded_ndvi, padded_ndvi], axis=0)
            img_t = torch.tensor(ndvi_rgb, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(img_t)
                mask_patch = torch.sigmoid(logits)[0, 0].cpu().numpy()
                
                # Crop back to original size
                mask_patch = mask_patch[:win_row, :win_col]
                
                # Aggregate predictions
                out_arr[row:row+win_row, col:col+win_col] += mask_patch
                count_arr[row:row+win_row, col:col+win_col] += 1

    # Average overlapping areas
    out_arr = np.divide(out_arr, count_arr, where=count_arr > 0)
    final_mask = (out_arr > THRESH).astype(np.uint8)

# ----------------------------------------------------------------------
# 4. SAVE PRED MASK AS GEOTIFF WITH CORRECT GEOREFERENCE -------------------
profile.update(
    count=1,
    dtype=rasterio.uint8,
    compress='lzw',
    nodata=0
)
with rasterio.open(MASK_TIF, "w", **profile) as dst:
    dst.write(final_mask, 1)
print(f"✓ Saved mask to {MASK_TIF}")

# ----------------------------------------------------------------------
# 5. POLYGONIZATION WITH GEOSPATIAL VALIDATION -----------------------------
with rasterio.open(MASK_TIF) as src:
    mask = src.read(1)
    transform = src.transform
    crs = src.crs
    
    # Generate polygons
    polys = []
    for geom, val in shapes(mask, mask==1, transform=transform):
        if val == 1:
            poly = shape(geom)
            if poly.is_valid and poly.area > 100:
                polys.append({
                    "geometry": poly,
                    "properties": {"value": val}
                })
    
    gdf = gpd.GeoDataFrame.from_features(polys, crs=crs)

# ----------------------------------------------------------------------
# 6. CRITICAL: VERIFY GEOSPATIAL COORDINATES -----------------------------
print("\n=== PREDICTION METADATA ===")
print(f"Polygon CRS: {gdf.crs}")
if not gdf.empty:
    centroid = gdf.geometry[0].centroid
    print(f"First polygon centroid: ({centroid.x:.6f}, {centroid.y:.6f})")
    print(f"Polygon bounds: {gdf.total_bounds}")

# Reproject to WGS84
gdf_wgs84 = gdf.to_crs("EPSG:4326")
gdf_wgs84.to_file(POLY_GEOJS, driver="GeoJSON")
print(f"✓ {len(gdf_wgs84)} polygons → {POLY_GEOJS}")

# ----------------------------------------------------------------------
# 7. ACCURATE FOLIUM MAP WITH GEOSPATIAL CONTROLS ----------------------
# Create map centered on Kutch
m = folium.Map(location=LOCATION, zoom_start=11, control_scale=True)

# Add predicted polygons
if not gdf_wgs84.empty:
    folium.GeoJson(
        POLY_GEOJS,
        name="Prosopis Prediction",
        style_function=lambda _: {
            "fillColor": "#ff0000",
            "color": "#ff0000",
            "weight": 2,
            "fillOpacity": 0.5
        },
        tooltip=folium.GeoJsonTooltip(fields=["value"])
    ).add_to(m)

# Add reference layers
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr='Google Satellite',
    name='Satellite',
    overlay=False,
    control=True
).add_to(m)

folium.TileLayer(
    tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attr='OpenStreetMap',
    name='Street Map',
    overlay=False
).add_to(m)

# Add coordinate display
m.add_child(folium.LatLngPopup())

# Add layer control
folium.LayerControl().add_to(m)

# Add scale bar
folium.plugins.MousePosition().add_to(m)
folium.plugins.MiniMap().add_to(m)

# Add boundary rectangle for verification
folium.Rectangle(
    bounds=[[LOCATION[0]-0.1, LOCATION[1]-0.1], 
            [LOCATION[0]+0.1, LOCATION[1]+0.1]],
    color='#00ff00',
    fill=False,
    weight=3
).add_to(m)

m.save(HTML_OUT)
print(f"🎉 Interactive map saved → {HTML_OUT}")
print("\n=== VERIFICATION INSTRUCTIONS ===")
print("1. Open the HTML map in browser")
print("2. Right-click → 'Inspect' → Console")
print("3. Check coordinates using mouse position plugin")
print("4. Verify green rectangle is centered on Kutch")