import os
import numpy as np
import rasterio
import cv2

def process_ndvi(input_tif, output_png):
    with rasterio.open(input_tif) as src:
        ndvi = src.read(1)
        
        # Replace nodata values with 0 and normalize
        ndvi = np.nan_to_num(ndvi)
        ndvi = ((ndvi + 1) * 127.5).astype(np.uint8)  # Convert (-1 to 1) range to (0-255)
        
        # Create 3-channel image (SAM expects RGB)
        rgb = np.stack([ndvi]*3, axis=-1)
        cv2.imwrite(output_png, rgb)

# Correct usage - either process single file or directory
input_path = "juliflora_ndvi_bboxfix.tif"  # Your input file
output_path = "juliflora_ndvi_bboxfix.png"  # Output file

# Process single file
if os.path.isfile(input_path):
    process_ndvi(input_path, output_path)
    print(f"Successfully converted {input_path} to {output_path}")
else:
    print(f"Error: File not found - {input_path}")

# Alternative: Process directory (uncomment if needed)

input_dir = "juliflora_ndvi_bboxfix.tif"
output_dir = "path/to/output_pngs/"

if os.path.isdir(input_dir):
    os.makedirs(output_dir, exist_ok=True)
    for tif_file in os.listdir(input_dir):
        if tif_file.endswith(".tif"):
            process_ndvi(
                os.path.join(input_dir, tif_file),
                os.path.join(output_dir, tif_file.replace(".tif", ".png"))
            )
    print(f"Converted all TIFFs in {input_dir} to PNGs in {output_dir}")
else:
    print(f"Error: Directory not found - {input_dir}")
