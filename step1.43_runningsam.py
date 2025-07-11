import os
import cv2
import numpy as np
import torch
import json
from segment_anything import sam_model_registry, SamPredictor

# ----------------- Configuration -----------------
sam_checkpoint = "sam_vit_h.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

image_path = "juliflora_screenshot.png"  # Replace with your image file path
prompt_path = "sam2_prompts.json"    # Replace with your prompt file path

output_mask_dir = "masks"
os.makedirs(output_mask_dir, exist_ok=True)

# ----------------- Load Image -----------------
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ----------------- Load Model -----------------
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(image)

# ----------------- Load Prompts -----------------
with open(prompt_path, "r") as f:
    prompts = json.load(f)

# ----------------- Run Predictions -----------------
successful_masks = 0
for idx, prompt in enumerate(prompts):
    try:
        # Validate prompt
        if 'point' not in prompt or 'label' not in prompt:
            print(f"Missing key in prompt {idx}: 'point' or 'label'")
            continue

        input_point = np.array(prompt['point'])  # Should be shape (N, 2)
        input_label = np.array(prompt['label'])  # Should be shape (N,)

        if input_point.ndim != 2 or input_point.shape[1] != 2:
            print(f"Invalid shape for point in prompt {idx}: {input_point.shape}")
            continue

        if input_label.ndim != 1 or input_label.shape[0] != input_point.shape[0]:
            print(f"Label shape mismatch in prompt {idx}: {input_label.shape} vs {input_point.shape}")
            continue

        # Predict mask
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )

        # Save the mask
        mask = masks[0].astype(np.uint8) * 255
        mask_filename = os.path.join(output_mask_dir, f"mask_{idx:03d}.png")
        cv2.imwrite(mask_filename, mask)
        successful_masks += 1

    except Exception as e:
        print(f"Error processing prompt {idx}: {e}")

# ----------------- Final Check -----------------
if successful_masks == 0:
    raise RuntimeError("No masks generated. Check prompt format or image.")
else:
    print(f"Generated {successful_masks} masks successfully.")
# import os
# os.makedirs("masks", exist_ok=True)
# for i, mask in enumerate(masks):
#     mask_uint8 = (mask * 255).astype(np.uint8)
#     cv2.imwrite(f"masks/mask_{i}.png", mask_uint8)
# composite = np.zeros_like(masks[0], dtype=np.uint8)
# for i, mask in enumerate(masks):
#     composite[mask > 0] = 255
# cv2.imwrite("composite_mask.png", composite)
import rasterio
from shapely.geometry import shape, mapping
import geopandas as gpd
from skimage import measure

# For one mask:
contours = measure.find_contours(mask, 0.5)
polygons = [shape({'type': 'Polygon', 'coordinates': [c[:, ::-1]]}) for c in contours]

gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")  # Replace with correct CRS
gdf.to_file("juliflora_segmented.geojson", driver="GeoJSON")
