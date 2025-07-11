import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force GUI window backend

import matplotlib.pyplot as plt

folder = "sam2_training_data"  # folder where samples were saved

# Get list of all sample folders
sample_dirs = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])

if not sample_dirs:
    print("❌ No samples found in", folder)
    exit()

# Load first sample
sample_path = os.path.join(folder, sample_dirs[0])
print("✅ Showing sample:", sample_path)

image = cv2.imread(os.path.join(sample_path, "image.png"))[..., ::-1]
mask = cv2.imread(os.path.join(sample_path, "mask.png"), 0)
with open(os.path.join(sample_path, "prompts.json"), "r") as f:
    prompts = json.load(f)

# Overlay mask
masked = image.copy()
masked[mask > 0] = [255, 0, 0]  # red overlay

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(masked)
plt.title("With Mask & Prompts")

# Plot prompts
for prompt in prompts['positive_points']:
    plt.plot(prompt[0], prompt[1], "go")  # green for positive
for prompt in prompts['negative_points']:
    plt.plot(prompt[0], prompt[1], "ro")  # red for negative

print("✅ Displaying sample...")

try:
    plt.show()
except Exception as e:
    print("❌ Couldn't open window. Saving to 'debug_sample.png' instead.")
    plt.savefig("debug_sample.png")
    print("✅ Saved visualization to debug_sample.png")
