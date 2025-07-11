import os
import torch

# Path to one of your training samples
sample_path = os.path.join("sam2_training_data", "sample_000.pt")

# Load the sample
data = torch.load(sample_path, weights_only=False)

# Print what's inside the sample
print("✅ Loaded sample_000.pt")
print("Available keys in the sample:")
for key in data.keys():
    print(f"  - {key} → type: {type(data[key])}")
