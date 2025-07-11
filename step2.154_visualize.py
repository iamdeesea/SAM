import os
import glob
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_samples(path):
    sample_files = sorted(glob.glob(os.path.join(path, "*.pt")))
    
    if not sample_files:
        raise ValueError("❌ No samples found in the provided directory.")

    for sample_file in sample_files:
        print(f"📂 Loading {sample_file}")
        data = torch.load(sample_file, weights_only=False)

        # Extract image and mask
        image = data["image"]
        mask = data["mask"]

        # Ensure image is in HWC format (Height, Width, Channels)
        if image.ndim == 3:
            if image.shape[0] == 3:  # CHW format
                image = image.transpose(1, 2, 0)
            elif image.shape[1] == 3:  # HWC format but channels in middle
                image = image.transpose(0, 2, 1)
            elif image.shape[2] != 3:  # No channel dimension size 3 found
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            # Handle 2D grayscale images by converting to RGB
            image = np.stack([image]*3, axis=-1)

        # Resize mask to match image dimensions
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), 
                             (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)

        # Create overlay with proper broadcasting
        overlay = image.copy()
        mask_bool = mask > 0  # Ensure mask is boolean
        
        # Apply red overlay using broadcasting
        red_channel = np.where(mask_bool, 255, overlay[..., 0])
        green_channel = np.where(mask_bool, 0, overlay[..., 1])
        blue_channel = np.where(mask_bool, 0, overlay[..., 2])
        
        overlay = np.stack([red_channel, green_channel, blue_channel], axis=-1)

        # Visualization
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="gray")
        axs[1].set_title("Binary Mask")
        axs[1].axis("off")

        axs[2].imshow(overlay)
        axs[2].set_title("Overlay (Red Mask)")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

        if sample_file.endswith("sample_004.pt"):
            break

if __name__ == "__main__":
    visualize_samples("sam2_training_data")