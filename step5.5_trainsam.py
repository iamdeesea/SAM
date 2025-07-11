import torch
import os
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

# Configuration
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 10

# Dataset Class
class GeoDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, f"mask_{img_name}")
        
        with rasterio.open(img_path) as img_src:
            image = img_src.read().astype(np.float32)
            image = np.nan_to_num(image)
            image = image / 10000.0  # Normalize NDVI
            image = np.transpose(image, (1, 2, 0))  # CHW to HWC
            
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1).astype(np.float32)
            
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        return image, mask

# Transformations
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=[0.5], std=[0.5]),  # For single-channel NDVI
    ToTensorV2()
])

# Initialize model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(DEVICE)

# Freeze encoder
for param in sam.image_encoder.parameters():
    param.requires_grad = False

# Create datasets
train_dataset = GeoDataset(
    "data/train/images", 
    "data/train/masks",
    transform=transform
)

val_dataset = GeoDataset(
    "data/val/images", 
    "data/val/masks",
    transform=transform
)
print(f"Number of training samples: {len(train_dataset)}")


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    sam.train()
    train_loss = 0.0
    
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        # Forward pass
        with torch.no_grad():
            image_embeddings = sam.image_encoder(images)
            
        pred_masks, _ = sam.mask_decoder(
            image_embeddings=image_embeddings,
            multimask_output=False,
        )
        
        # Compute loss
        loss = criterion(pred_masks.squeeze(1), masks)
        train_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    sam.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            image_embeddings = sam.image_encoder(images)
            pred_masks, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                multimask_output=False,
            )
            loss = criterion(pred_masks.squeeze(1), masks)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f}")

# Save trained model
torch.save(sam.mask_decoder.state_dict(), "trained_mask_decoder.pth")