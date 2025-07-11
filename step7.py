import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from segment_anything import sam_model_registry
from step6 import train_loader, val_loader
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F

# 1. Load Pretrained SAM ----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "sam_vit_h.pth"  # Download from Meta's SAM repo if not present

sam_model = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH).to(DEVICE)
predictor = SamPredictor(sam_model)

# 2. Replace mask decoder with a binary head --------------------------------
class SAMBinaryHead(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.encoder = predictor.model.image_encoder  # ViT-H encoder
        self.transform = predictor.transform          # ResizeLongestSide

        # Input channels should match the encoder's output channels (256 for ViT-H)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        B = x.size(0)
        outs = []

        for i in range(B):
            # Convert image to HWC uint8
            img_np = (x[i].clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().astype("uint8")

            # Apply SAM's transform
            img_proc = self.transform.apply_image(img_np)
            img_tensor = torch.tensor(img_proc.transpose(2, 0, 1), dtype=torch.float32) \
                            .unsqueeze(0).to(x.device) / 255.0

            with torch.no_grad():
                features = self.encoder(img_tensor)  # This directly outputs the embeddings tensor
                
            logits = self.decoder(features)
            logits = F.interpolate(logits, size=(1024, 1024),
                                 mode="bilinear", align_corners=False)
            outs.append(logits)

        return torch.cat(outs, dim=0)  # (B, 1, 1024, 1024)

model = SAMBinaryHead(predictor).to(DEVICE)

# 3. Loss and optimizer -----------------------------------------------------
def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

bce = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return 0.5 * bce(pred, target.float()) + 0.5 * dice_loss(pred, target)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4. Training loop ----------------------------------------------------------
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        masks = masks.unsqueeze(1).to(DEVICE)  # B×1×H×W

        preds = model(imgs)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Train Loss: {train_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.unsqueeze(1).to(DEVICE)
            preds = model(imgs)
            loss = combined_loss(preds, masks)
            val_loss += loss.item()
        print(f"Val Loss: {val_loss / len(val_loader):.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"sam_binary_epoch{epoch+1}.pth")