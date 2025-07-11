import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, jaccard_score
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from pathlib import Path
import cv2

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 512
EPOCHS_P1 = 20
EPOCHS_P2 = 10
BATCH = 8
SAM_CKPT = "sam_vit_h.pth"
DATA_DIR = Path("patches")

# ---------------- DATA ----------------
train_npz = np.load(DATA_DIR / "train.npz")
val_npz = np.load(DATA_DIR / "val.npz")

train_ds = TensorDataset(torch.tensor(train_npz["images"]), torch.tensor(train_npz["masks"]))
val_ds = TensorDataset(torch.tensor(val_npz["images"]), torch.tensor(val_npz["masks"]))

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

# ---------------- MODEL ----------------
sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(DEVICE)
predictor = SamPredictor(sam)

# ... [previous code remains the same]

# ... [previous code remains the same]

# ... [previous code remains the same]

class UNetDecoder(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.enc = predictor.model.image_encoder
        self.tfm = predictor.transform
        self.conv1 = nn.Sequential(
            nn.Conv2d(1280, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1))
        
        # Get normalization parameters from SAM model
        self.register_buffer("pixel_mean", predictor.model.pixel_mean)
        self.register_buffer("pixel_std", predictor.model.pixel_std)

    def forward(self, x):
        batch_size = x.shape[0]
        # Process entire batch at once
        img_np = (x * 255).permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
        processed_batch = []
        
        for i in range(batch_size):
            img_resized = cv2.resize(img_np[i], (1024, 1024))
            img_proc = self.tfm.apply_image(img_resized)
            processed_batch.append(img_proc.transpose(2, 0, 1))
        
        img_tensor = torch.tensor(np.array(processed_batch), 
                                 dtype=torch.float32).to(x.device)
        img_tensor = (img_tensor - self.pixel_mean) / self.pixel_std
        
        # Get features
        with torch.no_grad():
            features = self.enc(img_tensor)  # [B, 1280, 64, 64]
        
        # Decoding steps
        y = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.conv1(y)  # [B, 256, 128, 128]
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.conv2(y)  # [B, 128, 256, 256]
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.conv3(y)  # [B, 1, 512, 512]
        
        return y

# ... [rest of the code remains the same]

# ... [rest of the code remains the same]

# ... [rest of the code remains the same]

model = UNetDecoder(predictor).to(DEVICE)

# ---------------- LOSS ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def combined_loss(pred, target):
    return 0.5 * FocalLoss()(pred, target) + 0.5 * dice_loss(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
best_iou = 0

# ---------------- TRAINING ----------------
def evaluate(model, loader):
    model.eval()
    total_loss, total_iou, total_f1 = 0, 0, 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.unsqueeze(1).to(DEVICE)
            logits = model(imgs)
            loss = combined_loss(logits, masks)
            preds = (torch.sigmoid(logits) > 0.5).int()
            total_loss += loss.item()
            total_iou += jaccard_score(masks.cpu().flatten(), preds.cpu().flatten())
            total_f1 += f1_score(masks.cpu().flatten(), preds.cpu().flatten())
    n = len(loader)
    return total_loss / n, total_iou / n, total_f1 / n

for epoch in range(1, EPOCHS_P1 + EPOCHS_P2 + 1):
    if epoch == EPOCHS_P1 + 1:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
        print("\n↓ Learning Rate dropped to 1e-5")

    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch}"):
        imgs = imgs.to(DEVICE)
        masks = masks.unsqueeze(1).to(DEVICE)
        logits = model(imgs)
        loss = combined_loss(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss, val_iou, val_f1 = evaluate(model, val_dl)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss / len(train_dl):.4f} | Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | F1: {val_f1:.4f}")
    torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pth")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "best_sam_model.pth")

print(f"\n✓ Best IoU: {best_iou:.4f} — Model saved as best_sam_model.pth")
