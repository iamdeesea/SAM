#!/usr/bin/env python
import os, torch, matplotlib.pyplot as plt, numpy as np
from sklearn.metrics import f1_score
from step6 import val_loader
from step7 import SAMBinaryHead         # your model class
from segment_anything import sam_model_registry, SamPredictor

# ------------------------------------------------------------------
# 1.  Load trained decoder + frozen SAM encoder
# ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT   = "sam_binary_epoch10.pth"                 # choose your best epoch
sam    = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth").to(DEVICE)
predictor = SamPredictor(sam)
model  = SAMBinaryHead(predictor).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# ------------------------------------------------------------------
# 2.  Output dir
# ------------------------------------------------------------------
os.makedirs("visualizations", exist_ok=True)

# ------------------------------------------------------------------
# 3.  Loop over validation loader
# ------------------------------------------------------------------
with torch.no_grad():
    for batch_idx, (imgs, masks) in enumerate(val_loader):
        imgs   = imgs.to(DEVICE)                 # (B,3,1024,1024)
        preds  = model(imgs)                     # (B,1,1024,1024)

        for i in range(len(imgs)):
            ndvi_rgb = imgs[i].cpu().permute(1,2,0).numpy()
            gt_mask  = masks[i].cpu().numpy()            # (1024,1024)
            pr_mask  = (torch.sigmoid(preds[i]).cpu().numpy()[0] > 0.5).astype(np.uint8)

            # ------------ F1 score ------------
            f1 = f1_score(gt_mask.flatten(), pr_mask.flatten(), zero_division=1)

            # ------------ create overlay ------------
            overlay = ndvi_rgb.copy()
            overlay[..., 0] = np.where(gt_mask==1, 1.0, overlay[...,0])   # Red  channel = GT
            overlay[..., 2] = np.where(pr_mask==1, 1.0, overlay[...,2])   # Blue channel = Pred
            # Where both GT and Pred are 1 => magenta (R+B)

            # ------------ plot & save ------------
            fig, axs = plt.subplots(1,3, figsize=(13,4))
            axs[0].imshow(ndvi_rgb);                axs[0].set_title("NDVI RGB")
            axs[1].imshow(gt_mask, cmap="gray");    axs[1].set_title("Ground Truth")
            axs[2].imshow(overlay);                 axs[2].set_title(f"Overlay  |  F1={f1:.3f}")
            for ax in axs: ax.axis("off")

            idx = batch_idx*len(imgs)+i
            plt.tight_layout()
            plt.savefig(f"visualizations/vis_epoch10_sample{idx}.png", dpi=150)
            plt.close()

print("✅  Visuals (with F1) saved in  ./visualizations/")
