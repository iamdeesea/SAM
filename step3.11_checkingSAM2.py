import sys
import torch
sys.path.append('segment-anything-2')  # Adjust this as needed

from sam2.build_sam import build_sam2  # ✅ FIXED

checkpoint_path = 'segment-anything-2/checkpoints/sam2_hiera_large.pt'
cfg = 'segment-anything-2/configs/sam2_hiera_l.yaml'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sam2 = build_sam2(cfg, checkpoint_path, device=device, apply_postprocessing=False)
sam2.eval()

print("✅ SAM2 loaded successfully!")
