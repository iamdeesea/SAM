import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np  # Added for array operations
import os
import sys

# Either use this (if installed in dev mode):
from segment_anything_2.sam2.modeling.sam2_base import SAM2

# # OR if that doesn't work:
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from sam2.modeling.sam2_base import SAM2


# ---------- Dataset Loader ----------
class SAM2TrainingDataset(Dataset):
    def __init__(self, data_dir):
        self.sample_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.transform = T.Compose([
            T.ToTensor(),  # Already tensor, just normalize
        ])

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_paths[idx], weights_only=False)
        image = sample["image"]
        mask = sample["mask"]
        prompts = sample["prompts"]  # For now, not used

        # Convert grayscale to 3-channel if needed
        if image.ndim == 2:
            image = np.stack([image]*3, axis=0)

        # Ensure shape is (3, H, W)
        if image.shape[0] != 3:
            image = image.transpose(2, 0, 1)

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            "prompts": prompts  # optional for fine-tuning with prompt guidance
        }

# ---------- Training Function ----------
def train(model, dataloader, device, epochs=10, lr=1e-4):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # for binary masks

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)["masks"]  # adjust if your model uses different keys
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"✅ Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "sam2_prosopis_model.pt")
    print("✅ Model saved as sam2_prosopis_model.pt")

# ---------- Main ----------
if __name__ == "__main__":
    dataset = SAM2TrainingDataset("sam2_training_data")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SAM2()  # replace with your actual SAM2 initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(model, dataloader, device)