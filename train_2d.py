import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ==========================================
# 1. 2D asymmetric CNN
# ==========================================
class EdgeCameraDefectNet2D(nn.Module):
    def __init__(self, in_channels=4):
        super(EdgeCameraDefectNet2D, self).__init__()
        # Use asymmetric convolution (tall vertically, narrow horizontally)
        # This captures vertical line features while saving compute
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Output channel is 1, producing a 2D defect mask
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. 2D feature extraction
# ==========================================
def extract_2d_features(img_stack):
    """
    Keep the height H dimension and extract temporal statistics per pixel.
    Input shape: (Time, H, W)
    Output shape: (4, H, W)
    """
    f_mean = img_stack.mean(axis=0)
    f_std = img_stack.std(axis=0)
    f_max = img_stack.max(axis=0)
    f_min = img_stack.min(axis=0)

    features = np.stack([f_mean, f_std, f_max, f_min], axis=0)

    # Normalize spatially per channel
    for i in range(4):
        features[i] = (features[i] - features[i].mean()) / (features[i].std() + 1e-6)

    return features.astype(np.float32)


def load_2d_labels_from_json(json_path, H, W):
    """Generate a 2D label mask of shape (H, W)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label = np.zeros((H, W), dtype=np.float32)
    indices = set()

    for defect in data:
        x_coords = defect.get("x_coord", [])
        if not isinstance(x_coords, list):
            x_coords = [x_coords]
        indices.update(x_coords)

        y_starts = defect.get("y_start", [0])
        y_stops = defect.get("y_stop", [H])

        # Cross-product: Every X column in this cluster shares ALL the Y segments
        for x in x_coords:
            if 0 <= x < W:
                for y0, y1 in zip(y_starts, y_stops):
                    y0 = max(0, int(y0))
                    y1 = min(H, int(y1))
                    label[y0:y1, x] = 1.0

    return label, list(indices)


# ==========================================
# 3. Dynamic random-crop Dataset
# ==========================================
class RandomPatchDataset(Dataset):
    def __init__(
        self, features_list, labels_list, patch_size=(128, 128), patches_per_img=20
    ):
        self.features_list = features_list
        self.labels_list = labels_list
        self.patch_size = patch_size
        self.patches_per_img = patches_per_img
        self.num_images = len(features_list)

    def __len__(self):
        # For each image, we randomly extract patches_per_img patches for training
        return self.num_images * self.patches_per_img

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_img
        features = self.features_list[img_idx]  # (4, H, W)
        label = self.labels_list[img_idx]  # (H, W)

        _, H, W = features.shape
        ph, pw = self.patch_size

        # Randomly generate top-left coordinates for the crop
        y = np.random.randint(0, max(1, H - ph + 1))
        x = np.random.randint(0, max(1, W - pw + 1))

        # Slice to extract the patch
        feat_patch = features[:, y : y + ph, x : x + pw]
        label_patch = label[y : y + ph, x : x + pw]

        return torch.tensor(feat_patch), torch.tensor(label_patch).unsqueeze(0)


# ==========================================
# 4. Data loading helper functions
# ==========================================
def load_2d_data(root_path, sequences, simulations):
    features_list, labels_list, meta_list = [], [], []
    for seq in sequences:
        for sim in simulations:
            folder_path = Path(root_path) / seq / sim
            if not folder_path.exists():
                continue

            img_files = sorted(folder_path.glob("*.png"))
            if not img_files:
                continue

            # Read image stack
            img_stack = np.array(
                [cv2.imread(str(f), -1).astype(np.float32) for f in img_files]
            )
            _, H, W = img_stack.shape

            # Extract 2D features
            features = extract_2d_features(img_stack)

            json_files = list(folder_path.glob("*.json"))
            if not json_files:
                continue
            label, true_defects = load_2d_labels_from_json(json_files[0], H, W)

            features_list.append(features)
            labels_list.append(label)
            meta_list.append({"folder": folder_path, "true_defects": true_defects})

    return features_list, labels_list, meta_list


# ==========================================
# 5. Main training and evaluation pipeline
# ==========================================
def run_2d_neural_pipeline(root_dir="train"):
    sims = [
        "low dyn with columns 1",
        "low dyn with columns 2",
        "low dyn with columns 3",
    ]

    print(">>> 1. Extracting 2D features (VGA, SXGA, HD)...")

    # Get VGA
    X_train_v, Y_train_v, _ = load_2d_data(
        f"{root_dir}/VGA", ["sequence_1", "sequence_2"], sims
    )
    X_val_v, Y_val_v, _ = load_2d_data(f"{root_dir}/VGA", ["sequence_3"], sims)

    # Get SXGA
    X_train_s, Y_train_s, _ = load_2d_data(
        f"{root_dir}/SXGA", ["sequence_1", "sequence_2"], sims
    )
    X_val_s, Y_val_s, _ = load_2d_data(f"{root_dir}/SXGA", ["sequence_3"], sims)

    # Get HD
    X_train_h, Y_train_h, _ = load_2d_data(
        f"{root_dir}/HD", ["sequence_1", "sequence_2"], sims
    )
    X_val_h, Y_val_h, _ = load_2d_data(f"{root_dir}/HD", ["sequence_3"], sims)

    # Mix all sensors together for training and validation
    X_train = X_train_v + X_train_s + X_train_h
    Y_train = Y_train_v + Y_train_s + Y_train_h

    X_val = X_val_v + X_val_s + X_val_h
    Y_val = Y_val_v + Y_val_s + Y_val_h

    train_dataset = RandomPatchDataset(
        X_train, Y_train, patch_size=(128, 128), patches_per_img=30
    )
    val_dataset = RandomPatchDataset(
        X_val, Y_val, patch_size=(128, 128), patches_per_img=10
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(
        f">>> 2. Starting 2D training (Train Patches: {len(train_dataset)}, Val Patches: {len(val_dataset)})..."
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeCameraDefectNet2D(in_channels=4).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 100
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        avg_t_loss = t_loss / len(train_loader)
        train_losses.append(avg_t_loss)

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                v_loss += loss.item()

        avg_v_loss = v_loss / len(val_loader)
        val_losses.append(avg_v_loss)

        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            best_model_weights = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(
                f"    Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_t_loss:.4f} | Val Loss: {avg_v_loss:.4f}"
            )

    print(f">>> 3. Training complete! Best Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_weights)
    torch.save(best_model_weights, "best_defect_net_2d.pth")
    print(">>> Weights saved to best_defect_net_2d.pth")

    # Draw training curves
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("2D Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_2d_neural_pipeline(root_dir="train")
