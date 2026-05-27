import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# ==========================================
# 1. Define 1D-CNN
# ==========================================
class EdgeCameraDefectNet(nn.Module):
    def __init__(self, in_channels=4):
        super(EdgeCameraDefectNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. Utilities
# ==========================================
def extract_1d_features(img_stack):
    """
    Reduce an image stack of shape (Time, Height, Width)
    to column features of shape (4, Width).
    """
    # Average over the height to obtain a time series per column
    col_time_series = img_stack.mean(axis=1)  # Shape: (Time, Width)

    # Extract four temporal statistics
    f_mean = col_time_series.mean(axis=0)
    f_std = col_time_series.std(axis=0)
    f_max = col_time_series.max(axis=0)
    f_min = col_time_series.min(axis=0)

    features = np.stack([f_mean, f_std, f_max, f_min], axis=0)  # Shape: (4, W)

    # Standardization
    for i in range(4):
        features[i] = (features[i] - features[i].mean()) / (features[i].std() + 1e-6)

    return features.astype(np.float32)


def load_labels_from_json(json_path, width=640):
    """Parse JSON to obtain the ground-truth defective column mask."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    indices = []
    for c in data:
        if isinstance(c["x_coord"], list):
            indices.extend(c["x_coord"])
        else:
            indices.append(c["x_coord"])

    label = np.zeros(width, dtype=np.float32)
    label[indices] = 1.0
    return label, indices


def load_data_from_sequences(root_path, sequences, simulations, img_width=640):
    """Helper: load features and labels for the specified sequences."""
    features_list, labels_list, meta_list = [], [], []
    for seq in sequences:
        for sim in simulations:
            folder_path = root_path / seq / sim
            if not folder_path.exists():
                continue

            img_files = sorted(folder_path.glob("*.png"))
            if not img_files:
                continue

            img_stack = np.array(
                [cv2.imread(str(f), -1).astype(np.float32) for f in img_files]
            )
            features = extract_1d_features(img_stack)

            json_files = list(folder_path.glob("*.json"))
            if not json_files:
                continue
            label, true_defects = load_labels_from_json(json_files[0], img_width)

            features_list.append(features)
            labels_list.append(label)
            meta_list.append(
                {
                    "folder": folder_path,
                    "img_stack": img_stack,
                    "true_defects": true_defects,
                    "img_files": img_files,
                }
            )

    if not features_list:
        return None, None, []

    X = torch.tensor(np.array(features_list))  # (N, 4, 640)
    Y = torch.tensor(np.array(labels_list)).unsqueeze(1)  # (N, 1, 640)
    return X, Y, meta_list


# ==========================================
# 3. Train
# ==========================================
def run_neural_pipeline_with_eval(root_dir="train/VGA", img_width=640):
    root_path = Path(root_dir)
    simulations = [
        "low dyn with columns 1",
        "low dyn with columns 2",
        "low dyn with columns 3",
    ]

    # ==========================================
    # 1. First two sequences for training, last sequence for validation
    # ==========================================
    train_seqs = ["sequence_1", "sequence_2"]
    val_seqs = ["sequence_3"]

    print(">>> 1. Extracting training and validation features...")
    X_train, Y_train, meta_train = load_data_from_sequences(
        root_path, train_seqs, simulations, img_width
    )
    X_val, Y_val, meta_val = load_data_from_sequences(
        root_path, val_seqs, simulations, img_width
    )

    if X_train is None or X_val is None:
        raise ValueError("Not enough data found; please check the paths.")

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=4, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=4, shuffle=False)

    # ==========================================
    # 2. Network initialization and training setup
    # ==========================================
    print(f">>> 2. 开始训练 (Train: {len(X_train)}组, Val: {len(X_val)}组)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeCameraDefectNet(in_channels=4).to(device)
    criterion = nn.BCELoss()
    # L2 regularization to prevent overfitting on small datasets
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    epochs = 200
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_weights = None

    # ==========================================
    # 3. Training loop with validation
    # ==========================================
    for epoch in range(epochs):
        # --- Training phase ---
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

        # --- Validation phase ---
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

        # --- Save best model ---
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            best_model_weights = model.state_dict().copy()

        if (epoch + 1) % 50 == 0:
            print(
                f"    Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_t_loss:.4f} | Val Loss: {avg_v_loss:.4f}"
            )

    # ==========================================
    # 4. Visualize training curve
    # ==========================================
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f">>> 3. Training complete! Best Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_weights)  # restore best weights

    torch.save(best_model_weights, "best_defect_net.pth")
    print(">>> Weights saved to best_defect_net.pth")


# Run main
if __name__ == "__main__":
    run_neural_pipeline_with_eval(root_dir="train/VGA", img_width=640)
