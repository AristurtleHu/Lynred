import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ==========================================
# 1. CNN
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


def extract_1d_features(img_stack):
    # Compute column-wise time series by averaging over height
    col_time_series = img_stack.mean(axis=1)
    f_mean = col_time_series.mean(axis=0)
    f_std = col_time_series.std(axis=0)
    f_max = col_time_series.max(axis=0)
    f_min = col_time_series.min(axis=0)
    features = np.stack([f_mean, f_std, f_max, f_min], axis=0)
    for i in range(4):
        features[i] = (features[i] - features[i].mean()) / (features[i].std() + 1e-6)
    return features.astype(np.float32)


# ==========================================
# 2. Correction algorithm
# ==========================================
def correct_single_image(img, defective_cols):
    H, W = img.shape
    corrected = img.copy()
    for col in defective_cols:
        left = col - 1
        while left in defective_cols and left > 0:
            left -= 1
        right = col + 1
        while right in defective_cols and right < W - 1:
            right += 1

        if left >= 0 and right < W:
            w_left = (right - col) / (right - left)
            w_right = (col - left) / (right - left)
            corrected[:, col] = (
                w_left * corrected[:, left] + w_right * corrected[:, right]
            )
        elif left >= 0:
            corrected[:, col] = corrected[:, left]
        elif right < W:
            corrected[:, col] = corrected[:, right]
    return corrected


# ==========================================
# 3. Inference and save results following original folder structure
# ==========================================
def run_inference_and_save(model_path="best_defect_net.pth", root_dir="train/VGA"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f">>> 1. Loading model weights ({model_path})...")
    model = EdgeCameraDefectNet(in_channels=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Modify which sequences to test here
    seqs = ["sequence_3"]
    sims = [
        "low dyn with columns 1",
        "low dyn with columns 2",
        "low dyn with columns 3",
    ]
    root_path = Path(root_dir)

    with torch.no_grad():
        for seq in seqs:
            for sim in sims:
                folder_path = root_path / seq / sim
                if not folder_path.exists():
                    continue

                print(f"\n>>> Processing: {seq}/{sim}")

                # glob("*.png") is non-recursive
                # won't read images in the generated 'result' subfolder
                img_files = sorted(folder_path.glob("*.png"))
                if not img_files:
                    continue

                print("    - Extracting temporal-spatial features...")
                img_stack = np.array(
                    [cv2.imread(str(f), -1).astype(np.float32) for f in img_files]
                )
                features = extract_1d_features(img_stack)
                del img_stack  # free large memory

                x_input = torch.tensor(features).unsqueeze(0).to(device)  # (1, 4, W)
                pred_prob = model(x_input).cpu().squeeze().numpy()
                pred_defects = np.flatnonzero(pred_prob > 0.5)
                print(
                    f"    - Network predicted defective columns: {pred_defects.tolist()}"
                )

                # Save to 'result' subfolder
                out_folder = folder_path / "result"
                out_folder.mkdir(parents=True, exist_ok=True)
                print(f"    - Correcting frames and saving to: {out_folder} ...")

                for f_path in img_files:
                    img = cv2.imread(str(f_path), -1)
                    corr_img = correct_single_image(img, pred_defects)
                    cv2.imwrite(str(out_folder / f_path.name), corr_img)

    print(
        "\n>>> 🎉 Processing complete! You can now run python metrics.py to evaluate."
    )


if __name__ == "__main__":
    run_inference_and_save()
