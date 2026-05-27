import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ==========================================
# 1. 2D network architecture
# ==========================================
class EdgeCameraDefectNet2D(nn.Module):
    def __init__(self, in_channels=4):
        super(EdgeCameraDefectNet2D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def extract_2d_features(img_stack):
    f_mean = img_stack.mean(axis=0)
    f_std = img_stack.std(axis=0)
    f_max = img_stack.max(axis=0)
    f_min = img_stack.min(axis=0)
    features = np.stack([f_mean, f_std, f_max, f_min], axis=0)
    for i in range(4):
        features[i] = (features[i] - features[i].mean()) / (features[i].std() + 1e-6)
    return features.astype(np.float32)


# ==========================================
# 2. Correction
# ==========================================
def correct_single_image_median(img, defect_mask):
    H, W = img.shape
    corrected = img.copy().astype(np.float32)
    ys, xs = np.where(defect_mask)

    for y, x in zip(ys, xs):
        # Take a 5x5 neighborhood around the defective pixel
        y_min, y_max = max(0, y - 2), min(H, y + 3)
        x_min, x_max = max(0, x - 2), min(W, x + 3)

        # Extract the local image patch and mask patch
        local_patch = corrected[y_min:y_max, x_min:x_max]
        local_mask = defect_mask[y_min:y_max, x_min:x_max]

        # Sample only from healthy pixels not marked as defective
        valid_pixels = local_patch[~local_mask]

        if len(valid_pixels) > 0:
            # Use median: very smoothing while preserving edges
            corrected[y, x] = np.median(valid_pixels)
        else:
            # Rare case: if all pixels in 5x5 are defective, fall back to horizontal neighbors
            left = max(0, x - 1)
            right = min(W - 1, x + 1)
            corrected[y, x] = (corrected[y, left] + corrected[y, right]) / 2.0

    return corrected.astype(np.uint16)


# ==========================================
# 3. Inference pipeline
# ==========================================
def run_ultimate_inference(model_path="best_defect_net_2d.pth", root_dir="train/VGA"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeCameraDefectNet2D(in_channels=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    seqs = ["sequence_1", "sequence_2", "sequence_3"]
    sims = [
        "low dyn with columns 1",
        "low dyn with columns 2",
        "low dyn with columns 3",
    ]
    root_path = Path(root_dir)

    # Optimization 1: lengthen morphological opening kernel to (9,1)
    # This enforces that detected defects must be vertically continuous for at least 9 pixels
    # and will remove many isolated false positives caused by high-frequency texture
    morph_kernel = np.ones((9, 1), np.uint8)

    with torch.no_grad():
        for seq in seqs:
            for sim in sims:
                folder_path = root_path / seq / sim
                if not folder_path.exists():
                    continue

                img_files = sorted(folder_path.glob("*.png"))
                if not img_files:
                    continue

                print(f"[{seq}/{sim}] Extracting features...")
                img_stack = np.array(
                    [cv2.imread(str(f), -1).astype(np.float32) for f in img_files]
                )
                features = extract_2d_features(img_stack)
                del img_stack  # release large image memory

                x_input = torch.tensor(features).unsqueeze(0).to(device)
                pred_prob = model(x_input).cpu().squeeze().numpy()

                # Optimization 2: raise decision threshold to 0.6 to further reduce FP
                raw_mask = (pred_prob > 0.6).astype(np.uint8)

                # Perform strong vertical opening to remove non-column noise
                clean_mask = cv2.morphologyEx(
                    raw_mask, cv2.MORPH_OPEN, morph_kernel
                ).astype(bool)

                out_folder = folder_path / "result"
                out_folder.mkdir(parents=True, exist_ok=True)

                print(f"  -> Applying local median inpainting...")
                for f_path in img_files:
                    img = cv2.imread(str(f_path), -1)
                    corr_img = correct_single_image_median(img, clean_mask)
                    cv2.imwrite(str(out_folder / f_path.name), corr_img)


if __name__ == "__main__":
    run_ultimate_inference()
