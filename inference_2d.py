import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json


# ==========================================
# 1. Pure Detection Diagnostic Evaluator
# ==========================================
def evaluate_pure_detection(mask, json_path, stage_name="Pure AI Vision"):
    """
    Column-level detection evaluation.
    Added `stage_name` parameter to distinguish between Raw Mask and Clean Mask.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract Ground Truth just like metrics.py does
    true_def_cols = set()
    for defect in data:
        x_coords = defect.get("x_coord", [])
        if not isinstance(x_coords, list):
            x_coords = [x_coords]
        true_def_cols.update(x_coords)

    # Extract predicted defective columns from the 2D mask
    col_defect_counts = mask.sum(axis=0)
    pred_def_cols = set(np.where(col_defect_counts > 3)[0])

    # Calculate metrics
    TP = len(pred_def_cols.intersection(true_def_cols))
    FP = len(pred_def_cols.difference(true_def_cols))
    FN = len(true_def_cols.difference(pred_def_cols))

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(
        f"  [{stage_name}] TP: {TP:<3} | FP: {FP:<3} | FN: {FN:<3} ---> Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
    )


# ==========================================
# 2. 2D Network Architecture
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
# 3. Correction
# ==========================================
def correct_single_image(img, defect_mask):
    """Correct defective pixels."""
    H, W = img.shape
    corrected = img.copy().astype(np.float32)
    ys, xs = np.where(defect_mask)

    for y, x in zip(ys, xs):
        left = x - 1
        while left > 0 and defect_mask[y, left]:
            left -= 1

        right = x + 1
        while right < W - 1 and defect_mask[y, right]:
            right += 1

        if left >= 0 and right < W:
            w_left = (right - x) / (right - left)
            w_right = (x - left) / (right - left)
            corrected[y, x] = (
                w_left * corrected[y, left] + w_right * corrected[y, right]
            )
        elif left >= 0:
            corrected[y, x] = corrected[y, left]
        elif right < W:
            corrected[y, x] = corrected[y, right]

    return corrected.astype(np.uint16)


# ==========================================
# 4. Main Inference Pipeline
# ==========================================
def run_inference(model_path="best_defect_net_2d.pth", root_dir="train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeCameraDefectNet2D(in_channels=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sensors = ["VGA", "HD", "SXGA"]
    seqs = ["sequence_1", "sequence_2", "sequence_3"]
    sims = [
        "low dyn with columns 1",
        "low dyn with columns 2",
        "low dyn with columns 3",
    ]
    root_path = Path(root_dir)

    with torch.no_grad():
        for sensor in sensors:
            for seq in seqs:
                for sim in sims:
                    folder_path = root_path / sensor / seq / sim
                    if not folder_path.exists():
                        continue

                    img_files = sorted(folder_path.glob("*.png"))
                    if not img_files:
                        continue

                    print(f"[{sensor}/{seq}/{sim}] Inferencing...")
                    img_stack = np.array(
                        [cv2.imread(str(f), -1).astype(np.float32) for f in img_files]
                    )
                    features = extract_2d_features(img_stack)
                    del img_stack  # release large image memory

                    x_input = torch.tensor(features).unsqueeze(0).to(device)
                    pred_prob = model(x_input).cpu().squeeze().numpy()

                    json_files = list(folder_path.glob("*.json"))
                    json_path = json_files[0] if json_files else None

                    # 1. Threshold
                    raw_mask = (pred_prob > 0.25).astype(np.uint8)

                    # if json_path:
                    #     evaluate_pure_detection(
                    #         raw_mask, json_path, stage_name="1. Raw AI Mask  "
                    #     )

                    # 2. Column-level Filtering based on defect pixel counts

                    col_counts = raw_mask.sum(axis=0)

                    if sensor == "HD":
                        valid_cols = col_counts >= 35
                    elif sensor == "SXGA":
                        valid_cols = col_counts >= 55
                    else:
                        valid_cols = col_counts >= 35

                    # Clean mask after column filtering
                    filtered_mask = raw_mask.copy()
                    filtered_mask[:, ~valid_cols] = 0

                    # if json_path:
                    #     evaluate_pure_detection(
                    #         filtered_mask, json_path, stage_name="2. After Filter "
                    #     )

                    out_folder = folder_path / "result"
                    out_folder.mkdir(parents=True, exist_ok=True)

                    for f_path in img_files:
                        img = cv2.imread(str(f_path), -1)
                        corr_img = correct_single_image(img, filtered_mask)
                        cv2.imwrite(str(out_folder / f_path.name), corr_img)


if __name__ == "__main__":
    run_inference()
