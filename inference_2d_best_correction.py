import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. NN ARCHITECTURE
# ==============================================================================
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
    def forward(self, x): return self.net(x)

def extract_2d_features(img_stack):
    f_mean = img_stack.mean(axis=0)
    f_std = img_stack.std(axis=0)
    f_max = img_stack.max(axis=0)
    f_min = img_stack.min(axis=0)
    features = np.stack([f_mean, f_std, f_max, f_min], axis=0)
    for i in range(4):
        features[i] = (features[i] - features[i].mean()) / (features[i].std() + 1e-6)
    return features.astype(np.float32)

# ==============================================================================
# 2. CORRECTION UTILITIES
# ==============================================================================
def mask_to_col_list(mask: np.ndarray) -> list:
    """Convert 2D boolean mask to sorted list of defective column indices."""
    return sorted(np.where(mask.any(axis=0))[0].tolist())

def get_good_cols(W: int, defective_set: set) -> np.ndarray:
    return np.array([c for c in range(W) if c not in defective_set])

def neighbor_guide(frame_f, col, good_cols, n=3):
    """Estimate a guide signal from healthy neighbor columns."""
    left = good_cols[good_cols < col][-n:]
    right = good_cols[good_cols > col][:n]
    neighbors = np.concatenate([left, right])
    if len(neighbors) == 0: return np.full(frame_f.shape[0], frame_f[:, col].mean(), dtype=np.float32)
    w = 1.0 / (np.abs(neighbors - col).astype(np.float32) + 1e-6)
    w /= w.sum()
    return sum(ww * frame_f[:, nc] for nc, ww in zip(neighbors, w)).astype(np.float32)

def _edge_weight(u, window=16, eps=1e-6):
    H, chi = len(u), np.zeros(len(u), dtype=np.float32)
    for k in range(H):
        s, e = max(0, k - window // 2), min(H, k + window // 2)
        chi[k] = (np.std(u[k:k+1]) + eps) / (np.std(u[s:e]) + eps)
    chi_mean = np.mean(chi) + eps
    eta = 4.0 / (chi_mean - np.min(chi) + eps)
    return 1.0 - 1.0 / (1.0 + np.exp(eta * (chi - chi_mean)))

def _gdgif_1d(x, u, window=16, mu=0.022, eps=1e-6):
    """Gradient Domain Guided Image Filtering (GDGIF) 1D implementation."""
    H = len(x)
    gamma = _edge_weight(u, window, eps)
    a_sum, b_sum, counts = np.zeros(H), np.zeros(H), np.zeros(H)
    for k in range(H):
        s, e = max(0, k - window // 2), min(H, k + window // 2)
        if e - s < 2: continue
        u_w, x_w = u[s:e], x[s:e]
        mean_u, mean_x = u_w.mean(), x_w.mean()
        var_u = u_w.var() + eps
        reg = mu / (np.mean(gamma[s:e]) + eps)
        cov_ux = np.mean(u_w * x_w) - mean_u * mean_x
        a_ck = (cov_ux + reg * gamma[k]) / (var_u + reg)
        b_ck = mean_x - a_ck * mean_u
        a_sum[s:e] += a_ck; b_sum[s:e] += b_ck; counts[s:e] += 1.0
    counts = np.maximum(counts, 1.0)
    return (a_sum / counts) * u + (b_sum / counts)

def correct_side_window(sequence, defective_cols):
    """Side-Window Filter + 1D Guided Filter for stripe noise removal."""
    def swf_1d(signal, r=8):
        H, out = len(signal), np.zeros(len(signal), dtype=np.float32)
        for i in range(H):
            cands = []
            if i - r >= 0: cands.append(signal[i-r:i+1])
            if i + r < H: cands.append(signal[i:i+r+1])
            means = [w.mean() for w in cands]
            out[i] = cands[np.argmin([abs(m - signal[i]) for m in means])].mean()
        return out
    
    corrected = sequence.astype(np.float32).copy()
    T, H, W = corrected.shape
    good_cols = get_good_cols(W, set(defective_cols))
    
    for t in range(T):
        frame = corrected[t]
        for col in defective_cols:
            guide = neighbor_guide(frame, col, good_cols)
            u_smooth = swf_1d(guide, 8)
            stripe = _gdgif_1d(frame[:, col] - u_smooth, u_smooth)
            corrected[t, :, col] = frame[:, col] - stripe
            
    return np.clip(corrected, 0, 65535).astype(np.uint16)

# ==============================================================================
# 3. PIPELINE
# ==============================================================================
def run_pipeline(root_dir="train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeCameraDefectNet2D(in_channels=4).to(device)
    model.load_state_dict(torch.load("best_defect_net_2d.pth", map_location=device))
    model.eval()

    root_path = Path(root_dir)
    # Target all simulation folders
    folders = list(root_path.rglob("low dyn with columns *"))

    with torch.no_grad():
        for folder_path in folders:
            img_files = sorted(folder_path.glob("*.png"))
            if not img_files: continue
            
            print(f"Processing: {folder_path}")
            img_stack = np.array([cv2.imread(str(f), -1).astype(np.float32) for f in img_files])
            
            # Neural Network Detection
            features = extract_2d_features(img_stack)
            x_input = torch.tensor(features).unsqueeze(0).to(device)
            prob = model(x_input).cpu().squeeze().numpy()
            
            # Thresholding and Morphological Cleaning
            raw_mask = (prob > 0.6).astype(np.uint8)
            clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, np.ones((9, 1), np.uint8)).astype(bool)
            
            defective_cols = mask_to_col_list(clean_mask)
            if not defective_cols: 
                print(f" - No defects found in {folder_path.name}")
                continue

            # Correction and Saving to 'result' folder
            out_dir = folder_path / "result"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            print(f" - Applying Side Window correction...")
            corrected = correct_side_window(img_stack.astype(np.uint16), defective_cols)
            
            def save_frame(idx):
                Image.fromarray(corrected[idx]).save(out_dir / img_files[idx].name)
            
            with ThreadPoolExecutor(max_workers=8) as pool:
                pool.map(save_frame, range(len(img_files)))
            
            print(f" - Saved to {out_dir}")

if __name__ == "__main__":
    run_pipeline("train")