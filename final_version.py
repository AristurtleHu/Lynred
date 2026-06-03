import numpy as np
from skimage import io, util
from scipy.ndimage import uniform_filter1d
import os

def detect_columns(img):
    median = np.median(img, axis=0)
    smoothed = uniform_filter1d(median, size=20)
    ratio = median - smoothed 
    ratio = np.abs(ratio)
    threshold = 3 * np.std(ratio) + np.mean(ratio) 
    noisy_cols = np.where(ratio > threshold)[0]
    return noisy_cols
    
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
    H, W = corrected.shape
    good_cols = get_good_cols(W, set(defective_cols))
    
    frame = corrected
    for col in defective_cols:
        guide = neighbor_guide(frame, col, good_cols)
        u_smooth = swf_1d(guide, 8)
        stripe = _gdgif_1d(frame[:, col] - u_smooth, u_smooth)
        corrected[:, col] = frame[:, col] - stripe
            
    return np.clip(corrected, 0, 65535).astype(np.uint16)

def fix_img(sensor, seq, sim):
    def_path = os.path.join(seq,
                             f"low dyn with columns {sim}")
    res_path = os.path.join(def_path, "result")
    os.makedirs(res_path, exist_ok=True)
    def_files = sorted([os.path.join(def_path, f) for f in os.listdir(def_path) if f.lower().endswith(".png")])
    arr_img = []
    for i in def_files:
        arr_img.append(util.img_as_uint(io.imread(i)))
    arr_img = np.asarray(arr_img)
    for i, img in enumerate(arr_img):
        col = detect_columns(img)
        res = correct_side_window(img, col)
        
        img_path = os.path.join(res_path, f"frame_{i:04d}.png")
        io.imsave(img_path, res, check_contrast=False)

root_path = r"train"
sensors = ["HD", "SXGA", "VGA"]
for sensor in sensors:
    sensor_path = os.path.join(root_path, sensor)

    for entry in os.scandir(sensor_path):
        if not entry.is_dir():
            continue
        seq = entry.path
        for sim in (1, 2, 3 ):
            fix_img(sensor, seq, sim) 
