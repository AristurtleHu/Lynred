import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


IMAGE_FOLDER = r"train\VGA\sequence_1\low dyn with columns 1"
EXTENSION = "*.png"

OUTPUT_FOLDER = r"train\VGA\sequence_1\low dyn with columns 1\result"

STD_THRESHOLD_SIGMA = 3.0

DISPLAY_RESULTS = True

paths = sorted(
    glob.glob(
        os.path.join(IMAGE_FOLDER, EXTENSION)
    )
)

if len(paths) == 0:
    raise RuntimeError("No images found")


# Load frames
frames = []
for p in paths:

    img = cv2.imread(
        p,
        cv2.IMREAD_UNCHANGED
    )

    if img is None:
        continue

    frames.append(img.astype(np.float32))

# Set axis for time
stack = np.stack(frames, axis=0)
T, H, W = stack.shape

# Calculate standard deviation across time for each pixel over time
temporal_std = np.std(stack, axis=0)

temporal_med= np.median(
    np.abs(stack - np.median(stack, axis=0)),
    axis=0
)

# Calculate average temporal noise per column
column_noise = np.mean(temporal_std, axis=0)

mean_noise = np.mean(column_noise)
std_noise = np.std(column_noise)

temporal_median = np.median(stack, axis=0)
temporal_metric = np.median(np.abs(stack - temporal_median), axis=0)
column_noise = np.mean(temporal_metric, axis=0)

# Calculate a thresholds for defected columns
threshold = mean_noise + STD_THRESHOLD_SIGMA * std_noise
lower_threshold = mean_noise - 3 * std_noise
bad_columns = np.where(column_noise > threshold)[0]
#bad_columns = np.where((column_noise > threshold) | (column_noise < lower_threshold))[0]

# Visualise defects
plt.figure(figsize=(14, 5))
plt.plot(column_noise, label="Column Temporal Noise")
plt.axhline(
    threshold,
    color='r',
    linestyle='--',
    label='Threshold'
)
plt.title("Column Defect Detection")
plt.xlabel("Column")
plt.ylabel("Temporal Std Dev")

for c in bad_columns:
    plt.axvline(c, color='r', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Correct columns by replacing with average of neighbours
corrected_frames = []
for idx, img in enumerate(stack):

    corrected = img.copy()

    for c in bad_columns:

        # Skip border columns
        if c == 0 or c == W - 1:
            continue

        # Interpolate from left/right neighbours
        corrected[:, c] = 0.5 * (
            img[:, c - 1] +
            img[:, c + 1]
        )

    corrected_frames.append(corrected)

    out = np.clip(corrected, 0, 65535).astype(np.uint16)
    

    filename = os.path.join(
        OUTPUT_FOLDER,
        f"frame_{idx:04d}.png"
    )

    cv2.imwrite(filename, out)

print("\nCorrected frames saved")
