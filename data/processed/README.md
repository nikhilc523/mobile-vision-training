# Processed Data - Training-Ready Datasets

This folder contains final processed datasets ready for LSTM training. All data has been feature-engineered, windowed, and balanced.

---

## 📁 Files Overview

```
processed/
├── all_windows_30frame_raw.npz              # Main training dataset ⭐
├── all_windows_30frame_raw_metadata.json    # Dataset metadata
├── all_windows_30_raw_balanced.npz          # Balanced version (16/84 split)
├── all_windows_30_raw_balanced_hnm.npz      # With hard negative mining
├── all_windows_30_physics5.npz              # Physics-based features
├── all_windows_enhanced.npz                 # Enhanced features (10 features)
├── all_windows_full.npz                     # Full feature set
├── all_windows_v2.npz                       # Version 2 features
├── all_windows_60frame.npz                  # 60-frame windows (2 seconds)
├── urfd_windows.npz                         # URFD only
├── le2i_windows.npz                         # Le2i only
└── ucf101_windows.npz                       # UCF101 only
```

---

## ⭐ **Main Training Dataset**

### `all_windows_30frame_raw.npz`

**Description:** Primary training dataset with 30-frame windows and raw keypoint coordinates.

**Format:**
```python
{
    'X': np.array(shape=(N, 30, 34), dtype=float32),
    'y': np.array(shape=(N,), dtype=int),
    'video_names': list of str,
    'window_indices': list of int
}
```

**Details:**
- **Windows (N):** ~15,000 windows
- **Frames per Window:** 30 (1 second at 30 FPS)
- **Features per Frame:** 34 values
  - 17 keypoints × 2 coordinates (y, x)
  - Confidence values included in separate channel
- **Stride:** 10 frames (0.33 seconds overlap)
- **Labels:** 0 = non-fall, 1 = fall
- **Class Distribution:** ~16% fall, ~84% non-fall

**Feature Structure (34 values per frame):**
```
[y0, x0, y1, x1, ..., y16, x16]  # 17 keypoints × 2 = 34 values

Where:
- y0, x0 = nose position
- y1, x1 = left_eye position
- ...
- y16, x16 = right_ankle position
```

**Statistics:**
```
Total Windows: ~15,000
├── Fall Windows: ~2,400 (16%)
└── Non-Fall Windows: ~12,600 (84%)

Input Shape: (30, 34)
├── 30 frames (1 second)
└── 34 values per frame (17 keypoints × 2)

Size: ~20 MB compressed
```

---

## 📊 Dataset Variants

### `all_windows_30_raw_balanced.npz`

**Description:** Balanced version with realistic fall/non-fall ratio.

**Class Distribution:**
- Fall: 16%
- Non-fall: 84%

**Use Case:** Main training dataset (used for final model)

---

### `all_windows_30_raw_balanced_hnm.npz`

**Description:** Balanced dataset with hard negative mining.

**Features:**
- Includes challenging non-fall examples
- Reduces false positives
- Improves model robustness

**Use Case:** Advanced training with difficult examples

---

### `all_windows_30_physics5.npz`

**Description:** Physics-based features instead of raw keypoints.

**Features (5 per frame):**
1. Body height (hip_y - nose_y)
2. Hip velocity
3. Body orientation angle
4. Center of mass position
5. Keypoint spread (bounding box area)

**Input Shape:** (30, 5)

**Use Case:** Experimental - physics-informed model

---

### `all_windows_enhanced.npz`

**Description:** Enhanced feature set with 10 engineered features.

**Features (10 per frame):**
1. Body height
2. Hip velocity
3. Body orientation
4. Center of mass
5. Keypoint spread
6. Head velocity
7. Limb angles
8. Pose confidence
9. Bounding box area
10. Vertical acceleration

**Input Shape:** (30, 10)

**Use Case:** Feature-engineered model (Week 3 experiments)

---

### `all_windows_60frame.npz`

**Description:** 60-frame windows (2 seconds) for longer temporal context.

**Input Shape:** (60, 34)

**Use Case:** Experimental - longer context windows

---

### Dataset-Specific Files

**`urfd_windows.npz`** - URFD only (~1,000 windows)  
**`le2i_windows.npz`** - Le2i only (~3,000 windows)  
**`ucf101_windows.npz`** - UCF101 only (~11,000 windows)

**Use Case:** Dataset-specific analysis and ablation studies

---

## 🔧 How to Use

### Load Training Data

```python
import numpy as np

# Load main dataset
data = np.load('data/processed/all_windows_30frame_raw.npz')

X = data['X']  # Shape: (N, 30, 34)
y = data['y']  # Shape: (N,)

print(f"Total windows: {X.shape[0]}")
print(f"Fall windows: {np.sum(y == 1)} ({np.mean(y) * 100:.1f}%)")
print(f"Non-fall windows: {np.sum(y == 0)} ({(1 - np.mean(y)) * 100:.1f}%)")
```

### Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

# 80/10/10 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape[0]} windows")
print(f"Val: {X_val.shape[0]} windows")
print(f"Test: {X_test.shape[0]} windows")
```

### Load with TensorFlow

```python
import tensorflow as tf

# Load data
data = np.load('data/processed/all_windows_30frame_raw.npz')
X, y = data['X'], data['y']

# Create TF dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

# Use in training
model.fit(dataset, epochs=100)
```

---

## 📈 Data Statistics

### Window Generation

```
Source Videos: 964
├── URFD: 63 videos
├── Le2i: 190 videos
└── UCF101: 711 videos

Total Frames: 197,411
├── Fall Frames: 34,801 (17.6%)
└── Non-Fall Frames: 162,610 (82.4%)

Windows Generated: ~15,000
├── Window Size: 30 frames (1 second)
├── Stride: 10 frames (0.33 seconds)
└── Overlap: 66.7%

Quality Filtering:
├── Minimum Keypoints: 30% per frame
├── Minimum Frames: 30 consecutive frames
└── Filtered Out: ~5% of potential windows
```

### Class Distribution

```
Final Dataset (Balanced):
├── Fall Windows: 2,400 (16%)
└── Non-Fall Windows: 12,600 (84%)

Realistic Distribution:
- Matches real-world fall frequency
- Prevents model from always predicting "non-fall"
- Improves precision/recall balance
```

---

## 🚀 Regenerate Processed Data

If you need to regenerate the processed datasets:

```bash
# Create 30-frame raw keypoint windows
python ml/features/create_30frame_raw_keypoints.py \
    --input data/interim/keypoints/ \
    --output data/processed/all_windows_30frame_raw.npz \
    --window-size 30 \
    --stride 10

# Create balanced version
python ml/features/balance_dataset.py \
    --input data/processed/all_windows_30frame_raw.npz \
    --output data/processed/all_windows_30_raw_balanced.npz \
    --fall-ratio 0.16

# Create enhanced features
python ml/features/feature_engineering_enhanced.py \
    --input data/interim/keypoints/ \
    --output data/processed/all_windows_enhanced.npz
```

---

## ⚠️ Important Notes

1. **In GitHub:** These files ARE included in GitHub (small, ~50 MB total)
2. **Ready to Use:** No regeneration needed for training
3. **Balanced:** Main dataset is balanced for realistic distribution
4. **Validated:** All files passed integrity checks
5. **Metadata:** JSON file contains dataset statistics and parameters

---

## 📚 References

- **Feature Engineering:** `ml/features/create_30frame_raw_keypoints.py`
- **Balancing:** `ml/features/balance_dataset.py`
- **Training:** `ml/training/lstm_train_raw_keypoints.py`
- **Week 3 Report:** `docs/weekly_reports/WEEK_3_REPORT.md`

---

**Last Updated:** December 2024

