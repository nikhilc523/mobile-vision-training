# Feature Engineering Module

## Overview

The `ml.features` module provides feature extraction and windowing utilities for converting pose keypoints into LSTM-ready training data.

## Features Extracted

From MoveNet pose keypoints (17 keypoints × 3 values), we extract 6 engineered features per frame:

1. **Torso Angle (α)** - Angle between neck-hip line and vertical (degrees)
   - Uses midpoint of shoulders as neck proxy
   - Measures body tilt/orientation

2. **Hip Height (h)** - Normalized hip height: `1 - average(hip_y)`
   - Higher values = person is higher in frame (standing)
   - Lower values = person is lower in frame (falling/on ground)

3. **Vertical Velocity (v)** - Rate of change of hip height: `Δh / Δt`
   - Positive = moving up
   - Negative = moving down (falling)

4. **Motion Magnitude (m)** - Mean L2 displacement of all visible keypoints
   - Measures overall body movement between frames
   - Higher values = more rapid movement

5. **Shoulder Symmetry (s)** - Absolute difference in shoulder y-coordinates
   - Measures body balance/tilt
   - Higher values = more asymmetric (falling/tilted)

6. **Knee Angle (θ)** - Maximum knee angle (larger of left/right)
   - Computed using hip-knee-ankle vectors
   - Indicates leg bend/posture

## Windowing

Creates fixed-length temporal sequences for LSTM training:

- **Window Length:** 60 frames (~2 seconds @ 30 FPS)
- **Stride:** 10 frames (overlapping windows)
- **Quality Filter:** Drops windows with > 30% missing data

### Labeling Strategy

**URFD Dataset:**
- Uses video-level label (all windows from a fall video are labeled as fall)

**Le2i Dataset:**
- Uses per-frame labels
- Window labeled as fall if ≥ 6 frames (10%) are labeled as fall

## Usage

### Command Line

```bash
# Process all datasets
python -m ml.features.feature_engineering \
    --source data/interim/keypoints \
    --out data/processed \
    --stride 10 \
    --length 60 \
    --min-visible 0.7 \
    --datasets urfd,le2i,all \
    --le2i-min-fall-frames 6
```

### Python API

```python
from ml.features import extract_features_from_keypoints, create_windows
import numpy as np

# Load keypoints
data = np.load('data/interim/keypoints/video.npz')
keypoints = data['keypoints']  # (T, 17, 3)
label = data['label']
fps = data['fps']

# Extract features
features_raw, features_normalized = extract_features_from_keypoints(
    keypoints, 
    fps=fps, 
    apply_smoothing=True
)

# Create windows
windows, labels, metadata = create_windows(
    features_normalized,
    label=label,
    video_name='video',
    dataset='urfd',
    window_length=60,
    stride=10,
    max_missing_ratio=0.3
)
```

## Output Format

### Window Files (.npz)

Each output file contains:

```python
{
    'X': np.ndarray,           # (N, 60, 6) - feature windows
    'y': np.ndarray,           # (N,) - labels (0 or 1)
    'video_names': np.ndarray, # (N,) - source video names
    'start_indices': np.ndarray, # (N,) - window start frames
    'end_indices': np.ndarray,   # (N,) - window end frames
    'datasets': np.ndarray,      # (N,) - dataset names
    'missing_ratios': np.ndarray # (N,) - missing data ratios
}
```

### Output Files

- `data/processed/urfd_windows.npz` - URFD dataset windows
- `data/processed/le2i_windows.npz` - Le2i dataset windows
- `data/processed/all_windows.npz` - Combined dataset windows

## Quality Control

### Confidence Masking

- Keypoints with confidence < 0.3 are masked (set to NaN)
- Features computed only from high-confidence keypoints
- Prevents unreliable detections from affecting features

### Smoothing

- Savitzky-Golay filter applied to reduce noise
- Window length: 5 frames
- Polynomial order: 2
- Applied after confidence masking

### Normalization

- Per-video min-max normalization to [0, 1]
- Handles different video scales/resolutions
- Preserves temporal dynamics within each video

### Window Quality

- Windows with > 30% missing data are dropped
- Ensures sufficient valid data for LSTM training
- Tracks and reports dropped window statistics

## Visualization

Generate EDA plots:

```bash
python scripts/generate_eda_plots.py
```

Outputs:
- `docs/wiki_assets/phase1_features/class_balance.png`
- `docs/wiki_assets/phase1_features/feature_distributions.png`
- `docs/wiki_assets/phase1_features/temporal_traces.png`

Interactive notebook:
- `notebooks/eda.ipynb`

## Module Structure

```
ml/features/
├── __init__.py              # Module exports
├── feature_engineering.py   # Feature extraction + CLI
├── windowing.py            # Windowing utilities
└── README.md               # This file
```

## Dependencies

- numpy
- scipy (for Savitzky-Golay filter)
- matplotlib (for visualization)
- tqdm (for progress bars)

## Next Steps

1. **LSTM Training:** Use windowed features for fall detection model
2. **Feature Selection:** Analyze feature importance
3. **Data Augmentation:** Add noise, time warping
4. **Cross-validation:** Split data for training/validation

## References

- MoveNet keypoint format: COCO 17-keypoint skeleton
- Savitzky-Golay filter: `scipy.signal.savgol_filter`
- URFD Dataset: University of Rzeszow Fall Detection
- Le2i Dataset: Le2i Fall Detection Dataset

## License

Part of the mobile-vision-training project.

