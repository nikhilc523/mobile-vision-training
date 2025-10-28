# MoveNet Quick Reference

## Installation

```bash
pip install tensorflow tensorflow-hub opencv-python numpy matplotlib
```

## Basic Usage

```python
from ml.pose.movenet_loader import load_movenet, infer_keypoints
import cv2

# 1. Load model (once)
inference_fn = load_movenet()

# 2. Load image
frame = cv2.imread('image.png')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 3. Run inference
keypoints = infer_keypoints(inference_fn, frame_rgb)
# Returns: (17, 3) array with [y, x, confidence]
```

## CLI Usage

```bash
# Test on single image
python -m ml.pose.movenet_loader path/to/image.png
```

## Keypoint Indices

```
0: nose           9: left_wrist
1: left_eye      10: right_wrist
2: right_eye     11: left_hip
3: left_ear      12: right_hip
4: right_ear     13: left_knee
5: left_shoulder 14: right_knee
6: right_shoulder 15: left_ankle
7: left_elbow    16: right_ankle
8: right_elbow
```

## Common Patterns

### Process Video Sequence

```python
from pathlib import Path
import numpy as np

inference_fn = load_movenet()
sequence_dir = Path("data/raw/urfd/falls/fall-01-cam0-rgb")

keypoints_list = []
for frame_path in sorted(sequence_dir.glob("*.png")):
    frame = cv2.imread(str(frame_path))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = infer_keypoints(inference_fn, frame_rgb)
    keypoints_list.append(keypoints)

# Shape: (num_frames, 17, 3)
keypoints_array = np.array(keypoints_list)
```

### Extract LSTM Features

```python
# Flatten keypoints to feature vectors
features = []
for keypoints in keypoints_list:
    feature_vector = keypoints.flatten()  # 51 features
    features.append(feature_vector)

X = np.array(features)  # Shape: (num_frames, 51)
```

### Visualize Pose

```python
from ml.pose.movenet_loader import visualize_keypoints

visualize_keypoints(
    frame_rgb,
    keypoints,
    confidence_threshold=0.3,
    show=False,
    save_path='output.png'
)
```

### Filter High-Confidence Keypoints

```python
# Get only keypoints with confidence >= 0.5
high_conf_mask = keypoints[:, 2] >= 0.5
high_conf_keypoints = keypoints[high_conf_mask]

# Count high-confidence keypoints
count = np.sum(keypoints[:, 2] >= 0.3)
```

### Calculate Pose Metrics

```python
# Hip center
hip_y = (keypoints[11, 0] + keypoints[12, 0]) / 2
hip_x = (keypoints[11, 1] + keypoints[12, 1]) / 2

# Shoulder center
shoulder_y = (keypoints[5, 0] + keypoints[6, 0]) / 2
shoulder_x = (keypoints[5, 1] + keypoints[6, 1]) / 2

# Body height (normalized)
body_height = abs(hip_y - keypoints[0, 0])  # hip to nose

# Shoulder width (normalized)
shoulder_width = abs(keypoints[5, 1] - keypoints[6, 1])
```

## Testing

```bash
# Run unit tests
python3 ml/tests/test_movenet_loader.py

# Run examples
python3 examples/movenet_inference_example.py
```

## Performance

- **CPU:** ~30ms per frame (~33 FPS)
- **GPU:** ~5ms per frame (~200 FPS)
- **Model size:** 12MB (cached after first download)

## Troubleshooting

### SSL Certificate Error (macOS)
```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

### Low Detection Rate
```python
# Lower confidence threshold
keypoints = infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.2)
```

### Memory Issues
```python
# Process in batches
for i in range(0, len(frames), 100):
    batch = frames[i:i+100]
    # Process batch
```

## File Structure

```
ml/
├── pose/
│   ├── __init__.py
│   └── movenet_loader.py       # Main module
├── tests/
│   └── test_movenet_loader.py  # Unit tests
examples/
└── movenet_inference_example.py # Usage examples
docs/
├── movenet_pose_estimation.md   # Full documentation
└── movenet_quick_reference.md   # This file
```

## Next Steps

1. Extract pose features from all datasets
2. Train LSTM model for fall detection
3. Implement real-time inference pipeline
4. Add temporal smoothing for better tracking

