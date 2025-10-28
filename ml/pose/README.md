# Pose Estimation Module

## Overview

The `ml.pose` module provides pose estimation capabilities for fall detection using Google's MoveNet model from TensorFlow Hub.

## Features

- ✅ **Fast Inference:** MoveNet Lightning processes frames in ~30ms (CPU) or ~5ms (GPU)
- ✅ **17 Keypoints:** Full body pose in COCO format
- ✅ **Confidence Filtering:** Automatic masking of low-confidence keypoints
- ✅ **Visualization:** Skeleton overlay generation
- ✅ **Batch Processing:** Efficient processing of video sequences
- ✅ **LSTM Ready:** Feature extraction for temporal models

## Quick Start

```python
from ml.pose import load_movenet, infer_keypoints
import cv2

# Load model (once)
inference_fn = load_movenet()

# Process image
frame = cv2.imread('image.png')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Get keypoints
keypoints = infer_keypoints(inference_fn, frame_rgb)
# Returns: (17, 3) array with [y, x, confidence]
```

## Installation

```bash
pip install tensorflow tensorflow-hub opencv-python numpy matplotlib
```

## Module Contents

### Functions

- **`load_movenet(model_url=None)`** - Load MoveNet model from TensorFlow Hub
- **`preprocess_frame(frame_rgb)`** - Resize and prepare frame for inference
- **`infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.3)`** - Run pose estimation
- **`visualize_keypoints(frame_rgb, keypoints, ...)`** - Draw skeleton overlay

### Constants

- **`KEYPOINT_NAMES`** - List of 17 keypoint names
- **`KEYPOINT_EDGES`** - Skeleton connectivity for visualization

## Keypoint Format

MoveNet outputs 17 keypoints in COCO format:

```
Index  Name              Index  Name
-----  ----              -----  ----
0      nose              9      left_wrist
1      left_eye          10     right_wrist
2      right_eye         11     left_hip
3      left_ear          12     right_hip
4      right_ear         13     left_knee
5      left_shoulder     14     right_knee
6      right_shoulder    15     left_ankle
7      left_elbow        16     right_ankle
8      right_elbow
```

Each keypoint has 3 values: `[y, x, confidence]`, all normalized to [0, 1].

## Usage Examples

### Single Image

```python
from ml.pose import load_movenet, infer_keypoints, visualize_keypoints
import cv2

# Load model
inference_fn = load_movenet()

# Load and process image
frame = cv2.imread('person.png')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Run inference
keypoints = infer_keypoints(inference_fn, frame_rgb)

# Visualize
visualize_keypoints(frame_rgb, keypoints, save_path='output.png')
```

### Video Sequence

```python
from pathlib import Path
import numpy as np

# Load model
inference_fn = load_movenet()

# Process all frames
sequence_dir = Path("data/raw/urfd/falls/fall-01-cam0-rgb")
keypoints_list = []

for frame_path in sorted(sequence_dir.glob("*.png")):
    frame = cv2.imread(str(frame_path))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = infer_keypoints(inference_fn, frame_rgb)
    keypoints_list.append(keypoints)

# Convert to array: (num_frames, 17, 3)
keypoints_array = np.array(keypoints_list)
```

### Feature Extraction for LSTM

```python
# Extract flattened features
features = []
for keypoints in keypoints_list:
    # Flatten: 17 keypoints × 3 values = 51 features
    feature_vector = keypoints.flatten()
    features.append(feature_vector)

# Create feature matrix: (num_frames, 51)
X = np.array(features)
```

### Confidence Filtering

```python
# Get only high-confidence keypoints
high_conf_mask = keypoints[:, 2] >= 0.5
high_conf_keypoints = keypoints[high_conf_mask]

# Count detections
num_detected = np.sum(keypoints[:, 2] >= 0.3)
print(f"Detected {num_detected}/17 keypoints")
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
body_height = abs(hip_y - keypoints[0, 0])

# Shoulder width (normalized)
shoulder_width = abs(keypoints[5, 1] - keypoints[6, 1])

# Simple fall detection heuristic
if hip_y > shoulder_y + 0.2:
    print("Potential fall detected!")
```

## Command-Line Interface

Test the module on a single image:

```bash
python -m ml.pose.movenet_loader path/to/image.png
```

Output:
- First 5 keypoints with coordinates and confidence
- Summary statistics
- Visualization saved to `*_pose.png`

## Testing

Run the test suite:

```bash
python3 ml/tests/test_movenet_loader.py
```

Run examples:

```bash
python3 examples/movenet_inference_example.py
```

## Performance

### Inference Speed (640×480 images)

| Hardware | Time/Frame | FPS |
|----------|-----------|-----|
| CPU (M1) | 30ms | 33 |
| GPU (M1) | 5ms | 200 |

### Detection Quality (URFD Dataset)

| Sequence Type | Avg Confidence | Avg Keypoints |
|--------------|---------------|---------------|
| Fall | 0.461 | 15.3/17 |
| ADL | 0.218 | 2.2/17 |

## Troubleshooting

### SSL Certificate Error (macOS)

```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

### Low Detection Rate

Try lowering the confidence threshold:

```python
keypoints = infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.2)
```

### Memory Issues

Process frames in batches:

```python
batch_size = 100
for i in range(0, len(frames), batch_size):
    batch = frames[i:i+batch_size]
    # Process batch
```

## Documentation

- **Full Documentation:** `docs/movenet_pose_estimation.md`
- **Quick Reference:** `docs/movenet_quick_reference.md`
- **Implementation Summary:** `docs/movenet_implementation_summary.md`

## File Structure

```
ml/pose/
├── __init__.py              # Module exports
├── movenet_loader.py        # Main implementation
└── README.md                # This file

ml/tests/
└── test_movenet_loader.py   # Unit tests (14 tests)

examples/
└── movenet_inference_example.py  # 5 usage examples

docs/
├── movenet_pose_estimation.md
├── movenet_quick_reference.md
└── movenet_implementation_summary.md
```

## Next Steps

1. **Extract Features:** Process all URFD and Le2i datasets
2. **Train LSTM:** Use pose sequences for fall detection
3. **Real-Time:** Integrate with webcam for live detection
4. **Temporal Smoothing:** Add Kalman filtering for stability

## References

- [MoveNet on TensorFlow Hub](https://tfhub.dev/google/movenet/singlepose/lightning/4)
- [MoveNet Blog Post](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)

## License

Part of the mobile-vision-training project.

