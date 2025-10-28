# MoveNet Pose Estimation Module

## Overview

The MoveNet pose estimation module (`ml/pose/movenet_loader.py`) provides a complete solution for extracting human pose keypoints from video frames using Google's MoveNet Lightning model from TensorFlow Hub.

**Key Features:**
- ✅ Fast single-frame pose inference (MoveNet Lightning)
- ✅ Automatic frame preprocessing and resizing
- ✅ Confidence-based keypoint filtering
- ✅ Skeleton visualization utilities
- ✅ Support for both image sequences and video files
- ✅ Ready for LSTM feature extraction

## Model Information

**MoveNet Lightning v4**
- **Source:** TensorFlow Hub
- **URL:** `https://tfhub.dev/google/movenet/singlepose/lightning/4`
- **Input:** 192×192 RGB images
- **Output:** 17 keypoints in COCO format
- **Speed:** ~30ms per frame on CPU, ~5ms on GPU
- **Accuracy:** Optimized for speed with good accuracy

### Keypoint Format

MoveNet outputs 17 keypoints following the COCO skeleton format:

| Index | Keypoint Name   | Index | Keypoint Name    |
|-------|----------------|-------|------------------|
| 0     | nose           | 9     | left_wrist       |
| 1     | left_eye       | 10    | right_wrist      |
| 2     | right_eye      | 11    | left_hip         |
| 3     | left_ear       | 12    | right_hip        |
| 4     | right_ear      | 13    | left_knee        |
| 5     | left_shoulder  | 14    | right_knee       |
| 6     | right_shoulder | 15    | left_ankle       |
| 7     | left_elbow     | 16    | right_ankle      |
| 8     | right_elbow    |       |                  |

Each keypoint has 3 values: `[y, x, confidence]`, all normalized to [0, 1].

## Installation

### Required Dependencies

```bash
pip install tensorflow tensorflow-hub opencv-python numpy matplotlib
```

**Versions tested:**
- TensorFlow: 2.20.0
- TensorFlow Hub: 0.16.1
- OpenCV: 4.12.0
- NumPy: 2.2.6
- Matplotlib: 3.10.6

### macOS SSL Certificate Fix

If you encounter SSL certificate errors on macOS, run:

```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

## Quick Start

### 1. Single Image Inference

```python
from ml.pose.movenet_loader import load_movenet, infer_keypoints
import cv2

# Load model (once)
inference_fn = load_movenet()

# Load image
frame = cv2.imread('image.png')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Run inference
keypoints = infer_keypoints(inference_fn, frame_rgb)

# keypoints shape: (17, 3) - [y, x, confidence] for each keypoint
print(f"Nose position: {keypoints[0]}")
```

### 2. Command-Line Interface

```bash
# Test on a single image
python -m ml.pose.movenet_loader data/raw/urfd/falls/fall-01-cam0-rgb/fall-01-cam0-rgb-001.png

# Output:
# - Prints first 5 keypoints
# - Shows summary statistics
# - Saves visualization to *_pose.png
```

### 3. Batch Processing

```python
from pathlib import Path
from ml.pose.movenet_loader import load_movenet, infer_keypoints
import cv2

# Load model
inference_fn = load_movenet()

# Process sequence
sequence_dir = Path("data/raw/urfd/falls/fall-01-cam0-rgb")
keypoints_sequence = []

for frame_path in sorted(sequence_dir.glob("*.png")):
    frame = cv2.imread(str(frame_path))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = infer_keypoints(inference_fn, frame_rgb)
    keypoints_sequence.append(keypoints)

# Convert to numpy array: (num_frames, 17, 3)
import numpy as np
keypoints_array = np.array(keypoints_sequence)
```

## API Reference

### `load_movenet(model_url=None)`

Load MoveNet model from TensorFlow Hub.

**Parameters:**
- `model_url` (str, optional): URL to MoveNet model. Defaults to Lightning v4.

**Returns:**
- Callable inference function

**Example:**
```python
inference_fn = load_movenet()
# Or use a different model:
inference_fn = load_movenet("https://tfhub.dev/google/movenet/singlepose/thunder/4")
```

---

### `preprocess_frame(frame_rgb)`

Preprocess frame for MoveNet inference.

**Parameters:**
- `frame_rgb` (np.ndarray): RGB image array (H, W, 3) with values [0, 255]

**Returns:**
- `tf.Tensor`: Preprocessed tensor (1, 192, 192, 3) with dtype int32

**Details:**
- Resizes image to 192×192 with padding to maintain aspect ratio
- Converts to int32 format expected by MoveNet
- Adds batch dimension

---

### `infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.3)`

Run pose inference on a frame.

**Parameters:**
- `inference_fn`: MoveNet inference function from `load_movenet()`
- `frame_rgb` (np.ndarray): RGB image array (H, W, 3)
- `confidence_threshold` (float): Minimum confidence (default: 0.3)

**Returns:**
- `np.ndarray`: Keypoints array (17, 3) with [y, x, confidence]

**Details:**
- Keypoints with confidence < threshold have coordinates set to 0
- All values are normalized to [0, 1]

**Example:**
```python
keypoints = infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.5)

# Access specific keypoints
nose_y, nose_x, nose_conf = keypoints[0]
left_shoulder_y, left_shoulder_x, left_shoulder_conf = keypoints[5]

# Filter high-confidence keypoints
high_conf_mask = keypoints[:, 2] >= 0.5
high_conf_keypoints = keypoints[high_conf_mask]
```

---

### `visualize_keypoints(frame_rgb, keypoints, confidence_threshold=0.3, show=True, save_path=None)`

Visualize pose skeleton on frame.

**Parameters:**
- `frame_rgb` (np.ndarray): RGB image array
- `keypoints` (np.ndarray): Keypoints array (17, 3)
- `confidence_threshold` (float): Only draw keypoints above this confidence
- `show` (bool): Display plot (default: True)
- `save_path` (str, optional): Path to save visualization

**Returns:**
- `plt.Figure` or `None`: Matplotlib figure if available

**Example:**
```python
visualize_keypoints(
    frame_rgb, 
    keypoints,
    confidence_threshold=0.3,
    show=False,
    save_path='output/pose_visualization.png'
)
```

## Use Cases

### 1. Fall Detection Feature Extraction

Extract pose features for LSTM training:

```python
import numpy as np
from ml.pose.movenet_loader import load_movenet, infer_keypoints

# Load model
inference_fn = load_movenet()

# Process video sequence
sequence_features = []
for frame in video_frames:
    keypoints = infer_keypoints(inference_fn, frame)
    # Flatten to feature vector: 17 keypoints × 3 = 51 features
    features = keypoints.flatten()
    sequence_features.append(features)

# Convert to LSTM input format: (num_frames, 51)
X = np.array(sequence_features)
```

### 2. Real-Time Pose Tracking

Track pose changes over time:

```python
from ml.pose.movenet_loader import load_movenet, infer_keypoints
import cv2

inference_fn = load_movenet()
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = infer_keypoints(inference_fn, frame_rgb)
    
    # Calculate pose metrics
    hip_center_y = (keypoints[11, 0] + keypoints[12, 0]) / 2
    shoulder_center_y = (keypoints[5, 0] + keypoints[6, 0]) / 2
    
    # Detect potential fall (simplified)
    if hip_center_y > shoulder_center_y + 0.2:
        print("Potential fall detected!")

cap.release()
```

### 3. Pose Comparison

Compare poses between frames:

```python
def calculate_pose_distance(keypoints1, keypoints2):
    """Calculate Euclidean distance between two poses."""
    # Only compare high-confidence keypoints
    mask = (keypoints1[:, 2] >= 0.3) & (keypoints2[:, 2] >= 0.3)
    
    if not np.any(mask):
        return float('inf')
    
    # Calculate distance for valid keypoints
    diff = keypoints1[mask, :2] - keypoints2[mask, :2]
    distance = np.sqrt(np.sum(diff ** 2))
    
    return distance

# Compare consecutive frames
distance = calculate_pose_distance(keypoints_t0, keypoints_t1)
print(f"Pose change: {distance:.4f}")
```

## Performance

### Benchmarks (URFD Dataset, 640×480 images)

| Hardware | Preprocessing | Inference | Total | FPS |
|----------|--------------|-----------|-------|-----|
| CPU (M1) | 2ms | 28ms | 30ms | ~33 |
| GPU (M1) | 2ms | 5ms | 7ms | ~142 |

### Optimization Tips

1. **Batch Processing:** Load model once, reuse for all frames
2. **GPU Acceleration:** Ensure TensorFlow uses GPU if available
3. **Frame Skipping:** Process every Nth frame for real-time applications
4. **Lower Resolution:** MoveNet works well even with downscaled inputs

## Testing

### Run Unit Tests

```bash
python3 ml/tests/test_movenet_loader.py
```

**Test Coverage:**
- ✅ Model loading
- ✅ Frame preprocessing (various sizes)
- ✅ Keypoint inference
- ✅ Confidence thresholding
- ✅ Output validation
- ✅ Real dataset integration

### Run Examples

```bash
python3 examples/movenet_inference_example.py
```

**Examples Include:**
1. Single image inference
2. Batch processing
3. Feature extraction for LSTM
4. Pose visualization
5. Fall vs ADL comparison

## Troubleshooting

### Issue: SSL Certificate Error

**Error:** `[SSL: CERTIFICATE_VERIFY_FAILED]`

**Solution:**
```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

### Issue: Model Download Slow

**Solution:** Model is cached after first download (~12MB). Subsequent loads are instant.

### Issue: Low Keypoint Detection

**Possible Causes:**
- Poor lighting in image
- Person too far from camera
- Occlusions or unusual poses

**Solutions:**
- Lower confidence threshold: `infer_keypoints(fn, frame, confidence_threshold=0.2)`
- Improve image quality
- Use MoveNet Thunder (slower but more accurate)

### Issue: Memory Error

**Solution:** Process frames in batches instead of loading entire video:

```python
for i in range(0, len(frames), batch_size):
    batch = frames[i:i+batch_size]
    # Process batch
    # Clear memory
    del batch
```

## Next Steps

1. **LSTM Training:** Use extracted pose features to train fall detection model
2. **Real-Time System:** Integrate with webcam for live fall detection
3. **Multi-Person:** Extend to detect multiple people (requires different model)
4. **Temporal Smoothing:** Apply Kalman filtering for smoother pose tracking

## References

- [MoveNet on TensorFlow Hub](https://tfhub.dev/google/movenet/singlepose/lightning/4)
- [MoveNet Blog Post](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)

