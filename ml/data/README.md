# Data Processing Module

## Overview

The `ml.data` module provides tools for processing and extracting features from fall detection datasets.

## Modules

### `extract_pose_sequences.py`

Batch extraction of MoveNet pose keypoints from URFD and Le2i video datasets.

**Features:**
- Processes both URFD (image sequences) and Le2i (videos)
- Saves compressed .npz files with keypoints and metadata
- Progress tracking with tqdm
- Automatic result reporting
- Error handling and recovery

**Usage:**
```bash
# Process all datasets
python -m ml.data.extract_pose_sequences --all

# Process specific dataset
python -m ml.data.extract_pose_sequences --dataset urfd
python -m ml.data.extract_pose_sequences --dataset le2i

# Test with limited videos
python -m ml.data.extract_pose_sequences --dataset urfd --limit 5

# Skip already processed videos
python -m ml.data.extract_pose_sequences --all --skip-existing
```

**Output Format:**
```python
# Each .npz file contains:
{
    'keypoints': np.ndarray,  # (T, 17, 3) - [y, x, confidence]
    'label': int,             # 0 or 1
    'fps': int,               # Frame rate
    'dataset': str,           # 'urfd' or 'le2i'
    'video_name': str,        # Unique identifier
    'frame_labels': np.ndarray  # (T,) - Le2i only
}
```

**Python API:**
```python
from ml.data import PoseExtractor
from pathlib import Path

# Initialize extractor
extractor = PoseExtractor(
    output_dir=Path("data/interim/keypoints"),
    skip_existing=True
)

# Extract from image sequence (URFD)
extractor.extract_from_image_sequence(
    frame_dir=Path("data/raw/urfd/falls/fall-01-cam0-rgb"),
    label=1,
    video_name="urfd_fall_fall-01-cam0-rgb"
)

# Extract from video (Le2i)
extractor.extract_from_video(
    video_path=Path("data/raw/le2i/Home_01/Videos/video (1).avi"),
    fall_ranges=[(144, 164)],
    video_name="le2i_Home_01_video (1)"
)

# Check statistics
print(f"Processed: {extractor.stats['videos_processed']}")
print(f"Total frames: {extractor.stats['total_frames']}")
```

### `parsers/le2i_annotations.py`

Parser for Le2i fall detection annotation files.

**Functions:**
- `parse_annotation(txt_path)` - Extract fall frame ranges
- `match_video_for_annotation(ann_path)` - Find corresponding video
- `get_fall_ranges(scene_dir)` - Process entire scene directory

**Usage:**
```python
from ml.data.parsers import get_fall_ranges

# Get fall ranges for a scene
fall_data = get_fall_ranges("data/raw/le2i/Home_01")

# Returns: {"video (1).avi": [(144, 164)], ...}
```

## Directory Structure

```
ml/data/
├── __init__.py                    # Module exports
├── README.md                      # This file
├── extract_pose_sequences.py     # Pose extraction script
└── parsers/
    ├── __init__.py
    ├── le2i_annotations.py        # Le2i parser
    └── README.md
```

## Output Structure

```
data/interim/keypoints/
├── urfd_fall_fall-01-cam0-rgb.npz
├── urfd_fall_fall-02-cam0-rgb.npz
├── ...
├── urfd_adl_adl-01-cam0-rgb.npz
├── urfd_adl_adl-02-cam0-rgb.npz
├── ...
├── le2i_Home_01_video (1).npz
├── le2i_Home_01_video (2).npz
├── ...
└── le2i_Office_video (1).npz
```

## Validation

Validate extracted keypoints:

```bash
python scripts/validate_extracted_keypoints.py --verbose
```

**Checks:**
- File format and structure
- Array shapes and data types
- Value ranges (coordinates and confidence in [0,1])
- Label validity (0 or 1)
- Metadata completeness
- Le2i frame labels consistency

## Performance

### Benchmarks

| Dataset | Videos | Frames | Time | FPS | Size |
|---------|--------|--------|------|-----|------|
| URFD | 63 | ~34,000 | ~2min | ~280 | ~1.5MB |
| Le2i | 190 | ~57,000 | ~6min | ~160 | ~2.5MB |
| **Total** | **253** | **~91,000** | **~8min** | **~190** | **~4MB** |

*Benchmarks on M1 MacBook Pro with CPU inference*

### Compression

- **Uncompressed:** ~91,000 frames × 17 keypoints × 3 values × 4 bytes = ~18.5 MB
- **Compressed (.npz):** ~4 MB
- **Compression ratio:** ~4.6x

## Documentation

- **Full Guide:** [docs/pose_extraction_guide.md](../../docs/pose_extraction_guide.md)
- **Le2i Parser:** [parsers/README.md](parsers/README.md)
- **MoveNet Docs:** [docs/movenet_pose_estimation.md](../../docs/movenet_pose_estimation.md)

## Examples

### Load and Inspect Keypoints

```python
import numpy as np

# Load data
data = np.load('data/interim/keypoints/urfd_fall_fall-01-cam0-rgb.npz')

# Access arrays
keypoints = data['keypoints']  # (160, 17, 3)
label = data['label']          # 1
fps = data['fps']              # 30

print(f"Video: {data['video_name']}")
print(f"Frames: {keypoints.shape[0]}")
print(f"Label: {'Fall' if label == 1 else 'Non-fall'}")

# Get nose position in first frame
nose_y, nose_x, nose_conf = keypoints[0, 0]
print(f"Nose: ({nose_x:.3f}, {nose_y:.3f}) confidence: {nose_conf:.3f}")
```

### Batch Load All Videos

```python
import numpy as np
from pathlib import Path

keypoints_dir = Path("data/interim/keypoints")

# Load all URFD fall videos
fall_videos = []
for npz_file in keypoints_dir.glob("urfd_fall_*.npz"):
    data = np.load(npz_file)
    fall_videos.append({
        'name': data['video_name'],
        'keypoints': data['keypoints'],
        'label': data['label']
    })

print(f"Loaded {len(fall_videos)} fall videos")
```

### Extract Features for LSTM

```python
import numpy as np

# Load video
data = np.load('data/interim/keypoints/urfd_fall_fall-01-cam0-rgb.npz')
keypoints = data['keypoints']  # (T, 17, 3)

# Flatten keypoints to feature vectors
# Each frame: 17 keypoints × 3 values = 51 features
features = keypoints.reshape(keypoints.shape[0], -1)  # (T, 51)

print(f"Feature matrix shape: {features.shape}")
# Output: Feature matrix shape: (160, 51)
```

### Filter by Confidence

```python
import numpy as np

data = np.load('data/interim/keypoints/urfd_fall_fall-01-cam0-rgb.npz')
keypoints = data['keypoints']  # (T, 17, 3)

# Count high-confidence keypoints per frame
confidence_threshold = 0.5
high_conf_counts = np.sum(keypoints[:, :, 2] >= confidence_threshold, axis=1)

print(f"Average high-confidence keypoints: {high_conf_counts.mean():.1f}/17")
```

## Next Steps

1. **Feature Engineering:** Create derived features (angles, velocities)
2. **Temporal Windowing:** Create fixed-length sequences for LSTM
3. **Data Augmentation:** Add noise, rotations, scaling
4. **LSTM Training:** Use keypoints as input to fall detection model

## Troubleshooting

### Issue: Slow Extraction

**Solution:** Enable GPU acceleration in TensorFlow

### Issue: Out of Memory

**Solution:** Process in batches with `--limit` flag

### Issue: Corrupted Files

**Solution:** Delete and re-extract with `--skip-existing`

## References

- [Pose Extraction Guide](../../docs/pose_extraction_guide.md)
- [MoveNet Documentation](../../docs/movenet_pose_estimation.md)
- [Le2i Parser Documentation](parsers/README.md)
- [Project Status](../../PROJECT_STATUS.md)

