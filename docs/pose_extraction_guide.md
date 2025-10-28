# Pose Extraction Guide

## Overview

The `ml.data.extract_pose_sequences` module extracts MoveNet pose keypoints from URFD and Le2i video datasets and saves them as compressed `.npz` files for LSTM training.

## Features

- ✅ **Batch Processing:** Process entire datasets with progress bars
- ✅ **Two Dataset Support:** URFD (image sequences) and Le2i (videos)
- ✅ **Compressed Output:** Efficient `.npz` format with compression
- ✅ **Frame-Level Labels:** Le2i includes per-frame fall/non-fall labels
- ✅ **Skip Existing:** Option to skip already processed videos
- ✅ **Progress Tracking:** Real-time progress bars with tqdm
- ✅ **Automatic Reporting:** Results appended to `docs/results1.md`
- ✅ **Error Handling:** Graceful handling of corrupted/missing files

## Installation

Required dependencies:
```bash
pip install tensorflow tensorflow-hub opencv-python numpy tqdm
```

## Quick Start

### Process URFD Dataset

```bash
python -m ml.data.extract_pose_sequences --dataset urfd
```

### Process Le2i Dataset

```bash
python -m ml.data.extract_pose_sequences --dataset le2i
```

### Process Both Datasets

```bash
python -m ml.data.extract_pose_sequences --all
```

## Command-Line Options

```bash
python -m ml.data.extract_pose_sequences [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset {urfd,le2i,all}` | Which dataset to process | `all` |
| `--skip-existing` | Skip videos with existing .npz files | `False` |
| `--limit N` | Process only first N videos (testing) | `None` |
| `--output-dir DIR` | Output directory for .npz files | `data/interim/keypoints` |

### Examples

```bash
# Test with 5 videos
python -m ml.data.extract_pose_sequences --dataset urfd --limit 5

# Skip already processed videos
python -m ml.data.extract_pose_sequences --all --skip-existing

# Custom output directory
python -m ml.data.extract_pose_sequences --dataset le2i --output-dir data/processed/keypoints
```

## Output Format

### File Naming

- **URFD:** `urfd_{fall|adl}_{sequence_name}.npz`
  - Example: `urfd_fall_fall-01-cam0-rgb.npz`
  
- **Le2i:** `le2i_{scene_name}_{video_name}.npz`
  - Example: `le2i_Home_01_video (1).npz`

### NPZ Contents

Each `.npz` file contains:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `keypoints` | float32 | (T, 17, 3) | Pose keypoints [y, x, confidence] |
| `label` | int | scalar | Video label (0=no fall, 1=fall) |
| `fps` | int | scalar | Video frame rate |
| `dataset` | str | - | Dataset name ('urfd' or 'le2i') |
| `video_name` | str | - | Unique video identifier |
| `frame_labels` | int | (T,) | Per-frame labels (Le2i only) |

### Loading NPZ Files

```python
import numpy as np

# Load data
data = np.load('data/interim/keypoints/urfd_fall_fall-01-cam0-rgb.npz')

# Access arrays
keypoints = data['keypoints']  # (T, 17, 3)
label = data['label']          # 0 or 1
fps = data['fps']              # 30
dataset = data['dataset']      # 'urfd'
video_name = data['video_name']

print(f"Video: {video_name}")
print(f"Frames: {keypoints.shape[0]}")
print(f"Label: {label}")
```

## Dataset-Specific Details

### URFD Dataset

**Structure:**
- Image sequences in folders
- Each folder contains PNG frames
- Two categories: `falls/` and `adl/`

**Labels:**
- `fall` sequences → label = 1
- `adl` sequences → label = 0

**Processing:**
1. Scans `data/raw/urfd/falls/` and `data/raw/urfd/adl/`
2. For each folder, reads PNG frames in sorted order
3. Runs MoveNet inference on each frame
4. Stacks keypoints into (T, 17, 3) array
5. Saves with label and metadata

**Example Output:**
```
urfd_fall_fall-01-cam0-rgb.npz
  - keypoints: (160, 17, 3)
  - label: 1
  - fps: 30
  - dataset: 'urfd'
```

### Le2i Dataset

**Structure:**
- Video files (.avi) in scene directories
- Annotation files mark fall frame ranges
- 6 scenes: Home_01, Home_02, Coffee_room_01, Coffee_room_02, Lecture_room, Office

**Labels:**
- Videos with fall annotations → label = 1
- Videos without falls → label = 0
- Per-frame labels: 1 for fall frames, 0 for non-fall frames

**Processing:**
1. Scans all scene directories
2. Parses annotation files using `le2i_annotations.get_fall_ranges()`
3. Opens each video with OpenCV
4. Runs MoveNet inference on each frame
5. Creates per-frame labels based on fall ranges
6. Saves with label, frame labels, and metadata

**Example Output:**
```
le2i_Home_01_video (1).npz
  - keypoints: (216, 17, 3)
  - label: 1
  - fps: 24
  - dataset: 'le2i'
  - frame_labels: (216,) - [0,0,0,...,1,1,1,...,0,0,0]
```

## Performance

### Benchmarks

| Dataset | Videos | Frames | Time | FPS | Avg Time/Video |
|---------|--------|--------|------|-----|----------------|
| URFD | 63 | ~34,000 | ~2min | ~280 | ~2s |
| Le2i | 190 | ~57,000 | ~6min | ~160 | ~2s |
| **Total** | **253** | **~91,000** | **~8min** | **~190** | **~2s** |

*Benchmarks on M1 MacBook Pro with CPU inference*

### Optimization Tips

1. **GPU Acceleration:** Ensure TensorFlow uses GPU for faster inference
2. **Skip Existing:** Use `--skip-existing` to resume interrupted runs
3. **Batch Processing:** Process datasets separately if memory is limited
4. **Limit Testing:** Use `--limit` to test on small subset first

## Progress Tracking

### Console Output

```
======================================================================
MoveNet Pose Extraction
======================================================================
Dataset: all
Output directory: data/interim/keypoints
Skip existing: False
======================================================================

Loading MoveNet model...
✓ MoveNet model loaded successfully!

======================================================================
Processing URFD Dataset
======================================================================

Found 31 fall sequences
Found 33 ADL sequences

URFD: 100%|████████████████████████| 63/63 [02:05<00:00,  2.0s/video]

======================================================================
Processing Le2i Dataset
======================================================================

Found 6 scene directories
Found 190 total videos

Le2i: 100%|████████████████████████| 190/190 [06:20<00:00,  2.0s/video]

======================================================================
Processing Complete
======================================================================

Dataset Summary:
----------------------------------------------------------------------
URFD    | Videos:  63 | Falls: 31 | ADL: 32
Le2i    | Videos: 190 | Scenes:  6
----------------------------------------------------------------------
Total   | Processed: 253 | Skipped:   0 | Failed:   0
Frames  | Total: 91,234
Time    | Total: 8m25s | Avg: 2.0s/video
Output  | data/interim/keypoints
======================================================================

✓ Results appended to docs/results1.md
✅ Done!
```

### Results File

Results are automatically appended to `docs/results1.md`:

```markdown
## 🗓️ Date: 2025-10-28 14:30:00

**Phase:** 1.4 Pose Extraction

### Dataset Summary:

- **URFD** – 63 videos processed (31 fall, 32 ADL)
- **Le2i** – 190 videos processed (6 scenes)

### Statistics:

- **Total Processed:** 253 videos
- **Total Frames:** 91,234 frames
- **Skipped:** 0 videos
- **Failed:** 0 videos
- **Avg FPS:** 180.5 frames/sec
- **Total Runtime:** 8m25s
- **Avg Time:** 2.0s per video

### Output:

- **Directory:** `data/interim/keypoints`
- **Format:** Compressed .npz files
- **Contents:** keypoints (T, 17, 3), label, fps, dataset, video_name

✅ **Status:** Success
```

## Error Handling

### Common Issues

#### 1. Missing Frames
```
⚠️  Warning: No frames found in data/raw/urfd/falls/fall-XX-cam0-rgb
```
**Solution:** Check if folder contains PNG files

#### 2. Failed Video Read
```
⚠️  Warning: Failed to open video data/raw/le2i/Home_01/Videos/video (1).avi
```
**Solution:** Verify video file is not corrupted

#### 3. Invalid Annotations
```
⚠️  Warning: Invalid frame range (0, 0) in annotation file
```
**Solution:** Annotation file has invalid format (expected, will skip)

### Statistics

The script tracks:
- `videos_processed`: Successfully processed videos
- `videos_skipped`: Skipped (already exist)
- `videos_failed`: Failed to process
- `total_frames`: Total frames extracted
- `total_time`: Total processing time

## Python API

### Using PoseExtractor Directly

```python
from pathlib import Path
from ml.data.extract_pose_sequences import PoseExtractor

# Initialize extractor
output_dir = Path("data/interim/keypoints")
extractor = PoseExtractor(output_dir, skip_existing=True)

# Extract from image sequence (URFD)
frame_dir = Path("data/raw/urfd/falls/fall-01-cam0-rgb")
output_path = extractor.extract_from_image_sequence(
    frame_dir=frame_dir,
    label=1,
    video_name="urfd_fall_fall-01-cam0-rgb"
)

# Extract from video (Le2i)
video_path = Path("data/raw/le2i/Home_01/Videos/video (1).avi")
fall_ranges = [(144, 164)]  # From annotation parser
output_path = extractor.extract_from_video(
    video_path=video_path,
    fall_ranges=fall_ranges,
    video_name="le2i_Home_01_video (1)"
)

# Check statistics
print(f"Processed: {extractor.stats['videos_processed']}")
print(f"Failed: {extractor.stats['videos_failed']}")
print(f"Total frames: {extractor.stats['total_frames']}")
```

## Next Steps

After extraction:

1. **Verify Output:** Check `data/interim/keypoints/` for .npz files
2. **Inspect Data:** Load a few files to verify keypoints
3. **Feature Engineering:** Create derived features (angles, velocities)
4. **LSTM Training:** Use keypoints as input to LSTM model
5. **Evaluation:** Test fall detection accuracy

## Troubleshooting

### Issue: Slow Processing

**Possible Causes:**
- CPU inference (no GPU)
- Large videos
- Disk I/O bottleneck

**Solutions:**
- Enable GPU acceleration in TensorFlow
- Process datasets separately
- Use SSD for faster I/O

### Issue: Out of Memory

**Possible Causes:**
- Processing very long videos
- Insufficient RAM

**Solutions:**
- Process in batches with `--limit`
- Close other applications
- Increase swap space

### Issue: Corrupted Output

**Possible Causes:**
- Interrupted processing
- Disk full

**Solutions:**
- Delete partial files
- Use `--skip-existing` to resume
- Check disk space

## References

- [MoveNet Documentation](../docs/movenet_pose_estimation.md)
- [Le2i Parser Documentation](../docs/le2i_parser_summary.md)
- [Project Status](../PROJECT_STATUS.md)

