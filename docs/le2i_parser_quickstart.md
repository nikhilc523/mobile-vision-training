# Le2i Annotation Parser - Quick Start Guide

## Installation

No installation required! The parser uses only Python standard library.

**Requirements:**
- Python 3.11+
- Standard library: `os`, `re`, `pathlib`

## Quick Start

### 1. Import the module

```python
from ml.data.parsers import parse_annotation, match_video_for_annotation, get_fall_ranges
```

### 2. Parse a single annotation file

```python
fall_ranges = parse_annotation("data/raw/le2i/Home_01/Annotation_files/video (1).txt")
print(fall_ranges)  # [(144, 164)]
```

### 3. Find the corresponding video

```python
video_path = match_video_for_annotation("data/raw/le2i/Home_01/Annotation_files/video (1).txt")
print(video_path)  # data/raw/le2i/Home_01/Videos/video (1).avi
```

### 4. Process an entire scene

```python
fall_data = get_fall_ranges("data/raw/le2i/Home_01")

for video, ranges in fall_data.items():
    if ranges:
        print(f"{video}: {ranges}")
```

## Command-Line Usage

Inspect a scene directory:

```bash
python3 -m ml.data.parsers.le2i_annotations data/raw/le2i/Home_01
```

## Testing

Run the test suite:

```bash
# Without pytest
python3 ml/tests/run_tests.py

# With pytest (if installed)
pytest ml/tests/test_le2i_annotations.py -v
```

## Examples

Run the example script:

```bash
python3 examples/parse_le2i_example.py
```

## Common Use Cases

### Get all videos with falls

```python
fall_data = get_fall_ranges("data/raw/le2i/Home_01")
videos_with_falls = {v: r for v, r in fall_data.items() if r}
print(f"Found {len(videos_with_falls)} videos with falls")
```

### Calculate fall duration statistics

```python
fall_data = get_fall_ranges("data/raw/le2i/Home_01")

durations = []
for video, ranges in fall_data.items():
    for start, end in ranges:
        duration = end - start + 1
        durations.append(duration)

avg_duration = sum(durations) / len(durations)
print(f"Average fall duration: {avg_duration:.1f} frames")
```

### Process all scenes

```python
from pathlib import Path

le2i_dir = Path("data/raw/le2i")

for scene_dir in le2i_dir.iterdir():
    if scene_dir.is_dir():
        fall_data = get_fall_ranges(str(scene_dir))
        videos_with_falls = sum(1 for r in fall_data.values() if r)
        print(f"{scene_dir.name}: {videos_with_falls} videos with falls")
```

## Documentation

- **Full API Reference**: `ml/data/parsers/README.md`
- **Implementation Summary**: `docs/le2i_parser_summary.md`
- **Examples**: `examples/parse_le2i_example.py`

## Troubleshooting

### ModuleNotFoundError: No module named 'ml'

Make sure you're running from the project root directory:

```bash
cd /path/to/mobile-vision-training
python3 -m ml.data.parsers.le2i_annotations data/raw/le2i/Home_01
```

Or add the project root to your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Warning: Annotation file not found

Check that the Le2i dataset is properly extracted in `data/raw/le2i/`.

### Warning: Invalid frame range (0, 0)

This is normal - some annotation files have (0, 0) to indicate no falls in that video.

## Dataset Statistics

| Scene          | Videos | With Falls | Total Falls |
|----------------|--------|------------|-------------|
| Coffee_room_01 | 48     | 47         | 47          |
| Coffee_room_02 | 22     | 12         | 12          |
| Home_01        | 30     | 30         | 30          |
| Home_02        | 30     | 7          | 7           |
| Lecture room   | 27     | 0          | 0           |
| Office         | 33     | 0          | 0           |
| **TOTAL**      | **190**| **96**     | **96**      |

## Next Steps

1. ✅ Parse annotations - Done!
2. 🔄 Extract video frames
3. 🔄 Run pose estimation
4. 🔄 Train LSTM model
5. 🔄 Evaluate fall detection

## Support

For issues or questions:
- Check the full documentation in `ml/data/parsers/README.md`
- Review examples in `examples/parse_le2i_example.py`
- Run tests to verify installation: `python3 ml/tests/run_tests.py`

