# Le2i Annotation Parser

Python module for parsing Le2i fall detection dataset annotations and matching them with video files.

## Overview

The Le2i dataset contains video files with corresponding annotation files that mark fall events. This parser extracts fall frame ranges and links them to their corresponding video files.

## Annotation Format

Le2i annotation files follow this format:

```
144                          # Line 1: Fall start frame
164                          # Line 2: Fall end frame
1,1,205,70,259,170          # Line 3+: Bounding box data (frame_id,person_id,x1,y1,x2,y2)
2,1,205,70,259,170
...
```

## Directory Structure

The Le2i dataset has two common structures:

### Structure 1: With Videos/ folder
```
Home_01/
  Annotation_files/
    video (1).txt
    video (2).txt
  Videos/
    video (1).avi
    video (2).avi
```

### Structure 2: Direct videos (no annotations)
```
Lecture_room/
  video (1).avi
  video (2).avi
  ...
```

## API Reference

### `parse_annotation(txt_path: str) -> List[Tuple[int, int]]`

Parse a Le2i annotation file and extract fall frame ranges.

**Parameters:**
- `txt_path`: Path to the annotation .txt file

**Returns:**
- List of (start_frame, end_frame) tuples representing fall events
- Returns empty list if file doesn't exist or has invalid format

**Example:**
```python
from ml.data.parsers import parse_annotation

fall_ranges = parse_annotation("data/raw/le2i/Home_01/Annotation_files/video (1).txt")
# Returns: [(144, 164)]
```

### `match_video_for_annotation(ann_path: str) -> Optional[Path]`

Find the corresponding .avi video file for an annotation file.

**Parameters:**
- `ann_path`: Path to the annotation .txt file

**Returns:**
- Full path to the corresponding .avi file, or None if not found

**Matching Rule:**
- Same base name, ignoring extension
- Looks in Videos/ folder at the same level as Annotation_files/
- Falls back to scene directory if Videos/ doesn't exist

**Example:**
```python
from ml.data.parsers import match_video_for_annotation

video_path = match_video_for_annotation("data/raw/le2i/Home_01/Annotation_files/video (1).txt")
# Returns: Path("data/raw/le2i/Home_01/Videos/video (1).avi")
```

### `get_fall_ranges(scene_dir: str) -> Dict[str, List[Tuple[int, int]]]`

Scan an entire Le2i scene folder and build a dictionary of fall ranges.

**Parameters:**
- `scene_dir`: Path to a Le2i scene directory (e.g., "data/raw/le2i/Home_01")

**Returns:**
- Dictionary mapping video filenames to lists of fall frame ranges
- Videos without annotations will have empty lists

**Example:**
```python
from ml.data.parsers import get_fall_ranges

fall_data = get_fall_ranges("data/raw/le2i/Home_01")
# Returns:
# {
#     "video (1).avi": [(144, 164)],
#     "video (2).avi": [(120, 137)],
#     ...
# }
```

## Command-Line Usage

You can run the parser directly from the command line to inspect a scene directory:

```bash
python3 -m ml.data.parsers.le2i_annotations data/raw/le2i/Home_01
```

**Output:**
```
🔍 Scanning scene directory: data/raw/le2i/Home_01
======================================================================

📊 Found 30 videos:

  🎥 video (1).avi             Fall frames: [144-164]
  🎥 video (2).avi             Fall frames: [120-137]
  ...

======================================================================
📈 Summary:
   Total videos: 30
   Videos with falls: 30
   Total fall events: 30
======================================================================
```

## Testing

### With pytest (recommended)

If you have pytest installed:

```bash
# Install pytest
pip3 install pytest

# Run tests
pytest ml/tests/test_le2i_annotations.py -v
```

### Without pytest

Use the built-in test runner:

```bash
python3 ml/tests/run_tests.py
```

## Error Handling

The parser handles various error conditions gracefully:

- **Missing files**: Returns empty list/None and prints warning
- **Invalid format**: Returns empty list and prints warning
- **Invalid frame numbers**: Returns empty list and prints warning
- **Missing Videos/ folder**: Checks scene directory for direct .avi files

All errors are logged to stdout with emoji indicators:
- ⚠️  Warning: Non-critical issues
- ❌ Error: Critical parsing errors

## Features

✅ **Robust parsing**: Handles whitespace, invalid formats, and missing files  
✅ **Flexible structure**: Works with both Videos/ folder and direct .avi files  
✅ **Alternative spellings**: Supports both "Annotation_files" and "Annotations_files"  
✅ **Comprehensive tests**: 10+ unit tests covering edge cases  
✅ **Real data validation**: Tests against actual Le2i dataset  
✅ **CLI interface**: Inspect datasets from command line  

## Dependencies

- Python 3.11+
- Standard library only: `os`, `re`, `pathlib`

No external dependencies required!

## Examples

### Example 1: Process all scenes

```python
from pathlib import Path
from ml.data.parsers import get_fall_ranges

le2i_dir = Path("data/raw/le2i")
all_scenes = {}

for scene_dir in le2i_dir.iterdir():
    if scene_dir.is_dir():
        scene_name = scene_dir.name
        fall_data = get_fall_ranges(str(scene_dir))
        all_scenes[scene_name] = fall_data

# Now you have all fall ranges for all scenes
print(f"Processed {len(all_scenes)} scenes")
```

### Example 2: Filter videos with falls

```python
from ml.data.parsers import get_fall_ranges

fall_data = get_fall_ranges("data/raw/le2i/Home_01")

# Get only videos with falls
videos_with_falls = {
    video: ranges 
    for video, ranges in fall_data.items() 
    if ranges
}

print(f"Found {len(videos_with_falls)} videos with falls")
```

### Example 3: Extract fall duration statistics

```python
from ml.data.parsers import get_fall_ranges

fall_data = get_fall_ranges("data/raw/le2i/Home_01")

durations = []
for video, ranges in fall_data.items():
    for start, end in ranges:
        duration = end - start + 1
        durations.append(duration)

if durations:
    avg_duration = sum(durations) / len(durations)
    print(f"Average fall duration: {avg_duration:.1f} frames")
    print(f"Min: {min(durations)}, Max: {max(durations)}")
```

## Notes

- The parser currently extracts only the first fall range from each annotation file
- Some annotation files may have multiple falls (future enhancement)
- Frame numbers are 1-indexed (as in the annotation files)
- The parser validates that start_frame < end_frame and both are positive

