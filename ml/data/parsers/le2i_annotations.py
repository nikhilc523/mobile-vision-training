#!/usr/bin/env python3
"""
Le2i Fall Annotation Parser and Video Matcher

This module parses Le2i annotation files and links them to their corresponding video files.

Annotation Format:
- Line 1: Fall start frame (integer)
- Line 2: Fall end frame (integer)
- Remaining lines: Bounding box data (frame_id,person_id,x1,y1,x2,y2)

Example:
    144
    164
    1,1,205,70,259,170
    2,1,205,70,259,170
    ...
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def parse_annotation(txt_path: str) -> List[Tuple[int, int]]:
    """
    Parse a Le2i annotation file and extract fall frame ranges.
    
    The annotation format has:
    - Line 1: Fall start frame
    - Line 2: Fall end frame
    - Remaining lines: Bounding box data
    
    Args:
        txt_path: Path to the annotation .txt file
        
    Returns:
        List of (start_frame, end_frame) tuples representing fall events.
        Returns empty list if file doesn't exist or has invalid format.
        
    Examples:
        >>> parse_annotation("data/raw/le2i/Home_01/Annotation_files/video (1).txt")
        [(144, 164)]
    """
    txt_path = Path(txt_path)
    
    if not txt_path.exists():
        print(f"⚠️  Warning: Annotation file not found: {txt_path}")
        return []
    
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            print(f"⚠️  Warning: Invalid annotation format (less than 2 lines): {txt_path}")
            return []
        
        # Parse first two lines as start and end frames
        start_frame_str = lines[0].strip()
        end_frame_str = lines[1].strip()
        
        # Try to parse as integers
        try:
            start_frame = int(start_frame_str)
            end_frame = int(end_frame_str)
        except ValueError:
            print(f"⚠️  Warning: Could not parse frame numbers in: {txt_path}")
            return []
        
        # Validate frame range
        if start_frame <= 0 or end_frame <= 0:
            print(f"⚠️  Warning: Invalid frame range ({start_frame}, {end_frame}) in: {txt_path}")
            return []
        
        if start_frame > end_frame:
            print(f"⚠️  Warning: Start frame > end frame ({start_frame} > {end_frame}) in: {txt_path}")
            return []
        
        return [(start_frame, end_frame)]
        
    except Exception as e:
        print(f"❌ Error parsing annotation file {txt_path}: {e}")
        return []


def match_video_for_annotation(ann_path: str) -> Optional[Path]:
    """
    Find the corresponding .avi video file for an annotation file.
    
    Matching rule: Same base name, ignoring extension and spaces.
    Looks in the Videos/ folder at the same level as Annotation_files/.
    
    Args:
        ann_path: Path to the annotation .txt file
        
    Returns:
        Full path to the corresponding .avi file, or None if not found.
        
    Examples:
        >>> match_video_for_annotation("data/raw/le2i/Home_01/Annotation_files/video (1).txt")
        Path("data/raw/le2i/Home_01/Videos/video (1).avi")
    """
    ann_path = Path(ann_path)
    
    if not ann_path.exists():
        print(f"⚠️  Warning: Annotation file not found: {ann_path}")
        return None
    
    # Get the base name without extension
    base_name = ann_path.stem  # e.g., "video (1)"
    
    # Get the parent directory (should be Annotation_files or Annotations_files)
    ann_dir = ann_path.parent
    scene_dir = ann_dir.parent
    
    # Look for Videos folder
    videos_dir = scene_dir / "Videos"
    
    if not videos_dir.exists():
        # Maybe videos are directly in the scene directory
        # Check for .avi files with matching name
        video_path = scene_dir / f"{base_name}.avi"
        if video_path.exists():
            return video_path
        
        print(f"⚠️  Warning: Videos folder not found for: {ann_path}")
        return None
    
    # Look for matching .avi file
    video_path = videos_dir / f"{base_name}.avi"
    
    if not video_path.exists():
        print(f"⚠️  Warning: Video file not found: {video_path}")
        return None
    
    return video_path


def get_fall_ranges(scene_dir: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Scan an entire Le2i scene folder and build a dictionary of fall ranges.
    
    Args:
        scene_dir: Path to a Le2i scene directory (e.g., "data/raw/le2i/Home_01")
        
    Returns:
        Dictionary mapping video filenames to lists of fall frame ranges.
        Videos without annotations will have empty lists.
        
    Examples:
        >>> get_fall_ranges("data/raw/le2i/Home_01")
        {
            "video (1).avi": [(144, 164)],
            "video (2).avi": [(120, 137)],
            ...
        }
    """
    scene_dir = Path(scene_dir)
    
    if not scene_dir.exists():
        print(f"⚠️  Warning: Scene directory not found: {scene_dir}")
        return {}
    
    result = {}
    
    # Find all video files
    videos_dir = scene_dir / "Videos"
    video_files = []
    
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.avi"))
    else:
        # Videos might be directly in scene directory
        video_files = list(scene_dir.glob("*.avi"))
    
    # Find annotation directories (handle both spellings)
    ann_dirs = []
    for possible_name in ["Annotation_files", "Annotations_files"]:
        ann_dir = scene_dir / possible_name
        if ann_dir.exists():
            ann_dirs.append(ann_dir)
    
    # Build a mapping of video base names to annotation files
    ann_map = {}
    for ann_dir in ann_dirs:
        for ann_file in ann_dir.glob("*.txt"):
            base_name = ann_file.stem
            ann_map[base_name] = ann_file
    
    # Process each video file
    for video_file in video_files:
        video_name = video_file.name
        base_name = video_file.stem
        
        # Check if there's an annotation file
        if base_name in ann_map:
            ann_file = ann_map[base_name]
            fall_ranges = parse_annotation(str(ann_file))
            result[video_name] = fall_ranges
        else:
            # No annotation file - no falls
            result[video_name] = []
    
    return result


def main():
    """
    Command-line interface for testing the parser.
    
    Usage:
        python -m ml.data.parsers.le2i_annotations data/raw/le2i/Home_01
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m ml.data.parsers.le2i_annotations <scene_directory>")
        print("Example: python -m ml.data.parsers.le2i_annotations data/raw/le2i/Home_01")
        sys.exit(1)
    
    scene_dir = sys.argv[1]
    
    print(f"🔍 Scanning scene directory: {scene_dir}")
    print("="*70)
    
    fall_ranges = get_fall_ranges(scene_dir)
    
    if not fall_ranges:
        print("❌ No videos found or directory doesn't exist.")
        sys.exit(1)
    
    # Sort by video name for consistent output
    sorted_videos = sorted(fall_ranges.keys())
    
    print(f"\n📊 Found {len(sorted_videos)} videos:\n")
    
    for video_name in sorted_videos:
        ranges = fall_ranges[video_name]
        if ranges:
            ranges_str = ", ".join([f"[{start}-{end}]" for start, end in ranges])
            print(f"  🎥 {video_name:<25} Fall frames: {ranges_str}")
        else:
            print(f"  🎥 {video_name:<25} No falls detected")
    
    # Summary statistics
    videos_with_falls = sum(1 for ranges in fall_ranges.values() if ranges)
    total_falls = sum(len(ranges) for ranges in fall_ranges.values())
    
    print("\n" + "="*70)
    print(f"📈 Summary:")
    print(f"   Total videos: {len(sorted_videos)}")
    print(f"   Videos with falls: {videos_with_falls}")
    print(f"   Total fall events: {total_falls}")
    print("="*70)


if __name__ == "__main__":
    main()

