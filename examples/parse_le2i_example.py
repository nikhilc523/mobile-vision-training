#!/usr/bin/env python3
"""
Example script demonstrating Le2i annotation parser usage.

This script shows various ways to use the Le2i annotation parser
to extract fall detection information from the dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path to import ml module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.parsers import parse_annotation, match_video_for_annotation, get_fall_ranges


def example_1_parse_single_annotation():
    """Example 1: Parse a single annotation file."""
    print("="*70)
    print("Example 1: Parse a single annotation file")
    print("="*70)
    
    ann_path = "data/raw/le2i/Home_01/Annotation_files/video (1).txt"
    
    if not Path(ann_path).exists():
        print(f"⚠️  Annotation file not found: {ann_path}")
        return
    
    fall_ranges = parse_annotation(ann_path)
    
    print(f"Annotation file: {ann_path}")
    print(f"Fall ranges: {fall_ranges}")
    
    if fall_ranges:
        start, end = fall_ranges[0]
        duration = end - start + 1
        print(f"Fall duration: {duration} frames")
    
    print()


def example_2_match_video():
    """Example 2: Find the corresponding video for an annotation."""
    print("="*70)
    print("Example 2: Match annotation to video file")
    print("="*70)
    
    ann_path = "data/raw/le2i/Home_01/Annotation_files/video (1).txt"
    
    if not Path(ann_path).exists():
        print(f"⚠️  Annotation file not found: {ann_path}")
        return
    
    video_path = match_video_for_annotation(ann_path)
    
    print(f"Annotation: {ann_path}")
    print(f"Video: {video_path}")
    
    if video_path and video_path.exists():
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"Video size: {size_mb:.2f} MB")
    
    print()


def example_3_process_scene():
    """Example 3: Process an entire scene directory."""
    print("="*70)
    print("Example 3: Process entire scene directory")
    print("="*70)
    
    scene_dir = "data/raw/le2i/Home_01"
    
    if not Path(scene_dir).exists():
        print(f"⚠️  Scene directory not found: {scene_dir}")
        return
    
    fall_data = get_fall_ranges(scene_dir)
    
    print(f"Scene: {scene_dir}")
    print(f"Total videos: {len(fall_data)}")
    
    videos_with_falls = {v: r for v, r in fall_data.items() if r}
    print(f"Videos with falls: {len(videos_with_falls)}")
    
    # Show first 5 videos with falls
    print("\nFirst 5 videos with falls:")
    for i, (video, ranges) in enumerate(list(videos_with_falls.items())[:5]):
        start, end = ranges[0]
        print(f"  {i+1}. {video}: frames {start}-{end}")
    
    print()


def example_4_fall_statistics():
    """Example 4: Calculate fall duration statistics."""
    print("="*70)
    print("Example 4: Fall duration statistics")
    print("="*70)
    
    scene_dir = "data/raw/le2i/Home_01"
    
    if not Path(scene_dir).exists():
        print(f"⚠️  Scene directory not found: {scene_dir}")
        return
    
    fall_data = get_fall_ranges(scene_dir)
    
    durations = []
    for video, ranges in fall_data.items():
        for start, end in ranges:
            duration = end - start + 1
            durations.append(duration)
    
    if durations:
        print(f"Total falls: {len(durations)}")
        print(f"Average duration: {sum(durations) / len(durations):.1f} frames")
        print(f"Min duration: {min(durations)} frames")
        print(f"Max duration: {max(durations)} frames")
        print(f"Total frames: {sum(durations)} frames")
    else:
        print("No falls found in this scene")
    
    print()


def example_5_process_all_scenes():
    """Example 5: Process all scenes in the Le2i dataset."""
    print("="*70)
    print("Example 5: Process all scenes")
    print("="*70)
    
    le2i_dir = Path("data/raw/le2i")
    
    if not le2i_dir.exists():
        print(f"⚠️  Le2i directory not found: {le2i_dir}")
        return
    
    all_scenes = {}
    
    for scene_dir in sorted(le2i_dir.iterdir()):
        if scene_dir.is_dir():
            scene_name = scene_dir.name
            fall_data = get_fall_ranges(str(scene_dir))
            all_scenes[scene_name] = fall_data
    
    print(f"Processed {len(all_scenes)} scenes\n")
    
    # Summary table
    print(f"{'Scene':<20} {'Videos':<10} {'With Falls':<12} {'Total Falls':<12}")
    print("-"*70)
    
    for scene_name, fall_data in sorted(all_scenes.items()):
        total_videos = len(fall_data)
        videos_with_falls = sum(1 for ranges in fall_data.values() if ranges)
        total_falls = sum(len(ranges) for ranges in fall_data.values())
        
        print(f"{scene_name:<20} {total_videos:<10} {videos_with_falls:<12} {total_falls:<12}")
    
    # Grand totals
    total_videos = sum(len(fd) for fd in all_scenes.values())
    total_with_falls = sum(
        sum(1 for ranges in fd.values() if ranges) 
        for fd in all_scenes.values()
    )
    total_falls = sum(
        sum(len(ranges) for ranges in fd.values()) 
        for fd in all_scenes.values()
    )
    
    print("-"*70)
    print(f"{'TOTAL':<20} {total_videos:<10} {total_with_falls:<12} {total_falls:<12}")
    
    print()


def example_6_filter_by_duration():
    """Example 6: Filter falls by duration."""
    print("="*70)
    print("Example 6: Filter falls by duration")
    print("="*70)
    
    scene_dir = "data/raw/le2i/Coffee_room_01"
    
    if not Path(scene_dir).exists():
        print(f"⚠️  Scene directory not found: {scene_dir}")
        return
    
    fall_data = get_fall_ranges(scene_dir)
    
    # Find falls longer than 30 frames
    min_duration = 30
    long_falls = []
    
    for video, ranges in fall_data.items():
        for start, end in ranges:
            duration = end - start + 1
            if duration >= min_duration:
                long_falls.append((video, start, end, duration))
    
    print(f"Falls with duration >= {min_duration} frames:\n")
    
    if long_falls:
        # Sort by duration (descending)
        long_falls.sort(key=lambda x: x[3], reverse=True)
        
        for video, start, end, duration in long_falls[:10]:  # Show top 10
            print(f"  {video:<25} frames {start:>4}-{end:>4} ({duration:>3} frames)")
    else:
        print("  No falls found with that duration")
    
    print()


def main():
    """Run all examples."""
    print("\n🎓 Le2i Annotation Parser - Usage Examples\n")
    
    # Check if data exists
    if not Path("data/raw/le2i").exists():
        print("⚠️  Le2i dataset not found at data/raw/le2i/")
        print("Please ensure the dataset is downloaded and extracted.")
        return
    
    # Run examples
    example_1_parse_single_annotation()
    example_2_match_video()
    example_3_process_scene()
    example_4_fall_statistics()
    example_5_process_all_scenes()
    example_6_filter_by_duration()
    
    print("="*70)
    print("✅ All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()

