#!/usr/bin/env python3
"""
Batch Pose Extraction from URFD and Le2i Datasets

This module extracts MoveNet pose keypoints from video sequences and saves them
as compressed .npz files for LSTM training.

URFD Dataset:
- Image sequences (.png frames in folders)
- Labels: fall=1, adl=0

Le2i Dataset:
- Video files (.avi)
- Annotations mark fall frame ranges
- Labels: 1 for fall frames, 0 for non-fall frames

Output Format:
- Compressed .npz files in data/interim/keypoints/
- Each file contains:
  - keypoints: (T, 17, 3) array with [y, x, confidence]
  - label: int (0 or 1)
  - fps: int (30)
  - dataset: str ('urfd' or 'le2i')
  - video_name: str

Usage:
    python -m ml.data.extract_pose_sequences --dataset urfd
    python -m ml.data.extract_pose_sequences --dataset le2i
    python -m ml.data.extract_pose_sequences --all
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import warnings

import numpy as np
import cv2

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm is required for progress bars.")
    print("Install with: pip install tqdm")
    sys.exit(1)

# Import our modules
try:
    from ml.pose.movenet_loader import load_movenet, infer_keypoints
    from ml.data.parsers.le2i_annotations import get_fall_ranges
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


class PoseExtractor:
    """Extracts pose keypoints from video sequences."""

    def __init__(self, output_dir: Path, skip_existing: bool = False, model_variant: str = 'thunder'):
        """
        Initialize the pose extractor.

        Args:
            output_dir: Directory to save .npz files
            skip_existing: If True, skip videos that already have .npz files
            model_variant: MoveNet model variant ('lightning' or 'thunder')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        self.model_variant = model_variant

        # Load MoveNet model
        if model_variant == 'thunder':
            model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
            print("Loading MoveNet Thunder (higher accuracy)...")
        else:
            model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
            print("Loading MoveNet Lightning (faster)...")

        self.inference_fn = load_movenet(model_url)
        print()
        
        # Statistics
        self.stats = {
            'videos_processed': 0,
            'videos_skipped': 0,
            'videos_failed': 0,
            'total_frames': 0,
            'total_time': 0.0,
        }
    
    def extract_from_image_sequence(
        self,
        frame_dir: Path,
        label: int,
        video_name: str
    ) -> Optional[Path]:
        """
        Extract keypoints from a folder of PNG images (URFD format).
        
        Args:
            frame_dir: Directory containing .png frames
            label: Video label (0 or 1)
            video_name: Name for the output file
            
        Returns:
            Path to saved .npz file, or None if failed
        """
        output_path = self.output_dir / f"{video_name}.npz"
        
        # Check if already exists
        if self.skip_existing and output_path.exists():
            self.stats['videos_skipped'] += 1
            return output_path
        
        try:
            # Get all PNG files in sorted order
            frame_files = sorted(frame_dir.glob("*.png"))
            
            # Filter out any pose visualization files
            frame_files = [f for f in frame_files if '_pose' not in f.stem]
            
            if not frame_files:
                print(f"⚠️  Warning: No frames found in {frame_dir}")
                self.stats['videos_failed'] += 1
                return None
            
            # Extract keypoints from each frame
            keypoints_list = []
            
            for frame_path in frame_files:
                # Read frame
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    print(f"⚠️  Warning: Failed to read {frame_path}")
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference
                keypoints = infer_keypoints(self.inference_fn, frame_rgb, confidence_threshold=0.3)
                keypoints_list.append(keypoints)
            
            if not keypoints_list:
                print(f"⚠️  Warning: No valid frames in {frame_dir}")
                self.stats['videos_failed'] += 1
                return None
            
            # Stack into array: (T, 17, 3)
            keypoints_array = np.array(keypoints_list, dtype=np.float32)
            
            # Save as compressed .npz
            np.savez_compressed(
                output_path,
                keypoints=keypoints_array,
                label=label,
                fps=30,
                dataset='urfd',
                video_name=video_name
            )
            
            self.stats['videos_processed'] += 1
            self.stats['total_frames'] += len(keypoints_list)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error processing {frame_dir}: {e}")
            self.stats['videos_failed'] += 1
            return None
    
    def extract_from_video(
        self,
        video_path: Path,
        fall_ranges: List[Tuple[int, int]],
        video_name: str
    ) -> Optional[Path]:
        """
        Extract keypoints from a video file (Le2i format).
        
        Args:
            video_path: Path to .avi video file
            fall_ranges: List of (start_frame, end_frame) tuples for falls
            video_name: Name for the output file
            
        Returns:
            Path to saved .npz file, or None if failed
        """
        output_path = self.output_dir / f"{video_name}.npz"
        
        # Check if already exists
        if self.skip_existing and output_path.exists():
            self.stats['videos_skipped'] += 1
            return output_path
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"⚠️  Warning: Failed to open video {video_path}")
                self.stats['videos_failed'] += 1
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 30  # Default
            
            # Create frame labels (0 = no fall, 1 = fall)
            frame_labels = np.zeros(total_frames, dtype=np.int32)
            for start, end in fall_ranges:
                # Annotation frames are 1-indexed, convert to 0-indexed
                start_idx = max(0, start - 1)
                end_idx = min(total_frames, end)
                frame_labels[start_idx:end_idx] = 1
            
            # Determine overall label (1 if any falls, 0 otherwise)
            label = 1 if len(fall_ranges) > 0 else 0
            
            # Extract keypoints from each frame
            keypoints_list = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference
                keypoints = infer_keypoints(self.inference_fn, frame_rgb, confidence_threshold=0.3)
                keypoints_list.append(keypoints)
                
                frame_idx += 1
            
            cap.release()
            
            if not keypoints_list:
                print(f"⚠️  Warning: No valid frames in {video_path}")
                self.stats['videos_failed'] += 1
                return None
            
            # Stack into array: (T, 17, 3)
            keypoints_array = np.array(keypoints_list, dtype=np.float32)
            
            # Save as compressed .npz
            np.savez_compressed(
                output_path,
                keypoints=keypoints_array,
                label=label,
                fps=fps,
                dataset='le2i',
                video_name=video_name,
                frame_labels=frame_labels[:len(keypoints_list)]  # Per-frame labels
            )
            
            self.stats['videos_processed'] += 1
            self.stats['total_frames'] += len(keypoints_list)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            self.stats['videos_failed'] += 1
            return None


def process_urfd_dataset(
    extractor: PoseExtractor,
    urfd_root: Path,
    limit: Optional[int] = None
) -> Dict[str, int]:
    """
    Process all URFD videos (image sequences).
    
    Args:
        extractor: PoseExtractor instance
        urfd_root: Root directory of URFD dataset
        limit: Maximum number of videos to process (for testing)
        
    Returns:
        Dictionary with processing statistics
    """
    print("="*70)
    print("Processing URFD Dataset")
    print("="*70)
    print()
    
    # Find all fall and ADL sequences
    falls_dir = urfd_root / "falls"
    adl_dir = urfd_root / "adl"
    
    fall_sequences = sorted([d for d in falls_dir.iterdir() if d.is_dir()])
    adl_sequences = sorted([d for d in adl_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(fall_sequences)} fall sequences")
    print(f"Found {len(adl_sequences)} ADL sequences")
    print()
    
    # Combine and limit if requested
    all_sequences = [(d, 1, 'fall') for d in fall_sequences] + [(d, 0, 'adl') for d in adl_sequences]
    
    if limit:
        all_sequences = all_sequences[:limit]
        print(f"⚠️  Limiting to first {limit} videos")
        print()
    
    # Process each sequence
    start_time = time.time()
    
    for seq_dir, label, seq_type in tqdm(all_sequences, desc="URFD", unit="video"):
        video_name = f"urfd_{seq_type}_{seq_dir.name}"
        extractor.extract_from_image_sequence(seq_dir, label, video_name)
    
    elapsed = time.time() - start_time
    extractor.stats['total_time'] += elapsed
    
    return {
        'total': len(all_sequences),
        'falls': len(fall_sequences) if not limit else min(limit, len(fall_sequences)),
        'adl': len(adl_sequences) if not limit else max(0, limit - len(fall_sequences)),
    }


def process_le2i_dataset(
    extractor: PoseExtractor,
    le2i_root: Path,
    limit: Optional[int] = None
) -> Dict[str, int]:
    """
    Process all Le2i videos.
    
    Args:
        extractor: PoseExtractor instance
        le2i_root: Root directory of Le2i dataset
        limit: Maximum number of videos to process (for testing)
        
    Returns:
        Dictionary with processing statistics
    """
    print("="*70)
    print("Processing Le2i Dataset")
    print("="*70)
    print()
    
    # Find all scene directories
    scene_dirs = [d for d in le2i_root.iterdir() if d.is_dir()]
    
    print(f"Found {len(scene_dirs)} scene directories")
    print()
    
    # Collect all videos with their fall ranges
    all_videos = []
    
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        
        # Get fall ranges for this scene
        fall_data = get_fall_ranges(str(scene_dir))
        
        # Find video files
        videos_dir = scene_dir / "Videos"
        if videos_dir.exists():
            video_files = list(videos_dir.glob("*.avi"))
        else:
            # Videos might be directly in scene directory
            video_files = list(scene_dir.glob("*.avi"))
        
        for video_file in video_files:
            video_filename = video_file.name
            fall_ranges = fall_data.get(video_filename, [])
            
            video_name = f"le2i_{scene_name}_{video_file.stem}"
            all_videos.append((video_file, fall_ranges, video_name))
    
    print(f"Found {len(all_videos)} total videos")
    
    if limit:
        all_videos = all_videos[:limit]
        print(f"⚠️  Limiting to first {limit} videos")
    
    print()
    
    # Process each video
    start_time = time.time()
    
    for video_path, fall_ranges, video_name in tqdm(all_videos, desc="Le2i", unit="video"):
        extractor.extract_from_video(video_path, fall_ranges, video_name)
    
    elapsed = time.time() - start_time
    extractor.stats['total_time'] += elapsed
    
    return {
        'total': len(all_videos),
        'scenes': len(scene_dirs),
    }


def main():
    """Main entry point for the pose extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract MoveNet pose keypoints from URFD and Le2i datasets"
    )
    
    parser.add_argument(
        '--dataset',
        choices=['urfd', 'le2i', 'all'],
        default='all',
        help="Which dataset to process (default: all)"
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help="Skip videos that already have .npz files"
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help="Process only first N videos (for testing)"
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/interim/keypoints',
        help="Output directory for .npz files (default: data/interim/keypoints)"
    )

    parser.add_argument(
        '--model',
        choices=['lightning', 'thunder'],
        default='thunder',
        help="MoveNet model variant (default: thunder for higher accuracy)"
    )

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    urfd_root = project_root / "data" / "raw" / "urfd"
    le2i_root = project_root / "data" / "raw" / "le2i"
    output_dir = project_root / args.output_dir

    # Print configuration
    print()
    print("="*70)
    print("MoveNet Pose Extraction")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: MoveNet {args.model.capitalize()}")
    print(f"Output directory: {output_dir}")
    print(f"Skip existing: {args.skip_existing}")
    if args.limit:
        print(f"Limit: {args.limit} videos")
    print("="*70)
    print()

    # Initialize extractor
    extractor = PoseExtractor(output_dir, skip_existing=args.skip_existing, model_variant=args.model)
    
    # Process datasets
    urfd_stats = None
    le2i_stats = None
    
    start_time = time.time()
    
    if args.dataset in ['urfd', 'all']:
        if urfd_root.exists():
            urfd_stats = process_urfd_dataset(extractor, urfd_root, args.limit)
        else:
            print(f"⚠️  Warning: URFD dataset not found at {urfd_root}")
    
    if args.dataset in ['le2i', 'all']:
        if le2i_root.exists():
            le2i_stats = process_le2i_dataset(extractor, le2i_root, args.limit)
        else:
            print(f"⚠️  Warning: Le2i dataset not found at {le2i_root}")
    
    total_time = time.time() - start_time
    
    # Print summary
    print()
    print("="*70)
    print("Processing Complete")
    print("="*70)
    print()
    
    # Print statistics table
    print("Dataset Summary:")
    print("-" * 70)
    
    if urfd_stats:
        print(f"URFD    | Videos: {urfd_stats['total']:3d} | "
              f"Falls: {urfd_stats['falls']:2d} | ADL: {urfd_stats['adl']:2d}")
    
    if le2i_stats:
        print(f"Le2i    | Videos: {le2i_stats['total']:3d} | "
              f"Scenes: {le2i_stats['scenes']:2d}")
    
    print("-" * 70)
    print(f"Total   | Processed: {extractor.stats['videos_processed']:3d} | "
          f"Skipped: {extractor.stats['videos_skipped']:3d} | "
          f"Failed: {extractor.stats['videos_failed']:3d}")
    print(f"Frames  | Total: {extractor.stats['total_frames']:,}")
    print(f"Time    | Total: {format_time(total_time)} | "
          f"Avg: {total_time/max(1, extractor.stats['videos_processed']):.1f}s/video")
    print(f"Output  | {output_dir}")
    print("="*70)
    print()
    
    # Append to results file
    append_to_results(args, extractor.stats, urfd_stats, le2i_stats, total_time, output_dir)
    
    print("✅ Done!")


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m{secs:02d}s"
    else:
        return f"{secs}s"


def append_to_results(
    args,
    stats: Dict,
    urfd_stats: Optional[Dict],
    le2i_stats: Optional[Dict],
    total_time: float,
    output_dir: Path
):
    """Append results to docs/results1.md."""
    project_root = Path(__file__).parent.parent.parent
    results_file = project_root / "docs" / "results1.md"
    
    # Create results entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    entry = f"""
---

## 🗓️ Date: {timestamp}

**Phase:** 1.4 Pose Extraction

### Dataset Summary:

"""
    
    if urfd_stats:
        entry += f"- **URFD** – {urfd_stats['total']} videos processed ({urfd_stats['falls']} fall, {urfd_stats['adl']} ADL)\n"
    
    if le2i_stats:
        entry += f"- **Le2i** – {le2i_stats['total']} videos processed ({le2i_stats['scenes']} scenes)\n"
    
    entry += f"""
### Statistics:

- **Total Processed:** {stats['videos_processed']} videos
- **Total Frames:** {stats['total_frames']:,} frames
- **Skipped:** {stats['videos_skipped']} videos
- **Failed:** {stats['videos_failed']} videos
- **Avg FPS:** {stats['total_frames']/max(1, total_time):.1f} frames/sec
- **Total Runtime:** {format_time(total_time)}
- **Avg Time:** {total_time/max(1, stats['videos_processed']):.1f}s per video

### Output:

- **Directory:** `{output_dir.relative_to(project_root)}`
- **Format:** Compressed .npz files
- **Contents:** keypoints (T, 17, 3), label, fps, dataset, video_name

✅ **Status:** Success

"""
    
    # Append to file
    try:
        with open(results_file, 'a') as f:
            f.write(entry)
        print(f"✓ Results appended to {results_file}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to append to results file: {e}")


if __name__ == '__main__':
    main()

