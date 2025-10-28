"""
UCF101 Subset Keypoint Extraction Script

Extracts MoveNet pose keypoints from UCF101 subset videos (7 non-fall activity classes).
Processes videos from data/raw/ucf101_subset/ and saves keypoint .npz files.

Classes:
- ApplyEyeMakeup
- BodyWeightSquats
- JumpingJack
- Lunges
- MoppingFloor
- PullUps
- PushUps

Usage:
    python -m ml.data.ucf101_extract \\
        --source data/raw/ucf101_subset \\
        --out data/interim/keypoints \\
        --model thunder \\
        --skip-existing
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np

# OpenCV for video processing
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is required.")
    print("Install with: pip install opencv-python")
    sys.exit(1)

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm is required.")
    print("Install with: pip install tqdm")
    sys.exit(1)

# Import MoveNet utilities
try:
    from ml.pose.movenet_loader import load_movenet, infer_keypoints
except ImportError:
    print("ERROR: Could not import ml.pose.movenet_loader")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


# Model URLs
MOVENET_MODELS = {
    'lightning': 'https://tfhub.dev/google/movenet/singlepose/lightning/4',
    'thunder': 'https://tfhub.dev/google/movenet/singlepose/thunder/4'
}


def extract_keypoints_from_video(
    video_path: Path,
    inference_fn,
    confidence_threshold: float = 0.3
) -> Tuple[Optional[np.ndarray], int, float]:
    """
    Extract pose keypoints from all frames of a video.
    
    Args:
        video_path: Path to video file
        inference_fn: MoveNet inference function
        confidence_threshold: Minimum confidence for keypoints
    
    Returns:
        Tuple of (keypoints_array, num_frames, fps)
        - keypoints_array: (T, 17, 3) array or None if failed
        - num_frames: Number of frames processed
        - fps: Video frame rate
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None, 0, 0.0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # Default fallback
    
    keypoints_list = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Infer keypoints
            keypoints = infer_keypoints(
                inference_fn,
                frame_rgb,
                confidence_threshold=confidence_threshold
            )
            
            keypoints_list.append(keypoints)
    
    except Exception as e:
        print(f"    Error processing frames: {e}")
        cap.release()
        return None, 0, fps
    
    finally:
        cap.release()
    
    if not keypoints_list:
        return None, 0, fps
    
    # Stack into (T, 17, 3) array
    keypoints_array = np.stack(keypoints_list, axis=0)
    
    return keypoints_array, len(keypoints_list), fps


def process_ucf101_subset(
    source_dir: Path,
    output_dir: Path,
    model_name: str = 'thunder',
    skip_existing: bool = False,
    confidence_threshold: float = 0.3
) -> dict:
    """
    Process all UCF101 subset videos and extract keypoints.
    
    Args:
        source_dir: Root directory containing class subdirectories
        output_dir: Directory to save .npz keypoint files
        model_name: MoveNet model ('lightning' or 'thunder')
        skip_existing: Skip videos that already have keypoint files
        confidence_threshold: Minimum confidence for keypoints
    
    Returns:
        Dictionary with extraction statistics
    """
    # Validate source directory
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MoveNet model
    model_url = MOVENET_MODELS.get(model_name)
    if model_url is None:
        print(f"ERROR: Unknown model '{model_name}'. Choose 'lightning' or 'thunder'.")
        sys.exit(1)
    
    print(f"Loading MoveNet {model_name.capitalize()}...")
    inference_fn = load_movenet(model_url)
    print()
    
    # Find all class directories
    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    
    if not class_dirs:
        print(f"ERROR: No class directories found in {source_dir}")
        sys.exit(1)
    
    print(f"Found {len(class_dirs)} classes:")
    for class_dir in class_dirs:
        print(f"  - {class_dir.name}")
    print()
    
    # Collect all video files
    video_files = []
    for class_dir in class_dirs:
        videos = (
            list(class_dir.glob('*.avi')) +
            list(class_dir.glob('*.mp4')) +
            list(class_dir.glob('*.AVI')) +
            list(class_dir.glob('*.MP4'))
        )
        video_files.extend([(v, class_dir.name) for v in videos])
    
    print(f"Total videos to process: {len(video_files)}")
    print()
    
    # Statistics
    stats = {
        'total_videos': len(video_files),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'total_frames': 0,
        'fps_values': [],
        'start_time': datetime.utcnow()
    }
    
    # Process each video
    for video_path, class_name in tqdm(video_files, desc="Extracting keypoints"):
        # Generate output filename
        video_id = video_path.stem
        output_filename = f"ucf101_{class_name}_{video_id}.npz"
        output_path = output_dir / output_filename
        
        # Skip if exists
        if skip_existing and output_path.exists():
            stats['skipped'] += 1
            continue
        
        # Extract keypoints
        keypoints, num_frames, fps = extract_keypoints_from_video(
            video_path,
            inference_fn,
            confidence_threshold=confidence_threshold
        )
        
        if keypoints is None or num_frames == 0:
            stats['failed'] += 1
            tqdm.write(f"  ✗ Failed: {video_path.name}")
            continue
        
        # Save to .npz
        try:
            np.savez_compressed(
                output_path,
                keypoints=keypoints,
                label=0,  # Non-fall
                fps=fps,
                dataset='ucf101',
                class_name=class_name,
                video_name=video_id
            )
            
            stats['processed'] += 1
            stats['total_frames'] += num_frames
            stats['fps_values'].append(fps)
        
        except Exception as e:
            stats['failed'] += 1
            tqdm.write(f"  ✗ Save failed for {video_path.name}: {e}")
    
    stats['end_time'] = datetime.utcnow()
    
    return stats


def print_summary(stats: dict):
    """Print extraction summary statistics."""
    print()
    print("="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print()
    print(f"Total videos: {stats['total_videos']}")
    print(f"  ✓ Processed: {stats['processed']}")
    print(f"  ⊘ Skipped: {stats['skipped']}")
    print(f"  ✗ Failed: {stats['failed']}")
    print()
    print(f"Total frames extracted: {stats['total_frames']:,}")
    
    if stats['fps_values']:
        avg_fps = np.mean(stats['fps_values'])
        print(f"Average FPS: {avg_fps:.1f}")
    
    duration = (stats['end_time'] - stats['start_time']).total_seconds()
    print(f"Processing time: {duration:.1f} seconds")
    
    success_rate = (stats['processed'] / stats['total_videos']) * 100
    print(f"Success rate: {success_rate:.1f}%")
    print()
    print("="*70)


def main():
    """Main entry point for UCF101 extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract MoveNet keypoints from UCF101 subset videos"
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='data/raw/ucf101_subset',
        help='Source directory containing class subdirectories (default: data/raw/ucf101_subset)'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default='data/interim/keypoints',
        help='Output directory for .npz files (default: data/interim/keypoints)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['lightning', 'thunder'],
        default='thunder',
        help='MoveNet model variant (default: thunder)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip videos that already have keypoint files'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='Confidence threshold for keypoints (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    source_dir = Path(args.source)
    output_dir = Path(args.out)
    
    print("="*70)
    print("UCF101 SUBSET KEYPOINT EXTRACTION")
    print("="*70)
    print()
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: MoveNet {args.model.capitalize()}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Skip existing: {args.skip_existing}")
    print()
    
    # Run extraction
    stats = process_ucf101_subset(
        source_dir=source_dir,
        output_dir=output_dir,
        model_name=args.model,
        skip_existing=args.skip_existing,
        confidence_threshold=args.confidence
    )
    
    # Print summary
    print_summary(stats)
    
    # Check success rate
    success_rate = (stats['processed'] / stats['total_videos']) * 100
    if success_rate < 95:
        print(f"⚠️  Warning: Success rate ({success_rate:.1f}%) is below 95%")
        sys.exit(1)
    else:
        print("✓ Extraction completed successfully!")


if __name__ == '__main__':
    main()

