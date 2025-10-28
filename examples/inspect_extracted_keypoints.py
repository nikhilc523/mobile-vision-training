#!/usr/bin/env python3
"""
Inspect Extracted Keypoints

This script demonstrates how to load and inspect extracted pose keypoints.

Usage:
    python examples/inspect_extracted_keypoints.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def example_1_load_single_file():
    """Example 1: Load and inspect a single .npz file."""
    print("="*70)
    print("Example 1: Load Single File")
    print("="*70)
    print()
    
    keypoints_dir = project_root / "data" / "interim" / "keypoints"
    
    # Find first URFD fall video
    urfd_files = list(keypoints_dir.glob("urfd_fall_*.npz"))
    
    if not urfd_files:
        print("⚠️  No URFD fall videos found. Run extraction first.")
        return
    
    # Load first file
    npz_path = urfd_files[0]
    data = np.load(npz_path)
    
    print(f"File: {npz_path.name}")
    print()
    
    # Print metadata
    print("Metadata:")
    print(f"  Video name: {data['video_name']}")
    print(f"  Dataset: {data['dataset']}")
    print(f"  Label: {data['label']} ({'Fall' if data['label'] == 1 else 'Non-fall'})")
    print(f"  FPS: {data['fps']}")
    print()
    
    # Print keypoints info
    keypoints = data['keypoints']
    print("Keypoints:")
    print(f"  Shape: {keypoints.shape} (frames, keypoints, [y,x,conf])")
    print(f"  Data type: {keypoints.dtype}")
    print(f"  Value range: [{keypoints.min():.3f}, {keypoints.max():.3f}]")
    print()
    
    # Print first frame keypoints
    print("First frame keypoints (first 5):")
    print(f"  {'Index':<6} {'Name':<15} {'Y':<8} {'X':<8} {'Conf':<8}")
    print(f"  {'-'*50}")
    
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    for i in range(min(5, len(keypoint_names))):
        y, x, conf = keypoints[0, i]
        print(f"  {i:<6} {keypoint_names[i]:<15} {y:.4f}  {x:.4f}  {conf:.4f}")
    
    print()


def example_2_batch_load():
    """Example 2: Load all videos and compute statistics."""
    print("="*70)
    print("Example 2: Batch Load and Statistics")
    print("="*70)
    print()
    
    keypoints_dir = project_root / "data" / "interim" / "keypoints"
    
    if not keypoints_dir.exists():
        print("⚠️  Keypoints directory not found. Run extraction first.")
        return
    
    # Load all files
    all_files = list(keypoints_dir.glob("*.npz"))
    
    if not all_files:
        print("⚠️  No .npz files found. Run extraction first.")
        return
    
    print(f"Found {len(all_files)} files")
    print()
    
    # Collect statistics
    urfd_count = 0
    le2i_count = 0
    fall_count = 0
    non_fall_count = 0
    total_frames = 0
    avg_confidences = []
    
    for npz_path in all_files:
        data = np.load(npz_path)
        
        # Count by dataset
        if str(data['dataset']) == 'urfd':
            urfd_count += 1
        else:
            le2i_count += 1
        
        # Count by label
        if data['label'] == 1:
            fall_count += 1
        else:
            non_fall_count += 1
        
        # Count frames
        keypoints = data['keypoints']
        total_frames += keypoints.shape[0]
        
        # Average confidence
        avg_conf = keypoints[:, :, 2].mean()
        avg_confidences.append(avg_conf)
    
    # Print statistics
    print("Dataset Statistics:")
    print(f"  URFD videos: {urfd_count}")
    print(f"  Le2i videos: {le2i_count}")
    print(f"  Total videos: {len(all_files)}")
    print()
    
    print("Label Distribution:")
    print(f"  Fall videos: {fall_count}")
    print(f"  Non-fall videos: {non_fall_count}")
    print()
    
    print("Frame Statistics:")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Avg frames per video: {total_frames / len(all_files):.1f}")
    print()
    
    print("Confidence Statistics:")
    print(f"  Avg confidence: {np.mean(avg_confidences):.3f}")
    print(f"  Min confidence: {np.min(avg_confidences):.3f}")
    print(f"  Max confidence: {np.max(avg_confidences):.3f}")
    print()


def example_3_feature_extraction():
    """Example 3: Extract features for LSTM training."""
    print("="*70)
    print("Example 3: Feature Extraction for LSTM")
    print("="*70)
    print()
    
    keypoints_dir = project_root / "data" / "interim" / "keypoints"
    
    # Find first URFD fall video
    urfd_files = list(keypoints_dir.glob("urfd_fall_*.npz"))
    
    if not urfd_files:
        print("⚠️  No URFD fall videos found. Run extraction first.")
        return
    
    # Load first file
    data = np.load(urfd_files[0])
    keypoints = data['keypoints']  # (T, 17, 3)
    
    print(f"Video: {data['video_name']}")
    print(f"Original shape: {keypoints.shape}")
    print()
    
    # Method 1: Flatten keypoints
    features_flat = keypoints.reshape(keypoints.shape[0], -1)  # (T, 51)
    print(f"Method 1 - Flatten all keypoints:")
    print(f"  Shape: {features_flat.shape} (frames, features)")
    print(f"  Features per frame: {features_flat.shape[1]} (17 keypoints × 3 values)")
    print()
    
    # Method 2: Extract only coordinates (ignore confidence)
    features_coords = keypoints[:, :, :2].reshape(keypoints.shape[0], -1)  # (T, 34)
    print(f"Method 2 - Coordinates only:")
    print(f"  Shape: {features_coords.shape} (frames, features)")
    print(f"  Features per frame: {features_coords.shape[1]} (17 keypoints × 2 coords)")
    print()
    
    # Method 3: Filter by confidence and flatten
    confidence_threshold = 0.3
    features_filtered = keypoints.copy()
    mask = features_filtered[:, :, 2] < confidence_threshold
    features_filtered[mask, :2] = 0  # Zero out low-confidence keypoints
    features_filtered = features_filtered.reshape(features_filtered.shape[0], -1)
    
    print(f"Method 3 - Confidence-filtered:")
    print(f"  Shape: {features_filtered.shape}")
    print(f"  Threshold: {confidence_threshold}")
    print(f"  Avg keypoints above threshold: {np.sum(keypoints[:, :, 2] >= confidence_threshold, axis=1).mean():.1f}/17")
    print()


def example_4_le2i_frame_labels():
    """Example 4: Inspect Le2i frame-level labels."""
    print("="*70)
    print("Example 4: Le2i Frame-Level Labels")
    print("="*70)
    print()
    
    keypoints_dir = project_root / "data" / "interim" / "keypoints"
    
    # Find Le2i videos with falls
    le2i_files = list(keypoints_dir.glob("le2i_*.npz"))
    
    if not le2i_files:
        print("⚠️  No Le2i videos found. Run extraction first.")
        return
    
    # Find a video with falls
    fall_video = None
    for npz_path in le2i_files:
        data = np.load(npz_path)
        if data['label'] == 1 and 'frame_labels' in data:
            fall_video = (npz_path, data)
            break
    
    if not fall_video:
        print("⚠️  No Le2i fall videos found.")
        return
    
    npz_path, data = fall_video
    frame_labels = data['frame_labels']
    keypoints = data['keypoints']
    
    print(f"Video: {data['video_name']}")
    print(f"Total frames: {len(frame_labels)}")
    print()
    
    # Find fall segments
    fall_frames = np.where(frame_labels == 1)[0]
    
    if len(fall_frames) > 0:
        # Find continuous segments
        segments = []
        start = fall_frames[0]
        
        for i in range(1, len(fall_frames)):
            if fall_frames[i] != fall_frames[i-1] + 1:
                segments.append((start, fall_frames[i-1]))
                start = fall_frames[i]
        segments.append((start, fall_frames[-1]))
        
        print(f"Fall segments: {len(segments)}")
        for i, (start, end) in enumerate(segments, 1):
            duration = end - start + 1
            print(f"  Segment {i}: frames {start}-{end} ({duration} frames)")
        print()
        
        print(f"Fall frames: {len(fall_frames)} ({len(fall_frames)/len(frame_labels)*100:.1f}%)")
        print(f"Non-fall frames: {len(frame_labels) - len(fall_frames)} ({(len(frame_labels)-len(fall_frames))/len(frame_labels)*100:.1f}%)")
    else:
        print("No fall frames found (annotation may be missing)")
    
    print()


def example_5_confidence_analysis():
    """Example 5: Analyze keypoint detection confidence."""
    print("="*70)
    print("Example 5: Confidence Analysis")
    print("="*70)
    print()
    
    keypoints_dir = project_root / "data" / "interim" / "keypoints"
    
    # Load all URFD fall videos
    urfd_fall_files = list(keypoints_dir.glob("urfd_fall_*.npz"))
    
    if not urfd_fall_files:
        print("⚠️  No URFD fall videos found. Run extraction first.")
        return
    
    print(f"Analyzing {len(urfd_fall_files)} URFD fall videos")
    print()
    
    # Collect confidence statistics per keypoint
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    all_confidences = []
    
    for npz_path in urfd_fall_files[:5]:  # Analyze first 5 videos
        data = np.load(npz_path)
        keypoints = data['keypoints']
        all_confidences.append(keypoints[:, :, 2])
    
    # Average confidence per keypoint across all videos
    avg_confidences = np.concatenate(all_confidences, axis=0).mean(axis=0)
    
    print("Average confidence per keypoint:")
    print(f"  {'Keypoint':<20} {'Avg Confidence':<15} {'Detection Rate':<15}")
    print(f"  {'-'*50}")
    
    for i, name in enumerate(keypoint_names):
        conf = avg_confidences[i]
        detection_rate = (conf >= 0.3) * 100  # Rough estimate
        print(f"  {name:<20} {conf:.3f}           {detection_rate:.0f}%")
    
    print()
    print(f"Overall average confidence: {avg_confidences.mean():.3f}")
    print()


def main():
    """Run all examples."""
    print()
    print("="*70)
    print("Extracted Keypoints Inspection Examples")
    print("="*70)
    print()
    
    try:
        example_1_load_single_file()
        example_2_batch_load()
        example_3_feature_extraction()
        example_4_le2i_frame_labels()
        example_5_confidence_analysis()
        
        print("="*70)
        print("✅ All examples completed!")
        print("="*70)
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

