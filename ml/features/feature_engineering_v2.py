"""
Feature Engineering v2 with 16 Features

Extends the 14-feature set with 2 additional acceleration features:
- Features 0-9: Original 10 features (torso_angle, hip_height, vertical_velocity, etc.)
- Feature 10: Vertical acceleration (Δv_y / Δt)
- Feature 11: Angular velocity (Δθ / Δt)
- Feature 12: Stillness ratio
- Feature 13: Pose stability
- Feature 14: Angular acceleration (Δω / Δt) - NEW
- Feature 15: Jerk (Δa / Δt) - vertical jerk - NEW

Total: 16 features for Phase 2.4
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import sys
import argparse
from tqdm import tqdm

# Import from enhanced feature engineering
from ml.features.feature_engineering_enhanced import (
    extract_enhanced_features,
    compute_vertical_acceleration,
    compute_angular_velocity,
    compute_stillness_ratio,
    compute_pose_stability
)

# Import utilities from full feature engineering
from ml.features.feature_engineering_full import (
    extract_all_features,
    interpolate_missing_keypoints,
    normalize_features,
    smooth_features,
    create_windows,
    load_le2i_annotations,
    CONF_THRESHOLD
)


def compute_angular_acceleration(angular_velocity: np.ndarray, fps: int = 30) -> np.ndarray:
    """
    Feature 15: Angular acceleration (second derivative of torso angle).
    
    Falls show characteristic angular acceleration patterns:
    - Rapid change in rotational velocity
    - High magnitude during fall initiation
    
    Args:
        angular_velocity: (T,) array of angular velocities (degrees/second)
        fps: frames per second
        
    Returns:
        (T,) array of angular accelerations (degrees/second²)
    """
    T = len(angular_velocity)
    accelerations = np.full(T, np.nan)
    dt = 1.0 / fps
    
    for t in range(1, T):
        if not np.isnan(angular_velocity[t]) and not np.isnan(angular_velocity[t-1]):
            accelerations[t] = (angular_velocity[t] - angular_velocity[t-1]) / dt
    
    return accelerations


def compute_vertical_jerk(vertical_acceleration: np.ndarray, fps: int = 30) -> np.ndarray:
    """
    Feature 16: Vertical jerk (third derivative of hip height).
    
    Jerk captures the rate of change of acceleration:
    - Sudden impact events show high jerk
    - Smooth motions show low jerk
    
    Args:
        vertical_acceleration: (T,) array of vertical accelerations
        fps: frames per second
        
    Returns:
        (T,) array of vertical jerk values
    """
    T = len(vertical_acceleration)
    jerk = np.full(T, np.nan)
    dt = 1.0 / fps
    
    for t in range(1, T):
        if not np.isnan(vertical_acceleration[t]) and not np.isnan(vertical_acceleration[t-1]):
            jerk[t] = (vertical_acceleration[t] - vertical_acceleration[t-1]) / dt
    
    return jerk


def extract_all_16_features(keypoints: np.ndarray, fps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all 16 features (14 existing + 2 new acceleration features).
    
    Args:
        keypoints: (T, 17, 3) array with [y, x, conf]
        fps: frames per second
        
    Returns:
        raw_features: (T, 16) array of raw features
        smoothed_features: (T, 16) array of smoothed features
    """
    T = keypoints.shape[0]

    # Extract original 14 features
    features_14_raw, features_14_smooth = extract_enhanced_features(keypoints, fps)
    
    # Initialize 16-feature array
    features_16 = np.zeros((T, 16))
    features_16[:, :14] = features_14_smooth  # Use smoothed versions of first 14 features
    
    # Extract new acceleration features
    # Feature 15: Angular acceleration (from feature 11: angular velocity)
    features_16[:, 14] = compute_angular_acceleration(features_14_smooth[:, 11], fps)
    
    # Feature 16: Vertical jerk (from feature 10: vertical acceleration)
    features_16[:, 15] = compute_vertical_jerk(features_14_smooth[:, 10], fps)
    
    # Normalize all 16 features together
    features_16_norm = normalize_features(features_16)
    
    # Smooth the new derived features (first 14 already smoothed)
    features_16_smooth = features_16_norm.copy()
    features_16_smooth[:, 14:] = smooth_features(features_16_norm[:, 14:])
    
    return features_16_norm, features_16_smooth


def process_single_video(
    npz_path: Path,
    window_length: int = 60,
    stride: int = 10,
    drop_threshold: float = 0.5
) -> Dict:
    """
    Process a single video: extract 16 features and create windows.
    
    Args:
        npz_path: path to .npz keypoint file
        window_length: window size in frames
        stride: stride in frames
        drop_threshold: drop windows with >this fraction missing
        
    Returns:
        Dictionary with windows, labels, and metadata
    """
    try:
        # Load keypoints
        data = np.load(npz_path)
        keypoints = data['keypoints']
        video_label = int(data['label'])
        dataset = str(data['dataset'])
        fps = int(data.get('fps', 30))
        video_name = str(data.get('video_name', npz_path.stem))
        
        T = keypoints.shape[0]
        
        # Extract 16 features
        _, features_smooth = extract_all_16_features(keypoints, fps)
        
        # Create frame-level labels
        if dataset == 'le2i':
            # Try to load frame-level annotations
            frame_labels = load_le2i_annotations(video_name)
            if frame_labels is None or len(frame_labels) != T:
                # Fallback: use video-level label
                frame_labels = np.full(T, video_label)
        else:
            # URFD and UCF101: use video-level label for all frames
            frame_labels = np.full(T, video_label)
        
        # Create windows
        X, y, num_dropped = create_windows(
            features_smooth,
            frame_labels,
            window_length,
            stride,
            drop_threshold
        )
        
        return {
            'X': X,
            'y': y,
            'num_dropped': num_dropped,
            'dataset': dataset,
            'video_name': video_name,
            'success': True
        }
        
    except Exception as e:
        return {
            'X': np.array([]),
            'y': np.array([]),
            'num_dropped': 0,
            'dataset': 'unknown',
            'video_name': npz_path.stem,
            'success': False,
            'error': str(e)
        }


def process_dataset(
    source_dir: Path,
    output_dir: Path,
    window_length: int = 60,
    stride: int = 10,
    drop_threshold: float = 0.5
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Process all videos in the dataset.
    
    Args:
        source_dir: directory containing .npz keypoint files
        output_dir: directory to save processed windows
        window_length: window size
        stride: stride
        drop_threshold: quality threshold
        
    Returns:
        Tuple of (statistics dict, X_all, y_all)
    """
    # Find all .npz files
    npz_files = sorted(list(source_dir.glob('*.npz')))
    
    print(f"\n{'='*80}")
    print(f"PHASE 2.4a — FEATURE ENGINEERING V2 (16 FEATURES)")
    print(f"{'='*80}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Total videos: {len(npz_files)}")
    print(f"Window: {window_length} frames, Stride: {stride}")
    print(f"Drop threshold: {drop_threshold * 100:.0f}%")
    print(f"{'='*80}\n")
    
    # Process by dataset
    datasets = {'urfd': [], 'le2i': [], 'ucf101': []}
    all_X = []
    all_y = []
    all_video_ids = []
    
    stats = {
        'urfd': {'videos': 0, 'windows': 0, 'pos': 0, 'neg': 0, 'dropped': 0},
        'le2i': {'videos': 0, 'windows': 0, 'pos': 0, 'neg': 0, 'dropped': 0},
        'ucf101': {'videos': 0, 'windows': 0, 'pos': 0, 'neg': 0, 'dropped': 0}
    }
    
    # Process all videos
    for npz_path in tqdm(npz_files, desc="Processing videos"):
        result = process_single_video(npz_path, window_length, stride, drop_threshold)
        
        if not result['success']:
            continue
        
        dataset = result['dataset']
        X = result['X']
        y = result['y']
        
        if len(X) == 0:
            continue
        
        # Update statistics
        stats[dataset]['videos'] += 1
        stats[dataset]['windows'] += len(X)
        stats[dataset]['pos'] += np.sum(y == 1)
        stats[dataset]['neg'] += np.sum(y == 0)
        stats[dataset]['dropped'] += result['num_dropped']
        
        # Collect data
        all_X.append(X)
        all_y.append(y)
        all_video_ids.extend([npz_path.stem] * len(X))
        
        datasets[dataset].append({
            'X': X,
            'y': y,
            'video_id': npz_path.stem
        })
    
    # Combine all data
    if len(all_X) == 0:
        print("❌ No windows generated!")
        return stats, np.array([]), np.array([])
    
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    
    # Save combined dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'all_windows_v2.npz'
    
    np.savez_compressed(
        output_path,
        X=X_all,
        y=y_all,
        video_ids=np.array(all_video_ids, dtype=object)
    )
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"DATASET STATISTICS")
    print(f"{'='*80}")
    
    total_videos = sum(s['videos'] for s in stats.values())
    total_windows = sum(s['windows'] for s in stats.values())
    total_pos = sum(s['pos'] for s in stats.values())
    total_neg = sum(s['neg'] for s in stats.values())
    total_dropped = sum(s['dropped'] for s in stats.values())
    
    for dataset_name in ['urfd', 'le2i', 'ucf101']:
        s = stats[dataset_name]
        if s['videos'] > 0:
            print(f"\n{dataset_name.upper()}:")
            print(f"  Videos: {s['videos']}")
            print(f"  Windows: {s['windows']} (Fall: {s['pos']}, Non-fall: {s['neg']})")
            print(f"  Dropped: {s['dropped']}")
    
    print(f"\nTOTAL:")
    print(f"  Videos: {total_videos}")
    print(f"  Windows: {total_windows} (Fall: {total_pos}, Non-fall: {total_neg})")
    print(f"  Class ratio: {total_pos / total_windows * 100:.1f}% fall")
    print(f"  Dropped: {total_dropped}")
    print(f"\nOutput shape: X={X_all.shape}, y={y_all.shape}")
    print(f"Saved to: {output_path}")
    print(f"{'='*80}\n")
    
    return stats, X_all, y_all


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering v2 - Extract 16 features from full dataset"
    )
    parser.add_argument(
        '--source',
        type=str,
        default='data/interim/keypoints',
        help='Source directory with .npz keypoint files'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/processed',
        help='Output directory for windowed features'
    )
    parser.add_argument(
        '--length',
        type=int,
        default=60,
        help='Window length in frames'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='Window stride in frames'
    )
    parser.add_argument(
        '--drop-threshold',
        type=float,
        default=0.5,
        help='Drop windows with >this fraction missing'
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.out)
    
    if not source_dir.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    # Process dataset
    stats, X_all, y_all = process_dataset(
        source_dir,
        output_dir,
        window_length=args.length,
        stride=args.stride,
        drop_threshold=args.drop_threshold
    )
    
    print("✅ Feature engineering v2 complete!")


if __name__ == '__main__':
    main()

