"""
Create 60-frame windowed dataset from enhanced features for Phase 2.3a

This script creates a dataset with 60-frame windows (2 seconds at 30 fps)
instead of 90-frame windows, which may help the model focus on more
critical temporal patterns and reduce overfitting.

Input: data/interim/keypoints/*.npy (enhanced features with 14 channels)
Output: data/processed/all_windows_60frame.npz
"""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def load_enhanced_features(npy_path: Path) -> tuple:
    """
    Load enhanced features from .npy file.
    
    Returns:
        features: (T, 14) array
        label: 0 or 1
        video_id: str
    """
    data = np.load(npy_path, allow_pickle=True).item()
    features = data['features']  # (T, 14)
    label = data['label']
    video_id = data['video_id']
    
    return features, label, video_id


def create_windows(features: np.ndarray, label: int, video_id: str,
                   window_length: int = 60, stride: int = 15) -> list:
    """
    Create sliding windows from feature sequence.
    
    Args:
        features: (T, 14) array
        label: 0 or 1
        video_id: str
        window_length: Window size in frames (default: 60 = 2 seconds)
        stride: Stride in frames (default: 15 = 0.5 seconds)
        
    Returns:
        List of (window, label, video_id) tuples
    """
    T, num_features = features.shape
    
    if T < window_length:
        # Pad short sequences
        pad_length = window_length - T
        features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        T = window_length
    
    windows = []
    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        window = features[start:end]  # (60, 14)
        windows.append((window, label, video_id))
    
    return windows


def process_file(npy_path: Path, window_length: int, stride: int) -> list:
    """Process a single .npy file and return windows."""
    try:
        features, label, video_id = load_enhanced_features(npy_path)
        windows = create_windows(features, label, video_id, window_length, stride)
        return windows
    except Exception as e:
        print(f"Error processing {npy_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Create 60-frame windowed dataset')
    parser.add_argument('--source', type=str, default='data/interim/keypoints',
                        help='Source directory with enhanced .npy files')
    parser.add_argument('--out', type=str, default='data/processed',
                        help='Output directory')
    parser.add_argument('--length', type=int, default=60,
                        help='Window length in frames (default: 60 = 2 seconds)')
    parser.add_argument('--stride', type=int, default=15,
                        help='Stride in frames (default: 15 = 0.5 seconds)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .npy files
    npy_files = sorted(source_dir.glob('*.npy'))
    print(f"Found {len(npy_files)} .npy files in {source_dir}")
    
    if len(npy_files) == 0:
        print("ERROR: No .npy files found!")
        return
    
    # Process files in parallel
    print(f"Creating {args.length}-frame windows with stride={args.stride}...")
    process_fn = partial(process_file, window_length=args.length, stride=args.stride)
    
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(process_fn, npy_files),
            total=len(npy_files),
            desc='Processing files'
        ))
    
    # Flatten results
    all_windows = []
    for windows in results:
        all_windows.extend(windows)
    
    print(f"Total windows created: {len(all_windows)}")
    
    # Convert to arrays
    X = np.array([w[0] for w in all_windows], dtype=np.float32)  # (N, 60, 14)
    y = np.array([w[1] for w in all_windows], dtype=np.int32)    # (N,)
    video_ids = np.array([w[2] for w in all_windows], dtype=object)  # (N,)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"video_ids shape: {video_ids.shape}")
    print(f"Fall samples: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"Non-fall samples: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"Unique videos: {len(np.unique(video_ids))}")
    
    # Save to .npz
    out_path = out_dir / 'all_windows_60frame.npz'
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        video_ids=video_ids,
        window_length=args.length,
        stride=args.stride,
        num_features=14,
        feature_names=[
            'nose_x', 'nose_y', 'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y',
            'left_knee_x', 'left_knee_y', 'right_knee_x', 'right_knee_y',
            'hip_velocity', 'knee_velocity', 'hip_accel', 'knee_accel'
        ]
    )
    
    print(f"✅ Dataset saved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()

