"""
Create 30-frame windowed dataset with RAW KEYPOINTS (34 features)

Phase 3.2+ Optimization:
- Sequence length: 30 frames (1.0 second @ 30 FPS) - captures rapid falls
- Features: 34 (17 keypoints × 2 coordinates) - raw keypoints, no feature engineering
- Let the model learn features automatically (simpler is better)

Inspired by fall-detection-deep-learning-master approach.
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json


def extract_raw_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract raw keypoints (x, y coordinates only).
    
    Args:
        keypoints: (T, 17, 3) array with [y, x, confidence]
    
    Returns:
        (T, 34) array with flattened [x, y] coordinates
    """
    T = keypoints.shape[0]
    features = np.zeros((T, 34), dtype=np.float32)
    
    for t in range(T):
        kp = keypoints[t]  # (17, 3)
        
        # Extract x, y coordinates (swap y, x to x, y order)
        xy_coords = kp[:, [1, 0]]  # (17, 2) - [x, y]
        
        # Flatten to (34,)
        features[t] = xy_coords.flatten()
    
    return features


def create_windows(features: np.ndarray, label: int, window_length: int = 30,
                   stride: int = 10, drop_threshold: float = 0.5) -> tuple:
    """
    Create sliding windows from feature sequence.
    
    Args:
        features: (T, 34) feature array
        label: Video-level label (0 or 1)
        window_length: Window size in frames
        stride: Stride between windows
        drop_threshold: Drop windows with >this fraction of zeros
    
    Returns:
        windows: (N, window_length, 34) array
        labels: (N,) array
    """
    T = features.shape[0]
    
    if T < window_length:
        # Pad short sequences
        padding = np.zeros((window_length - T, 34), dtype=np.float32)
        features = np.vstack([features, padding])
        T = window_length
    
    windows = []
    labels = []
    
    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        window = features[start:end]
        
        # Check for missing data (all zeros in a frame)
        missing_frames = np.sum(np.all(window == 0, axis=1))
        missing_ratio = missing_frames / window_length
        
        if missing_ratio <= drop_threshold:
            windows.append(window)
            labels.append(label)
    
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int32)


def process_dataset(source_dir: Path, window_length: int = 30, stride: int = 10,
                    drop_threshold: float = 0.5) -> tuple:
    """
    Process all .npz files in source directory.
    
    Args:
        source_dir: Directory with .npz keypoint files
        window_length: Window size in frames
        stride: Stride between windows
        drop_threshold: Drop windows with >this fraction missing
    
    Returns:
        X: (N, window_length, 34) array
        y: (N,) array
        video_ids: (N,) array with video identifiers
    """
    all_windows = []
    all_labels = []
    all_video_ids = []
    
    npz_files = sorted(source_dir.glob('**/*.npz'))
    
    print(f"\nProcessing {len(npz_files)} files from {source_dir}")
    
    for npz_path in tqdm(npz_files, desc="Processing videos"):
        try:
            data = np.load(npz_path, allow_pickle=True)
            keypoints = data['keypoints']  # (T, 17, 3)
            
            # Determine label from filename
            filename = npz_path.stem.lower()
            if 'fall' in filename or 'chute' in filename:
                label = 1
            else:
                label = 0
            
            # Extract raw keypoints
            features = extract_raw_keypoints(keypoints)
            
            # Create windows
            windows, labels = create_windows(
                features, label, window_length, stride, drop_threshold
            )
            
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.append(labels)
                
                # Create video IDs
                video_id = npz_path.stem
                video_ids = np.array([video_id] * len(windows))
                all_video_ids.append(video_ids)
        
        except Exception as e:
            print(f"\nError processing {npz_path}: {e}")
            continue
    
    # Concatenate all windows
    X = np.vstack(all_windows)
    y = np.concatenate(all_labels)
    video_ids = np.concatenate(all_video_ids)
    
    return X, y, video_ids


def main():
    parser = argparse.ArgumentParser(
        description='Create 30-frame dataset with raw keypoints (34 features)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='data/interim/keypoints',
        help='Source directory with keypoints .npz files'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/processed',
        help='Output directory'
    )
    parser.add_argument(
        '--length',
        type=int,
        default=30,
        help='Window length in frames (default: 30 = 1.0 second)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='Stride between windows (default: 10 = 0.33 seconds)'
    )
    parser.add_argument(
        '--drop-threshold',
        type=float,
        default=0.5,
        help='Drop windows with >this fraction missing'
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CREATE 30-FRAME RAW KEYPOINTS DATASET")
    print("="*70)
    print(f"Source: {source_dir}")
    print(f"Output: {out_dir}")
    print(f"Window length: {args.length} frames ({args.length/30:.2f} seconds @ 30 FPS)")
    print(f"Stride: {args.stride} frames ({args.stride/30:.2f} seconds)")
    print(f"Drop threshold: {args.drop_threshold}")
    print(f"Features: 34 (17 keypoints × 2 coordinates)")
    
    # Process dataset
    X, y, video_ids = process_dataset(
        source_dir,
        window_length=args.length,
        stride=args.stride,
        drop_threshold=args.drop_threshold
    )
    
    # Print statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total windows: {len(X):,}")
    print(f"Fall windows: {np.sum(y == 1):,} ({100*np.mean(y):.1f}%)")
    print(f"Non-fall windows: {np.sum(y == 0):,} ({100*np.mean(y == 0):.1f}%)")
    print(f"Shape: {X.shape} (N, T, features)")
    print(f"Unique videos: {len(np.unique(video_ids))}")
    
    # Class distribution
    print(f"\nClass distribution:")
    print(f"  Class 0 (non-fall): {np.sum(y == 0):,} windows")
    print(f"  Class 1 (fall): {np.sum(y == 1):,} windows")
    print(f"  Imbalance ratio: 1:{np.sum(y == 0) / max(np.sum(y == 1), 1):.2f}")
    
    # Save dataset
    output_path = out_dir / f'all_windows_30frame_raw.npz'
    print(f"\nSaving to {output_path}")
    
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        video_ids=video_ids
    )
    
    # Save metadata
    metadata = {
        'window_length': args.length,
        'stride': args.stride,
        'drop_threshold': args.drop_threshold,
        'num_features': 34,
        'feature_type': 'raw_keypoints',
        'feature_description': '17 keypoints × 2 coordinates (x, y)',
        'total_windows': int(len(X)),
        'fall_windows': int(np.sum(y == 1)),
        'non_fall_windows': int(np.sum(y == 0)),
        'unique_videos': int(len(np.unique(video_ids))),
        'shape': list(X.shape)
    }
    
    metadata_path = out_dir / f'all_windows_30frame_raw_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    print("\n" + "="*70)
    print("✅ DATASET CREATION COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Train model: python -m ml.training.lstm_train_raw_keypoints --data {output_path}")
    print(f"2. Evaluate: python -m ml.training.evaluate_model --model <model_path>")


if __name__ == '__main__':
    main()

