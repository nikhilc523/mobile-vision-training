"""
Phase 4.1: Create Balanced 30-Frame Raw Keypoints Dataset

Oversamples fall windows to 1:3 ratio with smart augmentations:
- Time-warp (±15%)
- Gaussian jitter (σ=0.02 on x,y)
- Temporal crop (±3 frames with padding)

Usage:
    python3 -m ml.features.create_balanced_dataset \
        --input data/processed/all_windows_30frame_raw.npz \
        --output data/processed/all_windows_30_raw_balanced.npz \
        --target-ratio 0.33
"""

import numpy as np
import argparse
from pathlib import Path
from scipy.interpolate import interp1d
from collections import Counter
import pandas as pd


def time_warp_augment(sequence: np.ndarray, warp_factor: float = 0.15) -> np.ndarray:
    """
    Apply time-warping augmentation by randomly stretching/compressing time.
    
    Args:
        sequence: (T, F) array
        warp_factor: Maximum warp factor (0.15 = ±15%)
    
    Returns:
        Warped sequence with same shape (T, F)
    """
    T, F = sequence.shape
    
    # Random warp factor in [-warp_factor, +warp_factor]
    warp = np.random.uniform(-warp_factor, warp_factor)
    
    # Create warped time indices
    original_indices = np.linspace(0, T - 1, T)
    warped_indices = np.linspace(0, T - 1, int(T * (1 + warp)))
    
    # Interpolate each feature
    warped_sequence = np.zeros((T, F), dtype=np.float32)
    for f in range(F):
        interpolator = interp1d(original_indices, sequence[:, f], 
                               kind='linear', fill_value='extrapolate')
        # Sample at warped indices, then resample to original length
        warped_values = interpolator(warped_indices)
        # Resample back to T frames
        resampler = interp1d(np.linspace(0, T - 1, len(warped_values)), warped_values,
                            kind='linear', fill_value='extrapolate')
        warped_sequence[:, f] = resampler(original_indices)
    
    return warped_sequence


def gaussian_jitter_augment(sequence: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """
    Add Gaussian noise to keypoint coordinates.
    
    Args:
        sequence: (T, F) array where F = 34 (17 keypoints × 2 coords)
        sigma: Standard deviation of Gaussian noise
    
    Returns:
        Jittered sequence with same shape (T, F)
    """
    noise = np.random.normal(0, sigma, sequence.shape).astype(np.float32)
    jittered = sequence + noise
    
    # Clip to valid range [0, 1] (MoveNet outputs are normalized)
    jittered = np.clip(jittered, 0.0, 1.0)
    
    return jittered


def temporal_crop_augment(sequence: np.ndarray, max_crop: int = 3) -> np.ndarray:
    """
    Randomly crop temporal sequence and pad back to original length.
    
    Args:
        sequence: (T, F) array
        max_crop: Maximum frames to crop from start/end
    
    Returns:
        Cropped and padded sequence with same shape (T, F)
    """
    T, F = sequence.shape
    
    # Random crop amount
    crop_start = np.random.randint(0, max_crop + 1)
    crop_end = np.random.randint(0, max_crop + 1)
    
    # Crop
    cropped = sequence[crop_start:T - crop_end]
    
    # Pad back to original length
    pad_start = crop_start
    pad_end = crop_end
    
    if len(cropped) == 0:
        # Edge case: cropped too much, return original
        return sequence.copy()
    
    padded = np.pad(cropped, ((pad_start, pad_end), (0, 0)), mode='edge')
    
    return padded.astype(np.float32)


def augment_fall_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply random combination of augmentations to a fall sequence.
    
    Args:
        sequence: (T, F) array
    
    Returns:
        Augmented sequence with same shape (T, F)
    """
    augmented = sequence.copy()
    
    # Apply augmentations with probability
    if np.random.rand() < 0.7:  # 70% chance
        augmented = time_warp_augment(augmented, warp_factor=0.15)
    
    if np.random.rand() < 0.7:  # 70% chance
        augmented = gaussian_jitter_augment(augmented, sigma=0.02)
    
    if np.random.rand() < 0.5:  # 50% chance
        augmented = temporal_crop_augment(augmented, max_crop=3)
    
    return augmented


def oversample_fall_class(X: np.ndarray, y: np.ndarray, 
                          video_ids: np.ndarray,
                          target_ratio: float = 0.33) -> tuple:
    """
    Oversample fall class to achieve target ratio with augmentations.
    
    Args:
        X: (N, T, F) array of sequences
        y: (N,) array of labels (0=non-fall, 1=fall)
        video_ids: (N,) array of video IDs
        target_ratio: Target ratio of fall:total (0.33 = 1:3 ratio)
    
    Returns:
        Tuple of (X_balanced, y_balanced, video_ids_balanced)
    """
    # Separate fall and non-fall samples
    fall_mask = y == 1
    non_fall_mask = y == 0
    
    X_fall = X[fall_mask]
    X_non_fall = X[non_fall_mask]
    
    video_ids_fall = video_ids[fall_mask]
    video_ids_non_fall = video_ids[non_fall_mask]
    
    n_fall = len(X_fall)
    n_non_fall = len(X_non_fall)
    
    print(f"\n=== Original Dataset ===")
    print(f"Fall samples: {n_fall}")
    print(f"Non-fall samples: {n_non_fall}")
    print(f"Ratio (fall:total): {n_fall / (n_fall + n_non_fall):.4f}")
    print(f"Imbalance ratio (fall:non-fall): 1:{n_non_fall / n_fall:.2f}")
    
    # Calculate target number of fall samples
    # target_ratio = n_fall_target / (n_fall_target + n_non_fall)
    # n_fall_target = target_ratio * (n_fall_target + n_non_fall)
    # n_fall_target * (1 - target_ratio) = target_ratio * n_non_fall
    # n_fall_target = target_ratio * n_non_fall / (1 - target_ratio)
    
    n_fall_target = int(target_ratio * n_non_fall / (1 - target_ratio))
    
    print(f"\n=== Target Dataset ===")
    print(f"Target fall samples: {n_fall_target}")
    print(f"Target ratio (fall:total): {n_fall_target / (n_fall_target + n_non_fall):.4f}")
    print(f"Target imbalance ratio (fall:non-fall): 1:{n_non_fall / n_fall_target:.2f}")
    
    # Oversample fall class with augmentations
    n_augment = n_fall_target - n_fall
    
    print(f"\n=== Augmentation ===")
    print(f"Need to generate {n_augment} augmented fall samples")
    
    if n_augment <= 0:
        print("No augmentation needed (already balanced)")
        return X, y, video_ids
    
    # Generate augmented samples
    X_fall_augmented = []
    video_ids_fall_augmented = []
    
    for i in range(n_augment):
        # Randomly select a fall sample
        idx = np.random.randint(0, n_fall)
        original_sequence = X_fall[idx]
        original_video_id = video_ids_fall[idx]
        
        # Augment
        augmented_sequence = augment_fall_sequence(original_sequence)
        
        X_fall_augmented.append(augmented_sequence)
        video_ids_fall_augmented.append(f"{original_video_id}_aug{i}")
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_augment} augmented samples...")
    
    X_fall_augmented = np.array(X_fall_augmented, dtype=np.float32)
    video_ids_fall_augmented = np.array(video_ids_fall_augmented)
    
    # Combine original and augmented fall samples
    X_fall_combined = np.concatenate([X_fall, X_fall_augmented], axis=0)
    video_ids_fall_combined = np.concatenate([video_ids_fall, video_ids_fall_augmented], axis=0)
    
    # Combine with non-fall samples
    X_balanced = np.concatenate([X_fall_combined, X_non_fall], axis=0)
    y_balanced = np.concatenate([
        np.ones(len(X_fall_combined), dtype=np.int32),
        np.zeros(len(X_non_fall), dtype=np.int32)
    ], axis=0)
    video_ids_balanced = np.concatenate([video_ids_fall_combined, video_ids_non_fall], axis=0)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    video_ids_balanced = video_ids_balanced[shuffle_idx]
    
    print(f"\n=== Balanced Dataset ===")
    print(f"Total samples: {len(X_balanced)}")
    print(f"Fall samples: {np.sum(y_balanced == 1)}")
    print(f"Non-fall samples: {np.sum(y_balanced == 0)}")
    print(f"Ratio (fall:total): {np.mean(y_balanced):.4f}")
    print(f"Imbalance ratio (fall:non-fall): 1:{np.sum(y_balanced == 0) / np.sum(y_balanced == 1):.2f}")
    
    return X_balanced, y_balanced, video_ids_balanced


def main():
    parser = argparse.ArgumentParser(description='Create balanced 30-frame raw keypoints dataset')
    parser.add_argument('--input', type=str, 
                       default='data/processed/all_windows_30frame_raw.npz',
                       help='Input dataset path')
    parser.add_argument('--output', type=str,
                       default='data/processed/all_windows_30_raw_balanced.npz',
                       help='Output dataset path')
    parser.add_argument('--target-ratio', type=float, default=0.33,
                       help='Target ratio of fall:total (0.33 = 1:3 ratio)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("PHASE 4.1: CREATE BALANCED DATASET")
    print("=" * 70)
    
    # Load original dataset
    print(f"\n[1/4] Loading original dataset from {args.input}...")
    data = np.load(args.input, allow_pickle=True)
    
    X = data['X']  # (N, 30, 34)
    y = data['y']  # (N,)
    video_ids = data['video_ids']  # (N,)
    
    print(f"✓ Loaded dataset")
    print(f"  Shape: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Video IDs: {video_ids.shape}")
    
    # Oversample fall class
    print(f"\n[2/4] Oversampling fall class to target ratio {args.target_ratio:.2f}...")
    X_balanced, y_balanced, video_ids_balanced = oversample_fall_class(
        X, y, video_ids, target_ratio=args.target_ratio
    )
    
    # Save balanced dataset
    print(f"\n[3/4] Saving balanced dataset to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        X=X_balanced,
        y=y_balanced,
        video_ids=video_ids_balanced
    )
    
    print(f"✓ Saved balanced dataset")
    
    # Save class histogram CSV
    print(f"\n[4/4] Saving class histogram...")
    csv_path = Path('docs/wiki_assets/phase4_balanced/class_counts.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    class_counts = Counter(y_balanced)
    df = pd.DataFrame({
        'class': ['non-fall', 'fall'],
        'count': [class_counts[0], class_counts[1]],
        'percentage': [
            100 * class_counts[0] / len(y_balanced),
            100 * class_counts[1] / len(y_balanced)
        ]
    })
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Saved class histogram to {csv_path}")
    print(f"\n{df.to_string(index=False)}")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 4.1 COMPLETE")
    print("=" * 70)
    print(f"\nBalanced dataset saved to: {args.output}")
    print(f"Class histogram saved to: {csv_path}")
    print(f"\nNext steps:")
    print(f"1. Train new model with balanced dataset")
    print(f"2. Test on secondfall.mp4")
    print(f"3. Compare with previous model (ROC-AUC: 0.9205, Recall: 0.55)")


if __name__ == '__main__':
    main()

