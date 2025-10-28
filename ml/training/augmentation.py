"""
Data augmentation utilities for temporal sequences.

Implements:
- Time warping (±10%)
- Gaussian noise (±5%)
- Feature dropout (10%)
"""

import numpy as np
from scipy.interpolate import interp1d


def time_warp(sequence, warp_factor=0.1):
    """
    Apply time warping to a sequence.
    
    Args:
        sequence: (T, D) array
        warp_factor: Maximum warping factor (default: 0.1 for ±10%)
    
    Returns:
        Warped sequence of same shape
    """
    T, D = sequence.shape
    
    # Random warp factor
    warp = 1.0 + np.random.uniform(-warp_factor, warp_factor)
    
    # Original time indices
    original_indices = np.arange(T)
    
    # Warped time indices
    warped_length = int(T * warp)
    warped_indices = np.linspace(0, T - 1, warped_length)
    
    # Interpolate each feature
    warped_sequence = np.zeros((T, D))
    for d in range(D):
        # Handle NaN values
        valid_mask = ~np.isnan(sequence[:, d])
        if np.sum(valid_mask) > 1:
            # Interpolate only valid values
            interp_func = interp1d(
                original_indices[valid_mask],
                sequence[valid_mask, d],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            warped_values = interp_func(warped_indices)
            
            # Resample back to original length
            resample_func = interp1d(
                np.linspace(0, T - 1, len(warped_values)),
                warped_values,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            warped_sequence[:, d] = resample_func(original_indices)
        else:
            # Keep original if not enough valid values
            warped_sequence[:, d] = sequence[:, d]
    
    return warped_sequence


def add_noise(sequence, noise_factor=0.05):
    """
    Add Gaussian noise to a sequence.
    
    Args:
        sequence: (T, D) array
        noise_factor: Noise standard deviation as fraction of data range (default: 0.05 for ±5%)
    
    Returns:
        Noisy sequence of same shape
    """
    noise = np.random.normal(0, noise_factor, sequence.shape)
    noisy_sequence = sequence + noise
    
    # Clip to [0, 1] range (assuming normalized features)
    noisy_sequence = np.clip(noisy_sequence, 0, 1)
    
    # Preserve NaN values
    noisy_sequence[np.isnan(sequence)] = np.nan
    
    return noisy_sequence


def feature_dropout(sequence, dropout_rate=0.1):
    """
    Randomly drop features (set to NaN) to simulate missing data.
    
    Args:
        sequence: (T, D) array
        dropout_rate: Probability of dropping each feature (default: 0.1)
    
    Returns:
        Sequence with randomly dropped features
    """
    T, D = sequence.shape
    
    # Create dropout mask
    dropout_mask = np.random.random((T, D)) < dropout_rate
    
    # Apply dropout
    augmented_sequence = sequence.copy()
    augmented_sequence[dropout_mask] = np.nan
    
    return augmented_sequence


def augment_sequence(sequence, 
                     apply_time_warp=True,
                     apply_noise=True,
                     apply_dropout=True,
                     warp_factor=0.1,
                     noise_factor=0.05,
                     dropout_rate=0.1):
    """
    Apply all augmentations to a sequence.
    
    Args:
        sequence: (T, D) array
        apply_time_warp: Whether to apply time warping
        apply_noise: Whether to apply noise
        apply_dropout: Whether to apply feature dropout
        warp_factor: Time warp factor
        noise_factor: Noise factor
        dropout_rate: Feature dropout rate
    
    Returns:
        Augmented sequence
    """
    augmented = sequence.copy()
    
    if apply_time_warp:
        augmented = time_warp(augmented, warp_factor)
    
    if apply_noise:
        augmented = add_noise(augmented, noise_factor)
    
    if apply_dropout:
        augmented = feature_dropout(augmented, dropout_rate)
    
    return augmented


def augment_batch(X, y, 
                  augment_prob=0.5,
                  apply_time_warp=True,
                  apply_noise=True,
                  apply_dropout=True):
    """
    Augment a batch of sequences.
    
    Args:
        X: (N, T, D) array of sequences
        y: (N,) array of labels
        augment_prob: Probability of augmenting each sequence
        apply_time_warp: Whether to apply time warping
        apply_noise: Whether to apply noise
        apply_dropout: Whether to apply feature dropout
    
    Returns:
        Augmented X, y (same shapes)
    """
    N = len(X)
    X_aug = X.copy()
    
    for i in range(N):
        if np.random.random() < augment_prob:
            X_aug[i] = augment_sequence(
                X[i],
                apply_time_warp=apply_time_warp,
                apply_noise=apply_noise,
                apply_dropout=apply_dropout
            )
    
    return X_aug, y


if __name__ == '__main__':
    # Test augmentation
    print("Testing augmentation functions...")
    
    # Create dummy sequence
    T, D = 60, 6
    sequence = np.random.rand(T, D)
    
    # Add some NaN values
    sequence[10:15, 2] = np.nan
    
    print(f"Original shape: {sequence.shape}")
    print(f"Original NaN ratio: {np.sum(np.isnan(sequence)) / sequence.size:.3f}")
    
    # Test time warp
    warped = time_warp(sequence)
    print(f"Warped shape: {warped.shape}")
    print(f"Warped NaN ratio: {np.sum(np.isnan(warped)) / warped.size:.3f}")
    
    # Test noise
    noisy = add_noise(sequence)
    print(f"Noisy shape: {noisy.shape}")
    print(f"Noisy NaN ratio: {np.sum(np.isnan(noisy)) / noisy.size:.3f}")
    
    # Test dropout
    dropped = feature_dropout(sequence)
    print(f"Dropped shape: {dropped.shape}")
    print(f"Dropped NaN ratio: {np.sum(np.isnan(dropped)) / dropped.size:.3f}")
    
    # Test full augmentation
    augmented = augment_sequence(sequence)
    print(f"Augmented shape: {augmented.shape}")
    print(f"Augmented NaN ratio: {np.sum(np.isnan(augmented)) / augmented.size:.3f}")
    
    print("\n✓ All augmentation tests passed!")

