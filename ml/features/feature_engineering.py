#!/usr/bin/env python3
"""
Feature Engineering for Fall Detection

This module extracts engineered features from pose keypoints for LSTM training.

MoveNet Keypoint Indices (COCO format):
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

Features extracted per frame:
1. Torso angle (α) - angle between neck-hip line and vertical
2. Hip height (h) - 1 - average(hip_y_left, hip_y_right)
3. Vertical velocity (v) - Δ(hip height) / Δt
4. Motion magnitude (m) - mean L2 displacement of visible keypoints
5. Shoulder symmetry (s) - |left_shoulder_y - right_shoulder_y|
6. Knee angle (θ_knee) - max knee angle
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Optional
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# MoveNet keypoint indices
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Confidence threshold for masking
CONF_THRESHOLD = 0.3


def compute_torso_angle(keypoints: np.ndarray, conf_threshold: float = CONF_THRESHOLD) -> np.ndarray:
    """
    Compute torso angle relative to vertical.
    
    Uses neck proxy (midpoint of shoulders) to hip center.
    
    Args:
        keypoints: (T, 17, 3) array with [y, x, conf]
        conf_threshold: minimum confidence threshold
        
    Returns:
        (T,) array of angles in degrees, NaN where confidence is low
    """
    T = keypoints.shape[0]
    angles = np.full(T, np.nan)
    
    for t in range(T):
        # Get shoulder keypoints
        left_shoulder = keypoints[t, LEFT_SHOULDER]
        right_shoulder = keypoints[t, RIGHT_SHOULDER]
        left_hip = keypoints[t, LEFT_HIP]
        right_hip = keypoints[t, RIGHT_HIP]
        
        # Check confidence
        if (left_shoulder[2] < conf_threshold or right_shoulder[2] < conf_threshold or
            left_hip[2] < conf_threshold or right_hip[2] < conf_threshold):
            continue
        
        # Compute neck proxy (midpoint of shoulders)
        neck_y = (left_shoulder[0] + right_shoulder[0]) / 2
        neck_x = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Compute hip center
        hip_y = (left_hip[0] + right_hip[0]) / 2
        hip_x = (left_hip[1] + right_hip[1]) / 2
        
        # Compute angle with vertical (y-axis)
        # Vertical vector is (1, 0) in (y, x) coordinates
        dy = hip_y - neck_y
        dx = hip_x - neck_x
        
        # Angle from vertical
        angle = np.degrees(np.arctan2(dx, dy))
        angles[t] = abs(angle)
    
    return angles


def compute_hip_height(keypoints: np.ndarray, conf_threshold: float = CONF_THRESHOLD) -> np.ndarray:
    """
    Compute hip height as 1 - average(hip_y_left, hip_y_right).
    
    Higher values mean person is higher in frame (standing).
    
    Args:
        keypoints: (T, 17, 3) array with [y, x, conf]
        conf_threshold: minimum confidence threshold
        
    Returns:
        (T,) array of hip heights, NaN where confidence is low
    """
    T = keypoints.shape[0]
    heights = np.full(T, np.nan)
    
    for t in range(T):
        left_hip = keypoints[t, LEFT_HIP]
        right_hip = keypoints[t, RIGHT_HIP]
        
        # Check confidence
        if left_hip[2] < conf_threshold or right_hip[2] < conf_threshold:
            continue
        
        # Compute average hip y-coordinate
        avg_hip_y = (left_hip[0] + right_hip[0]) / 2
        
        # Height is 1 - y (since y=0 is top, y=1 is bottom)
        heights[t] = 1.0 - avg_hip_y
    
    return heights


def compute_vertical_velocity(
    hip_heights: np.ndarray,
    fps: int = 30
) -> np.ndarray:
    """
    Compute vertical velocity as Δ(hip height) / Δt.
    
    Args:
        hip_heights: (T,) array of hip heights (may contain NaN)
        fps: frames per second
        
    Returns:
        (T,) array of velocities, NaN where data is missing
    """
    T = len(hip_heights)
    velocities = np.full(T, np.nan)
    
    # Compute frame-to-frame differences
    dt = 1.0 / fps
    
    for t in range(1, T):
        if not np.isnan(hip_heights[t]) and not np.isnan(hip_heights[t-1]):
            velocities[t] = (hip_heights[t] - hip_heights[t-1]) / dt
    
    return velocities


def compute_motion_magnitude(
    keypoints: np.ndarray,
    conf_threshold: float = CONF_THRESHOLD
) -> np.ndarray:
    """
    Compute mean L2 displacement of all visible keypoints from t-1 to t.
    
    Args:
        keypoints: (T, 17, 3) array with [y, x, conf]
        conf_threshold: minimum confidence threshold
        
    Returns:
        (T,) array of motion magnitudes, NaN where insufficient data
    """
    T = keypoints.shape[0]
    motion = np.full(T, np.nan)
    
    for t in range(1, T):
        displacements = []
        
        for kp_idx in range(17):
            curr = keypoints[t, kp_idx]
            prev = keypoints[t-1, kp_idx]
            
            # Check confidence for both frames
            if curr[2] >= conf_threshold and prev[2] >= conf_threshold:
                # Compute L2 distance
                dy = curr[0] - prev[0]
                dx = curr[1] - prev[1]
                dist = np.sqrt(dy**2 + dx**2)
                displacements.append(dist)
        
        # Compute mean displacement if we have at least 3 visible keypoints
        if len(displacements) >= 3:
            motion[t] = np.mean(displacements)
    
    return motion


def compute_shoulder_symmetry(
    keypoints: np.ndarray,
    conf_threshold: float = CONF_THRESHOLD
) -> np.ndarray:
    """
    Compute shoulder symmetry as |left_shoulder_y - right_shoulder_y|.
    
    Args:
        keypoints: (T, 17, 3) array with [y, x, conf]
        conf_threshold: minimum confidence threshold
        
    Returns:
        (T,) array of symmetry values, NaN where confidence is low
    """
    T = keypoints.shape[0]
    symmetry = np.full(T, np.nan)
    
    for t in range(T):
        left_shoulder = keypoints[t, LEFT_SHOULDER]
        right_shoulder = keypoints[t, RIGHT_SHOULDER]
        
        # Check confidence
        if left_shoulder[2] < conf_threshold or right_shoulder[2] < conf_threshold:
            continue
        
        # Compute absolute difference in y-coordinates
        symmetry[t] = abs(left_shoulder[0] - right_shoulder[0])
    
    return symmetry


def compute_knee_angle(
    keypoints: np.ndarray,
    conf_threshold: float = CONF_THRESHOLD
) -> np.ndarray:
    """
    Compute maximum knee angle (larger of left and right knee angles).

    Knee angle is computed using hip-knee-ankle vectors.

    Args:
        keypoints: (T, 17, 3) array with [y, x, conf]
        conf_threshold: minimum confidence threshold

    Returns:
        (T,) array of knee angles in degrees, NaN where confidence is low
    """
    T = keypoints.shape[0]
    angles = np.full(T, np.nan)

    for t in range(T):
        knee_angles = []

        # Compute left knee angle
        left_hip = keypoints[t, LEFT_HIP]
        left_knee = keypoints[t, LEFT_KNEE]
        left_ankle = keypoints[t, LEFT_ANKLE]

        if (left_hip[2] >= conf_threshold and left_knee[2] >= conf_threshold and
            left_ankle[2] >= conf_threshold):
            # Vectors from knee to hip and knee to ankle
            v1 = np.array([left_hip[0] - left_knee[0], left_hip[1] - left_knee[1]])
            v2 = np.array([left_ankle[0] - left_knee[0], left_ankle[1] - left_knee[1]])

            # Compute angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            knee_angles.append(angle)

        # Compute right knee angle
        right_hip = keypoints[t, RIGHT_HIP]
        right_knee = keypoints[t, RIGHT_KNEE]
        right_ankle = keypoints[t, RIGHT_ANKLE]

        if (right_hip[2] >= conf_threshold and right_knee[2] >= conf_threshold and
            right_ankle[2] >= conf_threshold):
            # Vectors from knee to hip and knee to ankle
            v1 = np.array([right_hip[0] - right_knee[0], right_hip[1] - right_knee[1]])
            v2 = np.array([right_ankle[0] - right_knee[0], right_ankle[1] - right_knee[1]])

            # Compute angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            knee_angles.append(angle)

        # Use maximum knee angle
        if knee_angles:
            angles[t] = max(knee_angles)

    return angles


def smooth_features(features: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to features.

    Args:
        features: (T,) array of feature values (may contain NaN)
        window_length: length of smoothing window (must be odd)
        polyorder: polynomial order for smoothing

    Returns:
        (T,) array of smoothed features
    """
    # Handle NaN values
    valid_mask = ~np.isnan(features)

    if np.sum(valid_mask) < window_length:
        # Not enough valid data for smoothing
        return features

    # Create a copy
    smoothed = features.copy()

    # Find continuous segments of valid data
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < window_length:
        return features

    # Apply smoothing only to valid segments
    try:
        # Simple approach: interpolate NaN, smooth, then restore NaN
        if np.any(~valid_mask):
            # Linear interpolation for NaN values
            from scipy.interpolate import interp1d
            if len(valid_indices) >= 2:
                f = interp1d(valid_indices, features[valid_indices],
                           kind='linear', fill_value='extrapolate')
                features_interp = f(np.arange(len(features)))
            else:
                features_interp = features.copy()
        else:
            features_interp = features

        # Apply Savitzky-Golay filter
        if len(features_interp) >= window_length:
            smoothed_all = savgol_filter(features_interp, window_length, polyorder)
            # Restore NaN where original data was NaN
            smoothed_all[~valid_mask] = np.nan
            return smoothed_all
        else:
            return features
    except:
        # If smoothing fails, return original
        return features


def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize features to [0, 1] using min-max normalization.

    Args:
        features: (T,) array of feature values (may contain NaN)

    Returns:
        Tuple of (normalized_features, min_val, max_val)
    """
    valid_mask = ~np.isnan(features)

    if np.sum(valid_mask) == 0:
        return features, 0.0, 1.0

    min_val = np.nanmin(features)
    max_val = np.nanmax(features)

    if max_val - min_val < 1e-8:
        # Constant feature
        normalized = np.where(valid_mask, 0.5, np.nan)
        return normalized, min_val, max_val

    normalized = (features - min_val) / (max_val - min_val)

    return normalized, min_val, max_val


def extract_features_from_keypoints(
    keypoints: np.ndarray,
    fps: int = 30,
    conf_threshold: float = CONF_THRESHOLD,
    apply_smoothing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all engineered features from keypoints.

    Args:
        keypoints: (T, 17, 3) array with [y, x, conf]
        fps: frames per second
        conf_threshold: minimum confidence threshold
        apply_smoothing: whether to apply Savitzky-Golay smoothing

    Returns:
        Tuple of (features_raw, features_normalized)
        - features_raw: (T, 6) array of raw features
        - features_normalized: (T, 6) array of normalized features [0, 1]

        Feature order: [torso_angle, hip_height, vertical_velocity,
                       motion_magnitude, shoulder_symmetry, knee_angle]
    """
    T = keypoints.shape[0]

    # Extract raw features
    torso_angle = compute_torso_angle(keypoints, conf_threshold)
    hip_height = compute_hip_height(keypoints, conf_threshold)
    vertical_velocity = compute_vertical_velocity(hip_height, fps)
    motion_magnitude = compute_motion_magnitude(keypoints, conf_threshold)
    shoulder_symmetry = compute_shoulder_symmetry(keypoints, conf_threshold)
    knee_angle = compute_knee_angle(keypoints, conf_threshold)

    # Apply smoothing if requested
    if apply_smoothing:
        torso_angle = smooth_features(torso_angle)
        hip_height = smooth_features(hip_height)
        vertical_velocity = smooth_features(vertical_velocity)
        motion_magnitude = smooth_features(motion_magnitude)
        shoulder_symmetry = smooth_features(shoulder_symmetry)
        knee_angle = smooth_features(knee_angle)

    # Stack raw features
    features_raw = np.stack([
        torso_angle,
        hip_height,
        vertical_velocity,
        motion_magnitude,
        shoulder_symmetry,
        knee_angle
    ], axis=1)  # Shape: (T, 6)

    # Normalize each feature
    features_normalized = np.zeros_like(features_raw)
    for i in range(6):
        features_normalized[:, i], _, _ = normalize_features(features_raw[:, i])

    return features_raw, features_normalized


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Extract features and create windows from pose keypoints'
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
        help='Output directory for windowed .npz files'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='Window stride (default: 10)'
    )
    parser.add_argument(
        '--length',
        type=int,
        default=60,
        help='Window length in frames (default: 60)'
    )
    parser.add_argument(
        '--min-visible',
        type=float,
        default=0.7,
        help='Minimum visible ratio (1 - max_missing_ratio) (default: 0.7)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='urfd,le2i,all',
        help='Datasets to process: urfd, le2i, or all (default: urfd,le2i,all)'
    )
    parser.add_argument(
        '--save-raw',
        action='store_true',
        help='Save raw per-frame features to data/interim/features_per_frame/'
    )
    parser.add_argument(
        '--drop-threshold',
        type=float,
        default=0.3,
        help='Max missing ratio per window (default: 0.3)'
    )
    parser.add_argument(
        '--le2i-min-fall-frames',
        type=int,
        default=6,
        help='Minimum fall frames for Le2i window label (default: 6)'
    )

    args = parser.parse_args()

    # Import windowing module
    from .windowing import (
        create_windows,
        save_windows_to_npz,
        print_window_statistics
    )

    # Setup paths
    source_dir = Path(args.source)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_raw:
        raw_features_dir = Path('data/interim/features_per_frame')
        raw_features_dir.mkdir(parents=True, exist_ok=True)

    # Convert min_visible to max_missing_ratio
    max_missing_ratio = 1.0 - args.min_visible

    print("="*70)
    print("Feature Engineering & Windowing")
    print("="*70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Window: length={args.length}, stride={args.stride}")
    print(f"Quality: max_missing_ratio={max_missing_ratio:.2f}")
    print(f"Le2i: min_fall_frames={args.le2i_min_fall_frames}")
    print("="*70)
    print()

    # Get all keypoint files
    keypoint_files = sorted(source_dir.glob("*.npz"))

    if not keypoint_files:
        print(f"ERROR: No .npz files found in {source_dir}")
        sys.exit(1)

    print(f"Found {len(keypoint_files)} keypoint files")
    print()

    # Process files by dataset
    urfd_windows = []
    urfd_labels = []
    urfd_metadata = []

    le2i_windows = []
    le2i_labels = []
    le2i_metadata = []

    stats = {
        'total_files': 0,
        'urfd_files': 0,
        'le2i_files': 0,
        'total_windows': 0,
        'kept_windows': 0,
        'dropped_windows': 0,
        'dropped_missing': 0,
        'dropped_short': 0
    }

    # Process each file
    for npz_path in tqdm(keypoint_files, desc="Processing videos"):
        try:
            # Load keypoints
            data = np.load(npz_path)
            keypoints = data['keypoints']
            label = int(data['label'])
            fps = int(data['fps']) if 'fps' in data else 30
            dataset = str(data['dataset']) if 'dataset' in data else 'unknown'
            video_name = str(data['video_name']) if 'video_name' in data else npz_path.stem
            frame_labels = data['frame_labels'] if 'frame_labels' in data else None

            stats['total_files'] += 1

            # Skip if too short
            if keypoints.shape[0] < args.length:
                stats['dropped_short'] += 1
                continue

            # Extract features
            features_raw, features_normalized = extract_features_from_keypoints(
                keypoints, fps=fps, apply_smoothing=True
            )

            # Save raw features if requested
            if args.save_raw:
                raw_path = raw_features_dir / f"{video_name}.npz"
                np.savez_compressed(
                    raw_path,
                    features_raw=features_raw,
                    features_normalized=features_normalized,
                    video_name=video_name,
                    dataset=dataset,
                    label=label
                )

            # Create windows
            windows, labels, metadata = create_windows(
                features_normalized,
                label,
                video_name,
                dataset,
                window_length=args.length,
                stride=args.stride,
                max_missing_ratio=max_missing_ratio,
                frame_labels=frame_labels,
                le2i_min_fall_frames=args.le2i_min_fall_frames
            )

            # Track statistics
            total_possible = (keypoints.shape[0] - args.length) // args.stride + 1
            dropped = total_possible - len(windows)

            stats['total_windows'] += total_possible
            stats['kept_windows'] += len(windows)
            stats['dropped_windows'] += dropped
            stats['dropped_missing'] += dropped  # Assume all dropped due to missing data

            # Add to dataset-specific lists
            if dataset == 'urfd':
                stats['urfd_files'] += 1
                urfd_windows.extend(windows)
                urfd_labels.extend(labels)
                urfd_metadata.extend(metadata)
            elif dataset == 'le2i':
                stats['le2i_files'] += 1
                le2i_windows.extend(windows)
                le2i_labels.extend(labels)
                le2i_metadata.extend(metadata)

        except Exception as e:
            print(f"\nERROR processing {npz_path.name}: {e}")
            continue

    print()
    print("="*70)
    print("Processing Complete")
    print("="*70)
    print(f"Files processed: {stats['total_files']} (URFD: {stats['urfd_files']}, Le2i: {stats['le2i_files']})")
    print(f"Files dropped (too short): {stats['dropped_short']}")
    print(f"Windows: kept={stats['kept_windows']}, dropped={stats['dropped_windows']}")
    print(f"  Dropped reasons: missing={stats['dropped_missing']}, short={stats['dropped_short']}")
    print()

    # Save dataset-specific files
    if 'urfd' in args.datasets.lower() and urfd_windows:
        print_window_statistics(urfd_windows, urfd_labels, urfd_metadata, "URFD")
        save_windows_to_npz(
            urfd_windows,
            urfd_labels,
            urfd_metadata,
            str(output_dir / 'urfd_windows.npz')
        )

    if 'le2i' in args.datasets.lower() and le2i_windows:
        print_window_statistics(le2i_windows, le2i_labels, le2i_metadata, "Le2i")
        save_windows_to_npz(
            le2i_windows,
            le2i_labels,
            le2i_metadata,
            str(output_dir / 'le2i_windows.npz')
        )

    # Save combined file
    if 'all' in args.datasets.lower():
        all_windows = urfd_windows + le2i_windows
        all_labels = urfd_labels + le2i_labels
        all_metadata = urfd_metadata + le2i_metadata

        if all_windows:
            print_window_statistics(all_windows, all_labels, all_metadata, "Combined (All)")
            save_windows_to_npz(
                all_windows,
                all_labels,
                all_metadata,
                str(output_dir / 'all_windows.npz')
            )

    print()
    print("✓ Feature engineering complete!")
    print()


if __name__ == '__main__':
    main()

