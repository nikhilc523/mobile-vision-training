"""
Enhanced Feature Engineering with Derived Temporal Features

Adds 4 new features to the original 10:
- Feature 11: Vertical acceleration (Δv_y / Δt)
- Feature 12: Angular velocity (Δθ / Δt) 
- Feature 13: Stillness ratio (low-motion frames / total)
- Feature 14: Pose stability (variance of torso angle)

Total: 14 features for improved fall detection
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

# Import original feature extraction
from ml.features.feature_engineering_full import (
    extract_all_features,
    interpolate_missing_keypoints,
    normalize_features,
    smooth_features,
    create_windows,
    CONF_THRESHOLD
)


def compute_vertical_acceleration(vertical_velocity: np.ndarray, fps: int = 30) -> np.ndarray:
    """
    Feature 11: Vertical acceleration (second derivative of hip height).
    
    Falls show characteristic acceleration patterns:
    - Sudden negative acceleration (downward)
    - High magnitude during impact
    
    Args:
        vertical_velocity: (T,) array of vertical velocities
        fps: frames per second
        
    Returns:
        (T,) array of vertical accelerations
    """
    T = len(vertical_velocity)
    accelerations = np.full(T, np.nan)
    dt = 1.0 / fps
    
    for t in range(1, T):
        if not np.isnan(vertical_velocity[t]) and not np.isnan(vertical_velocity[t-1]):
            accelerations[t] = (vertical_velocity[t] - vertical_velocity[t-1]) / dt
    
    return accelerations


def compute_angular_velocity(torso_angles: np.ndarray, fps: int = 30) -> np.ndarray:
    """
    Feature 12: Angular velocity of torso (Δθ / Δt).
    
    Falls show rapid torso rotation:
    - High angular velocity during fall
    - Sudden changes in orientation
    
    Args:
        torso_angles: (T,) array of torso angles in degrees
        fps: frames per second
        
    Returns:
        (T,) array of angular velocities (degrees/second)
    """
    T = len(torso_angles)
    angular_velocities = np.full(T, np.nan)
    dt = 1.0 / fps
    
    for t in range(1, T):
        if not np.isnan(torso_angles[t]) and not np.isnan(torso_angles[t-1]):
            # Handle angle wrapping (e.g., 359° -> 1°)
            angle_diff = torso_angles[t] - torso_angles[t-1]
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
            
            angular_velocities[t] = angle_diff / dt
    
    return angular_velocities


def compute_stillness_ratio(motion_magnitude: np.ndarray, window_size: int = 15, 
                            threshold: float = 0.01) -> np.ndarray:
    """
    Feature 13: Stillness ratio (proportion of low-motion frames in local window).
    
    Falls show transition from stillness to motion:
    - High stillness before fall
    - Low stillness during fall
    - High stillness after fall (on ground)
    
    Args:
        motion_magnitude: (T,) array of motion magnitudes
        window_size: size of sliding window (frames)
        threshold: motion threshold for "stillness"
        
    Returns:
        (T,) array of stillness ratios [0, 1]
    """
    T = len(motion_magnitude)
    stillness = np.full(T, np.nan)
    half_window = window_size // 2
    
    for t in range(T):
        start = max(0, t - half_window)
        end = min(T, t + half_window + 1)
        
        window_motion = motion_magnitude[start:end]
        valid_mask = ~np.isnan(window_motion)
        
        if np.sum(valid_mask) > 0:
            still_frames = np.sum(window_motion[valid_mask] < threshold)
            total_frames = np.sum(valid_mask)
            stillness[t] = still_frames / total_frames
    
    return stillness


def compute_pose_stability(torso_angles: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Feature 14: Pose stability (variance of torso angle in local window).
    
    Falls show loss of postural stability:
    - Low variance when stable
    - High variance during fall
    - Low variance when on ground
    
    Args:
        torso_angles: (T,) array of torso angles
        window_size: size of sliding window (frames)
        
    Returns:
        (T,) array of pose stability (variance)
    """
    T = len(torso_angles)
    stability = np.full(T, np.nan)
    half_window = window_size // 2
    
    for t in range(T):
        start = max(0, t - half_window)
        end = min(T, t + half_window + 1)
        
        window_angles = torso_angles[start:end]
        valid_mask = ~np.isnan(window_angles)
        
        if np.sum(valid_mask) > 2:  # Need at least 3 points for variance
            stability[t] = np.var(window_angles[valid_mask])
    
    return stability


def extract_enhanced_features(keypoints: np.ndarray, fps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all 14 features (10 original + 4 derived).
    
    Args:
        keypoints: (T, 17, 3) array with [y, x, conf]
        fps: frames per second
        
    Returns:
        raw_features: (T, 14) array of raw features
        smoothed_features: (T, 14) array of smoothed features
    """
    T = keypoints.shape[0]
    
    # Extract original 10 features
    features_10_raw, features_10_smooth = extract_all_features(keypoints, fps)
    
    # Initialize 14-feature array
    features_14 = np.zeros((T, 14))
    features_14[:, :10] = features_10_smooth  # Use smoothed versions of original features
    
    # Extract derived features from original features
    # Feature 11: Vertical acceleration (from feature 2: vertical velocity)
    features_14[:, 10] = compute_vertical_acceleration(features_10_smooth[:, 2], fps)
    
    # Feature 12: Angular velocity (from feature 0: torso angle)
    features_14[:, 11] = compute_angular_velocity(features_10_smooth[:, 0], fps)
    
    # Feature 13: Stillness ratio (from feature 3: motion magnitude)
    features_14[:, 12] = compute_stillness_ratio(features_10_smooth[:, 3])
    
    # Feature 14: Pose stability (from feature 0: torso angle)
    features_14[:, 13] = compute_pose_stability(features_10_smooth[:, 0])
    
    # Normalize all 14 features together
    features_14_norm = normalize_features(features_14)
    
    # Smooth the derived features (original 10 already smoothed)
    features_14_smooth = features_14_norm.copy()
    features_14_smooth[:, 10:] = smooth_features(features_14_norm[:, 10:])
    
    return features_14_norm, features_14_smooth


def process_video_enhanced(npz_path: Path, window_length: int = 90, stride: int = 15,
                           drop_threshold: float = 0.5) -> dict:
    """
    Process a single video with enhanced 14-feature extraction.
    
    Args:
        npz_path: Path to keypoints .npz file
        window_length: Window size in frames (default 90 = 3 seconds @ 30fps)
        stride: Stride between windows (default 15 = 0.5 seconds)
        drop_threshold: Drop windows with >this fraction missing
        
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
        
        # Extract enhanced 14 features
        _, features_smooth = extract_enhanced_features(keypoints, fps)
        
        # Create frame-level labels (simplified - use video label for all frames)
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


def get_feature_names() -> list:
    """Return names of all 14 features."""
    return [
        'torso_angle',           # 0
        'hip_height',            # 1
        'vertical_velocity',     # 2
        'motion_magnitude',      # 3
        'shoulder_symmetry',     # 4
        'knee_angle',            # 5
        'head_hip_distance',     # 6
        'elbow_angle',           # 7
        'body_aspect_ratio',     # 8
        'centroid_velocity',     # 9
        'vertical_acceleration', # 10 (NEW)
        'angular_velocity',      # 11 (NEW)
        'stillness_ratio',       # 12 (NEW)
        'pose_stability'         # 13 (NEW)
    ]


if __name__ == '__main__':
    # Test feature extraction
    print("Enhanced Feature Engineering Module")
    print("=" * 60)
    print(f"Total features: 14 (10 original + 4 derived)")
    print("\nFeature list:")
    for i, name in enumerate(get_feature_names()):
        marker = " (NEW)" if i >= 10 else ""
        print(f"  {i:2d}. {name}{marker}")

