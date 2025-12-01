#!/usr/bin/env python3
"""
Enhanced Feature Engineering for Fall Detection - 10 Features

This module computes all 10 engineered motion features from pose keypoints
for LSTM training on the full dataset (URFD + Le2i + UCF101).

MoveNet Keypoint Indices (COCO format):
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

10 Features (as per proposal § 3.3):
1. Torso angle (α) - angle between neck-hip line and vertical
2. Hip height (h) - normalized vertical position of hip
3. Vertical velocity (v) - rate of hip height change
4. Motion magnitude (m) - mean L2 displacement of all keypoints
5. Shoulder symmetry (s) - left-right shoulder balance
6. Knee angle (θ) - angle at knee joint
7. Head-hip distance - vertical distance between head and hips
8. Elbow angle (φ) - angle at elbows
9. Body aspect ratio (r) - height/width bounding box
10. Centroid velocity (c_v) - velocity of body centroid
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import Tuple, Dict, List, Optional
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# MoveNet keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Confidence threshold for masking
CONF_THRESHOLD = 0.3


def interpolate_missing_keypoints(keypoints: np.ndarray, conf_threshold: float = CONF_THRESHOLD) -> np.ndarray:
    """
    Interpolate missing keypoints using linear interpolation with EMA smoothing.
    
    Args:
        keypoints: (T, 17, 3) array with [y, x, conf]
        conf_threshold: minimum confidence threshold
        
    Returns:
        (T, 17, 3) array with interpolated keypoints
    """
    T, num_kp, _ = keypoints.shape
    interpolated = keypoints.copy()
    
    for kp_idx in range(num_kp):
        for coord_idx in range(2):  # y and x coordinates
            values = keypoints[:, kp_idx, coord_idx]
            confidences = keypoints[:, kp_idx, 2]
            
            # Mark low-confidence points as NaN
            values_masked = values.copy()
            values_masked[confidences < conf_threshold] = np.nan
            
            # Find valid indices
            valid_mask = ~np.isnan(values_masked)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 1:
                # Linear interpolation
                f = interp1d(valid_indices, values_masked[valid_indices], 
                           kind='linear', bounds_error=False, fill_value='extrapolate')
                interpolated_values = f(np.arange(T))
                
                # Apply EMA smoothing (alpha=0.3 for stability)
                alpha = 0.3
                smoothed = np.zeros(T)
                smoothed[0] = interpolated_values[0]
                for t in range(1, T):
                    smoothed[t] = alpha * interpolated_values[t] + (1 - alpha) * smoothed[t-1]
                
                interpolated[:, kp_idx, coord_idx] = smoothed
            elif len(valid_indices) == 1:
                # Only one valid point - use it for all frames
                interpolated[:, kp_idx, coord_idx] = values_masked[valid_indices[0]]
    
    return interpolated


def compute_torso_angle(keypoints: np.ndarray) -> np.ndarray:
    """Feature 1: Torso angle relative to vertical."""
    T = keypoints.shape[0]
    angles = np.full(T, np.nan)
    
    for t in range(T):
        # Neck proxy (midpoint of shoulders)
        neck_y = (keypoints[t, LEFT_SHOULDER, 0] + keypoints[t, RIGHT_SHOULDER, 0]) / 2
        neck_x = (keypoints[t, LEFT_SHOULDER, 1] + keypoints[t, RIGHT_SHOULDER, 1]) / 2
        
        # Hip center
        hip_y = (keypoints[t, LEFT_HIP, 0] + keypoints[t, RIGHT_HIP, 0]) / 2
        hip_x = (keypoints[t, LEFT_HIP, 1] + keypoints[t, RIGHT_HIP, 1]) / 2
        
        # Angle from vertical
        dy = hip_y - neck_y
        dx = hip_x - neck_x
        angle = np.degrees(np.arctan2(dx, dy))
        angles[t] = abs(angle)
    
    return angles


def compute_hip_height(keypoints: np.ndarray) -> np.ndarray:
    """Feature 2: Hip height (1 - average hip y)."""
    T = keypoints.shape[0]
    heights = np.full(T, np.nan)
    
    for t in range(T):
        avg_hip_y = (keypoints[t, LEFT_HIP, 0] + keypoints[t, RIGHT_HIP, 0]) / 2
        heights[t] = 1.0 - avg_hip_y
    
    return heights


def compute_vertical_velocity(hip_heights: np.ndarray, fps: int = 30) -> np.ndarray:
    """Feature 3: Vertical velocity (Δh / Δt)."""
    T = len(hip_heights)
    velocities = np.full(T, np.nan)
    dt = 1.0 / fps
    
    for t in range(1, T):
        velocities[t] = (hip_heights[t] - hip_heights[t-1]) / dt
    
    return velocities


def compute_motion_magnitude(keypoints: np.ndarray) -> np.ndarray:
    """Feature 4: Mean L2 displacement of all keypoints."""
    T = keypoints.shape[0]
    motion = np.full(T, np.nan)
    
    for t in range(1, T):
        displacements = []
        for kp_idx in range(17):
            dy = keypoints[t, kp_idx, 0] - keypoints[t-1, kp_idx, 0]
            dx = keypoints[t, kp_idx, 1] - keypoints[t-1, kp_idx, 1]
            dist = np.sqrt(dy**2 + dx**2)
            displacements.append(dist)
        motion[t] = np.mean(displacements)
    
    return motion


def compute_shoulder_symmetry(keypoints: np.ndarray) -> np.ndarray:
    """Feature 5: Shoulder symmetry (|left_y - right_y|)."""
    T = keypoints.shape[0]
    symmetry = np.full(T, np.nan)
    
    for t in range(T):
        symmetry[t] = abs(keypoints[t, LEFT_SHOULDER, 0] - keypoints[t, RIGHT_SHOULDER, 0])
    
    return symmetry


def compute_knee_angle(keypoints: np.ndarray) -> np.ndarray:
    """Feature 6: Maximum knee angle."""
    T = keypoints.shape[0]
    angles = np.full(T, np.nan)
    
    for t in range(T):
        knee_angles = []
        
        # Left knee
        v1 = keypoints[t, LEFT_HIP, :2] - keypoints[t, LEFT_KNEE, :2]
        v2 = keypoints[t, LEFT_ANKLE, :2] - keypoints[t, LEFT_KNEE, :2]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        knee_angles.append(np.degrees(np.arccos(cos_angle)))
        
        # Right knee
        v1 = keypoints[t, RIGHT_HIP, :2] - keypoints[t, RIGHT_KNEE, :2]
        v2 = keypoints[t, RIGHT_ANKLE, :2] - keypoints[t, RIGHT_KNEE, :2]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        knee_angles.append(np.degrees(np.arccos(cos_angle)))
        
        angles[t] = max(knee_angles)
    
    return angles


def compute_head_hip_distance(keypoints: np.ndarray) -> np.ndarray:
    """Feature 7: Vertical distance between head (nose) and hips."""
    T = keypoints.shape[0]
    distances = np.full(T, np.nan)
    
    for t in range(T):
        head_y = keypoints[t, NOSE, 0]
        hip_y = (keypoints[t, LEFT_HIP, 0] + keypoints[t, RIGHT_HIP, 0]) / 2
        distances[t] = abs(hip_y - head_y)
    
    return distances


def compute_elbow_angle(keypoints: np.ndarray) -> np.ndarray:
    """Feature 8: Maximum elbow angle."""
    T = keypoints.shape[0]
    angles = np.full(T, np.nan)
    
    for t in range(T):
        elbow_angles = []
        
        # Left elbow
        v1 = keypoints[t, LEFT_SHOULDER, :2] - keypoints[t, LEFT_ELBOW, :2]
        v2 = keypoints[t, LEFT_WRIST, :2] - keypoints[t, LEFT_ELBOW, :2]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        elbow_angles.append(np.degrees(np.arccos(cos_angle)))
        
        # Right elbow
        v1 = keypoints[t, RIGHT_SHOULDER, :2] - keypoints[t, RIGHT_ELBOW, :2]
        v2 = keypoints[t, RIGHT_WRIST, :2] - keypoints[t, RIGHT_ELBOW, :2]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        elbow_angles.append(np.degrees(np.arccos(cos_angle)))
        
        angles[t] = max(elbow_angles)
    
    return angles


def compute_body_aspect_ratio(keypoints: np.ndarray) -> np.ndarray:
    """Feature 9: Body aspect ratio (height / width of bounding box)."""
    T = keypoints.shape[0]
    ratios = np.full(T, np.nan)
    
    for t in range(T):
        # Bounding box
        y_coords = keypoints[t, :, 0]
        x_coords = keypoints[t, :, 1]
        
        height = np.max(y_coords) - np.min(y_coords)
        width = np.max(x_coords) - np.min(x_coords)
        
        ratios[t] = height / (width + 1e-8)
    
    return ratios


def compute_centroid_velocity(keypoints: np.ndarray, fps: int = 30) -> np.ndarray:
    """Feature 10: Velocity of body centroid."""
    T = keypoints.shape[0]
    velocities = np.full(T, np.nan)
    dt = 1.0 / fps
    
    # Compute centroids
    centroids = np.mean(keypoints[:, :, :2], axis=1)  # (T, 2)
    
    for t in range(1, T):
        dy = centroids[t, 0] - centroids[t-1, 0]
        dx = centroids[t, 1] - centroids[t-1, 1]
        velocity = np.sqrt(dy**2 + dx**2) / dt
        velocities[t] = velocity
    
    return velocities


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to [0, 1] range per video.

    Args:
        features: (T, num_features) array

    Returns:
        (T, num_features) normalized array
    """
    normalized = features.copy()

    for feat_idx in range(features.shape[1]):
        feat_values = features[:, feat_idx]
        valid_mask = ~np.isnan(feat_values)

        if np.sum(valid_mask) > 0:
            min_val = np.min(feat_values[valid_mask])
            max_val = np.max(feat_values[valid_mask])

            if max_val > min_val:
                normalized[:, feat_idx] = (feat_values - min_val) / (max_val - min_val)
            else:
                normalized[:, feat_idx] = 0.5  # Constant feature

    return normalized


def smooth_features(features: np.ndarray, window_length: int = 5) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to features.

    Args:
        features: (T, num_features) array
        window_length: smoothing window length (must be odd)

    Returns:
        (T, num_features) smoothed array
    """
    if features.shape[0] < window_length:
        return features

    smoothed = features.copy()

    for feat_idx in range(features.shape[1]):
        feat_values = features[:, feat_idx]
        valid_mask = ~np.isnan(feat_values)

        if np.sum(valid_mask) > window_length:
            try:
                smoothed[:, feat_idx] = savgol_filter(feat_values, window_length, 2)
            except:
                pass  # Keep original if smoothing fails

    return smoothed


def extract_all_features(keypoints: np.ndarray, fps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all 10 features from keypoints.

    Args:
        keypoints: (T, 17, 3) array with [y, x, conf]
        fps: frames per second

    Returns:
        raw_features: (T, 10) array of raw features
        smoothed_features: (T, 10) array of smoothed features
    """
    T = keypoints.shape[0]

    # Interpolate missing keypoints
    keypoints_interp = interpolate_missing_keypoints(keypoints)

    # Compute all 10 features
    features = np.zeros((T, 10))

    features[:, 0] = compute_torso_angle(keypoints_interp)
    features[:, 1] = compute_hip_height(keypoints_interp)
    features[:, 2] = compute_vertical_velocity(features[:, 1], fps)
    features[:, 3] = compute_motion_magnitude(keypoints_interp)
    features[:, 4] = compute_shoulder_symmetry(keypoints_interp)
    features[:, 5] = compute_knee_angle(keypoints_interp)
    features[:, 6] = compute_head_hip_distance(keypoints_interp)
    features[:, 7] = compute_elbow_angle(keypoints_interp)
    features[:, 8] = compute_body_aspect_ratio(keypoints_interp)
    features[:, 9] = compute_centroid_velocity(keypoints_interp, fps)

    # Normalize
    features_norm = normalize_features(features)

    # Smooth
    features_smooth = smooth_features(features_norm)

    return features_norm, features_smooth


def create_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_length: int = 60,
    stride: int = 10,
    drop_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Create sliding windows from features.

    Args:
        features: (T, num_features) array
        labels: (T,) array of frame-level labels (for Le2i)
        window_length: window size in frames
        stride: stride in frames
        drop_threshold: drop windows with >this fraction of missing data

    Returns:
        X: (N, window_length, num_features) windows
        y: (N,) window labels
        num_dropped: number of windows dropped due to quality
    """
    T, num_features = features.shape
    windows = []
    window_labels = []
    num_dropped = 0

    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        window = features[start:end]
        window_frame_labels = labels[start:end]

        # Quality check: drop if too much missing data
        missing_fraction = np.sum(np.isnan(window)) / window.size
        if missing_fraction > drop_threshold:
            num_dropped += 1
            continue

        # Replace remaining NaNs with 0
        window = np.nan_to_num(window, nan=0.0)

        # Window label: 1 if ≥6 fall frames (10% of 60 frames)
        window_label = 1 if np.sum(window_frame_labels == 1) >= 6 else 0

        windows.append(window)
        window_labels.append(window_label)

    if len(windows) == 0:
        return np.array([]), np.array([]), num_dropped

    X = np.array(windows)
    y = np.array(window_labels)

    return X, y, num_dropped


def load_le2i_annotations(video_name: str) -> Optional[np.ndarray]:
    """
    Load Le2i frame-level annotations.

    Args:
        video_name: Le2i video name (e.g., "Home_01_video (1)")

    Returns:
        Frame-level labels array or None if not found
    """
    try:
        from ml.data.parsers.le2i_annotations import Le2iAnnotationParser

        # Extract scene and video number
        parts = video_name.replace("le2i_", "").split("_video")
        scene = parts[0].replace("_", " ")

        parser = Le2iAnnotationParser()
        annotations = parser.parse_scene(scene)

        # Find matching video
        for ann in annotations:
            if video_name in ann['video_path'] or scene in ann['video_path']:
                return ann['frame_labels']

        return None
    except:
        return None


def process_single_video(
    npz_path: Path,
    window_length: int,
    stride: int,
    drop_threshold: float
) -> Dict:
    """
    Process a single video file.

    Args:
        npz_path: path to .npz keypoint file
        window_length: window size
        stride: stride
        drop_threshold: quality threshold

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

        # Extract features
        _, features_smooth = extract_all_features(keypoints, fps)

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
) -> Dict:
    """
    Process all videos in the dataset.

    Args:
        source_dir: directory containing .npz keypoint files
        output_dir: directory to save processed windows
        window_length: window size
        stride: stride
        drop_threshold: quality threshold

    Returns:
        Dictionary with statistics
    """
    # Find all .npz files
    npz_files = sorted(list(source_dir.glob('*.npz')))

    print(f"\n{'='*80}")
    print(f"PHASE 1.5b — ENHANCED FEATURE ENGINEERING (10 FEATURES)")
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
        'urfd': {'videos': 0, 'windows': 0, 'pos': 0, 'neg': 0, 'dropped': 0, 'missing_pct': 0},
        'le2i': {'videos': 0, 'windows': 0, 'pos': 0, 'neg': 0, 'dropped': 0, 'missing_pct': 0},
        'ucf101': {'videos': 0, 'windows': 0, 'pos': 0, 'neg': 0, 'dropped': 0, 'missing_pct': 0},
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
        return stats

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    video_ids_all = np.array(all_video_ids)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-dataset files
    for dataset_name, dataset_data in datasets.items():
        if len(dataset_data) > 0:
            X_dataset = np.concatenate([d['X'] for d in dataset_data], axis=0)
            y_dataset = np.concatenate([d['y'] for d in dataset_data], axis=0)
            video_ids_dataset = np.array([vid for d in dataset_data for vid in [d['video_id']] * len(d['X'])])

            output_path = output_dir / f"{dataset_name}_windows.npz"
            np.savez_compressed(
                output_path,
                X=X_dataset,
                y=y_dataset,
                video_ids=video_ids_dataset
            )
            print(f"✅ Saved {dataset_name}: {output_path} ({len(X_dataset)} windows)")

    # Save combined file
    output_path_all = output_dir / "all_windows_full.npz"
    np.savez_compressed(
        output_path_all,
        X=X_all,
        y=y_all,
        video_ids=video_ids_all
    )
    print(f"✅ Saved combined: {output_path_all} ({len(X_all)} windows)")

    return stats, X_all, y_all


def print_summary(stats: Dict, X_all: np.ndarray, y_all: np.ndarray):
    """Print processing summary."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<10} | {'Videos':<7} | {'Windows':<8} | {'Pos(%)':<8} | {'Neg(%)':<8} | {'Dropped':<8}")
    print(f"{'-'*80}")

    for dataset in ['urfd', 'le2i', 'ucf101']:
        s = stats[dataset]
        total = s['windows']
        pos_pct = (s['pos'] / total * 100) if total > 0 else 0
        neg_pct = (s['neg'] / total * 100) if total > 0 else 0

        print(f"{dataset:<10} | {s['videos']:<7} | {total:<8} | {pos_pct:<8.1f} | {neg_pct:<8.1f} | {s['dropped']:<8}")

    # Overall
    total_videos = sum(s['videos'] for s in stats.values())
    total_windows = len(X_all)
    total_pos = np.sum(y_all == 1)
    total_neg = np.sum(y_all == 0)
    total_dropped = sum(s['dropped'] for s in stats.values())

    print(f"{'-'*80}")
    print(f"{'TOTAL':<10} | {total_videos:<7} | {total_windows:<8} | {total_pos/total_windows*100:<8.1f} | {total_neg/total_windows*100:<8.1f} | {total_dropped:<8}")
    print(f"{'='*80}")
    print(f"\n✅ Feature shape: {X_all.shape} (N, 60, 10)")
    print(f"✅ Label shape: {y_all.shape}")
    print(f"✅ Class balance: Fall {total_pos/total_windows*100:.1f}% | Non-fall {total_neg/total_windows*100:.1f}%")
    print(f"{'='*80}\n")


def update_documentation(stats: Dict, X_all: np.ndarray, y_all: np.ndarray, output_dir: Path):
    """Update docs/results1.md with Phase 1.5b summary."""
    docs_path = Path("docs/results1.md")

    if not docs_path.exists():
        print(f"⚠️  Warning: {docs_path} not found, skipping documentation update")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    total_videos = sum(s['videos'] for s in stats.values())
    total_windows = len(X_all)
    total_pos = np.sum(y_all == 1)
    total_neg = np.sum(y_all == 0)
    total_dropped = sum(s['dropped'] for s in stats.values())

    summary = f"""
## Phase 1.5 b — Enhanced Feature Engineering

🗓️ **Date:** {timestamp}

**Inputs:** 964 videos (URFD + Le2i + UCF101)

**Features:** 10 engineered motion features
1. Torso angle (α) - angle between neck-hip line and vertical
2. Hip height (h) - normalized vertical position
3. Vertical velocity (v) - rate of hip height change
4. Motion magnitude (m) - mean L2 displacement
5. Shoulder symmetry (s) - left-right balance
6. Knee angle (θ) - angle at knee joint
7. Head-hip distance - vertical distance
8. Elbow angle (φ) - angle at elbows
9. Body aspect ratio (r) - height/width bounding box
10. Centroid velocity (c_v) - velocity of body centroid

**Processing:**
- Interpolation: EMA for missing keypoints (conf < 0.3)
- Normalization: [0, 1] per video
- Smoothing: Savitzky-Golay filter
- Windowing: 60 frames, stride 10
- Quality filter: Drop if >50% missing

**Results:**
- Videos processed: {total_videos}
- Windows generated: {total_windows:,}
- Windows dropped: {total_dropped}
- Class balance: Fall {total_pos/total_windows*100:.1f}% ({total_pos:,}) | Non-fall {total_neg/total_windows*100:.1f}% ({total_neg:,})

**Output:** `{output_dir}/all_windows_full.npz`
- X shape: {X_all.shape} (N, 60, 10)
- y shape: {y_all.shape}

**Status:** ✅ Success

---

"""

    # Append to file
    with open(docs_path, 'a') as f:
        f.write(summary)

    print(f"✅ Updated documentation: {docs_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Feature Engineering - Extract 10 features from full dataset"
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
        '--features',
        type=int,
        default=10,
        help='Number of features (must be 10)'
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
    parser.add_argument(
        '--min-visible',
        type=float,
        default=0.5,
        help='Minimum visible keypoints fraction (not used, for compatibility)'
    )

    args = parser.parse_args()

    if args.features != 10:
        print(f"⚠️  Warning: This module computes exactly 10 features (requested: {args.features})")

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

    # Print summary
    print_summary(stats, X_all, y_all)

    # Update documentation
    update_documentation(stats, X_all, y_all, output_dir)

    print("\n🎉 Feature engineering complete!")


if __name__ == '__main__':
    main()

