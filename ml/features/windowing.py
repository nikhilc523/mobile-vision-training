#!/usr/bin/env python3
"""
Windowing utilities for creating fixed-length temporal sequences.

This module creates 60-frame windows with stride for LSTM training.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def create_window_label_urfd(label: int) -> int:
    """
    Create window label for URFD dataset.
    
    URFD has scalar labels per video.
    
    Args:
        label: video-level label (0 or 1)
        
    Returns:
        Window label (same as video label)
    """
    return label


def create_window_label_le2i(
    frame_labels: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_fall_frames: int = 6
) -> int:
    """
    Create window label for Le2i dataset.
    
    Le2i has per-frame labels. A window is labeled as fall (1) if
    at least min_fall_frames frames are labeled as fall.
    
    Args:
        frame_labels: (T,) array of per-frame labels
        start_idx: window start index
        end_idx: window end index (exclusive)
        min_fall_frames: minimum number of fall frames to label window as fall
        
    Returns:
        Window label (0 or 1)
    """
    window_labels = frame_labels[start_idx:end_idx]
    fall_count = np.sum(window_labels == 1)
    
    return 1 if fall_count >= min_fall_frames else 0


def compute_missing_ratio(features: np.ndarray) -> float:
    """
    Compute ratio of missing (NaN) values in feature window.
    
    Args:
        features: (T, D) array of features
        
    Returns:
        Ratio of missing values [0, 1]
    """
    total_values = features.size
    missing_values = np.sum(np.isnan(features))
    
    return missing_values / total_values if total_values > 0 else 1.0


def create_windows(
    features: np.ndarray,
    label: int,
    video_name: str,
    dataset: str,
    window_length: int = 60,
    stride: int = 10,
    max_missing_ratio: float = 0.3,
    frame_labels: Optional[np.ndarray] = None,
    le2i_min_fall_frames: int = 6
) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
    """
    Create fixed-length windows from feature sequence.
    
    Args:
        features: (T, D) array of features
        label: video-level label (for URFD)
        video_name: name of the video
        dataset: 'urfd' or 'le2i'
        window_length: length of each window (default: 60)
        stride: stride between windows (default: 10)
        max_missing_ratio: maximum allowed ratio of missing values (default: 0.3)
        frame_labels: (T,) array of per-frame labels (for Le2i)
        le2i_min_fall_frames: minimum fall frames for Le2i window label
        
    Returns:
        Tuple of (windows, labels, metadata)
        - windows: list of (window_length, D) arrays
        - labels: list of window labels
        - metadata: list of dicts with video_name, start_idx, end_idx, dataset
    """
    T, D = features.shape
    
    windows = []
    labels = []
    metadata = []
    
    # Generate windows with stride
    for start_idx in range(0, T - window_length + 1, stride):
        end_idx = start_idx + window_length
        
        # Extract window
        window = features[start_idx:end_idx]
        
        # Check missing ratio
        missing_ratio = compute_missing_ratio(window)
        if missing_ratio > max_missing_ratio:
            continue
        
        # Create label
        if dataset == 'le2i' and frame_labels is not None:
            window_label = create_window_label_le2i(
                frame_labels, start_idx, end_idx, le2i_min_fall_frames
            )
        else:
            window_label = create_window_label_urfd(label)
        
        # Store window
        windows.append(window)
        labels.append(window_label)
        metadata.append({
            'video_name': video_name,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'dataset': dataset,
            'missing_ratio': missing_ratio
        })
    
    return windows, labels, metadata


def save_windows_to_npz(
    windows: List[np.ndarray],
    labels: List[int],
    metadata: List[Dict],
    output_path: str
):
    """
    Save windows to compressed .npz file.
    
    Args:
        windows: list of (window_length, D) arrays
        labels: list of window labels
        metadata: list of metadata dicts
        output_path: path to save .npz file
    """
    if not windows:
        print(f"Warning: No windows to save to {output_path}")
        return
    
    # Stack windows into array
    X = np.stack(windows, axis=0)  # Shape: (N, window_length, D)
    y = np.array(labels, dtype=np.int32)  # Shape: (N,)
    
    # Convert metadata to arrays
    video_names = np.array([m['video_name'] for m in metadata], dtype=object)
    start_indices = np.array([m['start_idx'] for m in metadata], dtype=np.int32)
    end_indices = np.array([m['end_idx'] for m in metadata], dtype=np.int32)
    datasets = np.array([m['dataset'] for m in metadata], dtype=object)
    missing_ratios = np.array([m['missing_ratio'] for m in metadata], dtype=np.float32)
    
    # Save to compressed npz
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        video_names=video_names,
        start_indices=start_indices,
        end_indices=end_indices,
        datasets=datasets,
        missing_ratios=missing_ratios
    )
    
    print(f"✓ Saved {len(windows)} windows to {output_path}")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Class balance: pos={np.sum(y==1)} ({100*np.mean(y):.1f}%), neg={np.sum(y==0)} ({100*np.mean(y==0):.1f}%)")


def load_windows_from_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load windows from .npz file.
    
    Args:
        npz_path: path to .npz file
        
    Returns:
        Tuple of (X, y, metadata)
        - X: (N, window_length, D) array of features
        - y: (N,) array of labels
        - metadata: dict with video_names, start_indices, end_indices, datasets
    """
    data = np.load(npz_path, allow_pickle=True)
    
    X = data['X']
    y = data['y']
    
    metadata = {
        'video_names': data['video_names'],
        'start_indices': data['start_indices'],
        'end_indices': data['end_indices'],
        'datasets': data['datasets'],
        'missing_ratios': data['missing_ratios']
    }
    
    return X, y, metadata


def print_window_statistics(
    windows: List[np.ndarray],
    labels: List[int],
    metadata: List[Dict],
    dataset_name: str
):
    """
    Print statistics about generated windows.
    
    Args:
        windows: list of windows
        labels: list of labels
        metadata: list of metadata dicts
        dataset_name: name of the dataset
    """
    if not windows:
        print(f"\n{dataset_name}: No windows generated")
        return
    
    n_windows = len(windows)
    n_fall = sum(labels)
    n_non_fall = n_windows - n_fall
    
    # Compute average missing ratio
    avg_missing = np.mean([m['missing_ratio'] for m in metadata])
    
    print(f"\n{dataset_name} Windows:")
    print(f"  Total: {n_windows}")
    print(f"  Fall: {n_fall} ({100*n_fall/n_windows:.1f}%)")
    print(f"  Non-fall: {n_non_fall} ({100*n_non_fall/n_windows:.1f}%)")
    print(f"  Avg missing ratio: {avg_missing:.3f}")
    
    # Count unique videos
    unique_videos = len(set(m['video_name'] for m in metadata))
    print(f"  Unique videos: {unique_videos}")

