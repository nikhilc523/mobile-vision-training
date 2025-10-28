"""
Feature Engineering Module for Fall Detection

This module provides feature extraction and windowing utilities for
converting pose keypoints into LSTM-ready training data.
"""

from .feature_engineering import (
    extract_features_from_keypoints,
    compute_torso_angle,
    compute_hip_height,
    compute_vertical_velocity,
    compute_motion_magnitude,
    compute_shoulder_symmetry,
    compute_knee_angle,
)

from .windowing import (
    create_windows,
    create_window_label_urfd,
    create_window_label_le2i,
)

__all__ = [
    'extract_features_from_keypoints',
    'compute_torso_angle',
    'compute_hip_height',
    'compute_vertical_velocity',
    'compute_motion_magnitude',
    'compute_shoulder_symmetry',
    'compute_knee_angle',
    'create_windows',
    'create_window_label_urfd',
    'create_window_label_le2i',
]

