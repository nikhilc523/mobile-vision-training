"""
Convert 90-frame enhanced dataset to 60-frame dataset for Phase 2.3a

This script takes the existing all_windows_enhanced.npz (90 frames)
and converts it to 60 frames by taking the middle 60 frames of each window.

This preserves the most critical temporal information while reducing
sequence length for faster training and potentially better generalization.

Input: data/processed/all_windows_enhanced.npz (N, 90, 14)
Output: data/processed/all_windows_60frame.npz (N, 60, 14)
"""

import numpy as np
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description='Convert 90-frame dataset to 60-frame')
    parser.add_argument('--input', type=str, default='data/processed/all_windows_enhanced.npz',
                        help='Input 90-frame dataset')
    parser.add_argument('--output', type=str, default='data/processed/all_windows_60frame.npz',
                        help='Output 60-frame dataset')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Loading {input_path}...")
    data = np.load(input_path, allow_pickle=True)
    
    X = data['X']  # (N, 90, 14)
    y = data['y']  # (N,)
    video_ids = data['video_ids']  # (N,)
    
    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")
    print(f"Original video_ids shape: {video_ids.shape}")
    
    # Take middle 60 frames from each 90-frame window
    # Start at frame 15, end at frame 75 (middle 60 frames)
    start_idx = 15
    end_idx = 75
    
    X_60 = X[:, start_idx:end_idx, :]  # (N, 60, 14)
    
    print(f"\nConverted X shape: {X_60.shape}")
    print(f"Fall samples: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"Non-fall samples: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"Unique videos: {len(np.unique(video_ids))}")
    
    # Save to .npz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X_60,
        y=y,
        video_ids=video_ids,
        window_length=60,
        stride=data['stride'] if 'stride' in data else 15,
        num_features=14,
        feature_names=data['feature_names'] if 'feature_names' in data else [
            'nose_x', 'nose_y', 'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y',
            'left_knee_x', 'left_knee_y', 'right_knee_x', 'right_knee_y',
            'hip_velocity', 'knee_velocity', 'hip_accel', 'knee_accel'
        ]
    )
    
    print(f"\n✅ Dataset saved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()

