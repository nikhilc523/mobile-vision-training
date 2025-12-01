"""
Phase 4.5 — Physics-Inspired 5-Feature Stream

Computes 5 physics-inspired features per frame:
1. ratio_bbox: width/height of bounding box
2. log_angle: log(1 + |angle_from_vertical|)
3. rotational_energy: 0.5 * I * ω² (approximate moment of inertia and angular velocity)
4. ratio_derivative: d(ratio)/dt
5. generalized_force: double-pendulum approximation from head-neck & neck-hip segments

These features capture fall dynamics from a physics perspective.
"""

import numpy as np
from typing import List, Tuple


class Physics5FeatureExtractor:
    """Extract 5 physics-inspired features from keypoints."""
    
    def __init__(self, fps: int = 30):
        """
        Initialize extractor.
        
        Args:
            fps: Frames per second for temporal derivatives
        """
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Keypoint indices (COCO format)
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16
        
        # History for temporal derivatives
        self.prev_ratio = None
        self.prev_angle = None
        self.prev_angular_velocity = None
    
    def extract_sequence(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """
        Extract features from a sequence of keypoints.
        
        Args:
            keypoints_sequence: (T, 17, 3) array with [y, x, confidence]
        
        Returns:
            (T, 5) array with physics features
        """
        T = len(keypoints_sequence)
        features = np.zeros((T, 5))
        
        # Reset history
        self.prev_ratio = None
        self.prev_angle = None
        self.prev_angular_velocity = None
        
        for t in range(T):
            features[t] = self.extract_frame(keypoints_sequence[t])
        
        return features
    
    def extract_frame(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract 5 physics features from a single frame.
        
        Args:
            keypoints: (17, 3) array with [y, x, confidence]
        
        Returns:
            (5,) array with [ratio_bbox, log_angle, rotational_energy, ratio_derivative, generalized_force]
        """
        # Extract key points
        nose = keypoints[self.NOSE, :2]
        left_shoulder = keypoints[self.LEFT_SHOULDER, :2]
        right_shoulder = keypoints[self.RIGHT_SHOULDER, :2]
        left_hip = keypoints[self.LEFT_HIP, :2]
        right_hip = keypoints[self.RIGHT_HIP, :2]
        left_knee = keypoints[self.LEFT_KNEE, :2]
        right_knee = keypoints[self.RIGHT_KNEE, :2]
        left_ankle = keypoints[self.LEFT_ANKLE, :2]
        right_ankle = keypoints[self.RIGHT_ANKLE, :2]
        
        # Compute centers
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        # Feature 1: Bounding box ratio (width/height)
        ratio_bbox = self._compute_bbox_ratio(keypoints)
        
        # Feature 2: Log angle from vertical
        log_angle = self._compute_log_angle(shoulder_center, hip_center)
        
        # Feature 3: Rotational energy
        rotational_energy = self._compute_rotational_energy(
            nose, shoulder_center, hip_center, left_knee, right_knee
        )
        
        # Feature 4: Ratio derivative
        ratio_derivative = self._compute_ratio_derivative(ratio_bbox)
        
        # Feature 5: Generalized force (double-pendulum approximation)
        generalized_force = self._compute_generalized_force(
            nose, shoulder_center, hip_center
        )
        
        return np.array([ratio_bbox, log_angle, rotational_energy, ratio_derivative, generalized_force])
    
    def _compute_bbox_ratio(self, keypoints: np.ndarray) -> float:
        """Compute bounding box width/height ratio."""
        # Get all valid keypoints (confidence > 0)
        valid_mask = keypoints[:, 2] > 0
        if not np.any(valid_mask):
            return 1.0  # Default ratio
        
        valid_points = keypoints[valid_mask, :2]
        
        # Compute bounding box
        y_min, x_min = valid_points.min(axis=0)
        y_max, x_max = valid_points.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        if height < 1e-6:
            return 1.0
        
        return width / height
    
    def _compute_log_angle(self, shoulder_center: np.ndarray, hip_center: np.ndarray) -> float:
        """Compute log(1 + |angle_from_vertical|) in degrees."""
        # Torso vector (from shoulder to hip)
        torso_vec = hip_center - shoulder_center
        
        # Angle from vertical (y-axis)
        # atan2(x, y) gives angle from vertical
        angle_rad = np.arctan2(np.abs(torso_vec[1]), np.abs(torso_vec[0]))
        angle_deg = np.degrees(angle_rad)
        
        # Log transform to compress large angles
        log_angle = np.log(1.0 + angle_deg)
        
        return log_angle
    
    def _compute_rotational_energy(
        self,
        nose: np.ndarray,
        shoulder_center: np.ndarray,
        hip_center: np.ndarray,
        left_knee: np.ndarray,
        right_knee: np.ndarray
    ) -> float:
        """
        Compute rotational energy: E_rot = 0.5 * I * ω²
        
        Approximate moment of inertia I from segment lengths.
        Approximate angular velocity ω from temporal changes.
        """
        # Compute segment lengths (approximate body dimensions)
        head_neck_length = np.linalg.norm(nose - shoulder_center)
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        knee_center = (left_knee + right_knee) / 2
        leg_length = np.linalg.norm(hip_center - knee_center)
        
        # Approximate moment of inertia (simplified as sum of segment contributions)
        # I ≈ m * L² (assuming unit mass for each segment)
        I = head_neck_length**2 + torso_length**2 + leg_length**2
        
        # Compute angular velocity from torso angle change
        torso_vec = hip_center - shoulder_center
        current_angle = np.arctan2(torso_vec[1], torso_vec[0])
        
        if self.prev_angle is not None:
            # Angular velocity: ω = Δθ / Δt
            omega = (current_angle - self.prev_angle) / self.dt
        else:
            omega = 0.0
        
        self.prev_angle = current_angle
        
        # Rotational energy: E = 0.5 * I * ω²
        rotational_energy = 0.5 * I * omega**2
        
        return rotational_energy
    
    def _compute_ratio_derivative(self, current_ratio: float) -> float:
        """Compute temporal derivative of bbox ratio."""
        if self.prev_ratio is not None:
            ratio_derivative = (current_ratio - self.prev_ratio) / self.dt
        else:
            ratio_derivative = 0.0
        
        self.prev_ratio = current_ratio
        
        return ratio_derivative
    
    def _compute_generalized_force(
        self,
        nose: np.ndarray,
        shoulder_center: np.ndarray,
        hip_center: np.ndarray
    ) -> float:
        """
        Compute generalized force from double-pendulum approximation.
        
        Model body as two segments:
        - Segment 1: head-neck (nose to shoulder)
        - Segment 2: neck-hip (shoulder to hip)
        
        Compute angles θ₁, θ₂ and their derivatives, then approximate generalized force.
        """
        # Segment 1: head-neck
        seg1_vec = nose - shoulder_center
        seg1_length = np.linalg.norm(seg1_vec)
        theta1 = np.arctan2(seg1_vec[1], seg1_vec[0])  # Angle from horizontal
        
        # Segment 2: neck-hip
        seg2_vec = hip_center - shoulder_center
        seg2_length = np.linalg.norm(seg2_vec)
        theta2 = np.arctan2(seg2_vec[1], seg2_vec[0])  # Angle from horizontal
        
        # Compute angular velocities (first derivatives)
        if self.prev_angular_velocity is not None:
            omega1_prev, omega2_prev = self.prev_angular_velocity
            
            # Current angular velocities
            omega1 = (theta1 - self.prev_angle) / self.dt if self.prev_angle is not None else 0.0
            omega2 = (theta2 - self.prev_angle) / self.dt if self.prev_angle is not None else 0.0
            
            # Angular accelerations (second derivatives)
            alpha1 = (omega1 - omega1_prev) / self.dt
            alpha2 = (omega2 - omega2_prev) / self.dt
            
            # Generalized force approximation (simplified double-pendulum dynamics)
            # F ≈ m * L * α (torque-like quantity)
            generalized_force = seg1_length * alpha1 + seg2_length * alpha2
        else:
            omega1 = 0.0
            omega2 = 0.0
            generalized_force = 0.0
        
        # Store for next iteration
        self.prev_angular_velocity = (omega1, omega2)
        
        return generalized_force


def create_physics5_dataset(
    keypoints_dir: str,
    output_path: str,
    window_size: int = 30,
    stride: int = 10,
    fps: int = 30
):
    """
    Create physics5 feature dataset from keypoints.
    
    Args:
        keypoints_dir: Directory with .npz keypoint files
        output_path: Output path for dataset
        window_size: Window size in frames
        stride: Stride for sliding window
        fps: Frames per second
    """
    import os
    from pathlib import Path
    
    print(f"Creating Physics5 dataset from: {keypoints_dir}")
    
    extractor = Physics5FeatureExtractor(fps=fps)
    
    all_windows = []
    all_labels = []
    all_video_ids = []
    
    keypoints_files = sorted(Path(keypoints_dir).glob('*.npz'))
    print(f"Found {len(keypoints_files)} keypoint files")
    
    for i, kp_file in enumerate(keypoints_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{len(keypoints_files)}...")
        
        # Load keypoints
        data = np.load(kp_file)
        keypoints = data['keypoints']  # (T, 17, 3)
        label = data['label']
        video_name = str(kp_file.stem)
        
        # Extract physics features
        features = extractor.extract_sequence(keypoints)  # (T, 5)
        
        # Create sliding windows
        for start_idx in range(0, len(features) - window_size + 1, stride):
            window = features[start_idx:start_idx + window_size]  # (30, 5)
            all_windows.append(window)
            all_labels.append(label)
            all_video_ids.append(video_name)
    
    # Convert to arrays
    X = np.array(all_windows)  # (N, 30, 5)
    y = np.array(all_labels)  # (N,)
    video_ids = np.array(all_video_ids)  # (N,)
    
    print(f"\nDataset created:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Fall windows: {np.sum(y == 1)} ({100*np.sum(y == 1)/len(y):.1f}%)")
    print(f"  Non-fall windows: {np.sum(y == 0)} ({100*np.sum(y == 0)/len(y):.1f}%)")
    
    # Save
    np.savez_compressed(output_path, X=X, y=y, video_ids=video_ids)
    print(f"\n✓ Dataset saved to: {output_path}")


if __name__ == '__main__':
    # Create physics5 dataset from existing keypoints
    create_physics5_dataset(
        keypoints_dir='data/interim/keypoints',
        output_path='data/processed/all_windows_30_physics5.npz',
        window_size=30,
        stride=10,
        fps=30
    )

