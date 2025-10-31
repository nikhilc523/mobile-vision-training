"""
Real-Time Feature Extraction for Fall Detection Inference

Extracts the same 14 features used in training from a single frame of keypoints.
Designed for online inference with ring buffer support.
"""

import numpy as np
from typing import Optional, Tuple
from collections import deque

# COCO keypoint indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


class RealtimeFeatureExtractor:
    """
    Extracts 14 features from keypoints in real-time with temporal smoothing.
    
    Features (matching training):
    0. torso_angle - Angle of torso relative to vertical
    1. hip_height - Height of hips in frame (1 - avg_hip_y)
    2. vertical_velocity - Change in hip height over time
    3. motion_magnitude - Average displacement of all keypoints
    4. shoulder_symmetry - Absolute difference in shoulder y-coordinates
    5. knee_angle - Maximum knee angle (hip-knee-ankle)
    6. head_hip_distance - Vertical distance between nose and hips
    7. elbow_angle - Maximum elbow angle (shoulder-elbow-wrist)
    8. body_aspect_ratio - Height/width ratio of bounding box
    9. centroid_velocity - Speed of body centroid
    10. vertical_acceleration - Change in vertical velocity
    11. angular_velocity - Change in torso angle
    12. stillness_ratio - Proportion of low-motion frames
    13. pose_stability - Variance of torso angle
    """
    
    def __init__(self, fps: int = 30, window_size: int = 15, conf_threshold: float = 0.3):
        """
        Initialize feature extractor.
        
        Args:
            fps: Frames per second for temporal features
            window_size: Window size for temporal smoothing
            conf_threshold: Minimum confidence threshold for keypoints
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.window_size = window_size
        self.conf_threshold = conf_threshold
        
        # History buffers for temporal features
        self.hip_height_history = deque(maxlen=window_size)
        self.torso_angle_history = deque(maxlen=window_size)
        self.motion_history = deque(maxlen=window_size)
        self.centroid_history = deque(maxlen=window_size)
        self.prev_keypoints = None
        
    def extract_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract 14 features from a single frame of keypoints.
        
        Args:
            keypoints: (17, 3) array with [y, x, confidence]
            
        Returns:
            (14,) array of features
        """
        features = np.zeros(14)
        
        # Feature 0: Torso angle
        features[0] = self._compute_torso_angle(keypoints)
        
        # Feature 1: Hip height
        features[1] = self._compute_hip_height(keypoints)
        
        # Feature 2: Vertical velocity (requires history)
        features[2] = self._compute_vertical_velocity()
        
        # Feature 3: Motion magnitude (requires previous frame)
        features[3] = self._compute_motion_magnitude(keypoints)
        
        # Feature 4: Shoulder symmetry
        features[4] = self._compute_shoulder_symmetry(keypoints)
        
        # Feature 5: Knee angle
        features[5] = self._compute_knee_angle(keypoints)
        
        # Feature 6: Head-hip distance
        features[6] = self._compute_head_hip_distance(keypoints)
        
        # Feature 7: Elbow angle
        features[7] = self._compute_elbow_angle(keypoints)
        
        # Feature 8: Body aspect ratio
        features[8] = self._compute_body_aspect_ratio(keypoints)
        
        # Feature 9: Centroid velocity (requires history)
        features[9] = self._compute_centroid_velocity(keypoints)
        
        # Feature 10: Vertical acceleration (requires history)
        features[10] = self._compute_vertical_acceleration()
        
        # Feature 11: Angular velocity (requires history)
        features[11] = self._compute_angular_velocity()
        
        # Feature 12: Stillness ratio (requires history)
        features[12] = self._compute_stillness_ratio()
        
        # Feature 13: Pose stability (requires history)
        features[13] = self._compute_pose_stability()
        
        # Update history
        self.hip_height_history.append(features[1])
        self.torso_angle_history.append(features[0])
        self.prev_keypoints = keypoints.copy()
        
        # Normalize features to [0, 1] range (simple min-max based on training ranges)
        features = self._normalize_features(features)
        
        return features
    
    def _compute_torso_angle(self, kp: np.ndarray) -> float:
        """Compute torso angle relative to vertical."""
        ls, rs = kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]
        lh, rh = kp[LEFT_HIP], kp[RIGHT_HIP]
        
        if ls[2] < self.conf_threshold or rs[2] < self.conf_threshold:
            return 0.0
        if lh[2] < self.conf_threshold or rh[2] < self.conf_threshold:
            return 0.0
        
        # Neck proxy (midpoint of shoulders)
        neck_y = (ls[0] + rs[0]) / 2
        neck_x = (ls[1] + rs[1]) / 2
        
        # Hip center
        hip_y = (lh[0] + rh[0]) / 2
        hip_x = (lh[1] + rh[1]) / 2
        
        # Angle from vertical
        dy = hip_y - neck_y
        dx = hip_x - neck_x
        angle = np.degrees(np.arctan2(dx, dy))
        
        return abs(angle)
    
    def _compute_hip_height(self, kp: np.ndarray) -> float:
        """Compute hip height (1 - avg_hip_y)."""
        lh, rh = kp[LEFT_HIP], kp[RIGHT_HIP]
        
        if lh[2] < self.conf_threshold or rh[2] < self.conf_threshold:
            return 0.5  # Default to middle
        
        avg_hip_y = (lh[0] + rh[0]) / 2
        return 1.0 - avg_hip_y
    
    def _compute_vertical_velocity(self) -> float:
        """Compute vertical velocity from hip height history."""
        if len(self.hip_height_history) < 2:
            return 0.0
        
        return (self.hip_height_history[-1] - self.hip_height_history[-2]) / self.dt
    
    def _compute_motion_magnitude(self, kp: np.ndarray) -> float:
        """Compute average displacement of all keypoints."""
        if self.prev_keypoints is None:
            return 0.0
        
        displacements = []
        for i in range(17):
            if kp[i, 2] >= self.conf_threshold and self.prev_keypoints[i, 2] >= self.conf_threshold:
                dy = kp[i, 0] - self.prev_keypoints[i, 0]
                dx = kp[i, 1] - self.prev_keypoints[i, 1]
                dist = np.sqrt(dy**2 + dx**2)
                displacements.append(dist)
        
        return np.mean(displacements) if displacements else 0.0
    
    def _compute_shoulder_symmetry(self, kp: np.ndarray) -> float:
        """Compute shoulder symmetry."""
        ls, rs = kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]
        
        if ls[2] < self.conf_threshold or rs[2] < self.conf_threshold:
            return 0.0
        
        return abs(ls[0] - rs[0])
    
    def _compute_knee_angle(self, kp: np.ndarray) -> float:
        """Compute maximum knee angle."""
        angles = []
        
        # Left knee
        lh, lk, la = kp[LEFT_HIP], kp[LEFT_KNEE], kp[LEFT_ANKLE]
        if lh[2] >= self.conf_threshold and lk[2] >= self.conf_threshold and la[2] >= self.conf_threshold:
            v1 = np.array([lh[0] - lk[0], lh[1] - lk[1]])
            v2 = np.array([la[0] - lk[0], la[1] - lk[1]])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_angle)))
        
        # Right knee
        rh, rk, ra = kp[RIGHT_HIP], kp[RIGHT_KNEE], kp[RIGHT_ANKLE]
        if rh[2] >= self.conf_threshold and rk[2] >= self.conf_threshold and ra[2] >= self.conf_threshold:
            v1 = np.array([rh[0] - rk[0], rh[1] - rk[1]])
            v2 = np.array([ra[0] - rk[0], ra[1] - rk[1]])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_angle)))
        
        return max(angles) if angles else 90.0  # Default to 90 degrees
    
    def _compute_head_hip_distance(self, kp: np.ndarray) -> float:
        """Compute vertical distance between nose and hips."""
        nose = kp[NOSE]
        lh, rh = kp[LEFT_HIP], kp[RIGHT_HIP]
        
        if nose[2] < self.conf_threshold:
            return 0.3  # Default
        if lh[2] < self.conf_threshold or rh[2] < self.conf_threshold:
            return 0.3
        
        hip_y = (lh[0] + rh[0]) / 2
        return abs(hip_y - nose[0])
    
    def _compute_elbow_angle(self, kp: np.ndarray) -> float:
        """Compute maximum elbow angle."""
        angles = []
        
        # Left elbow
        ls, le, lw = kp[LEFT_SHOULDER], kp[LEFT_ELBOW], kp[LEFT_WRIST]
        if ls[2] >= self.conf_threshold and le[2] >= self.conf_threshold and lw[2] >= self.conf_threshold:
            v1 = np.array([ls[0] - le[0], ls[1] - le[1]])
            v2 = np.array([lw[0] - le[0], lw[1] - le[1]])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_angle)))
        
        # Right elbow
        rs, re, rw = kp[RIGHT_SHOULDER], kp[RIGHT_ELBOW], kp[RIGHT_WRIST]
        if rs[2] >= self.conf_threshold and re[2] >= self.conf_threshold and rw[2] >= self.conf_threshold:
            v1 = np.array([rs[0] - re[0], rs[1] - re[1]])
            v2 = np.array([rw[0] - re[0], rw[1] - re[1]])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_angle)))
        
        return max(angles) if angles else 90.0
    
    def _compute_body_aspect_ratio(self, kp: np.ndarray) -> float:
        """Compute height/width ratio of bounding box."""
        valid_kp = kp[kp[:, 2] >= self.conf_threshold]
        
        if len(valid_kp) < 3:
            return 1.5  # Default
        
        y_coords = valid_kp[:, 0]
        x_coords = valid_kp[:, 1]
        
        height = np.max(y_coords) - np.min(y_coords)
        width = np.max(x_coords) - np.min(x_coords)
        
        return height / (width + 1e-8)
    
    def _compute_centroid_velocity(self, kp: np.ndarray) -> float:
        """Compute velocity of body centroid."""
        valid_kp = kp[kp[:, 2] >= self.conf_threshold]
        
        if len(valid_kp) < 3:
            return 0.0
        
        centroid_y = np.mean(valid_kp[:, 0])
        centroid_x = np.mean(valid_kp[:, 1])
        
        self.centroid_history.append((centroid_y, centroid_x))
        
        if len(self.centroid_history) < 2:
            return 0.0
        
        prev_y, prev_x = self.centroid_history[-2]
        dy = centroid_y - prev_y
        dx = centroid_x - prev_x
        
        return np.sqrt(dy**2 + dx**2) / self.dt
    
    def _compute_vertical_acceleration(self) -> float:
        """Compute vertical acceleration from velocity history."""
        if len(self.hip_height_history) < 3:
            return 0.0
        
        v1 = (self.hip_height_history[-2] - self.hip_height_history[-3]) / self.dt
        v2 = (self.hip_height_history[-1] - self.hip_height_history[-2]) / self.dt
        
        return (v2 - v1) / self.dt
    
    def _compute_angular_velocity(self) -> float:
        """Compute angular velocity from torso angle history."""
        if len(self.torso_angle_history) < 2:
            return 0.0
        
        angle_diff = self.torso_angle_history[-1] - self.torso_angle_history[-2]
        
        # Handle angle wrapping
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        return angle_diff / self.dt
    
    def _compute_stillness_ratio(self) -> float:
        """Compute proportion of low-motion frames."""
        if len(self.motion_history) < 3:
            return 1.0  # Assume still at start
        
        threshold = 0.01
        still_count = sum(1 for m in self.motion_history if m < threshold)
        
        return still_count / len(self.motion_history)
    
    def _compute_pose_stability(self) -> float:
        """Compute variance of torso angle."""
        if len(self.torso_angle_history) < 3:
            return 0.0
        
        return np.var(list(self.torso_angle_history))
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range based on training statistics.
        
        Approximate ranges from training data:
        0. torso_angle: [0, 90]
        1. hip_height: [0, 1]
        2. vertical_velocity: [-10, 10]
        3. motion_magnitude: [0, 0.5]
        4. shoulder_symmetry: [0, 0.3]
        5. knee_angle: [0, 180]
        6. head_hip_distance: [0, 0.8]
        7. elbow_angle: [0, 180]
        8. body_aspect_ratio: [0.5, 3.0]
        9. centroid_velocity: [0, 10]
        10. vertical_acceleration: [-100, 100]
        11. angular_velocity: [-500, 500]
        12. stillness_ratio: [0, 1]
        13. pose_stability: [0, 100]
        """
        normalized = features.copy()
        
        # Apply normalization
        normalized[0] = np.clip(features[0] / 90.0, 0, 1)
        normalized[1] = np.clip(features[1], 0, 1)
        normalized[2] = np.clip((features[2] + 10) / 20.0, 0, 1)
        normalized[3] = np.clip(features[3] / 0.5, 0, 1)
        normalized[4] = np.clip(features[4] / 0.3, 0, 1)
        normalized[5] = np.clip(features[5] / 180.0, 0, 1)
        normalized[6] = np.clip(features[6] / 0.8, 0, 1)
        normalized[7] = np.clip(features[7] / 180.0, 0, 1)
        normalized[8] = np.clip((features[8] - 0.5) / 2.5, 0, 1)
        normalized[9] = np.clip(features[9] / 10.0, 0, 1)
        normalized[10] = np.clip((features[10] + 100) / 200.0, 0, 1)
        normalized[11] = np.clip((features[11] + 500) / 1000.0, 0, 1)
        normalized[12] = np.clip(features[12], 0, 1)
        normalized[13] = np.clip(features[13] / 100.0, 0, 1)
        
        return normalized

