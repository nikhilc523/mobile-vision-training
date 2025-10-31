"""
Real-time Feature Extraction for Raw Keypoints (34 features)

Phase 3.2+ Optimization:
- Extract raw keypoints (17 × 2 = 34 features)
- No feature engineering - let the model learn
- Simpler and faster than engineered features

IMPORTANT: Must match training data preprocessing!
- Low-confidence keypoints (< 0.3) are set to 0.0
- This matches the behavior in ml/pose/movenet_loader.py:infer_keypoints()
"""

import numpy as np


class RealtimeRawKeypointsExtractor:
    """Extract raw keypoints (x, y coordinates) for real-time inference."""

    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize extractor.

        Args:
            confidence_threshold: Minimum confidence for keypoints (default: 0.3)
        """
        self.num_features = 34  # 17 keypoints × 2 coordinates
        self.confidence_threshold = confidence_threshold

    def extract(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract raw keypoints from MoveNet output.

        Args:
            keypoints: (17, 3) array with [y, x, confidence] from MoveNet

        Returns:
            (34,) array with flattened [x, y] coordinates

        Note:
            Low-confidence keypoints (< threshold) are set to 0.0 to match
            training data preprocessing.
        """
        # Apply confidence threshold masking (match training data!)
        # Set coordinates to 0 for low-confidence keypoints
        keypoints_masked = keypoints.copy()
        mask = keypoints_masked[:, 2] < self.confidence_threshold
        keypoints_masked[mask, :2] = 0.0

        # Extract x, y coordinates (swap y, x to x, y order)
        xy_coords = keypoints_masked[:, [1, 0]]  # (17, 2) - [x, y]

        # Flatten to (34,)
        features = xy_coords.flatten()

        return features
    
    def get_feature_names(self) -> list:
        """Get feature names for logging."""
        names = []
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        for kp_name in keypoint_names:
            names.append(f'{kp_name}_x')
            names.append(f'{kp_name}_y')
        
        return names

