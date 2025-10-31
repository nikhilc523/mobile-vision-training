"""
YOLO Pose Estimation Loader

This module provides functions to load YOLO pose estimation models
and extract keypoints from images/video frames.

Alternative to MoveNet - uses YOLO11-Pose for potentially better accuracy.
"""

import numpy as np
from typing import Callable, Tuple

def load_yolo(model_name: str = 'yolo11n-pose.pt'):
    """
    Load YOLO pose estimation model.
    
    Args:
        model_name: YOLO model variant to load. Options:
            - 'yolo11n-pose.pt': Nano (fastest, smallest)
            - 'yolo11s-pose.pt': Small
            - 'yolo11m-pose.pt': Medium
            - 'yolo11l-pose.pt': Large
            - 'yolo11x-pose.pt': Extra Large (most accurate)
    
    Returns:
        YOLO model instance
    
    Raises:
        ImportError: If ultralytics is not installed
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics YOLO is not installed. "
            "Install with: pip install ultralytics"
        )
    
    print(f"Loading YOLO model: {model_name}")
    print("This may take a moment on first run (model will be downloaded)...")
    
    # Load model (will auto-download if not cached)
    model = YOLO(model_name, verbose=False)
    
    print(f"✓ YOLO model loaded successfully!")
    return model


def infer_keypoints_yolo(
    model,
    frame_rgb: np.ndarray,
    confidence_threshold: float = 0.3,
    normalize: bool = True
) -> np.ndarray:
    """
    Run pose inference on a single frame using YOLO and extract keypoints.
    
    Args:
        model: YOLO model instance from load_yolo()
        frame_rgb: Input frame as RGB numpy array (H, W, 3) with values in [0, 255]
        confidence_threshold: Minimum confidence threshold for keypoints (default: 0.3)
        normalize: If True, normalize coordinates to [0, 1] range (default: True)
                   If False, return pixel coordinates
    
    Returns:
        Numpy array of shape (17, 3) containing [y, x, confidence] for each
        of the 17 keypoints. Coordinates are normalized to [0, 1] if normalize=True,
        otherwise in pixel coordinates.
        Low-confidence keypoints have coordinates masked to 0.
        
        If no person detected, returns array of zeros.
    
    Note:
        YOLO outputs keypoints in [x, y] order, but we swap to [y, x] to match
        MoveNet format for compatibility with existing pipeline.
    """
    # Run inference
    results = model(frame_rgb, verbose=False)[0]
    
    # Check if any keypoints detected
    if len(results.keypoints.xy) == 0:
        # No person detected - return zeros
        return np.zeros((17, 3), dtype=np.float32)
    
    # Get keypoints for first detected person
    # YOLO outputs: (17, 2) with [x, y] in pixel coordinates
    keypoints_xy = results.keypoints.xy[0].cpu().numpy()  # (17, 2)
    confidences = results.keypoints.conf[0].cpu().numpy()  # (17,)
    
    # Get frame dimensions for normalization
    height, width = frame_rgb.shape[:2]
    
    # Normalize coordinates if requested
    if normalize:
        keypoints_xy[:, 0] /= width   # Normalize x to [0, 1]
        keypoints_xy[:, 1] /= height  # Normalize y to [0, 1]
    
    # Swap x, y to y, x to match MoveNet format
    keypoints_yx = keypoints_xy[:, [1, 0]]  # (17, 2) - [y, x]
    
    # Stack with confidences
    keypoints = np.concatenate([keypoints_yx, confidences[:, None]], axis=1)  # (17, 3)
    
    # Apply confidence threshold masking
    mask = keypoints[:, 2] < confidence_threshold
    keypoints[mask, :2] = 0.0
    
    return keypoints


def compare_yolo_movenet(frame_rgb: np.ndarray) -> dict:
    """
    Compare YOLO and MoveNet pose estimation on the same frame.
    
    Args:
        frame_rgb: Input frame as RGB numpy array
    
    Returns:
        Dictionary with comparison results:
        {
            'yolo_keypoints': (17, 3) array,
            'movenet_keypoints': (17, 3) array,
            'yolo_confidence': float,
            'movenet_confidence': float,
            'yolo_valid_kps': int,
            'movenet_valid_kps': int
        }
    """
    # Load models
    yolo_model = load_yolo('yolo11n-pose.pt')
    
    from ml.pose.movenet_loader import load_movenet, infer_keypoints
    movenet_fn = load_movenet()
    
    # Run inference
    yolo_kps = infer_keypoints_yolo(yolo_model, frame_rgb, normalize=True)
    movenet_kps = infer_keypoints(movenet_fn, frame_rgb)
    
    # Calculate statistics
    yolo_conf = yolo_kps[:, 2].mean()
    movenet_conf = movenet_kps[:, 2].mean()
    yolo_valid = np.sum(yolo_kps[:, 2] >= 0.3)
    movenet_valid = np.sum(movenet_kps[:, 2] >= 0.3)
    
    return {
        'yolo_keypoints': yolo_kps,
        'movenet_keypoints': movenet_kps,
        'yolo_confidence': yolo_conf,
        'movenet_confidence': movenet_conf,
        'yolo_valid_kps': yolo_valid,
        'movenet_valid_kps': movenet_valid
    }


if __name__ == '__main__':
    """Test YOLO pose estimation"""
    import cv2
    
    # Test on sample image
    print("Testing YOLO pose estimation...")
    
    # Load test image
    img = cv2.imread('data/test/finalfall.mp4')
    if img is None:
        # Try video frame
        cap = cv2.VideoCapture('data/test/finalfall.mp4')
        ret, img = cap.read()
        cap.release()
    
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load YOLO
        model = load_yolo('yolo11n-pose.pt')
        
        # Infer keypoints
        keypoints = infer_keypoints_yolo(model, img_rgb, normalize=True)
        
        print(f"\nResults:")
        print(f"  Keypoints shape: {keypoints.shape}")
        print(f"  Mean confidence: {keypoints[:, 2].mean():.3f}")
        print(f"  Valid keypoints: {np.sum(keypoints[:, 2] >= 0.3)}/17")
        print(f"  Coordinate range: x=[{keypoints[:, 1].min():.3f}, {keypoints[:, 1].max():.3f}], y=[{keypoints[:, 0].min():.3f}, {keypoints[:, 0].max():.3f}]")
    else:
        print("No test image found!")

