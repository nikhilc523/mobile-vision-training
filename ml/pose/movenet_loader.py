"""
MoveNet Lightning pose estimation loader and inference utilities.

This module provides functions to load the MoveNet Lightning model from TensorFlow Hub
and perform single-frame pose keypoint extraction.

MoveNet outputs 17 keypoints in COCO format:
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

Each keypoint has (y, x, confidence) normalized to [0, 1].
"""

import sys
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy as np

# TensorFlow imports
try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError:
    print("ERROR: TensorFlow and TensorFlow Hub are required.")
    print("Install with: pip install tensorflow tensorflow-hub")
    sys.exit(1)

# OpenCV for image loading
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is required.")
    print("Install with: pip install opencv-python")
    sys.exit(1)

# Optional: Matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# MoveNet keypoint names (COCO format)
KEYPOINT_NAMES = [
    'nose',
    'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

# Skeleton edges for visualization (pairs of keypoint indices)
KEYPOINT_EDGES = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 6),  # shoulders
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Default MoveNet Lightning model URL
DEFAULT_MOVENET_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"


def load_movenet(model_url: Optional[str] = None) -> Callable:
    """
    Load MoveNet Lightning model from TensorFlow Hub.
    
    Args:
        model_url: URL to the MoveNet model on TensorFlow Hub.
                   Defaults to MoveNet Lightning v4.
    
    Returns:
        A callable inference function that takes a preprocessed frame
        and returns keypoints as a numpy array of shape (17, 3).
    
    Example:
        >>> inference_fn = load_movenet()
        >>> frame = cv2.imread('image.png')
        >>> frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        >>> keypoints = infer_keypoints(inference_fn, frame_rgb)
    """
    if model_url is None:
        model_url = DEFAULT_MOVENET_URL
    
    print(f"Loading MoveNet model from: {model_url}")
    print("This may take a moment on first run (model will be cached)...")
    
    try:
        model = hub.load(model_url)
        movenet = model.signatures['serving_default']
        print("✓ MoveNet model loaded successfully!")
        return movenet
    except Exception as e:
        print(f"ERROR: Failed to load MoveNet model: {e}")
        sys.exit(1)


def preprocess_frame(frame_rgb: np.ndarray) -> tf.Tensor:
    """
    Preprocess a frame for MoveNet inference.
    
    Resizes and pads the frame to 192x192 pixels (MoveNet Lightning input size)
    and converts to int32 tensor format expected by the model.
    
    Args:
        frame_rgb: Input frame as RGB numpy array (H, W, 3) with values in [0, 255].
    
    Returns:
        Preprocessed tensor of shape (1, 192, 192, 3) with dtype int32.
    
    Example:
        >>> frame = cv2.imread('image.png')
        >>> frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        >>> input_tensor = preprocess_frame(frame_rgb)
    """
    # Convert to tensor
    img = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
    
    # Resize with padding to 192x192 (MoveNet Lightning input size)
    img = tf.image.resize_with_pad(img, target_height=192, target_width=192)
    
    # Convert to int32 as expected by MoveNet
    img = tf.cast(img, dtype=tf.int32)
    
    # Add batch dimension
    img = tf.expand_dims(img, axis=0)
    
    return img


def infer_keypoints(
    inference_fn: Callable,
    frame_rgb: np.ndarray,
    confidence_threshold: float = 0.3
) -> np.ndarray:
    """
    Run pose inference on a single frame and extract keypoints.
    
    Args:
        inference_fn: MoveNet inference function from load_movenet().
        frame_rgb: Input frame as RGB numpy array (H, W, 3).
        confidence_threshold: Minimum confidence threshold. Keypoints with
                            confidence below this value will have their
                            coordinates set to 0. Default: 0.3
    
    Returns:
        Numpy array of shape (17, 3) containing [y, x, confidence] for each
        of the 17 keypoints. Coordinates are normalized to [0, 1].
        Low-confidence keypoints have coordinates masked to 0.
    
    Example:
        >>> inference_fn = load_movenet()
        >>> frame = cv2.imread('image.png')
        >>> frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        >>> keypoints = infer_keypoints(inference_fn, frame_rgb)
        >>> print(f"Nose position: {keypoints[0]}")
    """
    # Preprocess frame
    input_tensor = preprocess_frame(frame_rgb)
    
    # Run inference
    outputs = inference_fn(input_tensor)
    
    # Extract keypoints from output
    # MoveNet output format: {'output_0': tensor of shape (1, 1, 17, 3)}
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # Shape: (17, 3)
    
    # Apply confidence threshold masking
    # Set coordinates to 0 for low-confidence keypoints
    mask = keypoints[:, 2] < confidence_threshold
    keypoints[mask, :2] = 0.0
    
    return keypoints


def visualize_keypoints(
    frame_rgb: np.ndarray,
    keypoints: np.ndarray,
    confidence_threshold: float = 0.3,
    show: bool = True,
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Visualize pose keypoints and skeleton on a frame.
    
    Draws keypoints as circles and connects them with lines to form a skeleton.
    Requires matplotlib to be installed.
    
    Args:
        frame_rgb: Input frame as RGB numpy array (H, W, 3).
        keypoints: Keypoints array of shape (17, 3) with [y, x, confidence].
        confidence_threshold: Only draw keypoints with confidence above this value.
        show: If True, display the plot. Default: True.
        save_path: If provided, save the visualization to this path.
    
    Returns:
        Matplotlib figure object if matplotlib is available, None otherwise.
    
    Example:
        >>> inference_fn = load_movenet()
        >>> frame = cv2.imread('image.png')
        >>> frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        >>> keypoints = infer_keypoints(inference_fn, frame_rgb)
        >>> visualize_keypoints(frame_rgb, keypoints, save_path='output.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: Matplotlib not available. Cannot visualize keypoints.")
        print("Install with: pip install matplotlib")
        return None
    
    height, width = frame_rgb.shape[:2]
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(frame_rgb)
    ax.axis('off')
    
    # Draw skeleton edges
    for edge in KEYPOINT_EDGES:
        kp1_idx, kp2_idx = edge
        y1, x1, conf1 = keypoints[kp1_idx]
        y2, x2, conf2 = keypoints[kp2_idx]
        
        # Only draw edge if both keypoints are confident
        if conf1 >= confidence_threshold and conf2 >= confidence_threshold:
            # Convert normalized coordinates to pixel coordinates
            x1_px, y1_px = x1 * width, y1 * height
            x2_px, y2_px = x2 * width, y2 * height
            
            ax.plot([x1_px, x2_px], [y1_px, y2_px], 
                   color='lime', linewidth=2, alpha=0.7)
    
    # Draw keypoints
    for idx, (y, x, conf) in enumerate(keypoints):
        if conf >= confidence_threshold:
            # Convert normalized coordinates to pixel coordinates
            x_px, y_px = x * width, y * height
            
            # Draw keypoint circle
            circle = patches.Circle((x_px, y_px), radius=5, 
                                   color='red', fill=True, alpha=0.8)
            ax.add_patch(circle)
            
            # Optionally add keypoint label
            # ax.text(x_px, y_px - 10, KEYPOINT_NAMES[idx], 
            #        color='white', fontsize=8, ha='center',
            #        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✓ Visualization saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def main():
    """
    CLI interface for testing MoveNet on a single image.
    
    Usage:
        python -m ml.pose.movenet_loader <image_path>
    """
    if len(sys.argv) < 2:
        print("Usage: python -m ml.pose.movenet_loader <image_path>")
        print("\nExample:")
        print("  python -m ml.pose.movenet_loader data/raw/urfd/falls/fall-01-cam0-rgb/fall-01-cam0-rgb-001.png")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"MoveNet Pose Estimation Test")
    print(f"{'='*60}\n")
    
    # Load image
    print(f"Loading image: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: Failed to load image: {image_path}")
        sys.exit(1)
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {frame_rgb.shape[1]}x{frame_rgb.shape[0]} pixels\n")
    
    # Load MoveNet model
    inference_fn = load_movenet()
    print()
    
    # Run inference
    print("Running pose inference...")
    keypoints = infer_keypoints(inference_fn, frame_rgb, confidence_threshold=0.3)
    print("✓ Inference complete!\n")
    
    # Print first 5 keypoints
    print(f"{'='*60}")
    print(f"First 5 Keypoints (out of 17 total):")
    print(f"{'='*60}")
    print(f"{'Index':<6} {'Name':<15} {'Y':<8} {'X':<8} {'Confidence':<12}")
    print(f"{'-'*60}")
    
    for idx in range(min(5, len(keypoints))):
        y, x, conf = keypoints[idx]
        name = KEYPOINT_NAMES[idx]
        print(f"{idx:<6} {name:<15} {y:.4f}  {x:.4f}  {conf:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Summary Statistics:")
    print(f"{'='*60}")
    
    # Count high-confidence keypoints
    high_conf_count = np.sum(keypoints[:, 2] >= 0.3)
    avg_confidence = np.mean(keypoints[:, 2])
    
    print(f"Total keypoints: 17")
    print(f"High-confidence keypoints (≥0.3): {high_conf_count}")
    print(f"Average confidence: {avg_confidence:.4f}")
    print()
    
    # Visualize
    if MATPLOTLIB_AVAILABLE:
        print("Generating visualization...")
        output_path = image_path.parent / f"{image_path.stem}_pose.png"
        visualize_keypoints(frame_rgb, keypoints, 
                          confidence_threshold=0.3,
                          show=False,
                          save_path=str(output_path))
        print()
    
    print(f"{'='*60}")
    print("✓ Test complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

