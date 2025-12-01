"""
Enhanced Fall Detection System with MoveNet + BiLSTM

This script implements the complete enhanced fall detection system with:
- Rule 1: High model confidence (sustained falls, person on ground)
- Rule 2: Hip position check (chair falls)
- Rule 3: Body orientation (fast falls)
- Rule 4: Duration tracking (slow falls)
- Rule 5: Combined probability + orientation (uncertain falls)
- Stability Filter: Prevents false positives from erratic keypoint detection

Usage:
    python -m ml.export.enhanced_fall_detection <video_path>
"""

import sys
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_movenet():
    """Load MoveNet Thunder model."""
    print("Loading MoveNet Thunder model...")
    model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    model = hub.load(model_url)
    movenet = model.signatures['serving_default']
    print("✓ MoveNet loaded")
    return movenet


def infer_keypoints_movenet(model, frame_rgb, confidence_threshold=0.3):
    """Extract keypoints using MoveNet."""
    img = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    outputs = model(img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # (17, 3)
    keypoints[keypoints[:, 2] < confidence_threshold] = 0
    return keypoints


def extract_raw_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Extract raw keypoints in model format."""
    xy_coords = keypoints[:, [1, 0]]  # Swap y,x to x,y
    features = xy_coords.flatten()
    return features


def calculate_body_metrics(keypoints: np.ndarray):
    """
    Calculate body position metrics from keypoints.
    
    Returns:
        dict with keys:
            - nose_y: Y position of nose
            - hip_y: Y position of hip center
            - shoulder_y: Y position of shoulder center
            - body_height: Vertical distance from nose to hips (negative = horizontal)
            - valid_keypoints: Number of valid keypoints
            - avg_confidence: Average confidence score
    """
    # COCO keypoint indices
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    
    # Extract positions
    nose_y = keypoints[NOSE, 0] if keypoints[NOSE, 2] >= 0.3 else 0
    left_hip_y = keypoints[LEFT_HIP, 0] if keypoints[LEFT_HIP, 2] >= 0.3 else 0
    right_hip_y = keypoints[RIGHT_HIP, 0] if keypoints[RIGHT_HIP, 2] >= 0.3 else 0
    left_shoulder_y = keypoints[LEFT_SHOULDER, 0] if keypoints[LEFT_SHOULDER, 2] >= 0.3 else 0
    right_shoulder_y = keypoints[RIGHT_SHOULDER, 0] if keypoints[RIGHT_SHOULDER, 2] >= 0.3 else 0
    
    # Hip center
    if left_hip_y > 0 and right_hip_y > 0:
        hip_y = (left_hip_y + right_hip_y) / 2
    elif left_hip_y > 0:
        hip_y = left_hip_y
    elif right_hip_y > 0:
        hip_y = right_hip_y
    else:
        hip_y = 0
    
    # Shoulder center
    if left_shoulder_y > 0 and right_shoulder_y > 0:
        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    elif left_shoulder_y > 0:
        shoulder_y = left_shoulder_y
    elif right_shoulder_y > 0:
        shoulder_y = right_shoulder_y
    else:
        shoulder_y = 0
    
    # Body height (positive = upright, negative = horizontal/inverted)
    if nose_y > 0 and hip_y > 0:
        body_height = hip_y - nose_y
    else:
        body_height = 0
    
    # Quality metrics
    valid_keypoints = np.sum(keypoints[:, 2] >= 0.3)
    avg_confidence = keypoints[:, 2].mean()
    
    return {
        'nose_y': nose_y,
        'hip_y': hip_y,
        'shoulder_y': shoulder_y,
        'body_height': body_height,
        'valid_keypoints': valid_keypoints,
        'avg_confidence': avg_confidence
    }


def calculate_stability(metrics_history):
    """
    Calculate stability metrics from recent history.
    
    Args:
        metrics_history: List of body metrics dicts
    
    Returns:
        tuple: (hip_stability, body_height_stability)
    """
    if len(metrics_history) < 10:
        return 0.0, 0.0
    
    # Extract values
    hip_positions = [m['hip_y'] for m in metrics_history if m['hip_y'] > 0]
    body_heights = [m['body_height'] for m in metrics_history if m['body_height'] != 0]
    
    # Calculate standard deviations
    hip_stability = np.std(hip_positions) if len(hip_positions) > 5 else 0.0
    body_height_stability = np.std(body_heights) if len(body_heights) > 5 else 0.0
    
    return hip_stability, body_height_stability


def calculate_horizontal_duration(metrics_history, threshold=-0.01):
    """
    Calculate how long the body has been horizontal.
    
    Args:
        metrics_history: List of body metrics dicts
        threshold: Body height threshold for "horizontal" (negative value)
    
    Returns:
        float: Duration in seconds (assuming 30 FPS)
    """
    if len(metrics_history) < 5:
        return 0.0
    
    # Count consecutive frames where body is horizontal
    consecutive_count = 0
    for m in reversed(metrics_history):
        if m['body_height'] < threshold and m['body_height'] != 0:
            consecutive_count += 1
        else:
            break
    
    # Convert to seconds (assuming 30 FPS)
    duration = consecutive_count / 30.0
    return duration


def enhanced_fall_detection(probability, body_metrics, metrics_history):
    """
    Enhanced fall detection with multiple rules and stability filtering.

    Args:
        probability: Model output (0-1)
        body_metrics: Current frame body metrics dict
        metrics_history: List of recent body metrics (30 frames)

    Returns:
        tuple: (is_fall: bool, confidence: str, rule_triggered: str)
    """
    # Extract metrics
    body_height = body_metrics['body_height']
    hip_y = body_metrics['hip_y']
    keypoint_quality = body_metrics['valid_keypoints'] / 17.0

    # Calculate stability
    hip_stability, body_height_stability = calculate_stability(metrics_history)

    # Calculate horizontal duration
    horizontal_duration = calculate_horizontal_duration(metrics_history)

    # STABILITY FILTER: Reject if keypoints are too unstable
    # This catches false positives from erratic detection
    # BUT allow if detection is lost (keypoint_quality < 0.3) - person may have fallen out of frame
    if (hip_stability > 0.04 or body_height_stability > 0.04) and keypoint_quality >= 0.3:
        # Keypoints are jumping around - likely detection errors
        # Only trust VERY high probability AND excellent keypoint quality
        if probability >= 0.99 and keypoint_quality >= 0.75:
            return True, "HIGH", "Rule 1 (model very confident despite instability)"
        else:
            return False, "REJECTED", f"Stability Filter (hip_std={hip_stability:.3f}, body_std={body_height_stability:.3f})"

    # Rule 1: High probability (original model - sustained falls, person on ground)
    # Require stable detection for high confidence
    if probability >= 0.85 and keypoint_quality >= 0.5:
        return True, "HIGH", "Rule 1 (model confident)"

    # Rule 2: Hip very low in frame + high probability (chair falls, ground position)
    # Require BOTH low hip AND reasonable probability
    if hip_y > 0.58 and probability >= 0.5 and keypoint_quality >= 0.4:
        return True, "HIGH", "Rule 2 (person on ground - low hip position)"

    # Rule 3: Body very horizontal + good detection + some probability (fast falls)
    # Require horizontal body AND good detection quality AND some model agreement
    if body_height < -0.06 and keypoint_quality >= 0.70 and probability >= 0.01:
        return True, "HIGH", "Rule 3 (very horizontal body)"

    # Rule 4: Body horizontal + sustained duration (slow falls)
    # Require sustained horizontal position with good keypoint quality
    if body_height < -0.02 and horizontal_duration >= 0.8 and keypoint_quality >= 0.7:
        return True, "MEDIUM", f"Rule 4 (sustained horizontal for {horizontal_duration:.2f}s)"

    # Rule 5: Body horizontal + moderate probability (uncertain falls)
    # Require BOTH horizontal AND moderate probability
    if body_height < -0.01 and probability >= 0.50:
        return True, "MEDIUM", "Rule 5 (horizontal + probability)"

    # Rule 6: Body horizontal + low probability (very uncertain falls)
    # Catch falls with horizontal body but low model confidence
    if body_height < -0.01 and probability >= 0.15 and keypoint_quality >= 0.5:
        return True, "LOW", "Rule 6 (horizontal body + low confidence)"

    # Rule 7: Detection lost + elevated probability (person disappeared/fell out of frame)
    # Catch falls where person disappears from view
    if keypoint_quality < 0.3 and probability >= 0.15:
        return True, "LOW", "Rule 7 (detection lost + elevated probability)"

    # Rule 8: No fall
    return False, "NORMAL", "No fall detected"


def test_enhanced_detection(video_path: str, model_path: str):
    """Test enhanced fall detection on a video."""
    
    print("="*80)
    print("ENHANCED FALL DETECTION SYSTEM")
    print("="*80)
    print()
    
    # Load models
    print("[1/3] Loading models...")
    movenet_model = load_movenet()
    lstm_model = tf.keras.models.load_model(model_path)
    print("✓ Models loaded")
    print()
    
    # Open video
    print(f"[2/3] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"✓ Video opened")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print()
    
    # Process video
    print("[3/3] Processing video with enhanced detection...")
    print()
    
    window_size = 30
    keypoint_buffer = deque(maxlen=window_size)
    metrics_history = deque(maxlen=window_size)
    
    frame_idx = 0
    fall_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints
        keypoints = infer_keypoints_movenet(movenet_model, frame_rgb, confidence_threshold=0.3)
        features = extract_raw_keypoints(keypoints)
        keypoint_buffer.append(features)
        
        # Calculate body metrics
        body_metrics = calculate_body_metrics(keypoints)
        metrics_history.append(body_metrics)
        
        # Make prediction
        probability = None
        if len(keypoint_buffer) == window_size:
            window = np.array(list(keypoint_buffer), dtype=np.float32)
            window_batch = np.expand_dims(window, axis=0)
            probability = lstm_model.predict(window_batch, verbose=0)[0][0]
            
            # Enhanced fall detection
            is_fall, confidence, rule = enhanced_fall_detection(
                probability, body_metrics, list(metrics_history)
            )
            
            if is_fall:
                fall_detections.append({
                    'frame': frame_idx,
                    'time': frame_idx / fps,
                    'probability': probability,
                    'confidence': confidence,
                    'rule': rule,
                    'body_metrics': body_metrics.copy()
                })
    
    cap.release()
    
    # Print results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    
    if fall_detections:
        print(f"🚨 FALL DETECTED! ({len(fall_detections)} frames)")
        print()
        
        # Group consecutive detections
        groups = []
        current_group = [fall_detections[0]]
        
        for detection in fall_detections[1:]:
            if detection['frame'] - current_group[-1]['frame'] <= 5:
                current_group.append(detection)
            else:
                groups.append(current_group)
                current_group = [detection]
        groups.append(current_group)
        
        # Print each group
        for i, group in enumerate(groups, 1):
            start_time = group[0]['time']
            end_time = group[-1]['time']
            max_prob_detection = max(group, key=lambda x: x['probability'])
            
            print(f"Fall Event #{i}:")
            print(f"  Time: {start_time:.2f}s - {end_time:.2f}s (duration: {end_time - start_time:.2f}s)")
            print(f"  Frames: {group[0]['frame']} - {group[-1]['frame']} ({len(group)} frames)")
            print(f"  Max probability: {max_prob_detection['probability']:.4f} ({max_prob_detection['probability']*100:.2f}%)")
            print(f"  Confidence: {max_prob_detection['confidence']}")
            print(f"  Rule triggered: {max_prob_detection['rule']}")
            
            m = max_prob_detection['body_metrics']
            print(f"  Body metrics:")
            print(f"    - Valid keypoints: {m['valid_keypoints']}/17")
            print(f"    - Hip Y position: {m['hip_y']:.3f}")
            print(f"    - Body height: {m['body_height']:.3f}")
            print()
    else:
        print("✅ No falls detected in this video")
        print()
    
    print("="*80)


def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        print("Usage: python -m ml.export.enhanced_fall_detection <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5'
    
    # Check if files exist
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found: {video_path}")
        return
    
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found: {model_path}")
        return
    
    # Run enhanced detection
    test_enhanced_detection(video_path, model_path)


if __name__ == '__main__':
    main()

