"""
Detailed frame-by-frame analysis of fall detection.

This script provides very detailed analysis of each frame to understand
why the model might miss a fall or have low confidence.

Usage:
    python -m ml.export.analyze_fall_detailed <video_path>
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


def analyze_fall_detailed(video_path: str, model_path: str):
    """Detailed frame-by-frame analysis."""
    
    print("="*80)
    print("DETAILED FALL ANALYSIS")
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
    print("[3/3] Processing video frame by frame...")
    print()
    
    window_size = 30
    keypoint_buffer = deque(maxlen=window_size)
    
    frame_idx = 0
    all_data = []
    
    # COCO keypoint names
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
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
        
        # Calculate detailed metrics
        valid_keypoints = np.sum(keypoints[:, 2] >= 0.3)
        avg_confidence = keypoints[:, 2].mean()
        
        # Body position analysis
        nose_y = keypoints[0, 0] if keypoints[0, 2] >= 0.3 else 0
        left_hip_y = keypoints[11, 0] if keypoints[11, 2] >= 0.3 else 0
        right_hip_y = keypoints[12, 0] if keypoints[12, 2] >= 0.3 else 0
        left_shoulder_y = keypoints[5, 0] if keypoints[5, 2] >= 0.3 else 0
        right_shoulder_y = keypoints[6, 0] if keypoints[6, 2] >= 0.3 else 0
        
        # Hip center
        if left_hip_y > 0 and right_hip_y > 0:
            hip_center_y = (left_hip_y + right_hip_y) / 2
        elif left_hip_y > 0:
            hip_center_y = left_hip_y
        elif right_hip_y > 0:
            hip_center_y = right_hip_y
        else:
            hip_center_y = 0
        
        # Shoulder center
        if left_shoulder_y > 0 and right_shoulder_y > 0:
            shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2
        elif left_shoulder_y > 0:
            shoulder_center_y = left_shoulder_y
        elif right_shoulder_y > 0:
            shoulder_center_y = right_shoulder_y
        else:
            shoulder_center_y = 0
        
        # Body orientation (vertical distance between head and hips)
        if nose_y > 0 and hip_center_y > 0:
            body_height = hip_center_y - nose_y  # Positive = upright
        else:
            body_height = 0
        
        # Make prediction
        probability = None
        if len(keypoint_buffer) == window_size:
            window = np.array(list(keypoint_buffer), dtype=np.float32)
            window_batch = np.expand_dims(window, axis=0)
            probability = lstm_model.predict(window_batch, verbose=0)[0][0]
        
        # Store detailed data
        all_data.append({
            'frame': frame_idx,
            'time': frame_idx / fps,
            'valid_kps': valid_keypoints,
            'avg_confidence': avg_confidence,
            'nose_y': nose_y,
            'hip_y': hip_center_y,
            'shoulder_y': shoulder_center_y,
            'body_height': body_height,
            'probability': probability,
            'keypoints': keypoints.copy()
        })
    
    cap.release()
    
    # Find frames with elevated probability
    print("="*80)
    print("FRAMES WITH ELEVATED PROBABILITY (>10%)")
    print("="*80)
    print()
    
    elevated_frames = [d for d in all_data if d['probability'] is not None and d['probability'] >= 0.1]
    
    if elevated_frames:
        for d in elevated_frames:
            print(f"Frame {d['frame']} ({d['time']:.2f}s):")
            print(f"  Probability: {d['probability']:.4f} ({d['probability']*100:.2f}%)")
            print(f"  Valid keypoints: {d['valid_kps']}/17")
            print(f"  Avg confidence: {d['avg_confidence']:.3f}")
            print(f"  Nose Y: {d['nose_y']:.3f}")
            print(f"  Hip Y: {d['hip_y']:.3f}")
            print(f"  Shoulder Y: {d['shoulder_y']:.3f}")
            print(f"  Body height: {d['body_height']:.3f} (positive = upright)")
            
            # Show which keypoints are detected
            kps = d['keypoints']
            detected = [keypoint_names[i] for i in range(17) if kps[i, 2] >= 0.3]
            missing = [keypoint_names[i] for i in range(17) if kps[i, 2] < 0.3]
            print(f"  Detected: {', '.join(detected)}")
            print(f"  Missing: {', '.join(missing)}")
            print()
    else:
        print("No frames with probability >10%")
        print()
    
    # Analyze the window around max probability
    max_prob_frame = max(all_data, key=lambda x: x['probability'] if x['probability'] is not None else 0)
    max_prob = max_prob_frame['probability']
    max_frame_idx = max_prob_frame['frame']
    
    print("="*80)
    print(f"DETAILED ANALYSIS AROUND MAX PROBABILITY")
    print(f"Max probability: {max_prob:.4f} ({max_prob*100:.2f}%) at frame {max_frame_idx} ({max_prob_frame['time']:.2f}s)")
    print("="*80)
    print()
    
    # Show 30-frame window (the window that produced this prediction)
    window_start = max(0, max_frame_idx - 30)
    window_end = max_frame_idx
    
    print(f"Analyzing 30-frame window (frames {window_start}-{window_end}):")
    print()
    
    window_data = all_data[window_start:window_end]
    
    # Calculate statistics for the window
    avg_valid_kps = np.mean([d['valid_kps'] for d in window_data])
    avg_conf = np.mean([d['avg_confidence'] for d in window_data])
    
    hip_positions = [d['hip_y'] for d in window_data if d['hip_y'] > 0]
    if len(hip_positions) > 5:
        hip_start = np.mean(hip_positions[:5])
        hip_end = np.mean(hip_positions[-5:])
        hip_change = hip_end - hip_start
        hip_velocity = hip_change / len(hip_positions)
    else:
        hip_start = 0
        hip_end = 0
        hip_change = 0
        hip_velocity = 0
    
    body_heights = [d['body_height'] for d in window_data if d['body_height'] != 0]
    avg_body_height = np.mean(body_heights) if body_heights else 0
    
    print(f"Window Statistics:")
    print(f"  Average valid keypoints: {avg_valid_kps:.1f}/17")
    print(f"  Average confidence: {avg_conf:.3f}")
    print(f"  Hip position start: {hip_start:.3f}")
    print(f"  Hip position end: {hip_end:.3f}")
    print(f"  Hip change: {hip_change:+.3f} (positive = downward)")
    print(f"  Hip velocity: {hip_velocity:+.4f} per frame")
    print(f"  Average body height: {avg_body_height:.3f} (positive = upright)")
    print()
    
    # Show frame-by-frame for the window
    print("Frame-by-frame breakdown:")
    print()
    for i, d in enumerate(window_data):
        marker = "→" if d['frame'] == max_frame_idx else " "
        print(f"{marker} Frame {d['frame']:3d} ({d['time']:5.2f}s): "
              f"kps={d['valid_kps']:2d}/17, "
              f"conf={d['avg_confidence']:.3f}, "
              f"hip_y={d['hip_y']:.3f}, "
              f"body_h={d['body_height']:+.3f}")
    
    print()
    
    # Analysis
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()
    
    if max_prob >= 0.85:
        print(f"✅ FALL DETECTED (probability {max_prob*100:.2f}% ≥ 85%)")
    elif max_prob >= 0.5:
        print(f"⚠️  UNCERTAIN (probability {max_prob*100:.2f}% between 50-85%)")
        print()
        print("Possible reasons for low confidence:")
    else:
        print(f"✅ NORMAL ACTIVITY (probability {max_prob*100:.2f}% < 50%)")
        print()
        return
    
    print()
    
    # Diagnose why confidence is low
    if avg_valid_kps < 12:
        print(f"  ⚠️  Poor keypoint detection ({avg_valid_kps:.1f}/17 average)")
        print(f"     → Model trained on videos with 15-17 keypoints")
        print(f"     → Missing keypoints reduce confidence")
    
    if avg_conf < 0.4:
        print(f"  ⚠️  Low confidence scores ({avg_conf:.3f} average)")
        print(f"     → MoveNet is uncertain about keypoint positions")
        print(f"     → May be due to lighting, angle, or occlusion")
    
    if abs(hip_velocity) < 0.01:
        print(f"  ⚠️  Very slow movement (velocity {hip_velocity:+.4f} per frame)")
        print(f"     → Model trained on RAPID falls (velocity >0.02)")
        print(f"     → Slow descents have lower confidence")
    
    if avg_body_height > 0.3:
        print(f"  ⚠️  Body appears upright (height {avg_body_height:.3f})")
        print(f"     → Falls typically have body_height near 0 (horizontal)")
        print(f"     → Person may be sitting/bending, not falling")
    
    print()
    print("="*80)


def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        print("Usage: python -m ml.export.analyze_fall_detailed <video_path>")
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
    
    # Run analysis
    analyze_fall_detailed(video_path, model_path)


if __name__ == '__main__':
    main()

