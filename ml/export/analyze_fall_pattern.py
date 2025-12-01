"""
Analyze the fall pattern at 3.1s to understand why model missed it.
"""

import sys
import cv2
from pathlib import Path
import numpy as np
from collections import deque
import tensorflow as tf

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.pose.yolo_loader import load_yolo, infer_keypoints_yolo


def extract_raw_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Extract raw keypoints (x, y coordinates only)."""
    xy_coords = keypoints[:, [1, 0]]  # (17, 2) - [x, y]
    features = xy_coords.flatten()
    return features


def analyze_fall_pattern(video_path: str, model_path: str):
    """Analyze the fall pattern to understand why model missed it."""
    
    print("="*80)
    print("FALL PATTERN ANALYSIS")
    print("="*80)
    print()
    
    # Load models
    print("Loading models...")
    yolo_model = load_yolo('yolo11n-pose.pt')
    lstm_model = tf.keras.models.load_model(model_path)
    print("✓ Models loaded")
    print()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Focus on the fall window: 3.1s ± 0.5s (frames 153-212)
    fall_start = int(2.6 * fps)  # 153
    fall_end = int(3.6 * fps)    # 212
    
    print(f"Analyzing fall window: frames {fall_start}-{fall_end}")
    print(f"(2.6s - 3.6s, centered at 3.1s)")
    print()
    
    # Process video
    window_size = 30
    keypoint_buffer = deque(maxlen=window_size)
    
    frame_idx = 0
    fall_data = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Skip frames before our window (but keep 30 frames before for buffer)
        if frame_idx < fall_start - 30:
            continue
        if frame_idx > fall_end:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints
        keypoints = infer_keypoints_yolo(yolo_model, frame_rgb, 
                                         confidence_threshold=0.3, 
                                         normalize=True)
        
        features = extract_raw_keypoints(keypoints)
        keypoint_buffer.append(features)
        
        # Calculate metrics
        valid_keypoints = np.sum(keypoints[:, 2] >= 0.3)
        
        # Body parts positions
        nose_y = keypoints[0, 0] if keypoints[0, 2] >= 0.3 else 0
        left_hip_y = keypoints[11, 0] if keypoints[11, 2] >= 0.3 else 0
        right_hip_y = keypoints[12, 0] if keypoints[12, 2] >= 0.3 else 0
        
        # Hip center (approximate body center)
        if left_hip_y > 0 and right_hip_y > 0:
            hip_center_y = (left_hip_y + right_hip_y) / 2
        elif left_hip_y > 0:
            hip_center_y = left_hip_y
        elif right_hip_y > 0:
            hip_center_y = right_hip_y
        else:
            hip_center_y = 0
        
        # Make prediction
        probability = None
        if len(keypoint_buffer) == window_size:
            window = np.array(list(keypoint_buffer), dtype=np.float32)
            window_batch = np.expand_dims(window, axis=0)
            probability = lstm_model.predict(window_batch, verbose=0)[0][0]
        
        if frame_idx >= fall_start:
            fall_data.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'valid_kps': valid_keypoints,
                'nose_y': nose_y,
                'hip_y': hip_center_y,
                'probability': probability,
                'keypoints': keypoints.copy()
            })
    
    cap.release()
    
    # Analyze the pattern
    print("="*80)
    print("KEYPOINT MOVEMENT ANALYSIS")
    print("="*80)
    print()
    
    # Calculate velocities
    for i in range(1, len(fall_data)):
        prev = fall_data[i-1]
        curr = fall_data[i]
        
        if prev['hip_y'] > 0 and curr['hip_y'] > 0:
            velocity = (curr['hip_y'] - prev['hip_y']) * fps  # pixels/second
            fall_data[i]['velocity'] = velocity
        else:
            fall_data[i]['velocity'] = 0
    
    fall_data[0]['velocity'] = 0
    
    # Find peak velocity (fastest descent)
    max_velocity = max(d['velocity'] for d in fall_data)
    max_velocity_frame = [d for d in fall_data if d['velocity'] == max_velocity][0]
    
    print(f"Peak descent velocity: {max_velocity:.3f} (frame {max_velocity_frame['frame']})")
    print()
    
    # Display key frames
    print(f"{'Frame':<8} {'Time(s)':<10} {'Hip Y':<10} {'Velocity':<12} {'Probability':<12} {'Status':<15}")
    print("-"*90)
    
    for d in fall_data[::3]:  # Every 3rd frame
        frame = d['frame']
        time = d['time']
        hip_y = d['hip_y']
        velocity = d.get('velocity', 0)
        prob = d['probability']
        
        if prob is None:
            status = "Buffering"
            prob_str = "---"
        elif prob >= 0.85:
            status = "🚨 FALL"
            prob_str = f"{prob:.4f}"
        elif prob >= 0.1:
            status = "⚠️  Elevated"
            prob_str = f"{prob:.4f}"
        else:
            status = "✅ Normal"
            prob_str = f"{prob:.4f}"
        
        marker = ">>>" if abs(velocity) > 5 else "   "
        
        print(f"{marker} {frame:<5} {time:<10.2f} {hip_y:<10.3f} {velocity:>8.3f}     {prob_str:>8}     {status}")
    
    print()
    print("="*80)
    print("PATTERN COMPARISON")
    print("="*80)
    print()
    
    # Compare with training data expectations
    print("What the model learned from training data:")
    print("  ✅ Rapid change in keypoint positions (sudden movement)")
    print("  ✅ Body orientation change (standing → horizontal)")
    print("  ✅ Temporal pattern over 1 second (30 frames)")
    print()
    
    print("What's happening in this video:")
    
    # Check for rapid movement
    high_velocity_frames = [d for d in fall_data if abs(d.get('velocity', 0)) > 5]
    if high_velocity_frames:
        print(f"  ✅ Rapid movement detected: {len(high_velocity_frames)} frames")
    else:
        print(f"  ❌ No rapid movement: Max velocity = {max_velocity:.3f}")
    
    # Check for position change
    start_hip = np.mean([d['hip_y'] for d in fall_data[:10] if d['hip_y'] > 0])
    end_hip = np.mean([d['hip_y'] for d in fall_data[-10:] if d['hip_y'] > 0])
    hip_change = end_hip - start_hip
    
    if abs(hip_change) > 0.15:
        print(f"  ✅ Significant position change: {hip_change:+.3f}")
    else:
        print(f"  ❌ Small position change: {hip_change:+.3f}")
    
    # Check detection quality
    avg_kps = np.mean([d['valid_kps'] for d in fall_data])
    if avg_kps >= 15:
        print(f"  ✅ Good detection quality: {avg_kps:.1f}/17 keypoints")
    else:
        print(f"  ⚠️  Poor detection quality: {avg_kps:.1f}/17 keypoints")
    
    print()
    print("="*80)
    print("HYPOTHESIS")
    print("="*80)
    print()
    
    if max_velocity < 5:
        print("🎯 SLOW FALL / SITTING DOWN")
        print()
        print("The person is descending SLOWLY (velocity < 5 pixels/frame).")
        print("This looks more like:")
        print("  • Sitting down intentionally")
        print("  • Slow descent to ground")
        print("  • Controlled movement")
        print()
        print("Training data likely contains:")
        print("  • RAPID falls (sudden collapse)")
        print("  • Fast forward/backward falls")
        print("  • Quick loss of balance")
        print()
        print("Model learned to detect RAPID falls, not slow descents!")
    else:
        print("🎯 RAPID FALL")
        print()
        print("The person is descending RAPIDLY.")
        print("Model should detect this, but isn't.")
        print()
        print("Possible reasons:")
        print("  • Different fall direction (training data bias)")
        print("  • Different body orientation")
        print("  • Camera angle difference")
        print("  • Pose estimator difference (MoveNet vs YOLO)")
    
    print()
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()
    
    if max_velocity < 5:
        print("✅ Model is working correctly!")
        print()
        print("This is a SLOW descent, not a dangerous fall.")
        print("The model was trained to detect RAPID falls (medical emergencies).")
        print()
        print("If you want to detect slow descents:")
        print("  1. Add slow fall examples to training data")
        print("  2. Retrain model with diverse fall speeds")
        print("  3. Consider separate 'sitting down' vs 'falling' classifier")
    else:
        print("⚠️  Model missed a rapid fall!")
        print()
        print("This indicates a training data issue:")
        print("  1. Training data may not include this fall type")
        print("  2. MoveNet vs YOLO keypoint differences")
        print("  3. Camera angle differences")
        print()
        print("Solutions:")
        print("  1. Retrain with YOLO keypoints (not MoveNet)")
        print("  2. Add more diverse fall examples")
        print("  3. Data augmentation for different angles")
    
    print()
    print("="*80)


def main():
    video_path = 'data/test/2.mp4'
    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5'
    
    analyze_fall_pattern(video_path, model_path)


if __name__ == '__main__':
    main()

