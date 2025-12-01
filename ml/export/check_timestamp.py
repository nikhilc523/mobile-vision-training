"""
Check what's happening at specific timestamp (3.1 seconds).
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


def check_timestamp(video_path: str, model_path: str, target_time: float):
    """Check what's happening at specific timestamp."""
    
    print("="*80)
    print(f"CHECKING TIMESTAMP: {target_time}s")
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    target_frame = int(target_time * fps)
    
    print(f"Video: {video_path}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    print()
    print(f"Target time: {target_time}s")
    print(f"Target frame: {target_frame}")
    print()
    
    # Process video
    window_size = 30
    keypoint_buffer = deque(maxlen=window_size)
    
    frame_idx = 0
    
    # Define range around target
    start_frame = max(1, target_frame - 60)  # 1 second before
    end_frame = min(total_frames, target_frame + 60)  # 1 second after
    
    print(f"Analyzing frames {start_frame} to {end_frame}")
    print(f"(±1 second around target)")
    print()
    
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Skip frames outside our range
        if frame_idx < start_frame - 30:  # Need 30 frames before for window
            continue
        if frame_idx > end_frame:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints
        keypoints = infer_keypoints_yolo(yolo_model, frame_rgb, 
                                         confidence_threshold=0.3, 
                                         normalize=True)
        
        features = extract_raw_keypoints(keypoints)
        keypoint_buffer.append(features)
        
        valid_keypoints = np.sum(keypoints[:, 2] >= 0.3)
        
        # Make prediction if we have 30 frames
        probability = None
        if len(keypoint_buffer) == window_size:
            window = np.array(list(keypoint_buffer), dtype=np.float32)
            window_batch = np.expand_dims(window, axis=0)
            probability = lstm_model.predict(window_batch, verbose=0)[0][0]
        
        # Calculate center of mass
        if valid_keypoints > 0:
            valid_coords = keypoints[keypoints[:, 2] >= 0.3]
            center_y = valid_coords[:, 0].mean()
        else:
            center_y = 0
        
        if frame_idx >= start_frame:
            results.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'valid_kps': valid_keypoints,
                'center_y': center_y,
                'probability': probability
            })
    
    cap.release()
    
    # Display results
    print("="*80)
    print("FRAME-BY-FRAME ANALYSIS")
    print("="*80)
    print()
    print(f"{'Frame':<8} {'Time(s)':<10} {'Keypoints':<12} {'Center Y':<12} {'Probability':<12} {'Status':<15}")
    print("-"*90)
    
    for r in results:
        frame = r['frame']
        time = r['time']
        kps = r['valid_kps']
        center_y = r['center_y']
        prob = r['probability']
        
        # Determine status
        if prob is None:
            status = "Buffering"
            prob_str = "---"
        elif prob >= 0.85:
            status = "🚨 FALL"
            prob_str = f"{prob:.4f}"
        elif prob >= 0.5:
            status = "⚠️  WARNING"
            prob_str = f"{prob:.4f}"
        else:
            status = "✅ Normal"
            prob_str = f"{prob:.4f}"
        
        # Highlight target frame
        marker = ">>>" if abs(frame - target_frame) <= 2 else "   "
        
        print(f"{marker} {frame:<5} {time:<10.2f} {kps}/17      "
              f"{center_y:>8.3f}      {prob_str:>8}     {status}")
    
    print()
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()
    
    # Find high probability frames
    high_prob_frames = [r for r in results if r['probability'] is not None and r['probability'] >= 0.5]
    
    if high_prob_frames:
        print(f"Found {len(high_prob_frames)} frames with probability ≥ 0.5:")
        print()
        for r in high_prob_frames:
            print(f"  Frame {r['frame']} ({r['time']:.2f}s): {r['probability']:.4f}")
            print(f"    Keypoints: {r['valid_kps']}/17")
            print(f"    Center Y: {r['center_y']:.3f}")
            
            if r['valid_kps'] < 10:
                print(f"    ⚠️  Person NOT consistently detected - likely FALSE POSITIVE")
            else:
                print(f"    ✅ Person detected - could be REAL FALL")
            print()
    else:
        print("No frames with probability ≥ 0.5 in this range")
        print()
    
    # Check vertical movement
    frames_with_person = [r for r in results if r['valid_kps'] >= 10]
    
    if len(frames_with_person) > 10:
        print("Vertical movement analysis:")
        y_positions = [r['center_y'] for r in frames_with_person]
        
        y_start = np.mean(y_positions[:5])
        y_end = np.mean(y_positions[-5:])
        y_change = y_end - y_start
        
        print(f"  Start Y: {y_start:.3f}")
        print(f"  End Y: {y_end:.3f}")
        print(f"  Change: {y_change:+.3f} (positive = downward)")
        
        if y_change > 0.15:
            print(f"  ⚠️  SIGNIFICANT DOWNWARD MOVEMENT - Likely a fall!")
        elif y_change > 0.05:
            print(f"  ⚠️  Moderate downward movement")
        else:
            print(f"  ✅ No significant vertical movement")
        print()
    
    print("="*80)


def main():
    video_path = 'data/test/2.mp4'
    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5'
    target_time = 3.1  # User said fall at 3.1s
    
    check_timestamp(video_path, model_path, target_time)


if __name__ == '__main__':
    main()

