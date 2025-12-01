"""
Analyze video content to understand what's actually happening.

This script extracts frames and shows keypoint detection quality
to understand if there's a real fall in the video.
"""

import sys
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.pose.yolo_loader import load_yolo, infer_keypoints_yolo


def extract_raw_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Extract raw keypoints (x, y coordinates only)."""
    xy_coords = keypoints[:, [1, 0]]  # (17, 2) - [x, y]
    features = xy_coords.flatten()
    return features


def analyze_video_content(video_path: str, model_path: str):
    """Analyze video to understand what's happening."""
    
    print("="*80)
    print("VIDEO CONTENT ANALYSIS")
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
    
    print(f"Video: {video_path}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print()
    
    # Process video WITHOUT filters to see all predictions
    window_size = 30
    keypoint_buffer = deque(maxlen=window_size)
    
    frame_idx = 0
    all_data = []
    
    print("Processing all frames (NO FILTERS)...")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints
        keypoints = infer_keypoints_yolo(yolo_model, frame_rgb, 
                                         confidence_threshold=0.3, 
                                         normalize=True)
        
        features = extract_raw_keypoints(keypoints)
        keypoint_buffer.append(features)
        
        valid_keypoints = np.sum(keypoints[:, 2] >= 0.3)
        avg_confidence = keypoints[:, 2].mean()
        
        # Calculate center of mass (approximate body position)
        if valid_keypoints > 0:
            valid_coords = keypoints[keypoints[:, 2] >= 0.3]
            center_y = valid_coords[:, 0].mean()  # y coordinate
            center_x = valid_coords[:, 1].mean()  # x coordinate
        else:
            center_y = 0
            center_x = 0
        
        # Make prediction if we have 30 frames
        probability = None
        if len(keypoint_buffer) == window_size:
            window = np.array(list(keypoint_buffer), dtype=np.float32)
            window_batch = np.expand_dims(window, axis=0)
            probability = lstm_model.predict(window_batch, verbose=0)[0][0]
        
        all_data.append({
            'frame': frame_idx,
            'valid_keypoints': valid_keypoints,
            'avg_confidence': avg_confidence,
            'center_y': center_y,
            'center_x': center_x,
            'probability': probability,
            'keypoints': keypoints.copy()
        })
    
    cap.release()
    
    # Analyze the data
    print("="*80)
    print("FRAME-BY-FRAME ANALYSIS")
    print("="*80)
    print()
    
    # Find segments where person is detected
    segments = []
    in_segment = False
    segment_start = 0
    
    for i, data in enumerate(all_data):
        if data['valid_keypoints'] >= 10 and not in_segment:
            in_segment = True
            segment_start = i
        elif data['valid_keypoints'] < 10 and in_segment:
            in_segment = False
            segments.append((segment_start, i-1))
    
    if in_segment:
        segments.append((segment_start, len(all_data)-1))
    
    print(f"Found {len(segments)} segments where person is visible:")
    print()
    
    for seg_idx, (start, end) in enumerate(segments, 1):
        duration = (end - start + 1) / fps
        print(f"Segment {seg_idx}: Frames {start+1}-{end+1} ({duration:.2f}s)")
        
        # Analyze this segment
        segment_data = all_data[start:end+1]
        
        # Check for vertical movement (potential fall)
        y_positions = [d['center_y'] for d in segment_data if d['center_y'] > 0]
        
        if len(y_positions) > 10:
            y_start = np.mean(y_positions[:5])
            y_end = np.mean(y_positions[-5:])
            y_change = y_end - y_start
            
            print(f"  Vertical movement: {y_change:+.3f} (positive = downward)")
            
            # Check for rapid descent
            max_descent = 0
            for i in range(len(y_positions) - 5):
                descent = y_positions[i+5] - y_positions[i]
                if descent > max_descent:
                    max_descent = descent
            
            print(f"  Max 5-frame descent: {max_descent:.3f}")
            
            # Check predictions in this segment
            probs = [d['probability'] for d in segment_data if d['probability'] is not None]
            if probs:
                max_prob = max(probs)
                avg_prob = np.mean(probs)
                print(f"  Max probability: {max_prob:.4f}")
                print(f"  Avg probability: {avg_prob:.4f}")
                
                # Find when max probability occurred
                for d in segment_data:
                    if d['probability'] == max_prob:
                        print(f"  Max prob at frame: {d['frame']}")
                        break
            
            # Determine if this looks like a fall
            if max_descent > 0.15:  # Significant downward movement
                print(f"  ⚠️  POTENTIAL FALL DETECTED (rapid descent)")
            elif y_change > 0.1:
                print(f"  ⚠️  POTENTIAL FALL DETECTED (overall descent)")
            else:
                print(f"  ✅ Normal activity (no significant descent)")
        
        print()
    
    # Show high probability predictions
    print("="*80)
    print("HIGH PROBABILITY PREDICTIONS")
    print("="*80)
    print()
    
    high_probs = [d for d in all_data if d['probability'] is not None and d['probability'] >= 0.5]
    
    if high_probs:
        print(f"Found {len(high_probs)} frames with probability ≥ 0.5:")
        print()
        print(f"{'Frame':<8} {'Keypoints':<12} {'Confidence':<12} {'Center Y':<12} {'Probability':<12}")
        print("-"*80)
        
        for d in high_probs:
            print(f"{d['frame']:<8} {d['valid_keypoints']}/17      "
                  f"{d['avg_confidence']:>8.3f}    "
                  f"{d['center_y']:>8.3f}      "
                  f"{d['probability']:>8.4f}")
        
        print()
        
        # Check if these are in segments with person detected
        for d in high_probs:
            in_segment = False
            for start, end in segments:
                if start <= d['frame']-1 <= end:
                    in_segment = True
                    break
            
            if not in_segment:
                print(f"⚠️  Frame {d['frame']}: High probability but person NOT consistently detected")
                print(f"   This is likely a FALSE POSITIVE")
            else:
                print(f"✅ Frame {d['frame']}: High probability and person IS detected")
                print(f"   This could be a REAL FALL")
        
        print()
    else:
        print("No frames with probability ≥ 0.5")
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    all_probs = [d['probability'] for d in all_data if d['probability'] is not None]
    
    if all_probs:
        print(f"Total predictions: {len(all_probs)}")
        print(f"Max probability: {max(all_probs):.6f}")
        print(f"Frames with prob ≥ 0.85: {sum(1 for p in all_probs if p >= 0.85)}")
        print(f"Frames with prob ≥ 0.5: {sum(1 for p in all_probs if p >= 0.5)}")
        print()
        
        # Check if there's a real fall
        real_fall_candidates = []
        for d in all_data:
            if d['probability'] is not None and d['probability'] >= 0.85:
                # Check if person is detected
                if d['valid_keypoints'] >= 10:
                    real_fall_candidates.append(d['frame'])
        
        if real_fall_candidates:
            print(f"🚨 POTENTIAL REAL FALL at frames: {real_fall_candidates}")
            print(f"   (High probability + person detected)")
        else:
            print(f"✅ No real falls detected")
            print(f"   (All high probabilities are when person not detected)")
    
    print()
    print("="*80)


def main():
    video_path = 'data/test/2.mp4'
    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5'
    
    analyze_video_content(video_path, model_path)


if __name__ == '__main__':
    main()

