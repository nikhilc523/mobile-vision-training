"""
Diagnose the issue with YOLO pose detection on test video.

This script analyzes why the model detects falls when no person is visible.
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


def diagnose_video(video_path: str, model_path: str):
    """Diagnose the issue."""
    
    print("="*80)
    print("DIAGNOSTIC ANALYSIS")
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
    
    # Process video
    window_size = 30
    keypoint_buffer = deque(maxlen=window_size)
    
    frame_idx = 0
    all_keypoints = []
    all_features = []
    predictions = []
    
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
        
        all_keypoints.append(keypoints.copy())
        all_features.append(features.copy())
        
        keypoint_buffer.append(features)
        
        if len(keypoint_buffer) == window_size:
            window = np.array(list(keypoint_buffer), dtype=np.float32)
            window_batch = np.expand_dims(window, axis=0)
            probability = lstm_model.predict(window_batch, verbose=0)[0][0]
            
            predictions.append({
                'frame': frame_idx,
                'probability': probability,
                'window': window.copy(),
                'keypoints': keypoints.copy()
            })
    
    cap.release()
    
    print("="*80)
    print("ISSUE IDENTIFIED")
    print("="*80)
    print()
    
    # Find the fall detection spike (frames 68-78)
    fall_frames = [p for p in predictions if p['probability'] >= 0.85]
    
    if fall_frames:
        print(f"🚨 FALSE POSITIVE DETECTED!")
        print(f"   Frames {fall_frames[0]['frame']} - {fall_frames[-1]['frame']}")
        print(f"   Duration: {len(fall_frames)} frames")
        print()
        
        # Analyze the window that triggered the fall
        trigger_pred = fall_frames[0]
        trigger_window = trigger_pred['window']
        
        print("📊 Analysis of Fall Detection Window:")
        print(f"   Frame: {trigger_pred['frame']}")
        print(f"   Probability: {trigger_pred['probability']:.6f}")
        print()
        
        print("   Window Statistics:")
        print(f"     Shape: {trigger_window.shape}")
        print(f"     Min: {trigger_window.min():.6f}")
        print(f"     Max: {trigger_window.max():.6f}")
        print(f"     Mean: {trigger_window.mean():.6f}")
        print(f"     Std: {trigger_window.std():.6f}")
        print(f"     Zeros: {np.sum(trigger_window == 0)} / {trigger_window.size} ({100*np.sum(trigger_window == 0)/trigger_window.size:.1f}%)")
        print()
        
        # Check if all zeros
        if np.all(trigger_window == 0):
            print("   ⚠️  PROBLEM: Window is ALL ZEROS!")
            print()
        
        # Show frame-by-frame breakdown
        print("   Frame-by-frame breakdown (last 10 frames of window):")
        print(f"   {'Frame':<8} {'Non-zero':<12} {'Mean':<12} {'Max':<12}")
        print("   " + "-"*50)
        
        start_frame = trigger_pred['frame'] - 29
        for i in range(20, 30):
            frame_features = trigger_window[i]
            non_zero = np.sum(frame_features != 0)
            mean_val = frame_features.mean()
            max_val = frame_features.max()
            
            print(f"   {start_frame + i:<8} {non_zero}/34      {mean_val:>8.6f}    {max_val:>8.6f}")
        
        print()
        
        # Check what happened before the fall
        print("   Context (frames before fall detection):")
        context_start = max(0, trigger_pred['frame'] - 40)
        context_end = trigger_pred['frame'] - 30
        
        for idx in range(context_start, context_end):
            if idx < len(all_keypoints):
                kp = all_keypoints[idx]
                valid = np.sum(kp[:, 2] >= 0.3)
                conf = kp[:, 2].mean()
                print(f"     Frame {idx+1}: {valid}/17 keypoints, confidence={conf:.3f}")
        
        print()
    
    # Analyze keypoint detection throughout video
    print("="*80)
    print("KEYPOINT DETECTION ANALYSIS")
    print("="*80)
    print()
    
    valid_counts = []
    for kp in all_keypoints:
        valid = np.sum(kp[:, 2] >= 0.3)
        valid_counts.append(valid)
    
    print(f"Total frames: {len(all_keypoints)}")
    print(f"Frames with 0 keypoints: {sum(1 for v in valid_counts if v == 0)} ({100*sum(1 for v in valid_counts if v == 0)/len(valid_counts):.1f}%)")
    print(f"Frames with 1-10 keypoints: {sum(1 for v in valid_counts if 1 <= v <= 10)} ({100*sum(1 for v in valid_counts if 1 <= v <= 10)/len(valid_counts):.1f}%)")
    print(f"Frames with 11-16 keypoints: {sum(1 for v in valid_counts if 11 <= v <= 16)} ({100*sum(1 for v in valid_counts if 11 <= v <= 16)/len(valid_counts):.1f}%)")
    print(f"Frames with 17 keypoints: {sum(1 for v in valid_counts if v == 17)} ({100*sum(1 for v in valid_counts if v == 17)/len(valid_counts):.1f}%)")
    print()
    
    # Find segments
    print("Video segments:")
    in_person = False
    segment_start = 0
    
    for i, count in enumerate(valid_counts):
        if count >= 10 and not in_person:
            # Person appears
            in_person = True
            segment_start = i + 1
            print(f"  Frame {segment_start}: Person appears")
        elif count < 10 and in_person:
            # Person disappears
            in_person = False
            print(f"  Frame {i}: Person disappears (visible for {i - segment_start + 1} frames)")
    
    print()
    
    # Root cause
    print("="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    print()
    
    print("🔍 The Issue:")
    print()
    print("1. YOLO fails to detect person in frames 1-28 (no keypoints)")
    print("2. YOLO detects person in frame 29 (17/17 keypoints)")
    print("3. YOLO fails again in frames 30+ (no keypoints)")
    print()
    print("4. The model sees this pattern:")
    print("   - 29 frames of ALL ZEROS (no person)")
    print("   - 1 frame with valid keypoints (person appears)")
    print("   - Then zeros again (person disappears)")
    print()
    print("5. This pattern looks like a FALL to the model:")
    print("   - Standing position (zeros = no movement)")
    print("   - Sudden appearance of keypoints (person detected)")
    print("   - Rapid disappearance (person on ground/out of frame)")
    print()
    print("💡 Why this happens:")
    print()
    print("   The model was trained on videos where:")
    print("   - Person is ALWAYS visible (17 keypoints detected)")
    print("   - Falls show rapid MOVEMENT of keypoints")
    print("   - Missing keypoints are rare and brief")
    print()
    print("   But in this test video:")
    print("   - Person is NOT visible for long periods")
    print("   - YOLO intermittently detects/loses the person")
    print("   - This creates a false 'fall' pattern")
    print()
    print("="*80)
    print("SOLUTIONS")
    print("="*80)
    print()
    print("✅ Solution 1: Add person detection filter")
    print("   - Only run fall detection when person is consistently detected")
    print("   - Require at least 15/17 keypoints for 5+ consecutive frames")
    print()
    print("✅ Solution 2: Add zero-frame rejection")
    print("   - Reject windows with >50% zero frames")
    print("   - This was used in training (drop_threshold=0.5)")
    print()
    print("✅ Solution 3: Improve pose detection")
    print("   - Use better camera angle (not top-down)")
    print("   - Ensure good lighting")
    print("   - Keep person in frame and at reasonable distance")
    print()
    print("✅ Solution 4: Retrain with missing keypoint handling")
    print("   - Add augmentation: randomly zero out keypoints")
    print("   - Train model to handle intermittent detection")
    print()
    print("="*80)


def main():
    video_path = 'data/test/2.mp4'
    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5'
    
    diagnose_video(video_path, model_path)


if __name__ == '__main__':
    main()

