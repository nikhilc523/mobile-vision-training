"""
Test fall detection using MoveNet (what the model was trained on).

This will help us understand if the model works correctly when using
the SAME pose estimator it was trained on.

Usage:
    python -m ml.export.test_with_movenet
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
    """
    Extract keypoints using MoveNet.
    
    Returns:
        keypoints: (17, 3) array with [y, x, confidence] (MoveNet format)
    """
    # Resize to 256x256 (MoveNet Thunder input)
    img = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    
    # Run inference
    outputs = model(img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # (17, 3)
    
    # Filter by confidence
    keypoints[keypoints[:, 2] < confidence_threshold] = 0
    
    return keypoints


def extract_raw_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract raw keypoints in the format the model was trained on.
    
    MoveNet outputs [y, x, confidence]
    Model expects [x, y] (swapped)
    
    Args:
        keypoints: (17, 3) array with [y, x, confidence]
    
    Returns:
        features: (34,) array with [x, y] for each keypoint
    """
    # Extract x, y coordinates (swap y, x to x, y order)
    xy_coords = keypoints[:, [1, 0]]  # (17, 2) - [x, y]
    features = xy_coords.flatten()  # (34,)
    return features


def analyze_video_second_by_second(video_path: str, model_path: str):
    """Analyze video second by second using MoveNet."""
    
    print("="*80)
    print("TESTING WITH MOVENET (WHAT MODEL WAS TRAINED ON)")
    print("="*80)
    print()
    
    # Load models
    print("[1/4] Loading models...")
    movenet_model = load_movenet()
    lstm_model = tf.keras.models.load_model(model_path)
    print("✓ Models loaded")
    print()
    
    # Open video
    print(f"[2/4] Opening video: {video_path}")
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
    print("[3/4] Processing video second by second...")
    print()
    
    window_size = 30
    keypoint_buffer = deque(maxlen=window_size)
    
    frame_idx = 0
    all_data = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints using MoveNet
        keypoints = infer_keypoints_movenet(movenet_model, frame_rgb, 
                                            confidence_threshold=0.3)
        
        features = extract_raw_keypoints(keypoints)
        keypoint_buffer.append(features)
        
        # Calculate metrics
        valid_keypoints = np.sum(keypoints[:, 2] >= 0.3)
        avg_confidence = keypoints[:, 2].mean()
        
        # Body position (hip center)
        left_hip_y = keypoints[11, 0] if keypoints[11, 2] >= 0.3 else 0
        right_hip_y = keypoints[12, 0] if keypoints[12, 2] >= 0.3 else 0
        
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
        
        all_data.append({
            'frame': frame_idx,
            'time': frame_idx / fps,
            'valid_kps': valid_keypoints,
            'avg_confidence': avg_confidence,
            'hip_y': hip_center_y,
            'probability': probability
        })
    
    cap.release()
    
    # Analyze second by second
    print("="*80)
    print("SECOND-BY-SECOND ANALYSIS")
    print("="*80)
    print()
    
    num_seconds = int(duration) + 1
    
    for second in range(num_seconds):
        start_frame = second * fps
        end_frame = min((second + 1) * fps, total_frames)
        
        # Get data for this second
        second_data = [d for d in all_data if start_frame <= d['frame'] < end_frame]
        
        if not second_data:
            continue
        
        print(f"{'='*80}")
        print(f"SECOND {second} ({second:.1f}s - {second+1:.1f}s) | Frames {start_frame}-{end_frame}")
        print(f"{'='*80}")
        
        # Calculate statistics
        avg_kps = np.mean([d['valid_kps'] for d in second_data])
        avg_conf = np.mean([d['avg_confidence'] for d in second_data])
        
        # Hip position change
        hip_positions = [d['hip_y'] for d in second_data if d['hip_y'] > 0]
        if len(hip_positions) > 5:
            hip_start = np.mean(hip_positions[:5])
            hip_end = np.mean(hip_positions[-5:])
            hip_change = hip_end - hip_start
        else:
            hip_change = 0
        
        # Probabilities
        probs = [d['probability'] for d in second_data if d['probability'] is not None]
        if probs:
            max_prob = max(probs)
            avg_prob = np.mean(probs)
            min_prob = min(probs)
        else:
            max_prob = None
            avg_prob = None
            min_prob = None
        
        print(f"Detection Quality:")
        print(f"  Average keypoints: {avg_kps:.1f}/17")
        print(f"  Average confidence: {avg_conf:.3f}")
        
        if avg_kps < 10:
            print(f"  ⚠️  Poor detection - person not consistently visible")
        elif avg_kps >= 15:
            print(f"  ✅ Good detection - person clearly visible")
        else:
            print(f"  ⚠️  Moderate detection - partial visibility")
        
        print()
        print(f"Body Movement:")
        if len(hip_positions) > 5:
            print(f"  Hip position start: {hip_start:.3f}")
            print(f"  Hip position end: {hip_end:.3f}")
            print(f"  Hip change: {hip_change:+.3f} (positive = downward)")
            
            if abs(hip_change) > 0.15:
                print(f"  ⚠️  SIGNIFICANT MOVEMENT")
            elif abs(hip_change) > 0.05:
                print(f"  ⚠️  Moderate movement")
            else:
                print(f"  ✅ Minimal movement")
        else:
            print(f"  ⚠️  Insufficient data to track movement")
        
        print()
        print(f"Fall Detection Predictions:")
        if probs:
            print(f"  Max probability: {max_prob:.4f} ({max_prob*100:.2f}%)")
            print(f"  Avg probability: {avg_prob:.4f} ({avg_prob*100:.2f}%)")
            print(f"  Min probability: {min_prob:.4f} ({min_prob*100:.2f}%)")
            
            if max_prob >= 0.85:
                print(f"  🚨 FALL DETECTED! (threshold: 0.85)")
                # Find which frame
                for d in second_data:
                    if d['probability'] == max_prob:
                        print(f"     At frame {d['frame']} ({d['time']:.2f}s)")
                        break
            elif max_prob >= 0.5:
                print(f"  ⚠️  WARNING - Elevated probability")
            elif max_prob >= 0.1:
                print(f"  ⚠️  Moderate probability")
            else:
                print(f"  ✅ Normal activity")
        else:
            print(f"  ⏳ Buffering (need 30 frames)")
        
        print()
        
        # Interpretation
        print(f"What's happening:")
        if avg_kps < 10:
            print(f"  • Person NOT consistently detected by MoveNet")
            print(f"  • May be out of frame, occluded, or poor lighting")
            if probs and max_prob >= 0.5:
                print(f"  • ⚠️  High probability despite poor detection = FALSE POSITIVE")
        elif abs(hip_change) > 0.15:
            print(f"  • Person is moving significantly (descending or ascending)")
            if probs and max_prob >= 0.85:
                print(f"  • 🚨 Model detected this as a FALL")
            elif probs and max_prob < 0.1:
                print(f"  • ✅ Model correctly identified as normal movement")
        elif abs(hip_change) > 0.05:
            print(f"  • Person is moving moderately")
            if probs and max_prob >= 0.5:
                print(f"  • ⚠️  Model is uncertain about this movement")
        else:
            print(f"  • Person is relatively stationary")
            if probs and max_prob < 0.1:
                print(f"  • ✅ Model correctly identified as normal")
        
        print()
    
    # Overall summary
    print("="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print()
    
    all_probs = [d['probability'] for d in all_data if d['probability'] is not None]
    
    if all_probs:
        print(f"Total predictions: {len(all_probs)}")
        print(f"Max probability: {max(all_probs):.6f} ({max(all_probs)*100:.2f}%)")
        print(f"Avg probability: {np.mean(all_probs):.6f} ({np.mean(all_probs)*100:.2f}%)")
        print()
        
        fall_frames = [d for d in all_data if d['probability'] is not None and d['probability'] >= 0.85]
        warning_frames = [d for d in all_data if d['probability'] is not None and 0.5 <= d['probability'] < 0.85]
        
        print(f"Fall detections (≥0.85): {len(fall_frames)}")
        if fall_frames:
            for d in fall_frames:
                print(f"  Frame {d['frame']} ({d['time']:.2f}s): {d['probability']:.4f} | {d['valid_kps']}/17 keypoints")
        
        print()
        print(f"Warnings (0.5-0.85): {len(warning_frames)}")
        if warning_frames:
            for d in warning_frames:
                print(f"  Frame {d['frame']} ({d['time']:.2f}s): {d['probability']:.4f} | {d['valid_kps']}/17 keypoints")
        
        print()
        
        # Final verdict
        print("="*80)
        print("VERDICT")
        print("="*80)
        print()
        
        if fall_frames:
            # Check if falls are real or false positives
            real_falls = [d for d in fall_frames if d['valid_kps'] >= 10]
            false_positives = [d for d in fall_frames if d['valid_kps'] < 10]
            
            if real_falls:
                print(f"✅ REAL FALLS DETECTED: {len(real_falls)}")
                for d in real_falls:
                    print(f"   Frame {d['frame']} ({d['time']:.2f}s): {d['probability']:.4f}")
                print()
            
            if false_positives:
                print(f"❌ FALSE POSITIVES: {len(false_positives)}")
                for d in false_positives:
                    print(f"   Frame {d['frame']} ({d['time']:.2f}s): {d['probability']:.4f} (only {d['valid_kps']}/17 keypoints)")
                print()
        else:
            print("✅ No falls detected in this video")
            print()
    
    print("="*80)


def main():
    """Main entry point."""

    # Allow video path as command line argument
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'data/test/niha.mp4'  # Default to niha.mp4

    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5'

    # Check if files exist
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found: {video_path}")
        return

    if not Path(model_path).exists():
        print(f"❌ Error: Model not found: {model_path}")
        return

    # Run test
    analyze_video_second_by_second(video_path, model_path)


if __name__ == '__main__':
    main()

