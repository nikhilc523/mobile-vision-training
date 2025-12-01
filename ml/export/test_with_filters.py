"""
Test fall detection with person detection filters.

This script demonstrates the fix for false positives caused by
intermittent pose detection.

Usage:
    python -m ml.export.test_with_filters
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


def is_person_detected(keypoint_buffer, min_keypoints=15, min_consecutive=5):
    """
    Check if person is consistently detected.
    
    Args:
        keypoint_buffer: Deque of (34,) feature arrays
        min_keypoints: Minimum valid keypoints required (default: 15/17)
        min_consecutive: Minimum consecutive frames (default: 5)
    
    Returns:
        (is_detected, valid_count, reason)
    """
    if len(keypoint_buffer) < min_consecutive:
        return False, 0, "Buffering"
    
    # Check last N frames
    recent_frames = list(keypoint_buffer)[-min_consecutive:]
    
    valid_counts = []
    for frame_features in recent_frames:
        # Count non-zero coordinates (each keypoint has x, y)
        non_zero_coords = np.sum(frame_features != 0)
        valid_keypoints = non_zero_coords // 2
        valid_counts.append(valid_keypoints)
        
        if valid_keypoints < min_keypoints:
            return False, min(valid_counts), f"Only {valid_keypoints}/{min_keypoints} keypoints"
    
    return True, min(valid_counts), "OK"


def has_sufficient_data(window, max_zero_ratio=0.5):
    """
    Check if window has sufficient non-zero data.
    
    Args:
        window: (30, 34) array
        max_zero_ratio: Maximum allowed ratio of zero values
    
    Returns:
        (has_data, zero_ratio, reason)
    """
    total_values = window.size
    zero_values = np.sum(window == 0)
    zero_ratio = zero_values / total_values
    
    if zero_ratio <= max_zero_ratio:
        return True, zero_ratio, "OK"
    else:
        return False, zero_ratio, f"Too many zeros ({100*zero_ratio:.0f}%)"


def should_run_fall_detection(keypoint_buffer, min_keypoints=15, min_consecutive=5, max_zero_ratio=0.5):
    """
    Determine if fall detection should run on current window.
    
    Returns:
        (should_run, reason, details)
    """
    if len(keypoint_buffer) < 30:
        return False, "Buffering", {}
    
    # Check 1: Person consistently detected
    person_ok, valid_count, person_reason = is_person_detected(
        keypoint_buffer, min_keypoints, min_consecutive
    )
    
    if not person_ok:
        return False, f"Person check failed: {person_reason}", {
            'valid_keypoints': valid_count,
            'required': min_keypoints
        }
    
    # Check 2: Sufficient data in window
    window = np.array(list(keypoint_buffer))
    data_ok, zero_ratio, data_reason = has_sufficient_data(window, max_zero_ratio)
    
    if not data_ok:
        return False, f"Data check failed: {data_reason}", {
            'zero_ratio': zero_ratio,
            'max_allowed': max_zero_ratio
        }
    
    return True, "All checks passed", {
        'valid_keypoints': valid_count,
        'zero_ratio': zero_ratio
    }


def test_video_with_filters(video_path: str, model_path: str):
    """Test fall detection with filters."""
    
    print("="*80)
    print("FALL DETECTION TEST WITH FILTERS")
    print("="*80)
    print()
    
    # Load models
    print("[1/4] Loading models...")
    yolo_model = load_yolo('yolo11n-pose.pt')
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
    
    print(f"✓ Video opened")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print()
    
    # Filter settings
    print("[3/4] Filter settings:")
    print("  Min keypoints: 15/17")
    print("  Min consecutive frames: 5")
    print("  Max zero ratio: 50%")
    print()
    
    # Process video
    print("[4/4] Processing video...")
    print()
    
    window_size = 30
    keypoint_buffer = deque(maxlen=window_size)
    
    frame_idx = 0
    predictions = []
    skipped = []
    
    print(f"{'Frame':<8} {'Keypoints':<12} {'Filter':<25} {'Probability':<12} {'Status':<15}")
    print("-"*90)
    
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
        
        # Check if we should run fall detection
        should_run, reason, details = should_run_fall_detection(keypoint_buffer)
        
        if should_run:
            # Run fall detection
            window = np.array(list(keypoint_buffer), dtype=np.float32)
            window_batch = np.expand_dims(window, axis=0)
            probability = lstm_model.predict(window_batch, verbose=0)[0][0]
            
            predictions.append({
                'frame': frame_idx,
                'probability': probability
            })
            
            # Determine status
            if probability >= 0.85:
                status = "🚨 FALL DETECTED"
            elif probability >= 0.5:
                status = "⚠️  WARNING"
            else:
                status = "✅ Normal"
            
            # Print every 5 frames or if fall detected
            if frame_idx % 5 == 0 or probability >= 0.5:
                print(f"{frame_idx:<8} {valid_keypoints}/17      "
                      f"{'✅ PASS':<25} "
                      f"{probability:>8.4f}     {status}")
        else:
            # Skipped
            skipped.append({
                'frame': frame_idx,
                'reason': reason,
                'details': details
            })
            
            # Print every 5 frames
            if frame_idx % 5 == 0:
                print(f"{frame_idx:<8} {valid_keypoints}/17      "
                      f"{reason[:23]:<25} "
                      f"{'---':>8}     ⏭️  Skipped")
    
    cap.release()
    
    print()
    print("="*80)
    print("RESULTS WITH FILTERS")
    print("="*80)
    print()
    
    # Statistics
    total_windows = len(predictions) + len(skipped)
    
    print("📊 Processing Statistics:")
    print(f"  Total windows: {total_windows}")
    print(f"  Predictions made: {len(predictions)} ({100*len(predictions)/total_windows:.1f}%)")
    print(f"  Windows skipped: {len(skipped)} ({100*len(skipped)/total_windows:.1f}%)")
    print()
    
    if predictions:
        probs = [p['probability'] for p in predictions]
        
        print("📈 Prediction Statistics:")
        print(f"  Min probability: {min(probs):.6f}")
        print(f"  Max probability: {max(probs):.6f}")
        print(f"  Mean probability: {np.mean(probs):.6f}")
        print(f"  Median probability: {np.median(probs):.6f}")
        print()
        
        # Count by threshold
        fall_count = sum(1 for p in probs if p >= 0.85)
        warning_count = sum(1 for p in probs if 0.5 <= p < 0.85)
        normal_count = sum(1 for p in probs if p < 0.5)
        
        print("🎯 Detection Results:")
        print(f"  🚨 Fall detected (≥0.85): {fall_count} ({100*fall_count/len(probs):.1f}%)")
        print(f"  ⚠️  Warning (0.5-0.85): {warning_count} ({100*warning_count/len(probs):.1f}%)")
        print(f"  ✅ Normal (<0.5): {normal_count} ({100*normal_count/len(probs):.1f}%)")
        print()
        
        if fall_count > 0:
            print("⚠️  Falls detected:")
            for pred in predictions:
                if pred['probability'] >= 0.85:
                    print(f"    Frame {pred['frame']}: {pred['probability']:.4f}")
            print()
    
    # Skip reasons
    if skipped:
        skip_reasons = {}
        for s in skipped:
            reason = s['reason'].split(':')[0]  # Get main reason
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        
        print("📋 Skip Reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} ({100*count/len(skipped):.1f}%)")
        print()
    
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print()
    print("Without filters (previous test):")
    print("  🚨 False positives: 10 frames (3.0%)")
    print("  Frames 69-78 incorrectly detected as falls")
    print()
    print("With filters (this test):")
    print(f"  🚨 False positives: {sum(1 for p in predictions if p['probability'] >= 0.85)} frames")
    print(f"  Windows skipped: {len(skipped)} (prevented false positives)")
    print()
    
    if len(predictions) > 0 and sum(1 for p in predictions if p['probability'] >= 0.85) == 0:
        print("✅ SUCCESS! No false positives detected!")
    elif len(predictions) > 0:
        print("⚠️  Some detections still present - may need stricter filters")
    else:
        print("⚠️  No predictions made - filters may be too strict")
    
    print()
    print("="*80)


def main():
    """Main entry point."""
    
    video_path = 'data/test/2.mp4'
    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5'
    
    # Check if files exist
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found: {video_path}")
        return
    
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found: {model_path}")
        return
    
    # Run test
    test_video_with_filters(video_path, model_path)


if __name__ == '__main__':
    main()

