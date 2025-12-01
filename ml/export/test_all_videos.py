"""
Test Enhanced Fall Detection on All Test Videos

This script tests the enhanced fall detection system on all test videos
and provides a comprehensive summary of results.

Usage:
    python -m ml.export.test_all_videos
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.export.enhanced_fall_detection import (
    load_movenet, infer_keypoints_movenet, extract_raw_keypoints,
    calculate_body_metrics, enhanced_fall_detection
)


def test_video(video_path: str, movenet_model, lstm_model, expected_fall: bool):
    """
    Test a single video and return results.
    
    Returns:
        dict with keys:
            - video_name: Name of video
            - expected_fall: Whether a fall was expected
            - detected_fall: Whether a fall was detected
            - correct: Whether detection was correct
            - max_probability: Maximum probability
            - fall_events: List of fall events
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    window_size = 30
    keypoint_buffer = deque(maxlen=window_size)
    metrics_history = deque(maxlen=window_size)
    
    frame_idx = 0
    fall_detections = []
    max_probability = 0.0
    
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
        if len(keypoint_buffer) == window_size:
            window = np.array(list(keypoint_buffer), dtype=np.float32)
            window_batch = np.expand_dims(window, axis=0)
            probability = lstm_model.predict(window_batch, verbose=0)[0][0]
            
            max_probability = max(max_probability, probability)
            
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
                    'rule': rule
                })
    
    cap.release()
    
    # Group consecutive detections
    fall_events = []
    if fall_detections:
        current_group = [fall_detections[0]]
        
        for detection in fall_detections[1:]:
            if detection['frame'] - current_group[-1]['frame'] <= 5:
                current_group.append(detection)
            else:
                fall_events.append(current_group)
                current_group = [detection]
        fall_events.append(current_group)
    
    detected_fall = len(fall_events) > 0
    correct = (detected_fall == expected_fall)
    
    return {
        'video_name': Path(video_path).name,
        'expected_fall': expected_fall,
        'detected_fall': detected_fall,
        'correct': correct,
        'max_probability': max_probability,
        'fall_events': fall_events
    }


def main():
    """Main entry point."""
    
    print("="*80)
    print("TESTING ENHANCED FALL DETECTION ON ALL VIDEOS")
    print("="*80)
    print()
    
    # Define test videos
    test_videos = [
        ('data/test/nihastand.mp4', False, 'Standing (normal activity)'),
        ('data/test/nihapass.mp4', True, 'Slow fall'),
        ('data/test/nihafast.mp4', True, 'Fast fall'),
        ('data/test/nihacase6.mp4', True, 'Sustained fall'),
        ('data/test/niha.mp4', True, 'Person on ground'),
        ('data/test/2.mp4', True, 'Slow fall (controlled descent with fall)'),
        ('data/test/nihaonelast.mp4', True, 'Chair fall'),
        ('data/test/idle.mp4', False, 'Idle/moving around'),
    ]
    
    # Load models
    print("[1/2] Loading models...")
    movenet_model = load_movenet()
    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5'
    lstm_model = tf.keras.models.load_model(model_path)
    print("✓ Models loaded")
    print()
    
    # Test each video
    print("[2/2] Testing videos...")
    print()
    
    results = []
    for video_path, expected_fall, description in test_videos:
        if not Path(video_path).exists():
            print(f"⚠️  Skipping {Path(video_path).name} (not found)")
            continue
        
        print(f"Testing {Path(video_path).name}... ", end='', flush=True)
        result = test_video(video_path, movenet_model, lstm_model, expected_fall)
        
        if result:
            result['description'] = description
            results.append(result)
            
            if result['correct']:
                print("✅ PASS")
            else:
                print("❌ FAIL")
        else:
            print("❌ ERROR")
    
    print()
    
    # Print detailed results
    print("="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print()
    
    for result in results:
        status = "✅ PASS" if result['correct'] else "❌ FAIL"
        
        print(f"{status} | {result['video_name']}")
        print(f"  Description: {result['description']}")
        print(f"  Expected fall: {'YES' if result['expected_fall'] else 'NO'}")
        print(f"  Detected fall: {'YES' if result['detected_fall'] else 'NO'}")
        print(f"  Max probability: {result['max_probability']:.4f} ({result['max_probability']*100:.2f}%)")
        
        if result['fall_events']:
            print(f"  Fall events: {len(result['fall_events'])}")
            for i, event in enumerate(result['fall_events'], 1):
                start_time = event[0]['time']
                end_time = event[-1]['time']
                max_prob = max(e['probability'] for e in event)
                rule = max(event, key=lambda x: x['probability'])['rule']
                print(f"    Event {i}: {start_time:.2f}s - {end_time:.2f}s, prob={max_prob:.2f}, {rule}")
        else:
            print(f"  Fall events: 0")
        
        print()
    
    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Count by category
    true_positives = sum(1 for r in results if r['expected_fall'] and r['detected_fall'])
    true_negatives = sum(1 for r in results if not r['expected_fall'] and not r['detected_fall'])
    false_positives = sum(1 for r in results if not r['expected_fall'] and r['detected_fall'])
    false_negatives = sum(1 for r in results if r['expected_fall'] and not r['detected_fall'])
    
    print(f"Total videos tested: {total}")
    print(f"Correct detections: {correct}/{total} ({accuracy:.1f}%)")
    print()
    print(f"True Positives (fall correctly detected): {true_positives}")
    print(f"True Negatives (no fall correctly detected): {true_negatives}")
    print(f"False Positives (false alarm): {false_positives}")
    print(f"False Negatives (missed fall): {false_negatives}")
    print()
    
    if accuracy == 100:
        print("🎉 PERFECT SCORE! All videos correctly classified!")
    elif accuracy >= 90:
        print("✅ EXCELLENT! Very high accuracy.")
    elif accuracy >= 75:
        print("⚠️  GOOD, but room for improvement.")
    else:
        print("❌ NEEDS IMPROVEMENT")
    
    print()
    print("="*80)


if __name__ == '__main__':
    main()

