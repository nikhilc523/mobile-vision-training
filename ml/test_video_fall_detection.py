#!/usr/bin/env python3
"""
Test Fall Detection on Video Using MoveNet + BiLSTM

This script:
1. Loads a test video (data/test/2.mp4)
2. Extracts keypoints using MoveNet (same as training)
3. Creates 30-frame sliding windows
4. Runs fall detection using the BiLSTM TFLite model
5. Shows frame-by-frame probability and detects falls

Usage:
    python ml/test_video_fall_detection.py
"""

import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.pose.movenet_loader import load_movenet, infer_keypoints


class FallDetectorTFLite:
    """Fall detection using TFLite BiLSTM model."""
    
    def __init__(self, model_path: str):
        """
        Initialize fall detector with TFLite model.
        
        Args:
            model_path: Path to fall_detection_model.tflite
        """
        print(f"Loading BiLSTM model from: {model_path}")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {self.input_details[0]['shape']}")
        print(f"   Output shape: {self.output_details[0]['shape']}")
        print()
    
    def detect_fall(self, keypoints_buffer: np.ndarray) -> float:
        """
        Run fall detection on 30-frame keypoint buffer.
        
        Args:
            keypoints_buffer: (30, 34) array of keypoints
        
        Returns:
            Fall probability [0, 1]
        """
        # Prepare input (add batch dimension)
        input_data = keypoints_buffer.reshape(1, 30, 34).astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        probability = float(output_data[0][0])
        
        return probability


def extract_keypoints_from_movenet(keypoints_raw: np.ndarray) -> np.ndarray:
    """
    Convert MoveNet keypoints to BiLSTM input format.

    MoveNet outputs: (17, 3) with [y, x, confidence]
    BiLSTM expects: (34,) with [x, y, x, y, ...] for 17 keypoints

    IMPORTANT: Training data has range [-0.0001, 1.008], so we DON'T clamp!

    Args:
        keypoints_raw: (17, 3) array from MoveNet with [y, x, confidence]

    Returns:
        (34,) array with [x, y] coordinates for 17 keypoints
    """
    features = np.zeros(34, dtype=np.float32)

    for i in range(17):
        y, x, conf = keypoints_raw[i]

        # Store as [x, y] (swap from MoveNet's [y, x])
        # DON'T clamp - training data has values slightly outside [0, 1]
        features[i * 2] = x
        features[i * 2 + 1] = y

    return features


def test_video_fall_detection(video_path: str, model_path: str, 
                               confidence_threshold: float = 0.3,
                               fall_threshold: float = 0.85):
    """
    Test fall detection on a video.
    
    Args:
        video_path: Path to test video
        model_path: Path to BiLSTM TFLite model
        confidence_threshold: Minimum keypoint confidence (default: 0.3)
        fall_threshold: Fall detection threshold (default: 0.85)
    """
    print("="*80)
    print("FALL DETECTION TEST ON VIDEO")
    print("="*80)
    print()
    
    # Check if video exists
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ ERROR: Video not found: {video_path}")
        return
    
    # Load MoveNet
    print("Loading MoveNet pose estimator...")
    movenet_fn = load_movenet()
    print()
    
    # Load BiLSTM fall detector
    fall_detector = FallDetectorTFLite(model_path)
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ ERROR: Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"✅ Video opened successfully!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    print()
    
    # Buffer for 30 frames
    keypoints_buffer = deque(maxlen=30)
    
    # Statistics
    frame_count = 0
    fall_detected_frames = []
    probabilities = []
    valid_keypoint_counts = []
    
    print("="*80)
    print("PROCESSING VIDEO")
    print("="*80)
    print()
    print(f"{'Frame':<8} {'Time':<8} {'Keypoints':<12} {'Probability':<15} {'Status':<20}")
    print("-"*80)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps if fps > 0 else 0
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints using MoveNet
        keypoints_raw = infer_keypoints(movenet_fn, frame_rgb, 
                                       confidence_threshold=confidence_threshold)
        
        # Convert to BiLSTM format (34 features)
        keypoints_features = extract_keypoints_from_movenet(keypoints_raw)
        
        # Add to buffer
        keypoints_buffer.append(keypoints_features)
        
        # Count valid keypoints
        valid_kps = np.sum(keypoints_raw[:, 2] >= confidence_threshold)
        valid_keypoint_counts.append(valid_kps)
        
        # Run fall detection if buffer is full
        if len(keypoints_buffer) == 30:
            # Convert buffer to numpy array
            buffer_array = np.array(keypoints_buffer, dtype=np.float32)
            
            # Run fall detection
            probability = fall_detector.detect_fall(buffer_array)
            probabilities.append(probability)
            
            # Check if fall detected
            is_fall = probability > fall_threshold
            if is_fall:
                fall_detected_frames.append(frame_count)
            
            # Print status
            status = "🚨 FALL DETECTED!" if is_fall else "✅ Normal"
            print(f"{frame_count:<8} {timestamp:<8.2f} {valid_kps}/17{'':<7} "
                  f"{probability:<15.6f} {status:<20}")
        else:
            # Buffer not full yet
            print(f"{frame_count:<8} {timestamp:<8.2f} {valid_kps}/17{'':<7} "
                  f"{'Buffering...':<15} {'(need 30 frames)':<20}")
    
    cap.release()
    
    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    print(f"📊 Video Statistics:")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Frames with fall detection: {len(probabilities)}")
    print(f"   Average keypoints detected: {np.mean(valid_keypoint_counts):.1f}/17")
    print()
    
    if probabilities:
        print(f"📈 Fall Detection Statistics:")
        print(f"   Min probability: {np.min(probabilities):.6f} ({np.min(probabilities)*100:.4f}%)")
        print(f"   Max probability: {np.max(probabilities):.6f} ({np.max(probabilities)*100:.4f}%)")
        print(f"   Mean probability: {np.mean(probabilities):.6f} ({np.mean(probabilities)*100:.4f}%)")
        print(f"   Std probability: {np.std(probabilities):.6f}")
        print()
        
        if fall_detected_frames:
            print(f"🚨 FALL DETECTED!")
            print(f"   Number of frames with fall: {len(fall_detected_frames)}")
            print(f"   First detection at frame: {fall_detected_frames[0]} ({fall_detected_frames[0]/fps:.2f}s)")
            print(f"   Last detection at frame: {fall_detected_frames[-1]} ({fall_detected_frames[-1]/fps:.2f}s)")
            print()
            print(f"✅ RESULT: Fall detection is WORKING! 🎉")
        else:
            print(f"✅ NO FALL DETECTED")
            print(f"   All probabilities below threshold ({fall_threshold})")
            print()
            
            if np.max(probabilities) < 0.01:
                print(f"⚠️  WARNING: Maximum probability is very low ({np.max(probabilities):.6f})")
                print(f"   This suggests a training/inference format mismatch!")
                print()
                print(f"🔍 Debugging Information:")
                print(f"   - Keypoints are being extracted: {np.mean(valid_keypoint_counts):.1f}/17 average")
                print(f"   - Buffer is filling correctly: 30 frames")
                print(f"   - Model is running: {len(probabilities)} inferences")
                print(f"   - BUT: Probabilities are near zero!")
                print()
                print(f"💡 Possible causes:")
                print(f"   1. Coordinate order mismatch ([x,y] vs [y,x])")
                print(f"   2. Normalization mismatch")
                print(f"   3. Keypoint order mismatch")
                print(f"   4. Different preprocessing in training vs inference")
            else:
                print(f"✅ RESULT: Model is working, but no fall in this video")
    else:
        print(f"❌ ERROR: No fall detection results (video too short?)")
        print(f"   Need at least 30 frames for fall detection")
    
    print()
    print("="*80)


def main():
    """Main entry point."""
    # Paths
    video_path = "data/test/finalfall.mp4"  # Changed to finalfall.mp4
    model_path = "ml/export/fall_detection_model.tflite"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ ERROR: Model not found: {model_path}")
        print(f"   Please run the TFLite conversion script first:")
        print(f"   python ml/export/convert_to_tflite.py")
        return
    
    # Run test
    test_video_fall_detection(
        video_path=video_path,
        model_path=model_path,
        confidence_threshold=0.3,
        fall_threshold=0.85
    )


if __name__ == '__main__':
    main()

