"""
Test fall detection model with YOLO pose estimation on video.

This script:
1. Loads the trained BiLSTM model
2. Uses YOLO11n-pose for keypoint extraction
3. Processes video frame-by-frame
4. Shows detailed analysis of predictions

Usage:
    python -m ml.export.test_yolo_on_video
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
    """
    Extract raw keypoints (x, y coordinates only).
    
    Args:
        keypoints: (17, 3) array with [y, x, confidence]
    
    Returns:
        (34,) array with flattened [x, y] coordinates
    """
    # Extract x, y coordinates (swap y, x to x, y order)
    xy_coords = keypoints[:, [1, 0]]  # (17, 2) - [x, y]
    
    # Flatten to (34,)
    features = xy_coords.flatten()
    
    return features


def test_video(video_path: str, model_path: str):
    """Test fall detection on video."""
    
    print("="*80)
    print("FALL DETECTION TEST WITH YOLO POSE")
    print("="*80)
    print()
    
    # Load YOLO model
    print("[1/4] Loading YOLO11n-pose model...")
    yolo_model = load_yolo('yolo11n-pose.pt')
    print("✓ YOLO model loaded")
    print()
    
    # Load BiLSTM model
    print("[2/4] Loading BiLSTM fall detection model...")
    print(f"Model path: {model_path}")
    lstm_model = tf.keras.models.load_model(model_path)
    print("✓ BiLSTM model loaded")
    print()
    
    # Print model info
    print("Model Architecture:")
    lstm_model.summary()
    print()
    
    # Open video
    print(f"[3/4] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Video opened")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print()
    
    # Process video
    print("[4/4] Processing video...")
    print()
    
    # Buffer for 30-frame windows
    window_size = 30
    keypoint_buffer = deque(maxlen=window_size)
    
    frame_idx = 0
    predictions = []
    keypoint_stats = []
    
    print(f"{'Frame':<8} {'Keypoints':<12} {'Confidence':<12} {'Probability':<12} {'Status':<15}")
    print("-"*80)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints with YOLO
        keypoints = infer_keypoints_yolo(yolo_model, frame_rgb, 
                                         confidence_threshold=0.3, 
                                         normalize=True)
        
        # Extract features (34 values)
        features = extract_raw_keypoints(keypoints)
        
        # Add to buffer
        keypoint_buffer.append(features)
        
        # Calculate stats
        valid_keypoints = np.sum(keypoints[:, 2] >= 0.3)
        avg_confidence = keypoints[:, 2].mean()
        
        keypoint_stats.append({
            'frame': frame_idx,
            'valid_keypoints': valid_keypoints,
            'avg_confidence': avg_confidence,
            'keypoints': keypoints.copy()
        })
        
        # Once we have 30 frames, make prediction
        if len(keypoint_buffer) == window_size:
            # Stack into (30, 34) array
            window = np.array(list(keypoint_buffer), dtype=np.float32)
            
            # Add batch dimension: (1, 30, 34)
            window_batch = np.expand_dims(window, axis=0)
            
            # Predict
            probability = lstm_model.predict(window_batch, verbose=0)[0][0]
            
            predictions.append({
                'frame': frame_idx,
                'probability': probability,
                'window': window.copy()
            })
            
            # Determine status
            if probability >= 0.85:
                status = "🚨 FALL DETECTED"
            elif probability >= 0.5:
                status = "⚠️  WARNING"
            else:
                status = "✅ Normal"
            
            # Print every 5 frames to avoid clutter
            if frame_idx % 5 == 0 or probability >= 0.5:
                print(f"{frame_idx:<8} {valid_keypoints}/17      "
                      f"{avg_confidence:>6.3f}       "
                      f"{probability:>8.4f}     {status}")
        else:
            # Not enough frames yet
            print(f"{frame_idx:<8} {valid_keypoints}/17      "
                  f"{avg_confidence:>6.3f}       "
                  f"{'---':>8}     Buffering...")
    
    cap.release()
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    
    # Statistics
    if predictions:
        probs = [p['probability'] for p in predictions]
        
        print("📊 Prediction Statistics:")
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Min probability: {min(probs):.6f}")
        print(f"  Max probability: {max(probs):.6f}")
        print(f"  Mean probability: {np.mean(probs):.6f}")
        print(f"  Median probability: {np.median(probs):.6f}")
        print(f"  Std deviation: {np.std(probs):.6f}")
        print()
        
        # Count by threshold
        fall_count = sum(1 for p in probs if p >= 0.85)
        warning_count = sum(1 for p in probs if 0.5 <= p < 0.85)
        normal_count = sum(1 for p in probs if p < 0.5)
        
        print("📈 Prediction Distribution:")
        print(f"  🚨 Fall detected (≥0.85): {fall_count} ({100*fall_count/len(probs):.1f}%)")
        print(f"  ⚠️  Warning (0.5-0.85): {warning_count} ({100*warning_count/len(probs):.1f}%)")
        print(f"  ✅ Normal (<0.5): {normal_count} ({100*normal_count/len(probs):.1f}%)")
        print()
        
        # Top 5 highest probabilities
        top_5 = sorted(predictions, key=lambda x: x['probability'], reverse=True)[:5]
        print("🔝 Top 5 Highest Probabilities:")
        for i, pred in enumerate(top_5, 1):
            print(f"  {i}. Frame {pred['frame']}: {pred['probability']:.6f}")
        print()
        
        # Keypoint quality
        valid_kps = [s['valid_keypoints'] for s in keypoint_stats]
        confidences = [s['avg_confidence'] for s in keypoint_stats]
        
        print("🎯 Keypoint Quality:")
        print(f"  Avg valid keypoints: {np.mean(valid_kps):.1f}/17")
        print(f"  Min valid keypoints: {min(valid_kps)}/17")
        print(f"  Max valid keypoints: {max(valid_kps)}/17")
        print(f"  Avg confidence: {np.mean(confidences):.3f}")
        print()
        
        # Analyze a sample window
        print("🔍 Sample Window Analysis (Frame 30):")
        if len(predictions) > 0:
            sample = predictions[0]
            window = sample['window']
            
            print(f"  Window shape: {window.shape}")
            print(f"  Value range: [{window.min():.6f}, {window.max():.6f}]")
            print(f"  Mean: {window.mean():.6f}")
            print(f"  Std: {window.std():.6f}")
            print(f"  Zeros: {np.sum(window == 0)} / {window.size} ({100*np.sum(window == 0)/window.size:.1f}%)")
            print()
            
            # Show first frame of window
            print("  First frame features (first 10 values):")
            print(f"    {window[0, :10]}")
            print()
    else:
        print("⚠️  No predictions made (video too short?)")
    
    print("="*80)


def main():
    """Main entry point."""
    
    # Paths
    video_path = 'data/test/2.mp4'
    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5'
    
    # Check if files exist
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found: {video_path}")
        return
    
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found: {model_path}")
        print("\nAvailable models:")
        checkpoint_dir = Path('ml/training/checkpoints')
        if checkpoint_dir.exists():
            for model_file in checkpoint_dir.glob('*.h5'):
                print(f"  - {model_file.name}")
        return
    
    # Run test
    test_video(video_path, model_path)


if __name__ == '__main__':
    main()

