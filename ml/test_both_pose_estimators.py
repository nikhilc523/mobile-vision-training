"""
Test Fall Detection with BOTH YOLO and MoveNet
Compare results side-by-side to see which works better.
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pose.movenet_loader import load_movenet, infer_keypoints


def load_tflite_model(model_path: str):
    """Load TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def extract_keypoints_movenet(keypoints_raw: np.ndarray) -> np.ndarray:
    """
    Convert MoveNet keypoints to BiLSTM format.
    MoveNet outputs: (17, 3) with [y, x, confidence]
    BiLSTM expects: (34,) with [x, y, x, y, ...]
    """
    features = np.zeros(34, dtype=np.float32)
    
    for i in range(17):
        y, x, conf = keypoints_raw[i]
        features[i * 2] = x      # x first
        features[i * 2 + 1] = y  # y second
    
    return features


def extract_keypoints_yolo(yolo_output: np.ndarray, conf_threshold: float = 0.3) -> np.ndarray:
    """
    Convert YOLO keypoints to BiLSTM format.
    YOLO outputs: (56, 8400) in CHW format
    BiLSTM expects: (34,) with [x, y, x, y, ...]
    """
    features = np.zeros(34, dtype=np.float32)
    
    # YOLO output format: [x, y, w, h, ...51 keypoint values, conf]
    # Keypoints start at index 4, format: [x, y, conf] for each of 17 keypoints
    
    # Find the detection with highest confidence
    conf_scores = yolo_output[4, :]  # Box confidence scores
    best_idx = np.argmax(conf_scores)
    detection = yolo_output[:, best_idx]
    
    # Extract keypoints (indices 5-55 contain 17 keypoints × 3 values)
    for i in range(17):
        kp_start = 5 + (i * 3)
        x = detection[kp_start]
        y = detection[kp_start + 1]
        conf = detection[kp_start + 2]
        
        # Apply confidence threshold
        if conf >= conf_threshold:
            features[i * 2] = x
            features[i * 2 + 1] = y
        else:
            features[i * 2] = 0.0
            features[i * 2 + 1] = 0.0
    
    return features


def run_fall_detection(bilstm_interpreter, buffer: np.ndarray) -> float:
    """Run fall detection on 30-frame buffer."""
    input_details = bilstm_interpreter.get_input_details()
    output_details = bilstm_interpreter.get_output_details()
    
    # Prepare input: (1, 30, 34)
    input_data = buffer.reshape(1, 30, 34).astype(np.float32)
    
    # Run inference
    bilstm_interpreter.set_tensor(input_details[0]['index'], input_data)
    bilstm_interpreter.invoke()
    
    # Get output
    output = bilstm_interpreter.get_tensor(output_details[0]['index'])
    probability = float(output[0, 0])
    
    return probability


def test_with_movenet(video_path: str, bilstm_model_path: str):
    """Test fall detection using MoveNet."""
    print("\n" + "="*80)
    print("🔵 TESTING WITH MOVENET")
    print("="*80)
    
    # Load models
    print("Loading MoveNet...")
    movenet_fn = load_movenet("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    
    print("Loading BiLSTM...")
    bilstm = load_tflite_model(bilstm_model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total_frames} frames @ {fps:.2f} FPS")
    
    # Buffer for 30 frames
    buffer = []
    frame_idx = 0
    
    results = {
        'probabilities': [],
        'keypoint_counts': [],
        'max_prob': 0.0,
        'fall_detected': False
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints
        keypoints_raw = infer_keypoints(movenet_fn, frame_rgb, confidence_threshold=0.3)
        keypoints = extract_keypoints_movenet(keypoints_raw)
        
        # Count detected keypoints
        kp_count = np.sum(keypoints_raw[:, 2] >= 0.3)
        results['keypoint_counts'].append(kp_count)
        
        # Add to buffer
        buffer.append(keypoints)
        
        # Keep only last 30 frames
        if len(buffer) > 30:
            buffer.pop(0)
        
        # Run fall detection when buffer is full
        if len(buffer) == 30:
            buffer_array = np.array(buffer, dtype=np.float32)
            prob = run_fall_detection(bilstm, buffer_array)
            results['probabilities'].append(prob)
            
            if prob > results['max_prob']:
                results['max_prob'] = prob
            
            if prob >= 0.85:
                results['fall_detected'] = True
                print(f"🚨 FALL DETECTED at frame {frame_idx}! Probability: {prob*100:.2f}%")
        
        frame_idx += 1
        
        # Print progress every 30 frames
        if frame_idx % 30 == 0:
            if results['probabilities']:
                latest_prob = results['probabilities'][-1]
                print(f"Frame {frame_idx}/{total_frames}: {kp_count}/17 keypoints, Prob: {latest_prob*100:.4f}%")
    
    cap.release()
    
    # Print summary
    print("\n" + "-"*80)
    print("📊 MOVENET RESULTS:")
    print("-"*80)
    
    if results['probabilities']:
        probs = np.array(results['probabilities'])
        print(f"Total inferences: {len(probs)}")
        print(f"Min probability: {probs.min()*100:.4f}%")
        print(f"Max probability: {probs.max()*100:.4f}%")
        print(f"Mean probability: {probs.mean()*100:.4f}%")
        print(f"Std probability: {probs.std()*100:.4f}%")
        print(f"Avg keypoints: {np.mean(results['keypoint_counts']):.1f}/17")
        print(f"Fall detected: {'✅ YES' if results['fall_detected'] else '❌ NO'}")
    
    return results


def test_with_yolo(video_path: str, yolo_model_path: str, bilstm_model_path: str):
    """Test fall detection using YOLO."""
    print("\n" + "="*80)
    print("🟡 TESTING WITH YOLO")
    print("="*80)
    
    # Load models
    print("Loading YOLO...")
    yolo = load_tflite_model(yolo_model_path)
    
    print("Loading BiLSTM...")
    bilstm = load_tflite_model(bilstm_model_path)
    
    # Get YOLO input/output details
    yolo_input_details = yolo.get_input_details()
    yolo_output_details = yolo.get_output_details()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total_frames} frames @ {fps:.2f} FPS")
    
    # Buffer for 30 frames
    buffer = []
    frame_idx = 0
    
    results = {
        'probabilities': [],
        'keypoint_counts': [],
        'max_prob': 0.0,
        'fall_detected': False
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess for YOLO (640x640)
        frame_resized = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(frame_normalized, axis=0)
        
        # Run YOLO
        yolo.set_tensor(yolo_input_details[0]['index'], input_tensor)
        yolo.invoke()
        yolo_output = yolo.get_tensor(yolo_output_details[0]['index'])[0]
        
        # Extract keypoints
        keypoints = extract_keypoints_yolo(yolo_output, conf_threshold=0.3)
        
        # Count detected keypoints (non-zero coordinates)
        kp_count = np.sum(keypoints != 0) // 2
        results['keypoint_counts'].append(kp_count)
        
        # Add to buffer
        buffer.append(keypoints)
        
        # Keep only last 30 frames
        if len(buffer) > 30:
            buffer.pop(0)
        
        # Run fall detection when buffer is full
        if len(buffer) == 30:
            buffer_array = np.array(buffer, dtype=np.float32)
            prob = run_fall_detection(bilstm, buffer_array)
            results['probabilities'].append(prob)
            
            if prob > results['max_prob']:
                results['max_prob'] = prob
            
            if prob >= 0.85:
                results['fall_detected'] = True
                print(f"🚨 FALL DETECTED at frame {frame_idx}! Probability: {prob*100:.2f}%")
        
        frame_idx += 1
        
        # Print progress every 30 frames
        if frame_idx % 30 == 0:
            if results['probabilities']:
                latest_prob = results['probabilities'][-1]
                print(f"Frame {frame_idx}/{total_frames}: {kp_count}/17 keypoints, Prob: {latest_prob*100:.4f}%")
    
    cap.release()
    
    # Print summary
    print("\n" + "-"*80)
    print("📊 YOLO RESULTS:")
    print("-"*80)
    
    if results['probabilities']:
        probs = np.array(results['probabilities'])
        print(f"Total inferences: {len(probs)}")
        print(f"Min probability: {probs.min()*100:.4f}%")
        print(f"Max probability: {probs.max()*100:.4f}%")
        print(f"Mean probability: {probs.mean()*100:.4f}%")
        print(f"Std probability: {probs.std()*100:.4f}%")
        print(f"Avg keypoints: {np.mean(results['keypoint_counts']):.1f}/17")
        print(f"Fall detected: {'✅ YES' if results['fall_detected'] else '❌ NO'}")
    
    return results


def main():
    """Main entry point."""
    video_path = "data/test/2.mp4"
    yolo_model_path = "ml/export/yolo11n-pose_float32.tflite"
    bilstm_model_path = "ml/export/fall_detection_model.tflite"
    
    print("\n" + "="*80)
    print("🧪 FALL DETECTION COMPARISON TEST")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"YOLO Model: {yolo_model_path}")
    print(f"BiLSTM Model: {bilstm_model_path}")
    
    # Test with MoveNet
    movenet_results = test_with_movenet(video_path, bilstm_model_path)
    
    # Test with YOLO
    yolo_results = test_with_yolo(video_path, yolo_model_path, bilstm_model_path)
    
    # Final comparison
    print("\n" + "="*80)
    print("🏆 FINAL COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'MoveNet':<20} {'YOLO':<20}")
    print("-" * 65)
    
    if movenet_results['probabilities'] and yolo_results['probabilities']:
        movenet_probs = np.array(movenet_results['probabilities'])
        yolo_probs = np.array(yolo_results['probabilities'])
        
        print(f"{'Max Probability':<25} {movenet_probs.max()*100:>18.4f}% {yolo_probs.max()*100:>18.4f}%")
        print(f"{'Mean Probability':<25} {movenet_probs.mean()*100:>18.4f}% {yolo_probs.mean()*100:>18.4f}%")
        print(f"{'Avg Keypoints':<25} {np.mean(movenet_results['keypoint_counts']):>18.1f}/17 {np.mean(yolo_results['keypoint_counts']):>18.1f}/17")
        print(f"{'Fall Detected':<25} {'YES' if movenet_results['fall_detected'] else 'NO':>20} {'YES' if yolo_results['fall_detected'] else 'NO':>20}")
        
        # Determine winner
        print("\n" + "="*80)
        if movenet_probs.max() > yolo_probs.max():
            improvement = movenet_probs.max() / yolo_probs.max() if yolo_probs.max() > 0 else float('inf')
            print(f"🏆 WINNER: MoveNet ({improvement:.1f}× better)")
        elif yolo_probs.max() > movenet_probs.max():
            improvement = yolo_probs.max() / movenet_probs.max() if movenet_probs.max() > 0 else float('inf')
            print(f"🏆 WINNER: YOLO ({improvement:.1f}× better)")
        else:
            print("🤝 TIE: Both performed equally")
        print("="*80)


if __name__ == "__main__":
    main()

