"""
Test Phase 4.2 balanced model on URFD fall video keypoints.

Usage:
    python -m ml.inference.test_on_urfd_keypoints
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.inference.realtime_features_raw import RealtimeRawKeypointsExtractor


def test_on_keypoints(keypoints_path: str, model_path: str, threshold: float = 0.81):
    """
    Test model on pre-extracted keypoints.
    
    Args:
        keypoints_path: Path to .npz file with keypoints
        model_path: Path to trained model
        threshold: Detection threshold
    """
    print("="*70)
    print("TESTING PHASE 4.2 MODEL ON URFD FALL VIDEO")
    print("="*70)
    
    # Load keypoints
    print(f"\n[1/4] Loading keypoints from: {keypoints_path}")
    data = np.load(keypoints_path)
    keypoints = data['keypoints']  # (T, 17, 3)
    label = data['label']
    fps = data['fps']
    video_name = data['video_name']
    
    print(f"✓ Keypoints loaded")
    print(f"  Video: {video_name}")
    print(f"  Frames: {len(keypoints)}")
    print(f"  FPS: {fps}")
    print(f"  Label: {'FALL' if label == 1 else 'NON-FALL'}")
    
    # Load model
    print(f"\n[2/4] Loading model from: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    print(f"✓ Model loaded")
    print(f"  Input shape: {model.input_shape}")
    
    # Extract features
    print(f"\n[3/4] Extracting raw keypoint features...")
    extractor = RealtimeRawKeypointsExtractor()
    features = []
    
    for i, kp in enumerate(keypoints):
        feat = extractor.extract(kp)
        features.append(feat)
    
    features = np.array(features)  # (T, 34)
    print(f"✓ Features extracted: {features.shape}")
    
    # Create sliding windows (30 frames, stride 1)
    print(f"\n[4/4] Running inference with sliding windows...")
    window_size = 30
    stride = 1
    
    probabilities = []
    frame_indices = []
    
    for i in range(0, len(features) - window_size + 1, stride):
        window = features[i:i+window_size]  # (30, 34)
        X = np.expand_dims(window, axis=0)  # (1, 30, 34)
        
        prob = model.predict(X, verbose=0)[0][0]
        probabilities.append(prob)
        frame_indices.append(i + window_size // 2)  # Center of window
    
    probabilities = np.array(probabilities)
    frame_indices = np.array(frame_indices)
    
    print(f"✓ Inference complete")
    print(f"  Windows processed: {len(probabilities)}")
    print(f"  Max probability: {probabilities.max():.6f}")
    print(f"  Mean probability: {probabilities.mean():.6f}")
    print(f"  Min probability: {probabilities.min():.6f}")
    
    # Detect falls
    fall_detections = probabilities >= threshold
    num_falls = np.sum(fall_detections)
    
    print(f"\n{'='*70}")
    print(f"DETECTION RESULTS (threshold={threshold:.2f})")
    print(f"{'='*70}")
    print(f"Fall windows detected: {num_falls} / {len(probabilities)} ({100*num_falls/len(probabilities):.1f}%)")
    
    if num_falls > 0:
        fall_frames = frame_indices[fall_detections]
        fall_probs = probabilities[fall_detections]
        print(f"\n✅ FALL DETECTED!")
        print(f"  First detection: Frame {fall_frames[0]} (prob={fall_probs[0]:.4f})")
        print(f"  Peak detection: Frame {fall_frames[np.argmax(fall_probs)]} (prob={fall_probs.max():.4f})")
        print(f"  Last detection: Frame {fall_frames[-1]} (prob={fall_probs[-1]:.4f})")
    else:
        print(f"\n❌ NO FALL DETECTED")
        print(f"  Max probability: {probabilities.max():.4f} (below threshold {threshold:.2f})")
    
    # Plot results
    print(f"\n[5/5] Creating visualization...")
    output_dir = Path('outputs/urfd_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Probability over time
    ax = axes[0]
    ax.plot(frame_indices, probabilities, 'b-', linewidth=2, label='Fall Probability')
    ax.axhline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    ax.fill_between(frame_indices, 0, probabilities, where=(probabilities >= threshold), 
                     color='red', alpha=0.3, label='Fall Detected')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Fall Detection: {video_name} (Label: {"FALL" if label == 1 else "NON-FALL"})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 2: Probability histogram
    ax = axes[1]
    ax.hist(probabilities, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    ax.axvline(probabilities.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean ({probabilities.mean():.3f})')
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f'{video_name}_detection.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    plt.close()
    
    # Save detailed results
    results = {
        'video_name': str(video_name),
        'label': int(label),
        'total_frames': int(len(keypoints)),
        'windows_processed': int(len(probabilities)),
        'threshold': float(threshold),
        'max_probability': float(probabilities.max()),
        'mean_probability': float(probabilities.mean()),
        'min_probability': float(probabilities.min()),
        'fall_windows_detected': int(num_falls),
        'detection_rate': float(num_falls / len(probabilities)),
        'fall_detected': bool(num_falls > 0)
    }
    
    import json
    results_path = output_dir / f'{video_name}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ TEST COMPLETE")
    print(f"{'='*70}")
    
    return results


def main():
    # Test on URFD fall video
    keypoints_path = 'data/interim/keypoints/urfd_fall_fall-01-cam0-rgb.npz'
    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_v2_best.h5'
    threshold = 0.81  # Optimal threshold from Phase 4.4
    
    results = test_on_keypoints(keypoints_path, model_path, threshold)
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"  Video: {results['video_name']}")
    print(f"  Ground Truth: {'FALL' if results['label'] == 1 else 'NON-FALL'}")
    print(f"  Prediction: {'FALL DETECTED ✅' if results['fall_detected'] else 'NO FALL ❌'}")
    print(f"  Max Probability: {results['max_probability']:.4f}")
    print(f"  Detection Rate: {100*results['detection_rate']:.1f}% of windows")


if __name__ == '__main__':
    main()

