"""
Phase 4.5 — Test Ensemble (RAW + PHYSICS5) on secondfall.mp4

Test both models (raw keypoints and physics5) and optionally combine with ensemble.

Usage:
    python -m ml.inference.test_ensemble_secondfall
"""

import sys
import cv2
import numpy as np
import json
from pathlib import Path
from tensorflow import keras

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.pose.movenet_loader import load_movenet, infer_keypoints
from ml.inference.realtime_features_raw import RealtimeRawKeypointsExtractor
from ml.features.physics5_stream import Physics5FeatureExtractor


def test_ensemble_on_video(
    video_path: str,
    raw_model_path: str,
    phys_model_path: str,
    raw_threshold: float = 0.81,
    phys_threshold: float = 0.52,
    ensemble_weight: float = 0.5,
    window_size: int = 30
):
    """
    Test both models on a video and compute ensemble predictions.
    
    Args:
        video_path: Path to video file
        raw_model_path: Path to raw keypoints model
        phys_model_path: Path to physics5 model
        raw_threshold: Threshold for raw model
        phys_threshold: Threshold for physics5 model
        ensemble_weight: Weight for ensemble (0.5 = equal weight)
        window_size: Window size in frames
    """
    print("="*70)
    print("PHASE 4.5 — ENSEMBLE TESTING ON SECONDFALL.MP4")
    print("="*70)
    
    # Load models
    print("\n[1/5] Loading models...")
    movenet_model = load_movenet()
    raw_model = keras.models.load_model(raw_model_path, compile=False)
    phys_model = keras.models.load_model(phys_model_path, compile=False)
    print(f"✓ MoveNet loaded")
    print(f"✓ RAW model loaded: {raw_model_path}")
    print(f"✓ PHYS model loaded: {phys_model_path}")
    
    # Initialize extractors
    raw_extractor = RealtimeRawKeypointsExtractor()
    phys_extractor = Physics5FeatureExtractor(fps=30)
    
    # Open video
    print(f"\n[2/5] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✓ Video opened: {total_frames} frames @ {fps} FPS")
    
    # Extract keypoints from all frames
    print(f"\n[3/5] Extracting keypoints...")
    all_keypoints = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract keypoints
        keypoints = infer_keypoints(movenet_model, frame)
        all_keypoints.append(keypoints)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")
    
    cap.release()
    all_keypoints = np.array(all_keypoints)  # (T, 17, 3)
    print(f"✓ Keypoints extracted: {all_keypoints.shape}")
    
    # Extract features
    print(f"\n[4/5] Extracting features...")
    raw_features = []
    for kp in all_keypoints:
        feat = raw_extractor.extract(kp)
        raw_features.append(feat)
    raw_features = np.array(raw_features)  # (T, 34)
    
    phys_features = phys_extractor.extract_sequence(all_keypoints)  # (T, 5)
    
    print(f"✓ RAW features: {raw_features.shape}")
    print(f"✓ PHYS features: {phys_features.shape}")
    
    # Run inference with sliding windows
    print(f"\n[5/5] Running inference...")
    
    raw_probs = []
    phys_probs = []
    ensemble_probs = []
    frame_indices = []
    
    for i in range(len(all_keypoints) - window_size + 1):
        # RAW model
        raw_window = raw_features[i:i+window_size]  # (30, 34)
        X_raw = np.expand_dims(raw_window, axis=0)  # (1, 30, 34)
        prob_raw = raw_model.predict(X_raw, verbose=0)[0][0]
        
        # PHYS model
        phys_window = phys_features[i:i+window_size]  # (30, 5)
        X_phys = np.expand_dims(phys_window, axis=0)  # (1, 30, 5)
        prob_phys = phys_model.predict(X_phys, verbose=0)[0][0]
        
        # Ensemble (simple average)
        prob_ensemble = ensemble_weight * prob_raw + (1 - ensemble_weight) * prob_phys
        
        raw_probs.append(prob_raw)
        phys_probs.append(prob_phys)
        ensemble_probs.append(prob_ensemble)
        frame_indices.append(i + window_size // 2)
    
    raw_probs = np.array(raw_probs)
    phys_probs = np.array(phys_probs)
    ensemble_probs = np.array(ensemble_probs)
    frame_indices = np.array(frame_indices)
    
    print(f"✓ Inference complete: {len(raw_probs)} windows")
    
    # Analyze results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    
    print(f"\n1. RAW MODEL (threshold={raw_threshold:.2f}):")
    print(f"   Max prob: {raw_probs.max():.6f}")
    print(f"   Mean prob: {raw_probs.mean():.6f}")
    print(f"   Min prob: {raw_probs.min():.6f}")
    raw_detections = np.sum(raw_probs >= raw_threshold)
    print(f"   Fall windows: {raw_detections} / {len(raw_probs)} ({100*raw_detections/len(raw_probs):.1f}%)")
    print(f"   Decision: {'✅ FALL DETECTED' if raw_detections > 0 else '❌ NO FALL'}")
    
    print(f"\n2. PHYSICS5 MODEL (threshold={phys_threshold:.2f}):")
    print(f"   Max prob: {phys_probs.max():.6f}")
    print(f"   Mean prob: {phys_probs.mean():.6f}")
    print(f"   Min prob: {phys_probs.min():.6f}")
    phys_detections = np.sum(phys_probs >= phys_threshold)
    print(f"   Fall windows: {phys_detections} / {len(phys_probs)} ({100*phys_detections/len(phys_probs):.1f}%)")
    print(f"   Decision: {'✅ FALL DETECTED' if phys_detections > 0 else '❌ NO FALL'}")
    
    # Ensemble threshold (average of individual thresholds)
    ensemble_threshold = ensemble_weight * raw_threshold + (1 - ensemble_weight) * phys_threshold
    
    print(f"\n3. ENSEMBLE (weight={ensemble_weight:.2f}, threshold={ensemble_threshold:.2f}):")
    print(f"   Max prob: {ensemble_probs.max():.6f}")
    print(f"   Mean prob: {ensemble_probs.mean():.6f}")
    print(f"   Min prob: {ensemble_probs.min():.6f}")
    ensemble_detections = np.sum(ensemble_probs >= ensemble_threshold)
    print(f"   Fall windows: {ensemble_detections} / {len(ensemble_probs)} ({100*ensemble_detections/len(ensemble_probs):.1f}%)")
    print(f"   Decision: {'✅ FALL DETECTED' if ensemble_detections > 0 else '❌ NO FALL'}")
    
    # Save results
    output_dir = Path('outputs/ensemble_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'video': str(video_path),
        'total_frames': int(total_frames),
        'windows_processed': int(len(raw_probs)),
        'raw_model': {
            'threshold': float(raw_threshold),
            'max_prob': float(raw_probs.max()),
            'mean_prob': float(raw_probs.mean()),
            'min_prob': float(raw_probs.min()),
            'fall_windows': int(raw_detections),
            'detection_rate': float(raw_detections / len(raw_probs)),
            'decision': 'FALL' if raw_detections > 0 else 'NO_FALL'
        },
        'phys_model': {
            'threshold': float(phys_threshold),
            'max_prob': float(phys_probs.max()),
            'mean_prob': float(phys_probs.mean()),
            'min_prob': float(phys_probs.min()),
            'fall_windows': int(phys_detections),
            'detection_rate': float(phys_detections / len(phys_probs)),
            'decision': 'FALL' if phys_detections > 0 else 'NO_FALL'
        },
        'ensemble': {
            'weight': float(ensemble_weight),
            'threshold': float(ensemble_threshold),
            'max_prob': float(ensemble_probs.max()),
            'mean_prob': float(ensemble_probs.mean()),
            'min_prob': float(ensemble_probs.min()),
            'fall_windows': int(ensemble_detections),
            'detection_rate': float(ensemble_detections / len(ensemble_probs)),
            'decision': 'FALL' if ensemble_detections > 0 else 'NO_FALL'
        },
        'per_frame_probs': {
            'frame_indices': frame_indices.tolist(),
            'raw_probs': raw_probs.tolist(),
            'phys_probs': phys_probs.tolist(),
            'ensemble_probs': ensemble_probs.tolist()
        }
    }
    
    results_path = output_dir / 'ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: RAW model
    ax = axes[0]
    ax.plot(frame_indices, raw_probs, 'b-', linewidth=2, label='RAW Probability')
    ax.axhline(raw_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({raw_threshold:.2f})')
    ax.fill_between(frame_indices, 0, raw_probs, where=(raw_probs >= raw_threshold), 
                     color='red', alpha=0.3, label='Fall Detected')
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('RAW Model (34 features)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 2: PHYS model
    ax = axes[1]
    ax.plot(frame_indices, phys_probs, 'g-', linewidth=2, label='PHYS Probability')
    ax.axhline(phys_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({phys_threshold:.2f})')
    ax.fill_between(frame_indices, 0, phys_probs, where=(phys_probs >= phys_threshold), 
                     color='red', alpha=0.3, label='Fall Detected')
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('PHYSICS5 Model (5 features)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 3: Ensemble
    ax = axes[2]
    ax.plot(frame_indices, ensemble_probs, 'purple', linewidth=2, label='Ensemble Probability')
    ax.axhline(ensemble_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({ensemble_threshold:.2f})')
    ax.fill_between(frame_indices, 0, ensemble_probs, where=(ensemble_probs >= ensemble_threshold), 
                     color='red', alpha=0.3, label='Fall Detected')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Ensemble (weight={ensemble_weight:.2f})', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plot_path = output_dir / 'ensemble_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"✅ ENSEMBLE TEST COMPLETE")
    print(f"{'='*70}")
    
    return results


def main():
    results = test_ensemble_on_video(
        video_path='data/test/secondfall.mp4',
        raw_model_path='ml/training/checkpoints/lstm_raw30_balanced_v2_best.h5',
        phys_model_path='ml/training/checkpoints/lstm_phys5_best.h5',
        raw_threshold=0.81,
        phys_threshold=0.52,
        ensemble_weight=0.5,
        window_size=30
    )
    
    print(f"\n📊 SUMMARY:")
    print(f"  RAW Model: {results['raw_model']['decision']}")
    print(f"  PHYS Model: {results['phys_model']['decision']}")
    print(f"  Ensemble: {results['ensemble']['decision']}")


if __name__ == '__main__':
    main()

