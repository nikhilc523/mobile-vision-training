"""
Phase 4.6 — Hard Negative Mining

Mine false positives from UCF101 non-fall clips to improve model robustness.

Usage:
    python -m ml.training.mine_hard_negatives
"""

import sys
import numpy as np
import json
from pathlib import Path
from tensorflow import keras
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.inference.realtime_features_raw import RealtimeRawKeypointsExtractor


def load_ucf101_non_fall_keypoints(keypoints_dir: str):
    """Load UCF101 non-fall keypoint files."""
    keypoints_path = Path(keypoints_dir)
    
    # UCF101 non-fall activities (exclude FallFloor)
    non_fall_patterns = [
        'ucf101_Basketball',
        'ucf101_Biking',
        'ucf101_Diving',
        'ucf101_GolfSwing',
        'ucf101_HorseRiding',
        'ucf101_IceDancing',
        'ucf101_RopeClimbing',
        'ucf101_SalsaSpin',
        'ucf101_SkateBoarding',
        'ucf101_Skiing',
        'ucf101_SoccerJuggling',
        'ucf101_Surfing',
        'ucf101_TrampolineJumping',
        'ucf101_VolleyballSpiking',
        'ucf101_WalkingWithDog'
    ]
    
    non_fall_files = []
    for pattern in non_fall_patterns:
        non_fall_files.extend(list(keypoints_path.glob(f'{pattern}*.npz')))
    
    print(f"Found {len(non_fall_files)} UCF101 non-fall keypoint files")
    
    return non_fall_files


def extract_windows_from_video(keypoints: np.ndarray, window_size: int = 30, stride: int = 10):
    """Extract sliding windows from keypoints."""
    extractor = RealtimeRawKeypointsExtractor()
    
    # Extract raw features
    features = []
    for kp in keypoints:
        feat = extractor.extract(kp)
        features.append(feat)
    features = np.array(features)  # (T, 34)
    
    # Create sliding windows
    windows = []
    for i in range(0, len(features) - window_size + 1, stride):
        window = features[i:i+window_size]  # (30, 34)
        windows.append(window)
    
    return np.array(windows)  # (N, 30, 34)


def mine_hard_negatives(
    model_path: str,
    keypoints_dir: str,
    threshold: float = 0.81,
    top_k: int = 500,
    window_size: int = 30,
    stride: int = 10
):
    """
    Mine hard negative examples (false positives) from UCF101 non-fall clips.
    
    Args:
        model_path: Path to trained model
        keypoints_dir: Directory with keypoint files
        threshold: Detection threshold
        top_k: Number of top false positives to keep
        window_size: Window size in frames
        stride: Stride for sliding window
    
    Returns:
        hard_negatives: (K, 30, 34) array of hard negative windows
        metadata: List of dicts with video name and window index
    """
    print("="*70)
    print("PHASE 4.6 — HARD NEGATIVE MINING")
    print("="*70)
    
    # Load model
    print(f"\n[1/4] Loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    print("✓ Model loaded")
    
    # Load UCF101 non-fall files
    print(f"\n[2/4] Loading UCF101 non-fall keypoints...")
    non_fall_files = load_ucf101_non_fall_keypoints(keypoints_dir)
    
    if len(non_fall_files) == 0:
        print("⚠️  No UCF101 non-fall files found. Using all non-fall files instead.")
        non_fall_files = [f for f in Path(keypoints_dir).glob('*.npz') 
                         if 'fall' not in f.stem.lower()]
        print(f"Found {len(non_fall_files)} non-fall files")
    
    # Extract windows and predict
    print(f"\n[3/4] Extracting windows and running inference...")
    all_windows = []
    all_probs = []
    all_metadata = []
    
    for kp_file in tqdm(non_fall_files, desc="Processing videos"):
        # Load keypoints
        data = np.load(kp_file)
        keypoints = data['keypoints']  # (T, 17, 3)
        video_name = kp_file.stem
        
        # Extract windows
        windows = extract_windows_from_video(keypoints, window_size, stride)
        
        if len(windows) == 0:
            continue
        
        # Predict
        probs = model.predict(windows, batch_size=64, verbose=0).flatten()
        
        # Store windows and metadata
        for i, (window, prob) in enumerate(zip(windows, probs)):
            all_windows.append(window)
            all_probs.append(prob)
            all_metadata.append({
                'video_name': video_name,
                'window_index': i,
                'probability': float(prob)
            })
    
    all_windows = np.array(all_windows)  # (N, 30, 34)
    all_probs = np.array(all_probs)  # (N,)
    
    print(f"✓ Processed {len(non_fall_files)} videos")
    print(f"✓ Total windows: {len(all_windows)}")
    print(f"✓ Max probability: {all_probs.max():.6f}")
    print(f"✓ Mean probability: {all_probs.mean():.6f}")
    
    # Find false positives (prob >= threshold)
    false_positives = all_probs >= threshold
    num_fp = np.sum(false_positives)
    
    print(f"\n✓ False positives (prob >= {threshold}): {num_fp} / {len(all_probs)} ({100*num_fp/len(all_probs):.2f}%)")
    
    # Select top-k hardest negatives (highest probabilities)
    print(f"\n[4/4] Selecting top-{top_k} hard negatives...")
    top_k_indices = np.argsort(all_probs)[-top_k:]  # Highest probabilities
    
    hard_negatives = all_windows[top_k_indices]
    hard_neg_probs = all_probs[top_k_indices]
    hard_neg_metadata = [all_metadata[i] for i in top_k_indices]
    
    print(f"✓ Selected {len(hard_negatives)} hard negatives")
    print(f"✓ Probability range: [{hard_neg_probs.min():.6f}, {hard_neg_probs.max():.6f}]")
    print(f"✓ Mean probability: {hard_neg_probs.mean():.6f}")
    
    # Save metadata
    output_dir = Path('docs/wiki_assets/phase4_hardneg')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'hard_negatives_metadata.json', 'w') as f:
        json.dump({
            'total_windows': int(len(all_windows)),
            'false_positives': int(num_fp),
            'false_positive_rate': float(num_fp / len(all_probs)),
            'top_k': int(top_k),
            'threshold': float(threshold),
            'hard_negatives': hard_neg_metadata
        }, f, indent=2)
    
    print(f"\n✓ Metadata saved to: {output_dir / 'hard_negatives_metadata.json'}")
    
    return hard_negatives, hard_neg_metadata


def create_hnm_dataset(
    original_dataset_path: str,
    hard_negatives: np.ndarray,
    output_path: str
):
    """
    Create new dataset with hard negatives added to negative pool.
    
    Args:
        original_dataset_path: Path to original balanced dataset
        hard_negatives: (K, 30, 34) array of hard negative windows
        output_path: Output path for new dataset
    """
    print(f"\n{'='*70}")
    print("CREATING HNM DATASET")
    print(f"{'='*70}")
    
    # Load original dataset
    print(f"\n[1/2] Loading original dataset: {original_dataset_path}")
    data = np.load(original_dataset_path)
    X_orig = data['X']  # (N, 30, 34)
    y_orig = data['y']  # (N,)
    video_ids_orig = data['video_ids'] if 'video_ids' in data else None
    
    print(f"✓ Original dataset loaded")
    print(f"  X shape: {X_orig.shape}")
    print(f"  Fall: {np.sum(y_orig == 1)} ({100*np.sum(y_orig == 1)/len(y_orig):.1f}%)")
    print(f"  Non-fall: {np.sum(y_orig == 0)} ({100*np.sum(y_orig == 0)/len(y_orig):.1f}%)")
    
    # Add hard negatives to dataset
    print(f"\n[2/2] Adding {len(hard_negatives)} hard negatives...")
    X_new = np.vstack([X_orig, hard_negatives])
    y_new = np.hstack([y_orig, np.zeros(len(hard_negatives))])
    
    # Create video IDs for hard negatives
    if video_ids_orig is not None:
        hnm_video_ids = np.array([f'hnm_{i}' for i in range(len(hard_negatives))])
        video_ids_new = np.hstack([video_ids_orig, hnm_video_ids])
    else:
        video_ids_new = None
    
    print(f"✓ New dataset created")
    print(f"  X shape: {X_new.shape}")
    print(f"  Fall: {np.sum(y_new == 1)} ({100*np.sum(y_new == 1)/len(y_new):.1f}%)")
    print(f"  Non-fall: {np.sum(y_new == 0)} ({100*np.sum(y_new == 0)/len(y_new):.1f}%)")
    print(f"  Imbalance ratio: 1:{np.sum(y_new == 0) / np.sum(y_new == 1):.2f}")
    
    # Save new dataset
    if video_ids_new is not None:
        np.savez_compressed(output_path, X=X_new, y=y_new, video_ids=video_ids_new)
    else:
        np.savez_compressed(output_path, X=X_new, y=y_new)
    
    print(f"\n✓ HNM dataset saved to: {output_path}")
    
    return X_new, y_new


def main():
    # Mine hard negatives
    hard_negatives, metadata = mine_hard_negatives(
        model_path='ml/training/checkpoints/lstm_raw30_balanced_v2_best.h5',
        keypoints_dir='data/interim/keypoints',
        threshold=0.81,
        top_k=500,
        window_size=30,
        stride=10
    )
    
    # Create HNM dataset
    X_new, y_new = create_hnm_dataset(
        original_dataset_path='data/processed/all_windows_30_raw_balanced.npz',
        hard_negatives=hard_negatives,
        output_path='data/processed/all_windows_30_raw_balanced_hnm.npz'
    )
    
    print(f"\n{'='*70}")
    print("✅ HARD NEGATIVE MINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext step: Retrain model on HNM dataset")
    print(f"  python -m ml.training.lstm_train_raw_balanced_hnm")


if __name__ == '__main__':
    main()

