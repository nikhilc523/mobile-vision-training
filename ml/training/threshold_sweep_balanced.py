"""
Phase 4.4 — Threshold Sweep for Balanced Model

Recompute optimal thresholds (balanced/safety/precision) for the Phase 4.2 balanced model.

Usage:
    python -m ml.training.threshold_sweep_balanced
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow import keras

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_test_data(data_path: str):
    """Load test data from balanced dataset."""
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    
    X = data['X']  # (N, 30, 34)
    y = data['y']  # (N,)
    
    # Split: 70/15/15 (subject-wise if video_ids available)
    if 'video_ids' in data:
        video_ids = data['video_ids']
        unique_videos = np.unique(video_ids)
        np.random.seed(42)
        np.random.shuffle(unique_videos)
        
        n_train = int(0.70 * len(unique_videos))
        n_val = int(0.15 * len(unique_videos))
        
        train_videos = unique_videos[:n_train]
        val_videos = unique_videos[n_train:n_train + n_val]
        test_videos = unique_videos[n_train + n_val:]
        
        test_mask = np.isin(video_ids, test_videos)
        X_test = X[test_mask]
        y_test = y[test_mask]
    else:
        # Fallback: simple split
        n_test = int(0.15 * len(X))
        X_test = X[-n_test:]
        y_test = y[-n_test:]
    
    print(f"Test set: {len(X_test)} samples")
    print(f"  Fall: {np.sum(y_test == 1)} ({100 * np.sum(y_test == 1) / len(y_test):.1f}%)")
    print(f"  Non-fall: {np.sum(y_test == 0)} ({100 * np.sum(y_test == 0) / len(y_test):.1f}%)")
    
    return X_test, y_test


def predict_probabilities(model, X_test):
    """Get model predictions."""
    print("\nGenerating predictions...")
    y_pred_proba = model.predict(X_test, batch_size=64, verbose=1)
    return y_pred_proba.flatten()


def threshold_sweep(y_true, y_pred_proba, thresholds):
    """Perform threshold sweep and compute metrics."""
    print(f"\nPerforming threshold sweep ({len(thresholds)} thresholds)...")
    
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Compute metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        results.append({
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        })
    
    return results


def find_optimal_thresholds(results):
    """Find optimal thresholds for different modes."""
    print("\nFinding optimal thresholds...")
    
    # 1. Balanced mode: Max F1
    max_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
    balanced_threshold = results[max_f1_idx]['threshold']
    balanced_metrics = results[max_f1_idx]
    
    print(f"\n1. BALANCED MODE (Max F1):")
    print(f"   Threshold: {balanced_threshold:.2f}")
    print(f"   F1: {balanced_metrics['f1']:.4f}")
    print(f"   Precision: {balanced_metrics['precision']:.4f}")
    print(f"   Recall: {balanced_metrics['recall']:.4f}")
    
    # 2. Safety mode: High recall (≥ 0.90 if possible)
    safety_candidates = [r for r in results if r['recall'] >= 0.90]
    if safety_candidates:
        # Among high-recall candidates, pick highest F1
        safety_idx = max(range(len(safety_candidates)), key=lambda i: safety_candidates[i]['f1'])
        safety_metrics = safety_candidates[safety_idx]
    else:
        # Fallback: highest recall
        safety_idx = max(range(len(results)), key=lambda i: results[i]['recall'])
        safety_metrics = results[safety_idx]
    
    safety_threshold = safety_metrics['threshold']
    
    print(f"\n2. SAFETY MODE (High Recall ≥ 0.90):")
    print(f"   Threshold: {safety_threshold:.2f}")
    print(f"   F1: {safety_metrics['f1']:.4f}")
    print(f"   Precision: {safety_metrics['precision']:.4f}")
    print(f"   Recall: {safety_metrics['recall']:.4f}")
    
    # 3. Precision mode: High precision (≥ 0.85 if possible)
    precision_candidates = [r for r in results if r['precision'] >= 0.85]
    if precision_candidates:
        # Among high-precision candidates, pick highest F1
        precision_idx = max(range(len(precision_candidates)), key=lambda i: precision_candidates[i]['f1'])
        precision_metrics = precision_candidates[precision_idx]
    else:
        # Fallback: highest precision
        precision_idx = max(range(len(results)), key=lambda i: results[i]['precision'])
        precision_metrics = results[precision_idx]
    
    precision_threshold = precision_metrics['threshold']
    
    print(f"\n3. PRECISION MODE (High Precision ≥ 0.85):")
    print(f"   Threshold: {precision_threshold:.2f}")
    print(f"   F1: {precision_metrics['f1']:.4f}")
    print(f"   Precision: {precision_metrics['precision']:.4f}")
    print(f"   Recall: {precision_metrics['recall']:.4f}")
    
    return {
        'balanced': {
            'threshold': balanced_threshold,
            'metrics': balanced_metrics
        },
        'safety': {
            'threshold': safety_threshold,
            'metrics': safety_metrics
        },
        'precision': {
            'threshold': precision_threshold,
            'metrics': precision_metrics
        }
    }


def plot_threshold_analysis(results, optimal_thresholds, output_path):
    """Create threshold analysis plot."""
    print(f"\nCreating threshold analysis plot...")
    
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Precision, Recall, F1 vs Threshold
    ax = axes[0, 0]
    ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    ax.plot(thresholds, f1s, 'g-', label='F1 Score', linewidth=2)
    
    # Mark optimal thresholds
    for mode, color in [('balanced', 'green'), ('safety', 'red'), ('precision', 'blue')]:
        t = optimal_thresholds[mode]['threshold']
        m = optimal_thresholds[mode]['metrics']
        ax.axvline(t, color=color, linestyle='--', alpha=0.5, label=f'{mode.capitalize()} ({t:.2f})')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score (zoomed)
    ax = axes[0, 1]
    ax.plot(thresholds, f1s, 'g-', linewidth=2)
    ax.axvline(optimal_thresholds['balanced']['threshold'], color='green', linestyle='--', linewidth=2)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'F1 Score (Max: {max(f1s):.4f} @ {optimal_thresholds["balanced"]["threshold"]:.2f})', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall Trade-off
    ax = axes[1, 0]
    ax.plot(recalls, precisions, 'purple', linewidth=2)
    
    # Mark optimal points
    for mode, color, marker in [('balanced', 'green', 'o'), ('safety', 'red', 's'), ('precision', 'blue', '^')]:
        m = optimal_thresholds[mode]['metrics']
        ax.scatter(m['recall'], m['precision'], color=color, s=200, marker=marker, 
                  label=f'{mode.capitalize()} ({m["recall"]:.3f}, {m["precision"]:.3f})', zorder=5)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix Counts
    ax = axes[1, 1]
    balanced_metrics = optimal_thresholds['balanced']['metrics']
    
    cm = np.array([[balanced_metrics['tn'], balanced_metrics['fp']],
                   [balanced_metrics['fn'], balanced_metrics['tp']]])
    
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Fall', 'Fall'])
    ax.set_yticklabels(['Non-Fall', 'Fall'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix (Balanced Mode, t={balanced_metrics["threshold"]:.2f})', 
                 fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=16, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("PHASE 4.4 — THRESHOLD SWEEP FOR BALANCED MODEL")
    print("="*70)
    
    # Paths
    data_path = 'data/processed/all_windows_30_raw_balanced.npz'
    model_path = 'ml/training/checkpoints/lstm_raw30_balanced_v2_best.h5'
    output_dir = Path('docs/wiki_assets/phase4_threshold_sweep')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1/5] Loading model...")
    model = keras.models.load_model(model_path, compile=False)
    print(f"✓ Model loaded: {model_path}")
    
    # Load test data
    print("\n[2/5] Loading test data...")
    X_test, y_test = load_test_data(data_path)
    
    # Get predictions
    print("\n[3/5] Generating predictions...")
    y_pred_proba = predict_probabilities(model, X_test)
    
    # Compute ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n✓ ROC-AUC: {roc_auc:.4f}")
    
    # Threshold sweep
    print("\n[4/5] Performing threshold sweep...")
    thresholds = np.arange(0.05, 0.96, 0.01)
    results = threshold_sweep(y_test, y_pred_proba, thresholds)
    
    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(results)
    
    # Save results
    print("\n[5/5] Saving results...")
    output_json = output_dir / 'deployment_thresholds_v2.json'
    with open(output_json, 'w') as f:
        json.dump({
            'model': model_path,
            'dataset': data_path,
            'test_samples': len(y_test),
            'roc_auc': float(roc_auc),
            'thresholds': optimal_thresholds,
            'all_results': results
        }, f, indent=2)
    print(f"✓ Results saved to: {output_json}")
    
    # Also save to checkpoints directory
    checkpoint_json = Path('ml/training/checkpoints/deployment_thresholds_v2.json')
    with open(checkpoint_json, 'w') as f:
        json.dump({
            'model': model_path,
            'roc_auc': float(roc_auc),
            'balanced': optimal_thresholds['balanced'],
            'safety': optimal_thresholds['safety'],
            'precision': optimal_thresholds['precision']
        }, f, indent=2)
    print(f"✓ Deployment config saved to: {checkpoint_json}")
    
    # Create plot
    plot_path = output_dir / 'threshold_analysis_v2.png'
    plot_threshold_analysis(results, optimal_thresholds, plot_path)
    
    print("\n" + "="*70)
    print("✅ THRESHOLD SWEEP COMPLETE")
    print("="*70)
    print(f"\nOptimal Thresholds:")
    print(f"  Balanced: {optimal_thresholds['balanced']['threshold']:.2f} (F1={optimal_thresholds['balanced']['metrics']['f1']:.4f})")
    print(f"  Safety:   {optimal_thresholds['safety']['threshold']:.2f} (Recall={optimal_thresholds['safety']['metrics']['recall']:.4f})")
    print(f"  Precision: {optimal_thresholds['precision']['threshold']:.2f} (Precision={optimal_thresholds['precision']['metrics']['precision']:.4f})")


if __name__ == '__main__':
    main()

