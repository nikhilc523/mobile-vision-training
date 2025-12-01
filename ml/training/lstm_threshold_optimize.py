"""
BiLSTM Threshold Optimization for Deployment - Phase 2.5

Evaluate the existing best BiLSTM model across multiple thresholds
to find optimal settings for different deployment modes without fine-tuning.

Target: Use Phase 2.3a model (F1=0.7456, ROC-AUC=0.9360)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import argparse
import json
from datetime import datetime
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import sys

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.training.lstm_train_full import (
    SigmoidFocalCrossEntropy,
    F1Metric
)
from ml.training.lstm_train_optimized import (
    build_optimized_bilstm_model,
    subject_wise_split
)


def load_data(data_path: str):
    """Load dataset."""
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    
    X = data['X']
    y = data['y']
    video_ids = data['video_ids']
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Fall samples: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"Non-fall samples: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    
    return X, y, video_ids


def evaluate_threshold_range(y_true, y_pred_proba, thresholds):
    """Evaluate model performance across multiple thresholds."""
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        })
    
    return results


def find_deployment_thresholds(y_true, y_pred_proba, thresholds):
    """Find optimal thresholds for different deployment modes."""
    results = evaluate_threshold_range(y_true, y_pred_proba, thresholds)
    
    # Find F1-optimal (balanced mode)
    f1_optimal = max(results, key=lambda x: x['f1'])
    
    # Find recall-optimal (safety mode - minimize false negatives)
    recall_optimal = max(results, key=lambda x: x['recall'])
    
    # Find precision-optimal (precision mode - minimize false positives)
    precision_optimal = max(results, key=lambda x: x['precision'])
    
    return {
        'balanced': f1_optimal,
        'safety': recall_optimal,
        'precision': precision_optimal,
        'all_results': results
    }


def save_threshold_config(thresholds, output_path):
    """Save threshold configuration for Android deployment."""
    config = {
        'model_version': '2.5',
        'model_source': 'Phase 2.3a Optimized BiLSTM (no fine-tuning)',
        'date': datetime.utcnow().isoformat(),
        'thresholds': {
            'balanced': {
                'value': float(thresholds['balanced']['threshold']),
                'description': 'F1-optimal threshold for balanced precision/recall',
                'metrics': {
                    'f1': float(thresholds['balanced']['f1']),
                    'precision': float(thresholds['balanced']['precision']),
                    'recall': float(thresholds['balanced']['recall'])
                }
            },
            'safety': {
                'value': float(thresholds['safety']['threshold']),
                'description': 'Recall-optimal threshold for maximum fall detection (safety-critical)',
                'metrics': {
                    'f1': float(thresholds['safety']['f1']),
                    'precision': float(thresholds['safety']['precision']),
                    'recall': float(thresholds['safety']['recall'])
                }
            },
            'precision': {
                'value': float(thresholds['precision']['threshold']),
                'description': 'Precision-optimal threshold for minimum false alarms',
                'metrics': {
                    'f1': float(thresholds['precision']['f1']),
                    'precision': float(thresholds['precision']['precision']),
                    'recall': float(thresholds['precision']['recall'])
                }
            }
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Threshold configuration saved to {output_path}")
    return config


def plot_threshold_analysis(results, output_dir):
    """Plot threshold analysis curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds = [r['threshold'] for r in results]
    f1_scores = [r['f1'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score', marker='o')
    plt.plot(thresholds, precisions, 'g-', linewidth=2, label='Precision', marker='s')
    plt.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall', marker='^')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Threshold Analysis for Deployment Modes', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(min(thresholds), max(thresholds))
    plt.ylim(0, 1)
    
    save_path = output_dir / 'threshold_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved threshold analysis: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 2.5 - BiLSTM Threshold Optimization')
    parser.add_argument('--data', type=str, default='data/processed/all_windows_60frame.npz')
    parser.add_argument('--model', type=str, default='ml/training/checkpoints/lstm_bilstm_opt_best.h5')
    args = parser.parse_args()
    
    print("="*70)
    print("PHASE 2.5 — BiLSTM THRESHOLD OPTIMIZATION FOR DEPLOYMENT")
    print("="*70)
    
    # Load data
    X, y, video_ids = load_data(args.data)
    
    # Subject-wise split
    _, X_test, _, y_test = subject_wise_split(X, y, video_ids)
    
    # Load existing model (Phase 2.3a)
    print(f"\nLoading model from {args.model}...")
    model_path = Path(args.model)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Rebuild model architecture and load weights
    input_shape = (X_test.shape[1], X_test.shape[2])
    model = build_optimized_bilstm_model(input_shape)
    model.load_weights(args.model)
    
    print("✅ Model loaded successfully (Phase 2.3a Optimized BiLSTM)")
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("Threshold Optimization for Deployment")
    print("="*70)
    
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    
    # Evaluate thresholds from 0.40 to 0.65 (step 0.05)
    thresholds = np.arange(0.40, 0.66, 0.05)
    deployment_thresholds = find_deployment_thresholds(y_test, y_pred_proba, thresholds)
    
    # Print results
    print("\n" + "="*70)
    print("DEPLOYMENT THRESHOLD RECOMMENDATIONS")
    print("="*70)
    
    for mode, result in [('balanced', 'Balanced Mode (F1-optimal)'), 
                         ('safety', 'Safety Mode (Recall-optimal)'),
                         ('precision', 'Precision Mode (Precision-optimal)')]:
        metrics = deployment_thresholds[mode]
        print(f"\n{result}:")
        print(f"  Threshold: {metrics['threshold']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  TP: {metrics['tp']:4d} | FP: {metrics['fp']:4d} | TN: {metrics['tn']:4d} | FN: {metrics['fn']:4d}")
    
    # Compute overall metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"\nOverall Metrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    
    # Save threshold configuration
    config_path = 'ml/training/checkpoints/deployment_thresholds.json'
    save_threshold_config(deployment_thresholds, config_path)
    
    # Plot threshold analysis
    output_dir = Path('docs/wiki_assets/phase2_final_deployment')
    plot_threshold_analysis(deployment_thresholds['all_results'], output_dir)
    
    print("\n" + "="*70)
    print("✅ PHASE 2.5 COMPLETE")
    print("="*70)
    print(f"\nModel: {args.model} (Phase 2.3a - no fine-tuning)")
    print(f"Threshold config: {config_path}")
    print(f"Visualizations: {output_dir}")


if __name__ == '__main__':
    main()

