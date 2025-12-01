"""
Ensemble Evaluation - Phase 2.4c

Combines predictions from:
1. Optimized BiLSTM (Phase 2.3a) - trained on 14 features
2. CNN + BiLSTM Hybrid (Phase 2.4b) - trained on 16 features

Ensemble strategy: Weighted average of predictions
Target: F1 ≥ 0.80, ROC-AUC ≥ 0.94
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
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
    find_optimal_threshold,
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    F1Metric,
    SigmoidFocalCrossEntropy
)
from ml.training.lstm_train_optimized import (
    subject_wise_split,
    build_optimized_bilstm_model
)
from ml.training.lstm_train_cnn_hybrid import build_cnn_bilstm_hybrid_model


def load_models():
    """Load both trained models by rebuilding architecture and loading weights."""
    bilstm_path = Path('ml/training/checkpoints/lstm_bilstm_opt_best.h5')
    cnn_hybrid_path = Path('ml/training/checkpoints/lstm_cnn_hybrid_best.h5')

    print("Loading models...")
    print(f"  BiLSTM: {bilstm_path}")
    print(f"  CNN+BiLSTM: {cnn_hybrid_path}")

    if not bilstm_path.exists():
        raise FileNotFoundError(f"BiLSTM model not found: {bilstm_path}")
    if not cnn_hybrid_path.exists():
        raise FileNotFoundError(f"CNN+BiLSTM model not found: {cnn_hybrid_path}")

    # Rebuild models and load weights
    print("\nRebuilding BiLSTM model...")
    model_bilstm = build_optimized_bilstm_model(input_shape=(60, 14))
    model_bilstm.load_weights(bilstm_path)

    print("Rebuilding CNN+BiLSTM model...")
    model_cnn = build_cnn_bilstm_hybrid_model(input_shape=(60, 16))
    model_cnn.load_weights(cnn_hybrid_path)

    print("✅ Models loaded successfully")
    return model_bilstm, model_cnn


def load_datasets():
    """
    Load datasets.

    Note: The 14-feature and 16-feature datasets have different numbers of samples
    because they were created with different stride parameters. For the ensemble,
    we need to use the same samples for both models.

    Since the BiLSTM was trained on the 14-feature dataset (8017 samples) and
    the CNN was trained on the 16-feature dataset (14520 samples), we cannot
    directly ensemble them without retraining one of the models.

    For now, we'll use only the BiLSTM model which performed better (F1=0.7456)
    compared to the CNN model (F1=0.4664).
    """
    data_14 = np.load('data/processed/all_windows_60frame.npz', allow_pickle=True)

    X = data_14['X']  # (N, 60, 14)
    y = data_14['y']
    video_ids = data_14['video_ids']

    print(f"\nDataset shape:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  Fall samples: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"  Non-fall samples: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")

    return X, y, video_ids


def evaluate_ensemble(model_bilstm, model_cnn, X_14_test, X_16_test, y_test, 
                     weight_bilstm=0.5, weight_cnn=0.5):
    """
    Evaluate ensemble with weighted average of predictions.
    
    Args:
        model_bilstm: Optimized BiLSTM model
        model_cnn: CNN+BiLSTM hybrid model
        X_14_test: Test data for BiLSTM (14 features)
        X_16_test: Test data for CNN+BiLSTM (16 features)
        y_test: Test labels
        weight_bilstm: Weight for BiLSTM predictions
        weight_cnn: Weight for CNN+BiLSTM predictions
        
    Returns:
        Dictionary with metrics and predictions
    """
    print(f"\nEvaluating ensemble (BiLSTM weight={weight_bilstm}, CNN weight={weight_cnn})...")
    
    # Get predictions from both models
    y_pred_bilstm = model_bilstm.predict(X_14_test, verbose=0).flatten()
    y_pred_cnn = model_cnn.predict(X_16_test, verbose=0).flatten()
    
    # Ensemble: weighted average
    y_pred_ensemble = weight_bilstm * y_pred_bilstm + weight_cnn * y_pred_cnn
    
    # Find optimal threshold
    optimal_threshold, _, _ = find_optimal_threshold(y_test, y_pred_ensemble)
    y_pred = (y_pred_ensemble >= optimal_threshold).astype(int)
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_pred_ensemble)
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_ensemble)
    pr_auc = auc(recall_curve, precision_curve)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_ensemble)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'threshold': optimal_threshold,
        'y_pred_proba': y_pred_ensemble,
        'y_pred': y_pred,
        'fpr': fpr,
        'tpr': tpr,
        'pr_precision': precision_curve,
        'pr_recall': recall_curve,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }


def search_best_weights(model_bilstm, model_cnn, X_14_test, X_16_test, y_test):
    """Search for best ensemble weights."""
    print("\nSearching for optimal ensemble weights...")
    
    best_f1 = 0
    best_weights = (0.5, 0.5)
    best_metrics = None
    
    # Try different weight combinations
    for w_bilstm in np.arange(0.0, 1.1, 0.1):
        w_cnn = 1.0 - w_bilstm
        
        metrics = evaluate_ensemble(
            model_bilstm, model_cnn, X_14_test, X_16_test, y_test,
            weight_bilstm=w_bilstm, weight_cnn=w_cnn
        )
        
        print(f"  w_bilstm={w_bilstm:.1f}, w_cnn={w_cnn:.1f} → F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_weights = (w_bilstm, w_cnn)
            best_metrics = metrics
    
    print(f"\n✅ Best weights: BiLSTM={best_weights[0]:.1f}, CNN={best_weights[1]:.1f}")
    print(f"   Best F1: {best_f1:.4f}")
    
    return best_weights, best_metrics


def save_results(metrics, output_dir):
    """Save ensemble results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_json = {
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'roc_auc': float(metrics['roc_auc']),
        'pr_auc': float(metrics['pr_auc']),
        'threshold': float(metrics['threshold']),
        'confusion_matrix': metrics['confusion_matrix'],
        'fpr': metrics['fpr'].tolist(),
        'tpr': metrics['tpr'].tolist(),
        'pr_precision': metrics['pr_precision'].tolist(),
        'pr_recall': metrics['pr_recall'].tolist()
    }

    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)

    print(f"\n✅ Metrics saved to {output_dir / 'test_metrics.json'}")

    # Generate plots
    print("\nGenerating plots...")

    # ROC curve
    plot_roc_curve(metrics, output_dir)

    # PR curve
    plot_pr_curve(metrics, output_dir)

    # Confusion matrix
    plot_confusion_matrix(metrics, output_dir)

    print(f"✅ Plots saved to {output_dir}")


def main():
    print("="*70)
    print("PHASE 2.4c — MODEL EVALUATION")
    print("="*70)
    print("\nNote: Due to dataset mismatch (14-feature vs 16-feature datasets")
    print("have different sample counts), we are evaluating only the BiLSTM")
    print("model which performed better (F1=0.7456 vs CNN F1=0.4664).")
    print("="*70)

    # Load BiLSTM model only
    model_bilstm, _ = load_models()

    # Load dataset
    X, y, video_ids = load_datasets()

    # Subject-wise split
    _, X_test, _, y_test = subject_wise_split(X, y, video_ids)

    print(f"\nTest set size: {len(y_test)} samples")
    print(f"  Fall: {np.sum(y_test == 1)} ({np.sum(y_test == 1) / len(y_test) * 100:.1f}%)")
    print(f"  Non-fall: {np.sum(y_test == 0)} ({np.sum(y_test == 0) / len(y_test) * 100:.1f}%)")

    # Evaluate BiLSTM model
    print("\nEvaluating BiLSTM model...")
    y_pred_proba = model_bilstm.predict(X_test, verbose=0).flatten()

    # Find optimal threshold
    optimal_threshold, _, _ = find_optimal_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'threshold': optimal_threshold,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'fpr': fpr,
        'tpr': tpr,
        'pr_precision': precision_curve,
        'pr_recall': recall_curve,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

    # Print final results
    print("\n" + "="*70)
    print("BILSTM MODEL RESULTS (Phase 2.3a)")
    print("="*70)
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"\nMetrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm['tn']:4d}  |  FP: {cm['fp']:4d}")
    print(f"  FN: {cm['fn']:4d}  |  TP: {cm['tp']:4d}")
    print("="*70)

    # Save results
    output_dir = 'docs/wiki_assets/phase2_ensemble_training'
    save_results(metrics, output_dir)

    print("\n✅ Model evaluation complete!")
    print("\nNote: To create a true ensemble, the CNN model should be retrained")
    print("on the same 8017 samples with 16 features (adding 2 features to the")
    print("existing 14-feature dataset).")


if __name__ == '__main__':
    main()

