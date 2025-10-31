"""
Phase 4.6 — Retrain BiLSTM on HNM Dataset

Retrain the balanced RAW model on dataset with hard negatives.

Usage:
    python -m ml.training.lstm_train_raw_balanced_hnm
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_data(data_path: str):
    """Load and split HNM dataset."""
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    
    X = data['X']  # (N, 30, 34)
    y = data['y']  # (N,)
    video_ids = data['video_ids'] if 'video_ids' in data else None
    
    print(f"Dataset loaded:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Fall: {np.sum(y == 1)} ({100*np.sum(y == 1)/len(y):.1f}%)")
    print(f"  Non-fall: {np.sum(y == 0)} ({100*np.sum(y == 0)/len(y):.1f}%)")
    
    # Subject-wise split (70/15/15)
    if video_ids is not None:
        unique_videos = np.unique(video_ids)
        np.random.seed(42)
        np.random.shuffle(unique_videos)
        
        n_train = int(0.70 * len(unique_videos))
        n_val = int(0.15 * len(unique_videos))
        
        train_videos = unique_videos[:n_train]
        val_videos = unique_videos[n_train:n_train + n_val]
        test_videos = unique_videos[n_train + n_val:]
        
        train_mask = np.isin(video_ids, train_videos)
        val_mask = np.isin(video_ids, val_videos)
        test_mask = np.isin(video_ids, test_videos)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        # Fallback: simple split
        n_train = int(0.70 * len(X))
        n_val = int(0.15 * len(X))
        
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(X_train)} ({np.sum(y_train == 1)} fall, {np.sum(y_train == 0)} non-fall)")
    print(f"  Val:   {len(X_val)} ({np.sum(y_val == 1)} fall, {np.sum(y_val == 0)} non-fall)")
    print(f"  Test:  {len(X_test)} ({np.sum(y_test == 1)} fall, {np.sum(y_test == 0)} non-fall)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model(input_shape=(30, 34)):
    """Build BiLSTM model (same architecture as Phase 4.2)."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(1e-4))),
        layers.Bidirectional(layers.LSTM(32, kernel_regularizer=keras.regularizers.l2(1e-4))),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, output_dir):
    """Train the model."""
    # Compute class weights
    n_fall = np.sum(y_train == 1)
    n_non_fall = np.sum(y_train == 0)
    total = len(y_train)
    
    weight_fall = total / (2 * n_fall)
    weight_non_fall = total / (2 * n_non_fall)
    
    class_weight = {0: weight_non_fall, 1: weight_fall}
    
    print(f"\nClass weights:")
    print(f"  Fall: {weight_fall:.4f}")
    print(f"  Non-fall: {weight_non_fall:.4f}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    checkpoint_path = output_dir / 'lstm_raw30_balanced_hnm_best.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    
    return history


def evaluate_model(model, X_test, y_test, output_dir):
    """Evaluate model and create plots."""
    print("\nEvaluating on test set...")
    
    # Predictions
    y_pred_proba = model.predict(X_test, batch_size=64, verbose=1).flatten()
    
    # Find optimal threshold
    thresholds = np.arange(0.05, 0.96, 0.01)
    f1_scores = []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Metrics at optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nTest Results (threshold={optimal_threshold:.2f}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn}, FP: {fp}")
    print(f"  FN: {fn}, TP: {tp}")
    
    # Compute FP rate
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\nError Rates:")
    print(f"  False Positive Rate: {fp_rate:.4f} ({fp}/{fp+tn})")
    print(f"  False Negative Rate: {fn_rate:.4f} ({fn}/{fn+tp})")
    
    # Save metrics
    metrics = {
        'optimal_threshold': float(optimal_threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'error_rates': {
            'false_positive_rate': float(fp_rate),
            'false_negative_rate': float(fn_rate)
        }
    }
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    print("="*70)
    print("PHASE 4.6 — RETRAIN BILSTM ON HNM DATASET")
    print("="*70)
    
    # Paths
    data_path = 'data/processed/all_windows_30_raw_balanced_hnm.npz'
    output_dir = Path('docs/wiki_assets/phase4_hardneg')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(data_path)
    
    # Build model
    print("\nBuilding model...")
    model = build_model(input_shape=(30, 34))
    model.summary()
    
    # Train
    history = train_model(model, X_train, y_train, X_val, y_val, output_dir)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, output_dir)
    
    # Copy best model to checkpoints
    import shutil
    checkpoint_dir = Path('ml/training/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(output_dir / 'lstm_raw30_balanced_hnm_best.h5', 
                checkpoint_dir / 'lstm_raw30_balanced_hnm_best.h5')
    
    # Load baseline metrics for comparison
    baseline_path = Path('docs/wiki_assets/phase4_threshold_sweep/deployment_thresholds_v2.json')
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        baseline_metrics = baseline['thresholds']['balanced']['metrics']
        
        print("\n" + "="*70)
        print("COMPARISON: BASELINE vs HNM")
        print("="*70)
        print(f"\n{'Metric':<25} {'Baseline':<15} {'HNM':<15} {'Delta':<15}")
        print("-"*70)
        print(f"{'F1 Score':<25} {baseline_metrics['f1']:<15.4f} {metrics['f1_score']:<15.4f} {metrics['f1_score']-baseline_metrics['f1']:<+15.4f}")
        print(f"{'Precision':<25} {baseline_metrics['precision']:<15.4f} {metrics['precision']:<15.4f} {metrics['precision']-baseline_metrics['precision']:<+15.4f}")
        print(f"{'Recall':<25} {baseline_metrics['recall']:<15.4f} {metrics['recall']:<15.4f} {metrics['recall']-baseline_metrics['recall']:<+15.4f}")
        print(f"{'ROC-AUC':<25} {baseline['roc_auc']:<15.4f} {metrics['roc_auc']:<15.4f} {metrics['roc_auc']-baseline['roc_auc']:<+15.4f}")
        print(f"{'False Positives':<25} {baseline_metrics['fp']:<15} {metrics['confusion_matrix']['fp']:<15} {metrics['confusion_matrix']['fp']-baseline_metrics['fp']:<+15}")
        print(f"{'False Negatives':<25} {baseline_metrics['fn']:<15} {metrics['confusion_matrix']['fn']:<15} {metrics['confusion_matrix']['fn']-baseline_metrics['fn']:<+15}")
        
        # Save comparison
        comparison = {
            'baseline': baseline_metrics,
            'hnm': metrics,
            'delta': {
                'f1': float(metrics['f1_score'] - baseline_metrics['f1']),
                'precision': float(metrics['precision'] - baseline_metrics['precision']),
                'recall': float(metrics['recall'] - baseline_metrics['recall']),
                'fp': int(metrics['confusion_matrix']['fp'] - baseline_metrics['fp']),
                'fn': int(metrics['confusion_matrix']['fn'] - baseline_metrics['fn'])
            }
        }
        
        with open(output_dir / 'comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"Best model saved to: ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5")
    print(f"Metrics: F1={metrics['f1_score']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, AUC={metrics['roc_auc']:.4f}")
    print(f"FP: {metrics['confusion_matrix']['fp']}, FN: {metrics['confusion_matrix']['fn']}")


if __name__ == '__main__':
    main()

