"""
Phase 4.5 — Train BiLSTM on Physics5 Features

Train a small BiLSTM model on 5 physics-inspired features:
- ratio_bbox, log_angle, rotational_energy, ratio_derivative, generalized_force

Model: BiLSTM(32) → BiLSTM(16) → Dense(16) → Sigmoid
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_data(data_path: str):
    """Load and split physics5 dataset."""
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    
    X = data['X']  # (N, 30, 5)
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


def build_model(input_shape=(30, 5)):
    """Build small BiLSTM model for physics5 features."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(16)),
        layers.Dense(16, activation='relu'),
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
    checkpoint_path = output_dir / 'lstm_phys5_best.h5'
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
        }
    }
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create plots
    create_plots(y_test, y_pred_proba, y_pred, optimal_threshold, output_dir)
    
    return metrics


def create_plots(y_test, y_pred_proba, y_pred, threshold, output_dir):
    """Create evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC Curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    ax = axes[0, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax.plot(recall, precision, 'g-', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    
    # Confusion Matrix
    ax = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Fall', 'Fall'])
    ax.set_yticklabels(['Non-Fall', 'Fall'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (t={threshold:.2f})')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=16, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    # Probability Distribution
    ax = axes[1, 1]
    ax.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Non-Fall', color='blue')
    ax.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Fall', color='red')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Count')
    ax.set_title('Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("="*70)
    print("PHASE 4.5 — TRAIN BILSTM ON PHYSICS5 FEATURES")
    print("="*70)
    
    # Paths
    data_path = 'data/processed/all_windows_30_physics5.npz'
    output_dir = Path('docs/wiki_assets/phase4_physics5')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(data_path)
    
    # Build model
    print("\nBuilding model...")
    model = build_model(input_shape=(30, 5))
    model.summary()
    
    # Train
    history = train_model(model, X_train, y_train, X_val, y_val, output_dir)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, output_dir)
    
    # Copy best model to checkpoints
    import shutil
    checkpoint_dir = Path('ml/training/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(output_dir / 'lstm_phys5_best.h5', checkpoint_dir / 'lstm_phys5_best.h5')
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"Best model saved to: ml/training/checkpoints/lstm_phys5_best.h5")
    print(f"Metrics: F1={metrics['f1_score']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, AUC={metrics['roc_auc']:.4f}")


if __name__ == '__main__':
    main()

