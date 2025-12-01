#!/usr/bin/env python3
"""
Phase 4.2: Retrain BiLSTM on Balanced RAW Keypoints (SIMPLIFIED VERSION)

This version uses standard batching with class weights instead of balanced batch sampling.

Author: Nikhil Chowdary
Date: 2025-10-30
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# Model Architecture
# ============================================================================

def build_bilstm_model(input_shape: tuple) -> keras.Model:
    """
    Build BiLSTM model for raw keypoints.
    
    Architecture:
    - BiLSTM(64) with L2 regularization
    - BiLSTM(32) with L2 regularization
    - Dropout(0.3)
    - Dense(32, ReLU)
    - Dense(1, Sigmoid)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First BiLSTM layer
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True,
                       kernel_regularizer=keras.regularizers.l2(1e-4))
        ),
        
        # Second BiLSTM layer
        layers.Bidirectional(
            layers.LSTM(32, kernel_regularizer=keras.regularizers.l2(1e-4))
        ),
        
        # Dropout
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dense(1, activation='sigmoid')
    ], name='BiLSTM_Raw')
    
    return model


# ============================================================================
# Data Splitting
# ============================================================================

def subject_wise_split(X, y, video_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Split data by subject (video) to prevent data leakage."""
    np.random.seed(random_state)
    
    unique_videos = np.unique(video_ids)
    np.random.shuffle(unique_videos)
    
    n_train = int(len(unique_videos) * train_ratio)
    n_val = int(len(unique_videos) * val_ratio)
    
    train_videos = unique_videos[:n_train]
    val_videos = unique_videos[n_train:n_train + n_val]
    test_videos = unique_videos[n_train + n_val:]
    
    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)
    test_mask = np.isin(video_ids, test_videos)
    
    return (X[train_mask], y[train_mask],
            X[val_mask], y[val_mask],
            X[test_mask], y[test_mask])


# ============================================================================
# Evaluation
# ============================================================================

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold by maximizing F1 score."""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def plot_training_history(history, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Train')
    axes[0, 1].plot(history['val_accuracy'], label='Val')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history['precision'], label='Train')
    axes[1, 0].plot(history['val_precision'], label='Val')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history['recall'], label='Train')
    axes[1, 1].plot(history['val_recall'], label='Val')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, output_dir):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curve(y_true, y_pred_proba, output_dir):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train BiLSTM on balanced raw keypoints (v2)')
    parser.add_argument('--data', type=str, required=True, help='Path to balanced dataset .npz file')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print("=" * 70)
    print("PHASE 4.2: RETRAIN BiLSTM ON BALANCED RAW KEYPOINTS (V2)")
    print("=" * 70)
    print()
    
    # Load data
    print(f"[1/7] Loading data from {args.data}...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    video_ids = data.get('video_ids', np.arange(len(y)))
    
    print("✓ Data loaded")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Features: {X.shape[2]}, Sequence length: {X.shape[1]}")
    print(f"  Class distribution: Fall={np.sum(y == 1)} ({np.mean(y == 1) * 100:.1f}%), "
          f"Non-fall={np.sum(y == 0)} ({np.mean(y == 0) * 100:.1f}%)")
    print(f"  Imbalance ratio: 1:{np.sum(y == 0) / np.sum(y == 1):.2f}")
    print()
    
    # Split data
    print("[2/7] Splitting data by subject (70/15/15)...")
    X_train, y_train, X_val, y_val, X_test, y_test = subject_wise_split(X, y, video_ids)
    
    print("✓ Data split")
    print(f"  Train: {len(X_train)} samples ({np.sum(y_train == 1)} fall, {np.sum(y_train == 0)} non-fall)")
    print(f"  Val: {len(X_val)} samples ({np.sum(y_val == 1)} fall, {np.sum(y_val == 0)} non-fall)")
    print(f"  Test: {len(X_test)} samples ({np.sum(y_test == 1)} fall, {np.sum(y_test == 0)} non-fall)")
    print()
    
    # Calculate class weights
    n_fall = np.sum(y_train == 1)
    n_non_fall = np.sum(y_train == 0)
    total = len(y_train)
    
    # Use inverse frequency weighting
    weight_fall = total / (2 * n_fall)
    weight_non_fall = total / (2 * n_non_fall)
    
    class_weight = {0: weight_non_fall, 1: weight_fall}
    
    print(f"[3/7] Calculated class weights:")
    print(f"  Fall (class 1): {weight_fall:.4f}")
    print(f"  Non-fall (class 0): {weight_non_fall:.4f}")
    print()
    
    # Build model
    print("[4/7] Building model...")
    model = build_bilstm_model(input_shape=(X.shape[1], X.shape[2]))
    model.summary()
    print()
    
    # Compile model
    print("[5/7] Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    print("✓ Model compiled")
    print()
    
    # Setup callbacks
    output_dir = Path('docs/wiki_assets/phase4_balanced_training_v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = Path('ml/training/checkpoints/lstm_raw30_balanced_v2_best.h5')
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=args.patience,
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("[6/7] Training model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping patience: {args.patience}")
    print()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✓ Training complete")
    print()
    
    # Evaluate on test set
    print("[7/7] Evaluating on test set...")
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print()
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    
    # Save results
    import pandas as pd
    
    # Training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    print(f"✓ Saved training history to {output_dir / 'training_history.csv'}")
    
    # Test metrics
    test_metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'optimal_threshold': float(optimal_threshold)
    }
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"✓ Saved test metrics to {output_dir / 'test_metrics.json'}")
    
    # Plots
    plot_training_history(history.history, output_dir)
    print(f"✓ Saved training history plot to {output_dir / 'training_history.png'}")
    
    plot_roc_curve(y_test, y_pred_proba, output_dir)
    print(f"✓ Saved ROC curve to {output_dir / 'roc_curve.png'}")
    
    plot_pr_curve(y_test, y_pred_proba, output_dir)
    print(f"✓ Saved PR curve to {output_dir / 'pr_curve.png'}")
    
    plot_confusion_matrix(y_test, y_pred, output_dir)
    print(f"✓ Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    
    print()
    print("=" * 70)
    print("✅ PHASE 4.2 COMPLETE (V2)")
    print("=" * 70)
    print()
    
    # Update documentation
    print("Updating documentation...")
    docs_path = Path('docs/results1.md')
    
    with open(docs_path, 'a') as f:
        f.write(f"\n## Phase 4.2 — BiLSTM(30×34 RAW) on balanced data (V2)\n")
        f.write(f"Test: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, ROC-AUC={roc_auc:.4f}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(f"Training: {len(history.history['loss'])} epochs, class weights used\n")
    
    print(f"✓ Updated {docs_path}")
    print()
    print("✅ All done!")


if __name__ == '__main__':
    main()

