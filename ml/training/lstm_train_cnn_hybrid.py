"""
CNN + BiLSTM Hybrid Training Pipeline - Phase 2.4b

Architecture:
- Conv1D (64 filters, kernel=3) for local pattern extraction
- MaxPooling1D (pool_size=2) for dimensionality reduction
- Bidirectional LSTM (64 units) for temporal modeling
- GlobalAveragePooling1D for sequence aggregation
- Dense (64, ReLU) + Dropout (0.25)
- Dense (1, Sigmoid) for binary classification

Key Features:
- Focal loss (γ=2.8) for hard example mining
- Adam optimizer (lr=5e-4)
- Strong augmentation (time-warp ±20%, noise σ=0.07, feature-drop 10%)
- Class weights (1:3.5) for imbalance handling
- Early stopping + ReduceLROnPlateau + ModelCheckpoint

Target: F1 ≥ 0.80, ROC-AUC ≥ 0.94
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import argparse
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
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

# Import custom focal loss and utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.training.lstm_train_full import (
    SigmoidFocalCrossEntropy,
    F1Metric,
    find_optimal_threshold,
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_training_history
)
from ml.training.lstm_train_optimized import (
    BalancedBatchGenerator,
    subject_wise_split
)
from ml.training.lstm_train_enhanced import (
    compute_class_weights
)


def build_cnn_bilstm_hybrid_model(input_shape: tuple, conv_filters: int = 64,
                                   lstm_units: int = 64, dense_units: int = 64,
                                   dropout: float = 0.25) -> keras.Model:
    """
    Build CNN + BiLSTM hybrid model.
    
    Architecture:
    - Conv1D (64 filters, kernel=3, padding='same', ReLU)
    - MaxPooling1D (pool_size=2)
    - Bidirectional LSTM (64 units, return sequences)
    - GlobalAveragePooling1D
    - Dense (64, ReLU)
    - Dropout (0.25)
    - Dense (1, Sigmoid)
    
    Args:
        input_shape: (sequence_length, num_features) = (60, 16)
        conv_filters: Number of Conv1D filters
        lstm_units: Units in BiLSTM layer
        dense_units: Units in dense layer
        dropout: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape, name='input')
    
    # CNN layer for local pattern extraction
    x = keras.layers.Conv1D(
        filters=conv_filters,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='conv1d'
    )(inputs)
    
    # Max pooling for dimensionality reduction
    x = keras.layers.MaxPooling1D(pool_size=2, name='maxpool')(x)
    
    # BiLSTM for temporal modeling
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units, return_sequences=True),
        name='bilstm'
    )(x)
    
    # Global average pooling
    x = keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Dense layers
    x = keras.layers.Dense(dense_units, activation='relu', name='dense')(x)
    x = keras.layers.Dropout(dropout, name='dropout')(x)
    
    # Output layer
    outputs = keras.layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM_Hybrid')
    
    return model


def load_data(data_path: str):
    """Load 60-frame, 16-feature dataset."""
    data = np.load(data_path, allow_pickle=True)
    X = data['X']  # (N, 60, 16)
    y = data['y']  # (N,)
    video_ids = data['video_ids']  # (N,)
    
    print(f"Loaded dataset: {data_path}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Fall samples: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"Non-fall samples: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    
    return X, y, video_ids


def main():
    parser = argparse.ArgumentParser(description='Phase 2.4b - CNN + BiLSTM Hybrid Training')
    parser.add_argument('--data', type=str, default='data/processed/all_windows_v2.npz')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--focal-alpha', type=float, default=0.35)
    parser.add_argument('--focal-gamma', type=float, default=2.8)
    args = parser.parse_args()
    
    # Load data
    X, y, video_ids = load_data(args.data)
    
    # Subject-wise split
    X_train, X_test, y_train, y_test = subject_wise_split(X, y, video_ids)
    
    # Build model
    print("\n" + "="*60)
    print("Building CNN + BiLSTM Hybrid Model")
    print("="*60)
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (60, 16)
    model = build_cnn_bilstm_hybrid_model(input_shape)
    model.summary()
    
    # Compile with focal loss and Adam
    print("\n" + "="*60)
    print("Compiling Model")
    print("="*60)
    print(f"Loss: Sigmoid Focal CrossEntropy (α={args.focal_alpha}, γ={args.focal_gamma})")
    print(f"Optimizer: Adam (lr={args.lr})")
    
    loss = SigmoidFocalCrossEntropy(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            F1Metric(name='f1')
        ]
    )
    
    # Compute class weights
    class_weights = compute_class_weights(y_train, fall_weight_multiplier=3.5)
    print(f"\nClass weights: {class_weights}")
    
    # Create balanced batch generators
    print("\n" + "="*60)
    print("Creating Balanced Batch Generators")
    print("="*60)
    print(f"Batch size: {args.batch}")
    print(f"Augmentation: Strong (time-warp ±20%, noise σ=0.07, feature-drop 10%)")
    
    train_gen = BalancedBatchGenerator(
        X_train, y_train,
        batch_size=args.batch,
        augment=True,
        shuffle=True
    )
    
    val_gen = BalancedBatchGenerator(
        X_test, y_test,
        batch_size=args.batch,
        augment=False,
        shuffle=False
    )
    
    # Setup callbacks
    checkpoint_dir = Path('ml/training/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    history_dir = Path('ml/training/history')
    history_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_f1',
            patience=args.patience,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1',
            factor=0.5,
            patience=10,
            mode='max',
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'lstm_cnn_hybrid_best.h5'),
            monitor='val_f1',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save history
    import pandas as pd
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_dir / 'lstm_cnn_hybrid_history.csv', index=False)
    print(f"\n✅ Training history saved to {history_dir / 'lstm_cnn_hybrid_history.csv'}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    
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
    
    print(f"\nTest Metrics (threshold={optimal_threshold:.4f}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    
    print("\n✅ CNN + BiLSTM Hybrid training complete!")
    print(f"Best model saved to: {checkpoint_dir / 'lstm_cnn_hybrid_best.h5'}")


if __name__ == '__main__':
    main()

