"""
BiLSTM Training with Raw Keypoints (34 features) - Phase 3.2+

Key Changes:
- Input: 34 features (17 keypoints × 2 coordinates) instead of 14 engineered features
- Sequence length: 30 frames (1.0 second) instead of 60 frames
- Let the model learn features automatically (simpler is better)
- Inspired by fall-detection-deep-learning-master approach

Target: Detect falls in secondfall.mp4 and improve generalization
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import json
from datetime import datetime


def load_data(data_path: str) -> tuple:
    """Load windowed dataset."""
    print(f"\nLoading data from {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    X = data['X']  # (N, 30, 34)
    y = data['y']  # (N,)
    video_ids = data['video_ids']  # (N,)
    
    print(f"Loaded {len(X)} windows")
    print(f"Shape: {X.shape}")
    print(f"Fall samples: {np.sum(y == 1)} ({100*np.mean(y):.1f}%)")
    print(f"Non-fall samples: {np.sum(y == 0)} ({100*np.mean(y == 0):.1f}%)")
    
    return X, y, video_ids


def subject_wise_split(X: np.ndarray, y: np.ndarray, video_ids: np.ndarray,
                       test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Split data by video ID to prevent data leakage."""
    unique_videos = np.unique(video_ids)
    
    # Split video IDs
    train_videos, test_videos = train_test_split(
        unique_videos, test_size=test_size, random_state=random_state,
        stratify=[1 if 'fall' in v.lower() or 'chute' in v.lower() else 0 for v in unique_videos]
    )
    
    # Create masks
    train_mask = np.isin(video_ids, train_videos)
    test_mask = np.isin(video_ids, test_videos)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"\nTrain: {len(X_train)} windows from {len(train_videos)} videos")
    print(f"  Fall: {np.sum(y_train == 1)} ({100*np.mean(y_train):.1f}%)")
    print(f"Test: {len(X_test)} windows from {len(test_videos)} videos")
    print(f"  Fall: {np.sum(y_test == 1)} ({100*np.mean(y_test):.1f}%)")
    
    return X_train, X_test, y_train, y_test


def build_bilstm_raw_keypoints_model(input_shape: tuple, lstm_units_1: int = 128,
                                      lstm_units_2: int = 64, dense_units: int = 64,
                                      dropout: float = 0.3) -> keras.Model:
    """
    Build BiLSTM model for raw keypoints.
    
    Architecture:
    - Bidirectional LSTM (128 units, return sequences)
    - Dropout (0.3)
    - Bidirectional LSTM (64 units)
    - Dropout (0.3)
    - Dense (64, ReLU)
    - Dense (1, Sigmoid)
    
    Args:
        input_shape: (sequence_length, num_features) = (30, 34)
        lstm_units_1: Units in first BiLSTM layer
        lstm_units_2: Units in second BiLSTM layer
        dense_units: Units in dense layer
        dropout: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First BiLSTM layer
        layers.Bidirectional(
            layers.LSTM(lstm_units_1, return_sequences=True,
                       kernel_regularizer=keras.regularizers.l2(1e-3))
        ),
        layers.Dropout(dropout),
        
        # Second BiLSTM layer
        layers.Bidirectional(
            layers.LSTM(lstm_units_2,
                       kernel_regularizer=keras.regularizers.l2(1e-3))
        ),
        layers.Dropout(dropout),
        
        # Dense layers
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='BiLSTM_RawKeypoints')
    
    return model


def compute_class_weights(y: np.ndarray) -> dict:
    """Compute class weights for imbalanced dataset."""
    n_samples = len(y)
    n_classes = 2
    n_fall = np.sum(y == 1)
    n_non_fall = np.sum(y == 0)
    
    # Inverse frequency weighting
    weight_fall = n_samples / (n_classes * n_fall)
    weight_non_fall = n_samples / (n_classes * n_non_fall)
    
    class_weights = {0: weight_non_fall, 1: weight_fall}
    
    print(f"\nClass weights:")
    print(f"  Non-fall (0): {weight_non_fall:.3f}")
    print(f"  Fall (1): {weight_fall:.3f}")
    print(f"  Ratio: 1:{weight_fall/weight_non_fall:.2f}")
    
    return class_weights


def plot_training_history(history, save_path: Path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")


def evaluate_model(model, X_test, y_test, save_dir: Path):
    """Evaluate model and save results."""
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fall', 'Fall']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\nROC-AUC: {roc_auc:.4f}")
    
    # Save results
    results = {
        'classification_report': classification_report(y_test, y_pred, target_names=['Non-Fall', 'Fall'], output_dict=True),
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = save_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return y_pred_prob, y_pred


def main():
    parser = argparse.ArgumentParser(description='Phase 3.2+ - BiLSTM Training with Raw Keypoints')
    parser.add_argument('--data', type=str, default='data/processed/all_windows_30frame_raw.npz')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    
    # Load data
    X, y, video_ids = load_data(args.data)
    
    # Subject-wise split
    X_train, X_test, y_train, y_test = subject_wise_split(X, y, video_ids)
    
    # Build model
    print("\n" + "="*70)
    print("Building BiLSTM Model (Raw Keypoints)")
    print("="*70)
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (30, 34)
    model = build_bilstm_raw_keypoints_model(input_shape)
    model.summary()
    
    # Compile
    print("\n" + "="*70)
    print("Compiling Model")
    print("="*70)
    print(f"Loss: Binary Crossentropy")
    print(f"Optimizer: Adam (lr={args.lr})")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Class weights
    class_weights = compute_class_weights(y_train)
    
    # Callbacks
    checkpoint_dir = Path('ml/training/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(checkpoint_dir / 'lstm_raw_keypoints_best.h5', monitor='val_loss',
                       save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot history
    plot_training_history(history, checkpoint_dir / 'training_history_raw_keypoints.png')
    
    # Evaluate
    evaluate_model(model, X_test, y_test, checkpoint_dir)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel saved to: {checkpoint_dir / 'lstm_raw_keypoints_best.h5'}")


if __name__ == '__main__':
    main()

