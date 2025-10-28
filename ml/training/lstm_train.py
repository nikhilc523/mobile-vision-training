"""
LSTM Training Pipeline for Fall Detection

Implements the custom LSTM classifier as described in the proposal (Section 3.4).

Architecture:
- Masking layer (mask_value=0.0)
- LSTM(64)
- Dropout(0.3)
- Dense(32, relu)
- Dense(1, sigmoid)

Training:
- Loss: Sigmoid Focal Cross Entropy (α=0.25, γ=2)
- Optimizer: Adam(lr=1e-3)
- Batch size: 32
- Early stopping on val_f1 (patience=10)
- Train/Val/Test: 70/15/15
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
from pathlib import Path
import json
from datetime import datetime
import sys

# Try to import tensorflow_addons for focal loss
try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except ImportError:
    HAS_TFA = False
    print("Warning: tensorflow_addons not available. Using binary crossentropy instead.")

from .augmentation import augment_batch


def build_model(seq_length=60, n_features=6, lstm_units=64, dropout_rate=0.3, dense_units=32):
    """
    Build the LSTM model as per proposal specification.
    
    Args:
        seq_length: Sequence length (default: 60)
        n_features: Number of features (default: 6)
        lstm_units: LSTM units (default: 64)
        dropout_rate: Dropout rate (default: 0.3)
        dense_units: Dense layer units (default: 32)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Masking layer to handle NaN/missing values
        layers.Masking(mask_value=0.0, input_shape=(seq_length, n_features)),
        
        # LSTM layer
        layers.LSTM(lstm_units),
        
        # Dropout for regularization
        layers.Dropout(dropout_rate),
        
        # Dense layer with ReLU
        layers.Dense(dense_units, activation='relu'),
        
        # Output layer with sigmoid
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def prepare_data(X, y, replace_nan_with=0.0):
    """
    Prepare data for training by replacing NaN with mask value.
    
    Args:
        X: (N, T, D) array with potential NaN values
        y: (N,) array of labels
        replace_nan_with: Value to replace NaN with (default: 0.0)
    
    Returns:
        X_clean, y
    """
    X_clean = X.copy()
    X_clean[np.isnan(X_clean)] = replace_nan_with
    return X_clean, y


def split_data(X, y, video_names, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data into train/val/test sets.
    
    For small datasets, we use random split. For larger datasets with subject info,
    this should be modified to split by subject to avoid data leakage.
    
    Args:
        X: (N, T, D) array
        y: (N,) array
        video_names: (N,) array of video names
        train_ratio: Training set ratio (default: 0.7)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    np.random.seed(random_state)
    
    N = len(X)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_class_weights(y):
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y: (N,) array of labels
    
    Returns:
        Dictionary of class weights
    """
    n_samples = len(y)
    n_classes = 2
    
    class_counts = np.bincount(y.astype(int))
    class_weights = n_samples / (n_classes * class_counts)
    
    return {0: class_weights[0], 1: class_weights[1]}


class F1Metric(keras.metrics.Metric):
    """Custom F1 score metric for Keras."""
    
    def __init__(self, name='f1', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
    
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return f1
    
    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


class DataGenerator(keras.utils.Sequence):
    """Data generator with augmentation support."""
    
    def __init__(self, X, y, batch_size=32, augment=False, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.X))
        batch_indices = self.indices[start_idx:end_idx]
        
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        if self.augment:
            X_batch, y_batch = augment_batch(X_batch, y_batch, augment_prob=0.5)
        
        # Replace NaN with 0.0 for masking
        X_batch = X_batch.copy()
        X_batch[np.isnan(X_batch)] = 0.0
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_lstm(config):
    """
    Train the LSTM model.
    
    Args:
        config: Dictionary with training configuration
    
    Returns:
        model, history, test_metrics
    """
    print("="*70)
    print("LSTM Fall Detection Training")
    print("="*70)
    print()
    
    # Load data
    print(f"Loading data from {config['data_path']}...")
    data = np.load(config['data_path'], allow_pickle=True)
    X = data['X']
    y = data['y']
    video_names = data['video_names']
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {X.shape[2]}, Sequence length: {X.shape[1]}")
    print(f"Class distribution: Fall={np.sum(y==1)}, Non-fall={np.sum(y==0)}")
    print()
    
    # Split data
    print("Splitting data (70/15/15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, video_names,
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15),
        random_state=config.get('random_state', 42)
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    print()
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    print()
    
    # Build model
    print("Building model...")
    model = build_model(
        seq_length=X.shape[1],
        n_features=X.shape[2],
        lstm_units=config.get('lstm_units', 64),
        dropout_rate=config.get('dropout_rate', 0.3),
        dense_units=config.get('dense_units', 32)
    )
    
    # Print model summary
    model.summary()
    print()
    
    # Compile model
    print("Compiling model...")
    
    # Loss function
    if HAS_TFA and config.get('use_focal_loss', True):
        loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
        print("Using Sigmoid Focal Cross Entropy (α=0.25, γ=2.0)")
    else:
        loss = 'binary_crossentropy'
        print("Using Binary Cross Entropy")
    
    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=config.get('learning_rate', 1e-3))
    
    # Metrics
    metrics = [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        F1Metric(name='f1'),
        keras.metrics.AUC(name='auc')
    ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print()
    
    # Create data generators
    train_gen = DataGenerator(X_train, y_train, batch_size=config.get('batch_size', 32), augment=config.get('augment', False))
    val_gen = DataGenerator(X_val, y_val, batch_size=config.get('batch_size', 32), augment=False, shuffle=False)
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    if config.get('save_best', True):
        checkpoint_path = Path(config.get('checkpoint_dir', 'ml/training/checkpoints')) / 'lstm_best.h5'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_f1',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        )
    
    # Early stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor='val_f1',
            mode='max',
            patience=config.get('patience', 10),
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # CSV logger
    if config.get('save_history', True):
        history_path = Path(config.get('history_dir', 'ml/training/history')) / 'lstm_history.csv'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.CSVLogger(str(history_path))
        )
    
    # Train model
    print("Training model...")
    print()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.get('epochs', 100),
        class_weight=class_weights if config.get('use_class_weights', True) else None,
        callbacks=callbacks,
        verbose=1
    )
    
    print()
    print("="*70)
    print("Training complete!")
    print("="*70)

    return model, history, (X_test, y_test)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model on test set.

    Args:
        model: Trained Keras model
        X_test: Test features (N, T, D)
        y_test: Test labels (N,)
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix, classification_report
    )

    print()
    print("="*70)
    print("Evaluating on test set...")
    print("="*70)
    print()

    # Prepare test data
    X_test_clean = X_test.copy()
    X_test_clean[np.isnan(X_test_clean)] = 0.0

    # Predict
    y_pred_proba = model.predict(X_test_clean, verbose=0).flatten()
    y_pred = (y_pred_proba > threshold).astype(int)

    # Compute metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ROC-AUC (only if both classes present)
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Print results
    print(f"Test Set Results:")
    print(f"  Samples: {len(y_test)}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print()
    print(f"Confusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fall', 'Fall'], zero_division=0))

    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'n_samples': int(len(y_test)),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist(),
        'y_true': y_test.tolist()
    }

    return metrics


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and F1 curves).

    Args:
        history: Keras History object
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # F1 curve
    if 'f1' in history.history:
        axes[1].plot(history.history['f1'], label='Train F1', linewidth=2)
        axes[1].plot(history.history['val_f1'], label='Val F1', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('F1 Score', fontsize=12)
        axes[1].set_title('Training and Validation F1', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training history plot: {save_path}")

    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot (optional)
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved ROC curve: {save_path}")

    plt.close()


def plot_confusion_matrix(cm_dict, save_path=None):
    """
    Plot confusion matrix.

    Args:
        cm_dict: Dictionary with tn, fp, fn, tp
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.array([
        [cm_dict['tn'], cm_dict['fp']],
        [cm_dict['fn'], cm_dict['tp']]
    ])

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'],
                ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix: {save_path}")

    plt.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train LSTM model for fall detection'
    )

    # Data arguments
    parser.add_argument('--data', type=str, default='data/processed/all_windows.npz',
                        help='Path to windowed data .npz file')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')

    # Model arguments
    parser.add_argument('--lstm-units', type=int, default=64,
                        help='LSTM units (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--dense-units', type=int, default=32,
                        help='Dense layer units (default: 32)')

    # Augmentation
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--no-focal-loss', action='store_true',
                        help='Disable focal loss (use binary crossentropy)')
    parser.add_argument('--no-class-weights', action='store_true',
                        help='Disable class weights')

    # Output arguments
    parser.add_argument('--save-best', action='store_true', default=True,
                        help='Save best model checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='ml/training/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--history-dir', type=str, default='ml/training/history',
                        help='History directory')
    parser.add_argument('--plot-dir', type=str, default='docs/wiki_assets/phase2_training',
                        help='Plot output directory')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Build config
    config = {
        'data_path': args.data,
        'epochs': args.epochs,
        'batch_size': args.batch,
        'learning_rate': args.lr,
        'patience': args.patience,
        'lstm_units': args.lstm_units,
        'dropout_rate': args.dropout,
        'dense_units': args.dense_units,
        'augment': args.augment,
        'use_focal_loss': not args.no_focal_loss,
        'use_class_weights': not args.no_class_weights,
        'save_best': args.save_best,
        'save_history': True,
        'checkpoint_dir': args.checkpoint_dir,
        'history_dir': args.history_dir,
        'random_state': args.seed
    }

    # Train model
    model, history, (X_test, y_test) = train_lstm(config)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Create plot directory
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot training history
    print()
    print("Generating plots...")
    plot_training_history(history, save_path=plot_dir / 'training_history.png')

    # Plot ROC curve
    if len(np.unique(y_test)) > 1:
        plot_roc_curve(
            metrics['y_true'],
            metrics['y_pred_proba'],
            save_path=plot_dir / 'roc_curve.png'
        )

    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=plot_dir / 'confusion_matrix.png'
    )

    # Save metrics to JSON
    metrics_path = plot_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        # Remove large arrays before saving
        metrics_to_save = {k: v for k, v in metrics.items()
                          if k not in ['y_pred', 'y_pred_proba', 'y_true']}
        json.dump(metrics_to_save, f, indent=2)
    print(f"✓ Saved test metrics: {metrics_path}")

    print()
    print("="*70)
    print("✓ Training pipeline complete!")
    print("="*70)
    print()
    print(f"Best model: {Path(args.checkpoint_dir) / 'lstm_best.h5'}")
    print(f"Training history: {Path(args.history_dir) / 'lstm_history.csv'}")
    print(f"Plots: {plot_dir}")
    print()
    print(f"Test Performance:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print()

    return model, history, metrics


if __name__ == '__main__':
    main()

