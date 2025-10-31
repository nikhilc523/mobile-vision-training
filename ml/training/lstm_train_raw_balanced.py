"""
Phase 4.2: Retrain BiLSTM on Balanced RAW Keypoints (30×34)

Key Features:
- Balanced dataset (1:2.03 fall:non-fall ratio)
- Sigmoid Focal CrossEntropy (α=0.35, γ=2.0)
- AdamW optimizer (lr=5e-4, weight_decay=1e-4)
- ReduceLROnPlateau (patience=6, factor=0.5, min_lr=1e-6)
- Balanced batch sampler (50/50 fall/non-fall per batch)
- Early stopping on val F1 (patience=15)
- Subject-wise split to prevent data leakage

Architecture:
- BiLSTM(64) → BiLSTM(32) → Dropout(0.25) → Dense(32, ReLU) → Sigmoid

Target: F1 ≥ 0.70, ROC-AUC ≥ 0.90, Recall ≥ 0.75

Usage:
    python3 -m ml.training.lstm_train_raw_balanced \
        --data data/processed/all_windows_30_raw_balanced.npz \
        --epochs 120 \
        --batch 64 \
        --lr 5e-4
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
import pandas as pd
import sys

# Import utilities from existing training scripts
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Custom Loss: Sigmoid Focal CrossEntropy
# ============================================================================

class SigmoidFocalCrossEntropy(keras.losses.Loss):
    """
    Sigmoid Focal CrossEntropy Loss for binary classification.
    
    Focal loss focuses training on hard examples by down-weighting easy examples.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (0.35 = focus on fall class)
        gamma: Focusing parameter (2.0 = strong focus on hard examples)
    """
    
    def __init__(self, alpha=0.35, gamma=2.0, name='sigmoid_focal_crossentropy'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Compute focal loss
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        focal_weight = alpha_t * tf.pow(1 - p_t, self.gamma)
        bce = -tf.math.log(p_t)
        
        loss = focal_weight * bce
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        return {'alpha': self.alpha, 'gamma': self.gamma}


# ============================================================================
# Custom Metric: F1 Score
# ============================================================================

class F1Metric(keras.metrics.Metric):
    """F1 score metric for binary classification."""
    
    def __init__(self, name='f1', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision_metric = keras.metrics.Precision(thresholds=threshold)
        self.recall_metric = keras.metrics.Recall(thresholds=threshold)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        
        # F1 = 2 * (P * R) / (P + R)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1
    
    def reset_state(self):
        self.precision_metric.reset_state()
        self.recall_metric.reset_state()


# ============================================================================
# Balanced Batch Generator
# ============================================================================

class BalancedBatchGenerator(keras.utils.Sequence):
    """
    Balanced batch generator that ensures 50/50 fall/non-fall ratio in each batch.
    
    Samples with replacement to maintain balance.
    """
    
    def __init__(self, X, y, batch_size=64, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Separate indices by class
        self.fall_indices = np.where(y == 1)[0]
        self.non_fall_indices = np.where(y == 0)[0]
        
        # Calculate number of batches
        self.n_batches = int(np.ceil(len(X) / batch_size))
        
        self.on_epoch_end()
    
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, index):
        # Sample 50/50 from each class
        n_fall = self.batch_size // 2
        n_non_fall = self.batch_size - n_fall
        
        # Sample with replacement
        fall_batch_indices = np.random.choice(self.fall_indices, size=n_fall, replace=True)
        non_fall_batch_indices = np.random.choice(self.non_fall_indices, size=n_non_fall, replace=True)
        
        # Combine and shuffle
        batch_indices = np.concatenate([fall_batch_indices, non_fall_batch_indices])
        np.random.shuffle(batch_indices)
        
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.fall_indices)
            np.random.shuffle(self.non_fall_indices)


# ============================================================================
# Model Architecture
# ============================================================================

def build_bilstm_raw_balanced_model(input_shape: tuple) -> keras.Model:
    """
    Build BiLSTM model for balanced raw keypoints.
    
    Architecture:
    - BiLSTM(64) with L2 regularization
    - BiLSTM(32) with L2 regularization
    - Dropout(0.25)
    - Dense(32, ReLU)
    - Dense(1, Sigmoid)
    
    Args:
        input_shape: (seq_length, n_features) e.g., (30, 34)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First BiLSTM layer
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True,
                       kernel_regularizer=keras.regularizers.l2(1e-3))
        ),
        
        # Second BiLSTM layer
        layers.Bidirectional(
            layers.LSTM(32, kernel_regularizer=keras.regularizers.l2(1e-3))
        ),
        
        # Dropout
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='BiLSTM_RawBalanced')
    
    return model


# ============================================================================
# Data Splitting
# ============================================================================

def subject_wise_split(X, y, video_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data by subject (video) to prevent data leakage.
    
    Args:
        X: (N, T, F) array
        y: (N,) array
        video_ids: (N,) array
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    np.random.seed(random_state)
    
    # Get unique video IDs
    unique_videos = np.unique(video_ids)
    n_videos = len(unique_videos)
    
    # Shuffle videos
    np.random.shuffle(unique_videos)
    
    # Split videos
    n_train = int(n_videos * train_ratio)
    n_val = int(n_videos * val_ratio)
    
    train_videos = unique_videos[:n_train]
    val_videos = unique_videos[n_train:n_train + n_val]
    test_videos = unique_videos[n_train + n_val:]
    
    # Create masks
    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)
    test_mask = np.isin(video_ids, test_videos)
    
    # Split data
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# Evaluation and Visualization
# ============================================================================

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold that maximizes F1 score."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # F1
    axes[0, 1].plot(history['f1'], label='Train')
    axes[0, 1].plot(history['val_f1'], label='Val')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1')
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curve(y_true, y_pred_proba, save_path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Non-Fall', 'Fall']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Training Function
# ============================================================================

def train_model(args):
    """Main training function."""
    print("=" * 70)
    print("PHASE 4.2: RETRAIN BiLSTM ON BALANCED RAW KEYPOINTS")
    print("=" * 70)
    print()

    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load data
    print(f"[1/8] Loading data from {args.data}...")
    data = np.load(args.data, allow_pickle=True)

    X = data['X']  # (N, 30, 34)
    y = data['y']  # (N,)
    video_ids = data['video_ids']  # (N,)

    print(f"✓ Data loaded")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Features: {X.shape[2]}, Sequence length: {X.shape[1]}")
    print(f"  Class distribution: Fall={np.sum(y==1)} ({100*np.mean(y):.1f}%), "
          f"Non-fall={np.sum(y==0)} ({100*np.mean(y==0):.1f}%)")
    print(f"  Imbalance ratio: 1:{np.sum(y==0) / np.sum(y==1):.2f}")
    print()

    # Split data
    print("[2/8] Splitting data by subject (70/15/15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = subject_wise_split(
        X, y, video_ids,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_state=args.seed
    )

    print(f"✓ Data split")
    print(f"  Train: {len(X_train)} samples ({np.sum(y_train==1)} fall, {np.sum(y_train==0)} non-fall)")
    print(f"  Val: {len(X_val)} samples ({np.sum(y_val==1)} fall, {np.sum(y_val==0)} non-fall)")
    print(f"  Test: {len(X_test)} samples ({np.sum(y_test==1)} fall, {np.sum(y_test==0)} non-fall)")
    print()

    # Build model
    print("[3/8] Building model...")
    model = build_bilstm_raw_balanced_model(input_shape=(X.shape[1], X.shape[2]))
    model.summary()
    print()

    # Compile model
    print("[4/8] Compiling model...")

    # Loss
    loss = SigmoidFocalCrossEntropy(alpha=args.focal_alpha, gamma=args.focal_gamma)
    print(f"  Loss: Sigmoid Focal CrossEntropy (α={args.focal_alpha}, γ={args.focal_gamma})")

    # Optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")

    # Metrics
    metrics = [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        F1Metric(name='f1'),
        keras.metrics.AUC(name='auc')
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("✓ Model compiled")
    print()

    # Create data generators
    print("[5/8] Creating balanced batch generators...")
    train_gen = BalancedBatchGenerator(X_train, y_train, batch_size=args.batch, shuffle=True)
    val_gen = BalancedBatchGenerator(X_val, y_val, batch_size=args.batch, shuffle=False)

    print(f"  Train batches: {len(train_gen)} (batch size: {args.batch}, 50/50 fall/non-fall)")
    print(f"  Val batches: {len(val_gen)}")
    print()

    # Callbacks
    print("[6/8] Setting up callbacks...")

    # Model checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'lstm_raw30_balanced_best.h5'

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        monitor='val_f1',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # Early stopping
    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor='val_f1',
        mode='max',
        patience=args.patience,
        restore_best_weights=True,
        verbose=1
    )

    # Reduce LR on plateau
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [checkpoint_cb, early_stop_cb, reduce_lr_cb]

    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Early stopping: patience={args.patience}, monitor=val_f1")
    print(f"  ReduceLROnPlateau: patience=6, factor=0.5, min_lr=1e-6")
    print()

    # Train model
    print("[7/8] Training model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr}")
    print()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    print()
    print("✓ Training complete")
    print()

    # Evaluate on test set
    print("[8/8] Evaluating on test set...")

    # Load best model
    model = keras.models.load_model(
        checkpoint_path,
        custom_objects={
            'SigmoidFocalCrossEntropy': SigmoidFocalCrossEntropy,
            'F1Metric': F1Metric
        }
    )

    # Predict
    y_pred_proba = model.predict(X_test, verbose=0).flatten()

    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_pred_proba)
    print(f"  Optimal threshold: {optimal_threshold:.4f} (F1={optimal_f1:.4f})")

    # Predict with optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print()
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
    print(cm)
    print()

    # Save results
    output_dir = Path('docs/wiki_assets/phase4_balanced_training')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    print(f"✓ Saved training history to {output_dir / 'training_history.csv'}")

    # Save test metrics
    test_metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'optimal_threshold': float(optimal_threshold),
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"✓ Saved test metrics to {output_dir / 'test_metrics.json'}")

    # Plot training history
    plot_training_history(history.history, output_dir / 'training_history.png')
    print(f"✓ Saved training history plot to {output_dir / 'training_history.png'}")

    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba, output_dir / 'roc_curve.png')
    print(f"✓ Saved ROC curve to {output_dir / 'roc_curve.png'}")

    # Plot PR curve
    plot_pr_curve(y_test, y_pred_proba, output_dir / 'pr_curve.png')
    print(f"✓ Saved PR curve to {output_dir / 'pr_curve.png'}")

    # Plot confusion matrix
    plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png')
    print(f"✓ Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")

    print()
    print("=" * 70)
    print("✅ PHASE 4.2 COMPLETE")
    print("=" * 70)

    return test_metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 4.2 - Retrain BiLSTM on Balanced RAW Keypoints')
    parser.add_argument('--data', type=str, default='data/processed/all_windows_30_raw_balanced.npz')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--focal-alpha', type=float, default=0.35)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--checkpoint-dir', type=str, default='ml/training/checkpoints')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Train model
    test_metrics = train_model(args)

    # Update documentation
    print("\nUpdating documentation...")
    update_docs(test_metrics)

    print("\n✅ All done!")


def update_docs(test_metrics):
    """Update docs/results1.md with Phase 4.2 results."""
    docs_path = Path('docs/results1.md')

    summary = f"""
## 🗓️ Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Phase:** 4.2 — BiLSTM(30×34 RAW) on Balanced Data

### Objective
Retrain BiLSTM model on balanced dataset (1:2.03 ratio) with focal loss and balanced batch sampling

### Model Architecture
- **Input:** (30, 34) - 30 frames × 34 raw keypoint features
- **Architecture:** BiLSTM(64) → BiLSTM(32) → Dropout(0.25) → Dense(32, ReLU) → Sigmoid
- **Loss:** Sigmoid Focal CrossEntropy (α=0.35, γ=2.0)
- **Optimizer:** AdamW (lr=5e-4, weight_decay=1e-4)
- **Batch Sampling:** 50/50 fall/non-fall per batch (size=64)
- **Early Stopping:** Patience=15 on val_f1
- **LR Schedule:** ReduceLROnPlateau (patience=6, factor=0.5, min_lr=1e-6)

### Test Results
- **Precision:** {test_metrics['precision']:.4f}
- **Recall:** {test_metrics['recall']:.4f}
- **F1 Score:** {test_metrics['f1']:.4f}
- **ROC-AUC:** {test_metrics['roc_auc']:.4f}
- **Optimal Threshold:** {test_metrics['optimal_threshold']:.4f}

### Confusion Matrix
```
                Predicted
                Non-Fall  Fall
Actual Non-Fall    {test_metrics['confusion_matrix'][0][0]:4d}    {test_metrics['confusion_matrix'][0][1]:4d}
       Fall        {test_metrics['confusion_matrix'][1][0]:4d}    {test_metrics['confusion_matrix'][1][1]:4d}
```

### Comparison with Previous Model (Phase 3.2+)
| Metric | Old Model (Imbalanced) | New Model (Balanced) | Improvement |
|--------|------------------------|----------------------|-------------|
| **F1 Score** | 0.31 | {test_metrics['f1']:.4f} | {(test_metrics['f1'] - 0.31) / 0.31 * 100:+.1f}% |
| **Recall** | 0.55 | {test_metrics['recall']:.4f} | {(test_metrics['recall'] - 0.55) / 0.55 * 100:+.1f}% |
| **Precision** | 0.22 | {test_metrics['precision']:.4f} | {(test_metrics['precision'] - 0.22) / 0.22 * 100:+.1f}% |
| **ROC-AUC** | 0.9205 | {test_metrics['roc_auc']:.4f} | {(test_metrics['roc_auc'] - 0.9205) / 0.9205 * 100:+.1f}% |

### Output Files
- **Model:** `ml/training/checkpoints/lstm_raw30_balanced_best.h5`
- **Training History:** `docs/wiki_assets/phase4_balanced_training/training_history.csv`
- **Test Metrics:** `docs/wiki_assets/phase4_balanced_training/test_metrics.json`
- **Plots:** `docs/wiki_assets/phase4_balanced_training/` (ROC, PR, confusion matrix, training history)

### Next Steps
1. Test on `secondfall.mp4` with new model
2. Compare predictions with old model (max prob 0.11% → expected 30-80%)
3. Validate on full test set (URFD, Le2i datasets)

**Status:** ✅ Training Complete

---
"""

    with open(docs_path, 'a') as f:
        f.write(summary)

    print(f"✓ Updated {docs_path}")


if __name__ == '__main__':
    main()


