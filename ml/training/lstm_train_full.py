"""
LSTM Training Pipeline for Full Dataset (14,520 windows × 10 features)

Implements Phase 2.1 training with:
- Subject-wise data splitting (70/15/15) to prevent data leakage
- Focal loss (α=0.25, γ=2.0) for class imbalance
- Data augmentation (time warp, noise, dropout)
- Early stopping on validation F1 (patience=15)
- LR scheduling with ReduceLROnPlateau
- Threshold tuning for optimal F1
- Comprehensive metrics and visualizations

Architecture:
- Masking layer (mask_value=0.0)
- LSTM(64)
- Dropout(0.25)
- Dense(32, relu)
- Dense(1, sigmoid)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_fscore_support,
    precision_recall_curve, average_precision_score
)

from .augmentation import augment_batch, get_augmentation_params
from .lstm_train import (
    build_model,
    compute_class_weights,
    F1Metric,
    prepare_data
)


# Custom Focal Loss Implementation (for TF 2.16+)
@tf.keras.utils.register_keras_serializable()
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    """
    Sigmoid Focal Cross Entropy Loss.

    Focal loss applies a modulating term to the cross entropy loss in order to
    focus learning on hard negative examples. It is particularly useful for
    addressing class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where p_t is the model's estimated probability for the class with label 1.

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
               or a list of weights for each class. Default: 0.25
        gamma: Exponent of the modulating factor (1 - p_t)^γ. Default: 2.0
        from_logits: Whether y_pred is expected to be a logits tensor. Default: False
    """

    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False,
                 reduction='sum_over_batch_size', name='sigmoid_focal_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """
        Compute focal loss.

        Args:
            y_true: Ground truth values, shape (batch_size, 1) or (batch_size,)
            y_pred: Predicted values, shape (batch_size, 1) or (batch_size,)

        Returns:
            Focal loss value
        """
        # Ensure y_true and y_pred have the same shape
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Flatten if needed
        if len(y_true.shape) > 1:
            y_true = tf.squeeze(y_true, axis=-1)
        if len(y_pred.shape) > 1:
            y_pred = tf.squeeze(y_pred, axis=-1)

        # Convert from logits if needed
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate focal loss
        # For positive class (y_true = 1): -α * (1 - p)^γ * log(p)
        # For negative class (y_true = 0): -(1-α) * p^γ * log(1 - p)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Modulating factor
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        # Alpha weighting
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        focal_loss = alpha_weight * modulating_factor * cross_entropy

        return focal_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


class DataGenerator(keras.utils.Sequence):
    """
    Data generator with configurable augmentation support.

    Supports different augmentation modes: 'none', 'moderate', 'strong'
    """

    def __init__(self, X, y, batch_size=32, augment_mode='moderate', shuffle=True):
        """
        Initialize data generator.

        Args:
            X: (N, T, D) array of sequences
            y: (N,) array of labels
            batch_size: Batch size
            augment_mode: 'none', 'moderate', or 'strong'
            shuffle: Whether to shuffle data at epoch end
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment_mode = augment_mode
        self.shuffle = shuffle
        self.indices = np.arange(len(X))

        # Get augmentation parameters
        self.aug_params = get_augmentation_params(augment_mode)

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.X))
        batch_indices = self.indices[start_idx:end_idx]

        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        # Apply augmentation if enabled
        if self.augment_mode != 'none':
            X_batch, y_batch = augment_batch(
                X_batch, y_batch,
                augment_prob=self.aug_params['augment_prob'],
                apply_time_warp=self.aug_params['apply_time_warp'],
                apply_noise=self.aug_params['apply_noise'],
                apply_dropout=self.aug_params['apply_dropout'],
                warp_factor=self.aug_params['warp_factor'],
                noise_factor=self.aug_params['noise_factor'],
                dropout_rate=self.aug_params['dropout_rate']
            )

        # Replace NaN with 0.0 for masking
        X_batch = X_batch.copy()
        X_batch[np.isnan(X_batch)] = 0.0

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def split_data_by_subject(X, y, video_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data by subject (video_id) to prevent data leakage.
    
    Windows from the same video are kept together in the same split.
    
    Args:
        X: (N, T, D) array
        y: (N,) array
        video_ids: (N,) array of video identifiers
        train_ratio: Training set ratio (default: 0.7)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    np.random.seed(random_state)
    
    # Get unique video IDs
    unique_videos = np.unique(video_ids)
    n_videos = len(unique_videos)
    
    # Shuffle videos
    shuffled_videos = unique_videos.copy()
    np.random.shuffle(shuffled_videos)
    
    # Split videos
    train_end = int(n_videos * train_ratio)
    val_end = train_end + int(n_videos * val_ratio)
    
    train_videos = shuffled_videos[:train_end]
    val_videos = shuffled_videos[train_end:val_end]
    test_videos = shuffled_videos[val_end:]
    
    # Get indices for each split
    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)
    test_mask = np.isin(video_ids, test_videos)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Subject-wise split:")
    print(f"  Train: {len(train_videos)} videos, {len(X_train)} windows")
    print(f"  Val: {len(val_videos)} videos, {len(X_val)} windows")
    print(f"  Test: {len(test_videos)} videos, {len(X_test)} windows")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal decision threshold by maximizing a metric.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        optimal_threshold, best_metric_value, threshold_metrics
    """
    thresholds = np.arange(0.1, 0.95, 0.05)
    metrics_at_thresholds = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        metrics_at_thresholds.append({
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        })

    # Find best threshold
    if metric == 'f1':
        best_idx = np.argmax([m['f1'] for m in metrics_at_thresholds])
    elif metric == 'precision':
        best_idx = np.argmax([m['precision'] for m in metrics_at_thresholds])
    elif metric == 'recall':
        best_idx = np.argmax([m['recall'] for m in metrics_at_thresholds])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    optimal_threshold = metrics_at_thresholds[best_idx]['threshold']
    best_metric_value = metrics_at_thresholds[best_idx][metric]

    return optimal_threshold, best_metric_value, metrics_at_thresholds


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model on test set.

    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        threshold: Decision threshold (default: 0.5)

    Returns:
        Dictionary of metrics
    """
    # Prepare data
    X_test_clean, _ = prepare_data(X_test, y_test)

    # Predictions
    y_pred_proba = model.predict(X_test_clean, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # PR curve
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    metrics = {
        'threshold': float(threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr,
        'pr_precision': pr_precision,
        'pr_recall': pr_recall,
        'pr_thresholds': pr_thresholds
    }

    return metrics


def plot_training_history(history, output_dir):
    """Plot training history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1].plot(history.history['f1'], label='Train F1', linewidth=2)
    axes[1].plot(history.history['val_f1'], label='Val F1', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved training history plot: {output_dir / 'training_history.png'}")


def plot_roc_curve(metrics, output_dir):
    """Plot ROC curve."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.plot(metrics['fpr'], metrics['tpr'], linewidth=3, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved ROC curve: {output_dir / 'roc_curve.png'}")


def plot_pr_curve(metrics, output_dir):
    """Plot Precision-Recall curve."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.plot(metrics['pr_recall'], metrics['pr_precision'], linewidth=3,
             label=f"PR (AP = {metrics['pr_auc']:.4f})")

    # Baseline (random classifier for imbalanced data)
    y_true = metrics['y_pred']  # Just for getting the positive rate
    pos_rate = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0.5
    plt.axhline(y=pos_rate, color='k', linestyle='--', linewidth=2,
                label=f'Random Classifier (AP = {pos_rate:.4f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved PR curve: {output_dir / 'pr_curve.png'}")


def plot_confusion_matrix(metrics, output_dir):
    """Plot confusion matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm = np.array([
        [metrics['confusion_matrix']['tn'], metrics['confusion_matrix']['fp']],
        [metrics['confusion_matrix']['fn'], metrics['confusion_matrix']['tp']]
    ])
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'],
                annot_kws={'fontsize': 16})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved confusion matrix: {output_dir / 'confusion_matrix.png'}")


def save_test_metrics(metrics, output_dir, threshold_metrics=None):
    """Save test metrics to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove non-serializable items
    metrics_to_save = {
        'threshold': metrics.get('threshold', 0.5),
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics.get('pr_auc', 0.0),
        'confusion_matrix': metrics['confusion_matrix']
    }

    # Add threshold sweep results if available
    if threshold_metrics:
        metrics_to_save['threshold_sweep'] = threshold_metrics

    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"✅ Saved test metrics: {output_dir / 'test_metrics.json'}")


def train_lstm_full(config):
    """
    Train the LSTM model on full dataset with subject-wise split.
    
    Args:
        config: Dictionary with training configuration
    
    Returns:
        model, history, test_metrics
    """
    print("="*70)
    print("LSTM Fall Detection Training - Phase 2.1 (Full Dataset)")
    print("="*70)
    print()
    
    # Load data
    print(f"Loading data from {config['data_path']}...")
    data = np.load(config['data_path'], allow_pickle=True)
    X = data['X']
    y = data['y']
    video_ids = data['video_ids']
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {X.shape[2]}, Sequence length: {X.shape[1]}")
    print(f"Class distribution: Fall={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%), "
          f"Non-fall={np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
    print(f"Unique videos: {len(np.unique(video_ids))}")
    print()

    # Split data by subject
    print("Splitting data by subject (70/15/15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_subject(
        X, y, video_ids,
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15),
        random_state=config.get('random_state', 42)
    )
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
    if config.get('use_focal_loss', True):
        loss = SigmoidFocalCrossEntropy(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )
        print(f"Using Sigmoid Focal Cross Entropy (α={config.get('focal_alpha', 0.25)}, γ={config.get('focal_gamma', 2.0)})")
    else:
        loss = 'binary_crossentropy'
        print("Using Binary Cross Entropy")

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=config.get('learning_rate', 5e-4))

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
    augment_mode = config.get('augment_mode', 'moderate')
    train_gen = DataGenerator(
        X_train, y_train,
        batch_size=config.get('batch_size', 32),
        augment_mode=augment_mode,
        shuffle=True
    )
    val_gen = DataGenerator(
        X_val, y_val,
        batch_size=config.get('batch_size', 32),
        augment_mode='none',
        shuffle=False
    )
    print(f"Augmentation mode: {augment_mode}")

    # Callbacks
    callbacks = []

    # Model checkpoint
    if config.get('save_best', True):
        checkpoint_path = Path(config.get('checkpoint_dir', 'ml/training/checkpoints')) / 'lstm_full_best.h5'
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
        print(f"Model checkpoint: {checkpoint_path}")

    # Early stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor='val_f1',
            mode='max',
            patience=config.get('patience', 15),
            restore_best_weights=True,
            verbose=1
        )
    )
    print(f"Early stopping patience: {config.get('patience', 15)}")

    # Learning rate reduction
    if config.get('reduce_lr', True):
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1',
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        )
        print("ReduceLROnPlateau: factor=0.5, patience=5, min_lr=1e-6")

    # CSV logger
    if config.get('save_history', True):
        history_path = Path(config.get('history_dir', 'ml/training/history')) / 'lstm_full_history.csv'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.CSVLogger(str(history_path))
        )
        print(f"Training history: {history_path}")

    print()

    # Train model
    print("Training model...")
    print(f"Epochs: {config.get('epochs', 100)}")
    print(f"Batch size: {config.get('batch_size', 32)}")
    print(f"Learning rate: {config.get('learning_rate', 5e-4)}")
    print(f"Augmentation mode: {config.get('augment_mode', 'moderate')}")
    print()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.get('epochs', 100),
        callbacks=callbacks,
        class_weight=class_weights if config.get('use_class_weights', True) else None,
        verbose=1
    )

    print()
    print("Training complete!")
    print()

    # Threshold tuning on validation set
    print("Finding optimal threshold on validation set...")
    X_val_clean, _ = prepare_data(X_val, y_val)
    y_val_proba = model.predict(X_val_clean, verbose=0).flatten()
    optimal_threshold, best_f1, threshold_metrics = find_optimal_threshold(
        y_val, y_val_proba, metric='f1'
    )
    print(f"Optimal threshold: {optimal_threshold:.2f} (Val F1: {best_f1:.4f})")
    print()

    # Evaluate on test set with default threshold
    print("Evaluating on test set (threshold=0.5)...")
    test_metrics_default = evaluate_model(model, X_test, y_test, threshold=0.5)

    # Evaluate on test set with optimal threshold
    print(f"Evaluating on test set (threshold={optimal_threshold:.2f})...")
    test_metrics = evaluate_model(model, X_test, y_test, threshold=optimal_threshold)

    print()
    print("="*70)
    print("Test Set Results (Default Threshold = 0.5)")
    print("="*70)
    print(f"Precision: {test_metrics_default['precision']:.4f}")
    print(f"Recall:    {test_metrics_default['recall']:.4f}")
    print(f"F1 Score:  {test_metrics_default['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics_default['roc_auc']:.4f}")
    print(f"PR-AUC:    {test_metrics_default['pr_auc']:.4f}")
    print()
    print("="*70)
    print(f"Test Set Results (Optimal Threshold = {optimal_threshold:.2f})")
    print("="*70)
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print()
    print("Confusion Matrix:")
    cm = test_metrics['confusion_matrix']
    print(f"  TN: {cm['tn']:4d}  |  FP: {cm['fp']:4d}")
    print(f"  FN: {cm['fn']:4d}  |  TP: {cm['tp']:4d}")
    print("="*70)
    print()

    # Generate visualizations
    if config.get('plot_dir'):
        plot_dir = Path(config['plot_dir'])
        print(f"Generating visualizations in {plot_dir}...")
        plot_training_history(history, plot_dir)
        plot_roc_curve(test_metrics, plot_dir)
        plot_pr_curve(test_metrics, plot_dir)
        plot_confusion_matrix(test_metrics, plot_dir)
        save_test_metrics(test_metrics, plot_dir, threshold_metrics)
        print()

    return model, history, test_metrics


def update_results_doc(config, test_metrics, history):
    """Update docs/results1.md with training results."""
    docs_path = Path('docs/results1.md')

    # Get training info
    n_epochs = len(history.history['loss'])
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

    # Load data info
    data = np.load(config['data_path'], allow_pickle=True)
    n_samples = len(data['X'])
    n_features = data['X'].shape[2]

    cm = test_metrics['confusion_matrix']

    summary = f"""
## Phase 2.1 — Full LSTM(64) Training

🗓️ **Date:** {timestamp}

**Dataset:** `all_windows_full.npz` (N={n_samples:,}, features={n_features})

**Split:** 70/15/15 (subject-wise)

**Training:**
- Epochs: {n_epochs} (early-stopped)
- Batch size: {config.get('batch_size', 32)}
- Learning rate: {config.get('learning_rate', 1e-3)}
- Loss: Focal Loss (α={config.get('focal_alpha', 0.25)}, γ={config.get('focal_gamma', 2.0)})
- Augmentation: {'Enabled' if config.get('augment', True) else 'Disabled'}

**Test Metrics:**
- **Precision:** {test_metrics['precision']:.4f}
- **Recall:** {test_metrics['recall']:.4f}
- **F1 Score:** {test_metrics['f1']:.4f}
- **ROC-AUC:** {test_metrics['roc_auc']:.4f}

**Confusion Matrix:**
- TP: {cm['tp']} | TN: {cm['tn']} | FP: {cm['fp']} | FN: {cm['fn']}

**Status:** ✅ Success

---

"""

    # Append to file
    with open(docs_path, 'a') as f:
        f.write(summary)

    print(f"✅ Updated documentation: {docs_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train LSTM model on full dataset with subject-wise split'
    )

    # Data arguments
    parser.add_argument('--data', type=str, default='data/processed/all_windows_full.npz',
                        help='Path to windowed data .npz file')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=80,
                        help='Maximum number of epochs (default: 80)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--reduce-lr', action='store_true', default=True,
                        help='Use ReduceLROnPlateau (default: True)')
    parser.add_argument('--no-reduce-lr', action='store_false', dest='reduce_lr',
                        help='Disable ReduceLROnPlateau')

    # Model arguments
    parser.add_argument('--lstm-units', type=int, default=64,
                        help='LSTM units (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate (default: 0.25)')
    parser.add_argument('--dense-units', type=int, default=32,
                        help='Dense layer units (default: 32)')

    # Loss arguments
    parser.add_argument('--use-focal', action='store_true', default=True,
                        help='Use focal loss (default: True)')
    parser.add_argument('--no-focal', action='store_false', dest='use_focal',
                        help='Disable focal loss (use binary crossentropy)')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal loss alpha (default: 0.25)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma (default: 2.0)')

    # Training options
    parser.add_argument('--augment', type=str, default='moderate',
                        choices=['none', 'moderate', 'strong'],
                        help='Augmentation mode: none, moderate, strong (default: moderate)')
    parser.add_argument('--subject-split', action='store_true', default=True,
                        help='Use subject-wise split (default: True)')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class weights (default: True)')

    # Output arguments
    parser.add_argument('--save-best', action='store_true', default=True,
                        help='Save best model (default: True)')
    parser.add_argument('--checkpoint-dir', type=str, default='ml/training/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--history-dir', type=str, default='ml/training/history',
                        help='History directory')
    parser.add_argument('--plot-dir', type=str, default='docs/wiki_assets/phase2_full_training',
                        help='Plot output directory')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--update-docs', action='store_true', default=True,
                        help='Update docs/results1.md (default: True)')

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
        'reduce_lr': args.reduce_lr,
        'lstm_units': args.lstm_units,
        'dropout_rate': args.dropout,
        'dense_units': args.dense_units,
        'augment': args.augment != 'none',
        'augment_mode': args.augment,
        'use_focal_loss': args.use_focal,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'use_class_weights': args.use_class_weights,
        'save_best': args.save_best,
        'save_history': True,
        'checkpoint_dir': args.checkpoint_dir,
        'history_dir': args.history_dir,
        'plot_dir': args.plot_dir,
        'random_state': args.seed
    }

    # Train model
    model, history, test_metrics = train_lstm_full(config)

    # Update documentation
    if args.update_docs:
        update_results_doc(config, test_metrics, history)

    print()
    print("✅ Training pipeline complete!")
    print()


if __name__ == '__main__':
    main()

