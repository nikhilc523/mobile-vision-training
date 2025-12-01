"""
Enhanced LSTM Training Pipeline - Phase 2.2

Tier 1 Improvements:
- BiLSTM architecture with increased capacity
- Longer training (80 epochs, patience=20)
- Recall-optimized threshold tuning
- Adjusted class weights (3.5× for falls)

Tier 2 Improvements:
- 14 features (10 original + 4 derived)
- Variable-length windowing (90 frames = 3 seconds)
- Enhanced time-stretch augmentation (±20%)

Target: F1 ≥ 0.80, ROC-AUC ≥ 0.90
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

# Import custom focal loss and utilities from lstm_train_full
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


def build_bilstm_model(input_shape: tuple, lstm_units_1: int = 128, lstm_units_2: int = 64,
                       dense_units: int = 32, dropout: float = 0.25) -> keras.Model:
    """
    Build enhanced BiLSTM model with increased capacity.
    
    Architecture:
    - Masking layer (handle variable-length sequences)
    - Bidirectional LSTM (128 units, return sequences)
    - Dropout (0.25)
    - LSTM (64 units)
    - Dropout (0.25)
    - Dense (32, ReLU)
    - Dense (1, Sigmoid)
    
    Total params: ~150k (vs 21k in original)
    
    Args:
        input_shape: (sequence_length, num_features)
        lstm_units_1: Units in first BiLSTM layer
        lstm_units_2: Units in second LSTM layer
        dense_units: Units in dense layer
        dropout: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        keras.layers.Masking(mask_value=0.0, input_shape=input_shape),
        
        # Bidirectional LSTM captures forward + backward temporal dependencies
        keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_units_1, return_sequences=True),
            name='bilstm_1'
        ),
        keras.layers.Dropout(dropout, name='dropout_1'),
        
        # Second LSTM layer for hierarchical temporal features
        keras.layers.LSTM(lstm_units_2, name='lstm_2'),
        keras.layers.Dropout(dropout, name='dropout_2'),
        
        # Dense layers
        keras.layers.Dense(dense_units, activation='relu', name='dense_1'),
        keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    return model


def compute_class_weights(y: np.ndarray, fall_weight_multiplier: float = 3.5) -> dict:
    """
    Compute class weights with adjustable fall weight.
    
    Args:
        y: Labels array
        fall_weight_multiplier: Multiplier for fall class weight (default 3.5)
        
    Returns:
        Dictionary of class weights
    """
    n_samples = len(y)
    n_fall = np.sum(y == 1)
    n_non_fall = np.sum(y == 0)
    
    # Balanced weights
    weight_fall = n_samples / (2 * n_fall)
    weight_non_fall = n_samples / (2 * n_non_fall)
    
    # Apply multiplier to fall class
    weight_fall *= fall_weight_multiplier
    
    return {0: weight_non_fall, 1: weight_fall}


class EnhancedDataGenerator(keras.utils.Sequence):
    """
    Data generator with enhanced time-stretch augmentation (±20%).
    """
    
    def __init__(self, X, y, batch_size=32, augment=True, shuffle=True):
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
        
        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices]
        
        if self.augment:
            X_batch = self._augment_batch(X_batch)
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _augment_batch(self, X_batch):
        """
        Apply enhanced augmentation with ±20% time-stretch.
        """
        augmented = []
        
        for x in X_batch:
            if np.random.rand() < 0.7:  # 70% augmentation probability
                x_aug = x.copy()
                
                # Time-stretch (±20%)
                if np.random.rand() < 0.5:
                    stretch_factor = np.random.uniform(0.8, 1.2)
                    T_orig = x.shape[0]
                    T_new = int(T_orig * stretch_factor)
                    
                    # Resample to new length
                    indices = np.linspace(0, T_orig - 1, T_new)
                    x_stretched = np.zeros((T_new, x.shape[1]))
                    
                    for feat_idx in range(x.shape[1]):
                        x_stretched[:, feat_idx] = np.interp(
                            indices,
                            np.arange(T_orig),
                            x[:, feat_idx]
                        )
                    
                    # Pad or crop to original length
                    if T_new < T_orig:
                        # Pad with zeros
                        x_aug = np.zeros_like(x)
                        x_aug[:T_new] = x_stretched
                    else:
                        # Crop
                        x_aug = x_stretched[:T_orig]
                
                # Gaussian noise (σ=0.07)
                if np.random.rand() < 0.5:
                    noise = np.random.normal(0, 0.07, x_aug.shape)
                    x_aug = np.clip(x_aug + noise, 0, 1)
                
                # Feature dropout (10%)
                if np.random.rand() < 0.3:
                    num_features = x_aug.shape[1]
                    num_drop = int(num_features * 0.1)
                    drop_features = np.random.choice(num_features, num_drop, replace=False)
                    x_aug[:, drop_features] = 0
                
                augmented.append(x_aug)
            else:
                augmented.append(x)
        
        return np.array(augmented)


def prepare_data_enhanced(data_path: str, test_size: float = 0.15, val_size: float = 0.15,
                         random_state: int = 42):
    """
    Load and split enhanced dataset (14 features, 90 frames).
    """
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path)
    
    X = data['X']
    y = data['y']
    video_ids = data.get('video_ids', None)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {X.shape[2]}, Sequence length: {X.shape[1]}")
    print(f"Class distribution: Fall={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%), "
          f"Non-fall={np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
    
    if video_ids is not None:
        print(f"Unique videos: {len(np.unique(video_ids))}")
    
    # Subject-wise split
    if video_ids is not None:
        unique_videos = np.unique(video_ids)
        
        # Split videos
        train_videos, temp_videos = train_test_split(
            unique_videos, test_size=(test_size + val_size), random_state=random_state
        )
        val_videos, test_videos = train_test_split(
            temp_videos, test_size=test_size/(test_size + val_size), random_state=random_state
        )
        
        # Get indices
        train_mask = np.isin(video_ids, train_videos)
        val_mask = np.isin(video_ids, val_videos)
        test_mask = np.isin(video_ids, test_videos)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"\nSubject-wise split:")
        print(f"  Train: {len(train_videos)} videos, {len(X_train)} windows")
        print(f"  Val: {len(val_videos)} videos, {len(X_val)} windows")
        print(f"  Test: {len(test_videos)} videos, {len(X_test)} windows")
    else:
        # Random split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=random_state
        )
        
        print(f"\nRandom split:")
        print(f"  Train: {len(X_train)} windows")
        print(f"  Val: {len(X_val)} windows")
        print(f"  Test: {len(X_test)} windows")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_enhanced_model(args):
    """
    Main training function with all Tier 1 + Tier 2 improvements.
    """
    print("=" * 70)
    print("Enhanced LSTM Training - Phase 2.2")
    print("=" * 70)
    print("\nTier 1 Improvements:")
    print("  ✓ BiLSTM architecture (128 → 64 units)")
    print("  ✓ Longer training (80 epochs, patience=20)")
    print("  ✓ Recall-optimized threshold tuning")
    print("  ✓ Adjusted class weights (3.5× for falls)")
    print("\nTier 2 Improvements:")
    print("  ✓ 14 features (10 original + 4 derived)")
    print("  ✓ Variable-length windowing (90 frames)")
    print("  ✓ Enhanced time-stretch augmentation (±20%)")
    print()

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_enhanced(
        args.data, test_size=0.15, val_size=0.15, random_state=42
    )

    # Compute class weights
    class_weights = compute_class_weights(y_train, fall_weight_multiplier=3.5)
    print(f"\nClass weights: {class_weights}")

    # Build model
    print("\nBuilding BiLSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_bilstm_model(
        input_shape,
        lstm_units_1=args.lstm1,
        lstm_units_2=args.lstm2,
        dense_units=args.dense,
        dropout=args.dropout
    )
    model.summary()

    # Compile model
    print("\nCompiling model...")
    if args.use_focal:
        loss = SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
        print("Using Sigmoid Focal Cross Entropy (α=0.25, γ=2.0)")
    else:
        loss = 'binary_crossentropy'
        print("Using Binary Cross Entropy")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            F1Metric(name='f1')
        ]
    )

    # Setup callbacks
    checkpoint_dir = Path('ml/training/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'lstm_enhanced_best.h5'

    history_dir = Path('ml/training/history')
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / 'lstm_enhanced_history.csv'

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_f1',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_f1',
            mode='max',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(history_path)
    ]

    if args.reduce_lr:
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

    print(f"\nModel checkpoint: {checkpoint_path}")
    print(f"Early stopping patience: {args.patience}")
    if args.reduce_lr:
        print("ReduceLROnPlateau: factor=0.5, patience=5, min_lr=1e-6")
    print(f"Training history: {history_path}")

    # Create data generators
    train_gen = EnhancedDataGenerator(
        X_train, y_train,
        batch_size=args.batch,
        augment=args.augment,
        shuffle=True
    )

    val_gen = EnhancedDataGenerator(
        X_val, y_val,
        batch_size=args.batch,
        augment=False,
        shuffle=False
    )

    # Train model
    print(f"\nTraining model...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Augmentation: {'Enabled (±20% time-stretch)' if args.augment else 'Disabled'}")
    print()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("\nTraining complete!")

    # Find optimal threshold (optimize for recall)
    print("\nFinding optimal threshold (optimizing for recall)...")
    y_val_proba = model.predict(X_val, verbose=0).flatten()

    # Find threshold that maximizes recall while maintaining reasonable precision
    optimal_threshold_recall, best_recall, _ = find_optimal_threshold(
        y_val, y_val_proba, metric='recall'
    )

    # Also find F1-optimal threshold for comparison
    optimal_threshold_f1, best_f1, threshold_metrics = find_optimal_threshold(
        y_val, y_val_proba, metric='f1'
    )

    print(f"Recall-optimal threshold: {optimal_threshold_recall:.2f} (Recall: {best_recall:.4f})")
    print(f"F1-optimal threshold: {optimal_threshold_f1:.2f} (F1: {best_f1:.4f})")

    # Use recall-optimal threshold for safety-critical fall detection
    optimal_threshold = optimal_threshold_recall

    # Evaluate on test set
    print(f"\nEvaluating on test set (threshold={optimal_threshold:.2f})...")
    y_test_proba = model.predict(X_test, verbose=0).flatten()
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='binary', zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_test_proba)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)

    # Compute PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall_curve, precision_curve)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    print("\n" + "=" * 70)
    print(f"Test Set Results (Threshold = {optimal_threshold:.2f})")
    print("=" * 70)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  |  TP: {tp:4d}")
    print("=" * 70)

    # Save results
    output_dir = Path('docs/wiki_assets/phase2_enhanced_training')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        'threshold': float(optimal_threshold),
        'threshold_recall_optimal': float(optimal_threshold_recall),
        'threshold_f1_optimal': float(optimal_threshold_f1),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'threshold_sweep': threshold_metrics,
        # Add curve data for plotting
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'pr_precision': precision_curve.tolist(),
        'pr_recall': recall_curve.tolist(),
        'y_pred': y_test.tolist()  # For baseline in PR curve
    }

    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Saved test metrics: {metrics_path}")

    # Generate visualizations
    print(f"\nGenerating visualizations in {output_dir}...")

    plot_training_history(history, output_dir)
    plot_roc_curve(metrics, output_dir)
    plot_pr_curve(metrics, output_dir)
    plot_confusion_matrix(metrics, output_dir)

    print("\n✅ Enhanced training pipeline complete!")

    return model, history, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced LSTM Training - Phase 2.2')

    parser.add_argument('--data', type=str, required=True,
                       help='Path to enhanced dataset .npz file')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Number of epochs (default: 80)')
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate (default: 5e-4)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--lstm1', type=int, default=128,
                       help='BiLSTM units (default: 128)')
    parser.add_argument('--lstm2', type=int, default=64,
                       help='LSTM units (default: 64)')
    parser.add_argument('--dense', type=int, default=32,
                       help='Dense units (default: 32)')
    parser.add_argument('--dropout', type=float, default=0.25,
                       help='Dropout rate (default: 0.25)')
    parser.add_argument('--use-focal', action='store_true', default=True,
                       help='Use focal loss (default: True)')
    parser.add_argument('--no-focal', action='store_false', dest='use_focal',
                       help='Disable focal loss')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Enable augmentation (default: True)')
    parser.add_argument('--no-augment', action='store_false', dest='augment',
                       help='Disable augmentation')
    parser.add_argument('--reduce-lr', action='store_true', default=True,
                       help='Enable ReduceLROnPlateau (default: True)')
    parser.add_argument('--no-reduce-lr', action='store_false', dest='reduce_lr',
                       help='Disable ReduceLROnPlateau')

    args = parser.parse_args()

    # Train model
    model, history, metrics = train_enhanced_model(args)

