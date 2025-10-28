# LSTM Training Module

## Overview

The `ml.training` module provides a complete training pipeline for the LSTM-based fall detection model as specified in the project proposal (Section 3.4).

## Model Architecture

The model follows the exact specification from the proposal:

```python
model = keras.Sequential([
    Masking(mask_value=0.0, input_shape=(60, 6)),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

**Parameters:**
- Total: 20,289 (~79 KB)
- LSTM: 18,176 parameters
- Dense layers: 2,113 parameters
- Well within mobile deployment constraints (< 2 MB target)

## Training Configuration

### Loss Function
- **Primary:** Sigmoid Focal Cross Entropy (α=0.25, γ=2.0) via TensorFlow Addons
- **Fallback:** Binary Cross Entropy (if TensorFlow Addons not available)
- Focal loss helps with class imbalance by focusing on hard examples

### Optimizer
- **Adam** with learning rate 1e-3
- Adaptive learning rate for efficient convergence

### Data Split
- **Train:** 70%
- **Validation:** 15%
- **Test:** 15%
- Random split (for larger datasets, should split by subject to avoid leakage)

### Class Weights
- Automatically computed to handle class imbalance
- Formula: `n_samples / (n_classes * class_counts)`
- Helps model learn minority class better

### Early Stopping
- Monitors validation F1 score
- Patience: 10 epochs
- Restores best weights automatically

### Callbacks
- **ModelCheckpoint:** Saves best model based on val_f1
- **EarlyStopping:** Prevents overfitting
- **CSVLogger:** Logs training history to CSV

## Data Augmentation

Three augmentation techniques to improve generalization:

### 1. Time Warping (±10%)
- Randomly speeds up or slows down sequences
- Simulates different fall speeds
- Uses interpolation to maintain sequence length

### 2. Gaussian Noise (±5%)
- Adds random noise to features
- Simulates sensor noise and variations
- Clipped to [0, 1] range

### 3. Feature Dropout (10%)
- Randomly sets features to NaN
- Simulates missing keypoints
- Forces model to be robust to occlusions

**Usage:**
```python
from ml.training.augmentation import augment_sequence

augmented = augment_sequence(
    sequence,
    apply_time_warp=True,
    apply_noise=True,
    apply_dropout=True
)
```

## Metrics

The model is evaluated on multiple metrics:

- **Precision:** Minimize false positives (important for user experience)
- **Recall:** Catch all true falls (critical for safety)
- **F1 Score:** Balanced metric (harmonic mean of precision and recall)
- **ROC-AUC:** Overall classification performance
- **Confusion Matrix:** Detailed breakdown (TP, TN, FP, FN)

**Target Performance (from proposal):**
- Precision ≥ 0.90
- Recall ≥ 0.90
- F1 ≥ 0.90
- ROC-AUC ≥ 0.90

## Usage

### Command Line

Basic training:
```bash
python -m ml.training.lstm_train \
    --data data/processed/all_windows.npz \
    --epochs 100 \
    --batch 32 \
    --lr 1e-3 \
    --augment \
    --save-best
```

Full options:
```bash
python -m ml.training.lstm_train \
    --data data/processed/all_windows.npz \
    --epochs 100 \
    --batch 32 \
    --lr 1e-3 \
    --patience 10 \
    --lstm-units 64 \
    --dropout 0.3 \
    --dense-units 32 \
    --augment \
    --save-best \
    --checkpoint-dir ml/training/checkpoints \
    --history-dir ml/training/history \
    --plot-dir docs/wiki_assets/phase2_training \
    --seed 42
```

### Python API

```python
from ml.training import train_lstm, evaluate_model

# Configure training
config = {
    'data_path': 'data/processed/all_windows.npz',
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'patience': 10,
    'lstm_units': 64,
    'dropout_rate': 0.3,
    'dense_units': 32,
    'augment': True,
    'use_focal_loss': True,
    'use_class_weights': True,
    'save_best': True,
    'checkpoint_dir': 'ml/training/checkpoints',
    'history_dir': 'ml/training/history',
    'random_state': 42
}

# Train model
model, history, (X_test, y_test) = train_lstm(config)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)

print(f"Test F1: {metrics['f1']:.4f}")
print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
```

## Output Files

### Model Checkpoint
- **Path:** `ml/training/checkpoints/lstm_best.h5`
- **Format:** HDF5 (Keras legacy format)
- **Size:** ~272 KB
- Contains best model weights based on validation F1

### Training History
- **Path:** `ml/training/history/lstm_history.csv`
- **Format:** CSV with columns: epoch, loss, accuracy, precision, recall, f1, auc, val_loss, val_accuracy, etc.
- Useful for analyzing training dynamics

### Visualizations
All saved to `docs/wiki_assets/phase2_training/`:

1. **training_history.png** - Loss and F1 curves over epochs
2. **roc_curve.png** - ROC curve with AUC score
3. **confusion_matrix.png** - Confusion matrix heatmap
4. **test_metrics.json** - Test metrics in JSON format

## Module Structure

```
ml/training/
├── __init__.py           # Module exports
├── lstm_train.py         # Main training pipeline (698 lines)
├── augmentation.py       # Data augmentation utilities (230 lines)
└── README.md            # This file
```

## Dependencies

- **TensorFlow 2.15+** (tested with 2.20.0)
- **TensorFlow Addons** (optional, for focal loss)
- **scikit-learn** (for metrics)
- **NumPy** (for data manipulation)
- **SciPy** (for interpolation in time warping)
- **Matplotlib** (for plotting)
- **seaborn** (optional, for better confusion matrix)

## Performance Notes

### Small Dataset Considerations
- Current dataset: N=17 samples
- Very small for deep learning
- Results may not generalize well
- Recommendations:
  - Collect more data (target: 1000+ samples)
  - Use cross-validation
  - Consider transfer learning
  - Apply more aggressive augmentation

### Training Time
- On CPU: ~1-2 seconds per epoch
- On GPU: < 1 second per epoch
- Total training time: < 1 minute (with early stopping)

### Model Size
- Parameters: 20,289
- File size: ~272 KB (HDF5)
- TFLite quantized: < 100 KB (estimated)
- Well within mobile constraints

## Next Steps

1. **Collect More Data:** Expand dataset to 1000+ samples
2. **Cross-Validation:** Implement k-fold CV for robust evaluation
3. **Hyperparameter Tuning:** Grid search for optimal parameters
4. **Model Export:** Convert to TFLite for mobile deployment
5. **Quantization:** Apply post-training quantization
6. **Latency Testing:** Measure inference time on target devices

## References

- Proposal Section 3.4: LSTM Classifier Architecture
- TensorFlow Keras API: https://www.tensorflow.org/api_docs/python/tf/keras
- Focal Loss Paper: https://arxiv.org/abs/1708.02002
- Data Augmentation for Time Series: https://arxiv.org/abs/1706.00527

## License

Part of the mobile-vision-training project.

