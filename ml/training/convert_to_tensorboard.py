#!/usr/bin/env python3
"""
Convert training history CSV to TensorBoard logs
This creates actual TensorBoard logs that you can view with `tensorboard --logdir=logs`
"""

import pandas as pd
import tensorflow as tf
from datetime import datetime
import os

# Read training history
history_file = 'docs/wiki_assets/phase4_balanced_training_v2/training_history.csv'
df = pd.read_csv(history_file)

# Create TensorBoard log directory
log_dir = 'logs/bilstm_fall_detection_' + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

print(f"📊 Converting training history to TensorBoard logs...")
print(f"📁 Input: {history_file}")
print(f"📁 Output: {log_dir}")
print(f"📈 Epochs: {len(df)}")

# Create TensorBoard writer
writer = tf.summary.create_file_writer(log_dir)

# Write all metrics to TensorBoard
with writer.as_default():
    for epoch, row in df.iterrows():
        # Training metrics
        tf.summary.scalar('epoch_loss/train', row['loss'], step=epoch)
        tf.summary.scalar('epoch_accuracy/train', row['accuracy'], step=epoch)
        tf.summary.scalar('epoch_auc/train', row['auc'], step=epoch)
        tf.summary.scalar('epoch_precision/train', row['precision'], step=epoch)
        tf.summary.scalar('epoch_recall/train', row['recall'], step=epoch)
        
        # Validation metrics
        tf.summary.scalar('epoch_loss/validation', row['val_loss'], step=epoch)
        tf.summary.scalar('epoch_accuracy/validation', row['val_accuracy'], step=epoch)
        tf.summary.scalar('epoch_auc/validation', row['val_auc'], step=epoch)
        tf.summary.scalar('epoch_precision/validation', row['val_precision'], step=epoch)
        tf.summary.scalar('epoch_recall/validation', row['val_recall'], step=epoch)
        
        # Learning rate
        tf.summary.scalar('learning_rate', row['learning_rate'], step=epoch)
        
        # F1 Score (calculated from precision and recall)
        train_f1 = 2 * (row['precision'] * row['recall']) / (row['precision'] + row['recall'] + 1e-7)
        val_f1 = 2 * (row['val_precision'] * row['val_recall']) / (row['val_precision'] + row['val_recall'] + 1e-7)
        tf.summary.scalar('epoch_f1/train', train_f1, step=epoch)
        tf.summary.scalar('epoch_f1/validation', val_f1, step=epoch)

writer.close()

print(f"\n✅ TensorBoard logs created successfully!")
print(f"\n🚀 To view in TensorBoard, run:")
print(f"   tensorboard --logdir=logs")
print(f"\n📊 Then open your browser to: http://localhost:6006")
print(f"\n💡 You'll see:")
print(f"   - Loss curves (train vs validation)")
print(f"   - Accuracy curves (train vs validation)")
print(f"   - AUC curves (train vs validation)")
print(f"   - Precision curves (train vs validation)")
print(f"   - Recall curves (train vs validation)")
print(f"   - F1 Score curves (train vs validation)")
print(f"   - Learning rate schedule")

