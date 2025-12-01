#!/usr/bin/env python3
"""
Generate TensorBoard-style training graphs from training history CSV
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read training history
history_file = 'docs/wiki_assets/phase4_balanced_training_v2/training_history.csv'
df = pd.read_csv(history_file)

# Create output directory
output_dir = 'docs/wiki_assets/tensorboard_graphs'
os.makedirs(output_dir, exist_ok=True)

print(f"📊 Generating TensorBoard-style training graphs...")
print(f"📁 Input: {history_file}")
print(f"📁 Output: {output_dir}/")
print(f"📈 Epochs: {len(df)}")

# Set style for professional look
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'train': '#1f77b4',  # Blue
    'val': '#ff7f0e',    # Orange
    'grid': '#cccccc'
}

# ============================================================================
# 1. LOSS CURVES (Training vs Validation)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

epochs = range(1, len(df) + 1)
ax.plot(epochs, df['loss'], color=colors['train'], linewidth=2, label='Training Loss', marker='o', markersize=3)
ax.plot(epochs, df['val_loss'], color=colors['val'], linewidth=2, label='Validation Loss', marker='s', markersize=3)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss (Binary Crossentropy)', fontsize=14, fontweight='bold')
ax.set_title('Training and Validation Loss Over Time', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, color=colors['grid'])
ax.set_xlim(0, len(df) + 1)

# Add annotations for key points
min_val_loss_idx = df['val_loss'].idxmin()
min_val_loss = df['val_loss'].min()
ax.annotate(f'Best Val Loss\n{min_val_loss:.4f} (Epoch {min_val_loss_idx + 1})',
            xy=(min_val_loss_idx + 1, min_val_loss),
            xytext=(min_val_loss_idx + 1 + 10, min_val_loss + 0.05),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{output_dir}/loss_curves.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir}/loss_curves.png")
plt.close()

# ============================================================================
# 2. ACCURACY CURVES (Training vs Validation)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(epochs, df['accuracy'] * 100, color=colors['train'], linewidth=2, label='Training Accuracy', marker='o', markersize=3)
ax.plot(epochs, df['val_accuracy'] * 100, color=colors['val'], linewidth=2, label='Validation Accuracy', marker='s', markersize=3)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Training and Validation Accuracy Over Time', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3, color=colors['grid'])
ax.set_xlim(0, len(df) + 1)
ax.set_ylim(80, 101)

# Add annotations for key points
max_val_acc_idx = df['val_accuracy'].idxmax()
max_val_acc = df['val_accuracy'].max() * 100
ax.annotate(f'Best Val Accuracy\n{max_val_acc:.2f}% (Epoch {max_val_acc_idx + 1})',
            xy=(max_val_acc_idx + 1, max_val_acc),
            xytext=(max_val_acc_idx + 1 - 15, max_val_acc - 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{output_dir}/accuracy_curves.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir}/accuracy_curves.png")
plt.close()

# ============================================================================
# 3. COMBINED: LOSS + ACCURACY (2x2 Grid)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top-left: Loss
ax = axes[0, 0]
ax.plot(epochs, df['loss'], color=colors['train'], linewidth=2, label='Training', marker='o', markersize=2)
ax.plot(epochs, df['val_loss'], color=colors['val'], linewidth=2, label='Validation', marker='s', markersize=2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Loss Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Top-right: Accuracy
ax = axes[0, 1]
ax.plot(epochs, df['accuracy'] * 100, color=colors['train'], linewidth=2, label='Training', marker='o', markersize=2)
ax.plot(epochs, df['val_accuracy'] * 100, color=colors['val'], linewidth=2, label='Validation', marker='s', markersize=2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom-left: AUC
ax = axes[1, 0]
ax.plot(epochs, df['auc'] * 100, color=colors['train'], linewidth=2, label='Training', marker='o', markersize=2)
ax.plot(epochs, df['val_auc'] * 100, color=colors['val'], linewidth=2, label='Validation', marker='s', markersize=2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('ROC-AUC (%)', fontsize=12, fontweight='bold')
ax.set_title('ROC-AUC Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom-right: Learning Rate
ax = axes[1, 1]
ax.plot(epochs, df['learning_rate'], color='green', linewidth=2, marker='o', markersize=3)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.suptitle('BiLSTM Fall Detection Training History (74 Epochs)', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{output_dir}/training_history_combined.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir}/training_history_combined.png")
plt.close()

print(f"\n🎉 All graphs generated successfully!")
print(f"\n📊 Generated Files:")
print(f"   1. {output_dir}/loss_curves.png")
print(f"   2. {output_dir}/accuracy_curves.png")
print(f"   3. {output_dir}/training_history_combined.png")

