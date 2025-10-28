#!/usr/bin/env python3
"""
Generate EDA plots for Phase 1.5 Feature Engineering

This script generates visualizations of the engineered features.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def main():
    """Generate all EDA plots."""
    
    # Setup paths
    data_path = Path('data/processed/all_windows.npz')
    output_dir = Path('docs/wiki_assets/phase1_features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)
    
    print("="*70)
    print("Generating EDA Plots")
    print("="*70)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print("="*70)
    print()
    
    # Load data
    print("Loading data...")
    data = np.load(data_path, allow_pickle=True)
    
    X = data['X']  # (N, 60, 6)
    y = data['y']  # (N,)
    video_names = data['video_names']
    datasets = data['datasets']
    
    print(f"Total windows: {len(X)}")
    print(f"Window shape: {X.shape}")
    print(f"Fall windows: {np.sum(y==1)} ({100*np.mean(y):.1f}%)")
    print(f"Non-fall windows: {np.sum(y==0)} ({100*np.mean(y==0):.1f}%)")
    print()
    
    # Feature names
    feature_names = ['Torso Angle', 'Hip Height', 'Vertical Velocity', 
                     'Motion Magnitude', 'Shoulder Symmetry', 'Knee Angle']
    
    # 1. Class Balance
    print("Generating class balance plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    counts = [np.sum(y==0), np.sum(y==1)]
    labels = ['Non-Fall', 'Fall']
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({100*count/len(y):.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Windows', fontsize=12)
    ax.set_title('Class Balance - 60-Frame Windows', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_balance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: class_balance.png")
    
    # 2. Feature Distributions
    print("Generating feature distributions...")
    X_mean = np.nanmean(X, axis=1)  # (N, 6)
    
    # Separate by class
    X_fall = X_mean[y == 1]
    X_non_fall = X_mean[y == 0]
    
    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        # Plot histograms
        ax.hist(X_non_fall[:, i], bins=20, alpha=0.6, label='Non-Fall', 
                color='#2ecc71', edgecolor='black')
        ax.hist(X_fall[:, i], bins=20, alpha=0.6, label='Fall', 
                color='#e74c3c', edgecolor='black')
        
        ax.set_xlabel(feature_name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{feature_name} Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: feature_distributions.png")
    
    # 3. Temporal Traces
    print("Generating temporal traces...")
    
    # Find example fall and non-fall windows
    fall_idx = np.where(y == 1)[0][0] if np.any(y == 1) else 0
    non_fall_idx = np.where(y == 0)[0][0] if np.any(y == 0) else 0
    
    # Plot temporal traces
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    frames = np.arange(60)
    
    for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        # Plot fall example
        if np.any(y == 1):
            ax.plot(frames, X[fall_idx, :, i], label='Fall Example', 
                    color='#e74c3c', linewidth=2, alpha=0.8)
        
        # Plot non-fall example
        if np.any(y == 0):
            ax.plot(frames, X[non_fall_idx, :, i], label='Non-Fall Example', 
                    color='#2ecc71', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel(feature_name, fontsize=11)
        ax.set_title(f'{feature_name} Over Time', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 59)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_traces.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: temporal_traces.png")
    
    # 4. Feature Statistics
    print()
    print("Feature Statistics (Mean ± Std)")
    print("="*70)
    print(f"{'Feature':<20} {'Fall':<25} {'Non-Fall':<25}")
    print("="*70)
    
    for i, feature_name in enumerate(feature_names):
        fall_mean = np.nanmean(X_fall[:, i])
        fall_std = np.nanstd(X_fall[:, i])
        non_fall_mean = np.nanmean(X_non_fall[:, i])
        non_fall_std = np.nanstd(X_non_fall[:, i])
        
        print(f"{feature_name:<20} {fall_mean:.3f} ± {fall_std:.3f}        {non_fall_mean:.3f} ± {non_fall_std:.3f}")
    
    print("="*70)
    print()
    
    # 5. Data Quality
    print("Data Quality Checks:")
    print(f"  Window length: {X.shape[1]} frames (expected: 60)")
    print(f"  All windows correct length: {X.shape[1] == 60}")
    
    nan_ratio = np.sum(np.isnan(X)) / X.size
    print(f"  NaN ratio: {nan_ratio:.3f} ({100*nan_ratio:.1f}%)")
    
    missing_ratios = data['missing_ratios']
    print(f"  Missing ratio - Mean: {np.mean(missing_ratios):.3f}, Max: {np.max(missing_ratios):.3f}")
    print()
    
    print("="*70)
    print("✓ All plots generated successfully!")
    print("="*70)


if __name__ == '__main__':
    main()

