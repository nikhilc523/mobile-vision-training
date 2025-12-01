"""
Create Enhanced Dataset with 14 Features and 90-Frame Windows

This script processes keypoints to create an enhanced dataset:
- 14 features (10 original + 4 derived)
- 90-frame windows (3 seconds @ 30fps)
- 15-frame stride (0.5 seconds)

Output: data/processed/all_windows_enhanced.npz
"""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.features.feature_engineering_enhanced import (
    process_video_enhanced,
    get_feature_names
)


def process_dataset_enhanced(source_dir: Path, output_dir: Path,
                             window_length: int = 90, stride: int = 15,
                             drop_threshold: float = 0.5, max_workers: int = 8):
    """
    Process all videos to create enhanced dataset.
    
    Args:
        source_dir: Directory containing keypoints .npz files
        output_dir: Output directory for processed dataset
        window_length: Window size in frames (default 90)
        stride: Stride between windows (default 15)
        drop_threshold: Drop windows with >this fraction missing
        max_workers: Number of parallel workers
        
    Returns:
        Statistics dictionary, X_all, y_all, video_ids_all
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all keypoints files
    npz_files = list(source_dir.rglob('*.npz'))
    print(f"Found {len(npz_files)} keypoints files")
    
    if len(npz_files) == 0:
        print(f"❌ No .npz files found in {source_dir}")
        sys.exit(1)
    
    # Process videos in parallel
    print(f"\nProcessing videos (window_length={window_length}, stride={stride})...")
    
    all_X = []
    all_y = []
    all_video_ids = []
    
    stats = {
        'total_videos': 0,
        'successful_videos': 0,
        'failed_videos': 0,
        'total_windows': 0,
        'total_dropped': 0,
        'total_pos': 0,
        'total_neg': 0,
        'datasets': {}
    }
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_video_enhanced, npz_file, window_length, stride, drop_threshold): npz_file
            for npz_file in npz_files
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(npz_files), desc="Processing"):
            npz_file = future_to_file[future]
            stats['total_videos'] += 1
            
            try:
                result = future.result()
                
                if result['success'] and len(result['X']) > 0:
                    stats['successful_videos'] += 1
                    
                    # Accumulate data
                    all_X.append(result['X'])
                    all_y.append(result['y'])
                    
                    # Create video IDs for subject-wise splitting
                    video_id = result['video_name']
                    video_ids = np.array([video_id] * len(result['y']))
                    all_video_ids.append(video_ids)
                    
                    # Update stats
                    stats['total_windows'] += len(result['y'])
                    stats['total_dropped'] += result['num_dropped']
                    stats['total_pos'] += np.sum(result['y'] == 1)
                    stats['total_neg'] += np.sum(result['y'] == 0)
                    
                    # Dataset stats
                    dataset = result['dataset']
                    if dataset not in stats['datasets']:
                        stats['datasets'][dataset] = {
                            'videos': 0, 'windows': 0, 'pos': 0, 'neg': 0
                        }
                    stats['datasets'][dataset]['videos'] += 1
                    stats['datasets'][dataset]['windows'] += len(result['y'])
                    stats['datasets'][dataset]['pos'] += np.sum(result['y'] == 1)
                    stats['datasets'][dataset]['neg'] += np.sum(result['y'] == 0)
                else:
                    stats['failed_videos'] += 1
                    if not result['success']:
                        print(f"\n⚠️  Failed: {npz_file.name} - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                stats['failed_videos'] += 1
                print(f"\n❌ Error processing {npz_file.name}: {e}")
    
    # Concatenate all data
    print("\nConcatenating data...")
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    video_ids_all = np.concatenate(all_video_ids, axis=0)
    
    # Save dataset
    output_path = output_dir / 'all_windows_enhanced.npz'
    print(f"\nSaving dataset to {output_path}...")
    np.savez_compressed(
        output_path,
        X=X_all,
        y=y_all,
        video_ids=video_ids_all,
        window_length=window_length,
        stride=stride,
        num_features=14,
        feature_names=get_feature_names()
    )
    
    print(f"✅ Saved: {output_path}")
    print(f"   X shape: {X_all.shape}")
    print(f"   y shape: {y_all.shape}")
    print(f"   video_ids shape: {video_ids_all.shape}")
    
    return stats, X_all, y_all, video_ids_all


def print_stats(stats, X_all, y_all):
    """Print dataset statistics."""
    print("\n" + "=" * 70)
    print("Enhanced Dataset Statistics")
    print("=" * 70)
    
    print(f"\nVideos:")
    print(f"  Total: {stats['total_videos']}")
    print(f"  Successful: {stats['successful_videos']}")
    print(f"  Failed: {stats['failed_videos']}")
    
    print(f"\nWindows:")
    print(f"  Generated: {stats['total_windows']:,}")
    print(f"  Dropped: {stats['total_dropped']:,}")
    print(f"  Final: {len(y_all):,}")
    
    print(f"\nClass Distribution:")
    print(f"  Fall: {stats['total_pos']:,} ({stats['total_pos']/stats['total_windows']*100:.1f}%)")
    print(f"  Non-fall: {stats['total_neg']:,} ({stats['total_neg']/stats['total_windows']*100:.1f}%)")
    
    print(f"\nDataset Breakdown:")
    for dataset, ds_stats in stats['datasets'].items():
        print(f"  {dataset}:")
        print(f"    Videos: {ds_stats['videos']}")
        print(f"    Windows: {ds_stats['windows']:,}")
        print(f"    Fall: {ds_stats['pos']:,} ({ds_stats['pos']/ds_stats['windows']*100:.1f}%)")
        print(f"    Non-fall: {ds_stats['neg']:,} ({ds_stats['neg']/ds_stats['windows']*100:.1f}%)")
    
    print(f"\nFeatures:")
    print(f"  Total: {X_all.shape[2]} (10 original + 4 derived)")
    print(f"  Window length: {X_all.shape[1]} frames (3 seconds @ 30fps)")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Create enhanced dataset with 14 features and 90-frame windows'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='data/interim/keypoints',
        help='Source directory with keypoints .npz files'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/processed',
        help='Output directory'
    )
    parser.add_argument(
        '--length',
        type=int,
        default=90,
        help='Window length in frames (default: 90 = 3 seconds)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=15,
        help='Stride between windows (default: 15 = 0.5 seconds)'
    )
    parser.add_argument(
        '--drop-threshold',
        type=float,
        default=0.5,
        help='Drop windows with >this fraction missing'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.out)
    
    if not source_dir.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("Enhanced Dataset Creation - Phase 2.2")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Window length: {args.length} frames (3 seconds @ 30fps)")
    print(f"  Stride: {args.stride} frames (0.5 seconds)")
    print(f"  Drop threshold: {args.drop_threshold}")
    print(f"  Workers: {args.workers}")
    print(f"  Features: 14 (10 original + 4 derived)")
    print()
    
    # Process dataset
    stats, X_all, y_all, video_ids_all = process_dataset_enhanced(
        source_dir,
        output_dir,
        window_length=args.length,
        stride=args.stride,
        drop_threshold=args.drop_threshold,
        max_workers=args.workers
    )
    
    # Print statistics
    print_stats(stats, X_all, y_all)
    
    print("\n✅ Enhanced dataset creation complete!")
    print(f"\nNext step:")
    print(f"  python -m ml.training.lstm_train_enhanced \\")
    print(f"    --data data/processed/all_windows_enhanced.npz \\")
    print(f"    --epochs 80 --batch 32 --lr 5e-4 \\")
    print(f"    --patience 20 --use-focal --augment")


if __name__ == '__main__':
    main()

