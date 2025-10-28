#!/usr/bin/env python3
"""
Verify Extraction Integrity

Validates the integrity of extracted pose keypoint files.
Checks for:
- File format validity
- Keypoint shape correctness
- Required metadata presence
- Data quality metrics

Usage:
    python scripts/verify_extraction_integrity.py
    python scripts/verify_extraction_integrity.py --dataset urfd
    python scripts/verify_extraction_integrity.py --dataset le2i
    python scripts/verify_extraction_integrity.py --verbose
"""

import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import sys


def verify_file(filepath: Path, verbose: bool = False) -> Tuple[bool, str]:
    """
    Verify a single .npz file.
    
    Args:
        filepath: Path to .npz file
        verbose: Print detailed information
        
    Returns:
        (is_valid, error_message)
    """
    try:
        # Load file
        data = np.load(filepath)
        
        # Check required keys
        required_keys = ['keypoints', 'label', 'fps', 'dataset']
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            return False, f"Missing keys: {missing_keys}"
        
        # Check keypoints shape
        keypoints = data['keypoints']
        if len(keypoints.shape) != 3:
            return False, f"Invalid keypoints shape: {keypoints.shape} (expected 3D)"
        
        T, num_kpts, coords = keypoints.shape
        if num_kpts != 17:
            return False, f"Invalid number of keypoints: {num_kpts} (expected 17)"
        
        if coords != 3:
            return False, f"Invalid coordinates: {coords} (expected 3: y, x, confidence)"
        
        if T == 0:
            return False, "Empty sequence (0 frames)"
        
        # Check label
        label = data['label']
        if label not in [0, 1]:
            return False, f"Invalid label: {label} (expected 0 or 1)"
        
        # Check FPS
        fps = data['fps']
        if fps <= 0 or fps > 120:
            return False, f"Invalid FPS: {fps} (expected 0 < fps <= 120)"
        
        # Check dataset name
        dataset = str(data['dataset'])
        if dataset not in ['urfd', 'le2i', 'ucf101']:
            return False, f"Invalid dataset: {dataset} (expected urfd, le2i, or ucf101)"
        
        # Check data quality
        # Confidence values should be in [0, 1]
        confidences = keypoints[:, :, 2]
        if np.any(confidences < 0) or np.any(confidences > 1):
            return False, f"Invalid confidence values (expected [0, 1])"
        
        # Coordinates should be in [0, 1] (normalized)
        coords_y = keypoints[:, :, 0]
        coords_x = keypoints[:, :, 1]
        
        # Allow some tolerance for coordinates (they might be slightly outside [0,1] due to padding)
        if np.any(coords_y < -0.1) or np.any(coords_y > 1.1):
            return False, f"Invalid y coordinates (expected ~[0, 1])"
        
        if np.any(coords_x < -0.1) or np.any(coords_x > 1.1):
            return False, f"Invalid x coordinates (expected ~[0, 1])"
        
        if verbose:
            print(f"  ✓ {filepath.name}: {T} frames, label={label}, fps={fps:.1f}, dataset={dataset}")
        
        return True, ""
        
    except Exception as e:
        return False, f"Error loading file: {str(e)}"


def verify_dataset(keypoints_dir: Path, dataset_filter: str = None, verbose: bool = False) -> Dict:
    """
    Verify all keypoint files in a directory.
    
    Args:
        keypoints_dir: Directory containing .npz files
        dataset_filter: Filter by dataset name (urfd, le2i, ucf101, or None for all)
        verbose: Print detailed information
        
    Returns:
        Dictionary with verification results
    """
    print("="*70)
    print("EXTRACTION INTEGRITY VERIFICATION")
    print("="*70)
    
    # Find files
    if dataset_filter:
        pattern = f"{dataset_filter}_*.npz"
        print(f"\nDataset filter: {dataset_filter}")
    else:
        pattern = "*.npz"
        print(f"\nVerifying all datasets")
    
    files = sorted(keypoints_dir.glob(pattern))
    
    if not files:
        print(f"\n❌ No files found matching pattern: {pattern}")
        return {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'success_rate': 0.0,
            'errors': []
        }
    
    print(f"Found {len(files)} files to verify")
    print()
    
    # Verify each file
    valid_count = 0
    invalid_count = 0
    errors = []
    
    for filepath in files:
        is_valid, error_msg = verify_file(filepath, verbose=verbose)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            errors.append((filepath.name, error_msg))
            if not verbose:
                print(f"  ❌ {filepath.name}: {error_msg}")
    
    # Print summary
    print()
    print("="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    total = len(files)
    success_rate = (valid_count / total * 100) if total > 0 else 0
    
    print(f"\nTotal files: {total}")
    print(f"Valid files: {valid_count} ({valid_count/total*100:.1f}%)")
    print(f"Invalid files: {invalid_count} ({invalid_count/total*100:.1f}%)")
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("\n✅ PASS: Success rate >= 95%")
        status = "PASS"
    else:
        print(f"\n❌ FAIL: Success rate < 95% (got {success_rate:.1f}%)")
        status = "FAIL"
    
    if errors and not verbose:
        print(f"\nErrors found in {len(errors)} files:")
        for filename, error_msg in errors[:10]:  # Show first 10 errors
            print(f"  - {filename}: {error_msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print("="*70)
    
    return {
        'total': total,
        'valid': valid_count,
        'invalid': invalid_count,
        'success_rate': success_rate,
        'errors': errors,
        'status': status
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify integrity of extracted pose keypoint files"
    )
    
    parser.add_argument(
        '--keypoints-dir',
        type=str,
        default='data/interim/keypoints',
        help="Directory containing .npz keypoint files (default: data/interim/keypoints)"
    )
    
    parser.add_argument(
        '--dataset',
        choices=['urfd', 'le2i', 'ucf101'],
        default=None,
        help="Filter by dataset (default: verify all)"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Print detailed information for each file"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    keypoints_dir = project_root / args.keypoints_dir
    
    if not keypoints_dir.exists():
        print(f"❌ Error: Keypoints directory not found: {keypoints_dir}")
        sys.exit(1)
    
    # Run verification
    results = verify_dataset(keypoints_dir, args.dataset, args.verbose)
    
    # Exit with appropriate code
    if results['status'] == 'PASS':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

