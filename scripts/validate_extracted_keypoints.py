#!/usr/bin/env python3
"""
Validate Extracted Keypoints

This script validates the extracted pose keypoints from URFD and Le2i datasets.

Checks:
- File existence and format
- Array shapes and data types
- Value ranges (coordinates in [0,1], confidence in [0,1])
- Label validity
- Metadata completeness

Usage:
    python scripts/validate_extracted_keypoints.py
    python scripts/validate_extracted_keypoints.py --verbose
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class KeypointValidator:
    """Validates extracted keypoint files."""
    
    def __init__(self, keypoints_dir: Path, verbose: bool = False):
        """
        Initialize validator.
        
        Args:
            keypoints_dir: Directory containing .npz files
            verbose: Print detailed information
        """
        self.keypoints_dir = Path(keypoints_dir)
        self.verbose = verbose
        
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'urfd_files': 0,
            'le2i_files': 0,
            'total_frames': 0,
            'fall_videos': 0,
            'non_fall_videos': 0,
        }
        
        self.errors = []
    
    def validate_file(self, npz_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate a single .npz file.
        
        Args:
            npz_path: Path to .npz file
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Load file
            data = np.load(npz_path)
            
            # Check required keys
            required_keys = ['keypoints', 'label', 'fps', 'dataset', 'video_name']
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing required key: {key}")
            
            if errors:
                return False, errors
            
            # Validate keypoints
            keypoints = data['keypoints']
            
            # Check shape
            if len(keypoints.shape) != 3:
                errors.append(f"Invalid keypoints shape: {keypoints.shape} (expected 3D)")
            elif keypoints.shape[1] != 17:
                errors.append(f"Invalid number of keypoints: {keypoints.shape[1]} (expected 17)")
            elif keypoints.shape[2] != 3:
                errors.append(f"Invalid keypoint dimensions: {keypoints.shape[2]} (expected 3)")
            
            # Check data type
            if keypoints.dtype not in [np.float32, np.float64]:
                errors.append(f"Invalid keypoints dtype: {keypoints.dtype} (expected float32/float64)")
            
            # Check value ranges
            if not np.all((keypoints >= 0) & (keypoints <= 1)):
                # Allow some tolerance for floating point errors
                if not np.all((keypoints >= -0.01) & (keypoints <= 1.01)):
                    errors.append(f"Keypoints out of range [0,1]: min={keypoints.min():.3f}, max={keypoints.max():.3f}")
            
            # Validate label
            label = data['label']
            if label not in [0, 1]:
                errors.append(f"Invalid label: {label} (expected 0 or 1)")
            
            # Validate FPS
            fps = int(data['fps'])
            if fps <= 0:
                errors.append(f"Invalid FPS: {fps} (expected positive integer)")
            
            # Validate dataset
            dataset = str(data['dataset'])
            if dataset not in ['urfd', 'le2i']:
                errors.append(f"Invalid dataset: {dataset} (expected 'urfd' or 'le2i')")
            
            # Check Le2i-specific fields
            if dataset == 'le2i':
                if 'frame_labels' not in data:
                    errors.append("Missing 'frame_labels' for Le2i video")
                else:
                    frame_labels = data['frame_labels']
                    if len(frame_labels) != keypoints.shape[0]:
                        errors.append(f"Frame labels length mismatch: {len(frame_labels)} vs {keypoints.shape[0]}")
                    if not np.all((frame_labels == 0) | (frame_labels == 1)):
                        errors.append(f"Invalid frame labels: must be 0 or 1")
            
            # Update statistics
            self.stats['total_frames'] += keypoints.shape[0]
            
            if dataset == 'urfd':
                self.stats['urfd_files'] += 1
            else:
                self.stats['le2i_files'] += 1
            
            if label == 1:
                self.stats['fall_videos'] += 1
            else:
                self.stats['non_fall_videos'] += 1
            
            if self.verbose and not errors:
                print(f"✓ {npz_path.name}: {keypoints.shape[0]} frames, label={label}, dataset={dataset}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Failed to load file: {e}")
            return False, errors
    
    def validate_all(self) -> Dict:
        """
        Validate all .npz files in the directory.
        
        Returns:
            Dictionary with validation results
        """
        if not self.keypoints_dir.exists():
            print(f"❌ Error: Directory not found: {self.keypoints_dir}")
            return self.stats
        
        # Find all .npz files
        npz_files = sorted(self.keypoints_dir.glob("*.npz"))
        
        if not npz_files:
            print(f"⚠️  Warning: No .npz files found in {self.keypoints_dir}")
            return self.stats
        
        print(f"Found {len(npz_files)} .npz files")
        print()
        
        # Validate each file
        for npz_path in npz_files:
            self.stats['total_files'] += 1
            
            is_valid, errors = self.validate_file(npz_path)
            
            if is_valid:
                self.stats['valid_files'] += 1
            else:
                self.stats['invalid_files'] += 1
                print(f"❌ {npz_path.name}:")
                for error in errors:
                    print(f"   - {error}")
                self.errors.append((npz_path.name, errors))
        
        return self.stats
    
    def print_summary(self):
        """Print validation summary."""
        print()
        print("="*70)
        print("Validation Summary")
        print("="*70)
        print()
        
        print(f"Total files:      {self.stats['total_files']}")
        print(f"Valid files:      {self.stats['valid_files']} ✓")
        print(f"Invalid files:    {self.stats['invalid_files']} ✗")
        print()
        
        print(f"URFD files:       {self.stats['urfd_files']}")
        print(f"Le2i files:       {self.stats['le2i_files']}")
        print()
        
        print(f"Fall videos:      {self.stats['fall_videos']}")
        print(f"Non-fall videos:  {self.stats['non_fall_videos']}")
        print()
        
        print(f"Total frames:     {self.stats['total_frames']:,}")
        print()
        
        if self.stats['invalid_files'] == 0:
            print("✅ All files are valid!")
        else:
            print(f"⚠️  {self.stats['invalid_files']} files have errors")
            print()
            print("Errors:")
            for filename, errors in self.errors:
                print(f"  {filename}:")
                for error in errors:
                    print(f"    - {error}")
        
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate extracted pose keypoints"
    )
    
    parser.add_argument(
        '--keypoints-dir',
        type=str,
        default='data/interim/keypoints',
        help="Directory containing .npz files (default: data/interim/keypoints)"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Print detailed information for each file"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    keypoints_dir = project_root / args.keypoints_dir
    
    print()
    print("="*70)
    print("Keypoint Validation")
    print("="*70)
    print(f"Directory: {keypoints_dir}")
    print("="*70)
    print()
    
    # Validate
    validator = KeypointValidator(keypoints_dir, verbose=args.verbose)
    validator.validate_all()
    validator.print_summary()
    
    # Exit with error code if any files are invalid
    if validator.stats['invalid_files'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()

