#!/usr/bin/env python3
"""
Safety test for validate_and_cleanup_datasets.py

This script verifies that the cleanup script never deletes valid dataset files.
"""

import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_and_cleanup_datasets import DatasetValidator


def test_cleanup_safety():
    """Test that cleanup never deletes valid files."""
    print("🧪 Testing cleanup safety...")
    
    # Create temporary test directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create test structure
        urfd_fall = tmp_path / 'urfd' / 'falls' / 'fall-01-cam0-rgb'
        urfd_fall.mkdir(parents=True)
        
        le2i_home = tmp_path / 'le2i' / 'Home_01'
        le2i_videos = le2i_home / 'Videos'
        le2i_ann = le2i_home / 'Annotation_files'
        le2i_videos.mkdir(parents=True)
        le2i_ann.mkdir(parents=True)
        
        # Create valid files (should NOT be deleted)
        valid_files = [
            urfd_fall / 'fall-01-cam0-rgb-001.png',
            urfd_fall / 'fall-01-cam0-rgb-002.png',
            le2i_videos / 'video (1).avi',
            le2i_videos / 'video (2).avi',
            le2i_ann / 'video (1).txt',
            le2i_ann / 'video (2).txt',
        ]
        
        for f in valid_files:
            f.write_text('test content')
        
        # Create cleanup candidates (SHOULD be deleted)
        cleanup_files = [
            tmp_path / '.DS_Store',
            tmp_path / 'urfd' / '.DS_Store',
            tmp_path / 'urfd' / 'falls' / 'fall-01-cam0-rgb.zip',
            tmp_path / 'urfd' / 'falls' / 'fall-02-cam0-rgb.zip',
            tmp_path / 'le2i' / 'backup~',
            tmp_path / 'le2i' / 'temp.tmp',
        ]
        
        for f in cleanup_files:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text('cleanup me')
        
        # Run validator
        validator = DatasetValidator(tmp_path, dry_run=False)
        
        # Validate datasets
        validator.validate_urfd_dataset()
        validator.validate_le2i_dataset()
        
        # Find cleanup candidates
        validator.find_cleanup_candidates()
        
        # Execute cleanup
        validator.cleanup(force=True)
        
        # Verify valid files still exist
        print("\n✅ Checking valid files are preserved:")
        all_valid = True
        for f in valid_files:
            if f.exists():
                print(f"  ✓ {f.name} - preserved")
            else:
                print(f"  ✗ {f.name} - DELETED (ERROR!)")
                all_valid = False
        
        # Verify cleanup files are deleted
        print("\n✅ Checking cleanup files are removed:")
        all_cleaned = True
        for f in cleanup_files:
            if not f.exists():
                print(f"  ✓ {f.name} - deleted")
            else:
                print(f"  ✗ {f.name} - still exists (ERROR!)")
                all_cleaned = False
        
        # Final result
        print("\n" + "="*70)
        if all_valid and all_cleaned:
            print("✅ SAFETY TEST PASSED")
            print("   - All valid files preserved")
            print("   - All cleanup files removed")
            return True
        else:
            print("❌ SAFETY TEST FAILED")
            if not all_valid:
                print("   - Some valid files were deleted!")
            if not all_cleaned:
                print("   - Some cleanup files remain")
            return False


def test_dry_run_safety():
    """Test that dry-run mode doesn't delete anything."""
    print("\n🧪 Testing dry-run safety...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create test files
        test_file = tmp_path / 'test.zip'
        test_file.write_text('test')
        
        # Run validator in dry-run mode
        validator = DatasetValidator(tmp_path, dry_run=True)
        validator.find_cleanup_candidates()
        validator.cleanup(force=True)
        
        # Verify file still exists
        if test_file.exists():
            print("  ✅ Dry-run mode: No files deleted")
            return True
        else:
            print("  ❌ Dry-run mode: File was deleted!")
            return False


def main():
    """Run all safety tests."""
    print("="*70)
    print("🔒 CLEANUP SCRIPT SAFETY TESTS")
    print("="*70)
    
    test1 = test_cleanup_safety()
    test2 = test_dry_run_safety()
    
    print("\n" + "="*70)
    if test1 and test2:
        print("✅ ALL SAFETY TESTS PASSED")
        print("="*70)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())

