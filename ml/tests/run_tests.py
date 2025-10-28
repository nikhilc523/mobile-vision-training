#!/usr/bin/env python3
"""
Simple test runner that doesn't require pytest.

Usage:
    python3 ml/tests/run_tests.py
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.parsers.le2i_annotations import (
    parse_annotation,
    match_video_for_annotation,
    get_fall_ranges,
)


class TestRunner:
    """Simple test runner."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def assert_equal(self, actual, expected, msg=""):
        """Assert that two values are equal."""
        if actual != expected:
            raise AssertionError(f"{msg}\nExpected: {expected}\nActual: {actual}")
    
    def assert_true(self, condition, msg=""):
        """Assert that condition is true."""
        if not condition:
            raise AssertionError(f"{msg}\nCondition is False")
    
    def run_test(self, test_func, test_name):
        """Run a single test function."""
        try:
            test_func()
            self.passed += 1
            print(f"  ✓ {test_name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((test_name, str(e)))
            print(f"  ✗ {test_name}")
        except Exception as e:
            self.failed += 1
            self.errors.append((test_name, f"Error: {e}"))
            print(f"  ✗ {test_name} (Error)")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print(f"Tests passed: {self.passed}")
        print(f"Tests failed: {self.failed}")
        
        if self.errors:
            print("\nFailures:")
            for test_name, error in self.errors:
                print(f"\n  {test_name}:")
                print(f"    {error}")
        
        print("="*70)
        
        return self.failed == 0


def test_parse_valid_annotation(runner):
    """Test parsing a valid annotation file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ann_file = tmp_path / "video1.txt"
        ann_file.write_text("144\n164\n1,1,205,70,259,170\n2,1,205,70,259,170\n")
        
        result = parse_annotation(str(ann_file))
        
        runner.assert_equal(result, [(144, 164)])


def test_parse_annotation_with_whitespace(runner):
    """Test parsing annotation with extra whitespace."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ann_file = tmp_path / "video2.txt"
        ann_file.write_text("  120  \n  137  \n1,1,212,75,264,153\n")
        
        result = parse_annotation(str(ann_file))
        
        runner.assert_equal(result, [(120, 137)])


def test_parse_annotation_missing_file(runner):
    """Test parsing non-existent file returns empty list."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        result = parse_annotation(str(tmp_path / "nonexistent.txt"))
        
        runner.assert_equal(result, [])


def test_parse_annotation_invalid_format(runner):
    """Test parsing file with less than 2 lines."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ann_file = tmp_path / "invalid.txt"
        ann_file.write_text("144\n")
        
        result = parse_annotation(str(ann_file))
        
        runner.assert_equal(result, [])


def test_parse_annotation_invalid_numbers(runner):
    """Test parsing file with non-numeric frame numbers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ann_file = tmp_path / "invalid.txt"
        ann_file.write_text("abc\n164\n1,1,205,70,259,170\n")
        
        result = parse_annotation(str(ann_file))
        
        runner.assert_equal(result, [])


def test_match_video_standard_structure(runner):
    """Test matching video in standard Videos/ folder structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        scene_dir = tmp_path / "Home_01"
        ann_dir = scene_dir / "Annotation_files"
        videos_dir = scene_dir / "Videos"
        
        ann_dir.mkdir(parents=True)
        videos_dir.mkdir(parents=True)
        
        ann_file = ann_dir / "video (1).txt"
        ann_file.write_text("144\n164\n")
        
        video_file = videos_dir / "video (1).avi"
        video_file.write_text("dummy video")
        
        result = match_video_for_annotation(str(ann_file))
        
        runner.assert_equal(result, video_file)


def test_match_video_not_found(runner):
    """Test when video file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        scene_dir = tmp_path / "Home_01"
        ann_dir = scene_dir / "Annotation_files"
        videos_dir = scene_dir / "Videos"
        
        ann_dir.mkdir(parents=True)
        videos_dir.mkdir(parents=True)
        
        ann_file = ann_dir / "video (1).txt"
        ann_file.write_text("144\n164\n")
        
        result = match_video_for_annotation(str(ann_file))
        
        runner.assert_equal(result, None)


def test_get_fall_ranges_with_annotations(runner):
    """Test getting fall ranges for scene with annotations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        scene_dir = tmp_path / "Home_01"
        ann_dir = scene_dir / "Annotation_files"
        videos_dir = scene_dir / "Videos"
        
        ann_dir.mkdir(parents=True)
        videos_dir.mkdir(parents=True)
        
        for i in [1, 2, 3]:
            video_file = videos_dir / f"video ({i}).avi"
            video_file.write_text("dummy")
            
            ann_file = ann_dir / f"video ({i}).txt"
            start = 100 + i * 10
            end = start + 20
            ann_file.write_text(f"{start}\n{end}\n1,1,0,0,0,0\n")
        
        result = get_fall_ranges(str(scene_dir))
        
        runner.assert_equal(len(result), 3)
        runner.assert_equal(result["video (1).avi"], [(110, 130)])
        runner.assert_equal(result["video (2).avi"], [(120, 140)])
        runner.assert_equal(result["video (3).avi"], [(130, 150)])


def test_get_fall_ranges_without_annotations(runner):
    """Test getting fall ranges for videos without annotations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        scene_dir = tmp_path / "Lecture_room"
        
        scene_dir.mkdir(parents=True)
        
        for i in [1, 2, 3]:
            video_file = scene_dir / f"video ({i}).avi"
            video_file.write_text("dummy")
        
        result = get_fall_ranges(str(scene_dir))
        
        runner.assert_equal(len(result), 3)
        runner.assert_equal(result["video (1).avi"], [])
        runner.assert_equal(result["video (2).avi"], [])
        runner.assert_equal(result["video (3).avi"], [])


def test_real_home_01_data(runner):
    """Test with real Home_01 data (if available)."""
    if not Path("data/raw/le2i/Home_01").exists():
        print("    (skipped - data not available)")
        return
    
    result = get_fall_ranges("data/raw/le2i/Home_01")
    
    runner.assert_true(len(result) > 0, "Should have videos")
    
    if "video (1).avi" in result:
        runner.assert_true(len(result["video (1).avi"]) > 0, "Should have fall ranges")
        start, end = result["video (1).avi"][0]
        runner.assert_true(start > 0, "Start frame should be positive")
        runner.assert_true(end > start, "End frame should be greater than start")


def main():
    """Run all tests."""
    print("🧪 Running Le2i Annotation Parser Tests")
    print("="*70)
    
    runner = TestRunner()
    
    # Parse annotation tests
    print("\n📝 Testing parse_annotation():")
    runner.run_test(lambda: test_parse_valid_annotation(runner), "parse_valid_annotation")
    runner.run_test(lambda: test_parse_annotation_with_whitespace(runner), "parse_annotation_with_whitespace")
    runner.run_test(lambda: test_parse_annotation_missing_file(runner), "parse_annotation_missing_file")
    runner.run_test(lambda: test_parse_annotation_invalid_format(runner), "parse_annotation_invalid_format")
    runner.run_test(lambda: test_parse_annotation_invalid_numbers(runner), "parse_annotation_invalid_numbers")
    
    # Match video tests
    print("\n🎥 Testing match_video_for_annotation():")
    runner.run_test(lambda: test_match_video_standard_structure(runner), "match_video_standard_structure")
    runner.run_test(lambda: test_match_video_not_found(runner), "match_video_not_found")
    
    # Get fall ranges tests
    print("\n📊 Testing get_fall_ranges():")
    runner.run_test(lambda: test_get_fall_ranges_with_annotations(runner), "get_fall_ranges_with_annotations")
    runner.run_test(lambda: test_get_fall_ranges_without_annotations(runner), "get_fall_ranges_without_annotations")
    
    # Real data tests
    print("\n🔍 Testing with real data:")
    runner.run_test(lambda: test_real_home_01_data(runner), "real_home_01_data")
    
    # Print summary
    success = runner.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

