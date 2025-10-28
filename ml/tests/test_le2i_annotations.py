"""
Tests for Le2i annotation parser.

Run with:
    pytest ml/tests/test_le2i_annotations.py
    pytest ml/tests/test_le2i_annotations.py -v
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from ml.data.parsers.le2i_annotations import (
    parse_annotation,
    match_video_for_annotation,
    get_fall_ranges,
)


class TestParseAnnotation:
    """Tests for parse_annotation function."""
    
    def test_parse_valid_annotation(self, tmp_path):
        """Test parsing a valid annotation file."""
        ann_file = tmp_path / "video1.txt"
        ann_file.write_text("144\n164\n1,1,205,70,259,170\n2,1,205,70,259,170\n")
        
        result = parse_annotation(str(ann_file))
        
        assert result == [(144, 164)]
    
    def test_parse_annotation_with_whitespace(self, tmp_path):
        """Test parsing annotation with extra whitespace."""
        ann_file = tmp_path / "video2.txt"
        ann_file.write_text("  120  \n  137  \n1,1,212,75,264,153\n")
        
        result = parse_annotation(str(ann_file))
        
        assert result == [(120, 137)]
    
    def test_parse_annotation_missing_file(self, tmp_path):
        """Test parsing non-existent file returns empty list."""
        result = parse_annotation(str(tmp_path / "nonexistent.txt"))
        
        assert result == []
    
    def test_parse_annotation_invalid_format_too_few_lines(self, tmp_path):
        """Test parsing file with less than 2 lines."""
        ann_file = tmp_path / "invalid.txt"
        ann_file.write_text("144\n")
        
        result = parse_annotation(str(ann_file))
        
        assert result == []
    
    def test_parse_annotation_invalid_numbers(self, tmp_path):
        """Test parsing file with non-numeric frame numbers."""
        ann_file = tmp_path / "invalid.txt"
        ann_file.write_text("abc\n164\n1,1,205,70,259,170\n")
        
        result = parse_annotation(str(ann_file))
        
        assert result == []
    
    def test_parse_annotation_negative_frames(self, tmp_path):
        """Test parsing file with negative frame numbers."""
        ann_file = tmp_path / "invalid.txt"
        ann_file.write_text("-10\n164\n1,1,205,70,259,170\n")
        
        result = parse_annotation(str(ann_file))
        
        assert result == []
    
    def test_parse_annotation_zero_frames(self, tmp_path):
        """Test parsing file with zero frame numbers."""
        ann_file = tmp_path / "invalid.txt"
        ann_file.write_text("0\n164\n1,1,205,70,259,170\n")
        
        result = parse_annotation(str(ann_file))
        
        assert result == []
    
    def test_parse_annotation_start_greater_than_end(self, tmp_path):
        """Test parsing file where start frame > end frame."""
        ann_file = tmp_path / "invalid.txt"
        ann_file.write_text("200\n100\n1,1,205,70,259,170\n")
        
        result = parse_annotation(str(ann_file))
        
        assert result == []
    
    def test_parse_annotation_large_frame_numbers(self, tmp_path):
        """Test parsing file with large frame numbers."""
        ann_file = tmp_path / "video.txt"
        ann_file.write_text("1088\n1116\n1,1,205,70,259,170\n")
        
        result = parse_annotation(str(ann_file))
        
        assert result == [(1088, 1116)]


class TestMatchVideoForAnnotation:
    """Tests for match_video_for_annotation function."""
    
    def test_match_video_standard_structure(self, tmp_path):
        """Test matching video in standard Videos/ folder structure."""
        # Create directory structure
        scene_dir = tmp_path / "Home_01"
        ann_dir = scene_dir / "Annotation_files"
        videos_dir = scene_dir / "Videos"
        
        ann_dir.mkdir(parents=True)
        videos_dir.mkdir(parents=True)
        
        # Create files
        ann_file = ann_dir / "video (1).txt"
        ann_file.write_text("144\n164\n")
        
        video_file = videos_dir / "video (1).avi"
        video_file.write_text("dummy video")
        
        # Test matching
        result = match_video_for_annotation(str(ann_file))
        
        assert result == video_file
    
    def test_match_video_with_spaces_in_name(self, tmp_path):
        """Test matching video with spaces in filename."""
        scene_dir = tmp_path / "Coffee_room_01"
        ann_dir = scene_dir / "Annotation_files"
        videos_dir = scene_dir / "Videos"
        
        ann_dir.mkdir(parents=True)
        videos_dir.mkdir(parents=True)
        
        ann_file = ann_dir / "video (25).txt"
        ann_file.write_text("137\n179\n")
        
        video_file = videos_dir / "video (25).avi"
        video_file.write_text("dummy video")
        
        result = match_video_for_annotation(str(ann_file))
        
        assert result == video_file
    
    def test_match_video_direct_in_scene_dir(self, tmp_path):
        """Test matching video directly in scene directory (no Videos/ folder)."""
        scene_dir = tmp_path / "Lecture_room"
        ann_dir = scene_dir / "Annotation_files"
        
        ann_dir.mkdir(parents=True)
        
        ann_file = ann_dir / "video (1).txt"
        ann_file.write_text("100\n120\n")
        
        video_file = scene_dir / "video (1).avi"
        video_file.write_text("dummy video")
        
        result = match_video_for_annotation(str(ann_file))
        
        assert result == video_file
    
    def test_match_video_not_found(self, tmp_path):
        """Test when video file doesn't exist."""
        scene_dir = tmp_path / "Home_01"
        ann_dir = scene_dir / "Annotation_files"
        videos_dir = scene_dir / "Videos"
        
        ann_dir.mkdir(parents=True)
        videos_dir.mkdir(parents=True)
        
        ann_file = ann_dir / "video (1).txt"
        ann_file.write_text("144\n164\n")
        
        # No video file created
        
        result = match_video_for_annotation(str(ann_file))
        
        assert result is None
    
    def test_match_video_annotation_not_found(self, tmp_path):
        """Test when annotation file doesn't exist."""
        result = match_video_for_annotation(str(tmp_path / "nonexistent.txt"))
        
        assert result is None
    
    def test_match_video_no_videos_folder(self, tmp_path):
        """Test when Videos/ folder doesn't exist and no direct videos."""
        scene_dir = tmp_path / "Home_01"
        ann_dir = scene_dir / "Annotation_files"
        
        ann_dir.mkdir(parents=True)
        
        ann_file = ann_dir / "video (1).txt"
        ann_file.write_text("144\n164\n")
        
        result = match_video_for_annotation(str(ann_file))
        
        assert result is None


class TestGetFallRanges:
    """Tests for get_fall_ranges function."""
    
    def test_get_fall_ranges_with_annotations(self, tmp_path):
        """Test getting fall ranges for scene with annotations."""
        scene_dir = tmp_path / "Home_01"
        ann_dir = scene_dir / "Annotation_files"
        videos_dir = scene_dir / "Videos"
        
        ann_dir.mkdir(parents=True)
        videos_dir.mkdir(parents=True)
        
        # Create video and annotation files
        for i in [1, 2, 3]:
            video_file = videos_dir / f"video ({i}).avi"
            video_file.write_text("dummy")
            
            ann_file = ann_dir / f"video ({i}).txt"
            start = 100 + i * 10
            end = start + 20
            ann_file.write_text(f"{start}\n{end}\n1,1,0,0,0,0\n")
        
        result = get_fall_ranges(str(scene_dir))
        
        assert len(result) == 3
        assert result["video (1).avi"] == [(110, 130)]
        assert result["video (2).avi"] == [(120, 140)]
        assert result["video (3).avi"] == [(130, 150)]
    
    def test_get_fall_ranges_without_annotations(self, tmp_path):
        """Test getting fall ranges for videos without annotations."""
        scene_dir = tmp_path / "Lecture_room"
        
        scene_dir.mkdir(parents=True)
        
        # Create video files without annotations
        for i in [1, 2, 3]:
            video_file = scene_dir / f"video ({i}).avi"
            video_file.write_text("dummy")
        
        result = get_fall_ranges(str(scene_dir))
        
        assert len(result) == 3
        assert result["video (1).avi"] == []
        assert result["video (2).avi"] == []
        assert result["video (3).avi"] == []
    
    def test_get_fall_ranges_mixed(self, tmp_path):
        """Test getting fall ranges with some videos having annotations."""
        scene_dir = tmp_path / "Office"
        ann_dir = scene_dir / "Annotation_files"
        videos_dir = scene_dir / "Videos"
        
        ann_dir.mkdir(parents=True)
        videos_dir.mkdir(parents=True)
        
        # Create 3 videos, but only 2 have annotations
        for i in [1, 2, 3]:
            video_file = videos_dir / f"video ({i}).avi"
            video_file.write_text("dummy")
        
        # Only create annotations for video 1 and 3
        for i in [1, 3]:
            ann_file = ann_dir / f"video ({i}).txt"
            start = 100 + i * 10
            end = start + 20
            ann_file.write_text(f"{start}\n{end}\n1,1,0,0,0,0\n")
        
        result = get_fall_ranges(str(scene_dir))
        
        assert len(result) == 3
        assert result["video (1).avi"] == [(110, 130)]
        assert result["video (2).avi"] == []  # No annotation
        assert result["video (3).avi"] == [(130, 150)]
    
    def test_get_fall_ranges_alternative_spelling(self, tmp_path):
        """Test with Annotations_files (alternative spelling)."""
        scene_dir = tmp_path / "Coffee_room_02"
        ann_dir = scene_dir / "Annotations_files"  # Note: Annotations not Annotation
        videos_dir = scene_dir / "Videos"
        
        ann_dir.mkdir(parents=True)
        videos_dir.mkdir(parents=True)
        
        video_file = videos_dir / "video (1).avi"
        video_file.write_text("dummy")
        
        ann_file = ann_dir / "video (1).txt"
        ann_file.write_text("50\n75\n1,1,0,0,0,0\n")
        
        result = get_fall_ranges(str(scene_dir))
        
        assert len(result) == 1
        assert result["video (1).avi"] == [(50, 75)]
    
    def test_get_fall_ranges_empty_directory(self, tmp_path):
        """Test with empty scene directory."""
        scene_dir = tmp_path / "Empty"
        scene_dir.mkdir()
        
        result = get_fall_ranges(str(scene_dir))
        
        assert result == {}
    
    def test_get_fall_ranges_nonexistent_directory(self, tmp_path):
        """Test with non-existent directory."""
        result = get_fall_ranges(str(tmp_path / "nonexistent"))
        
        assert result == {}


# Integration tests with real data (if available)
class TestRealData:
    """Integration tests with real Le2i data."""
    
    @pytest.mark.skipif(
        not Path("data/raw/le2i/Home_01").exists(),
        reason="Real Le2i data not available"
    )
    def test_real_home_01_data(self):
        """Test with real Home_01 data."""
        result = get_fall_ranges("data/raw/le2i/Home_01")
        
        # Home_01 should have 30 videos, all with falls
        assert len(result) > 0
        
        # Check that video (1).avi exists and has fall range
        if "video (1).avi" in result:
            assert len(result["video (1).avi"]) > 0
            start, end = result["video (1).avi"][0]
            assert start > 0
            assert end > start
    
    @pytest.mark.skipif(
        not Path("data/raw/le2i/Lecture room").exists(),
        reason="Real Le2i data not available"
    )
    def test_real_lecture_room_data(self):
        """Test with real Lecture room data (no annotations)."""
        result = get_fall_ranges("data/raw/le2i/Lecture room")
        
        # Lecture room should have videos but no falls
        assert len(result) > 0
        
        # All videos should have empty fall ranges
        for video_name, fall_ranges in result.items():
            assert fall_ranges == []

