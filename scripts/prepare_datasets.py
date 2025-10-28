#!/usr/bin/env python3
"""
Prepare and Normalize Fall-Detection Datasets (URFD + Le2i)

This script:
1. Unzips all files under data/raw/urfd/adl/ and data/raw/urfd/fall/
2. Flattens any double nesting (e.g., fall-01/fall-01/* -> fall-01/*)
3. Verifies Le2i folder structure
4. Prints a summary table of file counts
5. Updates docs/dataset_notes.md with the counts
"""

import os
import zipfile
import shutil
from pathlib import Path
from datetime import datetime


class DatasetPreparer:
    """Handles dataset preparation and normalization."""
    
    def __init__(self, base_path: str = "data/raw"):
        self.base_path = Path(base_path)
        self.urfd_path = self.base_path / "urfd"
        self.le2i_path = self.base_path / "le2i"
        self.stats = {
            "urfd_fall": 0,
            "urfd_adl": 0,
            "le2i_scenes": 0
        }
    
    def unzip_urfd_files(self, split: str):
        """Unzip all .zip files in URFD split directory."""
        split_path = self.urfd_path / split
        
        if not split_path.exists():
            print(f"⚠️  Warning: {split_path} does not exist")
            return
        
        zip_files = list(split_path.glob("*.zip"))
        print(f"\n📦 Processing {len(zip_files)} zip files in {split}/")
        
        for zip_file in zip_files:
            # Extract to a folder with the same name (without .zip)
            extract_folder = split_path / zip_file.stem
            
            # Skip if already extracted
            if extract_folder.exists():
                print(f"  ✓ Already extracted: {zip_file.name}")
                continue
            
            print(f"  📂 Extracting: {zip_file.name}")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)
            except Exception as e:
                print(f"  ❌ Error extracting {zip_file.name}: {e}")
    
    def flatten_nested_folders(self, split: str):
        """Flatten double-nested folders (e.g., fall-01/fall-01/* -> fall-01/*)."""
        split_path = self.urfd_path / split
        
        if not split_path.exists():
            return
        
        print(f"\n🔧 Flattening nested folders in {split}/")
        
        for item in split_path.iterdir():
            if not item.is_dir() or item.name.startswith('.'):
                continue
            
            # Check if there's a nested folder with the same name
            nested_folder = item / item.name
            if nested_folder.exists() and nested_folder.is_dir():
                print(f"  🔄 Flattening: {item.name}/{item.name}/* -> {item.name}/*")
                
                # Move all contents from nested folder to parent
                for nested_item in nested_folder.iterdir():
                    dest = item / nested_item.name
                    if not dest.exists():
                        shutil.move(str(nested_item), str(dest))
                
                # Remove the now-empty nested folder
                try:
                    nested_folder.rmdir()
                except OSError:
                    print(f"  ⚠️  Could not remove {nested_folder} (may not be empty)")
    
    def count_urfd_files(self, split: str) -> int:
        """Count video/image files in URFD split."""
        split_path = self.urfd_path / split
        
        if not split_path.exists():
            return 0
        
        # Count folders (each folder represents a video sequence)
        folders = [d for d in split_path.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
        
        return len(folders)
    
    def verify_le2i_structure(self):
        """Verify Le2i folder structure and count files."""
        if not self.le2i_path.exists():
            print(f"⚠️  Warning: {self.le2i_path} does not exist")
            return
        
        print(f"\n🔍 Verifying Le2i structure...")
        
        scene_folders = [d for d in self.le2i_path.iterdir() 
                        if d.is_dir() and not d.name.startswith('.')]
        
        total_videos = 0
        
        for scene in scene_folders:
            videos_folder = scene / "Videos"
            annotation_folder = scene / "Annotation_files"
            annotations_folder = scene / "Annotations_files"  # Alternative spelling
            
            # Count .avi files
            avi_files = list(scene.glob("*.avi"))
            videos_in_subfolder = []
            
            if videos_folder.exists():
                videos_in_subfolder = list(videos_folder.glob("*.avi"))
            
            video_count = len(avi_files) + len(videos_in_subfolder)
            total_videos += video_count
            
            # Check structure
            has_videos_folder = videos_folder.exists()
            has_annotation_folder = annotation_folder.exists() or annotations_folder.exists()
            has_direct_avi = len(avi_files) > 0
            
            status = "✓" if (has_videos_folder or has_direct_avi) else "⚠️"
            print(f"  {status} {scene.name}: {video_count} videos", end="")
            
            if has_videos_folder:
                print(" (Videos/)", end="")
            if has_annotation_folder:
                print(" (Annotation_files/)", end="")
            if has_direct_avi:
                print(" (direct .avi)", end="")
            print()
        
        return total_videos
    
    def collect_statistics(self):
        """Collect file counts for all datasets."""
        print("\n📊 Collecting statistics...")
        
        # URFD counts
        self.stats["urfd_fall"] = self.count_urfd_files("falls")
        self.stats["urfd_adl"] = self.count_urfd_files("adl")
        
        # Le2i counts
        self.stats["le2i_scenes"] = self.verify_le2i_structure()
    
    def print_summary_table(self):
        """Print a formatted summary table."""
        print("\n" + "="*50)
        print("📋 DATASET SUMMARY")
        print("="*50)
        print(f"{'Dataset':<15} | {'Split':<15} | {'File count':<10}")
        print("-"*50)
        print(f"{'URFD':<15} | {'fall':<15} | {self.stats['urfd_fall']:<10}")
        print(f"{'URFD':<15} | {'adl':<15} | {self.stats['urfd_adl']:<10}")
        print(f"{'Le2i':<15} | {'scenes':<15} | {self.stats['le2i_scenes']:<10}")
        print("="*50)
    
    def update_documentation(self):
        """Update or create docs/dataset_notes.md with counts."""
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)
        
        doc_file = docs_dir / "dataset_notes.md"
        
        # Prepare the new content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_content = f"""
## Dataset Statistics (Updated: {timestamp})

| Dataset | Split  | File Count |
|---------|--------|------------|
| URFD    | fall   | {self.stats['urfd_fall']:<10} |
| URFD    | adl    | {self.stats['urfd_adl']:<10} |
| Le2i    | scenes | {self.stats['le2i_scenes']:<10} |

### Notes
- URFD dataset contains image sequences (PNG files) organized by video
- Le2i dataset contains .avi video files across multiple scene locations
- Each URFD folder represents one video sequence
"""
        
        # Read existing content if file exists
        existing_content = ""
        if doc_file.exists():
            with open(doc_file, 'r') as f:
                existing_content = f.read()
        
        # Append new content
        with open(doc_file, 'w') as f:
            if existing_content:
                f.write(existing_content)
                f.write("\n\n---\n")
            f.write(new_content)
        
        print(f"\n✅ Updated documentation: {doc_file}")
    
    def run(self):
        """Execute the full preparation pipeline."""
        print("🚀 Starting dataset preparation...")
        print(f"📁 Base path: {self.base_path.absolute()}")
        
        # Process URFD
        for split in ["falls", "adl"]:
            self.unzip_urfd_files(split)
            self.flatten_nested_folders(split)
        
        # Collect statistics
        self.collect_statistics()
        
        # Print summary
        self.print_summary_table()
        
        # Update documentation
        self.update_documentation()
        
        print("\n✨ Dataset preparation complete!")


def main():
    """Main entry point."""
    preparer = DatasetPreparer()
    preparer.run()


if __name__ == "__main__":
    main()

