#!/usr/bin/env python3
"""
Dataset Validation and Cleanup Script

Validates URFD and Le2i datasets, detects issues, and optionally cleans up
unwanted files (zip archives, .DS_Store, empty folders, etc.).

Usage:
    python scripts/validate_and_cleanup_datasets.py --dry-run    # Preview only
    python scripts/validate_and_cleanup_datasets.py --force      # Execute cleanup
    python scripts/validate_and_cleanup_datasets.py              # Interactive mode
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
import argparse


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def colored(text: str, color: str) -> str:
    """Return colored text for terminal output."""
    return f"{color}{text}{Colors.RESET}"


class DatasetValidator:
    """Validates and cleans dataset files."""
    
    def __init__(self, data_root: Path, dry_run: bool = False):
        self.data_root = data_root
        self.dry_run = dry_run
        
        # Statistics
        self.stats = {
            'urfd_fall_videos': 0,
            'urfd_adl_videos': 0,
            'le2i_videos': 0,
            'le2i_annotations': 0,
            'videos_without_annotations': [],
            'annotations_without_videos': [],
            'empty_folders': [],
            'cleanup_files': [],
            'deleted_files': [],
            'deleted_folders': [],
        }
        
        # File patterns to clean up
        self.cleanup_patterns = ['.zip', '.tar', '.gz', '.tmp', '.DS_Store']
        self.backup_patterns = ['~']
        
    def validate_urfd_dataset(self) -> Dict[str, int]:
        """Validate URFD dataset structure and count files."""
        print(f"\n{colored('🔍 Validating URFD Dataset...', Colors.BLUE)}")
        
        urfd_root = self.data_root / 'urfd'
        if not urfd_root.exists():
            print(f"  {colored('⚠️  URFD directory not found', Colors.YELLOW)}")
            return {}
        
        results = {}
        
        # Check fall videos
        fall_dir = urfd_root / 'falls'
        if fall_dir.exists():
            fall_videos = self._count_urfd_videos(fall_dir)
            self.stats['urfd_fall_videos'] = fall_videos
            results['fall'] = fall_videos
            print(f"  {colored('🟢', Colors.GREEN)} Fall videos: {fall_videos}")
        else:
            print(f"  {colored('⚠️', Colors.YELLOW)} Fall directory not found")
        
        # Check ADL videos
        adl_dir = urfd_root / 'adl'
        if adl_dir.exists():
            adl_videos = self._count_urfd_videos(adl_dir)
            self.stats['urfd_adl_videos'] = adl_videos
            results['adl'] = adl_videos
            print(f"  {colored('🟢', Colors.GREEN)} ADL videos: {adl_videos}")
        else:
            print(f"  {colored('⚠️', Colors.YELLOW)} ADL directory not found")
        
        return results
    
    def _count_urfd_videos(self, directory: Path) -> int:
        """Count URFD video sequences (folders with PNG files)."""
        count = 0
        for item in directory.iterdir():
            if item.is_dir():
                # Check if folder contains PNG files
                png_files = list(item.glob('*.png'))
                if png_files:
                    count += 1
        return count
    
    def validate_le2i_dataset(self) -> Dict[str, Dict]:
        """Validate Le2i dataset structure and count files."""
        print(f"\n{colored('🔍 Validating Le2i Dataset...', Colors.BLUE)}")
        
        le2i_root = self.data_root / 'le2i'
        if not le2i_root.exists():
            print(f"  {colored('⚠️  Le2i directory not found', Colors.YELLOW)}")
            return {}
        
        results = {}
        
        # Process each scene
        for scene_dir in sorted(le2i_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            scene_name = scene_dir.name
            scene_stats = self._validate_le2i_scene(scene_dir)
            results[scene_name] = scene_stats
        
        return results
    
    def _validate_le2i_scene(self, scene_dir: Path) -> Dict:
        """Validate a single Le2i scene directory."""
        scene_name = scene_dir.name
        
        # Find videos
        videos = set()
        videos_dir = scene_dir / 'Videos'
        if videos_dir.exists():
            videos = {v.stem for v in videos_dir.glob('*.avi')}
        else:
            # Videos might be directly in scene directory
            videos = {v.stem for v in scene_dir.glob('*.avi')}
        
        # Find annotations
        annotations = set()
        ann_dir = scene_dir / 'Annotation_files'
        if not ann_dir.exists():
            ann_dir = scene_dir / 'Annotations_files'  # Alternative spelling
        
        if ann_dir.exists():
            annotations = {a.stem for a in ann_dir.glob('*.txt')}
        
        # Check for mismatches
        videos_without_ann = videos - annotations
        ann_without_videos = annotations - videos
        
        # Update global stats
        self.stats['le2i_videos'] += len(videos)
        self.stats['le2i_annotations'] += len(annotations)
        
        if videos_without_ann:
            for v in videos_without_ann:
                self.stats['videos_without_annotations'].append(f"{scene_name}/{v}.avi")
        
        if ann_without_videos:
            for a in ann_without_videos:
                self.stats['annotations_without_videos'].append(f"{scene_name}/{a}.txt")
        
        # Print scene summary
        status = colored('🟢', Colors.GREEN)
        if videos_without_ann or ann_without_videos:
            status = colored('⚠️', Colors.YELLOW)
        
        print(f"  {status} {scene_name:<20} Videos: {len(videos):<3} Annotations: {len(annotations):<3}", end='')
        
        if videos_without_ann:
            print(f" {colored(f'⚠️ {len(videos_without_ann)} without ann', Colors.YELLOW)}", end='')
        if ann_without_videos:
            print(f" {colored(f'⚠️ {len(ann_without_videos)} orphaned ann', Colors.YELLOW)}", end='')
        print()
        
        return {
            'videos': len(videos),
            'annotations': len(annotations),
            'videos_without_annotations': len(videos_without_ann),
            'annotations_without_videos': len(ann_without_videos),
        }
    
    def find_cleanup_candidates(self) -> List[Path]:
        """Find files that should be cleaned up."""
        print(f"\n{colored('🔍 Scanning for cleanup candidates...', Colors.BLUE)}")
        
        cleanup_files = []
        
        # Walk through all files
        for root, dirs, files in os.walk(self.data_root):
            root_path = Path(root)
            
            for file in files:
                file_path = root_path / file
                
                # Check for cleanup patterns
                should_cleanup = False
                
                # Check extensions
                for pattern in self.cleanup_patterns:
                    if file.endswith(pattern):
                        should_cleanup = True
                        break
                
                # Check backup files
                for pattern in self.backup_patterns:
                    if file.endswith(pattern):
                        should_cleanup = True
                        break
                
                if should_cleanup:
                    cleanup_files.append(file_path)
        
        self.stats['cleanup_files'] = cleanup_files
        
        # Print summary
        if cleanup_files:
            print(f"  Found {colored(str(len(cleanup_files)), Colors.YELLOW)} files to clean up:")
            
            # Group by type
            by_type = {}
            for f in cleanup_files:
                ext = f.suffix if f.suffix else f.name
                by_type[ext] = by_type.get(ext, 0) + 1
            
            for ext, count in sorted(by_type.items()):
                print(f"    {ext}: {count} files")
        else:
            print(f"  {colored('🟢 No cleanup needed', Colors.GREEN)}")
        
        return cleanup_files
    
    def find_empty_folders(self) -> List[Path]:
        """Find empty folders recursively."""
        empty_folders = []
        
        for root, dirs, files in os.walk(self.data_root, topdown=False):
            root_path = Path(root)
            
            # Skip if it's the root itself
            if root_path == self.data_root:
                continue
            
            # Check if folder is empty
            if not any(root_path.iterdir()):
                empty_folders.append(root_path)
        
        self.stats['empty_folders'] = empty_folders
        
        if empty_folders:
            print(f"\n  Found {colored(str(len(empty_folders)), Colors.YELLOW)} empty folders")
        
        return empty_folders
    
    def cleanup(self, force: bool = False) -> bool:
        """Execute cleanup operations."""
        cleanup_files = self.stats['cleanup_files']
        empty_folders = self.stats['empty_folders']
        
        if not cleanup_files and not empty_folders:
            print(f"\n{colored('✅ No cleanup needed!', Colors.GREEN)}")
            return True
        
        total_items = len(cleanup_files) + len(empty_folders)
        
        if self.dry_run:
            print(f"\n{colored('🔍 DRY RUN MODE - No changes will be made', Colors.CYAN)}")
            print(f"\nWould delete {total_items} items:")
            
            if cleanup_files:
                print(f"\n  Files ({len(cleanup_files)}):")
                for f in cleanup_files[:10]:  # Show first 10
                    print(f"    ❌ {f.relative_to(self.data_root)}")
                if len(cleanup_files) > 10:
                    print(f"    ... and {len(cleanup_files) - 10} more")
            
            if empty_folders:
                print(f"\n  Empty folders ({len(empty_folders)}):")
                for f in empty_folders[:10]:
                    print(f"    ❌ {f.relative_to(self.data_root)}/")
                if len(empty_folders) > 10:
                    print(f"    ... and {len(empty_folders) - 10} more")
            
            return True
        
        # Interactive confirmation
        if not force:
            print(f"\n{colored('⚠️  WARNING:', Colors.YELLOW)} About to delete {total_items} items")
            response = input(f"Continue? [y/N]: ").strip().lower()
            if response != 'y':
                print(f"{colored('❌ Cleanup cancelled', Colors.RED)}")
                return False
        
        # Delete files
        print(f"\n{colored('🗑️  Cleaning up files...', Colors.BLUE)}")
        for file_path in cleanup_files:
            try:
                file_path.unlink()
                self.stats['deleted_files'].append(file_path)
                print(f"  {colored('❌', Colors.RED)} Deleted: {file_path.relative_to(self.data_root)}")
            except Exception as e:
                print(f"  {colored('⚠️', Colors.YELLOW)} Failed to delete {file_path}: {e}")
        
        # Delete empty folders
        if empty_folders:
            print(f"\n{colored('🗑️  Removing empty folders...', Colors.BLUE)}")
            for folder_path in empty_folders:
                try:
                    folder_path.rmdir()
                    self.stats['deleted_folders'].append(folder_path)
                    print(f"  {colored('❌', Colors.RED)} Removed: {folder_path.relative_to(self.data_root)}/")
                except Exception as e:
                    print(f"  {colored('⚠️', Colors.YELLOW)} Failed to remove {folder_path}: {e}")
        
        print(f"\n{colored('✅ Cleanup complete!', Colors.GREEN)}")
        return True

    def generate_report(self, output_path: Path, urfd_results: Dict, le2i_results: Dict):
        """Generate a Markdown report of the validation and cleanup."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report_lines = [
            "# Dataset Validation and Cleanup Report",
            "",
            f"**Generated:** {timestamp}",
            "",
            "## Summary",
            "",
            "### URFD Dataset",
            "",
            f"- **Fall videos:** {self.stats['urfd_fall_videos']}",
            f"- **ADL videos:** {self.stats['urfd_adl_videos']}",
            f"- **Total:** {self.stats['urfd_fall_videos'] + self.stats['urfd_adl_videos']}",
            "",
            "### Le2i Dataset",
            "",
            f"- **Total videos:** {self.stats['le2i_videos']}",
            f"- **Total annotations:** {self.stats['le2i_annotations']}",
            f"- **Videos without annotations:** {len(self.stats['videos_without_annotations'])}",
            f"- **Annotations without videos:** {len(self.stats['annotations_without_videos'])}",
            "",
            "### Le2i Scene Breakdown",
            "",
            "| Scene | Videos | Annotations | Issues |",
            "|-------|--------|-------------|--------|",
        ]

        for scene_name, scene_stats in sorted(le2i_results.items()):
            issues = []
            if scene_stats['videos_without_annotations'] > 0:
                issues.append(f"{scene_stats['videos_without_annotations']} w/o ann")
            if scene_stats['annotations_without_videos'] > 0:
                issues.append(f"{scene_stats['annotations_without_videos']} orphaned")

            issue_str = ", ".join(issues) if issues else "✅ None"

            report_lines.append(
                f"| {scene_name} | {scene_stats['videos']} | "
                f"{scene_stats['annotations']} | {issue_str} |"
            )

        # Cleanup section
        report_lines.extend([
            "",
            "## Cleanup Operations",
            "",
        ])

        if self.dry_run:
            report_lines.append("**Mode:** Dry run (no changes made)")
        else:
            report_lines.append("**Mode:** Cleanup executed")

        report_lines.extend([
            "",
            f"- **Files identified for cleanup:** {len(self.stats['cleanup_files'])}",
            f"- **Empty folders found:** {len(self.stats['empty_folders'])}",
            f"- **Files deleted:** {len(self.stats['deleted_files'])}",
            f"- **Folders removed:** {len(self.stats['deleted_folders'])}",
            "",
        ])

        # Cleanup details
        if self.stats['cleanup_files']:
            report_lines.extend([
                "### Files Identified for Cleanup",
                "",
            ])

            # Group by type
            by_type = {}
            for f in self.stats['cleanup_files']:
                ext = f.suffix if f.suffix else f.name
                by_type[ext] = by_type.get(ext, [])
                by_type[ext].append(f)

            for ext, files in sorted(by_type.items()):
                report_lines.append(f"**{ext}** ({len(files)} files):")
                report_lines.append("")
                for f in files[:20]:  # Limit to first 20
                    status = "❌ Deleted" if f in self.stats['deleted_files'] else "🔍 Found"
                    report_lines.append(f"- {status}: `{f.relative_to(self.data_root)}`")
                if len(files) > 20:
                    report_lines.append(f"- ... and {len(files) - 20} more")
                report_lines.append("")

        # Issues section
        if self.stats['videos_without_annotations'] or self.stats['annotations_without_videos']:
            report_lines.extend([
                "## Issues Detected",
                "",
            ])

            if self.stats['videos_without_annotations']:
                report_lines.extend([
                    f"### Videos Without Annotations ({len(self.stats['videos_without_annotations'])})",
                    "",
                ])
                for video in self.stats['videos_without_annotations'][:20]:
                    report_lines.append(f"- ⚠️  `{video}`")
                if len(self.stats['videos_without_annotations']) > 20:
                    report_lines.append(f"- ... and {len(self.stats['videos_without_annotations']) - 20} more")
                report_lines.append("")

            if self.stats['annotations_without_videos']:
                report_lines.extend([
                    f"### Annotations Without Videos ({len(self.stats['annotations_without_videos'])})",
                    "",
                ])
                for ann in self.stats['annotations_without_videos'][:20]:
                    report_lines.append(f"- ⚠️  `{ann}`")
                if len(self.stats['annotations_without_videos']) > 20:
                    report_lines.append(f"- ... and {len(self.stats['annotations_without_videos']) - 20} more")
                report_lines.append("")

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(report_lines))

        print(f"\n{colored('📄 Report saved:', Colors.GREEN)} {output_path}")

    def print_summary_table(self, urfd_results: Dict, le2i_results: Dict):
        """Print a summary table to console."""
        print(f"\n{colored('='*70, Colors.BOLD)}")
        print(f"{colored('📊 VALIDATION SUMMARY', Colors.BOLD)}")
        print(f"{colored('='*70, Colors.BOLD)}")

        # URFD summary
        print(f"\n{colored('URFD Dataset:', Colors.CYAN)}")
        print(f"  Fall videos: {self.stats['urfd_fall_videos']}")
        print(f"  ADL videos:  {self.stats['urfd_adl_videos']}")
        print(f"  Total:       {self.stats['urfd_fall_videos'] + self.stats['urfd_adl_videos']}")

        # Le2i summary
        print(f"\n{colored('Le2i Dataset:', Colors.CYAN)}")
        print(f"  Total videos:      {self.stats['le2i_videos']}")
        print(f"  Total annotations: {self.stats['le2i_annotations']}")

        if self.stats['videos_without_annotations']:
            print(f"  {colored('⚠️  Videos w/o annotations:', Colors.YELLOW)} {len(self.stats['videos_without_annotations'])}")

        if self.stats['annotations_without_videos']:
            print(f"  {colored('⚠️  Orphaned annotations:', Colors.YELLOW)} {len(self.stats['annotations_without_videos'])}")

        # Cleanup summary
        print(f"\n{colored('Cleanup:', Colors.CYAN)}")
        print(f"  Files to clean:    {len(self.stats['cleanup_files'])}")
        print(f"  Empty folders:     {len(self.stats['empty_folders'])}")

        if not self.dry_run and (self.stats['deleted_files'] or self.stats['deleted_folders']):
            print(f"  Files deleted:     {len(self.stats['deleted_files'])}")
            print(f"  Folders removed:   {len(self.stats['deleted_folders'])}")

        print(f"\n{colored('='*70, Colors.BOLD)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate and clean up fall-detection datasets (URFD + Le2i)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be cleaned up
  python scripts/validate_and_cleanup_datasets.py --dry-run

  # Interactive cleanup (asks for confirmation)
  python scripts/validate_and_cleanup_datasets.py

  # Force cleanup without confirmation
  python scripts/validate_and_cleanup_datasets.py --force
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview cleanup operations without making changes'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Execute cleanup without confirmation'
    )

    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('data/raw'),
        help='Root directory of datasets (default: data/raw)'
    )

    parser.add_argument(
        '--report',
        type=Path,
        default=Path('docs/dataset_cleanup_report.md'),
        help='Output path for report (default: docs/dataset_cleanup_report.md)'
    )

    args = parser.parse_args()

    # Check if data directory exists
    if not args.data_root.exists():
        print(f"{colored('❌ Error:', Colors.RED)} Data directory not found: {args.data_root}")
        sys.exit(1)

    # Create validator
    validator = DatasetValidator(args.data_root, dry_run=args.dry_run)

    # Print header
    print(f"\n{colored('='*70, Colors.BOLD)}")
    print(f"{colored('🔍 DATASET VALIDATION AND CLEANUP', Colors.BOLD)}")
    print(f"{colored('='*70, Colors.BOLD)}")

    if args.dry_run:
        print(f"\n{colored('🔍 DRY RUN MODE - No changes will be made', Colors.CYAN)}")

    # Validate datasets
    urfd_results = validator.validate_urfd_dataset()
    le2i_results = validator.validate_le2i_dataset()

    # Find cleanup candidates
    validator.find_cleanup_candidates()
    validator.find_empty_folders()

    # Print summary
    validator.print_summary_table(urfd_results, le2i_results)

    # Execute cleanup
    if validator.stats['cleanup_files'] or validator.stats['empty_folders']:
        success = validator.cleanup(force=args.force)
        if not success:
            sys.exit(1)

    # Generate report
    validator.generate_report(args.report, urfd_results, le2i_results)

    print(f"\n{colored('✅ Done!', Colors.GREEN)}\n")


if __name__ == '__main__':
    main()


