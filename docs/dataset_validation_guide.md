# Dataset Validation and Cleanup Guide

## Overview

The `validate_and_cleanup_datasets.py` script audits and cleans both URFD and Le2i fall-detection datasets. It validates dataset structure, detects issues, and optionally removes unwanted files.

## Features

✅ **Validation**
- Counts valid video files (.avi, .mp4, .png sequences)
- Detects missing or mismatched files
- Identifies videos without annotations
- Identifies annotations without videos
- Finds empty folders

✅ **Cleanup**
- Removes archive files (.zip, .tar, .gz)
- Deletes system files (.DS_Store)
- Removes temporary files (.tmp)
- Deletes backup files (*~)
- Removes empty folders recursively

✅ **Reporting**
- Color-coded console output
- Detailed Markdown report
- Scene-level statistics
- Before/after file counts

✅ **Safety**
- Dry-run mode (preview only)
- Interactive confirmation
- Force mode for automation
- Never deletes valid videos or annotations

## Usage

### Dry Run (Preview Only)

Preview what would be cleaned up without making any changes:

```bash
python3 scripts/validate_and_cleanup_datasets.py --dry-run
```

**Output:**
```
🔍 DRY RUN MODE - No changes will be made

Would delete 68 items:
  Files (68):
    ❌ .DS_Store
    ❌ urfd/.DS_Store
    ❌ urfd/falls/fall-01-cam0-rgb.zip
    ...
```

### Interactive Mode (Default)

Run validation and cleanup with confirmation prompt:

```bash
python3 scripts/validate_and_cleanup_datasets.py
```

**Output:**
```
⚠️  WARNING: About to delete 68 items
Continue? [y/N]: 
```

Type `y` to proceed or `n` to cancel.

### Force Mode (No Confirmation)

Execute cleanup without confirmation (useful for automation):

```bash
python3 scripts/validate_and_cleanup_datasets.py --force
```

**⚠️ Warning:** This will delete files immediately without asking!

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview cleanup operations without making changes |
| `--force` | Execute cleanup without confirmation |
| `--data-root PATH` | Root directory of datasets (default: `data/raw`) |
| `--report PATH` | Output path for report (default: `docs/dataset_cleanup_report.md`) |
| `--help` | Show help message |

## Examples

### Example 1: Preview cleanup

```bash
python3 scripts/validate_and_cleanup_datasets.py --dry-run
```

### Example 2: Interactive cleanup

```bash
python3 scripts/validate_and_cleanup_datasets.py
# Type 'y' when prompted to confirm
```

### Example 3: Force cleanup with custom report path

```bash
python3 scripts/validate_and_cleanup_datasets.py --force --report reports/cleanup_$(date +%Y%m%d).md
```

### Example 4: Validate custom data directory

```bash
python3 scripts/validate_and_cleanup_datasets.py --data-root /path/to/data --dry-run
```

## Output

### Console Output

The script provides color-coded console output:

- 🟢 **Green** - Valid files/folders
- ⚠️ **Yellow** - Warnings (missing annotations, etc.)
- ❌ **Red** - Deleted files
- 🔍 **Blue** - Information

**Example:**
```
🔍 Validating URFD Dataset...
  🟢 Fall videos: 31
  🟢 ADL videos: 32

🔍 Validating Le2i Dataset...
  🟢 Coffee_room_01       Videos: 48  Annotations: 48 
  ⚠️ Lecture room         Videos: 27  Annotations: 0   ⚠️ 27 without ann
```

### Markdown Report

The script generates a detailed Markdown report at `docs/dataset_cleanup_report.md`:

**Report sections:**
1. **Summary** - Overall statistics for URFD and Le2i datasets
2. **Le2i Scene Breakdown** - Table with per-scene statistics
3. **Cleanup Operations** - Files and folders cleaned up
4. **Issues Detected** - Videos without annotations, orphaned annotations

**Example report:**

```markdown
# Dataset Validation and Cleanup Report

**Generated:** 2025-10-28 00:54:50

## Summary

### URFD Dataset
- **Fall videos:** 31
- **ADL videos:** 32
- **Total:** 63

### Le2i Dataset
- **Total videos:** 190
- **Total annotations:** 130
- **Videos without annotations:** 60

### Le2i Scene Breakdown

| Scene | Videos | Annotations | Issues |
|-------|--------|-------------|--------|
| Coffee_room_01 | 48 | 48 | ✅ None |
| Home_01 | 30 | 30 | ✅ None |
| Lecture room | 27 | 0 | 27 w/o ann |
```

## Validation Details

### URFD Dataset

The script validates URFD by:
1. Checking `data/raw/urfd/falls/` for fall video sequences
2. Checking `data/raw/urfd/adl/` for ADL video sequences
3. Counting folders containing PNG image sequences

**Expected structure:**
```
data/raw/urfd/
  falls/
    fall-01-cam0-rgb/
      fall-01-cam0-rgb-001.png
      fall-01-cam0-rgb-002.png
      ...
  adl/
    adl-01-cam0-rgb/
      adl-01-cam0-rgb-001.png
      ...
```

### Le2i Dataset

The script validates Le2i by:
1. Scanning each scene directory
2. Finding video files (.avi)
3. Finding annotation files (.txt)
4. Matching videos to annotations by base name
5. Detecting mismatches

**Expected structure:**
```
data/raw/le2i/
  Home_01/
    Videos/
      video (1).avi
      video (2).avi
    Annotation_files/
      video (1).txt
      video (2).txt
  Lecture_room/
    video (1).avi  # No annotations (expected)
```

## Cleanup Details

### Files Cleaned Up

The script removes the following file types:

| Pattern | Description | Example |
|---------|-------------|---------|
| `.zip` | Archive files | `fall-01-cam0-rgb.zip` |
| `.tar` | Tar archives | `dataset.tar` |
| `.gz` | Gzip archives | `data.tar.gz` |
| `.tmp` | Temporary files | `temp.tmp` |
| `.DS_Store` | macOS metadata | `.DS_Store` |
| `*~` | Backup files | `file.txt~` |

### Empty Folders

The script recursively removes empty folders after file cleanup.

### What's NOT Deleted

The script **never** deletes:
- Video files (.avi, .mp4)
- Image sequences (.png, .jpg)
- Annotation files (.txt)
- Documentation files (.md, .txt in docs/)
- Python scripts (.py)
- Any file in non-data directories

## Safety Features

### 1. Dry Run Mode

Always test with `--dry-run` first:

```bash
python3 scripts/validate_and_cleanup_datasets.py --dry-run
```

This shows exactly what would be deleted without making changes.

### 2. Interactive Confirmation

By default, the script asks for confirmation:

```
⚠️  WARNING: About to delete 68 items
Continue? [y/N]: 
```

Type `n` or press Enter to cancel.

### 3. Detailed Logging

Every deletion is logged to console and report:

```
🗑️  Cleaning up files...
  ❌ Deleted: urfd/falls/fall-01-cam0-rgb.zip
  ❌ Deleted: urfd/falls/fall-02-cam0-rgb.zip
```

### 4. Error Handling

If a file can't be deleted, the script logs a warning and continues:

```
⚠️ Failed to delete file.zip: Permission denied
```

## Common Issues

### Issue: "Videos without annotations"

**Cause:** Some Le2i scenes (Lecture room, Office) don't have fall annotations.

**Solution:** This is expected. These scenes contain only normal activities.

### Issue: "Annotations without videos"

**Cause:** Annotation file exists but corresponding video is missing.

**Solution:** Check if video was accidentally deleted or renamed.

### Issue: "Permission denied"

**Cause:** Script doesn't have permission to delete files.

**Solution:** Run with appropriate permissions or check file ownership.

## Integration with Other Scripts

### After Dataset Preparation

Run validation after `prepare_datasets.py`:

```bash
# 1. Prepare datasets
python3 scripts/prepare_datasets.py

# 2. Validate and cleanup
python3 scripts/validate_and_cleanup_datasets.py --dry-run
python3 scripts/validate_and_cleanup_datasets.py --force
```

### Before Training

Run validation before starting training:

```bash
# Validate datasets
python3 scripts/validate_and_cleanup_datasets.py --dry-run

# Check report
cat docs/dataset_cleanup_report.md
```

## Automation

### Cron Job

Add to crontab for weekly cleanup:

```bash
# Run every Sunday at 2 AM
0 2 * * 0 cd /path/to/project && python3 scripts/validate_and_cleanup_datasets.py --force
```

### CI/CD Pipeline

Add to your CI/CD pipeline:

```yaml
- name: Validate datasets
  run: python3 scripts/validate_and_cleanup_datasets.py --dry-run
```

## Troubleshooting

### Script doesn't find datasets

**Check:**
1. Current working directory: `pwd`
2. Data directory exists: `ls data/raw`
3. Use `--data-root` to specify custom path

### Colors not showing

**Cause:** Terminal doesn't support ANSI colors.

**Solution:** Use a modern terminal (iTerm2, Terminal.app, etc.)

### Report not generated

**Check:**
1. `docs/` directory exists
2. Write permissions
3. Use `--report` to specify custom path

## Dependencies

- **Python 3.11+**
- **Standard library only:**
  - `os` - File operations
  - `pathlib` - Path handling
  - `shutil` - File removal
  - `datetime` - Timestamps
  - `argparse` - Command-line parsing

No external dependencies required!

## Best Practices

1. **Always dry-run first:** `--dry-run` before actual cleanup
2. **Review the report:** Check `docs/dataset_cleanup_report.md`
3. **Backup important data:** Before running `--force`
4. **Run after extraction:** Clean up archives after unpacking
5. **Regular maintenance:** Run weekly to keep datasets clean

## Next Steps

After validation and cleanup:

1. ✅ Datasets validated and cleaned
2. 🔄 Parse Le2i annotations: `python3 -m ml.data.parsers.le2i_annotations`
3. 🔄 Extract video frames
4. 🔄 Run pose estimation
5. 🔄 Train LSTM model

