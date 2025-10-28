# Dataset Cleanup - Quick Reference

## One-Line Commands

```bash
# Preview cleanup (safe)
python3 scripts/validate_and_cleanup_datasets.py --dry-run

# Interactive cleanup (asks confirmation)
python3 scripts/validate_and_cleanup_datasets.py

# Force cleanup (no confirmation)
python3 scripts/validate_and_cleanup_datasets.py --force
```

## What Gets Cleaned Up

✅ **Removed:**
- `.zip` files (archives)
- `.DS_Store` (macOS metadata)
- `.tar`, `.gz` (archives)
- `.tmp` (temporary files)
- `*~` (backup files)
- Empty folders

❌ **Never Removed:**
- `.avi`, `.mp4` (videos)
- `.png`, `.jpg` (images)
- `.txt` (annotations)
- `.py` (scripts)
- `.md` (documentation)

## Current Dataset Status

| Dataset | Category | Count |
|---------|----------|-------|
| URFD | Fall videos | 31 |
| URFD | ADL videos | 32 |
| Le2i | Total videos | 190 |
| Le2i | Annotations | 130 |

**Cleanup candidates:** 68 files (64 .zip + 4 .DS_Store)

## Output Files

- **Report:** `docs/dataset_cleanup_report.md`
- **Console:** Color-coded summary

## Safety

✅ Dry-run mode available  
✅ Interactive confirmation  
✅ Never deletes valid files  
✅ Detailed logging  
✅ Safety tests pass  

## Common Workflows

### First-time cleanup
```bash
python3 scripts/validate_and_cleanup_datasets.py --dry-run
cat docs/dataset_cleanup_report.md
python3 scripts/validate_and_cleanup_datasets.py --force
```

### After dataset preparation
```bash
python3 scripts/prepare_datasets.py
python3 scripts/validate_and_cleanup_datasets.py --force
```

### Regular maintenance
```bash
python3 scripts/validate_and_cleanup_datasets.py --dry-run
```

## Help

```bash
python3 scripts/validate_and_cleanup_datasets.py --help
```

## Documentation

- **User Guide:** `docs/dataset_validation_guide.md`
- **Implementation:** `docs/cleanup_script_summary.md`
- **This Reference:** `docs/cleanup_quick_reference.md`

