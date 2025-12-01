# Data Directory

This directory contains all datasets, intermediate processing files, and final processed data for the fall detection project.

---

## 📁 Directory Structure

```
data/
├── raw/                    # Raw video datasets (URFD, Le2i, UCF101)
├── interim/                # Intermediate processing files
│   ├── keypoints/         # Extracted pose keypoints (.npz files)
│   └── features_per_frame/# Frame-level features
├── processed/              # Final processed datasets ready for training
├── test/                   # Test videos for evaluation
├── test_results/           # Test results and analysis
└── references/             # Reference papers and documentation
```

---

## 📊 Data Flow Pipeline

```
1. Raw Videos (data/raw/)
   ↓
2. Pose Keypoint Extraction (MoveNet Lightning)
   ↓
3. Keypoint Files (data/interim/keypoints/)
   ↓
4. Feature Engineering (6-10 features)
   ↓
5. Temporal Windowing (30 frames, stride 10)
   ↓
6. Processed Windows (data/processed/)
   ↓
7. LSTM Training
```

---

## 📦 Folder Descriptions

### `raw/` - Raw Datasets
- **URFD:** 63 sequences (31 falls, 32 ADL) - 7.4 GB
- **Le2i:** 190 videos (95 falls, 95 ADL) - 16 GB
- **UCF101 Subset:** 711 videos (non-fall activities) - 281 MB
- **Total:** 964 videos, ~24 GB

See [raw/README.md](raw/README.md) for details and Google Drive links.

### `interim/` - Intermediate Files
- **keypoints/:** 964 .npz files with extracted pose keypoints
  - Format: (T, 17, 3) - T frames × 17 keypoints × (y, x, confidence)
  - Size: ~9 MB compressed
  - Extracted using MoveNet Lightning
- **features_per_frame/:** Frame-level engineered features

See [interim/README.md](interim/README.md) for details.

### `processed/` - Training-Ready Data
- **all_windows_30frame_raw.npz:** 30-frame windows with raw keypoints
- **all_windows_enhanced.npz:** Enhanced features (10 features)
- **all_windows_full.npz:** Full feature set
- **all_windows_30_physics5.npz:** Physics-based features
- **all_windows_30_raw_balanced.npz:** Balanced dataset (16% fall / 84% non-fall)

See [processed/README.md](processed/README.md) for details.

### `test/` - Test Videos
- 15 custom test videos (9 falls, 6 non-falls)
- Used for final evaluation
- 100% accuracy achieved (15/15 correct)

See [test/README.md](test/README.md) for details.

### `test_results/` - Test Analysis
- Detailed test results for all 15 videos
- Frame-by-frame analysis
- Performance metrics

See [test_results/RESULTS.md](test_results/RESULTS.md) for details.

### `references/` - Documentation
- Research papers
- Dataset documentation
- Technical references

See [references/README.md](references/README.md) for details.

---

## 📈 Dataset Statistics

### Complete Dataset
```
Total Videos: 964
├── URFD: 63 (6.5%)
├── Le2i: 190 (19.7%)
└── UCF101: 711 (73.8%)

Total Frames: 197,411
├── Fall Frames: 34,801 (17.6%)
└── Non-Fall Frames: 162,610 (82.4%)

Storage:
├── Raw Videos: ~24 GB
├── Keypoint Files: ~9 MB
├── Processed Windows: ~50 MB
└── Total: ~24.1 GB
```

### Training Dataset (Final)
```
Windows: ~15,000
├── Fall Windows: ~2,400 (16%)
└── Non-Fall Windows: ~12,600 (84%)

Features per Window: 30 frames × 34 values = 1,020 values
Window Duration: 1 second (30 FPS)
Stride: 10 frames (0.33 seconds overlap)
```

---

## 🔧 Data Processing Scripts

### Extraction
- `ml/data/extract_pose_sequences.py` - Extract keypoints from URFD/Le2i
- `ml/data/ucf101_extract.py` - Extract keypoints from UCF101

### Feature Engineering
- `ml/features/feature_engineering.py` - Create engineered features
- `ml/features/create_30frame_raw_keypoints.py` - Create 30-frame windows

### Verification
- `scripts/verify_extraction_integrity.py` - Validate keypoint files
- `scripts/validate_and_cleanup_datasets.py` - Validate raw datasets

---

## 📝 Data Format Specifications

### Keypoint Files (.npz)
```python
{
    'keypoints': np.array(shape=(T, 17, 3), dtype=float32),
    'label': int (0=non-fall, 1=fall),
    'fps': float,
    'video_name': str
}
```

### Processed Windows (.npz)
```python
{
    'X': np.array(shape=(N, 30, 34), dtype=float32),  # N windows
    'y': np.array(shape=(N,), dtype=int),             # Labels
    'video_names': list of str,                        # Source videos
    'window_indices': list of int                      # Window positions
}
```

---

## ⚠️ Important Notes

1. **Raw datasets are NOT in GitHub** (too large, 24 GB)
   - Download from Google Drive links in `raw/README.md`

2. **Interim keypoints are NOT in GitHub** (excluded by .gitignore)
   - Regenerate using extraction scripts

3. **Processed data IS in GitHub** (small, ~50 MB)
   - Ready for training

4. **Test videos are NOT in GitHub** (706 MB)
   - Download from Google Drive link in `test/README.md`

---

## 🔗 Quick Links

- [Raw Datasets](raw/README.md) - Download links and descriptions
- [Interim Data](interim/README.md) - Keypoint extraction details
- [Processed Data](processed/README.md) - Training-ready datasets
- [Test Videos](test/README.md) - Test set description
- [Test Results](test_results/RESULTS.md) - Evaluation results

---

**Last Updated:** December 2024

