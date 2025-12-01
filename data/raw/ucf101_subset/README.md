# UCF101 Subset - Fall-Related Action Clips

## 📊 Dataset Overview

This is a subset of the UCF101 Action Recognition Dataset, containing only fall-related action clips used for training our fall detection model.

- **Size:** 281 MB
- **Total Videos:** ~50 clips
- **Source:** UCF101 Action Recognition Dataset
- **Resolution:** Various (320×240 to 640×480)
- **FPS:** 25-30
- **Format:** AVI

---

## 📁 Folder Structure

```
ucf101_subset/
├── falling/               # Fall-related action clips
│   ├── v_Falling_g01_c01.avi
│   ├── v_Falling_g01_c02.avi
│   └── ...
│
└── README.md             # This file
```

---

## 🎬 Action Categories Used

We selected clips from the following UCF101 categories:

1. **Falling** - People falling in various contexts
2. **Tripping** - People tripping and falling
3. **Collapsing** - People collapsing/fainting

---

## 📝 File Naming Convention

```
v_[ActionName]_g[GroupNumber]_c[ClipNumber].avi

Examples:
- v_Falling_g01_c01.avi
- v_Falling_g02_c03.avi

Where:
- ActionName = Type of action (Falling, etc.)
- GroupNumber = Group/subject number
- ClipNumber = Clip number within group
```

---

## 🔗 Download

**Google Drive:** [Download UCF101 Subset](https://drive.google.com/drive/folders/YOUR_UCF101_FOLDER_ID)

**Original UCF101 Dataset:** https://www.crcv.ucf.edu/data/UCF101.php

---

## 📚 Citation

If you use this dataset, please cite the original UCF101 paper:

```bibtex
@article{soomro2012ucf101,
  title={UCF101: A dataset of 101 human actions classes from videos in the wild},
  author={Soomro, Khurram and Zamir, Amir Roshan and Shah, Mubarak},
  journal={arXiv preprint arXiv:1212.0402},
  year={2012}
}
```

---

## 📊 Statistics

| Metric | Value |
|--------|-------|
| Total Videos | ~50 clips |
| Action Categories | 3 (Falling, Tripping, Collapsing) |
| Average Duration | 5-10 seconds |
| Resolution | Various (320×240 to 640×480) |
| FPS | 25-30 |
| Total Size | 281 MB |

---

## 🎯 Why UCF101?

We included UCF101 clips because:

1. **Diversity** - Real-world scenarios (not lab settings)
2. **Variety** - Different camera angles, lighting, backgrounds
3. **Realism** - Captured from movies, YouTube, sports
4. **Complementary** - Adds variety to URFD and Le2i datasets
5. **Action Recognition** - Helps model learn fall motion patterns

---

## ⚠️ Notes

- Only fall-related clips are included (not all 101 action categories)
- Videos are from real-world sources (movies, YouTube, sports)
- Various camera angles and qualities
- Some clips may have camera motion
- Dataset is for academic/research use only
- Smaller subset compared to URFD and Le2i

---

## 🔧 Selection Criteria

Clips were selected based on:

1. **Relevance** - Must show falling motion
2. **Quality** - Sufficient resolution and clarity
3. **Visibility** - Person clearly visible
4. **Duration** - At least 3 seconds long
5. **Diversity** - Various scenarios and angles

---

**Last Updated:** December 2024

