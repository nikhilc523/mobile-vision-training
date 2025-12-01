# URFD - University of Rzeszow Fall Detection Dataset

## 📊 Dataset Overview

The URFD dataset is a widely-used benchmark for fall detection research, created by the University of Rzeszow, Poland.

- **Size:** 7.4 GB
- **Total Videos:** 100
- **Falls:** 70 videos
- **ADL (Activities of Daily Living):** 30 videos
- **Resolution:** 640×480
- **FPS:** 30
- **Format:** AVI

---

## 📁 Folder Structure

```
urfd/
├── falls/                  # 70 fall videos
│   ├── fall-01-cam0.avi
│   ├── fall-02-cam0.avi
│   └── ...
│
├── adl/                    # 30 ADL videos
│   ├── adl-01-cam0.avi
│   ├── adl-02-cam0.avi
│   └── ...
│
└── README.md              # This file
```

---

## 🎬 Fall Types

The dataset includes various fall scenarios:

1. **Forward falls** - Person falling forward
2. **Backward falls** - Person falling backward
3. **Sideways falls** - Person falling to the left or right
4. **Chair falls** - Person falling from sitting position
5. **Tripping falls** - Person tripping and falling
6. **Fainting falls** - Person collapsing/fainting

---

## 🚶 ADL (Activities of Daily Living)

Non-fall activities included:

1. **Walking** - Normal walking
2. **Sitting down** - Sitting on chair
3. **Standing up** - Getting up from chair
4. **Bending** - Bending to pick up objects
5. **Lying down** - Lying on bed/floor intentionally
6. **Crouching** - Crouching/squatting

---

## 📝 File Naming Convention

```
fall-XX-camY.avi
adl-XX-camY.avi

Where:
- XX = sequence number (01-70 for falls, 01-30 for ADL)
- Y = camera number (0 = Kinect camera)
```

---

## 🔗 Download

**Google Drive:** [Download URFD Dataset](https://drive.google.com/drive/folders/YOUR_URFD_FOLDER_ID)

**Original Source:** http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html

---

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@article{kwolek2014human,
  title={Human fall detection on embedded platform using depth maps and wireless accelerometer},
  author={Kwolek, Bogdan and Kepski, Michal},
  journal={Computer methods and programs in biomedicine},
  volume={117},
  number={3},
  pages={489--501},
  year={2014},
  publisher={Elsevier}
}
```

---

## 📊 Statistics

| Metric | Value |
|--------|-------|
| Total Videos | 100 |
| Fall Videos | 70 |
| ADL Videos | 30 |
| Average Duration | 5-10 seconds |
| Resolution | 640×480 |
| FPS | 30 |
| Total Size | 7.4 GB |

---

## ⚠️ Notes

- Videos are recorded from a single Kinect camera
- Indoor environment with controlled lighting
- Actors perform falls on padded mats for safety
- Some videos may have motion blur during fast falls
- Dataset is for academic/research use only

---

**Last Updated:** December 2024

