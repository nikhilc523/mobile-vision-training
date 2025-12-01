# Le2i Fall Detection Dataset

## 📊 Dataset Overview

The Le2i Fall Detection Dataset is a comprehensive dataset created by the Le2i Laboratory at the University of Burgundy, France.

- **Size:** 16 GB
- **Total Videos:** 321
- **Falls:** 191 videos
- **ADL (Activities of Daily Living):** 130 videos
- **Resolution:** Various (320×240 to 640×480)
- **FPS:** 25-30
- **Format:** AVI

---

## 📁 Folder Structure

```
le2i/
├── falls/                  # 191 fall videos
│   ├── Coffee_room_01/    # Falls in coffee room
│   ├── Home_01/           # Falls in home environment
│   └── ...
│
├── adl/                    # 130 ADL videos
│   ├── Coffee_room_01/    # ADL in coffee room
│   ├── Home_01/           # ADL in home environment
│   └── ...
│
└── README.md              # This file
```

---

## 🎬 Fall Types

The dataset includes realistic fall scenarios:

1. **Forward falls** - Falling forward
2. **Backward falls** - Falling backward
3. **Lateral falls** - Falling to the side
4. **Syncope falls** - Fainting/collapsing
5. **From chair** - Falling from sitting position
6. **While walking** - Falling while walking

---

## 🚶 ADL (Activities of Daily Living)

Non-fall activities included:

1. **Walking** - Normal walking
2. **Sitting** - Sitting on chair/sofa
3. **Lying down** - Lying on bed/floor intentionally
4. **Crouching** - Crouching/squatting
5. **Bending** - Bending to pick up objects
6. **Standing** - Standing still
7. **Getting up** - Getting up from chair/bed

---

## 🏠 Scenarios

The dataset includes two main scenarios:

### Coffee Room
- Office/workplace environment
- Multiple camera angles
- Various lighting conditions
- Realistic office furniture

### Home Environment
- Residential setting
- Living room, bedroom scenarios
- Home furniture and objects
- Natural lighting

---

## 📝 File Naming Convention

```
[Scenario]_[Subject]_[Action]_[Camera].avi

Examples:
- Coffee_room_01_fall_forward_cam1.avi
- Home_01_sitting_cam2.avi
```

---

## 🔗 Download

**Google Drive:** [Download Le2i Dataset](https://drive.google.com/drive/folders/YOUR_LE2I_FOLDER_ID)

**Original Source:** http://le2i.cnrs.fr/Fall-detection-Dataset?lang=fr

---

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{charfi2012definition,
  title={Definition and performance evaluation of a robust SVM based fall detection solution},
  author={Charfi, Imen and Miteran, Johel and Dubois, J{\'e}r{\^o}me and Atri, Mohamed and Tourki, Rached},
  booktitle={2012 Eighth International Conference on Signal Image Technology and Internet Based Systems},
  pages={218--224},
  year={2012},
  organization={IEEE}
}
```

---

## 📊 Statistics

| Metric | Value |
|--------|-------|
| Total Videos | 321 |
| Fall Videos | 191 |
| ADL Videos | 130 |
| Scenarios | 2 (Coffee room, Home) |
| Average Duration | 5-15 seconds |
| Resolution | 320×240 to 640×480 |
| FPS | 25-30 |
| Total Size | 16 GB |

---

## ⚠️ Notes

- Multiple camera angles available for some scenarios
- Realistic environments (not lab settings)
- Various lighting conditions
- Some videos have occlusions (furniture, walls)
- Dataset is for academic/research use only
- Larger and more diverse than URFD

---

## 🎯 Why This Dataset?

The Le2i dataset is valuable because:

1. **Realistic scenarios** - Home and office environments
2. **Large scale** - 321 videos (largest in our training set)
3. **Diversity** - Multiple scenarios, angles, lighting
4. **Quality** - Well-labeled and documented
5. **Challenging** - Includes occlusions and complex backgrounds

---

**Last Updated:** December 2024

