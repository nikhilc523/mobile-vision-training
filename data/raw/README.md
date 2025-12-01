# Training Datasets

This folder contains the raw datasets used for training the fall detection model.

**⚠️ Note:** Due to size constraints, the full datasets are stored on Google Drive.

---

## 📊 Datasets Overview

| Dataset | Size | Falls | Non-Falls | Google Drive Link |
|---------|------|-------|-----------|-------------------|
| **URFD** | 7.4 GB | 70 videos | 30 videos | [Download URFD](https://drive.google.com/drive/folders/YOUR_URFD_FOLDER_ID) |
| **Le2i** | 16 GB | 191 videos | 130 videos | [Download Le2i](https://drive.google.com/drive/folders/YOUR_LE2I_FOLDER_ID) |
| **UCF101 Subset** | 281 MB | Selected clips | N/A | [Download UCF101](https://drive.google.com/drive/folders/YOUR_UCF101_FOLDER_ID) |

**Total Training Data:** ~24 GB, 1000+ video sequences

---

## 📁 Folder Structure

```
data/raw/
├── urfd/                    # University of Rzeszow Fall Detection Dataset
│   ├── falls/              # 70 fall videos
│   ├── adl/                # 30 ADL (Activities of Daily Living) videos
│   └── README.md           # Dataset description
│
├── le2i/                    # Le2i Fall Detection Dataset
│   ├── falls/              # 191 fall videos
│   ├── adl/                # 130 ADL videos
│   └── README.md           # Dataset description
│
├── ucf101_subset/           # UCF101 Action Recognition (selected clips)
│   ├── falling/            # Fall-related action clips
│   └── README.md           # Dataset description
│
└── README.md               # This file
```

---

## 🔗 How to Download

### Option 1: Download from Google Drive (Recommended)

1. Click the Google Drive links above
2. Download the entire folder
3. Extract to `data/raw/` directory
4. Verify folder structure matches above

### Option 2: Download from Original Sources

#### URFD Dataset
- **Source:** University of Rzeszow
- **Link:** http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html
- **Citation:** Kwolek, B., & Kepski, M. (2014). Human fall detection on embedded platform using depth maps and wireless accelerometer. Computer methods and programs in biomedicine, 117(3), 489-501.

#### Le2i Dataset
- **Source:** Le2i Laboratory, University of Burgundy
- **Link:** http://le2i.cnrs.fr/Fall-detection-Dataset?lang=fr
- **Citation:** Charfi, I., Miteran, J., Dubois, J., Atri, M., & Tourki, R. (2012). Definition and performance evaluation of a robust SVM based fall detection solution. In 2012 Eighth International Conference on Signal Image Technology and Internet Based Systems (pp. 218-224). IEEE.

#### UCF101 Dataset
- **Source:** University of Central Florida
- **Link:** https://www.crcv.ucf.edu/data/UCF101.php
- **Citation:** Soomro, K., Zamir, A. R., & Shah, M. (2012). UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402.
- **Note:** We only use fall-related action clips from this dataset

---

## 📝 Dataset Descriptions

### URFD (University of Rzeszow Fall Detection)
- **Resolution:** 640×480
- **FPS:** 30
- **Format:** AVI
- **Scenarios:** Indoor falls, various directions
- **Subjects:** Multiple actors
- **Falls:** Forward, backward, sideways, from chair
- **ADL:** Walking, sitting, standing, bending

### Le2i Fall Detection Dataset
- **Resolution:** Various (320×240 to 640×480)
- **FPS:** 25-30
- **Format:** AVI
- **Scenarios:** Home environment, coffee room
- **Subjects:** Multiple actors
- **Falls:** Various fall types in realistic scenarios
- **ADL:** Daily activities (sitting, lying, crouching, walking)

### UCF101 Subset
- **Resolution:** Various
- **FPS:** 25-30
- **Format:** AVI
- **Categories Used:** Falling, tripping, collapsing
- **Note:** Selected clips relevant to fall detection

---

## 🔧 Data Preprocessing

After downloading, preprocess the data using:

```bash
# Extract keypoints from videos
python scripts/extract_keypoints.py --dataset urfd
python scripts/extract_keypoints.py --dataset le2i
python scripts/extract_keypoints.py --dataset ucf101

# Preprocess and create training sequences
python scripts/preprocess_data.py
```

This will:
1. Extract 30-frame sequences at 30 FPS
2. Extract 17 COCO keypoints per frame using MoveNet Thunder
3. Normalize coordinates to [0, 1]
4. Save preprocessed data to `data/processed/`

---

## 📊 Data Statistics

### Combined Dataset Statistics
- **Total Videos:** ~1000+
- **Total Falls:** ~260+
- **Total Non-Falls:** ~160+
- **Fall/Non-Fall Ratio:** ~62% falls, 38% non-falls
- **Total Size:** ~24 GB

### Training/Validation/Test Split
- **Training:** 80% (~800 videos)
- **Validation:** 10% (~100 videos)
- **Test:** 10% (~100 videos)
- **Custom Test Set:** 15 videos (separate, in `data/test/`)

---

## ⚠️ Important Notes

1. **Copyright:** These datasets are for academic/research use only
2. **Citations:** Please cite the original papers if you use these datasets
3. **Privacy:** Datasets contain actors, not real fall victims
4. **Size:** Total download size is ~24 GB, ensure sufficient disk space
5. **Processing Time:** Keypoint extraction takes ~2-4 hours on CPU

---

## 📧 Questions?

If you have issues downloading or preprocessing the data, please contact:
- **Email:** nikhil@example.com
- **GitHub Issues:** [Open an issue](https://github.com/nikhilc523/mobile-vision-training/issues)

---

**Last Updated:** December 2024

