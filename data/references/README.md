# References - Research Papers and Documentation

This folder contains reference materials, research papers, and technical documentation used in the fall detection project.

---

## 📚 Contents

### Research Papers

**Paper_11-A_New_Method_for_Real_Time_Fall_Detection.pdf**
- **Title:** A New Method for Real-Time Fall Detection
- **Authors:** [Authors from paper]
- **Year:** [Year]
- **Description:** Reference paper on fall detection methods
- **Key Contributions:**
  - Real-time fall detection approach
  - Feature engineering techniques
  - LSTM-based classification

---

## 🔗 External References

### Datasets

**URFD (University of Rzeszow Fall Detection)**
- **Paper:** Kwolek, B., & Kepski, M. (2014). Human fall detection on embedded platform using depth maps and wireless accelerometer. *Computer methods and programs in biomedicine*, 117(3), 489-501.
- **Link:** http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html
- **Citation:**
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

**Le2i Fall Detection Dataset**
- **Paper:** Charfi, I., Miteran, J., Dubois, J., Atri, M., & Tourki, R. (2012). Definition and performance evaluation of a robust SVM based fall detection solution. In *2012 Eighth International Conference on Signal Image Technology and Internet Based Systems* (pp. 218-224). IEEE.
- **Link:** http://le2i.cnrs.fr/Fall-detection-Dataset?lang=fr
- **Citation:**
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

**UCF101 Action Recognition Dataset**
- **Paper:** Soomro, K., Zamir, A. R., & Shah, M. (2012). UCF101: A dataset of 101 human actions classes from videos in the wild. *arXiv preprint arXiv:1212.0402*.
- **Link:** https://www.crcv.ucf.edu/data/UCF101.php
- **Citation:**
```bibtex
@article{soomro2012ucf101,
  title={UCF101: A dataset of 101 human actions classes from videos in the wild},
  author={Soomro, Khurram and Zamir, Amir Roshan and Shah, Mubarak},
  journal={arXiv preprint arXiv:1212.0402},
  year={2012}
}
```

---

### Pose Estimation

**MoveNet**
- **Paper:** Next-generation pose detection with MoveNet and TensorFlow.js
- **Link:** https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
- **Model:** https://tfhub.dev/google/movenet/singlepose/lightning/4

**YOLO11 Pose**
- **Paper:** Ultralytics YOLO11
- **Link:** https://docs.ultralytics.com/tasks/pose/
- **Model:** yolo11n-pose.pt

**COCO Keypoints**
- **Link:** https://cocodataset.org/#keypoints-2020
- **Description:** 17-keypoint human pose annotation format

---

### Deep Learning

**LSTM Networks**
- **Paper:** Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.
- **Citation:**
```bibtex
@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997},
  publisher={MIT Press}
}
```

**Bidirectional LSTM**
- **Paper:** Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE transactions on Signal Processing*, 45(11), 2673-2681.

**Focal Loss**
- **Paper:** Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE international conference on computer vision* (pp. 2980-2988).
- **Link:** https://arxiv.org/abs/1708.02002
- **Use Case:** Handles class imbalance in fall detection

---

## 📖 Project Documentation

### Weekly Reports
- `docs/weekly_reports/WEEK_1_REPORT.md` - Dataset preparation and pose estimation
- `docs/weekly_reports/WEEK_2_REPORT.md` - Keypoint extraction
- `docs/weekly_reports/WEEK_3_REPORT.md` - Feature engineering and training
- `docs/weekly_reports/WEEK_4_REPORT.md` - Evaluation and deployment

### Technical Documentation
- `docs/dataset_notes.md` - Dataset overview
- `docs/movenet_pose_estimation.md` - Pose estimation guide
- `docs/MODEL_ARCHITECTURE.md` - Model architecture details
- `docs/ANDROID_INTEGRATION.md` - Android app integration

---

## 🎓 Related Work

### Fall Detection Methods

1. **Vision-based Fall Detection**
   - Pose estimation + classification
   - Depth sensors (Kinect)
   - Optical flow analysis

2. **Wearable Sensors**
   - Accelerometers
   - Gyroscopes
   - Pressure sensors

3. **Hybrid Approaches**
   - Vision + wearables
   - Multi-modal fusion

### Our Approach

**Method:** Vision-based fall detection using pose estimation and BiLSTM

**Advantages:**
- ✅ No wearable devices required
- ✅ Works with standard cameras
- ✅ Real-time processing (30 FPS)
- ✅ High accuracy (99.42% F1 score)

**Key Innovations:**
- 8-rule enhanced detection system
- Stability filtering for false positive reduction
- Hybrid model + rule-based approach
- On-device deployment (Android)

---

## 📝 How to Cite This Project

If you use this project in your research, please cite:

```bibtex
@misc{chowdary2024falldetection,
  title={Real-Time Fall Detection Using Pose Estimation and BiLSTM},
  author={Chowdary, Nikhil and Team},
  year={2024},
  howpublished={\url{https://github.com/nikhilc523/mobile-vision-training}}
}
```

---

## 🔗 Useful Links

- **TensorFlow:** https://www.tensorflow.org/
- **Keras:** https://keras.io/
- **Ultralytics:** https://ultralytics.com/
- **Android ML Kit:** https://developers.google.com/ml-kit
- **TensorFlow Lite:** https://www.tensorflow.org/lite

---

**Last Updated:** December 2024

