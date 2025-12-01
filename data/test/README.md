# Test Videos

## 📊 Overview

This folder contains 15 custom test videos used to evaluate the fall detection system.

- **Total Videos:** 15
- **Falls:** 9 videos
- **Non-Falls:** 6 videos
- **Size:** 706 MB
- **Resolution:** Various (720p to 1080p)
- **FPS:** 30
- **Format:** MP4

**Test Accuracy:** 100% (15/15 correct)

---

## 🔗 Download

**Google Drive:** [Download Test Videos](https://drive.google.com/drive/folders/YOUR_TEST_FOLDER_ID)

---

## 📁 Test Videos List

### Falls (9 videos) ✅

1. **nihapass.mp4** - Slow fall
   - Person falling slowly to the ground
   - BiLSTM: 50.23% | Rule 5
   - ✅ Detected

2. **nihafast.mp4** - Fast fall
   - Person falling quickly/rapidly
   - BiLSTM: 0.80% | Rule 3
   - ✅ Detected

3. **nihacase6.mp4** - Sustained fall
   - Person on ground after falling
   - BiLSTM: 99.57% | Rule 1
   - ✅ Detected

4. **niha.mp4** - Person on ground
   - Person lying on ground after fall
   - BiLSTM: 99.93% | Rule 1
   - ✅ Detected

5. **2.mp4** - Slow fall (controlled descent)
   - Person falling slowly in controlled manner
   - BiLSTM: 50.23% | Rule 5
   - ✅ Detected

6. **nihaonelast.mp4** - Chair fall
   - Person falling from chair to ground
   - BiLSTM: 99.93% | Rule 2
   - ✅ Detected

7. **finalfall.mp4** - Backward fall
   - Person falling backward
   - BiLSTM: 99.57% | Rule 1
   - ✅ Detected

8. **pleasefall.mp4** - Forward fall
   - Person falling forward
   - BiLSTM: 95.12% | Rule 1
   - ✅ Detected

9. **03.mp4** - Edge case (0% model probability)
   - Person horizontal on ground for 1 second
   - BiLSTM: 0.00% | Rule 4
   - ✅ Detected (model missed, rules caught!)

---

### Non-Falls (6 videos) ✅

10. **nihastand.mp4** - Standing still
    - Person standing still (normal activity)
    - BiLSTM: 0.66%
    - ✅ No alert

11. **idle.mp4** - Moving around
    - Person moving around, idle activity
    - BiLSTM: 89.62% (filtered by stability!)
    - ✅ No alert (false positive prevented!)

12. **haha.mp4** - Normal activity
    - Person in normal activity
    - BiLSTM: 0.00%
    - ✅ No alert

13. **hehe.mp4** - Normal activity
    - Person in normal activity
    - BiLSTM: 0.00%
    - ✅ No alert

14. **usinglap.mp4** - Using laptop
    - Person sitting and using laptop
    - BiLSTM: 0.00%
    - ✅ No alert

15. **kushal.mp4** - Standing/upright
    - Person standing upright
    - BiLSTM: 0.00%
    - ✅ No alert

---

## 📊 Test Results Summary

| Metric | Value |
|--------|-------|
| **Total Videos** | 15 |
| **Correct Detections** | 15/15 (100%) |
| **True Positives** | 9/9 |
| **True Negatives** | 6/6 |
| **False Positives** | 0 |
| **False Negatives** | 0 |

---

## 🎯 Key Test Cases

### Critical Test Case 1: 03.mp4 (Edge Case)
- **Model Probability:** 0.00% (model completely missed it!)
- **Detection:** Rule 4 (sustained horizontal position)
- **Why Important:** Demonstrates necessity of rule-based enhancement

### Critical Test Case 2: idle.mp4 (False Positive Prevention)
- **Model Probability:** 89.62% (model thinks it's a fall!)
- **Detection:** Filtered by stability filter
- **Why Important:** Demonstrates importance of stability filtering

---

## 🔧 How to Test

```bash
# Test single video
python -m ml.export.enhanced_fall_detection data/test/finalfall.mp4

# Test all videos
python -m ml.export.test_all_videos

# Detailed analysis
python -m ml.export.analyze_fall_detailed data/test/nihafast.mp4
```

---

## 📝 Test Methodology

1. **Diverse Scenarios:** Covers fast falls, slow falls, chair falls, edge cases
2. **False Positive Testing:** Includes normal activities that could trigger false alarms
3. **Edge Cases:** Tests model limitations (03.mp4, idle.mp4)
4. **Real-World Conditions:** Various lighting, angles, movements
5. **Comprehensive Coverage:** All major fall types and non-fall activities

---

## ⚠️ Notes

- Videos recorded specifically for this project
- Actors: Nikhil, Kushal, Nandini, Niharika
- Safe environment with padded mats for fall videos
- Various camera angles and lighting conditions
- Resolution varies (720p to 1080p)
- All videos are 30 FPS

---

## 📈 Detailed Results

For detailed frame-by-frame analysis of each test video, see:
- [Test Results](../test_results/RESULTS.md)
- [Main README](../../README.md#detailed-test-results)

---

**Last Updated:** December 2024

