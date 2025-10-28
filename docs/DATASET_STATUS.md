# Dataset Status Report

**Date:** 2025-10-28  
**Status:** ⚠️ Small Test Dataset (Not Full Scale)

---

## Current Dataset Summary

### Raw Data (Phase 1.4 - Keypoint Extraction)
- **Total videos processed:** 4
  - URFD: 2 videos (fall-01, fall-02)
  - Le2i: 2 videos (Home_01_video 7, 21)
- **Total frames:** 702
- **Format:** `.npz` files with keypoints (T, 17, 3)

### Processed Data (Phase 1.5 - Feature Engineering)
- **Total windows:** 17
- **Window shape:** (60 frames, 6 features)
- **Features:** 6 (not 10 as expected in prompt)
  1. Torso angle (α)
  2. Hip height (h)
  3. Vertical velocity (v)
  4. Motion magnitude (m)
  5. Shoulder symmetry (s)
  6. Knee angle (θ)
- **Class distribution:**
  - Fall: 13 (76.5%)
  - Non-fall: 4 (23.5%)

### Window Generation Statistics
- **Possible windows:** 49 (from 702 frames with stride=10)
- **Kept windows:** 17 (34.7%)
- **Dropped windows:** 32 (65.3%)
- **Drop reason:** Missing data > 30% threshold
- **Average missing ratio:** 9.7%
- **Max missing ratio:** 24.4%

---

## Gap Analysis

### Expected (from prompt)
- **Samples:** ~2,000 windows
- **Features:** 10 engineered features
- **Datasets:** URFD (~700), Le2i (~800), Custom (~20), Kinetics/UCF101 (~500)

### Actual (current)
- **Samples:** 17 windows ❌
- **Features:** 6 engineered features ❌
- **Datasets:** URFD (12), Le2i (5) ❌

### Gap
- **Missing samples:** ~1,983 (99.2% short)
- **Missing features:** 4 features
- **Missing datasets:** Custom clips, Kinetics/UCF101 subset

---

## Root Causes

### 1. Limited Video Processing (Phase 1.4)
**Issue:** Only 4 videos were processed for keypoint extraction

**Full datasets available:**
- URFD: ~70 fall videos + ADL videos
- Le2i: ~191 videos (falls + ADL)
- Custom clips: Not collected yet
- Kinetics/UCF101: Not processed yet

**Impact:** 702 frames instead of ~60,000+ frames

### 2. High Window Dropout Rate (65.3%)
**Issue:** 32 out of 49 possible windows were dropped

**Reasons:**
- Low confidence keypoint detections → NaN values
- Missing data exceeds 30% threshold
- MoveNet struggles with certain poses/angles

**Potential solutions:**
- Relax quality threshold (e.g., 40% or 50%)
- Improve video quality/lighting
- Use multiple pose estimation models
- Apply keypoint interpolation/smoothing

### 3. Missing Features
**Issue:** Only 6 features implemented, prompt expects 10

**Current features:**
1. Torso angle (α)
2. Hip height (h)
3. Vertical velocity (v)
4. Motion magnitude (m)
5. Shoulder symmetry (s)
6. Knee angle (θ)

**Potential additional features:**
7. Head-hip distance
8. Limb angles (elbow, wrist)
9. Body aspect ratio
10. Centroid velocity

### 4. Dataset Imbalance
**Issue:** 76.5% fall, 23.5% non-fall

**Reasons:**
- URFD videos are all falls
- Only 1 Le2i video has non-fall windows
- Need more ADL (Activities of Daily Living) videos

---

## Training Implications

### Current Training Results (Phase 2.1)
- **Test samples:** 4 (too small for reliable evaluation)
- **Precision:** 0.75
- **Recall:** 1.00
- **F1:** 0.857
- **ROC-AUC:** 1.00

### Issues with Current Dataset
1. **Overfitting risk:** Only 11 training samples
2. **High variance:** Test set of 4 samples is unreliable
3. **Data leakage:** Multiple windows from same video
4. **Limited generalization:** Only 3 unique videos
5. **Class imbalance:** 76.5% fall samples

### Why Results Look "Too Good"
- Perfect ROC-AUC (1.0) is suspicious with 4 test samples
- Model likely memorized the few training examples
- No true generalization to unseen subjects/scenarios

---

## Recommendations

### Short-term (Work with current data)
1. ✅ **Document limitations** in results
2. ✅ **Implement subject-wise split** (already done)
3. ✅ **Use cross-validation** instead of single split
4. ⚠️ **Relax quality threshold** to get more windows
5. ⚠️ **Add more features** (4 additional)

### Medium-term (Expand dataset)
1. 🔄 **Process full URFD dataset** (~70 videos)
2. 🔄 **Process full Le2i dataset** (~191 videos)
3. 🔄 **Collect custom clips** (~20 videos)
4. 🔄 **Add Kinetics/UCF101 subset** (~500 videos)
5. 🔄 **Balance fall/non-fall** samples

### Long-term (Production-ready)
1. 📋 **Collect 1000+ samples** from diverse sources
2. 📋 **Multiple subjects** per scenario
3. 📋 **Diverse environments** (indoor/outdoor, lighting)
4. 📋 **Multiple camera angles**
5. 📋 **Real-world testing** on unseen data

---

## Action Items for Full-Scale Training

### To achieve ~2000 samples with 10 features:

#### Phase 1.4b - Process Full Datasets
```bash
# Process all URFD videos
python -m ml.pose.extract_keypoints \
    --source data/raw/urfd/ \
    --output data/interim/keypoints/ \
    --model movenet_thunder

# Process all Le2i videos
python -m ml.pose.extract_keypoints \
    --source data/raw/le2i/ \
    --output data/interim/keypoints/ \
    --model movenet_thunder
```

#### Phase 1.5b - Add More Features
```python
# Add to ml/features/feature_engineering.py:
# 7. Head-hip distance
# 8. Elbow angles
# 9. Body aspect ratio
# 10. Centroid velocity
```

#### Phase 1.5c - Relax Quality Threshold
```bash
# Re-run with relaxed threshold
python -m ml.features.feature_engineering \
    --source data/interim/keypoints \
    --out data/processed \
    --min-visible 0.5  # Changed from 0.7
    --drop-threshold 0.5  # Changed from 0.3
```

#### Phase 2.1b - Retrain with Full Dataset
```bash
python -m ml.training.lstm_train \
    --data data/processed/all_windows.npz \
    --epochs 100 \
    --batch 32 \
    --use-focal \
    --split-mode subject \
    --full-run
```

---

## Current Status: Proof of Concept ✅

The current implementation successfully demonstrates:
- ✅ End-to-end pipeline (keypoints → features → training)
- ✅ LSTM architecture as per proposal
- ✅ Data augmentation
- ✅ Evaluation metrics
- ✅ Visualization

**However:** Results are not production-ready due to limited dataset.

---

## Conclusion

The current dataset (17 samples, 6 features) is a **proof-of-concept** that validates the pipeline architecture and training methodology. To achieve production-ready performance as outlined in the proposal (~2000 samples, 10 features), we need to:

1. Process the full URFD and Le2i datasets
2. Add 4 more engineered features
3. Collect additional data sources
4. Implement proper subject-wise cross-validation

The training pipeline is ready and working correctly. The bottleneck is data collection and processing, not the model or training code.

---

**Next Steps:** Decide whether to:
- A) Continue with current small dataset for demonstration purposes
- B) Process full datasets to achieve production-scale training
- C) Implement enhancements (focal loss, subject-wise split) on current data first

