# GitHub Issues for Fall Detection Project

**Project Timeline:** October 17 - November 3, 2025 (4 weeks)
**Repository:** https://github.com/nikhilc523/mobile-vision-training

This document contains detailed GitHub issues organized by week for the Fall Detection project.

---

## 📅 Timeline Overview

| Week | Dates | Focus | Report Due |
|------|-------|-------|------------|
| Week 1 | Oct 17-23 | Dataset Preparation & Pose Extraction | Oct 25 |
| Week 2 | Oct 24-30 | Full Dataset Extraction & Feature Engineering | Nov 1 |
| Week 3 | Oct 31-Nov 6 | LSTM Training & Evaluation | Nov 8 |
| Week 4 | Nov 7-13 | Optimization & Deployment | Nov 15 |

---

## 🗓️ WEEK 1 ISSUES (Oct 17-23) - ✅ COMPLETED

### Issue #1: Download and Prepare URFD and Le2i Datasets

**Status:** ✅ Done
**Priority:** 🔴 CRITICAL
**Labels:** `dataset`, `setup`, `week-1`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 1 - Dataset Preparation

**Description:**

Download and organize the URFD and Le2i fall detection datasets, validate integrity, and prepare for pose extraction.

**Tasks:**
- [x] Download URFD dataset (31 fall + 32 ADL sequences)
- [x] Download Le2i dataset (190 videos, 6 scenes)
- [x] Extract and organize files in `data/raw/`
- [x] Flatten nested directory structures
- [x] Validate all video files are readable
- [x] Match Le2i videos with annotation files
- [x] Clean up unnecessary files (.zip, .DS_Store)
- [x] Generate dataset statistics
- [x] Create documentation

**Deliverables:**
- [x] `scripts/prepare_datasets.py` (400 lines)
- [x] `scripts/validate_and_cleanup_datasets.py` (500 lines)
- [x] `docs/dataset_notes.md`
- [x] 253 validated video sequences

**Results:**
- ✅ 253 videos prepared (63 URFD + 190 Le2i)
- ✅ ~91,000 frames total
- ✅ 100% validation success
- ✅ 68 unnecessary files cleaned

**Time Spent:** ~8 hours

---

### Issue #2: Implement Le2i Annotation Parser

**Status:** ✅ Done
**Priority:** 🟠 HIGH
**Labels:** `parser`, `annotations`, `week-1`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 1 - Dataset Preparation

**Description:**

Create a robust parser for Le2i annotation files to extract fall frame ranges and match them with video files.

**Tasks:**
- [x] Implement `parse_annotation()` function
- [x] Implement `match_video_for_annotation()` function
- [x] Implement `get_fall_ranges()` function
- [x] Handle missing/malformed annotations gracefully
- [x] Create CLI interface for testing
- [x] Write 10 unit tests
- [x] Test on all 130 Le2i annotation files
- [x] Create documentation

**Deliverables:**
- [x] `ml/data/parsers/le2i_annotations.py` (250 lines)
- [x] `ml/tests/test_le2i_annotations.py` (280 lines)
- [x] `docs/le2i_parser_summary.md`
- [x] 10/10 tests passing

**Results:**
- ✅ 130/130 annotations parsed successfully
- ✅ 100% test pass rate
- ✅ Robust error handling

**Time Spent:** ~4 hours

---

### Issue #3: Implement MoveNet Pose Estimation

**Status:** ✅ Done
**Priority:** 🔴 CRITICAL
**Labels:** `pose-estimation`, `movenet`, `week-1`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 1 - Dataset Preparation

**Description:**

Implement MoveNet Lightning pose estimation pipeline for extracting 17 COCO keypoints from video frames.

**Tasks:**
- [x] Load MoveNet Lightning v4 from TensorFlow Hub
- [x] Implement frame preprocessing (192×192 with padding)
- [x] Implement single-frame inference
- [x] Extract 17 keypoints (y, x, confidence)
- [x] Implement confidence-based masking (threshold 0.3)
- [x] Create skeleton visualization
- [x] Write 14 unit tests
- [x] Benchmark inference speed
- [x] Create comprehensive documentation

**Deliverables:**
- [x] `ml/pose/movenet_loader.py` (318 lines)
- [x] `ml/tests/test_movenet_loader.py` (520 lines)
- [x] `docs/movenet_pose_estimation.md`
- [x] 14/14 tests passing

**Results:**
- ✅ 30ms/frame on CPU (33 FPS)
- ✅ 15.3/17 avg keypoints on falls
- ✅ 100% test pass rate

**Time Spent:** ~6 hours

---

## 🗓️ WEEK 2 ISSUES (Oct 24-30) - ✅ COMPLETED

### Issue #4: Extract Pose Keypoints from Full URFD + Le2i Dataset

**Status:** ✅ Done
**Priority:** 🔴 CRITICAL
**Labels:** `pose-extraction`, `data-pipeline`, `week-2`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 2 - Full Dataset Extraction

**Description:**

Process all 253 URFD and Le2i videos to extract MoveNet pose keypoints and save as compressed .npz files.

**Tasks:**
- [x] Create `ml/data/extract_pose_sequences.py`
- [x] Implement batch processing for URFD (image sequences)
- [x] Implement batch processing for Le2i (videos)
- [x] Add progress tracking with tqdm
- [x] Add `--skip-existing` flag
- [x] Add `--model` flag (lightning/thunder)
- [x] Process all 253 videos
- [x] Handle MoveNet Thunder failures (switch to Lightning)
- [x] Save compressed .npz files
- [x] Append results to `docs/results1.md`

**Deliverables:**
- [x] `ml/data/extract_pose_sequences.py` (597 lines)
- [x] 253 .npz keypoint files (~4 MB)
- [x] `docs/PHASE_1_4B_SUMMARY.md`

**Results:**
- ✅ 253 videos processed (63 URFD + 190 Le2i)
- ✅ 85,611 frames extracted
- ✅ 99.6% success rate
- ✅ 100% file validation

**Time Spent:** ~4 hours

---

### Issue #5: Extract Pose Keypoints from UCF101 Subset (Non-Fall Samples)

**Status:** ✅ Done
**Priority:** 🟠 HIGH
**Labels:** `pose-extraction`, `ucf101`, `week-2`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 2 - Full Dataset Extraction

**Description:**

Extract pose keypoints from 711 UCF101 videos (7 non-fall activity classes) to balance the dataset.

**Tasks:**
- [x] Create `ml/data/ucf101_extract.py`
- [x] Process 7 classes: ApplyEyeMakeup, BodyWeightSquats, JumpingJack, Lunges, MoppingFloor, PullUps, PushUps
- [x] Extract keypoints from all 711 videos
- [x] Label all as non-fall (label=0)
- [x] Save compressed .npz files
- [x] Verify extraction integrity
- [x] Update documentation

**Deliverables:**
- [x] `ml/data/ucf101_extract.py` (300 lines)
- [x] 711 .npz keypoint files (~5 MB)
- [x] `docs/PHASE_1_4C_VERIFICATION.md`

**Results:**
- ✅ 711 videos processed (100%)
- ✅ 111,800 frames extracted
- ✅ 100% success rate
- ✅ Improved class balance to ~18% fall / ~82% non-fall

**Time Spent:** ~3 hours

---

### Issue #6: Create Extraction Verification Script

**Status:** ✅ Done
**Priority:** 🟡 MEDIUM
**Labels:** `verification`, `testing`, `week-2`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 2 - Full Dataset Extraction

**Description:**

Create a comprehensive verification script to validate all extracted .npz keypoint files.

**Tasks:**
- [x] Create `scripts/verify_extraction_integrity.py`
- [x] Check keypoint shape (T, 17, 3)
- [x] Validate labels (0 or 1)
- [x] Validate FPS values
- [x] Validate confidence ranges [0, 1]
- [x] Validate coordinate ranges ~[0, 1]
- [x] Add CLI arguments for dataset filtering
- [x] Generate verification report
- [x] Test on all 964 files

**Deliverables:**
- [x] `scripts/verify_extraction_integrity.py` (250 lines)
- [x] Verification report

**Results:**
- ✅ 964/964 files valid (100%)
- ✅ All checks passed

**Time Spent:** ~2 hours

---

## 🗓️ WEEK 3 ISSUES (Oct 31-Nov 6) - ⏳ IN PROGRESS

### Issue #7: Implement Feature Engineering Pipeline

**Status:** 🟡 Partial (only 4 videos processed)
**Priority:** 🔴 CRITICAL
**Labels:** `feature-engineering`, `data-pipeline`, `week-3`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 3 - LSTM Training

**Description:**

Implement feature engineering to convert raw keypoints into 6 temporal features and create 60-frame windows for LSTM training. **CRITICAL: Must re-run on full 964-video dataset.**

**Current Status:**
- ✅ 6 features implemented
- ✅ Windowing strategy implemented (60 frames, stride 10)
- ✅ Quality filtering implemented (>30% missing data threshold)
- ❌ Only 17 windows from 4 videos (BLOCKER)
- ❌ Need ~10,000+ windows from 964 videos

**Tasks:**
- [x] Implement 6 features:
  1. Torso angle (α)
  2. Hip height (h)
  3. Vertical velocity (v)
  4. Motion magnitude (m)
  5. Shoulder symmetry (s)
  6. Knee angle (θ)
- [x] Implement temporal windowing (60 frames, stride 10)
- [x] Implement quality filtering
- [x] Create visualizations
- [ ] **TODO: Update to process all 964 videos**
- [ ] **TODO: Add CLI argument `--dataset all`**
- [ ] **TODO: Run on full dataset**
- [ ] **TODO: Verify ~10,000+ windows generated**
- [ ] **TODO: Check class balance (~30% fall / ~70% non-fall)**

**Command:**
```bash
# Current (only 4 videos)
python -m ml.features.feature_engineering

# TODO: Update and run
python -m ml.features.feature_engineering --dataset all --output data/processed/full_windows.npz
```

**Expected Output:**
- File: `data/processed/full_windows.npz`
- Shape: X: (~10000, 60, 6), y: (~10000,)
- Class balance: ~30% fall, ~70% non-fall

**Deliverables:**
- [x] `ml/features/feature_engineering.py` (450 lines)
- [x] Proof-of-concept: 17 windows from 4 videos
- [ ] **TODO: Full dataset: ~10,000+ windows from 964 videos**

**Acceptance Criteria:**
- [ ] ≥10,000 windows generated
- [ ] Class balance 20-40% fall, 60-80% non-fall
- [ ] All 964 videos processed
- [ ] Output file saved and verified
- [ ] Documentation updated

**Estimated Time:** 2-3 hours (including processing time)

---

### Issue #8: Add 4 Additional Engineered Features (Total 10)

**Status:** ⏳ TODO
**Priority:** 🟠 HIGH
**Labels:** `feature-engineering`, `enhancement`, `week-3`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 3 - LSTM Training

**Description:**

Current implementation has 6 features. Project proposal specifies 10 features. Add 4 more features to improve model performance and meet proposal requirements.

**Current Features (6):**
1. ✅ Torso angle (α)
2. ✅ Hip height (h)
3. ✅ Vertical velocity (v)
4. ✅ Motion magnitude (m)
5. ✅ Shoulder symmetry (s)
6. ✅ Knee angle (θ)

**New Features to Add (4):**

7. **Bounding Box Area** - Overall body size/scale
   - Calculate min/max x,y of all keypoints
   - Area = (max_x - min_x) * (max_y - min_y)
   - Fall indicator: Area increases when lying down
   - Helps detect body orientation changes

8. **Head Velocity** - Speed of head movement
   - Track nose keypoint velocity over time
   - L2 norm of (Δx, Δy) per frame
   - Fall indicator: High velocity during fall, low after
   - Helps detect sudden movements

9. **Limb Angles** - Arm and leg extension
   - Calculate elbow angles (shoulder-elbow-wrist)
   - Calculate knee angles (hip-knee-ankle)
   - Average all limb angles
   - Fall indicator: Extended limbs when falling/lying
   - Helps detect body posture

10. **Pose Confidence** - Overall detection quality
    - Average confidence across all keypoints
    - Indicates occlusion or distance from camera
    - Fall indicator: May drop during fall
    - Helps filter low-quality frames

**Tasks:**
- [ ] Implement bounding box area calculation
- [ ] Implement head velocity calculation
- [ ] Implement limb angles calculation
- [ ] Implement pose confidence calculation
- [ ] Update `ml/features/feature_engineering.py`
- [ ] Test on sample data
- [ ] Verify feature ranges and distributions
- [ ] Create visualizations for new features
- [ ] Update documentation
- [ ] Re-run on full dataset with 10 features

**Implementation Notes:**
```python
# Bounding box area
valid_keypoints = keypoints[keypoints[:, 2] > 0.3]  # High confidence only
if len(valid_keypoints) > 0:
    bbox_area = (valid_keypoints[:, 0].max() - valid_keypoints[:, 0].min()) * \
                (valid_keypoints[:, 1].max() - valid_keypoints[:, 1].min())

# Head velocity
nose_pos = keypoints[:, 0, :2]  # Nose keypoint (x, y)
head_velocity = np.linalg.norm(np.diff(nose_pos, axis=0), axis=1)

# Limb angles
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

# Pose confidence
pose_confidence = np.mean(keypoints[:, :, 2])
```

**Deliverables:**
- [ ] Updated `ml/features/feature_engineering.py`
- [ ] 10 features implemented
- [ ] Feature distribution plots
- [ ] Updated documentation

**Acceptance Criteria:**
- [ ] All 4 features implemented and tested
- [ ] Features validated on sample data
- [ ] Feature distributions visualized
- [ ] Documentation updated with feature descriptions
- [ ] Code committed and pushed

**Estimated Time:** 4-5 hours

---

### Issue #9: Implement Initial LSTM Training Pipeline

**Status:** ✅ Done (proof-of-concept)
**Priority:** 🔴 CRITICAL
**Labels:** `training`, `lstm`, `week-3`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 3 - LSTM Training

**Description:**

Implement LSTM training pipeline as specified in project proposal. Train initial model on proof-of-concept dataset (17 samples) to validate architecture and pipeline.

**Model Architecture (as per proposal):**
```python
model = keras.Sequential([
    Masking(mask_value=0.0, input_shape=(60, 6)),
    LSTM(64 units),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

**Tasks:**
- [x] Implement model architecture
- [x] Implement data loading and preprocessing
- [x] Implement train/val/test splitting
- [x] Implement class weighting
- [x] Implement data augmentation (time-warp, noise, dropout)
- [x] Implement early stopping
- [x] Implement metrics calculation (Precision, Recall, F1, ROC-AUC)
- [x] Create visualizations (training curves, ROC, confusion matrix)
- [x] Train on proof-of-concept dataset (17 samples)
- [x] Save model checkpoint
- [x] Document results

**Deliverables:**
- [x] `ml/training/lstm_train.py` (698 lines)
- [x] `ml/training/augmentation.py` (230 lines)
- [x] Model checkpoint: `ml/training/checkpoints/lstm_best.h5` (79 KB)
- [x] Training visualizations
- [x] `ml/training/README.md`

**Results (Proof-of-Concept):**
- ✅ Model trained successfully
- ✅ Test F1: 0.857, Recall: 1.0, Precision: 0.75
- ✅ Model size: 79 KB (mobile-ready)
- ⚠️ Only 17 samples - not production-ready

**Next Steps:**
- [ ] Re-train on full dataset (~10,000+ samples) after Issue #7 complete
- [ ] Implement focal loss (Issue #10)
- [ ] Implement subject-wise splitting (Issue #11)

**Time Spent:** ~4 hours

---

## 🗓️ WEEK 4 ISSUES (Nov 7-13) - ⏳ TODO

### Issue #10: Implement Focal Loss for Class Imbalance

**Status:** ⏳ TODO
**Priority:** 🔴 CRITICAL
**Labels:** `training`, `model-improvement`, `week-4`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 4 - Optimization & Deployment

**Description:**

Current training uses standard binary cross-entropy loss, which doesn't handle class imbalance well (~18% fall / ~82% non-fall). Implement focal loss to focus training on hard-to-classify examples and improve performance on minority class (falls).

**Background:**
- Current dataset: ~18% fall / ~82% non-fall imbalance
- Standard BCE treats all samples equally
- Focal loss down-weights easy examples, focuses on hard ones
- Improves recall on minority class (falls) without sacrificing precision

**What is Focal Loss?**
Focal loss adds a modulating factor to cross-entropy loss:
```
FL(pt) = -α(1-pt)^γ * log(pt)
```
- α (alpha): Weight for positive class (typically 0.25)
- γ (gamma): Focusing parameter (typically 2.0)
- pt: Model's estimated probability for the correct class

**Tasks:**
- [ ] Install TensorFlow Addons: `pip install tensorflow-addons`
- [ ] Import focal loss: `from tensorflow_addons.losses import SigmoidFocalCrossEntropy`
- [ ] Update `ml/training/lstm_train.py` to use focal loss
- [ ] Add CLI arguments:
  - `--focal-loss` (enable focal loss)
  - `--focal-alpha` (default: 0.25)
  - `--focal-gamma` (default: 2.0)
- [ ] Test on current 17-sample dataset
- [ ] Compare performance with standard BCE
- [ ] Train on full dataset with focal loss
- [ ] Document results and performance comparison

**Implementation:**
```python
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

# Focal loss with alpha=0.25, gamma=2.0 (standard values)
focal_loss = SigmoidFocalCrossEntropy(
    alpha=0.25,  # Weight for positive class (falls)
    gamma=2.0,   # Focusing parameter
    reduction=tf.keras.losses.Reduction.AUTO
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=focal_loss,
    metrics=['accuracy', Precision(), Recall(), AUC()]
)
```

**Expected Benefits:**
- Improved recall on falls (minority class)
- Better handling of class imbalance
- More robust model performance
- Higher F1 score

**Deliverables:**
- [ ] Updated `ml/training/lstm_train.py` with focal loss
- [ ] CLI arguments for focal loss configuration
- [ ] Performance comparison report
- [ ] Updated documentation

**Acceptance Criteria:**
- [ ] Focal loss implemented and tested
- [ ] CLI arguments added and working
- [ ] Performance comparison documented (BCE vs Focal Loss)
- [ ] Improved F1 score on test set
- [ ] Code committed and pushed

**Estimated Time:** 2-3 hours

---

### Issue #11: Implement Subject-Wise Data Splitting

**Status:** ⏳ TODO
**Priority:** 🔴 CRITICAL
**Labels:** `training`, `data-pipeline`, `week-4`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 4 - Optimization & Deployment

**Description:**

Current random splitting may leak information (same subject/video in train and test). Implement subject-wise splitting to ensure proper generalization evaluation and prevent data leakage.

**Background:**
- **Data Leakage:** When same subject appears in both train and test sets
- **Problem:** Model memorizes subject-specific patterns, not general fall patterns
- **Solution:** Split by subject/video to ensure true generalization

**Dataset Structure:**
- **URFD:** Each sequence is a different subject (63 unique subjects)
- **Le2i:** Multiple videos per scene, likely same subjects within scene
- **UCF101:** Multiple videos per person/class

**Why This Matters:**
- Without subject-wise splitting: Model may achieve 95% accuracy but fail on new subjects
- With subject-wise splitting: True measure of generalization to unseen people

**Tasks:**
- [ ] Extract subject/video identifiers from filenames
- [ ] Group windows by subject/video
- [ ] Implement subject-wise train/val/test split (70/15/15)
- [ ] Ensure no subject appears in multiple splits
- [ ] Update `ml/features/feature_engineering.py` to save subject IDs
- [ ] Update `ml/training/lstm_train.py` to use subject-wise splitting
- [ ] Add CLI argument `--split-by subject` (default: random)
- [ ] Create verification script to check split integrity
- [ ] Document approach and results

**Implementation:**
```python
# Extract subject IDs from filenames
subject_ids = []
for filename in filenames:
    if 'urfd' in filename:
        # urfd_fall_fall-01-cam0-rgb.npz → fall-01
        subject_id = 'urfd_' + filename.split('_')[2].rsplit('-', 2)[0]
    elif 'le2i' in filename:
        # le2i_Home_01_video (1).npz → le2i_Home_01_video (1)
        subject_id = 'le2i_' + '_'.join(filename.split('_')[1:])
    elif 'ucf101' in filename:
        # ucf101_ApplyEyeMakeup_v_ApplyEyeMakeup_g01_c01.npz → ucf101_v_ApplyEyeMakeup_g01_c01
        subject_id = 'ucf101_' + '_'.join(filename.split('_')[2:])
    subject_ids.append(subject_id)

# Split by subject
unique_subjects = np.unique(subject_ids)
print(f"Total unique subjects: {len(unique_subjects)}")

# Stratified split by fall/non-fall ratio
train_subjects, test_subjects = train_test_split(
    unique_subjects, test_size=0.3, random_state=42, stratify=subject_labels
)
val_subjects, test_subjects = train_test_split(
    test_subjects, test_size=0.5, random_state=42, stratify=test_labels
)

# Create splits
train_mask = np.isin(subject_ids, train_subjects)
val_mask = np.isin(subject_ids, val_subjects)
test_mask = np.isin(subject_ids, test_subjects)

# Verify no overlap
assert len(set(train_subjects) & set(val_subjects)) == 0
assert len(set(train_subjects) & set(test_subjects)) == 0
assert len(set(val_subjects) & set(test_subjects)) == 0
print("✓ No subject overlap between splits")
```

**Deliverables:**
- [ ] Updated `ml/features/feature_engineering.py` to save subject IDs
- [ ] Updated `ml/training/lstm_train.py` with subject-wise splitting
- [ ] Verification script: `scripts/verify_split_integrity.py`
- [ ] Documentation of approach

**Acceptance Criteria:**
- [ ] Subject-wise splitting implemented
- [ ] No subject overlap between splits (verified)
- [ ] Split ratios maintained (~70/15/15)
- [ ] CLI argument `--split-by subject` added
- [ ] Verification script created and passing
- [ ] Documentation updated
- [ ] Performance comparison (random vs subject-wise) documented

**Estimated Time:** 3-4 hours

---

### Issue #12: Train Final LSTM Model on Full Dataset

**Status:** ⏳ TODO
**Priority:** 🔴 CRITICAL
**Labels:** `training`, `lstm`, `week-4`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 4 - Optimization & Deployment

**Description:**

Train final LSTM model on full dataset (~10,000+ windows) with all improvements: 10 features, focal loss, subject-wise splitting, and data augmentation.

**Prerequisites:**
- ✅ Issue #7: Feature engineering on full dataset complete
- ✅ Issue #8: 10 features implemented
- ✅ Issue #10: Focal loss implemented
- ✅ Issue #11: Subject-wise splitting implemented

**Tasks:**
- [ ] Load full dataset (~10,000+ windows, 10 features)
- [ ] Apply subject-wise splitting (70/15/15)
- [ ] Configure model with focal loss
- [ ] Enable data augmentation
- [ ] Train for 100 epochs with early stopping
- [ ] Monitor training metrics (loss, F1, precision, recall)
- [ ] Save best model checkpoint
- [ ] Generate training visualizations
- [ ] Calculate test metrics
- [ ] Compare with baseline (17-sample model)
- [ ] Document results

**Training Configuration:**
```bash
python -m ml.training.lstm_train \
    --data data/processed/full_windows.npz \
    --epochs 100 \
    --batch 32 \
    --lr 1e-3 \
    --focal-loss \
    --focal-alpha 0.25 \
    --focal-gamma 2.0 \
    --split-by subject \
    --augment \
    --save-best \
    --output ml/training/checkpoints/lstm_final.h5
```

**Target Performance (from proposal):**
- Precision ≥ 0.90
- Recall ≥ 0.90
- F1 ≥ 0.90
- ROC-AUC ≥ 0.90

**Deliverables:**
- [ ] Trained model: `ml/training/checkpoints/lstm_final.h5`
- [ ] Training history: `ml/training/history/lstm_final_history.csv`
- [ ] Training visualizations (loss curves, ROC, confusion matrix)
- [ ] Test metrics report
- [ ] Performance comparison with baseline

**Acceptance Criteria:**
- [ ] Model trained on ≥10,000 samples
- [ ] Test F1 ≥ 0.85 (target: 0.90)
- [ ] Test Recall ≥ 0.85 (critical for safety)
- [ ] Test Precision ≥ 0.80
- [ ] Model size < 500 KB
- [ ] All visualizations generated
- [ ] Results documented

**Estimated Time:** 2-3 hours (+ compute time)

---

### Issue #13: Implement Comprehensive Evaluation Pipeline

**Status:** ⏳ TODO
**Priority:** 🟡 MEDIUM
**Labels:** `evaluation`, `metrics`, `week-4`
**Assignee:** Nikhil Chowdary
**Milestone:** Week 4 - Optimization & Deployment

**Description:**

Create comprehensive evaluation pipeline to assess model performance on held-out test set with detailed metrics, visualizations, and error analysis.

**Tasks:**
- [ ] Create `ml/evaluation/evaluate_model.py` script
- [ ] Load trained model and test data
- [ ] Calculate comprehensive metrics:
  - Precision, Recall, F1 Score
  - ROC-AUC, PR-AUC
  - Confusion matrix
  - Per-class metrics
  - False positive/negative analysis
  - Per-subject performance
  - Fall detection latency (time to detect after fall starts)
- [ ] Generate visualizations:
  - ROC curve with AUC score
  - Precision-Recall curve
  - Confusion matrix heatmap
  - Per-subject performance bar chart
  - Temporal analysis (detection latency histogram)
  - Error analysis (false positives/negatives)
- [ ] Create evaluation report (Markdown)
- [ ] Compare with baseline (17-sample model)
- [ ] Document results in `docs/results1.md`

**Evaluation Script Usage:**
```bash
python -m ml.evaluation.evaluate_model \
    --model ml/training/checkpoints/lstm_final.h5 \
    --data data/processed/full_windows.npz \
    --output docs/evaluation_report.md
```

**Deliverables:**
- [ ] `ml/evaluation/evaluate_model.py` script
- [ ] Evaluation report: `docs/evaluation_report.md`
- [ ] Visualizations in `docs/wiki_assets/evaluation/`
- [ ] Updated `docs/results1.md`

**Acceptance Criteria:**
- [ ] Evaluation script created and working
- [ ] All metrics calculated and documented
- [ ] All visualizations generated
- [ ] Evaluation report written
- [ ] Performance comparison with baseline
- [ ] Results documented

**Estimated Time:** 4-5 hours

---

## 📋 Summary of All Issues

### By Week

**Week 1 (Oct 17-23):** ✅ COMPLETED
- Issue #1: Download and Prepare Datasets
- Issue #2: Le2i Annotation Parser
- Issue #3: MoveNet Pose Estimation

**Week 2 (Oct 24-30):** ✅ COMPLETED
- Issue #4: Extract URFD + Le2i Keypoints
- Issue #5: Extract UCF101 Keypoints
- Issue #6: Verification Script

**Week 3 (Oct 31-Nov 6):** 🟡 IN PROGRESS
- Issue #7: Feature Engineering (partial - need full dataset)
- Issue #8: Add 4 More Features (TODO)
- Issue #9: Initial LSTM Training (done - proof-of-concept)

**Week 4 (Nov 7-13):** ⏳ TODO
- Issue #10: Implement Focal Loss
- Issue #11: Implement Subject-Wise Splitting
- Issue #12: Train Final Model on Full Dataset
- Issue #13: Comprehensive Evaluation

### By Priority

| Issue # | Title | Week | Priority | Time | Status |
|---------|-------|------|----------|------|--------|
| #1 | Download and Prepare Datasets | 1 | 🔴 CRITICAL | 8h | ✅ Done |
| #2 | Le2i Annotation Parser | 1 | 🟠 HIGH | 4h | ✅ Done |
| #3 | MoveNet Pose Estimation | 1 | 🔴 CRITICAL | 6h | ✅ Done |
| #4 | Extract URFD + Le2i Keypoints | 2 | 🔴 CRITICAL | 4h | ✅ Done |
| #5 | Extract UCF101 Keypoints | 2 | 🟠 HIGH | 3h | ✅ Done |
| #6 | Verification Script | 2 | 🟡 MEDIUM | 2h | ✅ Done |
| #7 | Feature Engineering Pipeline | 3 | 🔴 CRITICAL | 2-3h | 🟡 Partial |
| #8 | Add 4 More Features | 3 | 🟠 HIGH | 4-5h | ⏳ TODO |
| #9 | Initial LSTM Training | 3 | 🔴 CRITICAL | 4h | ✅ Done |
| #10 | Implement Focal Loss | 4 | 🔴 CRITICAL | 2-3h | ⏳ TODO |
| #11 | Subject-Wise Splitting | 4 | 🔴 CRITICAL | 3-4h | ⏳ TODO |
| #12 | Train Final Model | 4 | 🔴 CRITICAL | 2-3h | ⏳ TODO |
| #13 | Comprehensive Evaluation | 4 | 🟡 MEDIUM | 4-5h | ⏳ TODO |

**Total Time Spent (Weeks 1-3):** ~31 hours
**Remaining Time (Week 4):** ~15-20 hours

---

## 🎯 Week 4 Critical Path (Nov 7-13)

### Day 1-2: Complete Feature Engineering
1. **Issue #7:** Re-run feature engineering on full 964 videos (2-3h)
2. **Issue #8:** Add 4 additional features (4-5h)

### Day 3: Training Improvements
3. **Issue #10:** Implement focal loss (2-3h)
4. **Issue #11:** Implement subject-wise splitting (3-4h)

### Day 4-5: Final Training & Evaluation
5. **Issue #12:** Train final model on full dataset (2-3h + compute)
6. **Issue #13:** Comprehensive evaluation (4-5h)

---

*Last updated: October 28, 2025*

