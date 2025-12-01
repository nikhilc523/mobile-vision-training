# 🔄 Gemini Fine-Tuning Workflow

## 📊 Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR ASSIGNMENT WORKFLOW                      │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Prepare Dataset (YOU DO THIS)
┌──────────────────────────────────────────────────────────────┐
│  Your Videos (URFD + Le2i + UCF101)                          │
│  ├── data/raw/urfd/falls/     (70 fall videos)              │
│  ├── data/raw/urfd/adl/       (40 non-fall videos)          │
│  ├── data/raw/le2i/           (321 videos)                  │
│  └── data/raw/ucf101_subset/  (100 non-fall videos)         │
│                                                               │
│  Run: python finetune/prepare_urfd_le2i_dataset.py          │
│                                                               │
│  ↓ Extracts 3 frames per video                              │
│                                                               │
│  Output:                                                      │
│  ├── finetune/frames_full/                                  │
│  │   ├── fall/         (~783 images)                        │
│  │   └── non_fall/     (~810 images)                        │
│  ├── fall_detection_dataset_full.json                       │
│  ├── train_split_full.json                                  │
│  ├── val_split_full.json                                    │
│  └── test_split_full.json                                   │
└──────────────────────────────────────────────────────────────┘
                              ↓
                              
STEP 2: Zip Frames (YOU DO THIS)
┌──────────────────────────────────────────────────────────────┐
│  Run: cd finetune && zip -r frames_full.zip frames_full/    │
│                                                               │
│  Output: frames_full.zip (~200-300 MB)                      │
└──────────────────────────────────────────────────────────────┘
                              ↓
                              
STEP 3: Set Up Google Cloud (YOU DO THIS)
┌──────────────────────────────────────────────────────────────┐
│  1. Create Google Cloud Project                              │
│     → https://console.cloud.google.com/                     │
│                                                               │
│  2. Enable APIs                                              │
│     → Vertex AI API                                          │
│     → Cloud Storage API                                      │
│                                                               │
│  3. Create Service Account                                   │
│     → Download JSON key                                      │
│                                                               │
│  4. Create Storage Bucket                                    │
│     → Name: fall-detection-finetuning-<yourname>            │
│     → Region: us-central1                                    │
└──────────────────────────────────────────────────────────────┘
                              ↓
                              
STEP 4: Upload to Colab (YOU DO THIS)
┌──────────────────────────────────────────────────────────────┐
│  1. Open https://colab.research.google.com/                 │
│                                                               │
│  2. Upload notebook:                                         │
│     → GeminiMultiModalFineTune.ipynb                        │
│                                                               │
│  3. Upload files:                                            │
│     → frames_full.zip                                        │
│     → service-account-key.json                              │
│     → train_split_full.json                                 │
│                                                               │
│  4. Unzip: !unzip frames_full.zip -d /content/              │
│                                                               │
│  5. Add Colab Secrets:                                       │
│     → PROJECT_ID                                             │
│     → REGION                                                 │
│     → BUCKET_NAME                                            │
└──────────────────────────────────────────────────────────────┘
                              ↓
                              
STEP 5: Modify Notebook (YOU DO THIS)
┌──────────────────────────────────────────────────────────────┐
│  Find Cell 13 (dataset loading)                              │
│                                                               │
│  Replace:                                                     │
│    dataset = load_butterfly_dataset()                       │
│                                                               │
│  With:                                                        │
│    # Load YOUR fall detection dataset                       │
│    import json                                               │
│    with open('/content/train_split_full.json') as f:        │
│        train_data = json.load(f)                            │
│    ...                                                        │
│    (See YOUR_TODO_LIST.md for full code)                    │
└──────────────────────────────────────────────────────────────┘
                              ↓
                              
STEP 6: Run Fine-Tuning (YOU DO THIS)
┌──────────────────────────────────────────────────────────────┐
│  Run all cells in Colab notebook:                            │
│                                                               │
│  1. Install libraries                                        │
│  2. Authenticate with Google Cloud                           │
│  3. Choose model settings:                                   │
│     → Model: gemini-2.0-flash-exp                           │
│     → Epochs: 2                                              │
│     → Learning rate: 0.001                                   │
│  4. Load dataset (your modified code)                        │
│  5. Upload to Google Cloud Storage                           │
│  6. Start fine-tuning job ⏳ (2-4 hours)                    │
│  7. Test tuned model                                         │
│  8. Evaluate results                                         │
│                                                               │
│  Output:                                                      │
│  → Fine-tuned Gemini model                                   │
│  → Job ID: projects/.../tuningJobs/...                      │
│  → Evaluation metrics (accuracy, F1, etc.)                   │
└──────────────────────────────────────────────────────────────┘
                              ↓
                              
STEP 7: Evaluate & Compare (YOU DO THIS)
┌──────────────────────────────────────────────────────────────┐
│  Compare Gemini vs Your LSTM Model:                          │
│                                                               │
│  ┌─────────────────┬──────────────┬──────────────┐          │
│  │ Metric          │ Your LSTM    │ Gemini       │          │
│  ├─────────────────┼──────────────┼──────────────┤          │
│  │ F1 Score        │ 99.42%       │ ??%          │          │
│  │ Accuracy        │ 99.38%       │ ??%          │          │
│  │ Inference Time  │ 250ms        │ ??ms         │          │
│  │ Model Size      │ 94K params   │ Billions     │          │
│  │ Cost            │ Free         │ $6-11        │          │
│  │ Deployment      │ Mobile       │ Cloud API    │          │
│  └─────────────────┴──────────────┴──────────────┘          │
│                                                               │
│  Write comparison report:                                    │
│  → Which is better for fall detection?                       │
│  → Pros/cons of each approach                                │
│  → When to use Gemini vs LSTM?                               │
└──────────────────────────────────────────────────────────────┘
                              ↓
                              
FINAL: Submit to Professor
┌──────────────────────────────────────────────────────────────┐
│  1. Modified Colab notebook                                   │
│  2. Fine-tuning job ID                                       │
│  3. Evaluation results                                       │
│  4. Comparison report                                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Differences: Your LSTM vs Gemini

### Your LSTM Model (What You Already Built)
```
Input: 30 frames of keypoints (17 joints × 3 coords × 30 frames)
       ↓
Model: BiLSTM (2 layers, 128 units, 94K params)
       ↓
Output: Fall / Non-Fall
       ↓
Training: From scratch on 24K windows
Inference: 250ms on mobile device
Cost: Free (local)
```

### Gemini Fine-Tuning (What You'll Do Now)
```
Input: Single image (512×512 RGB)
       ↓
Model: Gemini 2.0 Flash (billions of params, pre-trained)
       ↓
Fine-tuning: Transfer learning on your 1593 frames
       ↓
Output: "Yes, a person is falling" / "No, the person is not falling"
       ↓
Inference: ~1-2 seconds via Cloud API
Cost: $6-11 for fine-tuning + inference
```

---

## 📊 Dataset Breakdown

### What You Have
```
URFD Dataset:
├── Falls: 70 videos × 3 frames = 210 images
└── ADL (non-falls): 40 videos × 3 frames = 120 images

Le2i Dataset:
├── Falls: ~191 videos × 3 frames = ~573 images
└── Non-falls: ~130 videos × 3 frames = ~390 images

UCF101 Subset (non-falls):
└── 100 videos × 3 frames = 300 images

TOTAL:
├── Fall images: ~783
├── Non-fall images: ~810
└── Total: ~1593 images
```

### Train/Val/Test Split (70/15/15)
```
Training Set:   ~1115 images (70%)
Validation Set: ~239 images (15%)
Test Set:       ~239 images (15%)
```

---

## ⏱️ Time Estimate

| Step | Task | Time |
|------|------|------|
| 1 | Run dataset script | 15 min |
| 2 | Zip frames | 3 min |
| 3 | Set up Google Cloud | 30 min |
| 4 | Upload to Colab | 10 min |
| 5 | Modify notebook | 15 min |
| 6 | Run fine-tuning | **2-4 hours** ⏳ |
| 7 | Evaluate & compare | 30 min |
| **TOTAL** | | **~4-6 hours** |

---

## 💰 Cost Breakdown

| Item | Cost |
|------|------|
| Fine-tuning (2 epochs, 1593 images) | $5-10 |
| Google Cloud Storage (~300 MB) | $0.50 |
| Inference testing (~100 requests) | $0.10 |
| **TOTAL** | **~$6-11** |

✅ Your Colab credits should cover this!

---

## 🚨 Common Issues & Solutions

### Issue 1: "No videos found"
**Solution:** Check dataset paths in `prepare_urfd_le2i_dataset.py`:
```python
URFD_DIR = "data/raw/urfd"
LE2I_DIR = "data/raw/le2i"
UCF101_DIR = "data/raw/ucf101_subset"
```

### Issue 2: "Out of memory in Colab"
**Solution:** Reduce `FRAMES_PER_VIDEO` in script:
```python
FRAMES_PER_VIDEO = 2  # Instead of 3
```

### Issue 3: "Authentication failed"
**Solution:** 
1. Re-download service account key from Google Cloud
2. Re-upload to Colab
3. Run authentication cell again

### Issue 4: "Bucket not found"
**Solution:** Check Colab secrets:
- `BUCKET_NAME` should match your bucket name exactly
- No `gs://` prefix needed

### Issue 5: "Fine-tuning job failed"
**Solution:**
1. Check job logs in Google Cloud Console
2. Verify dataset format (images + JSON)
3. Check if you have enough quota

---

## 📚 Additional Resources

- **Google Colab:** https://colab.research.google.com/
- **Vertex AI Docs:** https://cloud.google.com/vertex-ai/docs
- **Gemini API Docs:** https://ai.google.dev/docs
- **Your Guides:**
  - `finetune/YOUR_TODO_LIST.md` ← **START HERE**
  - `finetune/QUICK_START.md`
  - `finetune/INSTRUCTIONS.md`
  - `finetune/README.md`

---

## 🎓 What Your Professor Wants You to Learn

1. **Transfer Learning:** Fine-tuning a pre-trained model vs training from scratch
2. **Vision vs Features:** Raw images vs hand-crafted features (keypoints)
3. **Model Comparison:** When to use large models (Gemini) vs small models (LSTM)
4. **Cloud ML:** Using cloud services (Vertex AI) vs local training
5. **Trade-offs:** Accuracy vs speed vs cost vs deployment

---

## ✅ Success Criteria

You'll know you're done when you have:

- ✅ Fine-tuned Gemini model that can detect falls from images
- ✅ Evaluation metrics (accuracy, F1 score)
- ✅ Comparison with your LSTM model
- ✅ Report explaining which approach is better and why

---

## 🚀 Ready to Start?

**Open this file and follow along:**
```bash
open finetune/YOUR_TODO_LIST.md
```

**Then run STEP 1:**
```bash
python finetune/prepare_urfd_le2i_dataset.py
```

Good luck! 🎉

