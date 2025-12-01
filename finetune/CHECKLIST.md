# ✅ Gemini Fine-Tuning Checklist

Print this and check off items as you complete them!

---

## 📋 Pre-Flight Checklist

- [ ] Python environment ready (OpenCV, pandas, tqdm installed)
- [ ] URFD dataset at `data/raw/urfd/`
- [ ] Le2i dataset at `data/raw/le2i/`
- [ ] UCF101 dataset at `data/raw/ucf101_subset/`
- [ ] Google account ready
- [ ] Credit card ready (for Google Cloud)
- [ ] 4-6 hours of time available

---

## 🚀 STEP 1: Prepare Dataset (15 min)

- [ ] Open terminal
- [ ] Run: `python finetune/prepare_urfd_le2i_dataset.py`
- [ ] Wait for completion (~15 minutes)
- [ ] Verify output:
  - [ ] `finetune/frames_full/` folder created
  - [ ] `finetune/fall_detection_dataset_full.json` created
  - [ ] `finetune/train_split_full.json` created
  - [ ] `finetune/val_split_full.json` created
  - [ ] `finetune/test_split_full.json` created
- [ ] Check console output:
  - [ ] Total videos: ~531
  - [ ] Total frames: ~1593
  - [ ] Fall samples: ~783
  - [ ] Non-fall samples: ~810

**✅ Tell AI: "done step 1"**

---

## 📦 STEP 2: Zip Frames (3 min)

- [ ] Open terminal
- [ ] Run: `cd finetune`
- [ ] Run: `zip -r frames_full.zip frames_full/`
- [ ] Wait for completion (~3 minutes)
- [ ] Verify: `frames_full.zip` created (~200-300 MB)
- [ ] Run: `cd ..` (go back to root)

**✅ Tell AI: "done step 2"**

---

## ☁️ STEP 3: Set Up Google Cloud (30 min)

### 3.1: Create Project
- [ ] Go to: https://console.cloud.google.com/
- [ ] Click "Create Project"
- [ ] Name: `fall-detection-finetuning`
- [ ] Click "Create"
- [ ] Wait for project creation
- [ ] Copy Project ID: `_______________________`

### 3.2: Enable APIs
- [ ] Go to: https://console.cloud.google.com/apis/library
- [ ] Search: "Vertex AI API"
- [ ] Click "Enable"
- [ ] Wait for activation
- [ ] Go back to API Library
- [ ] Search: "Cloud Storage API"
- [ ] Click "Enable"
- [ ] Wait for activation

### 3.3: Create Service Account
- [ ] Go to: https://console.cloud.google.com/iam-admin/serviceaccounts
- [ ] Click "Create Service Account"
- [ ] Name: `gemini-finetuning`
- [ ] Click "Create and Continue"
- [ ] Add Role: "Vertex AI Administrator"
- [ ] Add Role: "Storage Admin"
- [ ] Click "Done"
- [ ] Click on the service account
- [ ] Go to "Keys" tab
- [ ] Click "Add Key" → "Create new key"
- [ ] Choose "JSON"
- [ ] Click "Create"
- [ ] Save file as: `service-account-key.json`
- [ ] Move to safe location

### 3.4: Create Storage Bucket
- [ ] Go to: https://console.cloud.google.com/storage
- [ ] Click "Create Bucket"
- [ ] Name: `fall-detection-finetuning-<yourname>`
  - [ ] My bucket name: `_______________________`
- [ ] Region: `us-central1`
- [ ] Storage class: Standard
- [ ] Access control: Uniform
- [ ] Click "Create"

**✅ Tell AI: "done step 3"**

---

## 📤 STEP 4: Upload to Colab (10 min)

### 4.1: Open Colab
- [ ] Go to: https://colab.research.google.com/
- [ ] Click "File" → "Upload notebook"
- [ ] Upload: `finetune/GeminiMultiModalFineTune.ipynb`
- [ ] Wait for upload

### 4.2: Upload Files
- [ ] Click folder icon (left sidebar)
- [ ] Upload: `finetune/frames_full.zip`
- [ ] Wait for upload (~5 minutes)
- [ ] Upload: `service-account-key.json`
- [ ] Upload: `finetune/train_split_full.json`

### 4.3: Unzip Frames
- [ ] Create new cell in Colab
- [ ] Run: `!unzip frames_full.zip -d /content/`
- [ ] Wait for completion (~2 minutes)
- [ ] Verify: `/content/frames_full/` folder exists

### 4.4: Add Secrets
- [ ] Click key icon (left sidebar)
- [ ] Click "Add new secret"
- [ ] Name: `PROJECT_ID`
- [ ] Value: (your project ID from Step 3.1)
- [ ] Click "Add new secret"
- [ ] Name: `REGION`
- [ ] Value: `us-central1`
- [ ] Click "Add new secret"
- [ ] Name: `BUCKET_NAME`
- [ ] Value: (your bucket name from Step 3.4)

**✅ Tell AI: "done step 4"**

---

## ✏️ STEP 5: Modify Notebook (15 min)

- [ ] Find Cell 13 in the notebook
- [ ] Look for: `dataset = load_butterfly_dataset()`
- [ ] Delete that line
- [ ] Copy code from `YOUR_TODO_LIST.md` (STEP 5)
- [ ] Paste into Cell 13
- [ ] Verify code looks correct
- [ ] Run Cell 13 to test
- [ ] Check output:
  - [ ] "✅ Loaded 1115 training samples" (or similar)
  - [ ] Fall samples: ~550
  - [ ] Non-fall samples: ~565

**✅ Tell AI: "done step 5"**

---

## 🏃 STEP 6: Run Fine-Tuning (2-4 hours)

### 6.1: Run Setup Cells
- [ ] Run Cell 1: Install libraries
- [ ] Run Cell 2: Import libraries
- [ ] Run Cell 3: Authenticate with Google Cloud
- [ ] Run Cell 4: Set project ID
- [ ] Run Cell 5: Initialize Vertex AI

### 6.2: Choose Model Settings
- [ ] Run Cell 6: Choose model
  - [ ] Model: `gemini-2.0-flash-exp`
- [ ] Run Cell 7: Set hyperparameters
  - [ ] Epochs: 2
  - [ ] Learning rate: 0.001
  - [ ] Batch size: 8

### 6.3: Load Dataset
- [ ] Run Cell 13: Load dataset (your modified code)
- [ ] Verify output shows correct number of samples

### 6.4: Upload to GCS
- [ ] Run Cell 14: Upload images to Google Cloud Storage
- [ ] Wait for upload (~10 minutes)
- [ ] Verify: Images uploaded successfully

### 6.5: Start Fine-Tuning
- [ ] Run Cell 15: Start fine-tuning job
- [ ] Copy Job ID: `_______________________`
- [ ] Wait for completion (2-4 hours)
- [ ] Check progress periodically

### 6.6: Test Model
- [ ] Run Cell 16: Load fine-tuned model
- [ ] Run Cell 17: Test on sample images
- [ ] Run Cell 18: Evaluate on test set

**✅ Tell AI: "done step 6"**

---

## 📊 STEP 7: Evaluate & Compare (30 min)

### 7.1: Calculate Metrics
- [ ] Run evaluation cells
- [ ] Record metrics:
  - [ ] Accuracy: `_______%`
  - [ ] Precision: `_______%`
  - [ ] Recall: `_______%`
  - [ ] F1 Score: `_______%`

### 7.2: Compare with LSTM
- [ ] Your LSTM F1 Score: 99.42%
- [ ] Gemini F1 Score: `_______%`
- [ ] Which is better? `_______`

### 7.3: Write Comparison Report
- [ ] Pros of Gemini:
  - [ ] _______________________
  - [ ] _______________________
  - [ ] _______________________
- [ ] Cons of Gemini:
  - [ ] _______________________
  - [ ] _______________________
  - [ ] _______________________
- [ ] Pros of LSTM:
  - [ ] _______________________
  - [ ] _______________________
  - [ ] _______________________
- [ ] Cons of LSTM:
  - [ ] _______________________
  - [ ] _______________________
  - [ ] _______________________
- [ ] When to use Gemini? `_______________________`
- [ ] When to use LSTM? `_______________________`

**✅ Tell AI: "done step 7"**

---

## 📝 FINAL: Submit to Professor

- [ ] Download modified Colab notebook
- [ ] Copy fine-tuning job ID
- [ ] Copy evaluation results
- [ ] Write comparison report (1-2 pages)
- [ ] Submit all materials to professor

---

## 💰 Cost Tracking

- [ ] Fine-tuning cost: $`_______`
- [ ] Storage cost: $`_______`
- [ ] Inference cost: $`_______`
- [ ] Total cost: $`_______`
- [ ] Within budget? (< $15) `_______`

---

## 🆘 Troubleshooting

If you encounter issues, check:

- [ ] Dataset paths correct in script?
- [ ] All files uploaded to Colab?
- [ ] Service account key valid?
- [ ] Colab secrets set correctly?
- [ ] Enough Google Cloud quota?
- [ ] Checked error logs?

---

## ✅ Final Checklist

Before submitting, verify:

- [ ] Fine-tuning completed successfully
- [ ] Model tested on sample images
- [ ] Evaluation metrics calculated
- [ ] Comparison with LSTM done
- [ ] Report written
- [ ] All materials ready to submit

---

## 🎉 Congratulations!

You've completed the Gemini fine-tuning assignment!

**What you learned:**
- ✅ Transfer learning vs training from scratch
- ✅ Vision models vs feature-based models
- ✅ Cloud ML vs local training
- ✅ Model comparison and evaluation

**What you built:**
- ✅ Fine-tuned Gemini model for fall detection
- ✅ Comprehensive evaluation and comparison
- ✅ Understanding of different ML approaches

Great job! 🚀

