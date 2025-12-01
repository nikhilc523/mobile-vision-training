# 🎯 YOUR TODO LIST - Gemini Fine-Tuning Assignment

## ✅ What I (AI) Already Did For You

I've prepared everything I can do automatically:

1. ✅ **Created the dataset preparation script** (`prepare_urfd_le2i_dataset.py`)
   - Extracts frames from URFD + Le2i + UCF101 videos
   - Creates train/val/test splits
   - Generates JSON files for Gemini

2. ✅ **Created comprehensive guides**:
   - `README.md` - Overview of the assignment
   - `INSTRUCTIONS.md` - Detailed step-by-step guide
   - `QUICK_START.md` - Fast-track checklist
   - `YOUR_TODO_LIST.md` - This file (what YOU need to do)

3. ✅ **Verified your dataset structure**:
   - URFD: ✅ Found (falls + adl folders)
   - Le2i: ✅ Found (multiple room folders)
   - UCF101: ✅ Found (7 activity categories)

---

## 🚀 WHAT YOU NEED TO DO (Step-by-Step)

### **STEP 1: Run the Dataset Preparation Script** ⏱️ 10-15 minutes

**What to do:**
```bash
python finetune/prepare_urfd_le2i_dataset.py
```

**What this does:**
- Extracts 3 frames from each video
- Creates `finetune/frames_full/` folder with all frames
- Creates JSON files: `fall_detection_dataset_full.json`, `train_split_full.json`, `val_split_full.json`, `test_split_full.json`

**Expected output:**
```
🚀 Starting dataset preparation for URFD + Le2i + UCF101...
📂 Finding videos...
  URFD: 110 videos
  Le2i: 321 videos
  UCF101: 100 videos
  Total: 531 videos
  
  Fall videos: 261
  Non-fall videos: 270

🎬 Extracting frames from videos...
Processing videos: 100%|████████████| 531/531 [05:23<00:00,  1.64it/s]

✅ DATASET CREATED SUCCESSFULLY!
Total samples: 1593
Fall samples: 783
Non-fall samples: 810
```

**When done, tell me:** "done step 1"

---

### **STEP 2: Zip the Frames Folder** ⏱️ 2-3 minutes

**What to do:**
```bash
cd finetune
zip -r frames_full.zip frames_full/
cd ..
```

**What this does:**
- Creates `finetune/frames_full.zip` (you'll upload this to Colab)

**Expected output:**
```
  adding: frames_full/ (stored 0%)
  adding: frames_full/fall/ (stored 0%)
  adding: frames_full/fall/fall-01-cam0-rgb_frame_0000.jpg (deflated 5%)
  ...
  (1593 files total)
```

**When done, tell me:** "done step 2"

---

### **STEP 3: Set Up Google Cloud** ⏱️ 30 minutes

**What to do:**

1. **Create Google Cloud Project**
   - Go to: https://console.cloud.google.com/
   - Click "Create Project"
   - Name: `fall-detection-finetuning`
   - Click "Create"

2. **Enable APIs**
   - Go to: https://console.cloud.google.com/apis/library
   - Search for "Vertex AI API" → Click → Enable
   - Search for "Cloud Storage API" → Click → Enable

3. **Create Service Account**
   - Go to: https://console.cloud.google.com/iam-admin/serviceaccounts
   - Click "Create Service Account"
   - Name: `gemini-finetuning`
   - Role: "Vertex AI Administrator" + "Storage Admin"
   - Click "Done"
   - Click on the service account → "Keys" tab → "Add Key" → "Create new key" → JSON
   - Download the JSON file (save as `service-account-key.json`)

4. **Create Storage Bucket**
   - Go to: https://console.cloud.google.com/storage
   - Click "Create Bucket"
   - Name: `fall-detection-finetuning-<yourname>` (must be globally unique)
   - Region: `us-central1`
   - Click "Create"

**When done, tell me:** "done step 3"

---

### **STEP 4: Upload to Google Colab** ⏱️ 10 minutes

**What to do:**

1. **Open Colab**
   - Go to: https://colab.research.google.com/
   - Click "File" → "Upload notebook"
   - Upload `finetune/GeminiMultiModalFineTune.ipynb`

2. **Upload Files to Colab**
   - In Colab, click the folder icon (left sidebar)
   - Upload these files:
     - `finetune/frames_full.zip`
     - `service-account-key.json` (from Step 3)
     - `finetune/train_split_full.json`

3. **Unzip Frames**
   - In Colab, run this cell:
   ```python
   !unzip frames_full.zip -d /content/
   ```

4. **Add Colab Secrets**
   - Click the key icon (left sidebar) → "Secrets"
   - Add these secrets:
     - `PROJECT_ID`: Your Google Cloud project ID (e.g., `fall-detection-finetuning`)
     - `REGION`: `us-central1`
     - `BUCKET_NAME`: Your bucket name (e.g., `fall-detection-finetuning-yourname`)

**When done, tell me:** "done step 4"

---

### **STEP 5: Modify the Notebook** ⏱️ 15 minutes

**What to do:**

Find **Cell 13** in the notebook (the one that loads the dataset).

**Original code:**
```python
# Load dataset
dataset = load_butterfly_dataset()  # or load_book_cover_dataset()
```

**Replace with:**
```python
# Load YOUR fall detection dataset
import json
import pandas as pd

# Load the train split
with open('/content/train_split_full.json', 'r') as f:
    train_data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(train_data)

# Create dataset in Gemini format
dataset = []
for _, row in df.iterrows():
    dataset.append({
        'image_path': row['image_path'].replace('finetune/', '/content/'),
        'prompt': row['question'],
        'response': row['answer']
    })

print(f"✅ Loaded {len(dataset)} training samples")
print(f"   Fall samples: {len(df[df['label'] == 'fall'])}")
print(f"   Non-fall samples: {len(df[df['label'] == 'non_fall'])}")
```

**When done, tell me:** "done step 5"

---

### **STEP 6: Run Fine-Tuning** ⏱️ 2-4 hours

**What to do:**

1. **Run all cells in order** (from top to bottom)
   - Cell 1-5: Install libraries, authenticate
   - Cell 6-10: Choose model settings
     - Model: `gemini-2.0-flash-exp`
     - Epochs: 2
     - Learning rate: 0.001
   - Cell 11-15: Load dataset (your modified code)
   - Cell 16-20: Upload to Google Cloud Storage
   - Cell 21-25: Start fine-tuning job ⏳ (2-4 hours)
   - Cell 26-30: Test tuned model
   - Cell 31-35: Evaluate results

2. **Wait for fine-tuning to complete**
   - You'll see: "Fine-tuning job started: projects/.../locations/.../tuningJobs/..."
   - Copy this job ID (you'll need it for your report)
   - The notebook will show progress updates

3. **Test the model**
   - Once done, run the evaluation cells
   - Test on some fall/non-fall images
   - Compare with base Gemini model

**When done, tell me:** "done step 6"

---

### **STEP 7: Evaluate & Compare** ⏱️ 30 minutes

**What to do:**

1. **Calculate metrics**
   - Accuracy, Precision, Recall, F1 Score
   - Compare with your LSTM model (99.42% F1)

2. **Create comparison report**
   - Which model is better?
   - Pros/cons of each approach
   - When to use Gemini vs LSTM?

**When done, tell me:** "done step 7"

---

## 📝 What to Submit to Your Professor

1. ✅ Modified Colab notebook (with your dataset code)
2. ✅ Fine-tuning job ID
3. ✅ Evaluation results (accuracy, F1 score)
4. ✅ Comparison report:
   - Gemini vs LSTM
   - Pros/cons
   - Which is better for fall detection?

---

## 💰 Cost Estimate

- Fine-tuning: ~$5-10
- Storage: ~$0.50
- Inference: ~$0.10 per 1000 requests
- **Total: ~$6-11** (covered by your Colab credits)

---

## 🆘 If You Get Stuck

1. **Check the detailed guides:**
   - `finetune/INSTRUCTIONS.md` (comprehensive guide)
   - `finetune/QUICK_START.md` (fast-track guide)

2. **Common issues:**
   - "No videos found" → Check dataset paths in script
   - "Out of memory" → Reduce `FRAMES_PER_VIDEO` in script
   - "Authentication failed" → Re-upload service account key
   - "Bucket not found" → Check bucket name in Colab secrets

3. **Ask me for help!** Tell me:
   - Which step you're on
   - What error you're seeing
   - What you've tried

---

## 🎯 Summary

**What I did:** ✅ Created scripts and guides
**What YOU do:** 
1. ⏳ Run dataset script (15 min)
2. ⏳ Zip frames (3 min)
3. ⏳ Set up Google Cloud (30 min)
4. ⏳ Upload to Colab (10 min)
5. ⏳ Modify notebook (15 min)
6. ⏳ Run fine-tuning (2-4 hours)
7. ⏳ Evaluate & compare (30 min)

**Total time:** ~4-6 hours
**Total cost:** ~$6-11

---

## 🚀 Ready to Start?

**Start with STEP 1:**
```bash
python finetune/prepare_urfd_le2i_dataset.py
```

Then tell me "done step 1" and I'll guide you to the next step! 😊

