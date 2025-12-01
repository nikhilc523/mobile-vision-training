# 🚀 **Quick Start Guide - Gemini Fine-Tuning for Fall Detection**

## ⚡ **TL;DR - What to Do**

1. **Prepare dataset** (30 min)
2. **Set up Google Cloud** (30 min)
3. **Upload to Colab** (10 min)
4. **Modify notebook** (15 min)
5. **Run fine-tuning** (2-4 hours)
6. **Evaluate results** (30 min)

**Total time:** ~4-6 hours

---

## 📋 **Step-by-Step Checklist**

### **STEP 1: Prepare Your Dataset** (30 minutes)

```bash
# Run the preparation script
cd /Users/nikhilchowdary/mobile-vision-training
python finetune/prepare_dataset.py
```

**This will:**
- ✅ Extract frames from your 8 test videos
- ✅ Create train/val/test splits
- ✅ Generate `fall_detection_frames.zip` for Colab upload
- ✅ Save dataset files (train_split.json, val_split.json, test_split.json)

**Output:**
- `finetune/frames/` - Extracted frames (fall/ and non_fall/ folders)
- `finetune/fall_detection_frames.zip` - Ready for Colab upload
- `finetune/train_split.json` - Training data
- `finetune/val_split.json` - Validation data
- `finetune/test_split.json` - Test data

---

### **STEP 2: Set Up Google Cloud** (30 minutes)

#### **A. Create Project**
1. Go to https://console.cloud.google.com/
2. Click "Select a project" → "New Project"
3. Name: `fall-detection-gemini`
4. Click "Create"
5. **Copy your Project ID** (e.g., `fall-detection-gemini-123456`)

#### **B. Enable APIs**
1. Go to "APIs & Services" → "Library"
2. Search and enable:
   - ✅ **Vertex AI API**
   - ✅ **Cloud Storage API**

#### **C. Create Service Account**
1. Go to "IAM & Admin" → "Service Accounts"
2. Click "Create Service Account"
3. Name: `gemini-finetuning`
4. Click "Create and Continue"
5. Add roles:
   - ✅ **Vertex AI User**
   - ✅ **Storage Admin**
6. Click "Continue" → "Done"
7. Click on the service account
8. Go to "Keys" tab
9. Click "Add Key" → "Create new key" → "JSON"
10. **Download the JSON file** (save as `service-account-key.json`)

#### **D. Create Storage Bucket**
1. Go to "Cloud Storage" → "Buckets"
2. Click "Create"
3. Name: `fall-detection-finetuning-<your-name>` (must be globally unique)
4. Region: **us-central1**
5. Storage class: **Standard**
6. Click "Create"

---

### **STEP 3: Upload to Google Colab** (10 minutes)

1. **Open Colab:**
   - Go to https://colab.research.google.com/
   - Click "File" → "Upload notebook"
   - Upload `finetune/GeminiMultiModalFineTune.ipynb`

2. **Upload frames:**
   - Click folder icon (left sidebar)
   - Click "Upload" button
   - Upload `finetune/fall_detection_frames.zip`
   - Wait for upload to complete

3. **Unzip frames:**
   ```python
   !unzip fall_detection_frames.zip -d /content/
   ```

4. **Upload service account key:**
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload service-account-key.json
   
   import os
   os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service-account-key.json'
   ```

5. **Add Colab secrets:**
   - Click key icon (left sidebar) → "Secrets"
   - Add:
     - `GOOGLE_CLOUD_PROJECT` = `your-project-id`
     - `GOOGLE_DEFAULT_REGION` = `us-central1`
     - `GOOGLE_DEFAULT_BUCKET` = `your-bucket-name`

---

### **STEP 4: Modify the Notebook** (15 minutes)

#### **Find Cell 13 (Load Dataset)**

**Original code:**
```python
LABEL_TO_REPO = {
    "Butterflies": "Dasool/butterflies_and_moths_vqa",
    ...
}
```

**Replace with:**
```python
# Load Fall Detection Dataset
import pandas as pd
from PIL import Image
import os

def load_fall_detection_dataset():
    """Load fall detection frames from uploaded zip"""
    frames_dir = '/content'
    
    data = []
    
    # Load fall frames
    fall_dir = os.path.join(frames_dir, 'fall')
    if os.path.exists(fall_dir):
        for img_file in os.listdir(fall_dir):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(fall_dir, img_file)
                data.append({
                    'image': Image.open(img_path),
                    'question': 'Is there a person falling in this image?',
                    'answer': 'Yes, a person is falling.'
                })
    
    # Load non-fall frames
    non_fall_dir = os.path.join(frames_dir, 'non_fall')
    if os.path.exists(non_fall_dir):
        for img_file in os.listdir(non_fall_dir):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(non_fall_dir, img_file)
                data.append({
                    'image': Image.open(img_path),
                    'question': 'Is there a person falling in this image?',
                    'answer': 'No, the person is not falling.'
                })
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} samples")
    print(f"   Fall: {len(df[df['answer'].str.contains('Yes')])}")
    print(f"   Non-fall: {len(df[df['answer'].str.contains('No')])}")
    
    return df

# Load dataset
ds = load_fall_detection_dataset()
SELECTED_HF_NAME = "fall_detection_custom"
```

---

### **STEP 5: Run the Notebook** (2-4 hours)

Run cells in order:

1. **Cell 5:** Install libraries ✅ (2 min)
2. **Cell 7:** Authenticate ✅ (1 min)
3. **Cell 9:** Choose settings:
   - Base model: `gemini-2.5-flash`
   - Epochs: `2` (start small)
   - Adapter: `Small (4)`
   - LR Multiplier: `1.0`
   - Click "Apply settings" ✅
4. **Cell 13:** Load dataset (your modified code) ✅ (1 min)
5. **Cell 15:** Sample data:
   - Shuffle: `Yes`
   - Click "Apply" ✅
6. **Cell 17:** Preview sample ✅
7. **Cell 21:** Build JSONL and upload to GCS ✅ (5 min)
8. **Cell 24:** Start fine-tuning job ✅ (1 min)
   - **Note the job ID!** (e.g., `4726005764240441344`)
9. **Cell 30:** Poll job status ⏳ (2-4 hours)
   - Wait for "JOB_STATE_SUCCEEDED"
10. **Cell 36:** Test tuned model ✅ (5 min)
11. **Cell 47:** Evaluate results ✅ (10 min)

---

### **STEP 6: Evaluate Results** (30 minutes)

After fine-tuning completes:

1. **Test on sample images:**
   ```python
   # Cell 36 will show predictions
   # Compare tuned model vs base model
   ```

2. **Check accuracy:**
   ```python
   # Cell 47 will show evaluation metrics
   # Look for:
   # - Accuracy
   # - Precision
   # - Recall
   # - F1 Score
   ```

3. **Compare with your LSTM model:**
   - Gemini F1 score: ?
   - LSTM F1 score: 99.42%
   - Which is better?

---

## 📊 **Expected Results**

### **Dataset Size:**
- ~80 frames total (8 videos × 10 frames)
- ~40 fall frames
- ~40 non-fall frames

### **Training Time:**
- 2 epochs: ~1-2 hours
- 5 epochs: ~2-4 hours

### **Cost:**
- Fine-tuning: ~$5-10
- Storage: ~$0.50
- Inference: ~$0.10 per 1000 requests

**Your Colab credits should cover this!**

---

## 🎯 **What to Submit to Professor**

1. **Modified Colab notebook** (with your dataset code)
2. **Fine-tuning job ID** (from Cell 24)
3. **Evaluation results** (from Cell 47)
4. **Comparison report:**
   - Gemini accuracy vs LSTM accuracy
   - Pros/cons of each approach
   - Which is better for fall detection?

---

## 🚨 **Troubleshooting**

### **"Dataset not found"**
- Check that you unzipped frames correctly
- Verify paths in Cell 13

### **"Authentication failed"**
- Re-upload service account JSON
- Check Colab secrets are set correctly

### **"Bucket not found"**
- Verify bucket name matches exactly
- Check bucket region is `us-central1`

### **"Out of memory"**
- Reduce number of frames (use 5 per video instead of 10)
- Resize images to 512x512

---

## 💡 **Pro Tips**

1. **Start small:** Test with 2 videos first (1 fall, 1 non-fall)
2. **Monitor costs:** Check Google Cloud billing dashboard
3. **Save job ID:** You'll need it to retrieve the model later
4. **Take screenshots:** Document your results for the report
5. **Compare models:** Test both Gemini and LSTM on same videos

---

## 📞 **Need Help?**

Check these resources:
- `INSTRUCTIONS.md` - Detailed guide
- `prepare_dataset.py` - Dataset preparation script
- Google Colab docs: https://colab.research.google.com/
- Vertex AI docs: https://cloud.google.com/vertex-ai/docs

---

## ✅ **Final Checklist**

Before submitting:

- [ ] Dataset prepared and uploaded
- [ ] Google Cloud project set up
- [ ] Notebook modified (Cell 13)
- [ ] Fine-tuning job completed
- [ ] Results evaluated
- [ ] Comparison report written
- [ ] Screenshots taken
- [ ] Notebook downloaded from Colab

---

**Good luck!** 🚀

**Estimated total time:** 4-6 hours
**Estimated cost:** $5-15 (covered by Colab credits)

