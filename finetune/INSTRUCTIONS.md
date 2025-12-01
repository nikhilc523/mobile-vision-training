# 📚 **Gemini Fine-Tuning for Fall Detection - Complete Guide**

## 🎯 **What Your Professor Wants**

Your professor wants you to:
1. Take the lab Colab notebook (`GeminiMultiModalFineTune.ipynb`)
2. **Replace the sample dataset** (Butterflies/Book Cover VQA) with **YOUR fall detection dataset**
3. Fine-tune Gemini to detect falls from video frames
4. Use Google Colab credits to run the fine-tuning

---

## 📖 **What is This Notebook?**

This is a **Gemini Multimodal Fine-Tuning** notebook that:
- Fine-tunes Google's Gemini model (vision + language)
- Uses **image + text** pairs for training
- Currently uses sample datasets: Butterflies, Book Cover VQA, Radiology, Endoscopy
- You need to **replace these with your fall detection data**

---

## 🔄 **How to Adapt for Fall Detection**

### **Current Notebook Flow:**
```
1. Install libraries
2. Authenticate with Google Cloud
3. Choose base model (gemini-2.5-flash)
4. Load dataset (Butterflies/Book Cover VQA)
5. Sample data (train/val/test split)
6. Convert to JSONL format
7. Upload to Google Cloud Storage
8. Start fine-tuning job
9. Poll job status
10. Test tuned model
11. Evaluate results
```

### **What You Need to Change:**
- **Step 4:** Replace dataset loading with YOUR fall detection videos
- **Step 5:** Create train/val/test splits from your videos
- **Step 6:** Convert fall detection data to JSONL format

---

## 📊 **Your Fall Detection Dataset**

### **What You Have:**
- **8 test videos** (finalfall.mp4, pleasefall.mp4, outdoor.mp4, etc.)
- **964 training videos** (URFD + Le2i + UCF101)
- **Labels:** Fall vs Non-fall

### **What Gemini Needs:**
Gemini expects **image + question + answer** format:

```json
{
  "image": "gs://bucket/frame_001.jpg",
  "question": "Is there a person falling in this image?",
  "answer": "Yes, a person is falling."
}
```

or

```json
{
  "image": "gs://bucket/frame_002.jpg",
  "question": "Is there a person falling in this image?",
  "answer": "No, the person is standing normally."
}
```

---

## 🛠️ **Step-by-Step Instructions**

### **STEP 1: Prepare Your Dataset** ⚠️ **DO THIS FIRST**

You need to convert your videos to **image frames + labels**.

#### **Option A: Extract Key Frames from Videos**
```python
import cv2
import os

def extract_frames(video_path, output_dir, label, num_frames=10):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract evenly spaced frames
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = f"{output_dir}/{os.path.basename(video_path)}_{idx:04d}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append({
                'image': frame_path,
                'question': 'Is there a person falling in this image?',
                'answer': 'Yes, a person is falling.' if label == 'fall' else 'No, the person is not falling.'
            })
    
    cap.release()
    return frames

# Example usage:
fall_videos = ['finalfall.mp4', 'pleasefall.mp4', 'outdoor.mp4', '2.mp4']
non_fall_videos = ['usinglap.mp4', '1.mp4']

dataset = []
for video in fall_videos:
    dataset.extend(extract_frames(f'data/test/{video}', 'finetune/frames', 'fall'))

for video in non_fall_videos:
    dataset.extend(extract_frames(f'data/test/{video}', 'finetune/frames', 'non-fall'))
```

#### **Option B: Use Existing URFD/Le2i Videos**
```python
# Extract frames from your 964 training videos
# Use the fall/non-fall labels you already have

import numpy as np

# Load your existing keypoints data
data = np.load('data/processed/all_windows_30_raw_balanced_hnm.npz')
# Extract corresponding video frames
# Create image + question + answer pairs
```

---

### **STEP 2: Create Dataset in Gemini Format**

Create a Python script to generate the dataset:

```python
import json
import pandas as pd
from PIL import Image

# Your extracted frames
frames_data = [
    {
        'image': 'finetune/frames/finalfall_0001.jpg',
        'question': 'Is there a person falling in this image?',
        'answer': 'Yes, a person is falling.'
    },
    {
        'image': 'finetune/frames/usinglap_0001.jpg',
        'question': 'Is there a person falling in this image?',
        'answer': 'No, the person is not falling.'
    },
    # ... more frames
]

# Convert to DataFrame (Gemini format)
df = pd.DataFrame(frames_data)

# Load images
df['image'] = df['image'].apply(lambda x: Image.open(x))

# Split into train/val/test
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

---

### **STEP 3: Modify the Colab Notebook**

Open `GeminiMultiModalFineTune.ipynb` in Google Colab and make these changes:

#### **Cell 13 (Load Dataset) - REPLACE THIS:**

**Original:**
```python
LABEL_TO_REPO = {
    "Butterflies": "Dasool/butterflies_and_moths_vqa",
    "Book Cover VQA (CONTAINS PEOPLE)": "howard-hou/OCR-VQA",
    "Radiology": "flaviagiammarino/vqa-rad",
    "Gastroinstestinal Endoscopy": "SimulaMet/Kvasir-VQA-x1",
}
```

**Replace with:**
```python
# Load YOUR fall detection dataset
import pandas as pd
from PIL import Image
import os

# Option 1: Load from local files (if you uploaded to Colab)
def load_fall_detection_dataset():
    frames_dir = '/content/fall_detection_frames'  # Upload your frames here
    
    data = []
    # Load fall frames
    for img_file in os.listdir(f'{frames_dir}/fall'):
        data.append({
            'image': Image.open(f'{frames_dir}/fall/{img_file}'),
            'question': 'Is there a person falling in this image?',
            'answer': 'Yes, a person is falling.'
        })
    
    # Load non-fall frames
    for img_file in os.listdir(f'{frames_dir}/non_fall'):
        data.append({
            'image': Image.open(f'{frames_dir}/non_fall/{img_file}'),
            'question': 'Is there a person falling in this image?',
            'answer': 'No, the person is not falling.'
        })
    
    return pd.DataFrame(data)

# Load dataset
ds = load_fall_detection_dataset()
print(f"Loaded {len(ds)} samples")
SELECTED_HF_NAME = "fall_detection_custom"
```

---

### **STEP 4: Upload Frames to Google Colab**

Before running the notebook, you need to upload your frames:

1. **Extract frames locally** (use the script from STEP 1)
2. **Zip the frames folder:**
   ```bash
   cd finetune
   zip -r fall_detection_frames.zip frames/
   ```
3. **Upload to Colab:**
   - In Colab, click the folder icon (left sidebar)
   - Click "Upload" button
   - Upload `fall_detection_frames.zip`
   - Unzip in Colab:
     ```python
     !unzip fall_detection_frames.zip -d /content/
     ```

---

### **STEP 5: Set Up Google Cloud**

You need Google Cloud credentials to run fine-tuning.

#### **A. Create Google Cloud Project**
1. Go to https://console.cloud.google.com/
2. Create a new project (or use existing)
3. Note your **Project ID** (e.g., `my-fall-detection-project`)

#### **B. Enable APIs**
1. Go to "APIs & Services" → "Enable APIs and Services"
2. Enable:
   - **Vertex AI API**
   - **Cloud Storage API**

#### **C. Create Service Account**
1. Go to "IAM & Admin" → "Service Accounts"
2. Click "Create Service Account"
3. Name: `gemini-finetuning`
4. Grant roles:
   - **Vertex AI User**
   - **Storage Admin**
5. Click "Create Key" → JSON
6. Download the JSON key file

#### **D. Create Cloud Storage Bucket**
1. Go to "Cloud Storage" → "Buckets"
2. Click "Create Bucket"
3. Name: `fall-detection-finetuning` (must be globally unique)
4. Region: `us-central1` (recommended)
5. Click "Create"

---

### **STEP 6: Add Credentials to Colab**

In Google Colab:

1. Click the **key icon** (left sidebar) → "Secrets"
2. Add these secrets:
   - **GOOGLE_CLOUD_PROJECT**: Your project ID (e.g., `my-fall-detection-project`)
   - **GOOGLE_DEFAULT_REGION**: `us-central1`
   - **GOOGLE_DEFAULT_BUCKET**: `fall-detection-finetuning`
3. Upload your service account JSON:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload your JSON key file
   
   # Set environment variable
   import os
   os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = list(uploaded.keys())[0]
   ```

---

### **STEP 7: Run the Notebook**

Now run the cells in order:

1. **Cell 5:** Install libraries ✅
2. **Cell 7:** Authenticate (uses your credentials) ✅
3. **Cell 9:** Choose model settings:
   - Base model: `gemini-2.5-flash` (recommended)
   - Epochs: `2-5` (start with 2)
   - Adapter: `Small (4)` (good balance)
   - LR Multiplier: `1.0`
4. **Cell 13:** Load YOUR dataset (modified) ✅
5. **Cell 15:** Sample data (train/val/test split) ✅
6. **Cell 21:** Convert to JSONL and upload to GCS ✅
7. **Cell 24:** Start fine-tuning job ✅
8. **Cell 30:** Poll job status (wait for completion) ⏳
9. **Cell 36:** Test tuned model ✅
10. **Cell 47:** Evaluate results ✅

---

## ⏱️ **Expected Timeline**

- **Data preparation:** 2-3 hours
- **Fine-tuning job:** 1-4 hours (depends on dataset size)
- **Evaluation:** 30 minutes

**Total:** ~4-8 hours

---

## 💰 **Cost Estimate**

With Google Colab credits:
- **Fine-tuning:** ~$5-20 (depends on dataset size and epochs)
- **Storage:** ~$0.50/month
- **Inference:** ~$0.10 per 1000 requests

**Your Colab credits should cover this!**

---

## ✅ **Checklist**

Before starting, make sure you have:

- [ ] Extracted frames from your fall detection videos
- [ ] Created train/val/test splits
- [ ] Set up Google Cloud project
- [ ] Enabled Vertex AI and Cloud Storage APIs
- [ ] Created service account and downloaded JSON key
- [ ] Created Cloud Storage bucket
- [ ] Added credentials to Colab secrets
- [ ] Modified Cell 13 in the notebook
- [ ] Uploaded frames to Colab

---

## 🚨 **Common Issues & Solutions**

### **Issue 1: "Dataset not found"**
**Solution:** Make sure you uploaded frames to Colab and modified Cell 13 correctly.

### **Issue 2: "Authentication failed"**
**Solution:** Check that you uploaded the service account JSON and set `GOOGLE_APPLICATION_CREDENTIALS`.

### **Issue 3: "Bucket not found"**
**Solution:** Make sure the bucket name in Colab secrets matches the actual bucket name.

### **Issue 4: "Out of memory"**
**Solution:** Reduce the number of frames or use smaller images (resize to 512x512).

---

## 📝 **Next Steps After Fine-Tuning**

Once fine-tuning is complete:

1. **Test the model** on your test videos
2. **Compare with your LSTM model** (which one is better?)
3. **Write a report** comparing:
   - Gemini fine-tuned model
   - Your LSTM model
   - Accuracy, speed, cost
4. **Submit to your professor**

---

## 🎓 **What to Submit**

Your professor likely wants:

1. **Modified Colab notebook** (with your dataset)
2. **Fine-tuning results** (accuracy, loss curves)
3. **Comparison report** (Gemini vs LSTM)
4. **Test predictions** (on your test videos)

---

## 💡 **Pro Tips**

1. **Start small:** Use 50-100 frames first to test the pipeline
2. **Balance classes:** Equal number of fall and non-fall frames
3. **Use diverse frames:** Different angles, lighting, people
4. **Monitor costs:** Check Google Cloud billing dashboard
5. **Save checkpoints:** Download the fine-tuned model

---

## 📞 **Need Help?**

If you get stuck, check:
1. Google Colab documentation: https://colab.research.google.com/
2. Vertex AI documentation: https://cloud.google.com/vertex-ai/docs
3. Gemini fine-tuning guide: https://ai.google.dev/gemini-api/docs/model-tuning

---

**Good luck with your fine-tuning!** 🚀

