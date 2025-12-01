# 📋 Colab Cells to Copy-Paste

Copy these cells **in order** into your Google Colab notebook.

---

## 🔧 CELL 1: Upload and Unzip Frames (Add at the TOP)

```python
# Upload frames_full.zip and service-account-key.json first using the Files panel (📁)
# Then run this cell:

!unzip -q frames_full.zip
!ls frames_full/
```

**Expected output:**
```
fall  non_fall
```

---

## 🔧 CELL 2: Set Up Authentication (Add after Cell 1)

```python
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service-account-key.json'

# Verify it works
!gcloud auth activate-service-account --key-file=service-account-key.json
```

**Expected output:**
```
Activated service account credentials for: [gemini-finetuning@...]
```

---

## 🔧 CELL 3: Replace the PROJECT_ID Cell

**Find the cell that has:**
```python
PROJECT_ID = userdata.get('GOOGLE_CLOUD_PROJECT')
```

**Replace the ENTIRE cell with:**

```python
from google.colab import userdata
import datetime

# Your project details
PROJECT_ID = "948622252329"
REGION = "us-central1"
GCS_BUCKET = "fall-detection-finetuning-nikhil"

# Initialize Vertex AI
import vertexai
vertexai.init(project=PROJECT_ID, location=REGION)

# Create unique prefix for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
GCS_PREFIX = f"multimodal_sft_{timestamp}"

# Helper for building gs:// URIs
def gs_uri(*parts):
    return "gs://" + "/".join([GCS_BUCKET] + [p.strip("/") for p in parts])

# Save globally for reuse
globals()["PROJECT_ID"] = PROJECT_ID
globals()["REGION"] = REGION
globals()["GCS_BUCKET"] = GCS_BUCKET
globals()["GCS_PREFIX"] = GCS_PREFIX
globals()["gs_uri"] = gs_uri

# Print summary
print("✅ Environment initialized")
print("✅ Project:", PROJECT_ID)
print("✅ Region:", REGION)
print("✅ Bucket:", f"gs://{GCS_BUCKET}")
print("📂 Prefix:", GCS_PREFIX)
```

---

## 🔧 CELL 4: Replace the Dataset Loading Cell

**Find the cell that has:**
```python
from datasets import load_dataset
import ipywidgets as widgets
...
LABEL_TO_REPO = {
    "Butterflies": "Dasool/butterflies_and_moths_vqa",
    ...
}
```

**Replace the ENTIRE cell with:**

```python
import json
import os
from PIL import Image

# Load your fall detection dataset
print("📂 Loading fall detection dataset from local frames...")

# Read the JSON files
with open('fall_detection_dataset_full.json', 'r') as f:
    full_dataset = json.load(f)

with open('train_split_full.json', 'r') as f:
    train_data = json.load(f)

with open('val_split_full.json', 'r') as f:
    val_data = json.load(f)

with open('test_split_full.json', 'r') as f:
    test_data = json.load(f)

# Create a dataset structure compatible with the notebook
class FallDetectionDataset:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        for item in self.data:
            # Convert to the format expected by the notebook
            yield {
                'image': Image.open(item['image_path']),
                'question': item['question'],
                'answer': item['answer'],
                'label': item['label']
            }

# Create dataset splits
ds = {
    'train': FallDetectionDataset(train_data),
    'validation': FallDetectionDataset(val_data),
    'test': FallDetectionDataset(test_data)
}

# Set global variables (required by the notebook)
globals()["ds"] = ds
globals()["SELECTED_HF_REPO"] = "local/fall_detection"
globals()["SELECTED_HF_NAME"] = "fall_detection"

# Print summary
print("✅ Loaded fall detection dataset into variable `ds`.")
print(f"Total samples: {len(train_data) + len(val_data) + len(test_data)}")
print(f"  Train: {len(train_data)} samples")
print(f"  Validation: {len(val_data)} samples")
print(f"  Test: {len(test_data)} samples")
print(f"  Fall samples: {sum(1 for item in full_dataset if item['label'] == 'fall')}")
print(f"  Non-fall samples: {sum(1 for item in full_dataset if item['label'] == 'non_fall')}")
print("SELECTED_HF_NAME = fall_detection")
```

---

## ✅ Summary

You need to:

1. **Upload 2 files** to Colab (using Files panel 📁):
   - `frames_full.zip` (from `finetune/frames_full.zip`)
   - `service-account-key.json` (from Google Cloud)

2. **Add 2 new cells** at the top:
   - Cell 1: Unzip frames
   - Cell 2: Set up authentication

3. **Replace 2 existing cells**:
   - Cell 3: Replace PROJECT_ID cell
   - Cell 4: Replace dataset loading cell

4. **Upload 3 JSON files** to Colab (using Files panel 📁):
   - `fall_detection_dataset_full.json`
   - `train_split_full.json`
   - `val_split_full.json`
   - `test_split_full.json`

---

## 🎯 After Making These Changes

Run all cells in order. The notebook will:
1. ✅ Connect to your Google Cloud project
2. ✅ Load your fall detection dataset
3. ✅ Upload frames to Cloud Storage
4. ✅ Start fine-tuning Gemini
5. ✅ Evaluate the model

---

## 🆘 If You Get Errors

Common issues:
- **"File not found"** → Make sure you uploaded all files to Colab
- **"Authentication failed"** → Check your service account key is correct
- **"Bucket not found"** → Verify your bucket name is correct

Let me know if you hit any issues! 😊

