# 🚀 Step-by-Step: Colab Setup with Google Drive

Follow these steps **exactly** in order.

---

## 📁 STEP 1: Upload Files to Google Drive

1. **Open Google Drive** in your browser: https://drive.google.com
2. **Create a new folder** called `tuning` in "My Drive"
3. **Upload these 6 files** to the `tuning` folder:
   - `frames_full.zip` (~100-200 MB)
   - `fall_detection_dataset_full.json`
   - `train_split_full.json`
   - `val_split_full.json`
   - `test_split_full.json`
   - `service-account-key.json`

4. **Wait for all uploads to complete!**

**✅ Your Google Drive should look like:**
```
Google Drive/
└── My Drive/
    └── tuning/
        ├── frames_full.zip
        ├── fall_detection_dataset_full.json
        ├── train_split_full.json
        ├── val_split_full.json
        ├── test_split_full.json
        └── service-account-key.json
```

---

## 🌐 STEP 2: Open Notebook in Colab

1. Go to: https://colab.research.google.com/
2. Click: **File** → **Upload notebook**
3. Select: `finetune/GeminiMultiModalFineTune.ipynb` from your computer
4. Wait for it to open

---

## ➕ STEP 3: Add New Cell #1 (Load Files from Google Drive)

1. **Click at the very top** of the notebook (above all cells)
2. **Click "+ Code"** to add a new cell
3. **Copy-paste this code:**

```python
# Mount Google Drive and load all files
from google.colab import drive
import shutil
import os

# Mount Google Drive
drive.mount('/content/drive')

# Set the path to your tuning folder
DRIVE_FOLDER = '/content/drive/MyDrive/tuning'

print("📂 Copying files from Google Drive...")

# Copy frames zip
frames_zip_path = f'{DRIVE_FOLDER}/frames_full.zip'
if os.path.exists(frames_zip_path):
    shutil.copy(frames_zip_path, '/content/frames_full.zip')
    print("✅ Copied frames_full.zip")

# Copy JSON files
json_files = [
    'fall_detection_dataset_full.json',
    'train_split_full.json',
    'val_split_full.json',
    'test_split_full.json'
]

for filename in json_files:
    src = f'{DRIVE_FOLDER}/{filename}'
    dst = f'/content/{filename}'
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"✅ Copied {filename}")

# Copy service account key (check for space in filename)
for key_name in ['service-account-key.json', ' service-account-key.json']:
    service_key_src = f'{DRIVE_FOLDER}/{key_name}'
    if os.path.exists(service_key_src):
        service_key_dst = '/content/service-account-key.json'
        shutil.copy(service_key_src, service_key_dst)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_key_dst
        print(f"✅ Copied service-account-key.json")
        print(f"   → Set GOOGLE_APPLICATION_CREDENTIALS")
        break

# Unzip frames
print("\n📦 Unzipping frames...")
!rm -rf /content/frames_full
!unzip -q /content/frames_full.zip -d /content/

# Verify
print("\n✅ Checking unzipped frames...")
if os.path.exists('/content/frames_full'):
    items = os.listdir('/content/frames_full')
    print(f"✅ frames_full/ exists with {len(items)} folders")
    for item in items:
        path = f'/content/frames_full/{item}'
        if os.path.isdir(path):
            count = len(os.listdir(path))
            print(f"   📁 {item}/ ({count} files)")

print("\n✅ All files loaded from Google Drive!")
```

4. **Run the cell** (click the ▶️ button or press Shift+Enter)
5. **Authorize Google Drive access** when prompted
6. **Expected output:**
```
Mounted at /content/drive
📂 Copying files from Google Drive...
✅ Copied frames_full.zip
✅ Copied fall_detection_dataset_full.json
✅ Copied train_split_full.json
✅ Copied val_split_full.json
✅ Copied test_split_full.json
✅ Copied service-account-key.json
   → Set GOOGLE_APPLICATION_CREDENTIALS

📦 Unzipping frames...

✅ Checking unzipped frames...
✅ frames_full/ exists with 2 folders
   📁 fall/ (93 files)
   📁 non_fall/ (969 files)

✅ All files loaded from Google Drive!
```

---

## 🔄 STEP 6: Replace PROJECT_ID Cell

1. **Scroll down** to find the cell that contains:
   ```python
   PROJECT_ID = userdata.get('GOOGLE_CLOUD_PROJECT')
   ```

2. **Delete everything** in that cell

3. **Copy-paste this code:**

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

4. **Run the cell**
5. **Expected output:**
```
✅ Environment initialized
✅ Project: 948622252329
✅ Region: us-central1
✅ Bucket: gs://fall-detection-finetuning-nikhil
📂 Prefix: multimodal_sft_20250111_143022
```

---

## 🔄 STEP 7: Replace Dataset Loading Cell

1. **Scroll down** to find the cell that contains:
   ```python
   from datasets import load_dataset
   import ipywidgets as widgets
   ...
   LABEL_TO_REPO = {
       "Butterflies": "Dasool/butterflies_and_moths_vqa",
   ```

2. **Delete everything** in that cell

3. **Copy-paste this code:**

```python
import json
import os
from PIL import Image
import shutil

# ========================================
# STEP 1: Copy JSON files from Google Drive
# ========================================
print("📂 Copying JSON files from Google Drive...")

DRIVE_FOLDER = '/content/drive/MyDrive/tuning'

json_files = [
    'fall_detection_dataset_full.json',
    'train_split_full.json',
    'val_split_full.json',
    'test_split_full.json'
]

for filename in json_files:
    src = f'{DRIVE_FOLDER}/{filename}'
    dst = f'/content/{filename}'

    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  ✅ Copied {filename}")
    else:
        print(f"  ❌ Not found: {filename}")

# ========================================
# STEP 2: Load JSON files
# ========================================
print("\n📂 Loading fall detection dataset from local frames...")

# Read the JSON files from /content/
with open('/content/fall_detection_dataset_full.json', 'r') as f:
    full_dataset = json.load(f)

with open('/content/train_split_full.json', 'r') as f:
    train_data = json.load(f)

with open('/content/val_split_full.json', 'r') as f:
    val_data = json.load(f)

with open('/content/test_split_full.json', 'r') as f:
    test_data = json.load(f)

print(f"✅ Loaded JSON files:")
print(f"  Full dataset: {len(full_dataset)} samples")
print(f"  Train: {len(train_data)} samples")
print(f"  Validation: {len(val_data)} samples")
print(f"  Test: {len(test_data)} samples")

# ========================================
# STEP 3: Fix image paths
# ========================================
print("\n🔧 Fixing image paths for Colab environment...")

def fix_path(item):
    """Remove 'finetune/' prefix from image paths."""
    if 'image_path' in item:
        # Change 'finetune/frames_full/...' to '/content/frames_full/...'
        item['image_path'] = item['image_path'].replace('finetune/', '/content/')
        # If path doesn't start with /content/, add it
        if not item['image_path'].startswith('/content/'):
            item['image_path'] = '/content/' + item['image_path']
    return item

# Fix paths in all datasets
train_data = [fix_path(item) for item in train_data]
val_data = [fix_path(item) for item in val_data]
test_data = [fix_path(item) for item in test_data]
full_dataset = [fix_path(item) for item in full_dataset]

print(f"✅ Fixed image paths")
print(f"  Example path: {train_data[0]['image_path']}")

# Verify first image can be loaded
try:
    test_img = Image.open(train_data[0]['image_path'])
    print(f"✅ Successfully loaded test image: {test_img.size}")
except Exception as e:
    print(f"❌ Error loading image: {e}")
    print(f"   Tried path: {train_data[0]['image_path']}")

# ========================================
# STEP 4: Create dataset class
# ========================================
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

# ========================================
# STEP 5: Print summary
# ========================================
print("\n" + "="*50)
print("✅ DATASET LOADED SUCCESSFULLY!")
print("="*50)
print(f"Total samples: {len(train_data) + len(val_data) + len(test_data)}")
print(f"  Train: {len(train_data)} samples")
print(f"  Validation: {len(val_data)} samples")
print(f"  Test: {len(test_data)} samples")
print(f"  Fall samples: {sum(1 for item in full_dataset if item['label'] == 'fall')}")
print(f"  Non-fall samples: {sum(1 for item in full_dataset if item['label'] == 'non_fall')}")
print(f"\nDataset name: {SELECTED_HF_NAME}")
print("="*50)
```

4. **Run the cell**
5. **Expected output:**
```
📂 Loading fall detection dataset from local frames...
✅ Loaded fall detection dataset into variable `ds`.
Total samples: 1062
  Train: 762 samples
  Validation: 141 samples
  Test: 159 samples
  Fall samples: 93
  Non-fall samples: 969
SELECTED_HF_NAME = fall_detection
```

---

## ▶️ STEP 8: Run All Cells

1. **Click:** Runtime → Run all
2. **Wait** for all cells to execute
3. **Watch for errors** - if you see any, tell me!

---

## ⏱️ STEP 9: Wait for Fine-Tuning

The fine-tuning will take **2-4 hours**. You'll see:

```
🚀 Starting fine-tuning job...
Job ID: 1234567890123456789
State: PENDING
...
State: RUNNING
...
```

**You can close the browser tab** - the job will continue running in Google Cloud.

To check status later:
1. Reopen the notebook
2. Run the "Check Job Status" cell

---

## ✅ STEP 10: Evaluate Results

After fine-tuning completes, run the evaluation cells to see:
- Accuracy
- F1 Score
- Confusion Matrix
- Sample predictions

Compare with your LSTM model (99.42% F1 score)!

---

## 🎯 Summary Checklist

Before running, make sure:

- [ ] All 6 files uploaded to Colab
- [ ] Cell 1 added (unzip frames)
- [ ] Cell 2 added (authentication)
- [ ] PROJECT_ID cell replaced
- [ ] Dataset loading cell replaced
- [ ] All cells run successfully
- [ ] No errors in output

---

## 🆘 Common Errors

### "FileNotFoundError: frames_full.zip"
→ Upload the file using the Files panel (📁)

### "Authentication failed"
→ Check your service-account-key.json is correct

### "Bucket not found"
→ Verify your bucket name: `fall-detection-finetuning-nikhil`

### "Permission denied"
→ Make sure your service account has the right roles

---

**When done, tell me: "done step 4"** 😊

