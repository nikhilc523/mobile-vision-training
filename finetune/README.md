# 🎓 **Gemini Fine-Tuning Assignment - What Your Professor Wants**

## ❓ **Your Question: "Custom Dataset" = URFD + Le2i?**

**YES! You're absolutely right!** ✅

Your professor wants you to use **YOUR FULL TRAINING DATASET** (URFD + Le2i + UCF101), NOT just the 8 test videos.

---

## 📊 **What is "Your Custom Dataset"?**

### **Your Custom Dataset = The SAME dataset you used for LSTM training**

- **URFD:** 70 fall videos + 40 ADL (non-fall) videos
- **Le2i:** 191 fall videos + 130 non-fall videos  
- **UCF101:** 500 non-fall videos (for hard negatives)

**Total:** ~964 videos with fall/non-fall labels

**This is YOUR custom dataset** because:
- ✅ You collected/downloaded it
- ✅ You labeled it (fall vs non-fall)
- ✅ You used it to train your LSTM model
- ✅ It's specific to YOUR fall detection project

---

## 🎯 **What Your Professor Wants**

Your professor wants you to:

1. ✅ Take the Gemini fine-tuning notebook
2. ✅ Replace the sample dataset (Butterflies/Book Cover VQA) with **YOUR URFD + Le2i dataset**
3. ✅ Fine-tune Gemini to detect falls using YOUR data
4. ✅ Compare Gemini vs your LSTM model

**Why?** To learn about different ML approaches:
- **Your LSTM:** Trains from scratch on keypoints
- **Gemini:** Fine-tunes a pre-trained vision model on images

---

## 🔄 **Two Options for Dataset**

### **Option 1: Full Dataset (URFD + Le2i + UCF101)** ✅ **RECOMMENDED**
- **Size:** ~964 videos → ~2,900 frames (3 frames per video)
- **Pros:** Same dataset as LSTM, fair comparison
- **Cons:** Large dataset, may take longer to upload/train
- **Use script:** `prepare_urfd_le2i_dataset.py`

### **Option 2: Small Test Dataset (8 videos)** ❌ **NOT RECOMMENDED**
- **Size:** 8 videos → ~80 frames (10 frames per video)
- **Pros:** Fast to prepare and upload
- **Cons:** Too small for fine-tuning, not a fair comparison
- **Use script:** `prepare_dataset.py`

**I recommend Option 1** because:
- It's what your professor means by "YOUR custom dataset"
- It's a fair comparison with your LSTM model
- It will give better fine-tuning results

---

## 🚀 **How to Prepare YOUR Dataset**

### **STEP 1: Run the Preparation Script**

```bash
cd /Users/nikhilchowdary/mobile-vision-training
python finetune/prepare_urfd_le2i_dataset.py
```

**This will:**
- ✅ Find all URFD videos (falls/ and adl/ folders)
- ✅ Find all Le2i videos (Coffee_room_01/, Home_01/, etc.)
- ✅ Find UCF101 videos (limit to 100 for balance)
- ✅ Extract 3 frames per video (resized to 512×512)
- ✅ Create train/val/test splits
- ✅ Save frames to `finetune/frames_full/`

**Expected output:**
```
URFD: 110 videos
Le2i: 321 videos
UCF101: 100 videos
Total: 531 videos

Total samples: ~1,593 frames
Fall samples: ~783 frames
Non-fall samples: ~810 frames
```

---

### **STEP 2: Zip the Frames**

```bash
cd finetune
zip -r frames_full.zip frames_full/
```

**This creates:** `finetune/frames_full.zip` (~200-500 MB)

---

### **STEP 3: Upload to Google Colab**

1. Open https://colab.research.google.com/
2. Upload `GeminiMultiModalFineTune.ipynb`
3. Upload `frames_full.zip`
4. Unzip:
   ```python
   !unzip frames_full.zip -d /content/
   ```

---

### **STEP 4: Modify Cell 13 in Notebook**

Replace the dataset loading code with:

```python
# Load YOUR Fall Detection Dataset (URFD + Le2i + UCF101)
import pandas as pd
from PIL import Image
import os

def load_fall_detection_dataset():
    """Load YOUR custom fall detection dataset"""
    frames_dir = '/content/frames_full'
    
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
    print(f"✅ Loaded YOUR custom dataset:")
    print(f"   Total: {len(df)} samples")
    print(f"   Fall: {len(df[df['answer'].str.contains('Yes')])}")
    print(f"   Non-fall: {len(df[df['answer'].str.contains('No')])}")
    print(f"   Dataset: URFD + Le2i + UCF101")
    
    return df

# Load YOUR custom dataset
ds = load_fall_detection_dataset()
SELECTED_HF_NAME = "fall_detection_urfd_le2i_ucf101"
```

---

### **STEP 5: Run Fine-Tuning**

Follow the notebook cells:
1. Install libraries
2. Authenticate with Google Cloud
3. Choose model settings (gemini-2.5-flash, 2-5 epochs)
4. Load YOUR dataset (modified Cell 13)
5. Sample data
6. Upload to GCS
7. Start fine-tuning (2-4 hours)
8. Test tuned model
9. Evaluate results

---

## 📊 **Expected Results**

### **Dataset Size:**
- ~1,593 frames (531 videos × 3 frames)
- ~783 fall frames
- ~810 non-fall frames

### **Training Time:**
- 2 epochs: ~2-3 hours
- 5 epochs: ~4-6 hours

### **Cost:**
- Fine-tuning: ~$10-20
- Storage: ~$1
- Inference: ~$0.10 per 1000 requests

**Your Colab credits should cover this!**

---

## 📝 **What to Submit**

1. **Modified Colab notebook** (with YOUR dataset code in Cell 13)
2. **Fine-tuning job ID**
3. **Evaluation results** (accuracy, F1 score)
4. **Comparison report:**
   - **Gemini F1 score:** ? (you'll find out)
   - **LSTM F1 score:** 99.42%
   - **Which is better?**
   - **Pros/cons of each approach**

---

## 🔍 **Comparison: Gemini vs LSTM**

| Feature | Your LSTM | Gemini Fine-Tuned |
|---------|-----------|-------------------|
| **Input** | 30 frames (keypoints) | Single image |
| **Training Data** | URFD + Le2i + UCF101 | URFD + Le2i + UCF101 |
| **Model** | BiLSTM (94K params) | Gemini (billions of params) |
| **Training** | From scratch | Fine-tuned (transfer learning) |
| **Accuracy** | 99.42% F1 | ? (you'll find out!) |
| **Speed** | 250ms | ? (likely slower) |
| **Cost** | Free (local) | $10-20 (cloud) |
| **Deployment** | Mobile-friendly | Cloud API |

**Your professor wants you to compare these two approaches!**

---

## 📂 **Files in This Folder**

1. **`GeminiMultiModalFineTune.ipynb`** - Lab notebook (provided by professor)
2. **`prepare_urfd_le2i_dataset.py`** - Script to prepare YOUR full dataset ✅ **USE THIS**
3. **`prepare_dataset.py`** - Script for small test dataset (8 videos) ❌ **DON'T USE**
4. **`INSTRUCTIONS.md`** - Detailed guide
5. **`QUICK_START.md`** - Quick reference
6. **`README.md`** - This file (what your professor wants)

---

## ✅ **Summary**

**Your professor wants:**
- ✅ Use YOUR custom dataset (URFD + Le2i + UCF101)
- ✅ Fine-tune Gemini on YOUR data
- ✅ Compare Gemini vs your LSTM model

**What to do:**
1. Run `python finetune/prepare_urfd_le2i_dataset.py`
2. Zip the frames: `cd finetune && zip -r frames_full.zip frames_full/`
3. Upload to Colab
4. Modify Cell 13 in notebook
5. Run fine-tuning
6. Compare results with your LSTM model

**Estimated time:** 6-8 hours (including fine-tuning)
**Estimated cost:** $10-20 (covered by Colab credits)

---

## 🚀 **Next Steps**

1. **Read this file** (you're doing it now! ✅)
2. **Run the script:** `python finetune/prepare_urfd_le2i_dataset.py`
3. **Follow QUICK_START.md** for step-by-step instructions
4. **Start fine-tuning!**

**Good luck!** 🎉

