# 📋 WHAT TO USE - Quick Reference Guide

**Last Updated:** November 5, 2025  
**Purpose:** Tell you exactly which files to use for YOLO integration

---

## 🎯 **FOR YOU (Human Developer)**

### **Step 1: Read These (in order)**

1. **FINAL_SUMMARY.md** ⭐ (5 minutes)
   - Overview of what you have
   - What problem we solved
   - What to do next

2. **YOLO_TO_MODEL_COMPLETE_GUIDE.md** ⭐⭐⭐ (15 minutes)
   - YOLO output format (CHW layout)
   - Model input format ([y,x] order)
   - Conversion process (step-by-step)
   - Common issues and solutions
   - **READ THIS BEFORE CODING!**

3. **DATA_FLOW_DIAGRAM.md** (5 minutes)
   - Visual guide of data flow
   - Format at each step
   - Coordinate transformation example

**Total time:** 25 minutes  
**Why:** Understand everything before starting

---

### **Step 2: Copy These Files**

1. **YOLO Model**
   ```bash
   cp ~/mobile-vision-training/ml/export/yolo11n-pose_float32.tflite \
      /path/to/android/app/src/main/assets/yolo11n-pose.tflite
   ```

2. **BiLSTM Model** (already done)
   ```bash
   # Should already be in:
   # /path/to/android/app/src/main/assets/fall_detection_model.tflite
   ```

---

### **Step 3: Use This Prompt**

**File:** `AUGMENT_PROMPT_YOLO_FIXED.txt` ⭐

**How to use:**
1. Open Android Studio
2. Open your fall detection project
3. Open `ml/export/AUGMENT_PROMPT_YOLO_FIXED.txt`
4. Copy the entire content
5. Paste to Augment AI
6. Tell Augment: "Read YOLO_TO_MODEL_COMPLETE_GUIDE.md first, then implement this"

**Why this one:**
- ✅ Correct CHW reading
- ✅ Correct coordinate conversion
- ✅ Complete implementation
- ✅ References the complete guide

---

### **Step 4: Reference These (if needed)**

- **YOLO_TFLITE_CONVERSION_SUCCESS.md** - Model specs and conversion details
- **IMPLEMENTATION_CHECKLIST.md** - Verification checklist
- **INDEX.md** - Master index of all files

---

## 🤖 **FOR AUGMENT AI (Other Instances)**

### **If another Augment is helping with Android integration:**

**Tell them to read (in order):**

1. **YOLO_TO_MODEL_COMPLETE_GUIDE.md** ⭐⭐⭐ (MUST READ!)
   - Explains YOLO output format (CHW layout)
   - Explains model input format ([y,x] order)
   - Explains conversion process
   - Prevents all confusion!

2. **AUGMENT_PROMPT_YOLO_FIXED.txt** (USE THIS!)
   - Complete implementation
   - Correct CHW reading
   - Correct coordinate conversion

**Critical points to emphasize:**

1. **YOLO output is CHW format (channels-first)**
   ```kotlin
   // CORRECT:
   val position = (featureIdx * 8400 + detectionIdx) * 4
   
   // WRONG:
   val position = (detectionIdx * 56 + featureIdx) * 4
   ```

2. **Coordinates are in pixels (0-640), must normalize**
   ```kotlin
   // CORRECT:
   val normX = x / 640.0f
   val normY = y / 640.0f
   
   // WRONG:
   val normX = x  // Already normalized? NO!
   ```

3. **Coordinates must be swapped to [y, x] order**
   ```kotlin
   // CORRECT:
   modelKeypoints[i * 2] = normY      // y first!
   modelKeypoints[i * 2 + 1] = normX  // x second!
   
   // WRONG:
   modelKeypoints[i * 2] = normX      // x first? NO!
   modelKeypoints[i * 2 + 1] = normY  // y second? NO!
   ```

4. **Last keypoint (right_ankle) has no confidence**
   ```kotlin
   // CORRECT:
   val conf = if (kptIdx < 16) {
       outputBuffer.getFloat()
   } else {
       1.0f  // Assume visible
   }
   
   // WRONG:
   val conf = outputBuffer.getFloat()  // Will overflow!
   ```

---

## 📁 **File Organization**

### **✅ USE THESE FILES**

| File | Purpose | When to Use |
|------|---------|-------------|
| **FINAL_SUMMARY.md** | Overview | Start here |
| **YOLO_TO_MODEL_COMPLETE_GUIDE.md** | Complete guide | Before coding |
| **AUGMENT_PROMPT_YOLO_FIXED.txt** | Implementation prompt | Give to Augment |
| **DATA_FLOW_DIAGRAM.md** | Visual guide | For understanding |
| **yolo11n-pose_float32.tflite** | YOLO model | Copy to Android |
| **fall_detection_model.tflite** | BiLSTM model | Copy to Android |

### **📚 REFERENCE FILES (Optional)**

| File | Purpose | When to Use |
|------|---------|-------------|
| YOLO_TFLITE_CONVERSION_SUCCESS.md | Conversion summary | For model specs |
| IMPLEMENTATION_CHECKLIST.md | Verification checklist | During testing |
| INDEX.md | Master index | To find files |

### **❌ OLD FILES (Don't Use)**

| File | Status | Why Not |
|------|--------|---------|
| AUGMENT_PROMPT_YOLO.txt | ❌ OLD | Missing CHW reading |
| YOLO_INTEGRATION_GUIDE.md | ❌ OLD | Incomplete conversion |
| YOLO_QUICK_START.md | ❌ OLD | Outdated instructions |

---

## 🎯 **Quick Decision Tree**

### **"I want to understand YOLO integration"**
→ Read: **YOLO_TO_MODEL_COMPLETE_GUIDE.md**

### **"I want to see the data flow"**
→ Read: **DATA_FLOW_DIAGRAM.md**

### **"I want to implement YOLO in Android"**
→ Use: **AUGMENT_PROMPT_YOLO_FIXED.txt**

### **"I want to know what files to copy"**
→ Copy: **yolo11n-pose_float32.tflite** to Android assets

### **"I want to verify everything works"**
→ Use: **IMPLEMENTATION_CHECKLIST.md**

### **"I want to troubleshoot issues"**
→ Read: **YOLO_TO_MODEL_COMPLETE_GUIDE.md** (Section 5)

### **"I want to tell another Augment what to do"**
→ Tell them: "Read YOLO_TO_MODEL_COMPLETE_GUIDE.md, then use AUGMENT_PROMPT_YOLO_FIXED.txt"

---

## 📊 **File Sizes**

| File | Size | Type |
|------|------|------|
| yolo11n-pose_float32.tflite | 11.3 MB | Model |
| fall_detection_model.tflite | 407 KB | Model |
| YOLO_TO_MODEL_COMPLETE_GUIDE.md | ~15 KB | Documentation |
| AUGMENT_PROMPT_YOLO_FIXED.txt | ~8 KB | Prompt |
| DATA_FLOW_DIAGRAM.md | ~12 KB | Documentation |
| FINAL_SUMMARY.md | ~10 KB | Documentation |

---

## ✅ **Success Criteria**

### **After reading the guides:**
- [ ] Understand YOLO output is CHW format
- [ ] Understand coordinates are in pixels (0-640)
- [ ] Understand coordinates must be swapped to [y, x]
- [ ] Understand last keypoint has no confidence

### **After implementation:**
- [ ] YOLO model loads successfully
- [ ] Person detected with confidence > 0.3
- [ ] Keypoints extracted in [y, x] format
- [ ] Keypoint values are 0.0-1.0
- [ ] Inference time < 100ms
- [ ] Probability updates in real-time
- [ ] Fall detection triggers at > 85%

---

## 🎊 **Summary**

### **For Human Developer:**
1. Read: **FINAL_SUMMARY.md** + **YOLO_TO_MODEL_COMPLETE_GUIDE.md** + **DATA_FLOW_DIAGRAM.md**
2. Copy: **yolo11n-pose_float32.tflite** to Android assets
3. Use: **AUGMENT_PROMPT_YOLO_FIXED.txt** with Augment AI
4. Test: Follow **IMPLEMENTATION_CHECKLIST.md**

### **For Augment AI:**
1. Read: **YOLO_TO_MODEL_COMPLETE_GUIDE.md** (MUST READ!)
2. Use: **AUGMENT_PROMPT_YOLO_FIXED.txt** (implementation)
3. Remember: CHW format, normalize by 640, swap to [y,x], last keypoint has no conf

### **Key Files:**
- ⭐⭐⭐ **YOLO_TO_MODEL_COMPLETE_GUIDE.md** - Read this first!
- ⭐⭐ **AUGMENT_PROMPT_YOLO_FIXED.txt** - Use this prompt!
- ⭐ **DATA_FLOW_DIAGRAM.md** - Visual guide
- ⭐ **yolo11n-pose_float32.tflite** - The model!

---

**Everything you need is here! Follow the guides and YOLO integration will work!** 🎯

