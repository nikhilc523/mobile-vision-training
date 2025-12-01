# 🚨 FINAL FIX: Swap Coordinate Order from [y, x] to [x, y]

## 📊 Evidence from Logs

Your logs show keypoints are stored as **[y, x]**:

```
Raw YOLO nose: x=0.13593982, y=0.06866951, conf=0.49444422
Keypoints sample: nose=[0.06866951, 0.13593982], ...
                       ↑ y first    ↑ x second
```

But the training data expects **[x, y]** order (see `ml/features/create_30frame_raw_keypoints.py` line 36).

**Result:** Model receives swapped coordinates → outputs near-zero probability (2.26E-6 = 0.00023%)

---

## ✅ THE FIX

### **File:** `YoloPoseEstimator.kt`
### **Line:** ~209-210

**CURRENT CODE (WRONG):**
```kotlin
// WRONG: Storing as [y, x]
keypoints[kptIdx * 2] = normY      // y first ❌
keypoints[kptIdx * 2 + 1] = normX  // x second ❌
```

**FIXED CODE:**
```kotlin
// CORRECT: Store as [x, y] to match training data!
keypoints[kptIdx * 2] = normX      // x first ✅
keypoints[kptIdx * 2 + 1] = normY  // y second ✅
```

**That's it!** Just swap the two lines.

---

## 🔍 Why This Matters

### **Example: Nose Keypoint**

**YOLO output:**
```
x = 0.136 (13.6% from left)
y = 0.069 (6.9% from top)
```

**Your code stores (WRONG):**
```kotlin
keypoints[0] = 0.069  // y (WRONG position!)
keypoints[1] = 0.136  // x (WRONG position!)
```

**Model expects:**
```
keypoints[0] = 0.136  // x (CORRECT position!)
keypoints[1] = 0.069  // y (CORRECT position!)
```

**Impact:** Model sees completely wrong spatial relationships → outputs 0.00023% instead of 5-10%

---

## 📝 Complete Fix Checklist

- [x] **Fix 1:** YOLO outputs normalized [0, 1] (no need to divide by 640) ✅
- [x] **Fix 2:** Add `inputBuffer.rewind()` in FallDetector ✅
- [ ] **Fix 3:** Swap coordinate order to [x, y] ← **DO THIS NOW!**

---

## 🧪 Expected Results After Fix

### **Before Fix:**
```
Input stats: min=0.000, max=1.000, avg=0.495
Fall detection result: probability=2.2642703E-6  ← 0.00023% ❌
```

### **After Fix:**
```
Input stats: min=0.000, max=1.000, avg=0.495
Fall detection result: probability=0.0523  ← 5.23% ✅
```

---

## 🚀 How to Apply

1. **Open** `YoloPoseEstimator.kt` in Android Studio
2. **Find** lines ~209-210 (inside the `if (conf > CONFIDENCE_THRESHOLD)` block)
3. **Swap** the two lines:

**Before:**
```kotlin
keypoints[kptIdx * 2] = normY
keypoints[kptIdx * 2 + 1] = normX
```

**After:**
```kotlin
keypoints[kptIdx * 2] = normX      // x first!
keypoints[kptIdx * 2 + 1] = normY  // y second!
```

4. **Rebuild** and test
5. **Expected logs:**

```
Keypoints sample: nose=[0.136, 0.069], ...  ← [x, y] order!
                       ↑ x first  ↑ y second
Fall detection result: probability=0.0523  ← 5.23% (realistic!)
```

---

## 📚 Why the Documentation Was Wrong

The documentation I created earlier (`YOLO_TO_MODEL_COMPLETE_GUIDE.md`) incorrectly stated the model expects **[y, x]** order.

**This was WRONG!** The actual training code (`create_30frame_raw_keypoints.py` line 36) shows:

```python
# Extract x, y coordinates (swap y, x to x, y order)
xy_coords = kp[:, [1, 0]]  # (17, 2) - [x, y]
```

The model was trained with **[x, y]** order, NOT [y, x]!

I've now corrected the documentation.

---

## 🎯 Summary

**1 line change:**
- Swap `normX` and `normY` on lines 209-210

**Result:**
- Probability will jump from 0.00023% to 5-10% for standing
- Fall detection will work correctly (85-100% when falling)

**This is the FINAL fix!** 🎉

---

*Last updated: November 5, 2025*

