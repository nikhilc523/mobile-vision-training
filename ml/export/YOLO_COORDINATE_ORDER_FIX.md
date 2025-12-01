# 🚨 CRITICAL FIX: Coordinate Order is [x, y] NOT [y, x]!

## 🐛 The REAL Problem

Your probability is stuck at near-zero (0.0004%) because of a **coordinate order mismatch**!

### **Training Data Format**

Looking at `ml/features/create_30frame_raw_keypoints.py` line 36:

```python
# Extract x, y coordinates (swap y, x to x, y order)
xy_coords = kp[:, [1, 0]]  # (17, 2) - [x, y]

# Flatten to (34,)
features[t] = xy_coords.flatten()
```

**The model was trained with [x, y] order!**

Flattened format:
```
[x0, y0, x1, y1, x2, y2, ..., x16, y16]
```

---

### **Your Android Code**

Your `YoloPoseEstimator.kt` stores keypoints as **[y, x]**:

```kotlin
// WRONG! Model expects [x, y] but you're storing [y, x]
keypoints[kptIdx * 2] = normY      // y first ❌
keypoints[kptIdx * 2 + 1] = normX  // x second ❌
```

**This is backwards!**

---

## ✅ **The Fix**

### **Change YoloPoseEstimator.kt (Line 165-166)**

**BEFORE (WRONG):**
```kotlin
if (conf > CONFIDENCE_THRESHOLD) {
    val normX = (x / 640.0f).coerceIn(0f, 1f)
    val normY = (y / 640.0f).coerceIn(0f, 1f)

    // WRONG: Storing as [y, x]
    keypoints[kptIdx * 2] = normY      // ❌ y first
    keypoints[kptIdx * 2 + 1] = normX  // ❌ x second
}
```

**AFTER (CORRECT):**
```kotlin
if (conf > CONFIDENCE_THRESHOLD) {
    val normX = (x / 640.0f).coerceIn(0f, 1f)
    val normY = (y / 640.0f).coerceIn(0f, 1f)

    // CORRECT: Store as [x, y] to match training data!
    keypoints[kptIdx * 2] = normX      // ✅ x first!
    keypoints[kptIdx * 2 + 1] = normY  // ✅ y second!
}
```

---

## 📊 Why This Matters

### **Example: Nose Keypoint**

**YOLO output:**
```
x = 320.5 pixels → normalized = 0.501
y = 240.3 pixels → normalized = 0.376
```

**Your code (WRONG):**
```kotlin
keypoints[0] = 0.376  // y (WRONG position!)
keypoints[1] = 0.501  // x (WRONG position!)
```

**Model expects:**
```
keypoints[0] = 0.501  // x (CORRECT position!)
keypoints[1] = 0.376  // y (CORRECT position!)
```

**Result:** Model sees completely wrong spatial relationships → outputs near-zero probability!

---

## 🎯 Complete Fix Summary

### **Fix 1: Normalize coordinates (ALREADY DONE)**

```kotlin
val normX = (x / 640.0f).coerceIn(0f, 1f)  // ✅
val normY = (y / 640.0f).coerceIn(0f, 1f)  // ✅
```

---

### **Fix 2: Swap coordinate order (NEW FIX!)**

**BEFORE:**
```kotlin
keypoints[kptIdx * 2] = normY      // ❌ WRONG!
keypoints[kptIdx * 2 + 1] = normX  // ❌ WRONG!
```

**AFTER:**
```kotlin
keypoints[kptIdx * 2] = normX      // ✅ CORRECT!
keypoints[kptIdx * 2 + 1] = normY  // ✅ CORRECT!
```

---

### **Fix 3: Add inputBuffer.rewind() in FallDetector (ALREADY DONE)**

```kotlin
inputBuffer.rewind()  // ✅
```

---

## 🧪 Expected Results After Fix

### **Test 1: Standing Still**

**Before fix:**
```
Probability: 0.0004%  ❌ (near zero)
```

**After fix:**
```
Probability: 2-10%  ✅ (realistic)
```

---

### **Test 2: Bending Forward**

**Before fix:**
```
Probability: 0.0005%  ❌ (near zero)
```

**After fix:**
```
Probability: 15-50%  ✅ (realistic)
```

---

### **Test 3: Falling**

**Before fix:**
```
Probability: 0.0006%  ❌ (near zero)
```

**After fix:**
```
Probability: 85-100%  ✅ (FALL DETECTED!)
```

---

## 📝 Quick Fix Checklist

- [x] **Fix 1:** Add `/ 640.0f` to normalize coordinates
- [ ] **Fix 2:** Swap coordinate order to [x, y] ← **DO THIS NOW!**
- [x] **Fix 3:** Add `inputBuffer.rewind()` in FallDetector

**Only Fix 2 is missing!** This is why your probability is still near-zero!

---

## 🔧 Exact Code Change

**File:** `YoloPoseEstimator.kt`  
**Line:** ~165-166

**Find this:**
```kotlin
keypoints[kptIdx * 2] = normY
keypoints[kptIdx * 2 + 1] = normX
```

**Replace with:**
```kotlin
keypoints[kptIdx * 2] = normX      // ✅ x first!
keypoints[kptIdx * 2 + 1] = normY  // ✅ y second!
```

**That's it!** Just swap the two lines.

---

## 🚀 After Applying the Fix

1. **Rebuild the app**
2. **Test with standing, bending, falling**
3. **Expected logs:**

```
YoloPoseEstimator: ✅ Person detected (conf: 0.95)
YoloPoseEstimator: Valid keypoints: 14 / 17
YoloPoseEstimator: Keypoints sample: nose=[0.501, 0.376], ...  ← [x, y] order!
FallDetector: Inference completed in 12ms, probability: 0.0523  ← 5.2% (realistic!)
```

4. **When you fall:**

```
FallDetector: Inference completed in 13ms, probability: 0.9234  ← 92.3% (FALL!)
MainActivity: ⚠️ FALL DETECTED! Triggering emergency alert...
```

---

## 📚 Why the Documentation Was Wrong

My earlier documentation (`YOLO_TO_MODEL_COMPLETE_GUIDE.md`) incorrectly stated the model expects **[y, x]** order.

**This was WRONG!** The actual training code shows the model was trained with **[x, y]** order.

I apologize for the confusion. The correct order is:

```
✅ CORRECT: [x, y] order
❌ WRONG: [y, x] order
```

---

*Last updated: November 5, 2025*

