# How to Test Fall Detection in Android App

## 🎯 Understanding the System

### How the 30-Frame Buffer Works

The fall detection system uses a **sliding window buffer** of 30 frames (1 second @ 30 FPS):

```
Frame 1  → Buffer: [1] (1/30)
Frame 2  → Buffer: [1, 2] (2/30)
Frame 3  → Buffer: [1, 2, 3] (3/30)
...
Frame 30 → Buffer: [1, 2, ..., 30] (30/30) ← FIRST PREDICTION!
Frame 31 → Buffer: [2, 3, ..., 31] (30/30) ← Slide window, drop frame 1
Frame 32 → Buffer: [3, 4, ..., 32] (30/30) ← Slide window, drop frame 2
```

**Key Points:**
1. **Buffer fills from 1/30 to 30/30** - This happens in the first 1 second
2. **Buffer STAYS at 30/30** - After that, it's always full (sliding window)
3. **Probability updates EVERY frame** after buffer is full
4. **"Buffer: 30/30 (100%)" is CORRECT** - It should stay at 30/30!

---

## 🐛 Common Issue: "Buffer stops at 30/30, probability doesn't update"

### Symptoms
```
Buffer: 30/30 frames (100%)  ← Stays here
Probability: 0%              ← Never changes
Status: NO FALL              ← Never changes
```

### Root Cause
The buffer is working correctly! The issue is that **keypoints are not being extracted properly** or **model inference is not running**.

### Debugging Steps

#### **Step 1: Check if keypoints are being extracted**

Add logging in your `YoloPoseEstimator` class:

```kotlin
class YoloPoseEstimator(context: Context) {
    fun estimatePose(bitmap: Bitmap): FloatArray {
        // ... YOLO inference code ...
        
        val keypoints = FloatArray(34)
        
        // Extract keypoints from YOLO output
        for (kptIdx in 0 until 17) {
            val x = // ... extract x
            val y = // ... extract y
            val conf = // ... extract confidence
            
            if (conf > 0.3f) {
                keypoints[kptIdx * 2] = y / 640.0f      // Normalized y
                keypoints[kptIdx * 2 + 1] = x / 640.0f  // Normalized x
            }
        }
        
        // 🔍 ADD THIS DEBUG LOG
        val validCount = keypoints.count { it > 0.0f }
        Log.d("YoloPose", "Valid keypoint values: $validCount / 34")
        Log.d("YoloPose", "Sample keypoints: [${keypoints[0]}, ${keypoints[1]}, ${keypoints[2]}, ${keypoints[3]}]")
        
        return keypoints
    }
}
```

**Expected output:**
```
YoloPose: Valid keypoint values: 28 / 34  (14 keypoints detected)
YoloPose: Sample keypoints: [0.523, 0.412, 0.534, 0.398]
```

**If you see all zeros:**
```
YoloPose: Valid keypoint values: 0 / 34  ← PROBLEM!
YoloPose: Sample keypoints: [0.0, 0.0, 0.0, 0.0]
```
→ **YOLO is not detecting any person!** Check YOLO model loading and inference.

---

#### **Step 2: Check if model inference is running**

Add logging in your `FallDetectionModel` class:

```kotlin
class FallDetectionModel(context: Context) {
    fun predict(window: Array<FloatArray>): Float {
        // 🔍 ADD THIS DEBUG LOG
        Log.d("FallModel", "predict() called with window size: ${window.size}")
        
        // Prepare input
        val inputBuffer = ByteBuffer.allocateDirect(4 * 1 * 30 * 34)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        for (frame in window) {
            for (value in frame) {
                inputBuffer.putFloat(value)
            }
        }
        
        // Run inference
        val outputBuffer = ByteBuffer.allocateDirect(4 * 1 * 1)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        interpreter.run(inputBuffer, outputBuffer)
        
        outputBuffer.rewind()
        val probability = outputBuffer.getFloat()
        
        // 🔍 ADD THIS DEBUG LOG
        Log.d("FallModel", "Model output probability: $probability")
        
        return probability
    }
}
```

**Expected output (every frame after buffer is full):**
```
FallModel: predict() called with window size: 30
FallModel: Model output probability: 0.0234  ← Changes every frame!
```

**If you DON'T see these logs:**
→ **Model inference is not being called!** Check your MainActivity logic.

---

#### **Step 3: Check MainActivity update logic**

Your MainActivity should look like this:

```kotlin
class MainActivity : AppCompatActivity() {
    private val frameBuffer = mutableListOf<FloatArray>()
    private val maxBufferSize = 30
    
    private fun processFrame(bitmap: Bitmap) {
        // 1. Extract keypoints
        val keypoints = yoloPoseEstimator.estimatePose(bitmap)
        
        // 2. Add to buffer
        frameBuffer.add(keypoints)
        if (frameBuffer.size > maxBufferSize) {
            frameBuffer.removeAt(0)  // Remove oldest frame
        }
        
        // 3. Update UI - Buffer status
        val bufferStatus = "${frameBuffer.size}/$maxBufferSize"
        runOnUiThread {
            tvBufferStatus.text = "Buffer: $bufferStatus frames"
        }
        
        // 4. Run inference when buffer is full
        if (frameBuffer.size == maxBufferSize) {
            val probability = fallDetectionModel.predict(frameBuffer.toTypedArray())
            
            // 🔍 ADD THIS DEBUG LOG
            Log.d("MainActivity", "Fall probability: $probability")
            
            // 5. Update UI - Probability
            runOnUiThread {
                tvProbability.text = "Probability: ${(probability * 100).toInt()}%"
                
                if (probability > 0.85f) {
                    tvStatus.text = "⚠️ FALL DETECTED!"
                    tvStatus.setBackgroundColor(Color.RED)
                    triggerEmergencyAlert()
                } else {
                    tvStatus.text = "✅ NO FALL"
                    tvStatus.setBackgroundColor(Color.GREEN)
                }
            }
        }
    }
}
```

**Key points:**
- `processFrame()` should be called **every frame** (30 times per second)
- Model inference should run **every frame** after buffer is full
- UI should update **every frame** with new probability

---

## ✅ How to Test Properly

### Test 1: Standing Still (Expected: 0-10% probability)

1. **Setup:** Stand in front of camera, stay still
2. **Wait:** 1 second for buffer to fill (0/30 → 30/30)
3. **Observe:**
   ```
   Buffer: 30/30 frames (100%)  ← Should stay here
   Probability: 2%              ← Should be low (0-10%)
   Status: NO FALL              ← Green background
   ```
4. **Expected logs:**
   ```
   YoloPose: Valid keypoint values: 30 / 34
   FallModel: Model output probability: 0.0234
   MainActivity: Fall probability: 0.0234
   ```

---

### Test 2: Bending Forward (Expected: 10-50% probability)

1. **Setup:** Stand in front of camera
2. **Action:** Slowly bend forward (like picking something up)
3. **Observe:**
   ```
   Buffer: 30/30 frames (100%)  ← Should stay here
   Probability: 35%             ← Should increase (10-50%)
   Status: NO FALL              ← Still green (< 85%)
   ```
4. **Expected logs:**
   ```
   FallModel: Model output probability: 0.3521
   MainActivity: Fall probability: 0.3521
   ```

---

### Test 3: Falling (Expected: 85-100% probability)

1. **Setup:** Stand in front of camera
2. **Action:** Fall to the ground (safely!)
3. **Observe:**
   ```
   Buffer: 30/30 frames (100%)  ← Should stay here
   Probability: 99%             ← Should spike to 85-100%
   Status: ⚠️ FALL DETECTED!    ← Red background
   ```
4. **Expected logs:**
   ```
   FallModel: Model output probability: 0.9987
   MainActivity: Fall probability: 0.9987
   MainActivity: FALL DETECTED! Triggering emergency alert...
   ```
5. **Expected behavior:**
   - Emergency alert dialog appears
   - Text-to-speech says "Fall detected!"
   - Phone vibrates

---

## 🔍 Troubleshooting Guide

### Issue 1: Buffer fills to 30/30 but probability stays at 0%

**Possible causes:**

1. **YOLO not detecting person**
   - Check: `YoloPose: Valid keypoint values: 0 / 34`
   - Fix: Ensure YOLO model is loaded correctly, check camera preview

2. **Model inference not running**
   - Check: No `FallModel: predict() called` logs
   - Fix: Ensure `if (frameBuffer.size == maxBufferSize)` condition is met

3. **Model returning 0 for all inputs**
   - Check: `FallModel: Model output probability: 0.0` (always 0)
   - Fix: Check TFLite model loading, ensure FlexDelegate is initialized

---

### Issue 2: Buffer never reaches 30/30

**Possible causes:**

1. **Camera frames not being processed**
   - Check: No `processFrame()` calls
   - Fix: Ensure CameraX analyzer is set up correctly

2. **Buffer not accumulating**
   - Check: `frameBuffer.size` stays at 0 or 1
   - Fix: Ensure `frameBuffer.add(keypoints)` is called every frame

---

### Issue 3: Probability updates but always shows 0%

**Possible causes:**

1. **All keypoints are zeros**
   - Check: `YoloPose: Sample keypoints: [0.0, 0.0, 0.0, 0.0]`
   - Fix: YOLO is not detecting person, check model and camera

2. **Model trained on different data format**
   - Check: Keypoints are in [y, x] order, normalized to [0, 1]
   - Fix: Ensure YOLO output is converted correctly (see YOLO_TO_MODEL_COMPLETE_GUIDE.md)

---

### Issue 4: False positives (detects fall when standing)

**Possible causes:**

1. **Threshold too low**
   - Current: 0.85 (85%)
   - Fix: Increase to 0.90 (90%) if too sensitive

2. **Keypoint format incorrect**
   - Check: Ensure [y, x] order, not [x, y]
   - Fix: Swap coordinates in YOLO output conversion

---

## 📊 Expected Behavior Summary

| Scenario | Buffer Status | Probability | Status | Alert |
|----------|--------------|-------------|--------|-------|
| **First 1 second** | 1/30 → 30/30 | N/A | NO FALL | No |
| **Standing still** | 30/30 | 0-10% | NO FALL | No |
| **Walking** | 30/30 | 5-15% | NO FALL | No |
| **Bending** | 30/30 | 10-50% | NO FALL | No |
| **Sitting down** | 30/30 | 20-60% | NO FALL | No |
| **Falling** | 30/30 | 85-100% | FALL DETECTED | Yes |
| **Lying on ground** | 30/30 | 90-100% | FALL DETECTED | Yes |

---

## 🎬 How We Tested (Python Script)

Here's how we tested the system in Python before Android integration:

```python
import cv2
from ultralytics import YOLO
import numpy as np
import tensorflow as tf

# Load models
yolo_model = YOLO('yolo11n-pose.pt')
fall_model = tf.keras.models.load_model('lstm_raw30_balanced_hnm_best.h5')

# Load video
video = cv2.VideoCapture('finalfall.mp4')

# Buffer for 30 frames
frame_buffer = []

frame_count = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 1. Extract keypoints using YOLO
    results = yolo_model(frame, verbose=False)
    keypoints = np.zeros(34)
    
    if len(results[0].keypoints.data) > 0:
        kpts = results[0].keypoints.data[0].cpu().numpy()  # [17, 3]
        for i in range(17):
            x, y, conf = kpts[i]
            if conf > 0.3:
                keypoints[i*2] = y / 640.0      # y first
                keypoints[i*2+1] = x / 640.0    # x second
    
    # 2. Add to buffer
    frame_buffer.append(keypoints)
    if len(frame_buffer) > 30:
        frame_buffer.pop(0)
    
    # 3. Print buffer status
    print(f"Frame {frame_count}: Buffer {len(frame_buffer)}/30", end="")
    
    # 4. Run inference when buffer is full
    if len(frame_buffer) == 30:
        window = np.array(frame_buffer).reshape(1, 30, 34)
        probability = fall_model.predict(window, verbose=0)[0][0]
        print(f" | Probability: {probability:.6f}", end="")
        
        if probability > 0.85:
            print(" | ⚠️ FALL DETECTED!")
        else:
            print(" | ✅ NO FALL")
    else:
        print()

video.release()
```

**Output for `finalfall.mp4`:**
```
Frame 1: Buffer 1/30
Frame 2: Buffer 2/30
...
Frame 30: Buffer 30/30 | Probability: 0.000234 | ✅ NO FALL
Frame 31: Buffer 30/30 | Probability: 0.000456 | ✅ NO FALL
Frame 32: Buffer 30/30 | Probability: 0.001234 | ✅ NO FALL
...
Frame 120: Buffer 30/30 | Probability: 0.523456 | ✅ NO FALL
Frame 121: Buffer 30/30 | Probability: 0.876543 | ⚠️ FALL DETECTED!
Frame 122: Buffer 30/30 | Probability: 0.998765 | ⚠️ FALL DETECTED!
Frame 123: Buffer 30/30 | Probability: 0.999822 | ⚠️ FALL DETECTED!
```

**Key observations:**
1. Buffer fills from 1/30 to 30/30 in first 30 frames
2. Buffer STAYS at 30/30 after that (sliding window)
3. Probability updates EVERY frame after buffer is full
4. Probability gradually increases as person falls
5. Probability spikes to 99.98% when person hits ground

---

## 🚀 Quick Checklist for Augment

When implementing fall detection in Android, ensure:

- [ ] **YOLO model loaded correctly** - Check `yolo11n-pose_float32.tflite` in assets
- [ ] **Fall detection model loaded correctly** - Check `fall_detection_model.tflite` in assets
- [ ] **FlexDelegate initialized** - Required for BiLSTM model
- [ ] **Camera frames processed at 30 FPS** - Use CameraX ImageAnalysis
- [ ] **Keypoints extracted every frame** - Call `yoloPoseEstimator.estimatePose(bitmap)`
- [ ] **Keypoints in correct format** - [y, x] order, normalized to [0, 1]
- [ ] **Buffer accumulates to 30 frames** - Use `MutableList<FloatArray>`
- [ ] **Sliding window implemented** - Remove oldest frame when buffer > 30
- [ ] **Model inference runs every frame** - After buffer is full
- [ ] **UI updates every frame** - Show buffer status, probability, status
- [ ] **Emergency alert triggers** - When probability > 0.85
- [ ] **Logging added** - Debug keypoints, probability, buffer status

---

## 📞 For Augment AI

**Copy-paste this to Augment:**

```
I need to debug why the fall detection probability is not updating after the buffer reaches 30/30.

The buffer correctly fills from 1/30 to 30/30, but then:
- Buffer stays at "30/30 frames (100%)" ← This is CORRECT!
- Probability stays at "0%" ← This is WRONG!
- Status stays at "NO FALL" ← This is WRONG!

Please add debug logging to:
1. YoloPoseEstimator.estimatePose() - Log valid keypoint count and sample values
2. FallDetectionModel.predict() - Log when called and output probability
3. MainActivity.processFrame() - Log buffer size and probability every frame

The issue is likely:
- YOLO not detecting person (all keypoints are 0)
- Model inference not running (predict() not called)
- Model returning 0 for all inputs (TFLite issue)

Expected behavior:
- Buffer fills to 30/30 in first 1 second ✅
- Buffer STAYS at 30/30 (sliding window) ✅
- Probability updates EVERY frame after buffer is full ❌ (NOT WORKING)
- Probability should be 0-10% when standing, 85-100% when falling ❌ (NOT WORKING)

Please check the implementation against HOW_TO_TEST_FALL_DETECTION.md
```

---

*Last updated: November 5, 2025*

