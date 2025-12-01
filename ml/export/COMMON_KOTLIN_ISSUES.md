# Common Kotlin Issues - Fall Detection Probability Stuck at 0%

## 🔍 Issues I'll Check in Your Code

### 1. YOLO Output Reading (CHW Format)

**WRONG (HWC format):**
```kotlin
// Reading as if data is in HWC format (Height, Width, Channels)
val x = outputBuffer.getFloat(detectionIdx * 56 + kptIdx * 3 + 0)  // ❌ WRONG!
val y = outputBuffer.getFloat(detectionIdx * 56 + kptIdx * 3 + 1)  // ❌ WRONG!
val conf = outputBuffer.getFloat(detectionIdx * 56 + kptIdx * 3 + 2)  // ❌ WRONG!
```

**CORRECT (CHW format):**
```kotlin
// YOLO output is in CHW format (Channels, Height, Width)
val x = outputBuffer.getFloat((kptIdx * 3 + 0) * 8400 + detectionIdx)  // ✅ CORRECT!
val y = outputBuffer.getFloat((kptIdx * 3 + 1) * 8400 + detectionIdx)  // ✅ CORRECT!
val conf = outputBuffer.getFloat((kptIdx * 3 + 2) * 8400 + detectionIdx)  // ✅ CORRECT!
```

---

### 2. Coordinate Order (Model Expects [y, x])

**WRONG ([x, y] order):**
```kotlin
keypoints[kptIdx * 2] = x / 640.0f      // ❌ WRONG! Model expects y first
keypoints[kptIdx * 2 + 1] = y / 640.0f  // ❌ WRONG! Model expects x second
```

**CORRECT ([y, x] order):**
```kotlin
keypoints[kptIdx * 2] = y / 640.0f      // ✅ CORRECT! y first
keypoints[kptIdx * 2 + 1] = x / 640.0f  // ✅ CORRECT! x second
```

---

### 3. Confidence Threshold

**WRONG (no confidence check):**
```kotlin
// Always use keypoints, even if confidence is low
keypoints[kptIdx * 2] = y / 640.0f      // ❌ Might use invalid keypoints
keypoints[kptIdx * 2 + 1] = x / 640.0f
```

**CORRECT (check confidence > 0.3):**
```kotlin
if (conf > 0.3f) {  // ✅ Only use high-confidence keypoints
    keypoints[kptIdx * 2] = y / 640.0f
    keypoints[kptIdx * 2 + 1] = x / 640.0f
}
```

---

### 4. Last Keypoint (No Confidence Value)

**WRONG (tries to read confidence for keypoint 16):**
```kotlin
for (kptIdx in 0 until 17) {
    val conf = outputBuffer.getFloat((kptIdx * 3 + 2) * 8400 + detectionIdx)  // ❌ Crashes for kptIdx=16!
}
```

**CORRECT (handle keypoint 16 separately):**
```kotlin
for (kptIdx in 0 until 16) {  // ✅ Only 0-15 have confidence
    val conf = outputBuffer.getFloat((kptIdx * 3 + 2) * 8400 + detectionIdx)
    if (conf > 0.3f) {
        // ...
    }
}

// Keypoint 16 (right_ankle) has no confidence, always use it
val x16 = outputBuffer.getFloat((16 * 3 + 0) * 8400 + detectionIdx)
val y16 = outputBuffer.getFloat((16 * 3 + 1) * 8400 + detectionIdx)
keypoints[32] = y16 / 640.0f
keypoints[33] = x16 / 640.0f
```

---

### 5. FlexDelegate Initialization

**WRONG (no FlexDelegate):**
```kotlin
val options = Interpreter.Options()
interpreter = Interpreter(model, options)  // ❌ BiLSTM won't work!
```

**CORRECT (FlexDelegate before Interpreter):**
```kotlin
val options = Interpreter.Options()
options.addDelegate(FlexDelegate())  // ✅ Required for BiLSTM!
interpreter = Interpreter(model, options)
```

---

### 6. ByteBuffer Order

**WRONG (wrong byte order):**
```kotlin
val inputBuffer = ByteBuffer.allocateDirect(4 * 1 * 30 * 34)
// Missing: inputBuffer.order(ByteOrder.nativeOrder())  // ❌ WRONG!
```

**CORRECT (native byte order):**
```kotlin
val inputBuffer = ByteBuffer.allocateDirect(4 * 1 * 30 * 34)
inputBuffer.order(ByteOrder.nativeOrder())  // ✅ CORRECT!
```

---

### 7. Buffer Not Rewinding

**WRONG (buffer position not reset):**
```kotlin
inputBuffer.putFloat(value)  // Fill buffer
interpreter.run(inputBuffer, outputBuffer)  // ❌ Buffer position is at end!
```

**CORRECT (rewind before inference):**
```kotlin
inputBuffer.putFloat(value)  // Fill buffer
inputBuffer.rewind()  // ✅ Reset position to 0
interpreter.run(inputBuffer, outputBuffer)
outputBuffer.rewind()  // ✅ Reset output position too
val probability = outputBuffer.getFloat()
```

---

### 8. Model Inference Not Called

**WRONG (condition never met):**
```kotlin
if (frameBuffer.size == 30) {  // ✅ Condition is correct
    val probability = fallDetectionModel.predict(frameBuffer.toTypedArray())
    // But this block never executes! Why?
}
```

**Possible reasons:**
- `frameBuffer` is being cleared somewhere
- `frameBuffer.size` never reaches 30
- Code is in wrong place (not called every frame)

**Fix: Add logging:**
```kotlin
Log.d("Buffer", "Size: ${frameBuffer.size}/30")  // Check if it reaches 30
if (frameBuffer.size == 30) {
    Log.d("Inference", "Running model...")  // Check if this executes
    val probability = fallDetectionModel.predict(frameBuffer.toTypedArray())
    Log.d("Inference", "Probability: $probability")  // Check output
}
```

---

### 9. UI Not Updating

**WRONG (UI update not on main thread):**
```kotlin
// In background thread
tvProbability.text = "Probability: ${probability}%"  // ❌ Crashes or doesn't update!
```

**CORRECT (use runOnUiThread):**
```kotlin
runOnUiThread {
    tvProbability.text = "Probability: ${(probability * 100).toInt()}%"  // ✅ CORRECT!
}
```

---

### 10. Bitmap Preprocessing

**WRONG (bitmap not resized or normalized):**
```kotlin
// Pass original bitmap to YOLO
val keypoints = yoloPoseEstimator.estimatePose(bitmap)  // ❌ Wrong size!
```

**CORRECT (resize to 640x640 and normalize):**
```kotlin
// Resize bitmap to 640x640
val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

// Normalize to [0, 1] in YoloPoseEstimator
val inputBuffer = ByteBuffer.allocateDirect(4 * 640 * 640 * 3)
inputBuffer.order(ByteOrder.nativeOrder())

for (y in 0 until 640) {
    for (x in 0 until 640) {
        val pixel = resizedBitmap.getPixel(x, y)
        inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)  // R
        inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)   // G
        inputBuffer.putFloat((pixel and 0xFF) / 255.0f)           // B
    }
}
```

---

### 11. Detection Index Selection

**WRONG (uses random detection):**
```kotlin
val detectionIdx = 0  // ❌ Might not be a person!
```

**CORRECT (find detection with highest confidence):**
```kotlin
// Find detection with highest confidence (first 4 values are bbox)
var maxConf = 0.0f
var bestDetectionIdx = 0

for (detIdx in 0 until 8400) {
    val conf = outputBuffer.getFloat(4 * 8400 + detIdx)  // Class confidence
    if (conf > maxConf) {
        maxConf = conf
        bestDetectionIdx = detIdx
    }
}

// Use bestDetectionIdx for keypoint extraction
```

---

### 12. Model Input Shape

**WRONG (wrong input shape):**
```kotlin
// Model expects (1, 30, 34) but you provide (30, 34)
val inputArray = Array(30) { FloatArray(34) }  // ❌ Missing batch dimension!
interpreter.run(inputArray, outputArray)
```

**CORRECT (include batch dimension):**
```kotlin
// Model expects (1, 30, 34)
val inputBuffer = ByteBuffer.allocateDirect(4 * 1 * 30 * 34)  // ✅ Batch=1
inputBuffer.order(ByteOrder.nativeOrder())

for (frame in window) {
    for (value in frame) {
        inputBuffer.putFloat(value)
    }
}

inputBuffer.rewind()
interpreter.run(inputBuffer, outputBuffer)
```

---

## 🎯 Debugging Checklist

When I review your code, I'll check:

- [ ] YOLO output reading (CHW vs HWC format)
- [ ] Coordinate order ([y, x] vs [x, y])
- [ ] Confidence threshold (> 0.3)
- [ ] Last keypoint handling (no confidence for keypoint 16)
- [ ] FlexDelegate initialization
- [ ] ByteBuffer byte order
- [ ] Buffer rewinding before inference
- [ ] Model inference condition (frameBuffer.size == 30)
- [ ] UI updates (runOnUiThread)
- [ ] Bitmap preprocessing (resize + normalize)
- [ ] Detection index selection (highest confidence)
- [ ] Model input shape (batch dimension)
- [ ] Logging (to verify each step)

---

## 📝 What to Share

Please provide:

1. **MainActivity.kt** - Main activity with camera processing
2. **YoloPoseEstimator.kt** - YOLO pose estimation class
3. **FallDetectionModel.kt** - BiLSTM model inference class

I'll identify the exact issue and provide a fix! 🚀

---

*Last updated: November 5, 2025*

