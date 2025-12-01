# Android Studio Implementation Prompt for Augment AI

Copy and paste this prompt to Augment when working in Android Studio:

---

## 🎯 **Task: Implement Fall Detection TFLite Model in Android App**

I need help implementing a fall detection system in Android using a TensorFlow Lite model. I have a pre-trained BiLSTM model that detects falls from pose keypoints.

---

## 📦 **What I Have**

1. **TFLite Model File:** `fall_detection_model.tflite` (407 KB)
   - Location: `mobile-vision-training/ml/export/fall_detection_model.tflite`
   - Copy this to Android project: `app/src/main/assets/fall_detection_model.tflite`

2. **Complete Documentation:**
   - `mobile-vision-training/ml/export/README.md` - Android integration guide
   - `mobile-vision-training/ml/export/TFLITE_CONVERSION_SUMMARY.md` - Model specs and test results

3. **Model Specifications:**
   - **Input:** (1, 30, 34) float32 array
     - 30 frames = 1 second of video @ 30 FPS
     - 34 features = 17 keypoints × 2 coordinates (y, x)
     - Values normalized to [0, 1]
   - **Output:** (1, 1) float32 array
     - Single probability [0, 1]
     - Threshold: 0.85 (if prob > 0.85 → FALL DETECTED)
   - **Keypoints Format:** COCO format (17 keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)

4. **Test Results (Validated):**
   - Normal activity: 18.13% → NO FALL ✅
   - Simulated fall: 99.47% → FALL DETECTED ✅
   - All zeros: 0.0003% → NO FALL ✅

---

## 🚨 **CRITICAL REQUIREMENTS**

### **1. TensorFlow Lite Dependencies (MUST INCLUDE!)**

The BiLSTM model uses TensorFlow ops that require the **Flex delegate**. You MUST add these dependencies to `app/build.gradle`:

```gradle
dependencies {
    // TensorFlow Lite (standard)
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    
    // TensorFlow Lite Select TF Ops (REQUIRED for BiLSTM!)
    // Without this, the model will fail with "FlexTensorListReserve not supported"
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0'
    
    // TensorFlow Lite GPU (optional, for faster inference)
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
}
```

**⚠️ WARNING:** If you don't include `tensorflow-lite-select-tf-ops`, the app will crash with:
```
RuntimeError: Select TensorFlow op(s), included in the given model, is(are) not supported by this interpreter.
```

### **2. Flex Delegate (MUST CREATE!)**

When creating the TFLite interpreter, you MUST use the Flex delegate:

```kotlin
import org.tensorflow.lite.flex.FlexDelegate

// CORRECT - With Flex delegate
val flexDelegate = FlexDelegate()
val options = Interpreter.Options()
options.addDelegate(flexDelegate)
interpreter = Interpreter(modelFile, options)

// WRONG - Without Flex delegate (will crash!)
interpreter = Interpreter(modelFile)  // ❌ DON'T DO THIS
```

### **3. Input Format (MUST FOLLOW EXACTLY!)**

The model expects a specific input format:

```kotlin
// Input shape: (1, 30, 34)
// - 1 = batch size (always 1)
// - 30 = number of frames (1 second @ 30 FPS)
// - 34 = features per frame (17 keypoints × 2 coordinates)

// Each frame has 17 keypoints in COCO format:
// 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
// 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
// 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
// 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

// Each keypoint: [y, x] (normalized to [0, 1])
// Example for one frame:
val frame = floatArrayOf(
    0.5f, 0.5f,      // nose (y, x)
    0.48f, 0.48f,    // left_eye
    0.48f, 0.52f,    // right_eye
    // ... 14 more keypoints (34 values total)
)

// Total input: 30 frames × 34 features = 1020 float values
val input = FloatArray(30 * 34)
```

---

## 📋 **Implementation Requirements**

Please help me implement the following:

### **1. FallDetector Class**

Create a `FallDetector.kt` class that:
- Loads the TFLite model from assets
- Creates Flex delegate and interpreter
- Provides `detectFall(keypoints: FloatArray): Float` method
- Provides `isFall(probability: Float): Boolean` method with threshold 0.85
- Handles ByteBuffer allocation and conversion
- Properly closes resources

**Reference implementation:** See `mobile-vision-training/ml/export/README.md` lines 80-150

### **2. KeypointsBuffer Class**

Create a `KeypointsBuffer.kt` class that:
- Maintains a sliding window of 30 frames
- Adds new frames and removes old frames (FIFO)
- Provides `isFull(): Boolean` method
- Provides `toFloatArray(): FloatArray` method to flatten buffer
- Thread-safe for real-time video processing

### **3. MainActivity Integration**

Update `MainActivity.kt` to:
- Initialize `FallDetector` in `onCreate()`
- Create `KeypointsBuffer` for 30-frame window
- Process each video frame:
  - Extract keypoints (placeholder for now - will integrate YOLO later)
  - Add to buffer
  - When buffer is full (30 frames), run fall detection
  - Display probability and result
- Show alert dialog when fall is detected
- Clean up resources in `onDestroy()`

### **4. UI Components**

Add to layout:
- TextView to display current probability
- TextView to display detection status (FALL / NO FALL)
- Button to test with sample data
- Alert dialog for fall detection

---

## 🔧 **Technical Specifications**

### **Model Loading**

```kotlin
private fun loadModelFile(context: Context, filename: String): ByteBuffer {
    val assetFileDescriptor = context.assets.openFd(filename)
    val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = assetFileDescriptor.startOffset
    val declaredLength = assetFileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}
```

### **ByteBuffer Preparation**

```kotlin
// Input buffer: 30 frames × 34 features × 4 bytes per float
val inputBuffer = ByteBuffer.allocateDirect(30 * 34 * 4)
inputBuffer.order(ByteOrder.nativeOrder())

for (value in keypoints) {
    inputBuffer.putFloat(value)
}

// Output buffer: 1 float × 4 bytes
val outputBuffer = ByteBuffer.allocateDirect(4)
outputBuffer.order(ByteOrder.nativeOrder())

// Run inference
interpreter.run(inputBuffer, outputBuffer)

// Get result
outputBuffer.rewind()
val probability = outputBuffer.float
```

### **Sliding Window Logic**

```kotlin
class KeypointsBuffer(private val windowSize: Int = 30) {
    private val buffer = mutableListOf<FloatArray>()
    
    fun add(keypoints: FloatArray) {
        require(keypoints.size == 34) { "Each frame must have 34 features" }
        buffer.add(keypoints)
        if (buffer.size > windowSize) {
            buffer.removeAt(0)  // Remove oldest frame
        }
    }
    
    fun isFull(): Boolean = buffer.size == windowSize
    
    fun toFloatArray(): FloatArray {
        val result = FloatArray(windowSize * 34)
        for (i in 0 until windowSize) {
            System.arraycopy(buffer[i], 0, result, i * 34, 34)
        }
        return result
    }
}
```

---

## 🧪 **Testing Requirements**

### **1. Test with Sample Data**

Create a test button that generates sample fall data:

```kotlin
fun generateTestFall(): FloatArray {
    val input = FloatArray(30 * 34)
    
    // Frames 0-10: Normal standing (y ≈ 0.5)
    for (t in 0 until 10) {
        for (i in 0 until 17) {
            input[t * 34 + i * 2] = 0.5f + Random.nextFloat() * 0.1f  // y
            input[t * 34 + i * 2 + 1] = 0.5f + Random.nextFloat() * 0.1f  // x
        }
    }
    
    // Frames 10-20: Falling (y decreasing)
    for (t in 10 until 20) {
        for (i in 0 until 17) {
            input[t * 34 + i * 2] = 0.5f - (t - 10) * 0.05f  // y decreasing
            input[t * 34 + i * 2 + 1] = 0.5f + Random.nextFloat() * 0.1f  // x
        }
    }
    
    // Frames 20-30: On ground (y ≈ 0.0)
    for (t in 20 until 30) {
        for (i in 0 until 17) {
            input[t * 34 + i * 2] = 0.05f + Random.nextFloat() * 0.05f  // y low
            input[t * 34 + i * 2 + 1] = 0.5f + Random.nextFloat() * 0.1f  // x
        }
    }
    
    return input
}
```

**Expected result:** Probability > 0.85 (FALL DETECTED)

### **2. Test with Normal Activity**

```kotlin
fun generateTestNormal(): FloatArray {
    val input = FloatArray(30 * 34)
    
    // All frames: Normal standing/walking (y ≈ 0.5, slight variation)
    for (t in 0 until 30) {
        for (i in 0 until 17) {
            input[t * 34 + i * 2] = 0.5f + Random.nextFloat() * 0.1f  // y
            input[t * 34 + i * 2 + 1] = 0.5f + Random.nextFloat() * 0.1f  // x
        }
    }
    
    return input
}
```

**Expected result:** Probability < 0.85 (NO FALL)

---

## 📊 **Expected Performance**

- **Inference time:** 10-20ms per detection
- **Memory usage:** 5-10 MB
- **CPU usage:** 15-35%
- **Model size:** 407 KB

---

## ⚠️ **Common Pitfalls to Avoid**

1. ❌ **Forgetting Flex delegate** → App will crash
2. ❌ **Wrong input shape** → Model will return garbage
3. ❌ **Not normalizing keypoints** → Model expects [0, 1] range
4. ❌ **Wrong coordinate order** → Model expects [y, x], not [x, y]
5. ❌ **Not using sliding window** → Need 30 frames for detection
6. ❌ **Memory leaks** → Must close interpreter in onDestroy()

---

## 📚 **Reference Documentation**

All documentation is in `mobile-vision-training/ml/export/`:
- `README.md` - Complete Android integration guide
- `TFLITE_CONVERSION_SUMMARY.md` - Model specs and test results
- `test_tflite_model.py` - Python test script showing expected behavior

---

## ✅ **Success Criteria**

The implementation is successful when:
1. ✅ App builds without errors
2. ✅ Model loads successfully with Flex delegate
3. ✅ Test fall data → Probability > 0.85 (FALL DETECTED)
4. ✅ Test normal data → Probability < 0.85 (NO FALL)
5. ✅ UI displays probability and status correctly
6. ✅ Alert dialog shows when fall is detected
7. ✅ No memory leaks or crashes

---

## 🚀 **Implementation Steps**

Please help me implement this in the following order:

1. **First:** Update `build.gradle` with correct dependencies (including Flex ops)
2. **Second:** Create `FallDetector.kt` class with model loading and inference
3. **Third:** Create `KeypointsBuffer.kt` class for sliding window
4. **Fourth:** Update `MainActivity.kt` with fall detection logic
5. **Fifth:** Add UI components and test buttons
6. **Finally:** Test with sample data and verify results

---

## 💡 **Additional Notes**

- The model is already trained and validated (99.42% F1 score)
- YOLO pose estimation will be integrated later (use placeholder keypoints for now)
- Focus on getting the TFLite inference working correctly first
- Follow the exact specifications in the documentation
- Ask if anything is unclear!

---

**Ready to implement! Please start with step 1 (build.gradle dependencies).**

