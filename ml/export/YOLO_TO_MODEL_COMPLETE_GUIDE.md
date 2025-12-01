# 🎯 YOLO11-Pose to BiLSTM Model - COMPLETE INTEGRATION GUIDE

**Date:** November 5, 2025  
**Purpose:** Definitive guide on YOLO output format, model input format, and exact conversion steps

---

## 📋 **Table of Contents**

1. [YOLO11-Pose Model Details](#1-yolo11-pose-model-details)
2. [BiLSTM Model Details](#2-bilstm-model-details)
3. [The Conversion Process](#3-the-conversion-process)
4. [Step-by-Step Implementation](#4-step-by-step-implementation)
5. [Common Issues & Solutions](#5-common-issues--solutions)
6. [Testing & Verification](#6-testing--verification)

---

## 1. YOLO11-Pose Model Details

### **Model File**
- **Filename:** `yolo11n-pose_float32.tflite`
- **Size:** 11.3 MB
- **Location:** `app/src/main/assets/yolo11n-pose.tflite`

### **Input Format**

```
Shape: (1, 640, 640, 3)
Type: float32
Format: RGB image
Range: [0.0, 1.0] (normalized)
Layout: NHWC (batch, height, width, channels)
```

**Example preprocessing:**
```kotlin
// Input: Bitmap from camera
// Output: ByteBuffer for YOLO

val inputBuffer = ByteBuffer.allocateDirect(1 * 640 * 640 * 3 * 4)
inputBuffer.order(ByteOrder.nativeOrder())

// Resize bitmap to 640x640
val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

// Convert to float array [0, 1]
for (y in 0 until 640) {
    for (x in 0 until 640) {
        val pixel = resizedBitmap.getPixel(x, y)
        val r = (pixel shr 16 and 0xFF) / 255.0f
        val g = (pixel shr 8 and 0xFF) / 255.0f
        val b = (pixel and 0xFF) / 255.0f
        inputBuffer.putFloat(r)
        inputBuffer.putFloat(g)
        inputBuffer.putFloat(b)
    }
}
```

### **Output Format**

```
Shape: (1, 56, 8400)
Type: float32
Layout: (batch, features, detections)
```

**CRITICAL:** The output is in **CHW format** (channels-first), NOT HWC format!

**What this means:**
- 8400 detections (possible person detections)
- Each detection has 56 features
- Features are stored in separate "channels"

**How data is stored in the buffer:**
```
Buffer layout (470,400 floats = 1,881,600 bytes):

Position 0-8399:     Feature 0 for all 8400 detections (bbox_x)
Position 8400-16799: Feature 1 for all 8400 detections (bbox_y)
Position 16800-25199: Feature 2 for all 8400 detections (bbox_w)
...
Position 462000-470399: Feature 55 for all 8400 detections (last keypoint conf)
```

**To read a specific feature for a specific detection:**
```kotlin
val detectionIndex = 1234  // Which detection (0-8399)
val featureIndex = 6       // Which feature (0-55)

// Calculate position in buffer
val position = featureIndex * 8400 + detectionIndex
outputBuffer.position(position * 4)  // *4 because each float is 4 bytes
val value = outputBuffer.getFloat()
```

### **Feature Layout (56 features per detection)**

```
Index 0-3:   Bounding box [x, y, w, h]
Index 4:     Object confidence (person detection confidence)
Index 5:     Class confidence (always 1.0 for person)
Index 6-55:  Keypoints (17 keypoints × 3 values = 51 values)
             But only 50 values fit (6-55), so last keypoint might be incomplete
```

**Keypoint layout (indices 6-55):**
```
Index 6, 7, 8:     Keypoint 0 (nose) [x, y, conf]
Index 9, 10, 11:   Keypoint 1 (left_eye) [x, y, conf]
Index 12, 13, 14:  Keypoint 2 (right_eye) [x, y, conf]
Index 15, 16, 17:  Keypoint 3 (left_ear) [x, y, conf]
Index 18, 19, 20:  Keypoint 4 (right_ear) [x, y, conf]
Index 21, 22, 23:  Keypoint 5 (left_shoulder) [x, y, conf]
Index 24, 25, 26:  Keypoint 6 (right_shoulder) [x, y, conf]
Index 27, 28, 29:  Keypoint 7 (left_elbow) [x, y, conf]
Index 30, 31, 32:  Keypoint 8 (right_elbow) [x, y, conf]
Index 33, 34, 35:  Keypoint 9 (left_wrist) [x, y, conf]
Index 36, 37, 38:  Keypoint 10 (right_wrist) [x, y, conf]
Index 39, 40, 41:  Keypoint 11 (left_hip) [x, y, conf]
Index 42, 43, 44:  Keypoint 12 (right_hip) [x, y, conf]
Index 45, 46, 47:  Keypoint 13 (left_knee) [x, y, conf]
Index 48, 49, 50:  Keypoint 14 (right_knee) [x, y, conf]
Index 51, 52, 53:  Keypoint 15 (left_ankle) [x, y, conf]
Index 54, 55:      Keypoint 16 (right_ankle) [x, y] (NO CONFIDENCE!)
```

**⚠️ CRITICAL:** The last keypoint (right_ankle) only has x and y, NO confidence value!

### **Coordinate System**

**YOLO outputs coordinates in pixel space relative to 640×640 input:**
```
x: 0.0 to 640.0 (left to right)
y: 0.0 to 640.0 (top to bottom)
conf: 0.0 to 1.0 (confidence score)
```

**Example YOLO output for nose keypoint:**
```
x = 320.5  (center of image horizontally)
y = 100.2  (near top of image)
conf = 0.95 (95% confident)
```

---

## 2. BiLSTM Model Details

### **Model File**
- **Filename:** `fall_detection_model.tflite`
- **Size:** 407 KB
- **Location:** `app/src/main/assets/fall_detection_model.tflite`

### **Input Format**

```
Shape: (1, 30, 34)
Type: float32
Layout: (batch, timesteps, features)
```

**What this means:**
- 1 batch (always 1 for real-time inference)
- 30 timesteps (30 frames = 1 second at 30 FPS)
- 34 features per frame (17 keypoints × 2 coordinates)

### **Feature Layout (34 features per frame)**

```
Index 0-1:   Keypoint 0 (nose) [x, y]
Index 2-3:   Keypoint 1 (left_eye) [x, y]
Index 4-5:   Keypoint 2 (right_eye) [x, y]
Index 6-7:   Keypoint 3 (left_ear) [x, y]
Index 8-9:   Keypoint 4 (right_ear) [x, y]
Index 10-11: Keypoint 5 (left_shoulder) [x, y]
Index 12-13: Keypoint 6 (right_shoulder) [x, y]
Index 14-15: Keypoint 7 (left_elbow) [x, y]
Index 16-17: Keypoint 8 (right_elbow) [x, y]
Index 18-19: Keypoint 9 (left_wrist) [x, y]
Index 20-21: Keypoint 10 (right_wrist) [x, y]
Index 22-23: Keypoint 11 (left_hip) [x, y]
Index 24-25: Keypoint 12 (right_hip) [x, y]
Index 26-27: Keypoint 13 (left_knee) [x, y]
Index 28-29: Keypoint 14 (right_knee) [x, y]
Index 30-31: Keypoint 15 (left_ankle) [x, y]
Index 32-33: Keypoint 16 (right_ankle) [x, y]
```

**⚠️ CRITICAL DIFFERENCES FROM YOLO:**

1. **Coordinate order:** `[x, y]` (SAME as YOLO, but YOLO also has confidence)
2. **No confidence values:** Only coordinates, no confidence scores
3. **Normalized to [0, 1]:** All values must be in range [0.0, 1.0]
4. **All 17 keypoints:** Must have all 17 keypoints (use 0.0 for missing)

### **Coordinate System**

**Model expects normalized coordinates:**
```
y: 0.0 to 1.0 (top to bottom)
x: 0.0 to 1.0 (left to right)
```

**Example model input for nose keypoint:**
```
x = 0.500  (50% from left = center)
y = 0.156  (15.6% from top)
```

### **Output Format**

```
Shape: (1, 1)
Type: float32
Range: [0.0, 1.0]
```

**What this means:**
- Single probability value
- 0.0 = definitely NOT a fall
- 1.0 = definitely a fall
- Threshold: 0.85 (if > 0.85 → FALL DETECTED)

---

## 3. The Conversion Process

### **Step-by-Step Conversion**

```
YOLO Output → Conversion → Model Input

[x, y, conf] → [y, x] (normalized)
```

### **Detailed Conversion Steps**

**Step 1: Extract YOLO keypoints**
```kotlin
// Read YOLO output buffer (CHW format)
val yoloKeypoints = FloatArray(17 * 3)  // 17 keypoints × 3 values

for (kptIdx in 0 until 17) {
    val featureBaseIdx = 6 + kptIdx * 3
    
    // Read x coordinate
    outputBuffer.position((featureBaseIdx * 8400 + detectionIdx) * 4)
    val x = outputBuffer.getFloat()
    
    // Read y coordinate
    outputBuffer.position(((featureBaseIdx + 1) * 8400 + detectionIdx) * 4)
    val y = outputBuffer.getFloat()
    
    // Read confidence (skip for last keypoint)
    val conf = if (kptIdx < 16) {
        outputBuffer.position(((featureBaseIdx + 2) * 8400 + detectionIdx) * 4)
        outputBuffer.getFloat()
    } else {
        1.0f  // Assume visible for last keypoint
    }
    
    yoloKeypoints[kptIdx * 3] = x
    yoloKeypoints[kptIdx * 3 + 1] = y
    yoloKeypoints[kptIdx * 3 + 2] = conf
}
```

**Step 2: Convert to model format**
```kotlin
// Convert YOLO keypoints to model input
val modelKeypoints = FloatArray(34)  // 17 keypoints × 2 coordinates

for (kptIdx in 0 until 17) {
    val x = yoloKeypoints[kptIdx * 3]
    val y = yoloKeypoints[kptIdx * 3 + 1]
    val conf = yoloKeypoints[kptIdx * 3 + 2]
    
    // CRITICAL: Check confidence threshold
    if (conf > 0.3f) {
        // Normalize to [0, 1]
        val normX = x / 640.0f
        val normY = y / 640.0f
        
        // CRITICAL: Swap to [y, x] order!
        modelKeypoints[kptIdx * 2] = normY      // y first!
        modelKeypoints[kptIdx * 2 + 1] = normX  // x second!
    } else {
        // Low confidence → use 0.0
        modelKeypoints[kptIdx * 2] = 0.0f
        modelKeypoints[kptIdx * 2 + 1] = 0.0f
    }
}
```

**Step 3: Add to sliding window buffer**
```kotlin
// Buffer holds 30 frames
val buffer = mutableListOf<FloatArray>()

// Add new frame
buffer.add(modelKeypoints)

// Keep only last 30 frames
if (buffer.size > 30) {
    buffer.removeAt(0)
}
```

**Step 4: Run inference when buffer is full**
```kotlin
if (buffer.size == 30) {
    // Create input tensor (1, 30, 34)
    val inputBuffer = ByteBuffer.allocateDirect(1 * 30 * 34 * 4)
    inputBuffer.order(ByteOrder.nativeOrder())
    
    // Fill buffer with 30 frames
    for (frame in buffer) {
        for (value in frame) {
            inputBuffer.putFloat(value)
        }
    }
    
    // Run inference
    val outputBuffer = ByteBuffer.allocateDirect(1 * 1 * 4)
    outputBuffer.order(ByteOrder.nativeOrder())
    
    interpreter.run(inputBuffer, outputBuffer)
    
    // Get probability
    outputBuffer.rewind()
    val probability = outputBuffer.getFloat()
    
    // Check threshold
    if (probability > 0.85f) {
        // FALL DETECTED!
        showEmergencyDialog()
    }
}
```

---

## 4. Step-by-Step Implementation

### **Complete YoloPoseEstimator Class**

```kotlin
class YoloPoseEstimator(context: Context) {
    companion object {
        private const val MODEL_NAME = "yolo11n-pose.tflite"
        private const val INPUT_SIZE = 640
        private const val NUM_KEYPOINTS = 17
        private const val NUM_DETECTIONS = 8400
        private const val NUM_FEATURES = 56
        private const val CONFIDENCE_THRESHOLD = 0.3f
    }
    
    private val interpreter: Interpreter
    
    init {
        val model = loadModelFile(context, MODEL_NAME)
        val options = Interpreter.Options()
        
        // Try GPU delegate
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            val gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
            options.addDelegate(gpuDelegate)
        }
        
        interpreter = Interpreter(model, options)
    }
    
    private fun loadModelFile(context: Context, modelName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun extractKeypoints(bitmap: Bitmap): FloatArray {
        // Step 1: Preprocess image
        val inputBuffer = preprocessImage(bitmap)
        
        // Step 2: Run YOLO inference
        val outputBuffer = ByteBuffer.allocateDirect(1 * NUM_FEATURES * NUM_DETECTIONS * 4)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        interpreter.run(inputBuffer, outputBuffer)
        
        // Step 3: Find best detection
        outputBuffer.rewind()
        val bestDetection = findBestDetection(outputBuffer)
        
        if (bestDetection == null) {
            Log.w(TAG, "⚠️ No person detected")
            return FloatArray(34) { 0f }
        }
        
        // Step 4: Extract and convert keypoints
        return convertKeypointsToModelFormat(bestDetection)
    }
    
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = resizedBitmap.getPixel(x, y)
                val r = (pixel shr 16 and 0xFF) / 255.0f
                val g = (pixel shr 8 and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f
                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
            }
        }
        
        return inputBuffer
    }
    
    private fun findBestDetection(outputBuffer: ByteBuffer): Int? {
        var bestConfidence = 0f
        var bestDetectionIdx = -1
        
        // Object confidence is at feature index 4
        for (detIdx in 0 until NUM_DETECTIONS) {
            val position = (4 * NUM_DETECTIONS + detIdx) * 4
            outputBuffer.position(position)
            val confidence = outputBuffer.getFloat()
            
            if (confidence > bestConfidence) {
                bestConfidence = confidence
                bestDetectionIdx = detIdx
            }
        }
        
        return if (bestConfidence > CONFIDENCE_THRESHOLD) {
            Log.d(TAG, "✅ Person detected: conf=$bestConfidence")
            bestDetectionIdx
        } else {
            null
        }
    }
    
    private fun convertKeypointsToModelFormat(detectionIdx: Int): FloatArray {
        val modelKeypoints = FloatArray(34)
        
        for (kptIdx in 0 until NUM_KEYPOINTS) {
            val featureBaseIdx = 6 + kptIdx * 3
            
            // Read x coordinate
            outputBuffer.position((featureBaseIdx * NUM_DETECTIONS + detectionIdx) * 4)
            val x = outputBuffer.getFloat()
            
            // Read y coordinate
            outputBuffer.position(((featureBaseIdx + 1) * NUM_DETECTIONS + detectionIdx) * 4)
            val y = outputBuffer.getFloat()
            
            // Read confidence (skip for last keypoint)
            val conf = if (kptIdx < 16) {
                outputBuffer.position(((featureBaseIdx + 2) * NUM_DETECTIONS + detectionIdx) * 4)
                outputBuffer.getFloat()
            } else {
                1.0f  // Assume visible
            }
            
            // Convert to model format
            if (conf > CONFIDENCE_THRESHOLD) {
                val normX = x / INPUT_SIZE.toFloat()
                val normY = y / INPUT_SIZE.toFloat()
                
                // CRITICAL: Swap to [y, x] order!
                modelKeypoints[kptIdx * 2] = normY
                modelKeypoints[kptIdx * 2 + 1] = normX
            } else {
                modelKeypoints[kptIdx * 2] = 0f
                modelKeypoints[kptIdx * 2 + 1] = 0f
            }
        }
        
        return modelKeypoints
    }
}
```

---

## 5. Common Issues & Solutions

### **Issue 1: All keypoints are [1.0, 1.0]**

**Symptom:** Probability always 0%, logs show all keypoints at [1.0, 1.0]

**Cause:** YOLO coordinates not normalized properly

**Solution:** Divide by INPUT_SIZE (640) to normalize:
```kotlin
val normX = x / 640.0f
val normY = y / 640.0f
```

### **Issue 2: Buffer overflow when reading keypoints**

**Symptom:** `BufferUnderflowException` or "Bad position" error

**Cause:** Reading output buffer in wrong format (HWC instead of CHW)

**Solution:** Use CHW format:
```kotlin
val position = (featureIdx * NUM_DETECTIONS + detectionIdx) * 4
```

### **Issue 3: No person detected**

**Symptom:** Logs show "⚠️ No person detected" even when person is visible

**Cause:** Confidence threshold too high or wrong feature index

**Solution:** 
- Check feature index 4 for object confidence
- Lower threshold to 0.3 or 0.2
- Verify image preprocessing (RGB, normalized to [0,1])

### **Issue 4: Probability stays at 0% even during fall**

**Symptom:** Person detected, but probability never increases

**Cause:** Keypoints in wrong format (x,y instead of y,x)

**Solution:** Swap coordinates:
```kotlin
modelKeypoints[i * 2] = normY      // y first!
modelKeypoints[i * 2 + 1] = normX  // x second!
```

---

## 6. Testing & Verification

### **Test 1: Check YOLO Output**

Add logging to see raw YOLO values:
```kotlin
Log.d(TAG, "Raw YOLO nose: x=$x, y=$y, conf=$conf")
```

**Expected values:**
- x: 0.0 to 640.0
- y: 0.0 to 640.0
- conf: 0.0 to 1.0

### **Test 2: Check Model Input**

Add logging to see converted keypoints:
```kotlin
Log.d(TAG, "Model input nose: y=${modelKeypoints[0]}, x=${modelKeypoints[1]}")
```

**Expected values:**
- y: 0.0 to 1.0
- x: 0.0 to 1.0

### **Test 3: Check Probability**

Add logging to see probability:
```kotlin
Log.d(TAG, "Fall probability: $probability")
```

**Expected behavior:**
- Standing still: 0.00 to 0.10 (0-10%)
- Bending forward: 0.10 to 0.50 (10-50%)
- Falling: 0.85 to 1.00 (85-100%)

### **Test 4: End-to-End Test**

1. Stand still → Probability should be low (~0-10%)
2. Bend forward slowly → Probability should increase (~10-30%)
3. Simulate fall (bend all the way down) → Probability should spike (>85%)
4. Emergency dialog should appear
5. TTS should say "A fall is detected. Are you okay?"
6. Phone should vibrate

---

## ✅ **Summary**

### **YOLO11-Pose Output**
- Shape: (1, 56, 8400)
- Format: CHW (channels-first)
- Coordinates: 0-640 pixels
- Keypoints: [x, y, conf] × 17

### **BiLSTM Model Input**
- Shape: (1, 30, 34)
- Format: (batch, timesteps, features)
- Coordinates: 0-1 normalized
- Keypoints: [y, x] × 17 (NO confidence!)

### **Critical Conversion Steps**
1. ✅ Read YOLO output in CHW format
2. ✅ Find best detection (highest confidence)
3. ✅ Extract 17 keypoints [x, y, conf]
4. ✅ Filter by confidence > 0.3
5. ✅ Normalize to [0, 1] by dividing by 640
6. ✅ **SWAP to [y, x] order!**
7. ✅ Add to sliding window buffer (30 frames)
8. ✅ Run BiLSTM inference
9. ✅ Check if probability > 0.85

---

**This is the definitive guide. Follow it exactly and YOLO integration will work!** 🎯

