# Keypoint Extraction & Fall Detection Integration Guide

This guide explains how to extract keypoints from camera frames and integrate fall detection into your Android app's image analyzer.

---

## 🎯 **Overview**

The fall detection pipeline:
```
Camera Frame → Pose Estimation → Keypoint Extraction → Buffer (30 frames) → Fall Detection → Alert
```

---

## 📋 **Phase 1: Dummy Keypoints (Testing)**

First, implement dummy keypoint generation to test the fall detection logic without YOLO.

### **Step 1: Add Dummy Keypoint Generator**

Create a helper class to generate test keypoints:

```kotlin
// File: DummyKeypointGenerator.kt
package com.example.falldetection

import kotlin.random.Random

object DummyKeypointGenerator {
    
    /**
     * Generate dummy keypoints for one frame (34 values)
     * 17 keypoints × 2 coordinates (y, x)
     * Normalized to [0, 1]
     */
    fun generateNormalFrame(): FloatArray {
        val keypoints = FloatArray(34)
        
        // Simulate person standing upright
        // Head keypoints (nose, eyes, ears) - high y values (0.2-0.3)
        for (i in 0 until 5) {  // First 5 keypoints (nose, eyes, ears)
            keypoints[i * 2] = 0.2f + Random.nextFloat() * 0.1f  // y (high)
            keypoints[i * 2 + 1] = 0.5f + Random.nextFloat() * 0.1f  // x (center)
        }
        
        // Upper body (shoulders, elbows, wrists) - mid y values (0.3-0.5)
        for (i in 5 until 11) {  // Keypoints 5-10
            keypoints[i * 2] = 0.3f + Random.nextFloat() * 0.2f  // y (mid)
            keypoints[i * 2 + 1] = 0.5f + Random.nextFloat() * 0.2f  // x (spread)
        }
        
        // Lower body (hips, knees, ankles) - low y values (0.5-0.8)
        for (i in 11 until 17) {  // Keypoints 11-16
            keypoints[i * 2] = 0.5f + Random.nextFloat() * 0.3f  // y (low)
            keypoints[i * 2 + 1] = 0.5f + Random.nextFloat() * 0.2f  // x (spread)
        }
        
        return keypoints
    }
    
    /**
     * Generate dummy keypoints for a fall sequence (30 frames)
     * Simulates: standing → falling → on ground
     */
    fun generateFallSequence(): List<FloatArray> {
        val sequence = mutableListOf<FloatArray>()
        
        // Frames 0-10: Normal standing
        for (t in 0 until 10) {
            sequence.add(generateNormalFrame())
        }
        
        // Frames 10-20: Falling (y-coordinates increasing = moving down)
        for (t in 10 until 20) {
            val keypoints = FloatArray(34)
            val fallProgress = (t - 10) / 10f  // 0.0 to 1.0
            
            for (i in 0 until 17) {
                // y-coordinate: move from 0.3 to 0.8 (falling down)
                keypoints[i * 2] = 0.3f + fallProgress * 0.5f + Random.nextFloat() * 0.05f
                // x-coordinate: slight horizontal movement
                keypoints[i * 2 + 1] = 0.5f + Random.nextFloat() * 0.1f
            }
            
            sequence.add(keypoints)
        }
        
        // Frames 20-30: On ground (stillness)
        val groundKeypoints = FloatArray(34)
        for (i in 0 until 17) {
            groundKeypoints[i * 2] = 0.75f + Random.nextFloat() * 0.05f  // y (on ground)
            groundKeypoints[i * 2 + 1] = 0.5f + Random.nextFloat() * 0.1f  // x
        }
        
        for (t in 20 until 30) {
            // Same position (stillness)
            sequence.add(groundKeypoints.copyOf())
        }
        
        return sequence
    }
    
    /**
     * Generate dummy keypoints for normal activity sequence (30 frames)
     * Simulates: walking, standing, slight movements
     */
    fun generateNormalSequence(): List<FloatArray> {
        val sequence = mutableListOf<FloatArray>()
        
        for (t in 0 until 30) {
            sequence.add(generateNormalFrame())
        }
        
        return sequence
    }
}
```

---

## 📱 **Phase 2: Integrate into MainActivity**

### **Step 2: Add Test Buttons to UI**

Update `activity_main.xml`:

```xml
<!-- Add these buttons below your existing UI -->
<Button
    android:id="@+id/btnTestFallSequence"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Test Fall Sequence"
    android:backgroundTint="@android:color/holo_red_dark" />

<Button
    android:id="@+id/btnTestNormalSequence"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Test Normal Sequence"
    android:backgroundTint="@android:color/holo_green_dark" />

<Button
    android:id="@+id/btnStartCamera"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Start Camera (Dummy Keypoints)" />
```

### **Step 3: Add Test Button Handlers**

In `MainActivity.kt`:

```kotlin
// Add to onCreate() after initializing fallDetector
findViewById<Button>(R.id.btnTestFallSequence).setOnClickListener {
    testFallSequence()
}

findViewById<Button>(R.id.btnTestNormalSequence).setOnClickListener {
    testNormalSequence()
}

findViewById<Button>(R.id.btnStartCamera).setOnClickListener {
    startCameraWithDummyKeypoints()
}

// Add these methods
private fun testFallSequence() {
    Log.d(TAG, "Testing fall sequence...")
    
    // Clear buffer
    keypointsBuffer.clear()
    
    // Generate fall sequence
    val sequence = DummyKeypointGenerator.generateFallSequence()
    
    // Process each frame
    sequence.forEachIndexed { index, keypoints ->
        processFrame(keypoints, index)
        Thread.sleep(33)  // Simulate 30 FPS (33ms per frame)
    }
}

private fun testNormalSequence() {
    Log.d(TAG, "Testing normal sequence...")
    
    // Clear buffer
    keypointsBuffer.clear()
    
    // Generate normal sequence
    val sequence = DummyKeypointGenerator.generateNormalSequence()
    
    // Process each frame
    sequence.forEachIndexed { index, keypoints ->
        processFrame(keypoints, index)
        Thread.sleep(33)  // Simulate 30 FPS
    }
}

private fun processFrame(keypoints: FloatArray, frameIndex: Int) {
    // Add to buffer
    keypointsBuffer.add(keypoints)
    
    // Run detection when buffer is full
    if (keypointsBuffer.isFull()) {
        val input = keypointsBuffer.toFloatArray()
        val probability = fallDetector.detectFall(input)
        
        // Update UI
        runOnUiThread {
            updateUI(probability, frameIndex)
        }
        
        // Check for fall
        if (fallDetector.isFall(probability)) {
            runOnUiThread {
                showFallAlert(probability)
            }
        }
    }
}

private fun updateUI(probability: Float, frameIndex: Int) {
    findViewById<TextView>(R.id.tvProbability).text = 
        "Frame $frameIndex: ${(probability * 100).toInt()}%"
    
    val status = if (probability > 0.85f) "FALL DETECTED" else "NO FALL"
    val color = if (probability > 0.85f) 
        android.graphics.Color.RED else android.graphics.Color.GREEN
    
    findViewById<TextView>(R.id.tvStatus).apply {
        text = status
        setTextColor(color)
    }
}
```

---

## 📷 **Phase 3: Camera Integration with Dummy Keypoints**

### **Step 4: Add Camera Analyzer**

```kotlin
// Add to MainActivity.kt
private fun startCameraWithDummyKeypoints() {
    Log.d(TAG, "Starting camera with dummy keypoints...")
    
    // TODO: Add camera permission check
    
    val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
    
    cameraProviderFuture.addListener({
        val cameraProvider = cameraProviderFuture.get()
        
        // Preview
        val preview = Preview.Builder().build()
        preview.setSurfaceProvider(
            findViewById<PreviewView>(R.id.previewView).surfaceProvider
        )
        
        // Image analyzer
        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(
                    ContextCompat.getMainExecutor(this),
                    DummyKeypointAnalyzer()
                )
            }
        
        // Bind to lifecycle
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        
        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )
        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
        }
        
    }, ContextCompat.getMainExecutor(this))
}

// Image analyzer with dummy keypoints
private inner class DummyKeypointAnalyzer : ImageAnalysis.Analyzer {
    private var frameCount = 0
    
    override fun analyze(image: ImageProxy) {
        // Generate dummy keypoints for this frame
        val keypoints = DummyKeypointGenerator.generateNormalFrame()
        
        // Process frame
        processFrame(keypoints, frameCount++)
        
        // Close image
        image.close()
    }
}
```

---

## 🤖 **Phase 4: Real YOLO Pose Estimation (Future)**

When you're ready to integrate real pose estimation, replace dummy keypoints with YOLO.

### **Option 1: TensorFlow Lite YOLO (Recommended)**

```kotlin
// File: YoloPoseEstimator.kt
package com.example.falldetection

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class YoloPoseEstimator(context: Context) {
    private val interpreter: Interpreter
    
    init {
        // Load YOLO11-Pose TFLite model
        val modelFile = loadModelFile(context, "yolo11n-pose.tflite")
        interpreter = Interpreter(modelFile)
    }
    
    /**
     * Extract keypoints from image
     * Returns: FloatArray of size 34 (17 keypoints × 2 coordinates)
     */
    fun extractKeypoints(bitmap: Bitmap): FloatArray {
        // Preprocess image
        val inputBuffer = preprocessImage(bitmap)
        
        // Run inference
        val outputBuffer = ByteBuffer.allocateDirect(17 * 3 * 4)  // 17 keypoints × 3 (y, x, conf)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        interpreter.run(inputBuffer, outputBuffer)
        
        // Parse output
        outputBuffer.rewind()
        val keypoints = FloatArray(34)
        
        for (i in 0 until 17) {
            val y = outputBuffer.float
            val x = outputBuffer.float
            val conf = outputBuffer.float
            
            // Only use keypoints with confidence > 0.3
            if (conf > 0.3f) {
                keypoints[i * 2] = y
                keypoints[i * 2 + 1] = x
            } else {
                keypoints[i * 2] = 0f
                keypoints[i * 2 + 1] = 0f
            }
        }
        
        return keypoints
    }
    
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize to 640×640 (YOLO input size)
        val resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        
        // Convert to ByteBuffer
        val buffer = ByteBuffer.allocateDirect(640 * 640 * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())
        
        val pixels = IntArray(640 * 640)
        resized.getPixels(pixels, 0, 640, 0, 0, 640, 640)
        
        for (pixel in pixels) {
            // Normalize to [0, 1]
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)  // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)   // G
            buffer.putFloat((pixel and 0xFF) / 255f)           // B
        }
        
        return buffer
    }
    
    fun close() {
        interpreter.close()
    }
}
```

### **Option 2: ML Kit Pose Detection (Easier, but less accurate)**

```kotlin
// Add to build.gradle
implementation 'com.google.mlkit:pose-detection:18.0.0-beta3'

// File: MLKitPoseEstimator.kt
package com.example.falldetection

import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions

class MLKitPoseEstimator {
    private val options = PoseDetectorOptions.Builder()
        .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
        .build()
    
    private val poseDetector = PoseDetection.getClient(options)
    
    fun extractKeypoints(bitmap: Bitmap, callback: (FloatArray) -> Unit) {
        val image = InputImage.fromBitmap(bitmap, 0)
        
        poseDetector.process(image)
            .addOnSuccessListener { pose ->
                val keypoints = FloatArray(34)
                
                // Map ML Kit landmarks to COCO format
                val landmarks = pose.allPoseLandmarks
                
                // TODO: Map ML Kit landmarks to COCO keypoints
                // ML Kit has different keypoint order than COCO
                
                callback(keypoints)
            }
            .addOnFailureListener { e ->
                // Return zeros on failure
                callback(FloatArray(34))
            }
    }
}
```

---

## 🔄 **Phase 5: Replace Dummy with Real Keypoints**

When YOLO is ready, update the analyzer:

```kotlin
private inner class RealKeypointAnalyzer : ImageAnalysis.Analyzer {
    private val yoloPose = YoloPoseEstimator(this@MainActivity)
    private var frameCount = 0
    
    override fun analyze(image: ImageProxy) {
        // Convert ImageProxy to Bitmap
        val bitmap = image.toBitmap()
        
        // Extract real keypoints using YOLO
        val keypoints = yoloPose.extractKeypoints(bitmap)
        
        // Process frame
        processFrame(keypoints, frameCount++)
        
        // Close image
        image.close()
    }
}

// Helper extension function
fun ImageProxy.toBitmap(): Bitmap {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}
```

---

## ✅ **Implementation Checklist**

### **Phase 1: Dummy Keypoints (Do This First!)**
- [ ] Create `DummyKeypointGenerator.kt`
- [ ] Add `generateNormalFrame()` method
- [ ] Add `generateFallSequence()` method
- [ ] Add `generateNormalSequence()` method
- [ ] Test: Generate 30 frames, verify shape (34 values each)

### **Phase 2: Test Integration**
- [ ] Add test buttons to UI
- [ ] Add `testFallSequence()` method
- [ ] Add `testNormalSequence()` method
- [ ] Add `processFrame()` method
- [ ] Add `updateUI()` method
- [ ] Test: Click "Test Fall" → Should detect fall
- [ ] Test: Click "Test Normal" → Should not detect fall

### **Phase 3: Camera with Dummy**
- [ ] Add camera permissions to manifest
- [ ] Add `PreviewView` to layout
- [ ] Add `startCameraWithDummyKeypoints()` method
- [ ] Create `DummyKeypointAnalyzer` class
- [ ] Test: Camera shows preview, generates dummy keypoints

### **Phase 4: Real YOLO (Later)**
- [ ] Get YOLO11-Pose TFLite model
- [ ] Create `YoloPoseEstimator.kt`
- [ ] Add image preprocessing
- [ ] Add keypoint extraction
- [ ] Test: Extract keypoints from test image

### **Phase 5: Replace Dummy**
- [ ] Create `RealKeypointAnalyzer`
- [ ] Replace dummy generator with YOLO
- [ ] Test: Real-time fall detection from camera

---

## 📊 **Expected Results**

| Test | Expected Behavior |
|------|-------------------|
| **Test Fall Sequence** | Probability increases from ~20% to ~99% over 30 frames, then shows FALL DETECTED alert |
| **Test Normal Sequence** | Probability stays low (~10-30%), no alert |
| **Camera (Dummy)** | Continuous low probability (~10-30%), no false positives |
| **Camera (Real YOLO)** | Accurate detection of real falls from camera feed |

---

## 🚀 **Next Steps**

1. **Start with Phase 1** - Implement dummy keypoint generator
2. **Test thoroughly** - Make sure fall sequence triggers detection
3. **Add camera** - Test with dummy keypoints from camera
4. **Integrate YOLO** - Replace dummy with real pose estimation
5. **Optimize** - Tune threshold, add FSM filter, improve UI

---

**This guide gives you a complete path from testing to production!** 🎉

