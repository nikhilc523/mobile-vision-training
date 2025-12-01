# YOLO11-Pose Integration Guide for Android

**Replace dummy keypoints with real YOLO11-Pose estimation**

---

## 🎯 **Overview**

This guide shows how to integrate YOLO11-Pose for real-time pose estimation in your Android fall detection app.

**What you'll do:**
1. Get YOLO11-Pose TFLite model
2. Create YoloPoseEstimator class
3. Replace DummyKeypointGenerator with real pose estimation
4. Test with real camera feed

---

## 📦 **Step 1: Get YOLO11-Pose TFLite Model**

### **Option A: Convert from PyTorch (Recommended)**

On your computer (not Android):

```bash
# Install ultralytics
pip install ultralytics

# Create conversion script
cat > convert_yolo_to_tflite.py << 'EOF'
from ultralytics import YOLO

# Load YOLO11-Pose model
model = YOLO('yolo11n-pose.pt')

# Export to TFLite
model.export(
    format='tflite',
    imgsz=640,
    int8=False,  # Use float32 for better accuracy
    nms=True,    # Include NMS in model
)

print("✅ YOLO11-Pose TFLite model created!")
print("📦 File: yolo11n-pose_saved_model/yolo11n-pose_float32.tflite")
EOF

# Run conversion
python convert_yolo_to_tflite.py
```

**Output:** `yolo11n-pose_float32.tflite` (~6 MB)

### **Option B: Download Pre-converted Model**

```bash
# Download from Ultralytics
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n-pose.tflite
```

### **Copy to Android Project**

```bash
# Copy to Android assets
cp yolo11n-pose_float32.tflite /path/to/android/app/src/main/assets/yolo11n-pose.tflite
```

---

## 🔧 **Step 2: Add Dependencies**

Update `app/build.gradle`:

```gradle
dependencies {
    // Existing dependencies
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
    
    // NEW: Add these for YOLO
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-metadata:0.4.4'
    
    // Camera
    implementation "androidx.camera:camera-core:1.3.0"
    implementation "androidx.camera:camera-camera2:1.3.0"
    implementation "androidx.camera:camera-lifecycle:1.3.0"
    implementation "androidx.camera:camera-view:1.3.0"
}
```

---

## 🤖 **Step 3: Create YoloPoseEstimator Class**

Create new file: `YoloPoseEstimator.kt`

```kotlin
package com.example.falldetection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class YoloPoseEstimator(context: Context) {
    
    companion object {
        private const val TAG = "YoloPoseEstimator"
        private const val MODEL_NAME = "yolo11n-pose.tflite"
        private const val INPUT_SIZE = 640  // YOLO11 input size
        private const val NUM_KEYPOINTS = 17  // COCO format
        private const val CONFIDENCE_THRESHOLD = 0.3f
    }
    
    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?
    
    init {
        // Load model
        val modelFile = loadModelFile(context, MODEL_NAME)
        
        // Try to use GPU if available
        val compatList = CompatibilityList()
        val options = Interpreter.Options()
        
        if (compatList.isDelegateSupportedOnThisDevice) {
            gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
            options.addDelegate(gpuDelegate)
            Log.d(TAG, "✅ GPU delegate enabled")
        } else {
            gpuDelegate = null
            Log.d(TAG, "⚠️ GPU not available, using CPU")
        }
        
        // Set number of threads for CPU
        options.setNumThreads(4)
        
        interpreter = Interpreter(modelFile, options)
        
        Log.d(TAG, "✅ YOLO11-Pose model loaded successfully")
        logModelInfo()
    }
    
    /**
     * Extract keypoints from camera frame
     * Returns: FloatArray of size 34 (17 keypoints × 2 coordinates [y, x])
     */
    fun extractKeypoints(bitmap: Bitmap): FloatArray {
        try {
            // Preprocess image
            val inputBuffer = preprocessImage(bitmap)
            
            // Prepare output buffers
            val outputBuffer = prepareOutputBuffer()
            
            // Run inference
            val startTime = System.currentTimeMillis()
            interpreter.run(inputBuffer, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - startTime
            
            // Parse output
            val keypoints = parseOutput(outputBuffer, bitmap.width, bitmap.height)
            
            Log.d(TAG, "Inference time: ${inferenceTime}ms")
            
            return keypoints
            
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting keypoints", e)
            // Return zeros on error
            return FloatArray(34)
        }
    }
    
    /**
     * Preprocess image for YOLO input
     * Input: Bitmap (any size)
     * Output: ByteBuffer (1, 640, 640, 3) normalized to [0, 1]
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize to 640×640
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        
        // Create ByteBuffer
        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())
        
        // Convert to float and normalize
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        
        for (pixel in pixels) {
            // Extract RGB and normalize to [0, 1]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
        
        return buffer
    }
    
    /**
     * Prepare output buffer for YOLO
     * YOLO11-Pose output: (1, 56, 8400)
     * - 56 = 4 (bbox) + 1 (obj conf) + 1 (class conf) + 17×3 (keypoints)
     * - 8400 = number of detections
     */
    private fun prepareOutputBuffer(): ByteBuffer {
        val outputSize = 1 * 56 * 8400 * 4  // float32
        val buffer = ByteBuffer.allocateDirect(outputSize)
        buffer.order(ByteOrder.nativeOrder())
        return buffer
    }
    
    /**
     * Parse YOLO output to extract keypoints
     * Returns: FloatArray(34) - 17 keypoints × 2 (y, x)
     */
    private fun parseOutput(
        outputBuffer: ByteBuffer,
        originalWidth: Int,
        originalHeight: Int
    ): FloatArray {
        outputBuffer.rewind()
        
        val keypoints = FloatArray(34)
        var maxConfidence = 0f
        var bestDetection: FloatArray? = null
        
        // Find detection with highest confidence
        for (i in 0 until 8400) {
            // Skip to confidence score (index 4)
            outputBuffer.position(i * 56 * 4 + 4 * 4)
            val objConf = outputBuffer.float
            val classConf = outputBuffer.float
            val confidence = objConf * classConf
            
            if (confidence > maxConfidence && confidence > CONFIDENCE_THRESHOLD) {
                maxConfidence = confidence
                
                // Extract keypoints (17 keypoints × 3 values)
                val kpts = FloatArray(17 * 3)
                outputBuffer.position(i * 56 * 4 + 6 * 4)  // Skip to keypoints
                for (j in 0 until 17 * 3) {
                    kpts[j] = outputBuffer.float
                }
                bestDetection = kpts
            }
        }
        
        // Process best detection
        if (bestDetection != null) {
            for (i in 0 until 17) {
                val x = bestDetection[i * 3]      // x coordinate
                val y = bestDetection[i * 3 + 1]  // y coordinate
                val conf = bestDetection[i * 3 + 2]  // confidence
                
                if (conf > CONFIDENCE_THRESHOLD) {
                    // Normalize to [0, 1]
                    val normX = x / INPUT_SIZE
                    val normY = y / INPUT_SIZE
                    
                    // Store as [y, x] (match training format)
                    keypoints[i * 2] = normY
                    keypoints[i * 2 + 1] = normX
                } else {
                    // Low confidence → set to 0
                    keypoints[i * 2] = 0f
                    keypoints[i * 2 + 1] = 0f
                }
            }
            
            Log.d(TAG, "✅ Person detected (conf: $maxConfidence)")
        } else {
            Log.d(TAG, "⚠️ No person detected")
            // Return zeros
        }
        
        return keypoints
    }
    
    /**
     * Load TFLite model from assets
     */
    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Log model input/output info
     */
    private fun logModelInfo() {
        val inputShape = interpreter.getInputTensor(0).shape()
        val outputShape = interpreter.getOutputTensor(0).shape()
        
        Log.d(TAG, "Input shape: ${inputShape.contentToString()}")
        Log.d(TAG, "Output shape: ${outputShape.contentToString()}")
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        Log.d(TAG, "YoloPoseEstimator closed")
    }
}
```

---

## 🔄 **Step 4: Replace Dummy with YOLO**

Update `MainActivity.kt`:

### **4.1: Add YOLO Instance**

```kotlin
class MainActivity : AppCompatActivity() {
    
    private lateinit var fallDetector: FallDetector
    private lateinit var keypointsBuffer: KeypointsBuffer
    
    // NEW: Add YOLO pose estimator
    private lateinit var yoloPose: YoloPoseEstimator
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize components
        fallDetector = FallDetector(this)
        keypointsBuffer = KeypointsBuffer()
        
        // NEW: Initialize YOLO
        yoloPose = YoloPoseEstimator(this)
        
        // ... rest of onCreate
    }
    
    override fun onDestroy() {
        super.onDestroy()
        fallDetector.close()
        
        // NEW: Close YOLO
        yoloPose.close()
    }
}
```

### **4.2: Update Image Analyzer**

Replace `DummyKeypointAnalyzer` with `RealKeypointAnalyzer`:

```kotlin
// OLD: DummyKeypointAnalyzer
private inner class DummyKeypointAnalyzer : ImageAnalysis.Analyzer {
    private var frameCount = 0
    
    override fun analyze(image: ImageProxy) {
        // Generate dummy keypoints
        val keypoints = DummyKeypointGenerator.generateNormalFrame()
        processFrame(keypoints, frameCount++)
        image.close()
    }
}

// NEW: RealKeypointAnalyzer
private inner class RealKeypointAnalyzer : ImageAnalysis.Analyzer {
    private var frameCount = 0
    
    override fun analyze(image: ImageProxy) {
        try {
            // Convert ImageProxy to Bitmap
            val bitmap = image.toBitmap()
            
            // Extract real keypoints using YOLO
            val keypoints = yoloPose.extractKeypoints(bitmap)
            
            // Process frame
            processFrame(keypoints, frameCount++)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing frame", e)
        } finally {
            image.close()
        }
    }
}

// Helper: Convert ImageProxy to Bitmap
private fun ImageProxy.toBitmap(): Bitmap {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}
```

### **4.3: Update Camera Start Method**

```kotlin
private fun startCamera() {
    Log.d(TAG, "Starting camera with YOLO pose estimation...")
    
    val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
    
    cameraProviderFuture.addListener({
        val cameraProvider = cameraProviderFuture.get()
        
        // Preview
        val preview = Preview.Builder().build()
        preview.setSurfaceProvider(
            findViewById<PreviewView>(R.id.previewView).surfaceProvider
        )
        
        // Image analyzer with YOLO
        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(
                    ContextCompat.getMainExecutor(this),
                    RealKeypointAnalyzer()  // Use YOLO instead of dummy
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
            
            Log.d(TAG, "✅ Camera started with YOLO")
            
        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
        }
        
    }, ContextCompat.getMainExecutor(this))
}
```

---

## 🧪 **Step 5: Test YOLO Integration**

### **5.1: Add Test Button**

Update `activity_main.xml`:

```xml
<Button
    android:id="@+id/btnStartYoloCamera"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Start Camera (YOLO)"
    android:backgroundTint="@android:color/holo_blue_dark" />
```

### **5.2: Add Button Handler**

```kotlin
findViewById<Button>(R.id.btnStartYoloCamera).setOnClickListener {
    startCamera()  // Now uses YOLO
}
```

### **5.3: Test Steps**

1. **Build and run app**
2. **Click "Start Camera (YOLO)"**
3. **Point camera at yourself**
4. **Check logs:**
   ```
   YoloPoseEstimator: ✅ GPU delegate enabled
   YoloPoseEstimator: ✅ YOLO11-Pose model loaded successfully
   YoloPoseEstimator: Input shape: [1, 640, 640, 3]
   YoloPoseEstimator: Output shape: [1, 56, 8400]
   YoloPoseEstimator: Inference time: 45ms
   YoloPoseEstimator: ✅ Person detected (conf: 0.87)
   ```
5. **Check UI:**
   - Probability should be low (~10-30%) when standing
   - Probability should increase when you simulate a fall

---

## 📊 **Expected Performance**

| Metric | Value | Notes |
|--------|-------|-------|
| **Inference Time (GPU)** | 20-50ms | ~20-30 FPS |
| **Inference Time (CPU)** | 50-100ms | ~10-20 FPS |
| **Model Size** | ~6 MB | YOLO11n-pose |
| **Keypoint Confidence** | 90-95% | Much better than MoveNet |
| **Detection Range** | 1-10 meters | Works at various distances |
| **Memory Usage** | 50-100 MB | Includes GPU buffers |

---

## ⚠️ **Common Issues & Solutions**

### **Issue 1: Model not found**
```
Error: yolo11n-pose.tflite not found
```
**Solution:** Make sure model is in `app/src/main/assets/`

### **Issue 2: Slow inference**
```
Inference time: 200ms (too slow!)
```
**Solution:** 
- Check if GPU delegate is enabled
- Reduce input size (try 320×320 instead of 640×640)
- Use YOLO11n (nano) instead of YOLO11s/m/l

### **Issue 3: No person detected**
```
⚠️ No person detected
```
**Solution:**
- Check camera is pointing at person
- Ensure good lighting
- Lower CONFIDENCE_THRESHOLD (try 0.2 instead of 0.3)

### **Issue 4: Wrong keypoint format**
```
Wrong probabilities from fall detector
```
**Solution:**
- Verify keypoints are [y, x], not [x, y]
- Verify values are normalized to [0, 1]
- Check COCO keypoint order matches training

---

## ✅ **Verification Checklist**

- [ ] YOLO model copied to assets folder
- [ ] Dependencies added to build.gradle
- [ ] YoloPoseEstimator.kt created
- [ ] RealKeypointAnalyzer created
- [ ] Camera uses YOLO instead of dummy
- [ ] App builds successfully
- [ ] Camera shows preview
- [ ] Logs show "Person detected"
- [ ] Inference time < 100ms
- [ ] Keypoints extracted correctly (34 values)
- [ ] Fall detection works with real keypoints
- [ ] No crashes or memory leaks

---

## 🎯 **Success Criteria**

Your YOLO integration is successful when:

1. ✅ Camera shows preview
2. ✅ Logs show "Person detected" when you're in frame
3. ✅ Inference time < 100ms (preferably < 50ms with GPU)
4. ✅ Probability stays low (~10-30%) when standing
5. ✅ Probability increases when you simulate a fall
6. ✅ Alert triggers when fall detected
7. ✅ No false positives during normal activity

---

## 🚀 **Next Steps**

After YOLO integration works:

1. ⏳ **Optimize performance** - Tune input size, threads
2. ⏳ **Add FSM filter** - Reduce false positives
3. ⏳ **Add notification system** - SMS/call emergency contacts
4. ⏳ **Add fall history** - Log all detected falls
5. ⏳ **Test with real users** - Elderly people, various scenarios
6. ⏳ **Deploy to production** - Publish to Play Store

---

**You're ready to integrate real YOLO pose estimation!** 🎉

