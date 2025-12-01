package edu.cs663.falldetect.ml

import android.content.Context
import edu.cs663.falldetect.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Fall detection using TensorFlow Lite BiLSTM model.
 * 
 * Model Specifications:
 * - Input: (1, 30, 34) float32 - 30 frames × 34 features (17 keypoints × 2 coords [y,x])
 * - Output: (1, 1) float32 - Probability [0, 1]
 * - Threshold: 0.85 (if prob > 0.85 → FALL DETECTED)
 * 
 * CRITICAL: This model requires the Flex delegate because it uses BiLSTM layers
 * with TensorFlow ops (FlexTensorListReserve, FlexTensorListSetItem, FlexTensorListStack).
 * 
 * Dependencies required in build.gradle:
 * - implementation 'org.tensorflow:tensorflow-lite:2.15.0'
 * - implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.15.0'  // REQUIRED!
 * - implementation 'org.tensorflow:tensorflow-lite-gpu:2.15.0'  // Optional
 */
class FallDetector(context: Context) {
    
    private val interpreter: Interpreter
    private val flexDelegate: FlexDelegate
    
    companion object {
        private const val TAG = "FallDetector"
        private const val MODEL_FILENAME = "fall_detection_model.tflite"
        private const val INPUT_SIZE = 30 * 34  // 30 frames × 34 features
        private const val OUTPUT_SIZE = 1
        private const val DEFAULT_THRESHOLD = 0.85f
        
        // COCO keypoint indices (17 keypoints)
        const val KEYPOINT_NOSE = 0
        const val KEYPOINT_LEFT_EYE = 1
        const val KEYPOINT_RIGHT_EYE = 2
        const val KEYPOINT_LEFT_EAR = 3
        const val KEYPOINT_RIGHT_EAR = 4
        const val KEYPOINT_LEFT_SHOULDER = 5
        const val KEYPOINT_RIGHT_SHOULDER = 6
        const val KEYPOINT_LEFT_ELBOW = 7
        const val KEYPOINT_RIGHT_ELBOW = 8
        const val KEYPOINT_LEFT_WRIST = 9
        const val KEYPOINT_RIGHT_WRIST = 10
        const val KEYPOINT_LEFT_HIP = 11
        const val KEYPOINT_RIGHT_HIP = 12
        const val KEYPOINT_LEFT_KNEE = 13
        const val KEYPOINT_RIGHT_KNEE = 14
        const val KEYPOINT_LEFT_ANKLE = 15
        const val KEYPOINT_RIGHT_ANKLE = 16
    }
    
    init {
        Log.i("Initializing FallDetector...", tag = TAG)
        
        try {
            // Load model from assets
            val modelFile = loadModelFile(context, MODEL_FILENAME)
            Log.i("Model file loaded: ${modelFile.capacity()} bytes", tag = TAG)
            
            // CRITICAL: Create Flex delegate for BiLSTM support
            flexDelegate = FlexDelegate()
            Log.i("Flex delegate created", tag = TAG)
            
            // Create interpreter options
            val options = Interpreter.Options()
            options.addDelegate(flexDelegate)
            options.setNumThreads(4)  // Use 4 CPU threads for better performance
            
            // Create interpreter
            interpreter = Interpreter(modelFile, options)
            Log.i("Interpreter created successfully", tag = TAG)
            
            // Log input/output tensor info
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            Log.i("Input tensor: shape=${inputTensor.shape().contentToString()}, dtype=${inputTensor.dataType()}", tag = TAG)
            Log.i("Output tensor: shape=${outputTensor.shape().contentToString()}, dtype=${outputTensor.dataType()}", tag = TAG)
            
            Log.i("✅ FallDetector initialized successfully", tag = TAG)
        } catch (e: Exception) {
            Log.e("Failed to initialize FallDetector", e, tag = TAG)
            throw RuntimeException("Failed to initialize FallDetector: ${e.message}", e)
        }
    }
    
    /**
     * Load TFLite model file from assets.
     */
    private fun loadModelFile(context: Context, filename: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Detect fall from 30 frames of keypoints.
     * 
     * @param keypoints FloatArray of size 1020 (30 frames × 34 features)
     *                  Each frame has 17 keypoints × 2 coordinates [y, x]
     *                  Values must be normalized to [0, 1]
     * @return Probability [0, 1] where higher values indicate fall
     * @throws IllegalArgumentException if input size is incorrect
     */
    fun detectFall(keypoints: FloatArray): Float {
        require(keypoints.size == INPUT_SIZE) {
            "Input must be 30 frames × 34 features = $INPUT_SIZE values, got ${keypoints.size}"
        }
        
        try {
            // Prepare input buffer
            // Model expects (1, 30, 34) = 1 batch × 30 frames × 34 features
            // Total: 1 × 30 × 34 × 4 bytes per float = 4080 bytes
            val inputBuffer = ByteBuffer.allocateDirect(1 * 30 * 34 * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            
            // Fill input buffer
            for (value in keypoints) {
                inputBuffer.putFloat(value)
            }
            
            // CRITICAL: Rewind buffer before inference!
            inputBuffer.rewind()
            
            // Prepare output buffer
            // Model outputs (1, 1) = 1 batch × 1 value
            // Total: 1 × 1 × 4 bytes per float = 4 bytes
            val outputBuffer = ByteBuffer.allocateDirect(1 * 1 * 4)
            outputBuffer.order(ByteOrder.nativeOrder())
            
            // Run inference
            val startTime = System.currentTimeMillis()
            interpreter.run(inputBuffer, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - startTime
            
            // Get probability
            outputBuffer.rewind()
            val probability = outputBuffer.float
            
            Log.d("Inference completed in ${inferenceTime}ms, probability: ${"%.4f".format(probability)}", tag = TAG)
            
            return probability
        } catch (e: Exception) {
            Log.e("Failed to run inference", e, tag = TAG)
            throw RuntimeException("Failed to run inference: ${e.message}", e)
        }
    }
    
    /**
     * Check if probability indicates a fall.
     * 
     * @param probability Fall probability [0, 1]
     * @param threshold Threshold for fall detection (default: 0.85)
     * @return true if fall detected, false otherwise
     */
    fun isFall(probability: Float, threshold: Float = DEFAULT_THRESHOLD): Boolean {
        return probability > threshold
    }
    
    /**
     * Get fall detection result with detailed information.
     * 
     * @param keypoints FloatArray of size 1020 (30 frames × 34 features)
     * @param threshold Threshold for fall detection (default: 0.85)
     * @return FallDetectionResult with probability and detection status
     */
    fun detectFallWithResult(keypoints: FloatArray, threshold: Float = DEFAULT_THRESHOLD): FallDetectionResult {
        val probability = detectFall(keypoints)
        val isFall = isFall(probability, threshold)
        return FallDetectionResult(probability, isFall, threshold)
    }
    
    /**
     * Close the interpreter and release resources.
     * MUST be called when done using the detector (e.g., in onDestroy()).
     */
    fun close() {
        try {
            interpreter.close()
            flexDelegate.close()
            Log.i("FallDetector closed", tag = TAG)
        } catch (e: Exception) {
            Log.e("Error closing FallDetector", e, tag = TAG)
        }
    }
}

/**
 * Result of fall detection.
 * 
 * @property probability Fall probability [0, 1]
 * @property isFall true if fall detected (probability > threshold)
 * @property threshold Threshold used for detection
 */
data class FallDetectionResult(
    val probability: Float,
    val isFall: Boolean,
    val threshold: Float
) {
    /**
     * Get probability as percentage string.
     */
    fun getProbabilityPercent(): String {
        return "${(probability * 100).toInt()}%"
    }
    
    /**
     * Get status string.
     */
    fun getStatusString(): String {
        return if (isFall) "FALL DETECTED" else "NO FALL"
    }
    
    override fun toString(): String {
        return "FallDetectionResult(probability=${getProbabilityPercent()}, status=${getStatusString()}, threshold=${(threshold * 100).toInt()}%)"
    }
}

