package edu.cs663.falldetect.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import edu.cs663.falldetect.util.Log

/**
 * YOLO11-Pose estimator for real-time pose detection.
 * Extracts 17 COCO keypoints from camera frames.
 */
class YoloPoseEstimator(context: Context) {
    
    companion object {
        private const val TAG = "YoloPoseEstimator"
        private const val MODEL_NAME = "yolo11n-pose.tflite"
        private const val INPUT_SIZE = 640  // Model is fixed at 640×640
        private const val NUM_KEYPOINTS = 17  // COCO format
        private const val CONFIDENCE_THRESHOLD = 0.3f
        private const val NUM_DETECTIONS = 8400  // Number of detections at 640×640
        private const val VALUES_PER_DETECTION = 56  // 4 bbox + 1 obj + 1 class + 17*3 keypoints
    }
    
    private val interpreter: Interpreter

    init {
        Log.d("Initializing YOLO11-Pose estimator (640×640, CPU optimized)...", tag = TAG)

        // Load model
        val modelFile = loadModelFile(context, MODEL_NAME)

        // Create interpreter options
        val options = Interpreter.Options()

        // Use CPU with 4 threads (GPU delegate has compatibility issues)
        options.setNumThreads(4)
        Log.d("Using CPU (4 threads)", tag = TAG)

        interpreter = Interpreter(modelFile, options)

        Log.d("✅ YOLO11-Pose model loaded successfully", tag = TAG)
        logModelInfo()
    }
    
    /**
     * Extract keypoints from camera frame.
     * 
     * @param bitmap Camera frame (any size)
     * @return FloatArray of size 34 (17 keypoints × 2 coordinates [y, x])
     */
    fun extractKeypoints(bitmap: Bitmap): FloatArray {
        try {
            // Preprocess image
            val inputBuffer = preprocessImage(bitmap)
            
            // Prepare output buffer
            val outputBuffer = prepareOutputBuffer()
            
            // Run inference
            val startTime = System.currentTimeMillis()
            interpreter.run(inputBuffer, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - startTime
            
            // Parse output
            val keypoints = parseOutput(outputBuffer)
            
            Log.d("Inference time: ${inferenceTime}ms", tag = TAG)
            
            return keypoints
            
        } catch (e: Exception) {
            Log.e("Error extracting keypoints: ${e.message}", e, tag = TAG)
            // Return zeros on error
            return FloatArray(34)
        }
    }
    
    /**
     * Preprocess image for YOLO input.
     * 
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
     * Prepare output buffer for YOLO.
     * 
     * YOLO11-Pose output: (1, 56, 8400)
     * - 56 = 4 (bbox) + 1 (obj conf) + 1 (class conf) + 17×3 (keypoints)
     * - 8400 = number of detections
     */
    private fun prepareOutputBuffer(): ByteBuffer {
        val outputSize = 1 * VALUES_PER_DETECTION * NUM_DETECTIONS * 4  // float32
        val buffer = ByteBuffer.allocateDirect(outputSize)
        buffer.order(ByteOrder.nativeOrder())
        return buffer
    }
    
    /**
     * Parse YOLO output to extract keypoints.
     * 
     * YOLO output format (per detection):
     * - Index 0-3: Bounding box [x, y, w, h]
     * - Index 4: Object confidence
     * - Index 5: Class confidence
     * - Index 6-55: 17 keypoints × 3 values [x, y, conf]
     * 
     * @return FloatArray(34) - 17 keypoints × 2 (y, x) normalized to [0, 1]
     */
    private fun parseOutput(outputBuffer: ByteBuffer): FloatArray {
        outputBuffer.rewind()

        val keypoints = FloatArray(34)

        // Output format: [1, 56, 8400] - CHW format (channels-first)!
        // Buffer layout: Feature 0 for all 8400 detections, then Feature 1 for all 8400 detections, etc.
        // To read feature F for detection D: position = (F * 8400 + D) * 4

        // Step 1: Find detection with highest confidence
        var maxConfidence = 0f
        var bestDetectionIdx = -1

        // Object confidence is at feature index 4
        for (detIdx in 0 until NUM_DETECTIONS) {
            val position = (4 * NUM_DETECTIONS + detIdx) * 4
            outputBuffer.position(position)
            val confidence = outputBuffer.float

            if (confidence > maxConfidence) {
                maxConfidence = confidence
                bestDetectionIdx = detIdx
            }
        }

        // Check if we found a person
        if (maxConfidence < CONFIDENCE_THRESHOLD || bestDetectionIdx < 0) {
            Log.d("⚠️ No person detected (max conf: ${"%.2f".format(maxConfidence)})", tag = TAG)
            return keypoints  // Return zeros
        }

        Log.d("✅ Person detected (conf: ${"%.2f".format(maxConfidence)})", tag = TAG)

        // Step 2: Extract keypoints from best detection
        // Keypoints are at feature indices 6-55 (50 values)
        // Format: [x, y, conf] × 16 keypoints + [x, y] for last keypoint (no conf!)

        var validKeypointCount = 0  // Track how many keypoints are valid

        for (kptIdx in 0 until NUM_KEYPOINTS) {
            val featureBaseIdx = 6 + kptIdx * 3

            // Read x coordinate (feature index: featureBaseIdx)
            val xPosition = (featureBaseIdx * NUM_DETECTIONS + bestDetectionIdx) * 4
            outputBuffer.position(xPosition)
            val x = outputBuffer.float

            // Read y coordinate (feature index: featureBaseIdx + 1)
            val yPosition = ((featureBaseIdx + 1) * NUM_DETECTIONS + bestDetectionIdx) * 4
            outputBuffer.position(yPosition)
            val y = outputBuffer.float

            // Read confidence (skip for last keypoint - it doesn't have conf!)
            val conf = if (kptIdx < 16) {
                val confPosition = ((featureBaseIdx + 2) * NUM_DETECTIONS + bestDetectionIdx) * 4
                outputBuffer.position(confPosition)
                outputBuffer.float
            } else {
                1.0f  // Assume visible for last keypoint
            }

            // Debug: Log first keypoint raw values
            if (kptIdx == 0) {
                Log.d("Raw YOLO nose: x=$x, y=$y, conf=$conf", tag = TAG)
            }

            // Convert to model format
            // CRITICAL FIX: YOLO outputs pixels (0-640), NOT normalized [0, 1]!
            // Must divide by 640.0f to normalize!
            if (conf > CONFIDENCE_THRESHOLD) {
                // Normalize from pixel space (0-640) to [0, 1]
                val normX = (x / 640.0f).coerceIn(0f, 1f)  // ✅ FIXED: Divide by 640!
                val normY = (y / 640.0f).coerceIn(0f, 1f)  // ✅ FIXED: Divide by 640!

                // CRITICAL: Store as [y, x] to match training format!
                keypoints[kptIdx * 2] = normY      // y first!
                keypoints[kptIdx * 2 + 1] = normX  // x second!
                
                validKeypointCount++
            } else {
                // Low confidence → use 0.0
                keypoints[kptIdx * 2] = 0f
                keypoints[kptIdx * 2 + 1] = 0f
            }
        }

        // Debug: Log valid keypoint count
        Log.d("Valid keypoints: $validKeypointCount / 17", tag = TAG)

        // Debug: Log first 3 keypoints (nose, left_eye, right_eye)
        Log.d("Keypoints sample: nose=[${keypoints[0]}, ${keypoints[1]}], left_eye=[${keypoints[2]}, ${keypoints[3]}], right_eye=[${keypoints[4]}, ${keypoints[5]}]", tag = TAG)

        // Debug: Count non-zero keypoints
        val nonZeroCount = keypoints.count { it > 0.0f }
        Log.d("Non-zero keypoint values: $nonZeroCount / 34", tag = TAG)

        return keypoints
    }
    
    /**
     * Load TFLite model from assets.
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
     * Log model input/output info.
     */
    private fun logModelInfo() {
        val inputShape = interpreter.getInputTensor(0).shape()
        val outputShape = interpreter.getOutputTensor(0).shape()
        
        Log.d("Input shape: ${inputShape.contentToString()}", tag = TAG)
        Log.d("Output shape: ${outputShape.contentToString()}", tag = TAG)
    }
    
    /**
     * Clean up resources.
     */
    fun close() {
        interpreter.close()
        Log.d("YoloPoseEstimator closed", tag = TAG)
    }
}

