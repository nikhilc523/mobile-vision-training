# 📊 YOLO to BiLSTM - Data Flow Diagram

**Visual guide showing how data flows from camera to fall detection**

---

## 🎥 **Complete Data Flow**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CAMERA FRAME                                    │
│                     (1920×1080 RGB Bitmap)                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEP 1: PREPROCESS IMAGE                           │
│                                                                          │
│  • Resize to 640×640                                                    │
│  • Convert to RGB float                                                 │
│  • Normalize to [0, 1]                                                  │
│                                                                          │
│  Output: ByteBuffer (1, 640, 640, 3) float32                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   STEP 2: YOLO11-POSE INFERENCE                         │
│                                                                          │
│  Model: yolo11n-pose.tflite (11.3 MB)                                  │
│  Input: (1, 640, 640, 3) float32                                       │
│  Output: (1, 56, 8400) float32                                         │
│  Time: 20-50ms (GPU) or 50-100ms (CPU)                                 │
│                                                                          │
│  Output format: CHW (channels-first)                                    │
│  • 8400 detections                                                      │
│  • 56 features per detection                                           │
│  • Features stored in separate "channels"                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  STEP 3: FIND BEST DETECTION                            │
│                                                                          │
│  • Read feature 4 (object confidence) for all 8400 detections          │
│  • Find detection with highest confidence                              │
│  • Check if confidence > 0.3                                           │
│                                                                          │
│  Example:                                                               │
│    Detection 1234: conf = 0.95 ← BEST!                                │
│    Detection 5678: conf = 0.12                                         │
│    ...                                                                  │
│                                                                          │
│  Output: detectionIdx = 1234                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STEP 4: EXTRACT YOLO KEYPOINTS (CHW FORMAT)                │
│                                                                          │
│  For each keypoint (0-16):                                             │
│    featureBaseIdx = 6 + kptIdx * 3                                     │
│                                                                          │
│    Read x: position = (featureBaseIdx * 8400 + detectionIdx) * 4      │
│    Read y: position = ((featureBaseIdx+1) * 8400 + detectionIdx) * 4  │
│    Read conf: position = ((featureBaseIdx+2) * 8400 + detectionIdx) * 4│
│                                                                          │
│  Example for nose (keypoint 0):                                        │
│    featureBaseIdx = 6                                                   │
│    x at position (6 * 8400 + 1234) * 4 = 210,736 bytes                │
│    y at position (7 * 8400 + 1234) * 4 = 244,336 bytes                │
│    conf at position (8 * 8400 + 1234) * 4 = 277,936 bytes             │
│                                                                          │
│  Output: 17 keypoints × 3 values = 51 values                          │
│    [x0, y0, conf0, x1, y1, conf1, ..., x16, y16]                      │
│    Note: Last keypoint (right_ankle) has no confidence!                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                STEP 5: CONVERT TO MODEL FORMAT                          │
│                                                                          │
│  For each keypoint:                                                     │
│    1. Check confidence > 0.3                                           │
│    2. Normalize: normX = x / 640, normY = y / 640                     │
│    3. SWAP to [y, x] order!                                            │
│    4. Store in output array                                            │
│                                                                          │
│  Example for nose:                                                      │
│    YOLO: x=320.5, y=100.2, conf=0.95                                  │
│    ↓ Normalize                                                          │
│    normX = 320.5 / 640 = 0.501                                        │
│    normY = 100.2 / 640 = 0.156                                        │
│    ↓ Swap to [y, x]                                                    │
│    modelKeypoints[0] = 0.156  (y first!)                              │
│    modelKeypoints[1] = 0.501  (x second!)                             │
│                                                                          │
│  Output: FloatArray(34) - 17 keypoints × 2 coordinates                │
│    [y0, x0, y1, x1, y2, x2, ..., y16, x16]                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  STEP 6: ADD TO SLIDING WINDOW                          │
│                                                                          │
│  Buffer: List<FloatArray> (max 30 frames)                             │
│                                                                          │
│  Frame 1: [y0, x0, y1, x1, ..., y16, x16]  ← Oldest                   │
│  Frame 2: [y0, x0, y1, x1, ..., y16, x16]                             │
│  ...                                                                    │
│  Frame 29: [y0, x0, y1, x1, ..., y16, x16]                            │
│  Frame 30: [y0, x0, y1, x1, ..., y16, x16] ← Newest                   │
│                                                                          │
│  When buffer is full (30 frames):                                      │
│    • Remove oldest frame (frame 1)                                     │
│    • Add newest frame (frame 31)                                       │
│    • Buffer slides forward in time                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 STEP 7: BiLSTM INFERENCE (WHEN FULL)                    │
│                                                                          │
│  Model: fall_detection_model.tflite (407 KB)                          │
│  Input: (1, 30, 34) float32                                            │
│  Output: (1, 1) float32                                                │
│  Time: 5-10ms                                                           │
│                                                                          │
│  Input format:                                                          │
│    • 30 frames (1 second at 30 FPS)                                    │
│    • 34 features per frame (17 keypoints × 2 coordinates)             │
│    • All values normalized to [0, 1]                                   │
│    • Coordinates in [y, x] order                                       │
│                                                                          │
│  Output: Single probability [0, 1]                                     │
│    • 0.0 = definitely NOT a fall                                       │
│    • 1.0 = definitely a fall                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 8: CHECK THRESHOLD                              │
│                                                                          │
│  if (probability > 0.85):                                              │
│    🚨 FALL DETECTED!                                                   │
│    • Show emergency dialog                                             │
│    • Play TTS: "A fall is detected. Are you okay?"                    │
│    • Vibrate phone                                                     │
│  else:                                                                  │
│    ✅ Normal activity                                                  │
│    • Update UI with probability                                        │
│    • Continue monitoring                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 **Data Format at Each Step**

### **Step 1: Camera Frame**
```
Type: Bitmap
Size: 1920×1080 (or camera resolution)
Format: ARGB_8888
Example pixel: Color(r=128, g=64, b=32, a=255)
```

### **Step 2: Preprocessed Image**
```
Type: ByteBuffer
Shape: (1, 640, 640, 3)
Format: RGB float32 [0, 1]
Size: 1 × 640 × 640 × 3 × 4 = 4,915,200 bytes
Example values: [0.502, 0.251, 0.125, ...]
```

### **Step 3: YOLO Output**
```
Type: ByteBuffer
Shape: (1, 56, 8400)
Format: CHW (channels-first) float32
Size: 1 × 56 × 8400 × 4 = 1,881,600 bytes
Layout:
  Position 0-8399:     Feature 0 (bbox_x) for all detections
  Position 8400-16799: Feature 1 (bbox_y) for all detections
  ...
  Position 33600-41999: Feature 4 (obj_conf) for all detections ← Use this!
  ...
  Position 50400-58799: Feature 6 (kpt0_x) for all detections
  Position 58800-67199: Feature 7 (kpt0_y) for all detections
  Position 67200-75599: Feature 8 (kpt0_conf) for all detections
  ...
```

### **Step 4: Best Detection**
```
Type: Int
Value: 0-8399 (detection index)
Example: 1234
Meaning: Detection at index 1234 has highest confidence
```

### **Step 5: YOLO Keypoints**
```
Type: FloatArray(51)
Format: [x, y, conf] × 17 keypoints
Coordinate range: 0-640 pixels
Confidence range: 0-1
Example:
  [320.5, 100.2, 0.95,  // nose
   310.3, 95.1, 0.92,   // left_eye
   330.7, 95.3, 0.91,   // right_eye
   ...]
```

### **Step 6: Model Keypoints**
```
Type: FloatArray(34)
Format: [y, x] × 17 keypoints
Coordinate range: 0-1 normalized
Example:
  [0.156, 0.501,  // nose [y, x]
   0.149, 0.485,  // left_eye [y, x]
   0.149, 0.517,  // right_eye [y, x]
   ...]
```

### **Step 7: Sliding Window Buffer**
```
Type: List<FloatArray>
Size: 30 frames
Each frame: FloatArray(34)
Total size: 30 × 34 = 1,020 values
Example:
  Frame 1:  [0.156, 0.501, 0.149, 0.485, ...]
  Frame 2:  [0.157, 0.502, 0.150, 0.486, ...]
  ...
  Frame 30: [0.165, 0.510, 0.158, 0.494, ...]
```

### **Step 8: BiLSTM Input**
```
Type: ByteBuffer
Shape: (1, 30, 34)
Format: float32
Size: 1 × 30 × 34 × 4 = 4,080 bytes
Layout: Sequential (frame 1, frame 2, ..., frame 30)
```

### **Step 9: BiLSTM Output**
```
Type: ByteBuffer
Shape: (1, 1)
Format: float32
Size: 1 × 1 × 4 = 4 bytes
Value: 0.0 to 1.0 (probability)
Example: 0.0234 (2.34% chance of fall)
```

---

## 🔄 **Coordinate Transformation Example**

### **Scenario: Person standing in center of frame**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CAMERA FRAME                                    │
│                         (640×640 after resize)                          │
│                                                                          │
│  (0,0)                                                        (640,0)   │
│    ┌────────────────────────────────────────────────────────┐          │
│    │                                                          │          │
│    │                         👤                              │          │
│    │                      (320, 100) ← Nose                  │          │
│    │                                                          │          │
│    │                                                          │          │
│    │                                                          │          │
│    │                                                          │          │
│    └────────────────────────────────────────────────────────┘          │
│  (0,640)                                                    (640,640)   │
└─────────────────────────────────────────────────────────────────────────┘

YOLO Output (nose keypoint):
  x = 320.5 pixels (horizontal position)
  y = 100.2 pixels (vertical position)
  conf = 0.95 (95% confident)

Conversion to Model Format:
  1. Normalize:
     normX = 320.5 / 640 = 0.501
     normY = 100.2 / 640 = 0.156
  
  2. Swap to [y, x]:
     modelKeypoints[0] = 0.156  (y first!)
     modelKeypoints[1] = 0.501  (x second!)

Model Input (nose keypoint):
  [0.156, 0.501]
  Meaning: Nose is at 15.6% from top, 50.1% from left (center)
```

---

## 🎯 **Key Takeaways**

### **1. YOLO Output is CHW Format**
- Data is stored in "channels" (features)
- Each feature has 8400 values (one per detection)
- To read: `position = (featureIdx * 8400 + detectionIdx) * 4`

### **2. Coordinates are in Pixels**
- YOLO outputs: 0-640 pixels
- Must normalize: divide by 640
- Result: 0-1 range

### **3. Coordinate Order Matters**
- YOLO: [x, y, conf]
- Model: [y, x] (NO confidence!)
- Must swap when converting!

### **4. Sliding Window is Key**
- Buffer holds 30 frames (1 second)
- Slides forward in time
- Removes oldest, adds newest
- Always 30 frames for inference

### **5. Two Models Work Together**
- YOLO: Extracts keypoints from single frame (20-50ms)
- BiLSTM: Analyzes 30 frames for fall detection (5-10ms)
- Total: ~30-60ms per frame (20-30 FPS)

---

## ✅ **Verification Checklist**

Use this to verify each step is working correctly:

- [ ] **Step 1:** Camera frame is captured (check preview)
- [ ] **Step 2:** Image is preprocessed (check size 640×640)
- [ ] **Step 3:** YOLO inference runs (check time 20-100ms)
- [ ] **Step 4:** Best detection found (check confidence > 0.3)
- [ ] **Step 5:** Keypoints extracted (check 17 keypoints)
- [ ] **Step 6:** Keypoints converted (check [y, x] order, 0-1 range)
- [ ] **Step 7:** Buffer fills up (check 30 frames)
- [ ] **Step 8:** BiLSTM inference runs (check time 5-10ms)
- [ ] **Step 9:** Probability calculated (check 0-1 range)
- [ ] **Step 10:** Threshold checked (check > 0.85 triggers alert)

---

**This diagram shows the complete data flow from camera to fall detection!** 📊

