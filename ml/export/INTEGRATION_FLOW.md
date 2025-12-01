# Fall Detection Integration Flow

Visual guide showing how all components work together.

---

## 🔄 **Complete System Flow**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ANDROID APP                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐                                                   │
│  │   Camera     │                                                   │
│  │   (CameraX)  │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                            │
│         │ Frame (ImageProxy)                                        │
│         ▼                                                            │
│  ┌──────────────────────┐                                          │
│  │  Image Analyzer      │                                          │
│  │  (analyze method)    │                                          │
│  └──────┬───────────────┘                                          │
│         │                                                            │
│         │ Bitmap                                                    │
│         ▼                                                            │
│  ┌──────────────────────┐         ┌─────────────────────┐         │
│  │  Pose Estimation     │         │ DummyKeypoint       │         │
│  │  (YOLO / ML Kit)     │   OR    │ Generator           │         │
│  │  [Future]            │         │ [Testing]           │         │
│  └──────┬───────────────┘         └─────────┬───────────┘         │
│         │                                     │                      │
│         │ Keypoints (34 values)              │                      │
│         └─────────────────┬───────────────────┘                      │
│                           │                                          │
│                           ▼                                          │
│                  ┌─────────────────┐                                │
│                  │ KeypointsBuffer │                                │
│                  │ (30 frames)     │                                │
│                  └────────┬────────┘                                │
│                           │                                          │
│                           │ When full (30 frames)                   │
│                           ▼                                          │
│                  ┌─────────────────┐                                │
│                  │  FallDetector   │                                │
│                  │  (TFLite Model) │                                │
│                  └────────┬────────┘                                │
│                           │                                          │
│                           │ Probability [0, 1]                      │
│                           ▼                                          │
│                  ┌─────────────────┐                                │
│                  │  Decision Logic │                                │
│                  │  (threshold)    │                                │
│                  └────────┬────────┘                                │
│                           │                                          │
│              ┌────────────┴────────────┐                            │
│              │                         │                            │
│              ▼                         ▼                            │
│      ┌──────────────┐         ┌──────────────┐                    │
│      │  Update UI   │         │  Show Alert  │                    │
│      │  (TextView)  │         │  (Dialog)    │                    │
│      └──────────────┘         └──────────────┘                    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 **Data Flow Details**

### **1. Camera Frame → Keypoints**

```
Camera Frame (ImageProxy)
    ↓
Convert to Bitmap (1920×1080 or similar)
    ↓
┌─────────────────────────────────────────┐
│ Option A: YOLO Pose Estimation (Future) │
│ - Resize to 640×640                     │
│ - Normalize to [0, 1]                   │
│ - Run YOLO inference                    │
│ - Extract 17 keypoints × 3 (y, x, conf)│
│ - Filter by confidence > 0.3            │
│ - Output: 34 values (17 × 2)           │
└─────────────────────────────────────────┘
    OR
┌─────────────────────────────────────────┐
│ Option B: Dummy Generator (Testing)     │
│ - Generate random keypoints             │
│ - Simulate person standing              │
│ - Output: 34 values (17 × 2)           │
└─────────────────────────────────────────┘
    ↓
Keypoints: FloatArray(34)
[nose_y, nose_x, left_eye_y, left_eye_x, ...]
All values in [0, 1]
```

### **2. Keypoints → Buffer**

```
Frame 1: [34 values] ──┐
Frame 2: [34 values] ──┤
Frame 3: [34 values] ──┤
...                     ├──→ KeypointsBuffer
Frame 28: [34 values] ──┤    (FIFO queue)
Frame 29: [34 values] ──┤
Frame 30: [34 values] ──┘
    ↓
When buffer is full (30 frames):
Flatten to FloatArray(1020)
[frame1_kp1_y, frame1_kp1_x, ..., frame30_kp17_y, frame30_kp17_x]
```

### **3. Buffer → Fall Detection**

```
Input: FloatArray(1020)
    ↓
Reshape to (1, 30, 34)
    ↓
Convert to ByteBuffer (4080 bytes)
    ↓
┌─────────────────────────────────────────┐
│ TFLite Model (BiLSTM)                   │
│ - BiLSTM(64) → BiLSTM(32)              │
│ - Dense(32) → Dense(1)                 │
│ - Sigmoid activation                    │
└─────────────────────────────────────────┘
    ↓
Output: Float (probability)
Range: [0, 1]
- 0.0 = Definitely not a fall
- 0.5 = Uncertain
- 1.0 = Definitely a fall
```

### **4. Probability → Decision**

```
Probability: 0.0 - 1.0
    ↓
Compare with threshold (0.85)
    ↓
┌─────────────────────────────────────────┐
│ If probability > 0.85:                  │
│   - Status: "FALL DETECTED" (RED)      │
│   - Show alert dialog                   │
│   - Log event                           │
│   - (Future: Send notification/SMS)    │
└─────────────────────────────────────────┘
    OR
┌─────────────────────────────────────────┐
│ If probability ≤ 0.85:                  │
│   - Status: "NO FALL" (GREEN)          │
│   - Update UI only                      │
│   - Continue monitoring                 │
└─────────────────────────────────────────┘
```

---

## 🧪 **Testing Flow (Phase 1-2)**

### **Test Fall Sequence**

```
User clicks "Test Fall" button
    ↓
Clear KeypointsBuffer
    ↓
Generate 30 frames:
┌─────────────────────────────────────────┐
│ Frames 0-10: Normal standing            │
│ - Head: y = 0.2-0.3 (high)             │
│ - Body: y = 0.3-0.5 (mid)              │
│ - Legs: y = 0.5-0.8 (low)              │
│ - Probability: ~10-30%                  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Frames 10-20: Falling                   │
│ - All keypoints: y increases            │
│ - y: 0.3 → 0.8 (moving down)           │
│ - Probability: increases gradually      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Frames 20-30: On ground                 │
│ - All keypoints: y = 0.75-0.8 (low)    │
│ - Same position (stillness)             │
│ - Probability: ~99%                     │
└─────────────────────────────────────────┘
    ↓
Alert: "FALL DETECTED! Probability: 99%"
```

### **Test Normal Sequence**

```
User clicks "Test Normal" button
    ↓
Clear KeypointsBuffer
    ↓
Generate 30 frames:
┌─────────────────────────────────────────┐
│ All frames: Normal standing/walking     │
│ - Head: y = 0.2-0.3 (high)             │
│ - Body: y = 0.3-0.5 (mid)              │
│ - Legs: y = 0.5-0.8 (low)              │
│ - Slight random variation               │
│ - Probability: ~10-30% (consistent)     │
└─────────────────────────────────────────┘
    ↓
No alert (probability stays below threshold)
```

---

## 📱 **Camera Flow (Phase 3)**

### **Real-time Processing**

```
Camera starts
    ↓
Every frame (30 FPS = 33ms per frame):
┌─────────────────────────────────────────┐
│ 1. Capture frame (ImageProxy)           │
│ 2. Convert to Bitmap                    │
│ 3. Extract keypoints (dummy or YOLO)   │
│ 4. Add to buffer                        │
│ 5. If buffer full (30 frames):         │
│    - Run fall detection                 │
│    - Update UI                          │
│    - Check for fall                     │
│ 6. Close ImageProxy                     │
└─────────────────────────────────────────┘
    ↓
Continuous monitoring
    ↓
If fall detected:
    ↓
Show alert + notification
```

---

## 🔢 **Data Sizes**

| Component | Size | Description |
|-----------|------|-------------|
| **Single Frame Keypoints** | 34 floats = 136 bytes | 17 keypoints × 2 coords |
| **30-Frame Buffer** | 1020 floats = 4080 bytes | 30 frames × 34 features |
| **TFLite Model** | 407 KB | BiLSTM model file |
| **Model Input Buffer** | 4080 bytes | ByteBuffer for inference |
| **Model Output Buffer** | 4 bytes | Single float probability |
| **Total Memory** | ~5-10 MB | Runtime memory usage |

---

## ⏱️ **Timing**

| Operation | Time | Notes |
|-----------|------|-------|
| **Camera Frame** | 33ms | 30 FPS |
| **Keypoint Extraction (Dummy)** | <1ms | Random generation |
| **Keypoint Extraction (YOLO)** | 20-50ms | Future implementation |
| **Buffer Add** | <1ms | Array copy |
| **Fall Detection (TFLite)** | 10-20ms | BiLSTM inference |
| **UI Update** | <1ms | TextView update |
| **Total (Dummy)** | ~35ms | Can run at 30 FPS |
| **Total (YOLO)** | ~60ms | Can run at 15-20 FPS |

---

## 🎯 **Key Points**

### **Coordinate System**
```
(0, 0) ────────────────── (0, 1)
  │                          │
  │     Person Standing      │
  │          👤              │
  │         /│\             │
  │        / │ \            │
  │         / \             │
  │        /   \            │
  │                          │
(1, 0) ────────────────── (1, 1)

y-axis: 0 (top) → 1 (bottom)
x-axis: 0 (left) → 1 (right)

Standing person:
- Head: y ≈ 0.2-0.3
- Torso: y ≈ 0.3-0.5
- Legs: y ≈ 0.5-0.8

Fallen person:
- All keypoints: y ≈ 0.7-0.9
```

### **Buffer Behavior**
```
Initial state (empty):
Buffer: []
Size: 0
isFull(): false

After 10 frames:
Buffer: [f1, f2, ..., f10]
Size: 10
isFull(): false

After 30 frames:
Buffer: [f1, f2, ..., f30]
Size: 30
isFull(): true ✅ Run detection!

After 31 frames:
Buffer: [f2, f3, ..., f31]  (f1 removed)
Size: 30
isFull(): true ✅ Run detection!

Sliding window: Always keeps last 30 frames
```

### **Threshold Behavior**
```
Probability: 0.00 - 0.84 → NO FALL (green)
Probability: 0.85 - 1.00 → FALL DETECTED (red + alert)

Examples:
- 0.18 (18%) → NO FALL ✅
- 0.50 (50%) → NO FALL ✅
- 0.84 (84%) → NO FALL ✅
- 0.85 (85%) → FALL DETECTED 🚨
- 0.99 (99%) → FALL DETECTED 🚨
```

---

## 📚 **Class Relationships**

```
MainActivity
    │
    ├── FallDetector (TFLite model)
    │   └── Interpreter (with Flex delegate)
    │
    ├── KeypointsBuffer (30-frame window)
    │   └── List<FloatArray>
    │
    ├── DummyKeypointGenerator (testing)
    │   ├── generateNormalFrame()
    │   ├── generateFallSequence()
    │   └── generateNormalSequence()
    │
    └── ImageAnalyzer (camera processing)
        ├── DummyKeypointAnalyzer (Phase 3)
        └── RealKeypointAnalyzer (Phase 4 - future)
            └── YoloPoseEstimator
```

---

## ✅ **Success Indicators**

### **Phase 1-2 (Testing)**
- ✅ Test fall sequence → Probability increases to ~99%
- ✅ Test fall sequence → Alert shows
- ✅ Test normal sequence → Probability stays ~18%
- ✅ Test normal sequence → No alert

### **Phase 3 (Camera)**
- ✅ Camera preview shows
- ✅ Probability updates continuously
- ✅ No false positives with dummy keypoints
- ✅ UI responsive (no lag)

### **Phase 4 (Real YOLO - Future)**
- ✅ Real keypoints extracted from camera
- ✅ Accurate fall detection from real falls
- ✅ Low false positive rate
- ✅ Fast inference (<50ms)

---

**This flow diagram shows exactly how everything connects!** 🎉

