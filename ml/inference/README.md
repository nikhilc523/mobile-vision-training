# Real-Time Fall Detection Inference - Phase 3.0

Real-time fall detection system using trained BiLSTM model and MoveNet pose estimation.

## Features

- ✅ **Real-time inference** on webcam or video files
- ✅ **90+ FPS** on CPU (11ms per frame)
- ✅ **Three detection modes**: Balanced, Safety, and Precision
- ✅ **Visual overlay**: Pose skeleton, probability bar, and fall alerts
- ✅ **Logging**: JSON logs with fall events and statistics
- ✅ **Video recording**: Save annotated videos with detections

## Architecture

```
Video/Webcam → MoveNet Pose → Feature Extraction → Ring Buffer (60 frames) → BiLSTM → Fall Detection
```

### Pipeline Components

1. **MoveNet Lightning**: Extracts 17 keypoints per frame (COCO format)
2. **Feature Extractor**: Computes 14 temporal features from keypoints
3. **Ring Buffer**: Maintains sliding window of 60 frames
4. **BiLSTM Model**: Predicts fall probability from feature sequence
5. **Threshold**: Classifies as fall/non-fall based on selected mode

### 14 Features Extracted

| # | Feature | Description |
|---|---------|-------------|
| 0 | torso_angle | Angle of torso relative to vertical |
| 1 | hip_height | Height of hips in frame (1 - avg_hip_y) |
| 2 | vertical_velocity | Change in hip height over time |
| 3 | motion_magnitude | Average displacement of all keypoints |
| 4 | shoulder_symmetry | Absolute difference in shoulder y-coordinates |
| 5 | knee_angle | Maximum knee angle (hip-knee-ankle) |
| 6 | head_hip_distance | Vertical distance between nose and hips |
| 7 | elbow_angle | Maximum elbow angle (shoulder-elbow-wrist) |
| 8 | body_aspect_ratio | Height/width ratio of bounding box |
| 9 | centroid_velocity | Speed of body centroid |
| 10 | vertical_acceleration | Change in vertical velocity |
| 11 | angular_velocity | Change in torso angle |
| 12 | stillness_ratio | Proportion of low-motion frames |
| 13 | pose_stability | Variance of torso angle |

## Usage

### Basic Usage

```bash
# Run on video file (balanced mode)
python -m ml.inference.run_fall_detection --video data/test/trailfall.mp4 --mode balanced

# Run on webcam (safety mode)
python -m ml.inference.run_fall_detection --camera 0 --mode safety

# Run with video recording and logging
python -m ml.inference.run_fall_detection --video path/to/video.mp4 --save-video --save-log
```

### Detection Modes

| Mode | Threshold | F1 | Precision | Recall | Use Case |
|------|-----------|----|-----------| -------|----------|
| **balanced** | 0.55 | 0.7456 | 0.7701 | 0.7226 | General-purpose (recommended) |
| **safety** | 0.40 | 0.6582 | 0.5221 | 0.8904 | Safety-critical (elderly care, hospitals) |
| **precision** | 0.60 | 0.7050 | 0.8357 | 0.6096 | Minimize false alarms (public spaces) |

### Command-Line Options

```bash
python -m ml.inference.run_fall_detection [OPTIONS]

Input Source (required, choose one):
  --video PATH              Path to video file
  --camera ID               Camera device ID (e.g., 0 for default webcam)

Model Configuration:
  --model PATH              Path to trained BiLSTM model
                           (default: ml/training/checkpoints/lstm_bilstm_opt_best.h5)
  --threshold-config PATH   Path to threshold configuration JSON
                           (default: ml/training/checkpoints/deployment_thresholds.json)
  --mode {balanced,safety,precision}
                           Detection mode (default: balanced)

Output Options:
  --save-video             Save annotated video to outputs/
  --save-log               Save detection log (JSON) to outputs/
  --output-dir PATH        Output directory (default: outputs/)

Display Options:
  --no-display             Disable video display (useful for headless servers)
  --fps FPS                Target FPS for processing (default: 30)
  --debug                  Print probability for every frame

Interactive Controls (when display is enabled):
  q                        Quit
  s                        Save screenshot
```

## Examples

### Example 1: Test on Video with All Outputs

```bash
python -m ml.inference.run_fall_detection \
    --video data/test/trailfall.mp4 \
    --mode balanced \
    --save-video \
    --save-log \
    --debug
```

**Output:**
```
Frame 30: p=0.0779 
Frame 31: p=0.1322 
Frame 32: p=0.1696 
...
Frame 56: p=0.1483 

DETECTION STATISTICS
Total frames processed: 56
Fall events detected: 0
Average inference time: 11.05 ms
Average FPS: 90.5

Log saved to: outputs/fall_log_trailfall_20251029_141900.json
```

### Example 2: Real-Time Webcam with Safety Mode

```bash
python -m ml.inference.run_fall_detection \
    --camera 0 \
    --mode safety
```

This will:
- Open your default webcam
- Use safety mode (threshold=0.40) for maximum fall detection
- Display live video with pose overlay and fall probability
- Print alerts when falls are detected

### Example 3: Batch Processing (No Display)

```bash
python -m ml.inference.run_fall_detection \
    --video path/to/video.mp4 \
    --mode precision \
    --save-video \
    --save-log \
    --no-display
```

Useful for:
- Processing videos on headless servers
- Batch processing multiple videos
- Automated testing

## Output Files

### Annotated Video

Saved to: `outputs/fall_detection_{source}_{timestamp}.mp4`

Contains:
- Pose skeleton overlay (green lines, red keypoints)
- Probability bar (top-right corner)
- Fall status label (bottom-left)
- Frame info (top-left)

### Detection Log (JSON)

Saved to: `outputs/fall_log_{source}_{timestamp}.json`

```json
{
  "total_frames": 56,
  "fall_events": 2,
  "avg_inference_time_ms": 11.05,
  "avg_fps": 90.5,
  "fall_event_details": [
    {
      "frame": 123,
      "timestamp": "2025-10-29T14:18:59.123456",
      "probability": 0.8734
    }
  ]
}
```

## Performance

### Inference Speed

- **Average inference time**: 11 ms per frame
- **Average FPS**: 90+ on CPU
- **Real-time capable**: Yes (30 FPS video = 33ms per frame budget)

### Hardware Requirements

- **Minimum**: CPU with 4+ cores, 8GB RAM
- **Recommended**: GPU for faster inference (optional)
- **Webcam**: Any USB webcam (640x480 or higher)

### Model Performance

From Phase 2.5 evaluation:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ROC-AUC | 0.9360 | ≥ 0.93 | ✅ Exceeded |
| F1 (balanced) | 0.7456 | ≥ 0.78 | ⚠️ 95.6% |
| Precision (balanced) | 0.7701 | - | ✅ Strong |
| Recall (balanced) | 0.7226 | - | ✅ Good |

## Troubleshooting

### Issue: "Model not found"

**Solution**: Ensure the model checkpoint exists:
```bash
ls -lh ml/training/checkpoints/lstm_bilstm_opt_best.h5
```

If missing, train the model first (Phase 2.3a).

### Issue: "Failed to open video source"

**Solution**: 
- Check video file path is correct
- For webcam, try different device IDs (0, 1, 2, ...)
- Ensure camera permissions are granted

### Issue: Low FPS / Slow inference

**Solution**:
- Close other applications
- Use `--no-display` to disable visualization
- Reduce video resolution
- Use GPU if available

### Issue: No falls detected

**Possible causes**:
1. Video doesn't contain actual falls
2. Threshold too high → Try `--mode safety`
3. Video too short (< 60 frames) → System needs at least 30 frames
4. Poor pose detection → Check if person is clearly visible

**Debug**:
```bash
python -m ml.inference.run_fall_detection --video path/to/video.mp4 --debug
```

This prints probability for every frame to help diagnose issues.

## Technical Details

### Feature Normalization

Features are normalized to [0, 1] range based on training statistics:

```python
# Example ranges
torso_angle: [0, 90] degrees
hip_height: [0, 1] (normalized y-coordinate)
vertical_velocity: [-10, 10] units/second
motion_magnitude: [0, 0.5] units
...
```

### Temporal Smoothing

Predictions are smoothed using Exponential Moving Average (EMA):

```python
p_smooth = 0.7 * p_prev + 0.3 * p_current
```

This reduces jitter and false positives from single-frame noise.

### Ring Buffer

The system maintains a sliding window of 60 frames:
- Minimum 30 frames required for prediction
- Frames < 30: No prediction (p=0.0)
- Frames 30-59: Zero-padded to 60 frames
- Frames ≥ 60: Full 60-frame window

## Integration with Android

The inference system is designed for easy Android integration:

1. **Model Export**: Convert `.h5` to TensorFlow Lite
   ```bash
   python -m ml.training.export_tflite \
       --model ml/training/checkpoints/lstm_bilstm_opt_best.h5 \
       --output models/fall_detection.tflite
   ```

2. **Threshold Config**: Use `deployment_thresholds.json` directly
   ```kotlin
   val config = loadThresholdConfig("deployment_thresholds.json")
   val threshold = config.thresholds.balanced.value
   ```

3. **Feature Extraction**: Port `RealtimeFeatureExtractor` to Kotlin/Java
   - Same 14 features
   - Same normalization ranges
   - Same temporal smoothing

4. **MoveNet**: Use TensorFlow Lite MoveNet model
   - Available on TensorFlow Hub
   - Optimized for mobile devices

## Next Steps

- [ ] Export model to TensorFlow Lite format
- [ ] Create Android app with fall detection
- [ ] Add sound alerts for fall events
- [ ] Implement fall event clip extraction (3-second clips)
- [ ] Add probability plot visualization
- [ ] Support multiple person detection
- [ ] Add fall recovery detection

## References

- **Phase 2.3a**: BiLSTM model training
- **Phase 2.5**: Threshold optimization
- **MoveNet**: https://tfhub.dev/google/movenet/singlepose/lightning/4
- **Training Dataset**: URFD, Le2i, UCF101 fall detection datasets

