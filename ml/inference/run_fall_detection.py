"""
Real-Time Fall Detection Inference - Phase 3.0 + 3.1

Runs fall detection on webcam or video using trained BiLSTM model and MoveNet pose estimation.
Phase 3.1 adds physics-inspired FSM for fall verification.

Usage:
    # Video file
    python -m ml.inference.run_fall_detection --video data/test/trailfall.mp4 --mode balanced

    # Webcam
    python -m ml.inference.run_fall_detection --camera 0 --mode safety

    # With FSM verification (Phase 3.1)
    python -m ml.inference.run_fall_detection --video data/test/trailfall.mp4 --enable-fsm

    # With visualization and logging
    python -m ml.inference.run_fall_detection --video data/test/trailfall.mp4 --save-video --save-log
"""

import sys
import argparse
import json
import time
import csv
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("ERROR: TensorFlow is required.")
    print("Install with: pip install tensorflow")
    sys.exit(1)

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.pose.movenet_loader import load_movenet, infer_keypoints, KEYPOINT_EDGES
from ml.inference.realtime_features import RealtimeFeatureExtractor
from ml.inference.fsm_filter import FallVerificationFSM, load_fsm_config
from ml.training.lstm_train_full import SigmoidFocalCrossEntropy, F1Metric
from ml.training.lstm_train_optimized import build_optimized_bilstm_model


class FallDetector:
    """Real-time fall detection system."""
    
    def __init__(
        self,
        model_path: str,
        threshold_config_path: str,
        mode: str = 'balanced',
        window_size: int = 30,
        fps: int = 30,
        smoothing_alpha: float = 0.7,
        enable_fsm: bool = False,
        fsm_config_path: Optional[str] = None
    ):
        """
        Initialize fall detector.

        Args:
            model_path: Path to trained BiLSTM model (.h5)
            threshold_config_path: Path to threshold configuration JSON
            mode: Detection mode ('balanced', 'safety', or 'precision')
            window_size: Number of frames for inference window (default: 30)
            fps: Frames per second (default: 30)
            smoothing_alpha: EMA smoothing factor for predictions (default: 0.7)
            enable_fsm: Enable physics-inspired FSM verification (Phase 3.1)
            fsm_config_path: Path to FSM configuration JSON
        """
        print("="*70)
        phase_str = "PHASE 3.0 + 3.1 (FSM)" if enable_fsm else "PHASE 3.0"
        print(f"FALL DETECTION SYSTEM - {phase_str}")
        print("="*70)

        # Load MoveNet
        print("\n[1/5] Loading MoveNet pose estimator...")
        self.movenet = load_movenet()

        # Load BiLSTM model
        print("\n[2/5] Loading BiLSTM fall detection model...")
        self.model = self._load_model(model_path)

        # Load threshold configuration
        print("\n[3/5] Loading threshold configuration...")
        self.threshold_config = self._load_threshold_config(threshold_config_path)
        self.mode = mode
        self.threshold = self.threshold_config['thresholds'][mode]['value']
        print(f"  Mode: {mode}")
        print(f"  Threshold: {self.threshold:.4f}")

        # Initialize feature extractor
        print("\n[4/5] Initializing feature extractor...")
        self.feature_extractor = RealtimeFeatureExtractor(fps=fps)

        # Initialize FSM (Phase 3.1)
        print("\n[5/5] Initializing FSM verification...")
        self.enable_fsm = enable_fsm
        self.fsm = None
        if enable_fsm:
            fsm_config = load_fsm_config(fsm_config_path)
            self.fsm = FallVerificationFSM(**fsm_config)
            print(f"  ✓ FSM enabled with thresholds:")
            print(f"    - Rapid descent: v < {fsm_config['v_threshold']}")
            print(f"    - Orientation flip: α ≥ {fsm_config['alpha_threshold']}°")
            print(f"    - Stillness: m ≤ {fsm_config['m_threshold']} for {fsm_config['stillness_frames']} frames")
        else:
            print("  FSM verification disabled (use --enable-fsm to enable)")

        # Ring buffer for 60-frame windows
        self.window_size = window_size
        self.feature_buffer = deque(maxlen=window_size)

        # Smoothing
        self.smoothing_alpha = smoothing_alpha
        self.prev_prob = 0.0
        
        # Statistics
        self.frame_count = 0
        self.fall_events = []
        self.inference_times = []
        self.frame_history = deque(maxlen=90)  # Store last 3 seconds of frames for clip extraction
        self.prob_history = []  # Store all probabilities for CSV logging
        
        print("\n✅ Fall detection system initialized!")
        print("="*70)
    
    def _load_model(self, model_path: str) -> keras.Model:
        """Load trained BiLSTM model."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Rebuild model architecture and load weights
        model = build_optimized_bilstm_model((60, 14))
        model.load_weights(str(model_path))
        
        print(f"  ✓ Model loaded: {model_path.name}")
        return model
    
    def _load_threshold_config(self, config_path: str) -> dict:
        """Load threshold configuration."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Threshold config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"  ✓ Threshold config loaded: {config_path.name}")
        return config
    
    def process_frame(self, frame: np.ndarray, save_clip_callback=None) -> Tuple[np.ndarray, float, bool]:
        """
        Process a single frame and detect falls.

        Args:
            frame: BGR frame from video/webcam
            save_clip_callback: Optional callback function to save fall event clips

        Returns:
            Tuple of (annotated_frame, fall_probability, is_fall_detected)
        """
        start_time = time.time()

        # Store frame in history for clip extraction
        self.frame_history.append(frame.copy())
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints
        keypoints = infer_keypoints(self.movenet, frame_rgb, confidence_threshold=0.3)

        # Extract features
        features = self.feature_extractor.extract_features(keypoints)

        # Add to buffer
        self.feature_buffer.append(features)

        # Run inference if buffer has enough frames (at least 30 frames)
        fall_prob = 0.0
        is_fall_lstm = False
        fsm_fall = False
        fsm_state_dict = {}

        if len(self.feature_buffer) >= min(30, self.window_size):
            # Prepare input - pad if necessary
            buffer_array = np.array(self.feature_buffer)

            if len(self.feature_buffer) < self.window_size:
                # Pad with zeros to reach window_size
                padding = np.zeros((self.window_size - len(self.feature_buffer), 14))
                buffer_array = np.vstack([padding, buffer_array])

            # Prepare input (1, window_size, 14)
            X = np.expand_dims(buffer_array, axis=0)

            # Predict
            fall_prob = self.model.predict(X, verbose=0)[0][0]

            # Apply EMA smoothing
            fall_prob = self.smoothing_alpha * self.prev_prob + (1 - self.smoothing_alpha) * fall_prob
            self.prev_prob = fall_prob

            # Check LSTM threshold
            is_fall_lstm = fall_prob >= self.threshold

        # Run FSM verification (Phase 3.1)
        if self.enable_fsm and self.fsm is not None:
            # Extract specific features for FSM
            # Features: [0]=torso_angle, [1]=hip_height, [2]=vertical_velocity, [3]=motion_magnitude, ...
            vertical_velocity = features[2]  # Index 2: vertical_velocity
            torso_angle = features[0]  # Index 0: torso_angle
            motion_magnitude = features[3]  # Index 3: motion_magnitude

            # Update FSM
            fsm_fall, fsm_state_dict = self.fsm.update(
                vertical_velocity=vertical_velocity,
                torso_angle=torso_angle,
                motion_magnitude=motion_magnitude
            )

        # Combined decision logic
        if self.enable_fsm:
            # Phase 3.1: Require both LSTM and FSM agreement
            is_fall = is_fall_lstm and fsm_fall
            decision_type = self._get_decision_type(is_fall_lstm, fsm_fall)
        else:
            # Phase 3.0: LSTM only
            is_fall = is_fall_lstm
            decision_type = "FALL DETECTED" if is_fall else "Normal"

        # Record inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        # Store probability for CSV logging
        log_entry = {
            'frame': self.frame_count,
            'timestamp': datetime.now().isoformat(),
            'probability': float(fall_prob),
            'is_fall_lstm': is_fall_lstm,
            'is_fall_fsm': fsm_fall if self.enable_fsm else None,
            'is_fall_combined': is_fall,
            'decision': decision_type,
            'mode': self.mode
        }

        # Add FSM state details if enabled
        if self.enable_fsm:
            log_entry.update({
                'fsm_rapid_descent': fsm_state_dict.get('rapid_descent', False),
                'fsm_orientation_flip': fsm_state_dict.get('orientation_flip', False),
                'fsm_stillness': fsm_state_dict.get('stillness', False)
            })

        self.prob_history.append(log_entry)

        # Annotate frame
        annotated_frame = self._annotate_frame(
            frame, keypoints, fall_prob, is_fall,
            fsm_state_dict if self.enable_fsm else None
        )

        # Record fall event and save clip
        if is_fall:
            event = {
                'frame': self.frame_count,
                'timestamp': datetime.now().isoformat(),
                'probability': float(fall_prob),
                'lstm_decision': is_fall_lstm,
                'fsm_decision': fsm_fall if self.enable_fsm else None,
                'decision_type': decision_type
            }
            self.fall_events.append(event)

            # Save fall event clip if callback provided
            if save_clip_callback:
                save_clip_callback(self.frame_history, event)

        self.frame_count += 1

        return annotated_frame, fall_prob, is_fall

    def _get_decision_type(self, is_fall_lstm: bool, fsm_fall: bool) -> str:
        """Get decision type string based on LSTM and FSM results."""
        if is_fall_lstm and fsm_fall:
            return "FALL DETECTED (LSTM + FSM)"
        elif fsm_fall:
            return "Candidate (FSM only)"
        elif is_fall_lstm:
            return "Candidate (LSTM only)"
        else:
            return "Normal"
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        fall_prob: float,
        is_fall: bool,
        fsm_state_dict: Optional[Dict] = None
    ) -> np.ndarray:
        """Annotate frame with pose skeleton and fall detection results."""
        annotated = frame.copy()
        height, width = frame.shape[:2]

        # Draw skeleton
        for edge in KEYPOINT_EDGES:
            kp1_idx, kp2_idx = edge
            y1, x1, conf1 = keypoints[kp1_idx]
            y2, x2, conf2 = keypoints[kp2_idx]

            if conf1 >= 0.3 and conf2 >= 0.3:
                x1_px, y1_px = int(x1 * width), int(y1 * height)
                x2_px, y2_px = int(x2 * width), int(y2 * height)
                cv2.line(annotated, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 2)

        # Draw keypoints
        for y, x, conf in keypoints:
            if conf >= 0.3:
                x_px, y_px = int(x * width), int(y * height)
                cv2.circle(annotated, (x_px, y_px), 4, (0, 0, 255), -1)

        # Draw probability bar
        bar_width = 200
        bar_height = 20
        bar_x = width - bar_width - 20
        bar_y = 20

        # Background
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Probability fill
        fill_width = int(bar_width * fall_prob)
        color = (0, 0, 255) if is_fall else (0, 255, 0)
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)

        # Text
        prob_text = f"Fall Prob: {fall_prob:.2f}"
        cv2.putText(annotated, prob_text, (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FSM State Overlay (Phase 3.1)
        if fsm_state_dict is not None:
            fsm_y = 60

            # Rapid Descent
            if fsm_state_dict.get('rapid_descent', False):
                cv2.putText(annotated, "FSM: Rapid Descent", (20, fsm_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Yellow
                fsm_y += 30

            # Orientation Flip
            if fsm_state_dict.get('orientation_flip', False):
                cv2.putText(annotated, "FSM: Orientation Flip", (20, fsm_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)  # Orange
                fsm_y += 30

            # Stillness
            if fsm_state_dict.get('stillness', False):
                cv2.putText(annotated, "FSM: Stillness", (20, fsm_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red
                fsm_y += 30

            # All conditions met
            if fsm_state_dict.get('all_conditions_met', False):
                cv2.putText(annotated, "FSM: ALL CONDITIONS MET", (20, fsm_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)  # Magenta

        # Status label
        if is_fall:
            label = "🚨 FALL DETECTED"
            label_color = (0, 0, 255)
        else:
            label = "Normal"
            label_color = (0, 255, 0)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        label_x = 20
        label_y = height - 40
        cv2.rectangle(annotated, (label_x - 10, label_y - label_size[1] - 10),
                     (label_x + label_size[0] + 10, label_y + 10), (0, 0, 0), -1)
        
        # Draw label text
        cv2.putText(annotated, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)
        
        # Frame info
        info_text = f"Frame: {self.frame_count} | Mode: {self.mode} | Threshold: {self.threshold:.2f}"
        cv2.putText(annotated, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def get_statistics(self) -> dict:
        """Get detection statistics."""
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        return {
            'total_frames': self.frame_count,
            'fall_events': len(self.fall_events),
            'avg_inference_time_ms': avg_inference_time * 1000,
            'avg_fps': avg_fps,
            'fall_event_details': self.fall_events
        }

    def save_csv_log(self, output_path: Path):
        """Save frame-by-frame probability log as CSV."""
        if not self.prob_history:
            return

        # Get fieldnames from first entry
        fieldnames = list(self.prob_history[0].keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.prob_history)

        print(f"CSV log saved to: {output_path}")


def save_fall_clip(frame_history: deque, event: dict, output_dir: Path, fps: int = 30):
    """
    Save a 3-second video clip around the fall event.

    Args:
        frame_history: Deque of recent frames (up to 90 frames = 3 seconds)
        event: Fall event dictionary with frame, timestamp, probability
        output_dir: Directory to save clips
        fps: Frames per second for output video
    """
    if len(frame_history) == 0:
        return

    # Create fall_events subdirectory
    clips_dir = output_dir / 'fall_events'
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_num = event['frame']
    prob = event['probability']
    clip_path = clips_dir / f"fall_event_frame{frame_num}_p{prob:.2f}_{timestamp}.mp4"

    # Get frame dimensions
    height, width = frame_history[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

    # Write all frames from history
    for frame in frame_history:
        writer.write(frame)

    writer.release()

    print(f"  💾 Fall clip saved: {clip_path.name} ({len(frame_history)} frames)")


def main():
    parser = argparse.ArgumentParser(description='Phase 3.0 - Real-Time Fall Detection')
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to video file')
    input_group.add_argument('--camera', type=int, help='Camera device ID (e.g., 0)')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='ml/training/checkpoints/lstm_bilstm_opt_best.h5',
                       help='Path to trained BiLSTM model')
    parser.add_argument('--threshold-config', type=str,
                       default='ml/training/checkpoints/deployment_thresholds.json',
                       help='Path to threshold configuration')
    parser.add_argument('--mode', type=str, default='balanced',
                       choices=['balanced', 'safety', 'precision'],
                       help='Detection mode (balanced/safety/precision)')

    # FSM options (Phase 3.1)
    parser.add_argument('--enable-fsm', action='store_true',
                       help='Enable physics-inspired FSM verification (Phase 3.1)')
    parser.add_argument('--fsm-config', type=str, default='ml/inference/fsm_config.json',
                       help='Path to FSM configuration JSON')

    # Output options
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    parser.add_argument('--save-log', action='store_true', help='Save detection log (JSON)')
    parser.add_argument('--save-csv', action='store_true', help='Save frame-by-frame CSV log')
    parser.add_argument('--save-clips', action='store_true', help='Save 3-second clips of fall events')
    parser.add_argument('--save-fsm-log', action='store_true', help='Save FSM state log (CSV)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')

    # Display options
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS for processing')
    parser.add_argument('--debug', action='store_true', help='Print probability for every frame')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    detector = FallDetector(
        model_path=args.model,
        threshold_config_path=args.threshold_config,
        mode=args.mode,
        fps=args.fps,
        enable_fsm=args.enable_fsm,
        fsm_config_path=args.fsm_config if args.enable_fsm else None
    )
    
    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = Path(args.video).stem
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"camera_{args.camera}"
    
    if not cap.isOpened():
        print(f"ERROR: Failed to open video source")
        sys.exit(1)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nVideo source: {source_name}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {source_fps:.1f}")

    if args.save_clips:
        print(f"Fall clips will be saved to: {output_dir / 'fall_events'}/")

    print("\nPress 'q' to quit, 's' to save screenshot\n")
    
    # Setup video writer
    video_writer = None
    if args.save_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = output_dir / f"fall_detection_{source_name}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, args.fps, 
                                       (frame_width, frame_height))
        print(f"Saving video to: {output_video_path}")
    
    # Setup fall clip callback
    clip_callback = None
    if args.save_clips:
        clip_callback = lambda history, event: save_fall_clip(history, event, output_dir, args.fps)

    # Process frames
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Process frame
            annotated_frame, fall_prob, is_fall = detector.process_frame(frame, clip_callback)

            # Debug mode - print every frame
            if args.debug:
                print(f"Frame {detector.frame_count}: p={fall_prob:.4f} {'[FALL]' if is_fall else ''}")

            # Print fall detection
            if is_fall and not args.debug:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"🚨 FALL DETECTED (p={fall_prob:.4f}) at {timestamp} (frame {detector.frame_count})")
            
            # Save video
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display
            if not args.no_display:
                cv2.imshow('Fall Detection', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = output_dir / f"screenshot_{detector.frame_count}.png"
                    cv2.imwrite(str(screenshot_path), annotated_frame)
                    print(f"Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        stats = detector.get_statistics()
        print("\n" + "="*70)
        print("DETECTION STATISTICS")
        print("="*70)
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Fall events detected: {stats['fall_events']}")
        print(f"Average inference time: {stats['avg_inference_time_ms']:.2f} ms")
        print(f"Average FPS: {stats['avg_fps']:.1f}")

        # Print FSM statistics if enabled
        if args.enable_fsm and detector.fsm is not None:
            fsm_stats = detector.fsm.get_statistics()
            print(f"\nFSM Statistics:")
            print(f"  FSM fall candidates: {fsm_stats['fsm_falls_detected']}")
            print(f"  Thresholds: v<{fsm_stats['thresholds']['vertical_velocity']}, "
                  f"α≥{fsm_stats['thresholds']['torso_angle']}°, "
                  f"m≤{fsm_stats['thresholds']['motion_magnitude']}")

        # Save logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.save_log:
            log_path = output_dir / f"fall_log_{source_name}_{timestamp}.json"

            # Add FSM stats to JSON if enabled
            if args.enable_fsm and detector.fsm is not None:
                stats['fsm_statistics'] = detector.fsm.get_statistics()

            with open(log_path, 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"\nJSON log saved to: {log_path}")

        if args.save_csv:
            csv_path = output_dir / f"fall_logs_{source_name}_{timestamp}.csv"
            detector.save_csv_log(csv_path)

        # Save FSM log if enabled
        if args.save_fsm_log and args.enable_fsm and detector.fsm is not None:
            fsm_log_path = output_dir / f"fsm_logs_{source_name}_{timestamp}.csv"
            detector.fsm.save_log(fsm_log_path)

        print("="*70)


if __name__ == '__main__':
    main()

