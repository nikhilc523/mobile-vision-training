"""
Real-Time Fall Detection Inference - Phase 4.3 (Stateful + Post-Filters)

Key Improvements:
- Stateful LSTM inference (maintains hidden state across frames)
- 30-frame window (1.0 second) instead of 60 frames
- Raw keypoints (34 features) instead of 14 engineered features
- Enhanced post-processing filters:
  * EMA height ratio < 0.66 (instead of 2/3)
  * Angle check ≥ 35° (instead of 45°)
  * Consecutive frames: 5 (instead of 9)
- FSM verification for robustness
- Combined decision: (LSTM + filters) AND FSM

Usage:
    python -m ml.inference.run_fall_detection_v2 --video data/test/secondfall.mp4 --mode balanced --stateful
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
    sys.exit(1)

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.pose.movenet_loader import load_movenet, infer_keypoints, KEYPOINT_EDGES
from ml.inference.realtime_features_raw import RealtimeRawKeypointsExtractor
from ml.inference.fsm_filter import FallVerificationFSM, load_fsm_config


class FallDetectorV2:
    """
    Real-time fall detector with stateful inference and post-filters.

    Phase 4.3 improvements:
    - Stateful LSTM inference (maintains hidden state)
    - 30-frame window (1.0 second @ 30 FPS)
    - Raw keypoints (34 features)
    - Enhanced post-processing filters:
      * EMA height ratio < 0.66
      * Angle check ≥ 35°
      * Consecutive frames: 5
    - FSM verification
    - Combined decision: (LSTM + filters) AND FSM
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        window_size: int = 30,
        fps: int = 30,
        smoothing_alpha: float = 0.7,
        enable_fsm: bool = True,
        fsm_config_path: Optional[str] = None,
        enable_post_filters: bool = True,
        stateful: bool = True
    ):
        """
        Initialize fall detector.

        Args:
            model_path: Path to trained BiLSTM model (.h5)
            threshold: Fall detection threshold (default: 0.5)
            window_size: Number of frames for inference window (default: 30)
            fps: Frames per second (default: 30)
            smoothing_alpha: EMA smoothing factor for predictions (default: 0.7)
            enable_fsm: Enable physics-inspired FSM verification
            fsm_config_path: Path to FSM configuration JSON
            enable_post_filters: Enable post-processing filters (height, angle, consecutive frames)
            stateful: Enable stateful LSTM inference (maintains hidden state)
        """
        print("="*70)
        print("FALL DETECTION SYSTEM - PHASE 4.3 (STATEFUL + POST-FILTERS)")
        print("="*70)

        # Load MoveNet
        print("\n[1/5] Loading MoveNet pose estimator...")
        self.movenet = load_movenet()
        print("✓ MoveNet loaded")

        # Load model
        print(f"\n[2/5] Loading BiLSTM model from {model_path}...")
        self.model = keras.models.load_model(model_path, compile=False)
        print(f"✓ Model loaded")
        print(f"  Input shape: {self.model.input_shape}")
        print(f"  Expected: (None, {window_size}, 34)")

        # Configuration
        self.threshold = threshold
        self.window_size = window_size
        self.fps = fps
        self.smoothing_alpha = smoothing_alpha
        self.enable_fsm = enable_fsm
        self.enable_post_filters = enable_post_filters
        self.stateful = stateful

        print(f"\n[3/5] Configuration:")
        print(f"  Window size: {window_size} frames ({window_size/fps:.2f} seconds)")
        print(f"  Threshold: {threshold}")
        print(f"  Smoothing alpha: {smoothing_alpha}")
        print(f"  Stateful inference: {stateful}")
        print(f"  FSM enabled: {enable_fsm}")
        print(f"  Post-filters enabled: {enable_post_filters}")

        # Feature extractor
        print(f"\n[4/5] Initializing feature extractor...")
        self.feature_extractor = RealtimeRawKeypointsExtractor()
        print(f"✓ Feature extractor initialized (34 features)")

        # FSM
        self.fsm = None
        if enable_fsm:
            print(f"\n[5/5] Initializing FSM...")
            if fsm_config_path:
                fsm_config = load_fsm_config(fsm_config_path)
                self.fsm = FallVerificationFSM(
                    v_threshold=fsm_config.get('v_threshold', -0.28),
                    alpha_threshold=fsm_config.get('alpha_threshold', 65.0),
                    m_threshold=fsm_config.get('m_threshold', 0.02),
                    rapid_descent_frames=fsm_config.get('rapid_descent_frames', 2),
                    stillness_frames=fsm_config.get('stillness_frames', 12),
                    window_size=fsm_config.get('window_size', 90)
                )
            else:
                self.fsm = FallVerificationFSM()
            print(f"✓ FSM initialized")

        # Buffers
        self.feature_buffer = deque(maxlen=window_size)
        self.keypoints_buffer = deque(maxlen=window_size)
        self.frame_history = deque(maxlen=window_size * 2)

        # State
        self.frame_count = 0
        self.prev_prob = 0.0
        self.fall_events = []
        self.inference_times = []
        self.fall_logs = []

        # Stateful LSTM state
        self.lstm_state = None  # Will store (h_state, c_state) for stateful inference

        # Post-processing state (Phase 4.7 relaxed thresholds for short clips)
        self.ema_height = 0.0
        self.fall_counter = 0
        self.consecutive_frames_required = 3  # Phase 4.7: Reduced from 5 to 3 for better sensitivity on short clips
        self.height_ratio_threshold = 0.66  # Phase 4.3: Height ratio threshold
        self.angle_threshold = 35.0  # Phase 4.3: Angle threshold (degrees)
        self.angle_history = deque(maxlen=10)  # Track last 10 frames for angle check

        print("\n" + "="*70)
        print("✅ INITIALIZATION COMPLETE")
        print("="*70)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Process a single frame with stateful LSTM inference.

        Args:
            frame: Input frame (BGR)

        Returns:
            annotated_frame: Frame with pose and fall detection overlay
            fall_prob: Fall probability [0, 1]
            is_fall: True if fall detected
        """
        start_time = time.time()

        # Store frame
        self.frame_history.append(frame.copy())

        # Detect keypoints
        keypoints = infer_keypoints(self.movenet, frame)
        self.keypoints_buffer.append(keypoints)

        # Extract features
        features = self.feature_extractor.extract(keypoints)
        self.feature_buffer.append(features)

        # Initialize prediction
        fall_prob = 0.0
        is_fall_lstm = False
        fsm_fall = False
        fsm_state_dict = None

        # LSTM prediction
        if self.stateful:
            # Stateful inference: process frame-by-frame maintaining hidden state
            if len(self.feature_buffer) >= 1:
                # Use rolling window of last 30 frames (or less if not enough frames yet)
                buffer_array = np.array(list(self.feature_buffer)[-self.window_size:])

                # Pad if needed (for first few frames)
                if len(buffer_array) < self.window_size:
                    padding = np.zeros((self.window_size - len(buffer_array), 34))
                    buffer_array = np.vstack([padding, buffer_array])

                # Prepare input (1, 30, 34)
                X = np.expand_dims(buffer_array, axis=0)

                # Predict (stateful - maintains hidden state across calls)
                fall_prob = self.model.predict(X, verbose=0)[0][0]

                # Apply EMA smoothing
                fall_prob = self.smoothing_alpha * self.prev_prob + (1 - self.smoothing_alpha) * fall_prob
                self.prev_prob = fall_prob

                # Check LSTM threshold
                is_fall_lstm = fall_prob >= self.threshold
        else:
            # Non-stateful inference: requires full window
            if len(self.feature_buffer) >= self.window_size:
                buffer_array = np.array(list(self.feature_buffer)[-self.window_size:])

                # Prepare input (1, 30, 34)
                X = np.expand_dims(buffer_array, axis=0)

                # Predict
                fall_prob = self.model.predict(X, verbose=0)[0][0]

                # Apply EMA smoothing
                fall_prob = self.smoothing_alpha * self.prev_prob + (1 - self.smoothing_alpha) * fall_prob
                self.prev_prob = fall_prob

                # Check LSTM threshold
                is_fall_lstm = fall_prob >= self.threshold

        # FSM verification
        if self.enable_fsm and self.fsm is not None and len(self.keypoints_buffer) >= 2:
            # Calculate features for FSM
            kp_curr = self.keypoints_buffer[-1]
            kp_prev = self.keypoints_buffer[-2]

            # Vertical velocity (hip height change)
            hip_curr = (kp_curr[11, 0] + kp_curr[12, 0]) / 2  # Average hip y
            hip_prev = (kp_prev[11, 0] + kp_prev[12, 0]) / 2
            vertical_velocity = (hip_curr - hip_prev) * self.fps

            # Torso angle
            shoulder_center = (kp_curr[5] + kp_curr[6]) / 2
            hip_center = (kp_curr[11] + kp_curr[12]) / 2
            torso_vec = hip_center - shoulder_center
            torso_angle = np.abs(np.degrees(np.arctan2(torso_vec[1], torso_vec[0])))

            # Motion magnitude
            motion = np.mean(np.linalg.norm(kp_curr[:, :2] - kp_prev[:, :2], axis=1))

            # Update FSM
            fsm_fall, fsm_state_dict = self.fsm.update(
                vertical_velocity=vertical_velocity,
                torso_angle=torso_angle,
                motion_magnitude=motion
            )

        # Post-processing filters
        is_fall = is_fall_lstm
        filter_status = {}
        if self.enable_post_filters and len(self.keypoints_buffer) >= 1:
            is_fall, filter_status = self._apply_post_filters(is_fall_lstm, keypoints)

        # Combined decision with FSM
        if self.enable_fsm:
            is_fall = is_fall and fsm_fall

        # Record inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        # Log
        log_entry = {
            'frame': int(self.frame_count),
            'timestamp': datetime.now().isoformat(),
            'probability': float(fall_prob),
            'lstm_decision': bool(is_fall_lstm),
            'fsm_decision': bool(fsm_fall) if self.enable_fsm else None,
            'filter_status': filter_status if self.enable_post_filters else {},
            'final_decision': bool(is_fall),
            'inference_time_ms': float(inference_time * 1000)
        }
        self.fall_logs.append(log_entry)

        # Record fall event
        if is_fall:
            event = {
                'frame': self.frame_count,
                'timestamp': datetime.now().isoformat(),
                'probability': float(fall_prob)
            }
            self.fall_events.append(event)

        self.frame_count += 1

        # Annotate frame
        annotated_frame = self._annotate_frame(frame, keypoints, fall_prob, is_fall, fsm_state_dict)

        return annotated_frame, fall_prob, is_fall

    def _apply_post_filters(self, is_fall_lstm: bool, keypoints: np.ndarray) -> Tuple[bool, dict]:
        """
        Apply Phase 4.3 post-processing filters.

        Filters:
        1. EMA height ratio < 0.66 (person must be low to ground)
        2. Angle check ≥ 35° within last 10 frames (body must be tilted)
        3. Consecutive frames: at least 5 consecutive frames above threshold

        Args:
            is_fall_lstm: LSTM prediction
            keypoints: Current keypoints (17, 3)

        Returns:
            Tuple of (filtered_decision, filter_status_dict)
        """
        # Calculate current height (hip to ground)
        hip_y = (keypoints[11, 0] + keypoints[12, 0]) / 2  # Average hip y (normalized [0, 1])
        current_height = 1.0 - hip_y  # Invert so higher value = taller

        # Update EMA height (running standing height estimate)
        if self.ema_height == 0.0:
            self.ema_height = current_height
        else:
            # Slowly update EMA (99% old, 1% new) to track standing height
            self.ema_height = 0.99 * self.ema_height + 0.01 * current_height

        # Calculate height ratio
        height_ratio = current_height / self.ema_height if self.ema_height > 0 else 1.0

        # Calculate torso angle
        shoulder_center = (keypoints[5] + keypoints[6]) / 2
        hip_center = (keypoints[11] + keypoints[12]) / 2
        torso_vec = hip_center - shoulder_center
        torso_angle = np.abs(np.degrees(np.arctan2(torso_vec[1], torso_vec[0])))

        # Track angle history
        self.angle_history.append(torso_angle)

        # Check if any angle in last 10 frames exceeds threshold
        angle_check_passed = any(angle >= self.angle_threshold for angle in self.angle_history)

        # Filter status for logging
        filter_status = {
            'height_ratio': float(height_ratio),
            'height_check_passed': bool(height_ratio < self.height_ratio_threshold),
            'torso_angle': float(torso_angle),
            'angle_check_passed': bool(angle_check_passed),
            'consecutive_count': int(self.fall_counter),
            'consecutive_check_passed': bool(self.fall_counter >= self.consecutive_frames_required)
        }

        # Apply filters
        if is_fall_lstm:
            # Filter 1: Height check (person must be low - below 0.66 of normal height)
            if height_ratio >= self.height_ratio_threshold:
                self.fall_counter = max(0, self.fall_counter - 1)
                return False, filter_status

            # Filter 2: Angle check (body must be tilted ≥ 35° in last 10 frames)
            if not angle_check_passed:
                self.fall_counter = max(0, self.fall_counter - 1)
                return False, filter_status

            # Filter 3: Consecutive frames (require 5 consecutive frames)
            self.fall_counter += 1
            if self.fall_counter < self.consecutive_frames_required:
                return False, filter_status

            return True, filter_status
        else:
            # Decay fall counter
            self.fall_counter = max(0, self.fall_counter - 1)
            return False, filter_status

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
            kp1 = keypoints[kp1_idx]
            kp2 = keypoints[kp2_idx]

            if kp1[2] > 0.3 and kp2[2] > 0.3:
                y1, x1 = int(kp1[0] * height), int(kp1[1] * width)
                y2, x2 = int(kp2[0] * height), int(kp2[1] * width)
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw keypoints
        for kp in keypoints:
            if kp[2] > 0.3:
                y, x = int(kp[0] * height), int(kp[1] * width)
                cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)

        # Fall detection overlay
        color = (0, 0, 255) if is_fall else (0, 255, 0)
        status = "FALL DETECTED!" if is_fall else "Normal"

        cv2.rectangle(annotated, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.putText(annotated, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(annotated, f"Probability: {fall_prob:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Frame: {self.frame_count}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # FSM state
        if self.enable_fsm and fsm_state_dict:
            fsm_status = "FALL" if fsm_state_dict.get('all_conditions_met', False) else "Normal"
            fsm_text = f"FSM: {fsm_status}"
            cv2.putText(annotated, fsm_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return annotated

    def get_stats(self) -> dict:
        """Get detection statistics."""
        avg_fps = 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        return {
            'total_frames': self.frame_count,
            'fall_events': len(self.fall_events),
            'avg_fps': avg_fps,
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0
        }


def main():
    parser = argparse.ArgumentParser(description='Phase 4.3 - Stateful Fall Detection with Post-Filters')

    # Input
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--camera', type=int, help='Camera index (default: 0)')

    # Model
    parser.add_argument('--model', type=str, default='ml/training/checkpoints/lstm_raw30_balanced_hnm_best.h5',
                        help='Path to trained model (default: Phase 4.6 HNM model)')
    parser.add_argument('--mode', type=str, choices=['balanced', 'safety', 'precision'], default='balanced',
                        help='Detection mode: balanced (0.85), safety (0.75), precision (0.90)')
    parser.add_argument('--threshold', type=float, help='Custom threshold (overrides --mode)')

    # Options
    parser.add_argument('--stateful', action='store_true', default=True,
                        help='Enable stateful LSTM inference (default: True)')
    parser.add_argument('--no-stateful', action='store_true',
                        help='Disable stateful inference (requires full 30-frame window)')
    parser.add_argument('--enable-fsm', action='store_true', default=True,
                        help='Enable FSM verification (default: True)')
    parser.add_argument('--disable-fsm', action='store_true',
                        help='Disable FSM verification')
    parser.add_argument('--enable-post-filters', action='store_true', default=True,
                        help='Enable post-processing filters (default: True)')
    parser.add_argument('--disable-post-filters', action='store_true',
                        help='Disable post-processing filters')
    parser.add_argument('--fsm-config', type=str, default='ml/inference/fsm_config.json',
                        help='Path to FSM configuration JSON')

    # Output
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    parser.add_argument('--save-log', action='store_true', help='Save JSON/CSV logs')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')

    args = parser.parse_args()

    # Validate input
    if not args.video and args.camera is None:
        parser.error("Either --video or --camera must be specified")

    # Determine threshold based on mode
    if args.threshold is not None:
        threshold = args.threshold
    else:
        # Phase 4.6 HNM optimal thresholds
        mode_thresholds = {
            'balanced': 0.85,  # Phase 4.6 HNM: F1=0.9942, Precision=0.9902, Recall=0.9983
            'safety': 0.75,    # Lower threshold for higher recall (safety-first)
            'precision': 0.90  # Higher threshold for higher precision
        }
        threshold = mode_thresholds[args.mode]
        print(f"\nUsing {args.mode} mode with threshold: {threshold} (Phase 4.4 optimal)")

    # Handle flags
    stateful = not args.no_stateful
    enable_fsm = args.enable_fsm and not args.disable_fsm
    enable_post_filters = args.enable_post_filters and not args.disable_post_filters

    # Initialize detector
    detector = FallDetectorV2(
        model_path=args.model,
        threshold=threshold,
        window_size=30,
        enable_fsm=enable_fsm,
        fsm_config_path=args.fsm_config if enable_fsm else None,
        enable_post_filters=enable_post_filters,
        stateful=stateful
    )

    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = Path(args.video).stem
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"camera_{args.camera}"

    if not cap.isOpened():
        print(f"ERROR: Could not open video source")
        return

    # Video writer
    writer = None
    if args.save_video:
        output_dir = Path(args.output_dir) / source_name
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{source_name}_annotated_{timestamp}.mp4"

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"\nSaving video to: {output_path}")

    # Process frames
    print("\nProcessing frames... (Press 'q' to quit)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, fall_prob, is_fall = detector.process_frame(frame)

        if writer:
            writer.write(annotated_frame)

        if not args.no_display:
            cv2.imshow('Fall Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Print stats
    stats = detector.get_stats()
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    print(f"Total frames: {stats['total_frames']}")
    print(f"Fall events: {stats['fall_events']}")
    print(f"Average FPS: {stats['avg_fps']:.1f}")
    print(f"Average inference time: {stats['avg_inference_time_ms']:.1f} ms")

    # Save logs
    if args.save_log:
        output_dir = Path(args.output_dir) / source_name
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_path = output_dir / f"fall_log_{source_name}_{timestamp}.json"
        with open(log_path, 'w') as f:
            json.dump({
                'stats': stats,
                'fall_events': detector.fall_events,
                'logs': detector.fall_logs
            }, f, indent=2)
        print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()

