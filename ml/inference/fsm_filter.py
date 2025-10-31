"""
Physics-Inspired Rule-Based FSM for Fall Verification - Phase 3.1

Implements a finite-state machine that verifies falls using three sequential physical cues:
1. Rapid Descent: Sudden downward velocity (v < -0.28)
2. Orientation Flip: Body becomes horizontal (α ≥ 65°)
3. Stillness: Minimal motion after impact (m ≤ 0.02 for ≥12 frames)

The FSM runs in parallel with the LSTM model and only confirms a fall when
all three conditions are met within a sliding window.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path


class FallVerificationFSM:
    """
    Finite-State Machine for physics-based fall verification.
    
    States:
    - NORMAL: Default state, no fall indicators
    - RAPID_DESCENT: Detected sudden downward velocity
    - ORIENTATION_FLIP: Detected horizontal body orientation
    - STILLNESS: Detected prolonged stillness after descent
    
    A fall is confirmed only when all three conditions are active simultaneously.
    """
    
    # State constants
    STATE_NORMAL = "Normal"
    STATE_RAPID_DESCENT = "Rapid Descent"
    STATE_ORIENTATION_FLIP = "Orientation Flip"
    STATE_STILLNESS = "Stillness"
    
    def __init__(
        self,
        v_threshold: float = -0.28,
        alpha_threshold: float = 65.0,
        m_threshold: float = 0.02,
        rapid_descent_frames: int = 2,
        stillness_frames: int = 12,
        window_size: int = 90  # 3 seconds at 30 FPS
    ):
        """
        Initialize FSM with configurable thresholds.
        
        Args:
            v_threshold: Vertical velocity threshold for rapid descent (negative value)
            alpha_threshold: Torso angle threshold for orientation flip (degrees)
            m_threshold: Motion magnitude threshold for stillness
            rapid_descent_frames: Minimum consecutive frames for rapid descent
            stillness_frames: Minimum consecutive frames for stillness
            window_size: Size of sliding window to check all conditions (frames)
        """
        # Thresholds
        self.v_threshold = v_threshold
        self.alpha_threshold = alpha_threshold
        self.m_threshold = m_threshold
        self.rapid_descent_frames = rapid_descent_frames
        self.stillness_frames = stillness_frames
        self.window_size = window_size
        
        # State tracking
        self.current_states = set()  # Active states
        self.state_history = deque(maxlen=window_size)  # History of state sets
        
        # Feature history for condition checking
        self.v_history = deque(maxlen=rapid_descent_frames)
        self.m_history = deque(maxlen=stillness_frames)
        
        # Counters
        self.frame_count = 0
        self.fall_count = 0
        
        # Logging
        self.state_log = []
        
    def update(
        self,
        vertical_velocity: float,
        torso_angle: float,
        motion_magnitude: float
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Update FSM with new frame features and check for fall.
        
        Args:
            vertical_velocity: Vertical velocity (normalized, negative = downward)
            torso_angle: Torso angle in degrees (0 = vertical, 90 = horizontal)
            motion_magnitude: Motion magnitude (normalized)
            
        Returns:
            Tuple of (is_fall_detected, state_dict)
            - is_fall_detected: True if all three conditions are met
            - state_dict: Dictionary of active states
        """
        # Update feature history
        self.v_history.append(vertical_velocity)
        self.m_history.append(motion_magnitude)
        
        # Reset current states
        self.current_states = set()
        
        # Check Condition 1: Rapid Descent
        if self._check_rapid_descent():
            self.current_states.add(self.STATE_RAPID_DESCENT)
        
        # Check Condition 2: Orientation Flip
        if self._check_orientation_flip(torso_angle):
            self.current_states.add(self.STATE_ORIENTATION_FLIP)
        
        # Check Condition 3: Stillness
        if self._check_stillness():
            self.current_states.add(self.STATE_STILLNESS)
        
        # Add to state history
        self.state_history.append(self.current_states.copy())
        
        # Check if all three conditions are met within the window
        is_fall = self._check_all_conditions_met()
        
        # Log state
        self.state_log.append({
            'frame': self.frame_count,
            'timestamp': datetime.now().isoformat(),
            'rapid_descent': self.STATE_RAPID_DESCENT in self.current_states,
            'orientation_flip': self.STATE_ORIENTATION_FLIP in self.current_states,
            'stillness': self.STATE_STILLNESS in self.current_states,
            'fsm_fall': is_fall,
            'vertical_velocity': float(vertical_velocity),
            'torso_angle': float(torso_angle),
            'motion_magnitude': float(motion_magnitude)
        })
        
        if is_fall:
            self.fall_count += 1
        
        self.frame_count += 1
        
        # Create state dictionary
        state_dict = {
            'rapid_descent': self.STATE_RAPID_DESCENT in self.current_states,
            'orientation_flip': self.STATE_ORIENTATION_FLIP in self.current_states,
            'stillness': self.STATE_STILLNESS in self.current_states,
            'all_conditions_met': is_fall
        }
        
        return is_fall, state_dict
    
    def _check_rapid_descent(self) -> bool:
        """Check if rapid descent condition is met."""
        if len(self.v_history) < self.rapid_descent_frames:
            return False
        
        # Check if last N frames all have v < threshold
        return all(v < self.v_threshold for v in self.v_history)
    
    def _check_orientation_flip(self, torso_angle: float) -> bool:
        """Check if orientation flip condition is met."""
        return torso_angle >= self.alpha_threshold
    
    def _check_stillness(self) -> bool:
        """Check if stillness condition is met."""
        if len(self.m_history) < self.stillness_frames:
            return False
        
        # Check if last N frames all have m <= threshold
        return all(m <= self.m_threshold for m in self.m_history)
    
    def _check_all_conditions_met(self) -> bool:
        """
        Check if all three conditions have been met within the sliding window.
        
        Returns True if there exists at least one frame in the window where
        all three conditions are simultaneously active.
        """
        if len(self.state_history) == 0:
            return False
        
        # Check if all three states are currently active
        required_states = {
            self.STATE_RAPID_DESCENT,
            self.STATE_ORIENTATION_FLIP,
            self.STATE_STILLNESS
        }
        
        # Check current frame first (most recent)
        if required_states.issubset(self.current_states):
            return True
        
        # Check if all three conditions have been active at any point in the window
        # (not necessarily simultaneously, but within the window)
        rapid_descent_seen = False
        orientation_flip_seen = False
        stillness_seen = False
        
        for states in self.state_history:
            if self.STATE_RAPID_DESCENT in states:
                rapid_descent_seen = True
            if self.STATE_ORIENTATION_FLIP in states:
                orientation_flip_seen = True
            if self.STATE_STILLNESS in states:
                stillness_seen = True
        
        # All three must have been seen in the window
        return rapid_descent_seen and orientation_flip_seen and stillness_seen
    
    def get_state_string(self) -> str:
        """Get human-readable string of current active states."""
        if not self.current_states:
            return self.STATE_NORMAL
        
        # Return states in priority order
        if self.STATE_STILLNESS in self.current_states:
            return self.STATE_STILLNESS
        elif self.STATE_ORIENTATION_FLIP in self.current_states:
            return self.STATE_ORIENTATION_FLIP
        elif self.STATE_RAPID_DESCENT in self.current_states:
            return self.STATE_RAPID_DESCENT
        else:
            return self.STATE_NORMAL
    
    def get_statistics(self) -> Dict:
        """Get FSM statistics."""
        return {
            'total_frames': self.frame_count,
            'fsm_falls_detected': self.fall_count,
            'thresholds': {
                'vertical_velocity': self.v_threshold,
                'torso_angle': self.alpha_threshold,
                'motion_magnitude': self.m_threshold,
                'rapid_descent_frames': self.rapid_descent_frames,
                'stillness_frames': self.stillness_frames
            }
        }
    
    def save_log(self, output_path: Path):
        """Save FSM state log to CSV."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            if not self.state_log:
                return
            
            fieldnames = self.state_log[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.state_log)
        
        print(f"FSM log saved to: {output_path}")


def load_fsm_config(config_path: Optional[str] = None) -> Dict:
    """
    Load FSM configuration from JSON file.
    
    Args:
        config_path: Path to FSM config JSON file
        
    Returns:
        Dictionary of FSM parameters
    """
    default_config = {
        'v_threshold': -0.28,
        'alpha_threshold': 65.0,
        'm_threshold': 0.02,
        'rapid_descent_frames': 2,
        'stillness_frames': 12,
        'window_size': 90
    }
    
    if config_path is None:
        return default_config
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Warning: FSM config not found at {config_path}, using defaults")
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Remove comments key if present
        config.pop('_comments', None)

        # Merge with defaults (in case some keys are missing)
        merged_config = default_config.copy()
        merged_config.update(config)

        print(f"✓ FSM config loaded from: {config_path}")
        return merged_config

    except Exception as e:
        print(f"Warning: Failed to load FSM config: {e}, using defaults")
        return default_config

