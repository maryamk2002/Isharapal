#!/usr/bin/env python3
"""
Keypoint filtering and denoising for PSL recognition.
Handles jitter detection, stuck sequence detection, and moving average smoothing.
"""

import numpy as np
import hashlib
from collections import deque
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class KeypointFilter:
    """
    Filter and denoise incoming keypoints from MediaPipe.
    
    Features:
    - Moving average smoothing (reduces webcam jitter)
    - Jitter detection (flags noisy frames)
    - Stuck sequence detection (detects frozen hand positions)
    
    Why these techniques?
    - Moving average: Simple, fast, preserves gesture shape
    - Jitter detection: Prevents model from learning noise patterns
    - Stuck detection: Prevents buffer from filling with identical frames
    """
    
    def __init__(
        self,
        window_size: int = 5,
        jitter_threshold: float = 0.01,
        stuck_threshold_frames: int = 60,
        stuck_movement_threshold: float = 0.001
    ):
        """
        Initialize keypoint filter.
        
        Args:
            window_size: Number of frames for moving average (default: 5)
                Why 5? Enough to smooth jitter, not too large to lag gestures
            jitter_threshold: Max allowed movement for "stable" frame (default: 0.01)
                Why 0.01? ~1% of normalized coords, typical webcam noise level
            stuck_threshold_frames: Frames to check for identical data (default: 60)
                Why 60? ~4 seconds at 15 FPS, detects truly frozen frames
            stuck_movement_threshold: Not used (kept for compatibility)
        """
        self.window_size = window_size
        self.jitter_threshold = jitter_threshold
        self.stuck_threshold_frames = stuck_threshold_frames
        
        # Circular buffer for moving average
        self.keypoint_buffer = deque(maxlen=window_size)
        
        # Last keypoint for movement detection
        self.last_keypoint = None
        
        # Hash-based stuck detection (detects truly frozen frames)
        self.frame_hashes = deque(maxlen=stuck_threshold_frames)
        self.stuck_hash_threshold = 0.95  # 95% identical frames = stuck
        
        # Stats
        self.stats = {
            'total_frames': 0,
            'jittery_frames': 0,
            'stuck_detected': 0,
            'avg_movement': 0.0
        }
    
    def add_keypoint(self, keypoints: np.ndarray) -> None:
        """
        Add new keypoint to filter buffer.
        
        Args:
            keypoints: Raw keypoints from MediaPipe (shape: (189,))
        """
        if keypoints is None or len(keypoints) == 0:
            return
        
        # Ensure 1D array
        if keypoints.ndim > 1:
            keypoints = keypoints.flatten()
        
        # Add to buffer
        self.keypoint_buffer.append(keypoints.copy())
        self.stats['total_frames'] += 1
        
        # Compute frame hash for stuck detection
        frame_hash = hashlib.md5(keypoints.tobytes()).hexdigest()[:8]
        self.frame_hashes.append(frame_hash)
        
        # Update movement detection
        if self.last_keypoint is not None:
            movement = self._calculate_movement(self.last_keypoint, keypoints)
            self.stats['avg_movement'] = (
                (self.stats['avg_movement'] * (self.stats['total_frames'] - 1) + movement)
                / self.stats['total_frames']
            )
        
        self.last_keypoint = keypoints
    
    def get_filtered(self) -> Optional[np.ndarray]:
        """
        Get filtered keypoints (moving average).
        
        Returns:
            Filtered keypoints or None if buffer not full
        """
        if len(self.keypoint_buffer) == 0:
            return None
        
        # Return moving average of buffer
        return self._moving_average()
    
    def is_jittery(self) -> bool:
        """
        Check if current frame is jittery (noisy).
        
        Returns:
            True if jitter detected, False otherwise
        """
        if len(self.keypoint_buffer) < 2:
            return False
        
        # Compare last two frames
        movement = self._calculate_movement(
            self.keypoint_buffer[-2],
            self.keypoint_buffer[-1]
        )
        
        is_jitter = movement < self.jitter_threshold
        if is_jitter:
            self.stats['jittery_frames'] += 1
        
        return is_jitter
    
    def is_stuck(self) -> bool:
        """
        Check if sequence is stuck (identical frames, not just still hands).
        
        Uses frame hashing to detect truly frozen camera/identical data.
        Natural hand stillness (held signs) will NOT trigger this.
        
        Returns:
            True if stuck detected, False otherwise
        """
        if len(self.frame_hashes) < self.stuck_threshold_frames:
            return False
        
        # Count unique hashes in recent frames
        unique_hashes = len(set(self.frame_hashes))
        total_frames = len(self.frame_hashes)
        
        # If >95% of frames are identical (same hash) → stuck
        identical_ratio = 1.0 - (unique_hashes / total_frames)
        
        if identical_ratio >= self.stuck_hash_threshold:
            self.stats['stuck_detected'] += 1
            logger.warning(
                f"Stuck sequence detected: {identical_ratio*100:.1f}% identical frames "
                f"({unique_hashes}/{total_frames} unique)"
            )
            return True
        
        return False
    
    def reset(self) -> None:
        """Clear all buffers and reset state."""
        self.keypoint_buffer.clear()
        self.frame_hashes.clear()
        self.last_keypoint = None
        logger.debug("Keypoint filter reset")
    
    def get_stats(self) -> dict:
        """Get filtering statistics."""
        return self.stats.copy()
    
    # Private helper methods
    
    def _moving_average(self) -> np.ndarray:
        """
        Calculate moving average of keypoint buffer.
        
        Why moving average?
        - Simple and fast O(n) computation
        - Effective at removing high-frequency noise
        - Preserves gesture shape (low-pass filter)
        
        Returns:
            Average of all keypoints in buffer
        """
        if len(self.keypoint_buffer) == 0:
            return None
        
        # Stack all keypoints and compute mean
        stacked = np.stack(list(self.keypoint_buffer), axis=0)
        return np.mean(stacked, axis=0)
    
    def _calculate_movement(self, kp1: np.ndarray, kp2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two keypoint arrays.
        
        Args:
            kp1: First keypoint array
            kp2: Second keypoint array
        
        Returns:
            Average per-coordinate distance (normalized)
        """
        if kp1 is None or kp2 is None:
            return 0.0
        
        # Euclidean distance
        diff = kp2 - kp1
        distance = np.sqrt(np.sum(diff ** 2))
        
        # Normalize by number of coordinates
        return distance / len(kp1)

