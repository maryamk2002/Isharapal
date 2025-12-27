#!/usr/bin/env python3
"""
MediaPipe utilities for hand landmark extraction and processing.
Optimized for Pakistan Sign Language recognition with robust error handling.

This module provides:
- Efficient hand landmark extraction using MediaPipe
- Quality assessment of detected hands
- Batch processing support for videos
- Error recovery mechanisms
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Constants
FEATURE_SIZE = 189  # Padded feature dimension (126 + padding)
HAND_LANDMARKS_COUNT = 21  # MediaPipe hand landmarks per hand
COORDINATES_PER_LANDMARK = 3  # x, y, z
MAX_HANDS = 2  # Maximum hands to detect


class HandednessLabel(Enum):
    """Enum for handedness labels."""
    LEFT = "Left"
    RIGHT = "Right"
    UNKNOWN = "Unknown"


@dataclass
class HandLandmarks:
    """Container for hand landmark data."""
    landmarks: np.ndarray
    handedness: str
    confidence: float
    quality_score: float


class MediaPipeProcessor:
    """
    MediaPipe processor for hand landmark extraction with robust error handling.
    
    Features:
    - Real-time hand detection and tracking
    - Landmark extraction with validation
    - Quality scoring and confidence assessment
    - Batch processing support for videos
    - Automatic error recovery
    
    Attributes:
        min_detection_confidence: Minimum confidence for hand detection
        min_tracking_confidence: Minimum confidence for hand tracking  
        model_complexity: Model complexity (0=fast, 1=balanced, 2=accurate)
        max_num_hands: Maximum number of hands to detect (1-2)
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0,
        max_num_hands: int = 2,
        static_image_mode: bool = False
    ):
        """
        Initialize MediaPipe processor with validation.
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for hand tracking (0.0-1.0)
            model_complexity: Model complexity (0=fast, 1=balanced, 2=accurate)
            max_num_hands: Maximum number of hands to detect (1-2)
            static_image_mode: Whether to treat images as static (False for video)
            
        Raises:
            ValueError: If parameters are out of valid range
        """
        # Validate parameters
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError(f"min_detection_confidence must be in [0, 1], got {min_detection_confidence}")
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError(f"min_tracking_confidence must be in [0, 1], got {min_tracking_confidence}")
        if model_complexity not in (0, 1, 2):
            raise ValueError(f"model_complexity must be 0, 1, or 2, got {model_complexity}")
        if not 1 <= max_num_hands <= 2:
            raise ValueError(f"max_num_hands must be 1 or 2, got {max_num_hands}")
        
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.max_num_hands = max_num_hands
        self.static_image_mode = static_image_mode
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self._initialize_hands()
        
        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Error tracking for recovery
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
        
        logger.info(f"[OK] MediaPipe processor initialized")
        logger.info(f"  Detection confidence: {min_detection_confidence}")
        logger.info(f"  Tracking confidence: {min_tracking_confidence}")
        logger.info(f"  Model complexity: {model_complexity}")
        logger.info(f"  Max hands: {max_num_hands}")
    
    def _initialize_hands(self) -> None:
        """Initialize or reinitialize MediaPipe Hands."""
        try:
            if self.hands is not None:
                self.hands.close()
                
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=self.model_complexity
            )
            self._consecutive_errors = 0
            logger.debug("MediaPipe Hands initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Hands: {e}")
            raise RuntimeError(f"MediaPipe initialization failed: {e}")
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a single frame.
        OPTIMIZED FOR DEMO: Downscale for speed, accept single hand.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            Flattened landmarks array (189 features) or None if extraction fails
        """
        # Validate input
        if frame is None or frame.size == 0:
            return None
        
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None
        
        try:
            orig_h, orig_w = frame.shape[:2]
            
            # Downscale to ~320x240 for faster processing
            target_w, target_h = 320, 240
            small_frame = cv2.resize(frame, (target_w, target_h))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            # No hands detected
            if not results.multi_hand_landmarks:
                return None
            
            # Extract landmarks (accept single hand or both)
            all_keypoints = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                hand_keypoints = []
                for landmark in hand_landmarks.landmark:
                    # Scale landmarks back to original resolution coordinates
                    # (though normalized, this ensures consistency if needed later)
                    hand_keypoints.extend([landmark.x, landmark.y, landmark.z])
                
                all_keypoints.extend(hand_keypoints)
            
            # Pad to 126 features (accept 1 or 2 hands)
            while len(all_keypoints) < 126:
                all_keypoints.append(0.0)
            
            all_keypoints = all_keypoints[:126]
            
            # Pad to 189 for model
            while len(all_keypoints) < FEATURE_SIZE:
                all_keypoints.append(0.0)
            
            self._consecutive_errors = 0
            return np.array(all_keypoints[:FEATURE_SIZE], dtype=np.float32)
            
        except cv2.error as e:
            logger.error(f"OpenCV error during landmark extraction: {e}")
            self._handle_extraction_error()
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting landmarks: {e}")
            self._handle_extraction_error()
            return None
    
    def _handle_extraction_error(self) -> None:
        """Handle extraction errors with recovery mechanism."""
        self._consecutive_errors += 1
        
        if self._consecutive_errors >= self._max_consecutive_errors:
            logger.warning(
                f"Reached {self._consecutive_errors} consecutive errors. Reinitializing MediaPipe..."
            )
            try:
                self._initialize_hands()
                logger.info("[OK] MediaPipe reinitialized after consecutive errors")
            except Exception as e:
                logger.error(f"Failed to reinitialize MediaPipe: {e}")
                # Reset counter to avoid infinite reinit attempts
                self._consecutive_errors = 0
    
    def extract_landmarks_with_quality(self, frame: np.ndarray) -> Optional[HandLandmarks]:
        """
        Extract hand landmarks with quality assessment.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            HandLandmarks object or None if no hands detected
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                return None
            
            # Extract landmarks and quality metrics
            all_landmarks = []
            handedness_list = []
            confidence_scores = []
            
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Extract landmarks
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.extend([landmark.x, landmark.y, landmark.z])
                
                all_landmarks.extend(hand_points)
                handedness_list.append(handedness.classification[0].label)
                confidence_scores.append(handedness.classification[0].score)
            
            if len(all_landmarks) < FEATURE_SIZE:
                all_landmarks.extend([0.0] * (FEATURE_SIZE - len(all_landmarks)))
            elif len(all_landmarks) > FEATURE_SIZE:
                all_landmarks = all_landmarks[:FEATURE_SIZE]
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(all_landmarks, confidence_scores)
            
            return HandLandmarks(
                landmarks=np.array(all_landmarks, dtype=np.float32),
                handedness=", ".join(handedness_list),
                confidence=np.mean(confidence_scores) if confidence_scores else 0.0,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Error extracting landmarks with quality: {e}")
            return None
    
    def _calculate_quality_score(self, landmarks: List[float], confidence_scores: List[float]) -> float:
        """
        Calculate quality score for extracted landmarks.
        
        Args:
            landmarks: Extracted landmarks
            confidence_scores: Confidence scores for each hand
        
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Base score from confidence
            base_score = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Check for valid landmark ranges
            landmarks_array = np.array(landmarks)
            valid_landmarks = np.sum((landmarks_array >= 0) & (landmarks_array <= 1))
            landmark_ratio = valid_landmarks / len(landmarks)
            
            # Check for hand movement (variance in coordinates)
            if len(landmarks) >= 63:  # At least one hand
                hand1_coords = landmarks_array[:63].reshape(-1, 3)
                if len(hand1_coords) > 0:
                    coord_variance = np.var(hand1_coords, axis=0)
                    movement_score = min(1.0, np.mean(coord_variance) * 10)  # Scale variance
                else:
                    movement_score = 0.0
            else:
                movement_score = 0.0
            
            # Combine scores
            quality_score = (base_score * 0.4 + landmark_ratio * 0.4 + movement_score * 0.2)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: Input frame
            landmarks: Landmarks array (126 features)
        
        Returns:
            Frame with drawn landmarks
        """
        try:
            # Convert landmarks back to MediaPipe format
            if len(landmarks) < 126:
                return frame
            
            # Reshape landmarks
            landmarks_reshaped = landmarks[:126].reshape(-1, 3)
            
            # Create MediaPipe landmark objects
            mp_landmarks = []
            for point in landmarks_reshaped:
                if np.any(point != 0):  # Skip zero-padded points
                    landmark = mp.framework.formats.landmark_pb2.Landmark()
                    landmark.x = point[0]
                    landmark.y = point[1]
                    landmark.z = point[2]
                    mp_landmarks.append(landmark)
            
            # Draw landmarks
            if mp_landmarks:
                # Create hand landmarks object
                hand_landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                hand_landmarks.landmark.extend(mp_landmarks)
                
                # Draw on frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return frame
    
    def process_video(self, video_path: str) -> List[np.ndarray]:
        """
        Process entire video and extract landmarks for all frames.
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of landmark arrays for each frame
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []
            
            landmarks_sequence = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks = self.extract_landmarks(frame)
                if landmarks is not None:
                    landmarks_sequence.append(landmarks)
            
            cap.release()
            return landmarks_sequence
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return []
    
    def get_hand_bbox(self, landmarks: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box for detected hands.
        
        Args:
            landmarks: Landmarks array (126 features)
        
        Returns:
            Bounding box (x, y, width, height) or None
        """
        try:
            if len(landmarks) < 63:
                return None
            
            # Get first hand landmarks
            hand_landmarks = landmarks[:63].reshape(-1, 3)
            
            # Filter out zero-padded points
            valid_points = hand_landmarks[np.any(hand_landmarks != 0, axis=1)]
            
            if len(valid_points) == 0:
                return None
            
            # Calculate bounding box
            x_min, y_min = np.min(valid_points[:, :2], axis=0)
            x_max, y_max = np.max(valid_points[:, :2], axis=0)
            
            width = x_max - x_min
            height = y_max - y_min
            
            return (int(x_min), int(y_min), int(width), int(height))
            
        except Exception as e:
            logger.error(f"Error calculating hand bbox: {e}")
            return None
    
    def close(self):
        """Close MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


# Utility functions
def validate_landmarks(landmarks: np.ndarray) -> bool:
    """
    Validate extracted landmarks.
    
    Args:
        landmarks: Landmarks array
    
    Returns:
        True if landmarks are valid, False otherwise
    """
    if landmarks is None or len(landmarks) != FEATURE_SIZE:
        return False
    
    # Check for reasonable coordinate ranges
    # MediaPipe normalizes x, y to [0, 1], but z can be negative (depth)
    # Check only first 126 features (actual hand data, not padding)
    landmarks_hands = landmarks[:126].reshape(-1, 3)
    
    # Validate x coordinates (every 3rd starting at 0)
    x_coords = landmarks_hands[:, 0]
    if np.any((x_coords < 0) | (x_coords > 1)):
        return False
    
    # Validate y coordinates (every 3rd starting at 1)
    y_coords = landmarks_hands[:, 1]
    if np.any((y_coords < 0) | (y_coords > 1)):
        return False
    
    # z coordinates can be negative (depth), so we don't validate range
    # Just check they're reasonable (not extremely large)
    z_coords = landmarks_hands[:, 2]
    if np.any(np.abs(z_coords) > 10):  # Sanity check
        return False
    
    # Check for sufficient non-zero landmarks
    non_zero_count = np.sum(landmarks != 0)
    if non_zero_count < 20:  # At least 20 non-zero landmarks
        return False
    
    return True


def landmarks_to_sequence(landmarks_list: List[np.ndarray]) -> np.ndarray:
    """
    Convert list of landmarks to sequence array.
    
    Args:
        landmarks_list: List of landmark arrays
    
    Returns:
        Sequence array of shape (sequence_length, FEATURE_SIZE)
    """
    if not landmarks_list:
        return np.array([])
    
    return np.array(landmarks_list)


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = MediaPipeProcessor()
    
    # Test with dummy frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Extract landmarks
    landmarks = processor.extract_landmarks(test_frame)
    print(f"Landmarks shape: {landmarks.shape if landmarks is not None else None}")
    
    # Extract with quality
    hand_data = processor.extract_landmarks_with_quality(test_frame)
    if hand_data:
        print(f"Quality score: {hand_data.quality_score}")
        print(f"Confidence: {hand_data.confidence}")
    
    # Close processor
    processor.close()

