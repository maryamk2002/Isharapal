#!/usr/bin/env python3
"""
Real-time predictor V2 for PSL recognition system.
FIXES: Lagging, flickering, old predictions, stopping mid-session.

Key Improvements:
- Smaller sliding window (24 frames for demo, configurable)
- Faster response time (<1 second latency)
- Better stability filtering (no flickering)
- Automatic buffer clearing when hands disappear
- Synchronized keypoint visualization
"""

import time
import logging
import numpy as np
import json
import torch
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import threading
from pathlib import Path

from config import inference_config

logger = logging.getLogger(__name__)


class RealTimePredictorV2:
    """
    Enhanced real-time predictor with optimized performance.
    
    Improvements over V1:
    - Reduced latency (24 frames default, configurable)
    - Better stability filtering
    - Auto-reset on hand disappearance
    - Synchronized predictions
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        labels: List[str],
        device: Optional[torch.device] = None,
        # V2 Parameters (optimized for real-time)
        sliding_window_size: int = 32,  # Faster response (was 60)
        min_prediction_frames: int = 32,  # Start predicting sooner
        stability_votes: int = 5,  # Need 3 out of 5 votes
        stability_threshold: int = 3,  # Lower for faster transitions
        min_confidence: float = 0.55,  # Confidence threshold
        reset_on_no_hands: bool = True,  # Auto-reset buffer
    ):
        """
        Initialize enhanced real-time predictor.
        
        Args:
            model: Trained PyTorch model
            config: Model configuration dictionary
            labels: List of class labels
            device: Device to run inference on (auto-detects if None)
            sliding_window_size: Number of frames in sliding window (default: 32)
            min_prediction_frames: Minimum frames before making predictions (default: 32)
            stability_votes: How many recent predictions to consider (default: 5)
            stability_threshold: How many votes needed for stable prediction (default: 3)
            min_confidence: Minimum confidence threshold (default: 0.55)
            reset_on_no_hands: Whether to reset buffer when hands disappear (default: True)
        """
        # Validate inputs
        if model is None:
            raise ValueError("Model cannot be None")
        if not config or not isinstance(config, dict):
            raise ValueError("Config must be a non-empty dictionary")
        if not labels or not isinstance(labels, list) or len(labels) == 0:
            raise ValueError("Labels must be a non-empty list")
        
        self.model = model
        self.config = config
        self.labels = labels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # V2 Configuration (optimized parameters)
        self.sliding_window_size = sliding_window_size
        self.min_prediction_frames = min_prediction_frames
        self.stability_votes = stability_votes
        self.stability_threshold = stability_threshold
        self.min_confidence = min_confidence
        self.reset_on_no_hands = reset_on_no_hands
        
        # Load per-sign confidence thresholds
        self.sign_thresholds = self._load_sign_thresholds()
        logger.info(f"  Per-sign thresholds: {len(self.sign_thresholds)} signs loaded")
        
        # Get model's expected sequence length
        self.model_sequence_length = config.get('sequence_length', 60)
        
        # Move model to device and set to eval mode
        try:
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model on device {self.device}: {e}")
        
        # Sliding window buffer for frames
        self.frame_buffer = deque(maxlen=self.sliding_window_size)
        
        # Prediction history for stability voting
        self.prediction_history = deque(maxlen=self.stability_votes)
        
        # Last stable prediction tracking
        self.last_stable_prediction = None
        self.last_stable_confidence = 0.0
        self.last_stable_time = None  # Use None instead of 0.0 to avoid falsy check bug
        
        # NEW: Exponential moving average for softmax smoothing
        self.ema_alpha = 0.7  # Higher = more responsive (0.7 = 70% new, 30% old)
        self.ema_probabilities = None  # Will be initialized on first prediction
        self.last_raw_prediction = None  # Track raw prediction for EMA reset
        
        # NEW: Low confidence threshold for "searching" state
        self.low_confidence_threshold = 0.45  # Slightly lower for better responsiveness
        
        # Performance monitoring
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'avg_prediction_time': 0.0,
            'avg_confidence': 0.0,
            'buffer_resets': 0,
        }
        
        # Thread lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Processing state flag for graceful shutdown
        # When True, the predictor is actively running inference
        # Stop commands should wait for this to become False
        self.is_processing = False
        self._stop_requested = False  # Flag to request graceful stop
        
        logger.info(f"[OK] Real-time predictor V2 initialized on {self.device}")
        logger.info(f"  Model: {type(model).__name__}")
        logger.info(f"  Classes: {len(self.labels)}")
        logger.info(f"  Sliding window: {self.sliding_window_size} frames")
        logger.info(f"  Model sequence length: {self.model_sequence_length}")
        logger.info(f"  Stability: {self.stability_threshold}/{self.stability_votes} votes")
        logger.info(f"  Labels: {', '.join(self.labels)}")
    
    def _load_sign_thresholds(self) -> dict:
        """
        Load per-sign confidence thresholds from config.
        
        Returns:
            Dict mapping label → threshold, or empty dict if not found
        """
        threshold_file = Path(__file__).parent.parent / 'config' / 'sign_thresholds.json'
        
        if threshold_file.exists():
            try:
                with open(threshold_file, 'r') as f:
                    thresholds = json.load(f)
                logger.info(f"  Loaded {len(thresholds)} per-sign thresholds from {threshold_file.name}")
                return thresholds
            except Exception as e:
                logger.warning(f"Failed to load sign thresholds: {e}")
                return {}
        else:
            logger.info("  No sign_thresholds.json found, using default threshold for all signs")
            return {}
    
    def add_frame(self, keypoints: np.ndarray) -> None:
        """
        Add a frame to the sliding window buffer.
        
        Args:
            keypoints: Extracted keypoints from current frame (shape: (features,))
        """
        with self.lock:
            if keypoints is None or len(keypoints) == 0:
                if self.reset_on_no_hands and len(self.frame_buffer) > 0:
                    # Clear buffer when hands disappear (prevents old predictions)
                    self.clear_buffer()
                    logger.info("[RESET] Buffer cleared: no hands detected")
                return
            
            # Ensure keypoints are 1D
            if keypoints.ndim == 2:
                keypoints = keypoints.flatten()
            
            # Add to sliding window
            self.frame_buffer.append(keypoints)
    
    def predict_current(
        self,
        return_all_predictions: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Make prediction on current sliding window with stability filtering.
        
        Returns:
            Dictionary with prediction results or None if not ready:
            {
                'prediction': str,          # Predicted label
                'confidence': float,        # Confidence score
                'is_stable': bool,          # Whether prediction is stable
                'is_new': bool,             # Whether this is a new stable prediction
                'all_predictions': list,    # All class predictions (if requested)
                'buffer_size': int,         # Current buffer size
                'ready': bool,              # Whether buffer has enough frames
            }
        """
        with self.lock:
            # Check if buffer has enough frames
            if len(self.frame_buffer) < self.min_prediction_frames:
                return {
                    'ready': False,
                    'buffer_size': len(self.frame_buffer),
                    'min_required': self.min_prediction_frames,
                    'status': 'collecting_frames'
                }
            
            try:
                # Set processing flag (for graceful shutdown)
                self.is_processing = True
                
                start_time = time.time()
                
                # Get current sequence from buffer
                sequence = np.array(list(self.frame_buffer))
                
                # Pad or truncate to model's expected length
                if len(sequence) != self.model_sequence_length:
                    sequence = self._prepare_sequence(sequence, self.model_sequence_length)
                
                # Convert to tensor
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                
                # Make prediction
                with torch.no_grad():
                    logits = self.model(sequence_tensor)
                    raw_probabilities = torch.softmax(logits, dim=1)
                    raw_probs_np = raw_probabilities.squeeze().cpu().numpy()
                    
                    # Get raw prediction (before smoothing)
                    raw_predicted_idx = np.argmax(raw_probs_np)
                    raw_predicted_label = self.labels[raw_predicted_idx]
                    raw_confidence = float(raw_probs_np[raw_predicted_idx])
                    
                    # Reset EMA if raw prediction changed (for responsiveness)
                    if self.last_raw_prediction is not None and raw_predicted_label != self.last_raw_prediction:
                        # New sign detected - reset EMA for faster response
                        self.ema_probabilities = raw_probs_np.copy()
                        logger.debug(f"EMA reset: {self.last_raw_prediction} -> {raw_predicted_label}")
                    elif self.ema_probabilities is None:
                        self.ema_probabilities = raw_probs_np.copy()
                    else:
                        # EMA: new_ema = alpha * new_value + (1 - alpha) * old_ema
                        self.ema_probabilities = (
                            self.ema_alpha * raw_probs_np + 
                            (1 - self.ema_alpha) * self.ema_probabilities
                        )
                    
                    self.last_raw_prediction = raw_predicted_label
                    
                    # Use smoothed probabilities for prediction
                    smoothed_probs = self.ema_probabilities
                    predicted_idx = np.argmax(smoothed_probs)
                    confidence_score = float(smoothed_probs[predicted_idx])
                    predicted_label = self.labels[predicted_idx]
                
                # Get all predictions if requested (using smoothed probabilities)
                all_predictions = []
                if return_all_predictions:
                    for label, prob in zip(self.labels, smoothed_probs):
                        all_predictions.append({'label': label, 'confidence': float(prob)})
                    # Sort by confidence
                    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Get per-sign threshold or use default
                sign_threshold = self.sign_thresholds.get(predicted_label, self.min_confidence)
                
                # Add to prediction history if confidence is sufficient
                if confidence_score >= sign_threshold:
                    self.prediction_history.append({
                        'label': predicted_label,
                        'confidence': confidence_score,
                        'timestamp': time.time()
                    })
                    logger.debug(f"Prediction: {predicted_label} ({confidence_score:.3f}, threshold={sign_threshold:.2f})")
                
                # Apply stability voting
                stable_prediction = self._get_stable_prediction()
                
                # Determine if this is a new stable prediction
                is_new = False
                if stable_prediction is not None:
                    stable_label, stable_confidence = stable_prediction
                    
                    # Check if this is truly new:
                    # 1. Different label than last stable, OR
                    # 2. Same label but enough time has passed (allow repeated signs)
                    time_since_last = time.time() - self.last_stable_time if self.last_stable_time is not None else 999
                    is_different_label = (stable_label != self.last_stable_prediction)
                    is_timeout_passed = (time_since_last > 30.0)  # 30 seconds - only re-trigger same sign after long pause
                    
                    is_new = is_different_label or is_timeout_passed
                    
                    if is_new:
                        self.last_stable_prediction = stable_label
                        self.last_stable_confidence = stable_confidence
                        self.last_stable_time = time.time()
                        logger.info(f"NEW stable prediction: {stable_label} ({stable_confidence:.3f})")
                        
                        # Only clear history if this is a DIFFERENT label
                        # Don't clear on timeout for same label (user might be holding continuously)
                        if is_different_label:
                            self.prediction_history.clear()
                            logger.debug("Prediction history cleared for next sign")
                        else:
                            logger.debug(f"Same sign re-triggered after timeout ({time_since_last:.1f}s)")
                    else:
                        logger.debug(f"Stable prediction held: {stable_label} ({stable_confidence:.3f})")
                
                # Update statistics
                prediction_time = time.time() - start_time
                self._update_stats(prediction_time, confidence_score, True)
                
                # Determine status based on confidence
                if confidence_score < self.low_confidence_threshold:
                    status = 'low_confidence'  # Frontend should show "Searching..."
                elif stable_prediction is None:
                    status = 'collecting_stability'
                else:
                    status = 'success'
                
                # Prepare result
                result = {
                    'ready': True,
                    'prediction': predicted_label,
                    'confidence': confidence_score,
                    'stable_prediction': stable_prediction[0] if stable_prediction else None,
                    'stable_confidence': stable_prediction[1] if stable_prediction else 0.0,
                    'is_stable': stable_prediction is not None,
                    'is_new': is_new,
                    'is_low_confidence': confidence_score < self.low_confidence_threshold,
                    'all_predictions': all_predictions,
                    'buffer_size': len(self.frame_buffer),
                    'prediction_time_ms': prediction_time * 1000,
                    'status': status
                }
                
                logger.debug(
                    f"Prediction: {predicted_label} ({confidence_score:.3f}) | "
                    f"Stable: {result['stable_prediction']} ({result['stable_confidence']:.3f}) | "
                    f"New: {is_new}"
                )
                
                return result
            
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                self._update_stats(0.0, 0.0, False)
                return {
                    'ready': True,
                    'status': 'error',
                    'error': str(e),
                    'buffer_size': len(self.frame_buffer)
                }
            
            finally:
                # Always clear processing flag when done
                self.is_processing = False
    
    def _get_stable_prediction(self) -> Optional[Tuple[str, float]]:
        """
        Get stable prediction using majority voting.
        
        Returns:
            Tuple of (label, confidence) if stable, None otherwise
        """
        if len(self.prediction_history) < self.stability_threshold:
            return None
        
        # Only look at RECENT predictions (last N frames from the history)
        # This prevents old predictions from blocking new ones
        recent_predictions = list(self.prediction_history)[-self.stability_votes:]
        
        # Count occurrences of each label
        label_counts = {}
        label_confidences = {}
        
        for pred in recent_predictions:
            label = pred['label']
            confidence = pred['confidence']
            
            if label not in label_counts:
                label_counts[label] = 0
                label_confidences[label] = []
            
            label_counts[label] += 1
            label_confidences[label].append(confidence)
        
        # Find most frequent label
        most_frequent_label = max(label_counts.keys(), key=lambda x: label_counts[x])
        count = label_counts[most_frequent_label]
        
        # Check if it meets stability threshold
        if count >= self.stability_threshold:
            avg_confidence = sum(label_confidences[most_frequent_label]) / len(label_confidences[most_frequent_label])
            return most_frequent_label, avg_confidence
        
        return None
    
    def _prepare_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """
        Prepare sequence to match model's expected length.
        
        Args:
            sequence: Input sequence of shape (seq_len, features)
            target_length: Target sequence length
            
        Returns:
            Sequence adjusted to target length
        """
        current_length = len(sequence)
        
        if current_length == target_length:
            return sequence
        
        elif current_length < target_length:
            # Pad with zeros at the beginning (preserve recent frames)
            padding_length = target_length - current_length
            padding = np.zeros((padding_length, sequence.shape[1]), dtype=sequence.dtype)
            return np.vstack([padding, sequence])
        
        else:
            # Truncate from the beginning (keep most recent frames)
            return sequence[-target_length:]
    
    def _update_stats(self, prediction_time: float, confidence: float, success: bool):
        """Update performance statistics."""
        self.stats['total_predictions'] += 1
        
        if success:
            self.stats['successful_predictions'] += 1
            # Exponential moving average
            self.stats['avg_prediction_time'] = (
                self.stats['avg_prediction_time'] * 0.9 + prediction_time * 0.1
            )
            self.stats['avg_confidence'] = (
                self.stats['avg_confidence'] * 0.9 + confidence * 0.1
            )
        else:
            self.stats['failed_predictions'] += 1
    
    def clear_buffer(self):
        """Clear frame buffer and prediction history."""
        with self.lock:
            self.frame_buffer.clear()
            self.prediction_history.clear()
            self.last_stable_prediction = None
            self.last_stable_confidence = 0.0
            self.last_stable_time = None  # Reset timer
            self.ema_probabilities = None  # Reset EMA smoothing
            self.last_raw_prediction = None  # Reset raw prediction tracking
            self.stats['buffer_resets'] += 1
            logger.info("[RESET] Buffer and prediction state cleared")
    
    def reset_stable_prediction(self):
        """Reset only the stable prediction tracking (keep buffer)."""
        with self.lock:
            self.prediction_history.clear()
            self.last_stable_prediction = None
            self.last_stable_confidence = 0.0
            self.last_stable_time = None  # Reset timer
            logger.debug("Stable prediction state reset (buffer kept)")
    
    def request_stop(self):
        """Request a graceful stop. Processing will complete current inference."""
        self._stop_requested = True
        logger.info("Graceful stop requested")
    
    def wait_for_processing(self, timeout_sec: float = 5.0) -> bool:
        """
        Wait for current processing to complete.
        
        Args:
            timeout_sec: Maximum time to wait (default: 5 seconds)
            
        Returns:
            True if processing completed, False if timeout occurred
        """
        start_time = time.time()
        while self.is_processing:
            if time.time() - start_time > timeout_sec:
                logger.warning(f"Timeout waiting for processing to complete after {timeout_sec}s")
                return False
            time.sleep(0.05)  # 50ms poll interval
        return True
    
    def is_stop_requested(self) -> bool:
        """Check if a graceful stop has been requested."""
        return self._stop_requested
    
    def clear_stop_request(self):
        """Clear the stop request flag (for restart)."""
        self._stop_requested = False
    
    def get_current_sequence(self) -> Optional[np.ndarray]:
        """
        Get the current frame buffer as a numpy array.
        Useful for saving feedback samples.
        
        Returns:
            Numpy array of shape (N, 189) or None if buffer is empty
        """
        with self.lock:
            if len(self.frame_buffer) == 0:
                return None
            return np.array(list(self.frame_buffer))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.lock:
            return {
                **self.stats,
                'buffer_size': len(self.frame_buffer),
                'history_size': len(self.prediction_history),
                'last_stable_prediction': self.last_stable_prediction,
                'last_stable_confidence': self.last_stable_confidence,
            }
    
    def reset_stats(self):
        """Reset performance statistics."""
        with self.lock:
            self.stats = {
                'total_predictions': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'avg_prediction_time': 0.0,
                'avg_confidence': 0.0,
                'buffer_resets': 0,
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': type(self.model).__name__,
            'num_classes': len(self.labels),
            'labels': self.labels,
            'device': str(self.device),
            'config': self.config,
            'sliding_window_size': self.sliding_window_size,
            'min_prediction_frames': self.min_prediction_frames,
            'stability_threshold': f"{self.stability_threshold}/{self.stability_votes}",
            'stats': self.get_stats()
        }


# Backward compatibility: alias for existing code
RealTimePredictor = RealTimePredictorV2


# Example usage
if __name__ == "__main__":
    print("Real-time Predictor V2 - Example Usage")
    print("=" * 60)
    
    # This would typically be used with a trained model
    from models.tcn_model import create_model
    
    model = create_model(input_dim=189, num_classes=4)
    config = {'sequence_length': 60, 'input_dim': 189, 'num_classes': 4}
    labels = ['2-Hay', 'Alifmad', 'Aray', 'Jeem']
    
    # Create V2 predictor with optimized settings
    # Import config for consistency
    from config_v2 import inference_config_v2
    
    predictor = RealTimePredictorV2(
        model=model,
        config=config,
        labels=labels,
        sliding_window_size=inference_config_v2.SLIDING_WINDOW_SIZE,
        min_prediction_frames=inference_config_v2.MIN_FRAMES_FOR_PREDICTION,
        stability_votes=5,
        stability_threshold=3,
        min_confidence=0.55
    )
    
    print("\n Simulating frame processing...")
    print("-" * 60)
    
    # Simulate adding frames
    for i in range(40):
        # Generate dummy keypoints
        dummy_keypoints = np.random.randn(189)
        predictor.add_frame(dummy_keypoints)
        
        # Try to predict
        result = predictor.predict_current(return_all_predictions=True)
        
        if result and result.get('ready'):
            if result.get('is_new'):
                print(f"Frame {i+1}: NEW STABLE PREDICTION: {result['stable_prediction']} "
                      f"({result['stable_confidence']:.3f})")
            elif result.get('is_stable'):
                print(f"Frame {i+1}: Stable: {result['stable_prediction']} "
                      f"({result['stable_confidence']:.3f}) [holding]")
            else:
                print(f"Frame {i+1}: Raw: {result['prediction']} "
                      f"({result['confidence']:.3f}) [not stable yet]")
        else:
            print(f"Frame {i+1}: Collecting frames... ({result['buffer_size']}/{result['min_required']})")
    
    # Print stats
    print("\n" + "=" * 60)
    print("Final Statistics:")
    print("-" * 60)
    stats = predictor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n Model Info:")
    print("-" * 60)
    info = predictor.get_model_info()
    for key, value in info.items():
        if key != 'stats':  # Already printed
            print(f"  {key}: {value}")

