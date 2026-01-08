#!/usr/bin/env python3
"""
Real-time predictor for PSL recognition system.
Optimized for low-latency inference with prediction smoothing.
"""

import time
import logging
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import threading

from config import inference_config

logger = logging.getLogger(__name__)


class RealTimePredictor:
    """
    Real-time predictor for sign language recognition.
    
    Features:
    - Low-latency inference
    - Prediction smoothing and temporal voting
    - Confidence thresholding
    - Performance monitoring
    - Thread-safe operations
    
    Compatible with:
    - Original TCN models
    - Advanced Spatiotemporal models
    - Any PyTorch model with compatible interface
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        labels: List[str],
        device: Optional[torch.device] = None
    ):
        """
        Initialize real-time predictor with validation.
        
        Args:
            model: Trained model (TCN, Advanced Spatiotemporal, etc.)
            config: Model configuration dictionary
            labels: List of class labels
            device: Device to run inference on (auto-detects if None)
            
        Raises:
            ValueError: If model, config, or labels are invalid
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
        
        # Move model to device and set to eval mode
        try:
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model on device {self.device}: {e}")
        
        # Validate config has required keys
        required_keys = ['sequence_length', 'num_classes']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            logger.warning(f"Config missing recommended keys: {missing_keys}")
        
        # Validate num_classes matches labels
        num_classes_config = config.get('num_classes', len(labels))
        if num_classes_config != len(labels):
            logger.warning(
                f"Config num_classes ({num_classes_config}) doesn't match labels count ({len(labels)})"
            )
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=inference_config.PREDICTION_HISTORY_SIZE)
        
        # Performance monitoring
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'avg_prediction_time': 0.0,
            'avg_confidence': 0.0
        }
        
        # Thread lock for thread-safe operations
        self.lock = threading.Lock()
        
        logger.info(f"[OK] Real-time predictor initialized on {self.device}")
        logger.info(f"  Model: {type(model).__name__}")
        logger.info(f"  Classes: {len(self.labels)}")
        logger.info(f"  Sequence length: {config.get('sequence_length', 'not specified')}")
        logger.info(f"  Labels: {', '.join(self.labels)}")
    
    def predict(
        self,
        sequence: np.ndarray,
        return_all_predictions: bool = True
    ) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
        """
        Make prediction on input sequence with proper validation.
        
        Args:
            sequence: Input sequence of shape (seq_len, features)
            return_all_predictions: Whether to return all class predictions
        
        Returns:
            Tuple of (predicted_label, confidence, all_predictions) or None if failed
        """
        try:
            start_time = time.time()
            
            with self.lock:
                # Validate input
                if sequence is None or len(sequence) == 0:
                    logger.warning("Empty sequence provided")
                    return None
                
                # Ensure sequence is 2D
                if sequence.ndim == 1:
                    sequence = sequence.reshape(1, -1)
                elif sequence.ndim != 2:
                    logger.error(f"Invalid sequence dimensions: {sequence.ndim}. Expected 2D array")
                    return None
                
                # Get target sequence length from config (trust the config!)
                target_length = self.config.get('sequence_length', inference_config.MODEL_SEQUENCE_LENGTH)
                
                # Validate feature dimensions
                expected_features = self.config.get('input_dim', 189)
                if sequence.shape[1] != expected_features:
                    logger.warning(
                        f"Feature dimension mismatch: got {sequence.shape[1]}, expected {expected_features}"
                    )
                
                # Pad or truncate sequence to model input length
                if len(sequence) != target_length:
                    sequence = self._pad_truncate_sequence(sequence, target_length)
                
                # Convert to tensor with device fallback
                try:
                    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                except RuntimeError as cuda_error:
                    # CUDA error - attempt fallback to CPU
                    if 'CUDA' in str(cuda_error).upper() or 'cuda' in str(cuda_error):
                        logger.warning(f"CUDA error, falling back to CPU: {cuda_error}")
                        self.device = torch.device("cpu")
                        self.model = self.model.to(self.device)
                        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    else:
                        raise
                
                # Make prediction with device fallback
                try:
                    with torch.no_grad():
                        logits = self.model(sequence_tensor)
                        probabilities = torch.softmax(logits, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, dim=1)
                        
                        predicted_label = self.labels[predicted_idx.item()]
                        confidence_score = confidence.item()
                except RuntimeError as cuda_error:
                    # CUDA error during inference - fallback to CPU
                    if 'CUDA' in str(cuda_error).upper() or 'cuda' in str(cuda_error):
                        logger.warning(f"CUDA error during inference, falling back to CPU: {cuda_error}")
                        self.device = torch.device("cpu")
                        self.model = self.model.to(self.device)
                        sequence_tensor = sequence_tensor.to(self.device)
                        
                        with torch.no_grad():
                            logits = self.model(sequence_tensor)
                            probabilities = torch.softmax(logits, dim=1)
                            confidence, predicted_idx = torch.max(probabilities, dim=1)
                            
                            predicted_label = self.labels[predicted_idx.item()]
                            confidence_score = confidence.item()
                    else:
                        raise
                
                # Get all predictions if requested
                all_predictions = []
                if return_all_predictions:
                    probs = probabilities.squeeze().cpu().numpy()
                    for i, (label, prob) in enumerate(zip(self.labels, probs)):
                        all_predictions.append((label, float(prob)))
                    
                    # Sort by probability
                    all_predictions.sort(key=lambda x: x[1], reverse=True)
                
                # Update statistics
                prediction_time = time.time() - start_time
                self._update_stats(prediction_time, confidence_score, True)
                
                logger.debug(f"Prediction: {predicted_label} (confidence: {confidence_score:.3f})")
                
                return predicted_label, confidence_score, all_predictions
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            self._update_stats(0.0, 0.0, False)
            return None
    
    def predict_with_smoothing(
        self,
        sequence: np.ndarray,
        smoothing_alpha: float = None
    ) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
        """
        Make prediction with temporal smoothing.
        
        Args:
            sequence: Input sequence
            smoothing_alpha: Smoothing factor for exponential moving average
        
        Returns:
            Smoothed prediction result
        """
        try:
            # Set default smoothing alpha if not provided
            if smoothing_alpha is None:
                smoothing_alpha = inference_config.SMOOTHING_ALPHA
            
            # Get current prediction (this already acquires lock internally)
            result = self.predict(sequence, return_all_predictions=True)
            if result is None:
                return None
            
            predicted_label, confidence, all_predictions = result
            
            # Add to history with lock for thread safety
            with self.lock:
                self.prediction_history.append({
                    'label': predicted_label,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
            
            # Apply smoothing if we have enough history
            if len(self.prediction_history) >= 3:
                smoothed_result = self._apply_temporal_smoothing(smoothing_alpha)
                if smoothed_result is not None:
                    return smoothed_result
            
            return result
        
        except Exception as e:
            logger.error(f"Smoothing prediction failed: {e}")
            return None
    
    def _apply_temporal_smoothing(self, alpha: float) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
        """Apply temporal smoothing to predictions."""
        try:
            if len(self.prediction_history) < 2:
                return None
            
            # Count recent predictions
            label_counts = {}
            total_confidence = 0.0
            
            for pred in self.prediction_history:
                label = pred['label']
                confidence = pred['confidence']
                
                if label not in label_counts:
                    label_counts[label] = {'count': 0, 'confidence': 0.0}
                
                label_counts[label]['count'] += 1
                label_counts[label]['confidence'] += confidence
                total_confidence += confidence
            
            # Find most frequent label
            most_frequent_label = max(label_counts.keys(), key=lambda x: label_counts[x]['count'])
            avg_confidence = label_counts[most_frequent_label]['confidence'] / label_counts[most_frequent_label]['count']
            
            # Create smoothed predictions
            smoothed_predictions = []
            for label in self.labels:
                if label in label_counts:
                    count = label_counts[label]['count']
                    confidence = label_counts[label]['confidence'] / count
                    smoothed_predictions.append((label, confidence))
                else:
                    smoothed_predictions.append((label, 0.0))
            
            # Sort by confidence
            smoothed_predictions.sort(key=lambda x: x[1], reverse=True)
            
            return most_frequent_label, avg_confidence, smoothed_predictions
        
        except Exception as e:
            logger.error(f"Temporal smoothing failed: {e}")
            return None
    
    def _pad_truncate_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or truncate sequence to target length with proper error handling.
        
        Args:
            sequence: Input sequence of shape (seq_len, features)
            target_length: Desired sequence length
            
        Returns:
            Sequence adjusted to target length
            
        Raises:
            ValueError: If target_length is invalid or sequence is malformed
        """
        if target_length <= 0:
            raise ValueError(f"target_length must be positive, got {target_length}")
        
        if sequence.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got {sequence.ndim}D")
        
        current_length = len(sequence)
        
        if current_length == target_length:
            return sequence
        
        elif current_length < target_length:
            # Pad with zeros (could use other strategies like repeat-last-frame)
            padding_length = target_length - current_length
            padding = np.zeros((padding_length, sequence.shape[1]), dtype=sequence.dtype)
            padded_sequence = np.vstack([sequence, padding])
            logger.debug(f"Padded sequence from {current_length} to {target_length} frames")
            return padded_sequence
        
        else:
            # Truncate from the beginning (keep most recent frames)
            truncated_sequence = sequence[-target_length:]
            logger.debug(f"Truncated sequence from {current_length} to {target_length} frames")
            return truncated_sequence
    
    def _update_stats(self, prediction_time: float, confidence: float, success: bool):
        """Update performance statistics."""
        self.stats['total_predictions'] += 1
        
        if success:
            self.stats['successful_predictions'] += 1
            self.stats['avg_prediction_time'] = (
                self.stats['avg_prediction_time'] * 0.9 + prediction_time * 0.1
            )
            self.stats['avg_confidence'] = (
                self.stats['avg_confidence'] * 0.9 + confidence * 0.1
            )
        else:
            self.stats['failed_predictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        with self.lock:
            self.stats = {
                'total_predictions': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'avg_prediction_time': 0.0,
                'avg_confidence': 0.0
            }
    
    def clear_history(self):
        """Clear prediction history."""
        with self.lock:
            self.prediction_history.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'enhanced_tcn',
            'num_classes': len(self.labels),
            'labels': self.labels,
            'device': str(self.device),
            'config': self.config,
            'stats': self.get_stats()
        }


class BatchPredictor:
    """
    Batch predictor for processing multiple sequences.
    Optimized for training and evaluation scenarios.
    
    Compatible with all model types.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        labels: List[str],
        device: Optional[torch.device] = None,
        batch_size: int = 32
    ):
        """
        Initialize batch predictor.
        
        Args:
            model: Trained model (any PyTorch model)
            config: Model configuration
            labels: List of class labels
            device: Device to run inference on
            batch_size: Batch size for processing
        """
        self.model = model
        self.config = config
        self.labels = labels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Batch predictor initialized on {self.device}")
    
    def predict_batch(
        self,
        sequences: np.ndarray,
        return_probabilities: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict on batch of sequences.
        
        Args:
            sequences: Batch of sequences of shape (batch_size, seq_len, features)
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            # Convert to tensor
            sequences_tensor = torch.FloatTensor(sequences).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                logits = self.model(sequences_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
            
            # Convert to numpy
            predictions_np = predictions.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy() if return_probabilities else None
            
            return predictions_np, probabilities_np
        
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return None, None


# Example usage
if __name__ == "__main__":
    # This would typically be used with a trained model
    # For demonstration purposes, we'll show the interface
    
    # Create dummy model and config
    from ..models.tcn_model import create_model
    
    model = create_model(input_dim=189, num_classes=4)
    config = {'sequence_length': 30}
    labels = ['2-Hay', 'Alifmad', 'Aray', 'Jeem']
    
    # Create predictor
    predictor = RealTimePredictor(model, config, labels)
    
    # Test prediction
    dummy_sequence = np.random.randn(30, 189)
    result = predictor.predict(dummy_sequence)
    
    if result:
        predicted_label, confidence, all_predictions = result
        print(f"Prediction: {predicted_label}")
        print(f"Confidence: {confidence:.3f}")
        print(f"All predictions: {all_predictions}")
    
    # Test with smoothing
    for i in range(5):
        result = predictor.predict_with_smoothing(dummy_sequence)
        if result:
            predicted_label, confidence, all_predictions = result
            print(f"Smoothed prediction {i+1}: {predicted_label} ({confidence:.3f})")
    
    # Get stats
    stats = predictor.get_stats()
    print(f"Stats: {stats}")
