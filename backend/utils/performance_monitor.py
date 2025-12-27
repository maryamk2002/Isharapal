#!/usr/bin/env python3
"""
Performance monitoring and benchmarking for PSL recognition.
Tracks FPS, inference time, and per-label accuracy metrics.
"""

import time
import json
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Performance metrics at a point in time."""
    timestamp: str
    fps: float
    avg_inference_ms: float
    frames_sent: int
    frames_processed: int
    landmarks_extracted: int
    predictions_made: int
    accuracy_per_label: Dict[str, Dict[str, float]]


class PerformanceMonitor:
    """
    Monitor and log real-time performance metrics.
    
    Features:
    - FPS calculation (frontend & backend)
    - Inference time tracking
    - Per-label accuracy (via user feedback)
    - Auto-save metrics to JSON for analysis
    
    Why track these metrics?
    - FPS: Ensure real-time performance (target: ≥15 FPS)
    - Inference time: Detect bottlenecks
    - Accuracy: Identify problematic signs for retraining
    """
    
    def __init__(
        self,
        fps_window: int = 30,
        inference_window: int = 100,
        save_dir: Optional[Path] = None,
        auto_save_interval: float = 60.0
    ):
        """
        Initialize performance monitor.
        
        Args:
            fps_window: Number of frames for FPS calculation (default: 30)
                Why 30? ~2 seconds at 15 FPS, smooth but responsive
            inference_window: Number of inferences for average (default: 100)
                Why 100? Statistical significance without excessive memory
            save_dir: Directory to save metrics (default: backend/logs/metrics)
            auto_save_interval: Save metrics every N seconds (default: 60)
        """
        self.fps_window = fps_window
        self.inference_window = inference_window
        self.auto_save_interval = auto_save_interval
        
        # Set up save directory
        if save_dir is None:
            save_dir = Path(__file__).resolve().parent.parent / "logs" / "metrics"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Frame tracking (for FPS)
        self.frame_sent_times = deque(maxlen=fps_window)
        self.frame_processed_times = deque(maxlen=fps_window)
        
        # Inference timing
        self.inference_times = deque(maxlen=inference_window)
        
        # Counters
        self.total_frames_sent = 0
        self.total_frames_processed = 0
        self.total_landmarks_extracted = 0
        self.total_predictions = 0
        
        # Per-label accuracy tracking (via user feedback)
        # Structure: {label: {'correct': count, 'incorrect': count, 'total': count}}
        self.label_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'total': 0})
        
        # Session info
        self.session_id = None
        self.session_start = time.time()
        self.last_save_time = time.time()
        
        # Snapshots for history
        self.snapshots: List[PerformanceSnapshot] = []
        
        logger.info("Performance monitor initialized")
    
    def start_session(self, session_id: str) -> None:
        """Start new monitoring session."""
        self.session_id = session_id
        self.session_start = time.time()
        logger.info(f"Performance monitoring started: {session_id}")
    
    def record_frame_sent(self) -> None:
        """Record timestamp when frame is sent from frontend."""
        self.frame_sent_times.append(time.time())
        self.total_frames_sent += 1
    
    def record_frame_processed(self) -> None:
        """Record timestamp when frame is processed by backend."""
        self.frame_processed_times.append(time.time())
        self.total_frames_processed += 1
    
    def record_landmarks_extracted(self) -> None:
        """Record successful landmark extraction."""
        self.total_landmarks_extracted += 1
    
    def record_inference_time(self, duration_ms: float) -> None:
        """
        Record inference duration.
        
        Args:
            duration_ms: Inference time in milliseconds
        """
        self.inference_times.append(duration_ms)
    
    def record_prediction(self, label: str) -> None:
        """
        Record prediction made.
        
        Args:
            label: Predicted sign label
        """
        self.total_predictions += 1
        self.label_stats[label]['total'] += 1
    
    def record_feedback(self, label: str, is_correct: bool) -> None:
        """
        Record user feedback for accuracy tracking.
        
        Args:
            label: Predicted label that was shown
            is_correct: True if user confirmed correct, False if incorrect
        """
        if is_correct:
            self.label_stats[label]['correct'] += 1
            logger.debug(f"Feedback: {label} marked CORRECT")
        else:
            self.label_stats[label]['incorrect'] += 1
            logger.debug(f"Feedback: {label} marked INCORRECT")
    
    def get_current_fps(self) -> float:
        """
        Calculate current FPS from recent frames.
        
        Returns:
            FPS (frames per second)
        """
        if len(self.frame_processed_times) < 2:
            return 0.0
        
        # Time span of recent frames
        time_span = self.frame_processed_times[-1] - self.frame_processed_times[0]
        if time_span == 0:
            return 0.0
        
        # FPS = frames / time
        return (len(self.frame_processed_times) - 1) / time_span
    
    def get_avg_inference_time(self) -> float:
        """
        Get average inference time.
        
        Returns:
            Average inference time in milliseconds
        """
        if len(self.inference_times) == 0:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
    
    def get_label_accuracy(self, label: str) -> Optional[float]:
        """
        Get accuracy for specific label.
        
        Args:
            label: Sign label
        
        Returns:
            Accuracy (0-1) or None if no feedback yet
        """
        stats = self.label_stats.get(label)
        if stats is None or (stats['correct'] + stats['incorrect']) == 0:
            return None
        
        total_feedback = stats['correct'] + stats['incorrect']
        return stats['correct'] / total_feedback
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get complete metrics summary.
        
        Returns:
            Dictionary with all current metrics
        """
        # Calculate per-label accuracy
        accuracy_per_label = {}
        for label, stats in self.label_stats.items():
            total_feedback = stats['correct'] + stats['incorrect']
            if total_feedback > 0:
                accuracy = stats['correct'] / total_feedback
                accuracy_per_label[label] = {
                    'accuracy': accuracy,
                    'correct': stats['correct'],
                    'incorrect': stats['incorrect'],
                    'total_predictions': stats['total']
                }
        
        return {
            'fps': round(self.get_current_fps(), 2),
            'avg_inference_ms': round(self.get_avg_inference_time(), 2),
            'frames_sent': self.total_frames_sent,
            'frames_processed': self.total_frames_processed,
            'landmarks_extracted': self.total_landmarks_extracted,
            'predictions_made': self.total_predictions,
            'landmark_detection_rate': (
                round(self.total_landmarks_extracted / max(1, self.total_frames_processed), 3)
            ),
            'accuracy_per_label': accuracy_per_label,
            'session_duration': round(time.time() - self.session_start, 1)
        }
    
    def create_snapshot(self) -> PerformanceSnapshot:
        """Create performance snapshot for history."""
        metrics = self.get_metrics_summary()
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            fps=metrics['fps'],
            avg_inference_ms=metrics['avg_inference_ms'],
            frames_sent=metrics['frames_sent'],
            frames_processed=metrics['frames_processed'],
            landmarks_extracted=metrics['landmarks_extracted'],
            predictions_made=metrics['predictions_made'],
            accuracy_per_label=metrics['accuracy_per_label']
        )
        self.snapshots.append(snapshot)
        return snapshot
    
    def should_auto_save(self) -> bool:
        """Check if it's time to auto-save metrics."""
        return (time.time() - self.last_save_time) >= self.auto_save_interval
    
    def save_metrics(self, filepath: Optional[Path] = None) -> Path:
        """
        Save metrics to JSON file.
        
        Args:
            filepath: Output file path (default: auto-generated)
        
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"metrics_{self.session_id}_{timestamp}.json"
        
        data = {
            'session_id': self.session_id,
            'session_start': self.session_start,
            'session_duration': time.time() - self.session_start,
            'current_metrics': self.get_metrics_summary(),
            'history': [asdict(snap) for snap in self.snapshots]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        self.last_save_time = time.time()
        logger.info(f"Metrics saved to {filepath}")
        return filepath
    
    def reset(self) -> None:
        """Reset all metrics (keep session)."""
        self.frame_sent_times.clear()
        self.frame_processed_times.clear()
        self.inference_times.clear()
        self.total_frames_sent = 0
        self.total_frames_processed = 0
        self.total_landmarks_extracted = 0
        self.total_predictions = 0
        self.label_stats.clear()
        logger.debug("Performance metrics reset")

