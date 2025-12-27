#!/usr/bin/env python3
"""
Recording session manager for PSL recognition.
Handles automatic recording start/stop based on sign detection.
"""

import time
import json
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class RecordingSegment:
    """A single recorded sign segment."""
    label: str
    confidence: float
    start_time: float
    end_time: float
    duration: float
    frame_count: int
    session_id: str
    timestamp: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class RecordingManager:
    """
    Manage automatic recording sessions with idle timeout.
    
    Features:
    - Auto-start recording on first valid prediction
    - Auto-stop after idle timeout (no valid predictions)
    - Save segments to JSON/CSV for analysis
    - Emit real-time segment updates
    
    Why automatic recording?
    - Natural user experience (no manual start/stop)
    - Clear segment boundaries (2-second pauses between signs)
    - Easier data collection for analysis
    """
    
    def __init__(
        self,
        idle_timeout_sec: float = 2.0,
        min_confidence: float = 0.6,
        save_dir: Optional[Path] = None,
        auto_save: bool = True
    ):
        """
        Initialize recording manager.
        
        Args:
            idle_timeout_sec: Stop recording after N seconds of no valid predictions
                Why 2.0? Natural pause between signs, not too short/long
            min_confidence: Minimum confidence to count as "valid" prediction
                Why 0.6? Filters out uncertain/transient predictions
            save_dir: Directory to save recording logs (default: backend/data/recordings)
            auto_save: Automatically save segments to disk
        """
        self.idle_timeout = idle_timeout_sec
        self.min_confidence = min_confidence
        self.auto_save = auto_save
        
        # Set up save directory
        if save_dir is None:
            save_dir = Path(__file__).resolve().parent.parent / "data" / "recordings"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.current_segment: Optional[Dict[str, Any]] = None
        self.all_segments: List[RecordingSegment] = []
        
        # Timing
        self.last_valid_prediction_time = 0.0
        self.recording_start_time = 0.0
        self.segment_frame_count = 0
        
        # Session info
        self.session_id = None
        self.session_start_time = None
        
        logger.info(f"Recording manager initialized (idle_timeout={idle_timeout_sec}s, min_conf={min_confidence})")
    
    def start_session(self, session_id: str) -> None:
        """
        Start new recording session.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self.session_start_time = time.time()
        self.all_segments.clear()
        logger.info(f"Recording session started: {session_id}")
    
    def on_prediction(self, label: str, confidence: float, timestamp: Optional[float] = None) -> Optional[Dict]:
        """
        Handle new prediction from model.
        
        Args:
            label: Predicted sign label
            confidence: Prediction confidence (0-1)
            timestamp: Frame timestamp (default: current time)
        
        Returns:
            Finalized segment dict if recording stopped, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Check if prediction is valid (above confidence threshold)
        if confidence >= self.min_confidence:
            self.last_valid_prediction_time = timestamp
            self.segment_frame_count += 1
            
            # Start recording if not already
            if not self.is_recording:
                self._start_recording(label, confidence, timestamp)
            else:
                # Update current segment
                self.current_segment['label'] = label
                self.current_segment['confidence'] = max(confidence, self.current_segment.get('confidence', 0))
                self.current_segment['last_update'] = timestamp
            
            return None
        else:
            # Low confidence prediction - check for timeout
            return self._check_timeout(timestamp)
    
    def on_no_hands(self, timestamp: Optional[float] = None) -> Optional[Dict]:
        """
        Handle frame with no hands detected.
        
        Args:
            timestamp: Frame timestamp
        
        Returns:
            Finalized segment dict if recording stopped, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        return self._check_timeout(timestamp)
    
    def should_stop_recording(self, current_time: Optional[float] = None) -> bool:
        """
        Check if recording should stop (idle timeout exceeded).
        
        Args:
            current_time: Current timestamp (default: now)
        
        Returns:
            True if should stop, False otherwise
        """
        if not self.is_recording:
            return False
        
        if current_time is None:
            current_time = time.time()
        
        idle_duration = current_time - self.last_valid_prediction_time
        return idle_duration >= self.idle_timeout
    
    def finalize_segment(self, timestamp: Optional[float] = None) -> Optional[RecordingSegment]:
        """
        Finalize current recording segment.
        
        Args:
            timestamp: End timestamp (default: now)
        
        Returns:
            Completed RecordingSegment or None
        """
        if not self.is_recording or self.current_segment is None:
            return None
        
        if timestamp is None:
            timestamp = time.time()
        
        # Create segment object
        segment = RecordingSegment(
            label=self.current_segment['label'],
            confidence=self.current_segment['confidence'],
            start_time=self.current_segment['start_time'],
            end_time=timestamp,
            duration=timestamp - self.current_segment['start_time'],
            frame_count=self.segment_frame_count,
            session_id=self.session_id or 'unknown',
            timestamp=datetime.now().isoformat()
        )
        
        # Save segment
        self.all_segments.append(segment)
        
        if self.auto_save:
            self._save_segment(segment)
        
        logger.info(
            f"Segment finalized: {segment.label} "
            f"({segment.confidence:.2f}, {segment.duration:.1f}s, {segment.frame_count} frames)"
        )
        
        # Reset state
        self.is_recording = False
        self.current_segment = None
        self.segment_frame_count = 0
        
        return segment
    
    def get_current_segment(self) -> Optional[Dict]:
        """Get current segment in progress."""
        return self.current_segment
    
    def get_all_segments(self) -> List[Dict]:
        """Get all completed segments."""
        return [seg.to_dict() for seg in self.all_segments]
    
    def get_recording_status(self) -> str:
        """Get current recording status."""
        if self.is_recording:
            return 'recording'
        elif len(self.all_segments) > 0:
            return 'idle'
        else:
            return 'idle'
    
    def save_session(self, filepath: Optional[Path] = None) -> Path:
        """
        Save entire session to JSON file.
        
        Args:
            filepath: Output file path (default: auto-generated)
        
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"session_{self.session_id}_{timestamp}.json"
        
        data = {
            'session_id': self.session_id,
            'start_time': self.session_start_time,
            'segments': self.get_all_segments(),
            'total_segments': len(self.all_segments),
            'total_duration': sum(seg.duration for seg in self.all_segments)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session saved to {filepath}")
        return filepath
    
    def reset(self) -> None:
        """Reset recording manager (keep session)."""
        if self.is_recording:
            self.finalize_segment()
        self.current_segment = None
        self.segment_frame_count = 0
    
    # Private methods
    
    def _start_recording(self, label: str, confidence: float, timestamp: float) -> None:
        """Start new recording segment."""
        self.is_recording = True
        self.recording_start_time = timestamp
        self.last_valid_prediction_time = timestamp
        self.segment_frame_count = 1
        
        self.current_segment = {
            'label': label,
            'confidence': confidence,
            'start_time': timestamp,
            'last_update': timestamp
        }
        
        logger.debug(f"Recording started: {label} ({confidence:.2f})")
    
    def _check_timeout(self, current_time: float) -> Optional[Dict]:
        """Check and handle idle timeout."""
        if self.should_stop_recording(current_time):
            segment = self.finalize_segment(current_time)
            if segment:
                return segment.to_dict()
        return None
    
    def _save_segment(self, segment: RecordingSegment) -> None:
        """Save segment to CSV file."""
        csv_file = self.save_dir / f"segments_{self.session_id}.csv"
        
        # Check if file exists (to write header)
        file_exists = csv_file.exists()
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=segment.to_dict().keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(segment.to_dict())

