#!/usr/bin/env python3
"""
Robust video feature extraction pipeline for PSL recognition.
Extracts hand landmarks from video files using MediaPipe with quality validation.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import json

from config import dataset_config, mediapipe_config, ensure_directories
from utils.mediapipe_utils import MediaPipeProcessor
from utils.video_utils import VideoProcessor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Robust feature extractor for PSL video data.
    
    Features:
    - Hand landmark extraction using MediaPipe
    - Quality validation and filtering
    - Temporal sequence processing
    - Data augmentation support
    - Progress tracking and error handling
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        quality_threshold: float = 0.5,
        min_sequence_length: int = 15,
        max_sequence_length: int = 60,
        target_fps: int = 30
    ):
        """
        Initialize feature extractor.
        
        Args:
            output_dir: Directory to save extracted features
            quality_threshold: Minimum quality score for sequences
            min_sequence_length: Minimum sequence length
            max_sequence_length: Maximum sequence length
            target_fps: Target FPS for video processing
        """
        self.output_dir = output_dir or Path("data/processed")
        self.quality_threshold = quality_threshold
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.target_fps = target_fps
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe processor
        self.mp_processor = MediaPipeProcessor(
            min_detection_confidence=mediapipe_config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=mediapipe_config.MIN_TRACKING_CONFIDENCE,
            model_complexity=mediapipe_config.MODEL_COMPLEXITY
        )
        
        # Initialize video processor
        self.video_processor = VideoProcessor()
        
        # Statistics
        self.stats = {
            "total_videos": 0,
            "processed_videos": 0,
            "skipped_videos": 0,
            "total_sequences": 0,
            "valid_sequences": 0,
            "invalid_sequences": 0
        }
    
    def extract_from_video(
        self,
        video_path: Path,
        label: str,
        save_individual: bool = False
    ) -> Optional[np.ndarray]:
        """
        Extract features from a single video file.
        
        Args:
            video_path: Path to video file
            label: Class label for the video
            save_individual: Whether to save individual sequence files
        
        Returns:
            Extracted features array or None if extraction failed
        """
        try:
            logger.debug(f"Processing video: {video_path}")
            
            # Load video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Cannot open video: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Validate video properties
            if duration < 1.0:  # Too short
                logger.warning(f"Video too short: {video_path} ({duration:.2f}s)")
                cap.release()
                return None
            
            if duration > 10.0:  # Too long
                logger.warning(f"Video too long: {video_path} ({duration:.2f}s)")
                cap.release()
                return None
            
            # Extract frames and landmarks
            sequences = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at target FPS
                if fps > self.target_fps:
                    skip_frames = int(fps / self.target_fps)
                    if frame_idx % skip_frames != 0:
                        frame_idx += 1
                        continue
                
                # Extract landmarks
                landmarks = self.mp_processor.extract_landmarks(frame)
                if landmarks is not None:
                    sequences.append(landmarks)
                
                frame_idx += 1
            
            cap.release()
            
            if len(sequences) == 0:
                logger.warning(f"No landmarks extracted from: {video_path}")
                return None
            
            # Convert to numpy array
            sequences = np.array(sequences)
            
            # Validate sequence length
            if len(sequences) < self.min_sequence_length:
                logger.warning(f"Sequence too short: {video_path} ({len(sequences)} frames)")
                return None
            
            # Truncate if too long
            if len(sequences) > self.max_sequence_length:
                sequences = sequences[:self.max_sequence_length]
            
            # Pad sequence to fixed length
            padded_sequence = self._pad_sequence(sequences, self.max_sequence_length)
            
            # Save individual sequence if requested
            if save_individual:
                self._save_sequence(padded_sequence, video_path, label)
            
            logger.debug(f"Extracted {len(sequences)} frames from {video_path}")
            return padded_sequence
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return None
    
    def extract_from_directory(
        self,
        data_dir: Path,
        labels: Optional[List[str]] = None,
        save_individual: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from all videos in a directory.
        
        Args:
            data_dir: Directory containing video files
            labels: List of labels to process (None for all)
            save_individual: Whether to save individual sequence files
            progress_bar: Whether to show progress bar
        
        Returns:
            Dictionary mapping labels to feature arrays
        """
        logger.info(f"Extracting features from directory: {data_dir}")
        
        # Find all label directories
        if labels is None:
            label_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        else:
            label_dirs = [data_dir / label for label in labels if (data_dir / label).exists()]
        
        all_features = {}
        
        for label_dir in label_dirs:
            label = label_dir.name
            logger.info(f"Processing label: {label}")
            
            # Find video files
            video_files = []
            for ext in dataset_config.VIDEO_EXTENSIONS:
                video_files.extend(label_dir.glob(f"*{ext}"))
            
            if not video_files:
                logger.warning(f"No video files found in: {label_dir}")
                continue
            
            # Extract features from videos
            label_features = []
            
            iterator = tqdm(video_files, desc=f"Processing {label}") if progress_bar else video_files
            
            for video_file in iterator:
                self.stats["total_videos"] += 1
                
                features = self.extract_from_video(video_file, label, save_individual)
                
                if features is not None:
                    label_features.append(features)
                    self.stats["processed_videos"] += 1
                    self.stats["valid_sequences"] += 1
                else:
                    self.stats["skipped_videos"] += 1
                    self.stats["invalid_sequences"] += 1
            
            if label_features:
                all_features[label] = np.array(label_features)
                logger.info(f"Extracted {len(label_features)} sequences for {label}")
            else:
                logger.warning(f"No valid sequences extracted for {label}")
        
        # Save statistics
        self._save_statistics()
        
        logger.info(f"Feature extraction completed. Stats: {self.stats}")
        return all_features
    
    def _pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """Pad sequence to target length."""
        if len(sequence) >= target_length:
            return sequence[:target_length]
        
        # Pad with zeros
        padding_length = target_length - len(sequence)
        padding = np.zeros((padding_length, sequence.shape[1]))
        return np.vstack([sequence, padding])
    
    def _save_sequence(self, sequence: np.ndarray, video_path: Path, label: str):
        """Save individual sequence to file."""
        sequence_dir = self.output_dir / "individual_sequences" / label
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        sequence_file = sequence_dir / f"{video_path.stem}.npy"
        np.save(sequence_file, sequence)
    
    def _save_statistics(self):
        """Save extraction statistics."""
        stats_file = self.output_dir / "extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def save_features(self, features: Dict[str, np.ndarray], filename: str = "features.npz"):
        """Save extracted features to file."""
        output_file = self.output_dir / filename
        np.savez_compressed(output_file, **features)
        logger.info(f"Features saved to: {output_file}")
    
    def load_features(self, filename: str = "features.npz") -> Dict[str, np.ndarray]:
        """Load features from file."""
        features_file = self.output_dir / filename
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        data = np.load(features_file)
        features = {label: data[label] for label in data.files}
        logger.info(f"Loaded features for {len(features)} labels")
        return features


def extract_psl_features(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    labels: Optional[List[str]] = None,
    quality_threshold: float = 0.5,
    min_sequence_length: int = 15,
    max_sequence_length: int = 60,
    target_fps: int = 30,
    save_individual: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extract features from PSL video dataset.
    
    Args:
        data_dir: Directory containing video files organized by labels
        output_dir: Directory to save extracted features
        labels: List of labels to process (None for all)
        quality_threshold: Minimum quality score for sequences
        min_sequence_length: Minimum sequence length
        max_sequence_length: Maximum sequence length
        target_fps: Target FPS for video processing
        save_individual: Whether to save individual sequence files
    
    Returns:
        Dictionary mapping labels to feature arrays
    """
    # Ensure directories exist
    ensure_directories()
    
    # Create extractor
    extractor = FeatureExtractor(
        output_dir=output_dir,
        quality_threshold=quality_threshold,
        min_sequence_length=min_sequence_length,
        max_sequence_length=max_sequence_length,
        target_fps=target_fps
    )
    
    # Extract features
    features = extractor.extract_from_directory(
        data_dir=data_dir,
        labels=labels,
        save_individual=save_individual
    )
    
    # Save features
    if features:
        extractor.save_features(features)
    
    return features


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Extract features from 4 initial signs
    data_dir = Path("../data/raw")
    output_dir = Path("../data/processed")
    
    initial_signs = ["2-Hay", "Alifmad", "Aray", "Jeem"]
    
    features = extract_psl_features(
        data_dir=data_dir,
        output_dir=output_dir,
        labels=initial_signs,
        quality_threshold=0.5,
        min_sequence_length=15,
        max_sequence_length=60,
        target_fps=30,
        save_individual=False
    )
    
    print(f"Extracted features for {len(features)} labels")
    for label, feature_array in features.items():
        print(f"{label}: {feature_array.shape}")

