#!/usr/bin/env python3
"""
Video processing utilities for PSL recognition system.
Handles video loading, preprocessing, and frame extraction.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Container for video information."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    format: str


class VideoProcessor:
    """
    Video processor for PSL video data.
    
    Features:
    - Video loading and validation
    - Frame extraction and preprocessing
    - Video information extraction
    - Batch processing support
    """
    
    def __init__(self, target_fps: int = 30):
        """
        Initialize video processor.
        
        Args:
            target_fps: Target FPS for frame extraction
        """
        self.target_fps = target_fps
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    def get_video_info(self, video_path: Path) -> Optional[VideoInfo]:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
        
        Returns:
            VideoInfo object or None if failed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            return VideoInfo(
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                codec=codec,
                format=video_path.suffix.lower()
            )
            
        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {e}")
            return None
    
    def validate_video(self, video_path: Path) -> bool:
        """
        Validate video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            True if video is valid, False otherwise
        """
        try:
            # Check file extension
            if video_path.suffix.lower() not in self.supported_formats:
                logger.warning(f"Unsupported video format: {video_path.suffix}")
                return False
            
            # Check if file exists
            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                return False
            
            # Check file size
            file_size = video_path.stat().st_size
            if file_size == 0:
                logger.error(f"Empty video file: {video_path}")
                return False
            
            # Check if video can be opened
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return False
            
            # Check basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Validate properties
            if fps <= 0:
                logger.warning(f"Invalid FPS: {fps}")
                return False
            
            if frame_count <= 0:
                logger.warning(f"Invalid frame count: {frame_count}")
                return False
            
            if duration < 1.0:  # Too short
                logger.warning(f"Video too short: {duration:.2f}s")
                return False
            
            if duration > 30.0:  # Too long
                logger.warning(f"Video too long: {duration:.2f}s")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating video {video_path}: {e}")
            return False
    
    def extract_frames(
        self,
        video_path: Path,
        target_fps: Optional[int] = None,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            target_fps: Target FPS for extraction (None for original FPS)
            max_frames: Maximum number of frames to extract
        
        Returns:
            List of frames
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame skip
            if target_fps is None:
                frame_skip = 1
            else:
                frame_skip = max(1, int(original_fps / target_fps))
            
            frames = []
            frame_idx = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on target FPS
                if frame_idx % frame_skip == 0:
                    frames.append(frame.copy())
                    extracted_count += 1
                    
                    # Check max frames limit
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_idx += 1
            
            cap.release()
            
            logger.debug(f"Extracted {len(frames)} frames from {video_path}")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return []
    
    def extract_frames_generator(
        self,
        video_path: Path,
        target_fps: Optional[int] = None,
        max_frames: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Extract frames as generator (memory efficient).
        
        Args:
            video_path: Path to video file
            target_fps: Target FPS for extraction
            max_frames: Maximum number of frames to extract
        
        Yields:
            Frames one by one
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame skip
            if target_fps is None:
                frame_skip = 1
            else:
                frame_skip = max(1, int(original_fps / target_fps))
            
            frame_idx = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on target FPS
                if frame_idx % frame_skip == 0:
                    yield frame.copy()
                    extracted_count += 1
                    
                    # Check max frames limit
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_idx += 1
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
    
    def preprocess_frame(
        self,
        frame: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess frame for better hand detection.
        
        Args:
            frame: Input frame
            target_size: Target size (width, height)
            normalize: Whether to normalize pixel values
        
        Returns:
            Preprocessed frame
        """
        try:
            # Resize if target size specified
            if target_size:
                frame = cv2.resize(frame, target_size)
            
            # Enhance image quality
            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            frame = cv2.filter2D(frame, -1, kernel * 0.1 + np.eye(3))
            
            # Apply bilateral filter for noise reduction
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
            
            # Normalize if requested
            if normalize:
                frame = frame.astype(np.float32) / 255.0
            
            return frame
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return frame
    
    def create_video_from_frames(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: int = 30,
        codec: str = 'mp4v'
    ) -> bool:
        """
        Create video from frames.
        
        Args:
            frames: List of frames
            output_path: Output video path
            fps: Output FPS
            codec: Video codec
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not frames:
                logger.error("No frames provided")
                return False
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Cannot create video writer: {output_path}")
                return False
            
            # Write frames
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            logger.info(f"Video created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            return False
    
    def get_frame_at_time(
        self,
        video_path: Path,
        time_seconds: float
    ) -> Optional[np.ndarray]:
        """
        Get frame at specific time.
        
        Args:
            video_path: Path to video file
            time_seconds: Time in seconds
        
        Returns:
            Frame at specified time or None if failed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None
            
            # Get video FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                logger.error(f"Invalid FPS: {fps}")
                cap.release()
                return None
            
            # Calculate frame number
            frame_number = int(time_seconds * fps)
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return frame
            else:
                logger.warning(f"Cannot read frame at time {time_seconds}s")
                return None
            
        except Exception as e:
            logger.error(f"Error getting frame at time {time_seconds}s: {e}")
            return None


# Utility functions
def get_video_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """
    Get all video files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
    
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    video_files = []
    for ext in extensions:
        video_files.extend(directory.glob(f"*{ext}"))
        video_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(video_files)


def calculate_video_duration(video_path: Path) -> float:
    """
    Calculate video duration in seconds.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Duration in seconds or 0 if failed
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        if fps > 0:
            return frame_count / fps
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating duration for {video_path}: {e}")
        return 0.0


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = VideoProcessor(target_fps=30)
    
    # Test video validation
    test_video = Path("test_video.mp4")
    if test_video.exists():
        is_valid = processor.validate_video(test_video)
        print(f"Video valid: {is_valid}")
        
        if is_valid:
            # Get video info
            info = processor.get_video_info(test_video)
            if info:
                print(f"Video info: {info}")
            
            # Extract frames
            frames = processor.extract_frames(test_video, target_fps=10, max_frames=5)
            print(f"Extracted {len(frames)} frames")

