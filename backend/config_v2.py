#!/usr/bin/env python3
"""
Configuration V2 for PSL Recognition System.
Complete configuration for training on ALL Urdu alphabet signs (37+ classes).

Safety: This file does NOT modify v1 config. Both can coexist.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# ==============================================================================
# BASE PATHS (V2)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BACKEND_DIR / "data"

# V2 Specific directories
V2_MODELS_DIR = BACKEND_DIR / "saved_models" / "v2"
V2_CHECKPOINTS_DIR = V2_MODELS_DIR / "checkpoints"
V2_LOGS_DIR = BACKEND_DIR / "logs" / "v2"

# Data paths
RAW_DATA_DIR = DATA_DIR / "Pakistan Sign Language Urdu Alphabets"
FEATURES_DIR = DATA_DIR / "features_temporal"
V2_FEATURES_DIR = DATA_DIR / "features_v2"  # New v2 processed features
V2_SPLITS_DIR = DATA_DIR / "splits_v2"

# ==============================================================================
# ALL URDU ALPHABET SIGNS (37+ classes)
# ==============================================================================
# Complete list of Urdu alphabet signs in the dataset
ALL_URDU_SIGNS = [
    "1-Hay", "2-Hay", "Ain", "Alif", "Alifmad", "Aray", 
    "Bay", "Byeh", "Chay", "Cyeh",
    "Daal", "Dal", "Dochahay",
    "Fay",
    "Gaaf", "Ghain",
    "Hamza",
    "Jeem",
    "Kaf", "Khay", "Kiaf",
    "Lam",
    "Meem",
    "Nuun", "Nuungh",
    "Pay",
    "Ray",
    "Say", "Seen", "Sheen", "Suad",
    "Taay", "Tay", "Tuey",
    "Wao",
    "Zaal", "Zaey", "Zay", "Zuad", "Zuey"
]

# Signs that might need special handling (video-only signs)
VIDEO_ONLY_SIGNS = ["Aray", "Jeem", "2-Hay", "Alifmad"]

# Signs that might be excluded (single sample or missing data)
EXCLUDED_SIGNS = ["Bariyay", "Chotiyay"]  # Only have 1 video each

# ==============================================================================
# MODEL CONFIGURATION V2
# ==============================================================================
@dataclass
class ModelConfigV2:
    """Model architecture configuration for V2."""
    
    # Input dimensions
    INPUT_DIM: int = 189  # 126 hand landmarks + padding
    HAND_LANDMARKS: int = 21 * 2 * 3  # 2 hands × 21 points × 3 coords = 126
    
    # Sequence settings
    SEQUENCE_LENGTH: int = 60  # Frames per sequence (matches training data)
    SLIDING_WINDOW_SIZE: int = 24  # DEMO: Faster response (was 32)
    SLIDING_STRIDE: int = 8  # Stride for sliding windows during training
    
    # Architecture (Enhanced TCN)
    NUM_CHANNELS: List[int] = field(default_factory=lambda: [256, 256, 256, 256, 128])
    KERNEL_SIZE: int = 5
    DROPOUT: float = 0.4
    USE_ATTENTION: bool = True
    ATTENTION_HEADS: int = 4
    ATTENTION_DIM: int = 64
    
    # Classification
    NUM_CLASSES: int = 40  # Will be set dynamically based on dataset
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.INPUT_DIM > 0, "INPUT_DIM must be positive"
        assert self.SEQUENCE_LENGTH > 0, "SEQUENCE_LENGTH must be positive"
        assert len(self.NUM_CHANNELS) > 0, "NUM_CHANNELS must not be empty"


@dataclass 
class TrainingConfigV2:
    """Training hyperparameters for V2."""
    
    # Training settings
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    
    # Optimizer settings
    OPTIMIZER: str = "AdamW"
    SCHEDULER: str = "ReduceLROnPlateau"
    SCHEDULER_PATIENCE: int = 10
    SCHEDULER_FACTOR: float = 0.5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 15
    MIN_DELTA: float = 1e-4
    
    # Gradient clipping
    GRADIENT_CLIP_NORM: float = 1.0
    
    # Mixed precision (for GPU)
    USE_MIXED_PRECISION: bool = True
    
    # Data splits
    TRAIN_SPLIT: float = 0.70
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    
    # Augmentation
    USE_AUGMENTATION: bool = True
    TEMPORAL_JITTER: float = 0.1
    NOISE_STD: float = 0.01
    ROTATION_RANGE: float = 5.0
    SCALE_RANGE: tuple = (0.9, 1.1)
    AUGMENTATION_PROB: float = 0.5
    
    # CTC Loss (disabled by default for isolated classification)
    USE_CTC: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.TRAIN_SPLIT < 1, "TRAIN_SPLIT must be in (0, 1)"
        assert 0 < self.VAL_SPLIT < 1, "VAL_SPLIT must be in (0, 1)"
        assert 0 < self.TEST_SPLIT < 1, "TEST_SPLIT must be in (0, 1)"
        total = self.TRAIN_SPLIT + self.VAL_SPLIT + self.TEST_SPLIT
        assert abs(total - 1.0) < 1e-6, f"Splits must sum to 1.0, got {total}"


@dataclass
class InferenceConfigV2:
    """Inference settings for V2 real-time recognition."""
    
    # Frame collection
    TARGET_FPS: int = 15
    FRAME_INTERVAL_MS: int = 67  # 1000/15
    
    # Sliding window
    SLIDING_WINDOW_SIZE: int = 45  # Closer to model's 60-frame training (45 frames ~= 3s at 15 FPS)
    MIN_FRAMES_FOR_PREDICTION: int = 30  # Start predicting at 30 frames (~2s)
    
    # Stability (DEMO OPTIMIZED: faster response)
    STABILITY_VOTES: int = 5
    STABILITY_THRESHOLD: int = 3  # Need 3/5 votes (60% - balanced)
    MIN_CONFIDENCE: float = 0.60  # Balanced confidence
    
    # Buffer management
    RESET_ON_NO_HANDS: bool = True
    NO_HANDS_TIMEOUT_MS: int = 1000
    
    # Keypoints
    SEND_KEYPOINTS_TO_FRONTEND: bool = True


@dataclass
class FilteringConfig:
    """Keypoint filtering and denoising settings."""
    
    # Moving average smoothing
    MOVING_AVERAGE_WINDOW: int = 5  # Number of frames for smoothing
    
    # Jitter detection
    JITTER_THRESHOLD: float = 0.01  # Max movement for "stable" frame (~1% of coords)
    
    # Stuck sequence detection (HASH-BASED)
    STUCK_THRESHOLD_FRAMES: int = 60  # Frames to check for identical data (~4s at 15 FPS)
    STUCK_MOVEMENT_THRESHOLD: float = 0.001  # Not used (kept for compatibility)
    
    # Enable/disable filtering
    ENABLE_FILTERING: bool = False  # DISABLED - causing system to hang


@dataclass
class RecordingConfig:
    """Automatic recording management settings."""
    
    # Automatic recording control
    IDLE_TIMEOUT_SEC: float = 2.0  # Stop recording after N seconds of no valid predictions
    MIN_CONFIDENCE_FOR_RECORDING: float = 0.6  # Minimum confidence to count as "valid"
    AUTO_STOP_ENABLED: bool = True
    
    # Segment saving
    AUTO_SAVE_SEGMENTS: bool = True
    SAVE_TO_CSV: bool = True
    SAVE_TO_JSON: bool = True


@dataclass
class MonitoringConfig:
    """Performance monitoring and metrics settings."""
    
    # FPS tracking
    FPS_WINDOW_SIZE: int = 30  # Number of frames for FPS calculation (~2s at 15 FPS)
    
    # Inference timing
    INFERENCE_WINDOW_SIZE: int = 100  # Number of inferences to average
    
    # Metrics emission
    METRICS_EMIT_INTERVAL_MS: int = 1000  # Send metrics to frontend every 1 second
    
    # Auto-save metrics
    AUTO_SAVE_METRICS: bool = True
    AUTO_SAVE_INTERVAL_SEC: float = 60.0  # Save metrics file every 60 seconds


@dataclass
class DatasetConfigV2:
    """Dataset configuration for V2."""
    
    # Supported extensions
    VIDEO_EXTENSIONS: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    IMAGE_EXTENSIONS: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"])
    
    # Feature extraction
    FEATURE_DIM: int = 189
    MAX_VIDEO_DURATION: float = 10.0  # seconds
    MIN_VIDEO_DURATION: float = 0.5  # seconds
    EXTRACT_FPS: int = 30  # FPS for video extraction
    
    # Quality thresholds
    MIN_HAND_DETECTION_CONFIDENCE: float = 0.5
    MIN_HAND_TRACKING_CONFIDENCE: float = 0.5
    MIN_VALID_FRAMES_RATIO: float = 0.3  # At least 30% of frames must have valid hands
    
    # Minimum samples per class
    MIN_SAMPLES_PER_CLASS: int = 30


# ==============================================================================
# VERSION INFO
# ==============================================================================
V2_VERSION = "2.0.0"
V2_CREATED = datetime.now().isoformat()
V2_DESCRIPTION = "Complete PSL Recognition v2 - All Urdu Alphabet Signs"


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def ensure_v2_directories() -> None:
    """Create all v2-specific directories."""
    directories = [
        V2_MODELS_DIR,
        V2_CHECKPOINTS_DIR,
        V2_LOGS_DIR,
        V2_FEATURES_DIR,
        V2_SPLITS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_available_signs(features_dir: Path = FEATURES_DIR) -> List[str]:
    """Get list of signs that have extracted features."""
    if not features_dir.exists():
        return []
    
    available = []
    for sign_dir in features_dir.iterdir():
        if sign_dir.is_dir() and sign_dir.name not in ['train', 'val', 'test']:
            # Check if it has any .npy files
            npy_files = list(sign_dir.glob("*.npy"))
            if len(npy_files) >= 10:  # At least 10 samples
                available.append(sign_dir.name)
    
    return sorted(available)


def get_dataset_stats(features_dir: Path = FEATURES_DIR) -> Dict[str, Any]:
    """Get statistics about the dataset."""
    stats = {
        'total_signs': 0,
        'total_samples': 0,
        'samples_per_sign': {},
        'min_samples': float('inf'),
        'max_samples': 0,
        'avg_samples': 0,
    }
    
    for sign in get_available_signs(features_dir):
        sign_dir = features_dir / sign
        npy_files = list(sign_dir.glob("*.npy"))
        count = len(npy_files)
        
        stats['samples_per_sign'][sign] = count
        stats['total_samples'] += count
        stats['total_signs'] += 1
        stats['min_samples'] = min(stats['min_samples'], count)
        stats['max_samples'] = max(stats['max_samples'], count)
    
    if stats['total_signs'] > 0:
        stats['avg_samples'] = stats['total_samples'] / stats['total_signs']
    
    return stats


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information for training configuration."""
    import torch
    
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'num_gpus': 0,
        'recommended_batch_size': 16,
        'use_mixed_precision': False,
    }
    
    if torch.cuda.is_available():
        info['num_gpus'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Recommended settings based on GPU memory
        if info['gpu_memory_gb'] >= 8:
            info['recommended_batch_size'] = 64
            info['use_mixed_precision'] = True
        elif info['gpu_memory_gb'] >= 4:
            info['recommended_batch_size'] = 32
            info['use_mixed_precision'] = True
        else:
            info['recommended_batch_size'] = 16
            info['use_mixed_precision'] = False
    else:
        info['recommended_batch_size'] = 8  # Conservative for CPU
    
    return info


# ==============================================================================
# CREATE CONFIG INSTANCES
# ==============================================================================
model_config_v2 = ModelConfigV2()
training_config_v2 = TrainingConfigV2()
inference_config_v2 = InferenceConfigV2()
dataset_config_v2 = DatasetConfigV2()
filtering_config = FilteringConfig()
recording_config = RecordingConfig()
monitoring_config = MonitoringConfig()

# Ensure directories exist
ensure_v2_directories()


# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    # Paths
    "BASE_DIR",
    "BACKEND_DIR",
    "DATA_DIR",
    "V2_MODELS_DIR",
    "V2_CHECKPOINTS_DIR",
    "V2_LOGS_DIR",
    "RAW_DATA_DIR",
    "FEATURES_DIR",
    "V2_FEATURES_DIR",
    "V2_SPLITS_DIR",
    
    # Sign lists
    "ALL_URDU_SIGNS",
    "VIDEO_ONLY_SIGNS",
    "EXCLUDED_SIGNS",
    
    # Config classes
    "ModelConfigV2",
    "TrainingConfigV2",
    "InferenceConfigV2",
    "DatasetConfigV2",
    "FilteringConfig",
    "RecordingConfig",
    "MonitoringConfig",
    
    # Config instances
    "model_config_v2",
    "training_config_v2",
    "inference_config_v2",
    "dataset_config_v2",
    "filtering_config",
    "recording_config",
    "monitoring_config",
    
    # Functions
    "ensure_v2_directories",
    "get_available_signs",
    "get_dataset_stats",
    "get_hardware_info",
    
    # Version
    "V2_VERSION",
]


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 70)
    print("PSL RECOGNITION SYSTEM - V2 CONFIGURATION")
    print("=" * 70)
    
    print(f"\nVersion: {V2_VERSION}")
    print(f"Created: {V2_CREATED}")
    
    print("\n--- Paths ---")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Features Directory: {FEATURES_DIR}")
    print(f"V2 Models Directory: {V2_MODELS_DIR}")
    
    print("\n--- Dataset ---")
    stats = get_dataset_stats()
    print(f"Available Signs: {stats['total_signs']}")
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Samples per Sign: min={stats['min_samples']}, max={stats['max_samples']}, avg={stats['avg_samples']:.1f}")
    
    print("\n--- Hardware ---")
    hw = get_hardware_info()
    print(f"Device: {hw['device']}")
    if hw['cuda_available']:
        print(f"GPU: {hw['gpu_name']}")
        print(f"GPU Memory: {hw['gpu_memory_gb']:.1f} GB")
    print(f"Recommended Batch Size: {hw['recommended_batch_size']}")
    print(f"Mixed Precision: {hw['use_mixed_precision']}")
    
    print("\n--- Available Signs ---")
    signs = get_available_signs()
    print(f"Count: {len(signs)}")
    print(f"Signs: {', '.join(signs[:10])}...")
    
    print("\n" + "=" * 70)

