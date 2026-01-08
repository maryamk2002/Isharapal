#!/usr/bin/env python3
"""
Configuration management for PSL Recognition System.
Centralized configuration for all components with state-of-the-art practices.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"  # Main model storage
SAVED_MODELS_DIR = BACKEND_DIR / "saved_models"  # Training checkpoints

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
FEATURES_DIR = DATA_DIR / "features_temporal"
UPLOADS_DIR = DATA_DIR / "uploads"


@dataclass
class ModelConfig:
    """Model architecture and training configuration with type safety."""
    
    # Architecture - Hand landmarks
    INPUT_DIM: int = 126  # 2 hands × 21 landmarks × 3 coords
    PADDED_DIM: int = 189  # Padded to 189 for model compatibility
    SEQUENCE_LENGTH: int = 60  # Default sequence length (MATCHES TRAINING DATA)
    MIN_SEQUENCE_LENGTH: int = 30  # Minimum for real-time
    MAX_SEQUENCE_LENGTH: int = 90  # Maximum for complex signs
    
    # TCN Architecture (for legacy models)
    TCN_CHANNELS: List[int] = field(default_factory=lambda: [128, 128, 128, 128])
    TCN_KERNEL_SIZE: int = 3
    TCN_DROPOUT: float = 0.3
    TCN_USE_ATTENTION: bool = True
    
    # Attention mechanism
    ATTENTION_HEADS: int = 4
    ATTENTION_DIM: int = 64
    
    # Training hyperparameters
    BATCH_SIZE: int = 16
    EPOCHS: int = 100
    LEARNING_RATE: float = 5e-4  # 0.0005, proven optimal
    WEIGHT_DECAY: float = 1e-4
    EARLY_STOPPING_PATIENCE: int = 15
    GRADIENT_CLIP_NORM: float = 1.0
    
    # Data augmentation
    TEMPORAL_JITTER: float = 0.1  # 10% temporal jittering
    NOISE_STD: float = 0.01  # Gaussian noise standard deviation
    ROTATION_RANGE: float = 5.0  # Degrees for rotation augmentation
    AUGMENTATION_PROB: float = 0.5  # Probability of applying augmentation

@dataclass
class MediaPipeConfig:
    """MediaPipe hands detection configuration with validation."""
    
    MAX_NUM_HANDS: int = 2
    MIN_DETECTION_CONFIDENCE: float = 0.7  # HIGHER to avoid detecting arm/face as hands
    MIN_TRACKING_CONFIDENCE: float = 0.6   # HIGHER for better tracking accuracy
    MODEL_COMPLEXITY: int = 1  # 1=balanced (more accurate, still fast enough)
    STATIC_IMAGE_MODE: bool = False
    
    def __post_init__(self):
        """Validate configuration values."""
        assert 1 <= self.MAX_NUM_HANDS <= 2, "MAX_NUM_HANDS must be 1 or 2"
        assert 0.0 <= self.MIN_DETECTION_CONFIDENCE <= 1.0, "Confidence must be in [0, 1]"
        assert 0.0 <= self.MIN_TRACKING_CONFIDENCE <= 1.0, "Confidence must be in [0, 1]"
        assert self.MODEL_COMPLEXITY in (0, 1, 2), "MODEL_COMPLEXITY must be 0, 1, or 2"


@dataclass
class InferenceConfig:
    """Real-time inference configuration optimized for speed and accuracy."""
    
    # Frame collection - CONTINUOUS SLIDING WINDOW MODE
    TARGET_FPS: int = 15  # Increased to 15 FPS for faster response
    FRAME_INTERVAL_MS: int = 67  # ~67ms = 15 FPS
    MODEL_SEQUENCE_LENGTH: int = 60  # Model trained on 60 frames
    MIN_FRAMES_FOR_PREDICTION: int = 60  # Must have full 60-frame sequence
    MAX_FRAMES_FOR_PREDICTION: int = 60  # Keep exactly 60 frames (rolling buffer)
    
    # CONTINUOUS PREDICTION MODE (Sliding Window)
    SLIDING_WINDOW_MODE: bool = True  # Enable continuous sliding window
    SLIDING_WINDOW_STRIDE: int = 1  # Predict on every new frame
    SEND_KEYPOINTS_TO_FRONTEND: bool = True  # Send hand landmarks for visualization
    
    # Prediction thresholds - OPTIMIZED FOR CONTINUOUS FLOW
    MIN_CONFIDENCE_THRESHOLD: float = 0.55  # Lower for continuous detection
    MIN_HAND_QUALITY: float = 0.3  # Minimum hand detection quality
    MIN_KEYPOINTS: int = 10  # Minimum keypoints for valid detection
    
    # Continuous prediction settings (NO COOLDOWN - predict every frame)
    PREDICTION_COOLDOWN_MS: int = 0  # No cooldown - truly continuous
    PREDICTION_CHANGE_THRESHOLD: float = 0.05  # Low threshold for smooth transitions
    
    # Enhanced smoothing for stable predictions without lag
    PREDICTION_HISTORY_SIZE: int = 7  # Slightly larger for better stability
    STABILITY_THRESHOLD: int = 4  # Need 4/7 matches for stable prediction
    SMOOTHING_ALPHA: float = 0.7  # Exponential moving average factor
    
    # Performance constraints
    MAX_PREDICTION_TIME_MS: int = 100  # Maximum time allowed for prediction
    SESSION_TIMEOUT_SECONDS: int = 300  # 5 minutes session timeout
    
    def __post_init__(self):
        """Validate inference configuration."""
        assert self.TARGET_FPS > 0, "TARGET_FPS must be positive"
        assert 0.0 <= self.MIN_CONFIDENCE_THRESHOLD <= 1.0, "Confidence must be in [0, 1]"
        assert self.MODEL_SEQUENCE_LENGTH > 0, "Sequence length must be positive"

@dataclass
class WebSocketConfig:
    """WebSocket communication configuration."""
    
    PING_INTERVAL: int = 10  # seconds (more frequent pings for stability)
    PING_TIMEOUT: int = 120  # seconds (longer timeout to prevent disconnects)
    MAX_CONNECTIONS: int = 100
    CORS_ORIGINS: List[str] = field(default_factory=lambda: ["*"])
    MAX_MESSAGE_SIZE: int = 16 * 1024 * 1024  # 16MB
    COMPRESSION_THRESHOLD: int = 1024  # Compress messages > 1KB
    
    def __post_init__(self):
        """Validate WebSocket configuration."""
        assert self.PING_INTERVAL > 0, "PING_INTERVAL must be positive"
        assert self.PING_TIMEOUT > self.PING_INTERVAL, "PING_TIMEOUT must be > PING_INTERVAL"
        assert self.MAX_CONNECTIONS > 0, "MAX_CONNECTIONS must be positive"


@dataclass
class DatasetConfig:
    """Dataset processing configuration."""
    
    # Initial 4 signs for testing (scalable to 40+)
    INITIAL_SIGNS: List[str] = field(default_factory=lambda: ["2-Hay", "Alifmad", "Aray", "Jeem"])
    
    # Video processing
    VIDEO_EXTENSIONS: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    IMAGE_EXTENSIONS: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"])
    
    # Feature extraction parameters
    TARGET_FRAME_RATE: int = 30  # FPS for video processing
    MAX_VIDEO_DURATION: float = 10.0  # seconds
    MIN_VIDEO_DURATION: float = 1.0  # seconds
    MIN_FRAMES_PER_VIDEO: int = 30  # Minimum frames required
    
    # Data splits (must sum to 1.0)
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    
    def __post_init__(self):
        """Validate dataset configuration."""
        total_split = self.TRAIN_SPLIT + self.VAL_SPLIT + self.TEST_SPLIT
        assert abs(total_split - 1.0) < 1e-6, f"Splits must sum to 1.0, got {total_split}"
        assert all(0.0 < s < 1.0 for s in [self.TRAIN_SPLIT, self.VAL_SPLIT, self.TEST_SPLIT]), \
            "All splits must be in (0, 1)"


@dataclass
class LoggingConfig:
    """Logging configuration with best practices."""
    
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Path = field(default_factory=lambda: BACKEND_DIR / "logs" / "psl_recognition.log")
    MAX_LOG_SIZE: int = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT: int = 5
    LOG_TO_CONSOLE: bool = True
    LOG_TO_FILE: bool = True
    
    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert self.LOG_LEVEL.upper() in valid_levels, f"LOG_LEVEL must be one of {valid_levels}"
        assert self.MAX_LOG_SIZE > 0, "MAX_LOG_SIZE must be positive"
        assert self.BACKUP_COUNT >= 0, "BACKUP_COUNT must be non-negative"


def _generate_secret_key() -> str:
    """Generate or retrieve secret key securely."""
    import secrets
    env_key = os.environ.get("SECRET_KEY")
    if env_key:
        return env_key
    # Generate a secure random key for development
    # In production, always set SECRET_KEY environment variable
    import logging
    logging.warning(
        "SECRET_KEY not set in environment! Using auto-generated key. "
        "Set SECRET_KEY environment variable for production."
    )
    return secrets.token_hex(32)


@dataclass
class FlaskConfig:
    """Flask server configuration with security best practices."""
    
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = False
    SECRET_KEY: str = field(default_factory=_generate_secret_key)
    
    # CORS configuration
    CORS_ORIGINS: List[str] = field(default_factory=lambda: ["*"])  # Restrict in production!
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # File upload configuration
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER: Path = field(default_factory=lambda: UPLOADS_DIR)
    ALLOWED_EXTENSIONS: List[str] = field(default_factory=lambda: [
        "mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"
    ])
    
    # Security headers
    SEND_FILE_MAX_AGE_DEFAULT: int = 0  # Disable caching for development
    SESSION_COOKIE_SECURE: bool = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    
    def __post_init__(self):
        """Validate Flask configuration."""
        assert 1024 <= self.PORT <= 65535, "PORT must be in range [1024, 65535]"
        assert self.MAX_CONTENT_LENGTH > 0, "MAX_CONTENT_LENGTH must be positive"
        if self.DEBUG:
            import warnings
            warnings.warn("DEBUG mode is enabled. Disable in production!", UserWarning)

class EnvironmentConfig:
    """Environment-specific settings with type-safe configurations."""
    
    VALID_ENVIRONMENTS = ["development", "production", "testing"]
    
    @staticmethod
    def get_config(env: str = "development") -> Dict[str, Any]:
        """
        Get configuration for specific environment.
        
        Args:
            env: Environment name (development, production, testing)
            
        Returns:
            Dictionary of environment-specific settings
            
        Raises:
            ValueError: If environment is invalid
        """
        if env not in EnvironmentConfig.VALID_ENVIRONMENTS:
            raise ValueError(
                f"Invalid environment '{env}'. Must be one of {EnvironmentConfig.VALID_ENVIRONMENTS}"
            )
        
        configs = {
            "development": {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "MODEL_COMPLEXITY": 0,  # Fast processing for dev
                "MIN_DETECTION_CONFIDENCE": 0.3,
                "MIN_TRACKING_CONFIDENCE": 0.3,
                "ENABLE_PROFILING": True,
                "CORS_ORIGINS": ["*"],
            },
            "production": {
                "DEBUG": False,
                "LOG_LEVEL": "INFO",
                "MODEL_COMPLEXITY": 1,  # Balanced for production
                "MIN_DETECTION_CONFIDENCE": 0.5,
                "MIN_TRACKING_CONFIDENCE": 0.5,
                "ENABLE_PROFILING": False,
                # IMPORTANT: Set CORS_ORIGINS via environment variable in production!
                # Example: CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
                "CORS_ORIGINS": os.environ.get(
                    "CORS_ORIGINS", 
                    "https://yourdomain.com"
                ).split(","),
            },
            "testing": {
                "DEBUG": True,
                "LOG_LEVEL": "WARNING",
                "MODEL_COMPLEXITY": 0,
                "MIN_DETECTION_CONFIDENCE": 0.2,
                "MIN_TRACKING_CONFIDENCE": 0.2,
                "ENABLE_PROFILING": False,
                "CORS_ORIGINS": ["*"],
            }
        }
        
        return configs[env]
    
    @staticmethod
    def get_current_env() -> str:
        """Get current environment from environment variable."""
        return os.environ.get("PSL_ENV", "development").lower()


# Utility functions with proper error handling
def ensure_directories() -> None:
    """
    Ensure all required directories exist.
    Creates directories with appropriate permissions.
    """
    directories = [
        MODELS_DIR,
        SAVED_MODELS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        FEATURES_DIR,
        UPLOADS_DIR,
        BACKEND_DIR / "logs",
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            import logging
            logging.warning(f"Could not create directory {directory}: {e}")


def get_model_path(model_name: str = "best_model", base_dir: Optional[Path] = None) -> Path:
    """
    Get path for model file.
    
    Args:
        model_name: Name of the model file (without extension)
        base_dir: Base directory for models (defaults to MODELS_DIR)
        
    Returns:
        Path to model file
    """
    if base_dir is None:
        base_dir = MODELS_DIR
    return base_dir / f"{model_name}.pth"


def get_config_path(model_name: str = "best_model", base_dir: Optional[Path] = None) -> Path:
    """
    Get path for model configuration file.
    
    Args:
        model_name: Name of the model (without extension)
        base_dir: Base directory for models (defaults to MODELS_DIR)
        
    Returns:
        Path to config file
    """
    if base_dir is None:
        base_dir = MODELS_DIR
    return base_dir / f"{model_name}_config.json"


def get_label_map_path(model_name: str = "best_model", base_dir: Optional[Path] = None) -> Path:
    """
    Get path for label map file.
    
    Args:
        model_name: Name of the model (without extension)
        base_dir: Base directory for models (defaults to MODELS_DIR)
        
    Returns:
        Path to label map file
    """
    if base_dir is None:
        base_dir = MODELS_DIR
    return base_dir / f"{model_name}_labels.txt"


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Validate if file has an allowed extension.
    
    Args:
        filename: Name of the file to validate
        allowed_extensions: List of allowed extensions (without dots)
        
    Returns:
        True if extension is allowed, False otherwise
    """
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in [ext.lower().lstrip('.') for ext in allowed_extensions]


# Initialize directories on import (safe to call multiple times)
ensure_directories()

# Create config instances (singletons)
model_config = ModelConfig()
mediapipe_config = MediaPipeConfig()
inference_config = InferenceConfig()
websocket_config = WebSocketConfig()
dataset_config = DatasetConfig()
logging_config = LoggingConfig()
flask_config = FlaskConfig()

def get_frontend_config() -> Dict[str, Any]:
    """
    Get configuration values that should be synced with the frontend.
    
    Returns:
        Dictionary containing configuration values for frontend synchronization
    """
    return {
        "MODEL_SEQUENCE_LENGTH": inference_config.MODEL_SEQUENCE_LENGTH,
        "TARGET_FPS": inference_config.TARGET_FPS,
        "PREDICTION_HISTORY_SIZE": inference_config.PREDICTION_HISTORY_SIZE,
        "STABILITY_THRESHOLD": inference_config.STABILITY_THRESHOLD,
        "MIN_CONFIDENCE_THRESHOLD": inference_config.MIN_CONFIDENCE_THRESHOLD,
        "FRAME_INTERVAL_MS": inference_config.FRAME_INTERVAL_MS,
        "MIN_FRAMES_FOR_PREDICTION": inference_config.MIN_FRAMES_FOR_PREDICTION,
    }


# Export main configurations and utilities
__all__ = [
    # Path constants
    "BASE_DIR",
    "BACKEND_DIR",
    "FRONTEND_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "SAVED_MODELS_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "SPLITS_DIR",
    "FEATURES_DIR",
    "UPLOADS_DIR",
    # Config classes
    "ModelConfig",
    "MediaPipeConfig", 
    "InferenceConfig",
    "WebSocketConfig",
    "DatasetConfig",
    "LoggingConfig",
    "FlaskConfig",
    "EnvironmentConfig",
    # Config instances (singletons)
    "model_config",
    "mediapipe_config",
    "inference_config",
    "websocket_config",
    "dataset_config",
    "logging_config",
    "flask_config",
    # Utility functions
    "ensure_directories",
    "get_model_path",
    "get_config_path", 
    "get_label_map_path",
    "validate_file_extension",
    "get_frontend_config",
]

