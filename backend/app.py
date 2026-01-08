#!/usr/bin/env python3
"""
Flask application with WebSocket support for PSL recognition system.
Real-time sign language recognition with optimized performance.
"""

import os
import json
import base64
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from config import (
    flask_config, websocket_config, inference_config,
    mediapipe_config, ensure_directories, EnvironmentConfig
)
from models.model_manager import ModelManager
from inference.predictor import RealTimePredictor
from utils.mediapipe_utils import MediaPipeProcessor, validate_landmarks

# Configure logging
from config import logging_config

# Ensure log directory exists
log_dir = logging_config.LOG_FILE.parent
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging handlers
handlers = []
if logging_config.LOG_TO_FILE:
    handlers.append(logging.FileHandler(logging_config.LOG_FILE))
if logging_config.LOG_TO_CONSOLE:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=getattr(logging, logging_config.LOG_LEVEL),
    format=logging_config.LOG_FORMAT,
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = flask_config.SECRET_KEY

# Initialize CORS
CORS(app, origins=flask_config.CORS_ORIGINS)

# Initialize SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_interval=websocket_config.PING_INTERVAL,
    ping_timeout=websocket_config.PING_TIMEOUT,
    max_http_buffer_size=16 * 1024 * 1024  # 16MB
)

# Initialize components
MODELS_DIR = Path(__file__).resolve().parent / "models"
model_manager = ModelManager(models_dir=MODELS_DIR)
predictor = None
mp_processor = None

# Session management
sessions: Dict[str, Dict[str, Any]] = {}

# Temporal smoothing configuration (from config)
PREDICTION_HISTORY_SIZE = inference_config.PREDICTION_HISTORY_SIZE  # Keep last N predictions
STABILITY_THRESHOLD = inference_config.STABILITY_THRESHOLD  # Must appear N times before displaying
MIN_CONFIDENCE_FOR_VOTE = 0.50  # Lowered for continuous flow - majority voting handles stability

# Performance monitoring
performance_stats = {
    'total_connections': 0,
    'active_connections': 0,
    'total_predictions': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'avg_prediction_time': 0.0,
    'avg_hand_detection_time': 0.0
}


def initialize_system():
    """Initialize the PSL recognition system."""
    global predictor, mp_processor
    
    try:
        logger.info("Initializing PSL recognition system...")
        
        # Ensure directories exist
        ensure_directories()
        
        # Initialize MediaPipe processor
        mp_processor = MediaPipeProcessor(
            min_detection_confidence=mediapipe_config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=mediapipe_config.MIN_TRACKING_CONFIDENCE,
            model_complexity=mediapipe_config.MODEL_COMPLEXITY
        )
        
        # Load best model
        best_model_name = model_manager.get_best_model()
        if best_model_name:
            model, config, labels = model_manager.load_model(best_model_name)
            if model is not None:
                config = config or {}
                if 'sequence_length' not in config and 'target_seq_len' in config:
                    config['sequence_length'] = config['target_seq_len']
                predictor = RealTimePredictor(model, config, labels)
                logger.info(f"Loaded model: {best_model_name}")
            else:
                logger.error("Failed to load model")
                predictor = None
        else:
            logger.warning("No trained model found")
            predictor = None
        
        logger.info("System initialization completed")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        predictor = None
        mp_processor = None


def get_majority_vote(prediction_history: List[Dict[str, Any]]) -> Optional[Tuple[str, float]]:
    """
    Get majority vote from prediction history with stability threshold.
    
    Args:
        prediction_history: List of recent predictions with labels and confidences
        
    Returns:
        Tuple of (label, avg_confidence) if stable, None otherwise
    """
    if not prediction_history or len(prediction_history) < STABILITY_THRESHOLD:
        return None
    
    # Count occurrences of each prediction (only high confidence ones)
    label_counts = {}
    label_confidences = {}
    
    for pred in prediction_history:
        label = pred.get('prediction')
        confidence = pred.get('confidence', 0.0)
        
        # Skip if prediction is None
        if label is None:
            continue
        
        # Only count predictions with sufficient confidence
        if confidence >= MIN_CONFIDENCE_FOR_VOTE:
            label_counts[label] = label_counts.get(label, 0) + 1
            if label not in label_confidences:
                label_confidences[label] = []
            label_confidences[label].append(confidence)
    
    if not label_counts:
        return None
    
    # Find most common prediction
    most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
    count = label_counts[most_common_label]
    
    # Check if it meets stability threshold
    if count >= STABILITY_THRESHOLD:
        # Calculate average confidence for this label
        avg_confidence = sum(label_confidences[most_common_label]) / len(label_confidences[most_common_label])
        return most_common_label, avg_confidence
    
    return None


def cleanup_old_sessions():
    """Clean up old sessions."""
    current_time = time.time()
    expired_sessions = []
    
    for session_id, session_data in sessions.items():
        last_activity = session_data.get('last_activity', 0)
        if current_time - last_activity > inference_config.SESSION_TIMEOUT_SECONDS:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del sessions[session_id]
        logger.debug(f"Cleaned up expired session: {session_id}")


def extract_keypoints_from_frame(frame_data: str) -> Optional[np.ndarray]:
    """Extract keypoints from base64 encoded frame."""
    try:
        if mp_processor is None:
            logger.error("MediaPipe processor not initialized")
            return None
        
        # Decode base64 image
        image_data = base64.b64decode(frame_data.split(',')[-1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
        
        # Extract landmarks
        landmarks = mp_processor.extract_landmarks(frame)
        
        if landmarks is not None and validate_landmarks(landmarks):
            return landmarks
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error extracting keypoints: {e}")
        return None


# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    global performance_stats
    
    performance_stats['total_connections'] += 1
    performance_stats['active_connections'] += 1
    
    session_id = request.sid
    sessions[session_id] = {
        'connected_at': time.time(),
        'last_activity': time.time(),
        'frames': [],
        'predictions': [],
        'prediction_history': [],  # Initialize prediction history on connect
        'recognition_active': False,
        'last_prediction_time': 0,
        'last_prediction_label': None,
        'last_prediction_confidence': 0,
        'last_stable_prediction': None,
        'last_stable_confidence': 0
    }
    
    logger.info(f"Client connected: {session_id}")
    emit('connected', {'session_id': session_id, 'status': 'ready'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    global performance_stats
    
    performance_stats['active_connections'] -= 1
    
    session_id = request.sid
    if session_id in sessions:
        del sessions[session_id]
    
    logger.info(f"Client disconnected: {session_id}")


@socketio.on('start_recognition')
def handle_start_recognition(data):
    """Handle start recognition request."""
    session_id = request.sid
    
    if session_id not in sessions:
        emit('error', {'message': 'Session not found'})
        return
    
    sessions[session_id]['recognition_active'] = True
    sessions[session_id]['frames'] = []
    sessions[session_id]['predictions'] = []
    sessions[session_id]['prediction_history'] = []  # For temporal smoothing
    sessions[session_id]['last_prediction_time'] = 0
    sessions[session_id]['last_prediction_label'] = None
    sessions[session_id]['last_prediction_confidence'] = 0
    sessions[session_id]['last_stable_prediction'] = None  # Last stable majority vote
    sessions[session_id]['last_stable_confidence'] = 0
    
    logger.info(f"Recognition started for session: {session_id}")
    emit('recognition_started', {'session_id': session_id})


@socketio.on('stop_recognition')
def handle_stop_recognition(data):
    """Handle stop recognition request."""
    session_id = request.sid
    
    if session_id not in sessions:
        emit('error', {'message': 'Session not found'})
        return
    
    sessions[session_id]['recognition_active'] = False
    sessions[session_id]['frames'] = []  # Clear frames when stopping
    sessions[session_id]['prediction_history'] = []  # Clear history to prevent stale data
    sessions[session_id]['last_stable_prediction'] = None  # Reset stable prediction
    
    logger.info(f"Recognition stopped for session: {session_id}")
    emit('recognition_stopped', {'session_id': session_id})


@socketio.on('reset_recognition')
def handle_reset_recognition(data):
    """Handle reset recognition request - clears frame buffer."""
    session_id = request.sid
    
    if session_id not in sessions:
        emit('error', {'message': 'Session not found'})
        return
    
    # Clear frames, predictions, and history
    sessions[session_id]['frames'] = []
    sessions[session_id]['predictions'] = []
    sessions[session_id]['prediction_history'] = []
    sessions[session_id]['last_stable_prediction'] = None
    sessions[session_id]['last_stable_confidence'] = 0
    
    logger.info(f"Recognition reset for session: {session_id}")
    emit('recognition_reset', {
        'session_id': session_id,
        'message': 'Frame buffer and prediction history cleared'
    })


@socketio.on('update_settings')
def handle_update_settings(data):
    """Handle settings updates from the client."""
    session_id = request.sid
    settings = (data or {}).get('settings', {})

    if session_id in sessions:
        sessions[session_id]['settings'] = settings

    emit('settings_updated', {
        'session_id': session_id,
        'settings': settings,
        'timestamp': time.time()
    })


@socketio.on('frame_data')
def handle_frame_data(data):
    """Handle incoming frame data with CONTINUOUS SLIDING WINDOW prediction."""
    global performance_stats
    
    session_id = request.sid
    
    # CRITICAL: Check if session still exists (prevents errors after disconnect)
    if session_id not in sessions:
        logger.warning(f"Frame received for non-existent session: {session_id}")
        return
    
    if not sessions[session_id].get('recognition_active', False):
        return

    if predictor is None or mp_processor is None:
        emit('frame_processed', {
            'session_id': session_id,
            'status': 'system_not_ready',
            'frames_collected': len(sessions[session_id].get('frames', [])),
            'target_frames': inference_config.MIN_FRAMES_FOR_PREDICTION
        })
        return
    
    # Defensive initialization of session keys (in case of corruption)
    if 'prediction_history' not in sessions[session_id]:
        sessions[session_id]['prediction_history'] = []
    if 'frames' not in sessions[session_id]:
        sessions[session_id]['frames'] = []
    
    try:
        start_time = time.time()
        
        # Extract keypoints
        keypoints = extract_keypoints_from_frame(data['frame'])
        
        # Prepare keypoints for frontend visualization
        keypoints_for_frontend = None
        if keypoints is not None and inference_config.SEND_KEYPOINTS_TO_FRONTEND:
            # Send first 126 features (actual hand landmarks, not padding)
            keypoints_for_frontend = keypoints[:126].tolist()
        
        if keypoints is None:
            emit('frame_processed', {
                'session_id': session_id,
                'status': 'no_hands',
                'keypoints': None,
                'prediction': None
            })
            return
        
        # Validate keypoints array shape before appending
        if not isinstance(keypoints, np.ndarray):
            logger.warning(f"Invalid keypoints type: {type(keypoints)}, expected numpy array")
            return
        
        expected_dim = inference_config.MODEL_SEQUENCE_LENGTH if hasattr(inference_config, 'MODEL_SEQUENCE_LENGTH') else 189
        # keypoints should be 1D array with feature dimension
        if keypoints.ndim != 1 or len(keypoints) < 63:  # Minimum: 1 hand * 21 landmarks * 3 coords
            logger.warning(f"Invalid keypoints shape: {keypoints.shape}, skipping frame")
            emit('frame_processed', {
                'session_id': session_id,
                'status': 'invalid_keypoints',
                'keypoints': None
            })
            return
        
        # Add to session frames (SLIDING WINDOW)
        sessions[session_id]['frames'].append(keypoints)
        sessions[session_id]['last_activity'] = time.time()
        
        # ROLLING BUFFER: Keep exactly MODEL_SEQUENCE_LENGTH frames (60 frames)
        target_frames = inference_config.MODEL_SEQUENCE_LENGTH
        if len(sessions[session_id]['frames']) > target_frames:
            # Remove oldest frame (FIFO queue)
            sessions[session_id]['frames'] = sessions[session_id]['frames'][-target_frames:]
        
        frames_collected = len(sessions[session_id]['frames'])
        
        # CONTINUOUS SLIDING WINDOW PREDICTION: Predict on EVERY frame once buffer is full
        if frames_collected == target_frames and predictor is not None:
            current_time = time.time()
            
            # Make prediction on EVERY frame (CONTINUOUS MODE - no cooldown)
            sequence = np.array(sessions[session_id]['frames'])
            prediction_result = predictor.predict(sequence)
            
            if prediction_result is not None:
                prediction, confidence, all_predictions = prediction_result
                
                # ALWAYS add to prediction history (even lower confidence for better smoothing)
                if confidence >= MIN_CONFIDENCE_FOR_VOTE:
                    # Initialize history if needed
                    if 'prediction_history' not in sessions[session_id]:
                        sessions[session_id]['prediction_history'] = []
                    
                    sessions[session_id]['prediction_history'].append({
                        'prediction': prediction,
                        'confidence': confidence,
                        'timestamp': current_time
                    })
                    
                    # Keep only recent history for majority voting
                    if len(sessions[session_id]['prediction_history']) > PREDICTION_HISTORY_SIZE:
                        sessions[session_id]['prediction_history'] = \
                            sessions[session_id]['prediction_history'][-PREDICTION_HISTORY_SIZE:]
                    
                    # Get majority vote for stability
                    stable_prediction = get_majority_vote(sessions[session_id]['prediction_history'])
                    
                    if stable_prediction is not None:
                        stable_label, stable_confidence = stable_prediction
                        last_stable = sessions[session_id].get('last_stable_prediction')
                        
                        # Only emit if this is a NEW stable prediction
                        if stable_label != last_stable:
                            # Update session
                            sessions[session_id]['last_stable_prediction'] = stable_label
                            sessions[session_id]['last_stable_confidence'] = stable_confidence
                            sessions[session_id]['last_prediction_time'] = current_time
                            
                            # Store in predictions log
                            sessions[session_id]['predictions'].append({
                                'prediction': stable_label,
                                'confidence': stable_confidence,
                                'timestamp': current_time
                            })
                            
                            # Keep only recent predictions
                            if len(sessions[session_id]['predictions']) > 10:
                                sessions[session_id]['predictions'] = \
                                    sessions[session_id]['predictions'][-10:]
                            
                            # Update performance stats
                            prediction_time = current_time - start_time
                            performance_stats['total_predictions'] += 1
                            performance_stats['successful_predictions'] += 1
                            performance_stats['avg_prediction_time'] = (
                                performance_stats['avg_prediction_time'] * 0.9 + prediction_time * 0.1
                            )
                            
                            # EMIT STABLE PREDICTION to frontend with keypoints
                            emit('prediction', {
                                'session_id': session_id,
                                'prediction': stable_label,
                                'confidence': stable_confidence,
                                'all_predictions': all_predictions,
                                'frames_collected': frames_collected,
                                'keypoints': keypoints_for_frontend,  # Add keypoints for skeleton
                                'is_stable': True,
                                'timestamp': current_time
                            })
                            
                            logger.info(f"STABLE Prediction: {stable_label} ({stable_confidence:.2f})")
                        else:
                            # Same stable prediction, send keypoints for continuous skeleton update
                            emit('frame_processed', {
                                'session_id': session_id,
                                'status': 'stable_prediction_continues',
                                'current_prediction': stable_label,
                                'frames_collected': frames_collected,
                                'keypoints': keypoints_for_frontend  # Keep skeleton updated
                            })
                    else:
                        # Not yet stable, gathering more data (still send keypoints)
                        emit('frame_processed', {
                            'session_id': session_id,
                            'status': 'stabilizing',
                            'history_size': len(sessions[session_id]['prediction_history']),
                            'threshold': STABILITY_THRESHOLD,
                            'frames_collected': frames_collected,
                            'keypoints': keypoints_for_frontend,  # Keep skeleton visible while stabilizing
                            'current_top_prediction': prediction,
                            'current_confidence': confidence
                        })
                else:
                    # Low confidence, but still send keypoints for skeleton
                    emit('frame_processed', {
                        'session_id': session_id,
                        'status': 'low_confidence',
                        'confidence': confidence,
                        'frames_collected': frames_collected,
                        'keypoints': keypoints_for_frontend  # Always show skeleton when hands detected
                    })
            else:
                performance_stats['failed_predictions'] += 1
                emit('frame_processed', {
                    'session_id': session_id,
                    'status': 'prediction_failed',
                    'frames_collected': frames_collected
                })
        else:
            # Still collecting initial frames - send keypoints for skeleton
            emit('frame_processed', {
                'session_id': session_id,
                'status': 'collecting_frames',
                'frames_collected': frames_collected,
                'target_frames': inference_config.MIN_FRAMES_FOR_PREDICTION,
                'keypoints': keypoints_for_frontend,  # Show skeleton even during initial collection
                'progress_percent': int((frames_collected / target_frames) * 100)
            })
    
    except Exception as e:
        logger.error(f"Error processing frame for session {session_id}: {e}", exc_info=True)
        performance_stats['failed_predictions'] += 1
        try:
            emit('error', {'message': 'Frame processing failed', 'session_id': session_id})
        except Exception as emit_error:
            logger.error(f"Failed to emit error to client: {emit_error}")


@socketio.on('get_session_info')
def handle_get_session_info(data):
    """Handle session info request."""
    session_id = request.sid
    
    if session_id not in sessions:
        emit('error', {'message': 'Session not found'})
        return
    
    session_data = sessions[session_id]
    
    emit('session_info', {
        'session_id': session_id,
        'connected_at': session_data.get('connected_at'),
        'last_activity': session_data.get('last_activity'),
        'recognition_active': session_data.get('recognition_active', False),
        'frames_collected': len(session_data.get('frames', [])),
        'predictions_count': len(session_data.get('predictions', []))
    })


@socketio.on('get_system_status')
def handle_get_system_status(data):
    """Handle system status request."""
    global performance_stats
    
    # Clean up old sessions
    cleanup_old_sessions()
    
    status = {
        'system_ready': predictor is not None and mp_processor is not None,
        'model_loaded': predictor is not None,
        'active_sessions': len(sessions),
        'performance_stats': performance_stats,
        'timestamp': time.time()
    }
    
    emit('system_status', status)


# HTTP routes
@app.route('/')
def index():
    """Serve main application."""
    return send_from_directory('../frontend', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('../frontend', filename)


@app.route('/api/status')
def api_status():
    """Get system status via HTTP."""
    global performance_stats
    
    cleanup_old_sessions()
    
    return jsonify({
        'status': 'running',
        'system_ready': predictor is not None and mp_processor is not None,
        'model_loaded': predictor is not None,
        'active_sessions': len(sessions),
        'performance_stats': performance_stats,
        'timestamp': time.time()
    })


@app.route('/api/models')
def api_models():
    """Get available models."""
    models = model_manager.list_models()
    return jsonify({'models': models})


@app.route('/api/model/<model_name>')
def api_model_info(model_name):
    """Get model information."""
    info = model_manager.get_model_info(model_name)
    if info is None:
        return jsonify({'error': 'Model not found'}), 404
    
    return jsonify(info)


@app.route('/api/load_model/<model_name>', methods=['POST'])
def api_load_model(model_name):
    """Load a specific model."""
    global predictor
    
    try:
        model, config, labels = model_manager.load_model(model_name)
        if model is not None:
            predictor = RealTimePredictor(model, config, labels)
            logger.info(f"Model loaded: {model_name}")
            return jsonify({'status': 'success', 'model': model_name})
        else:
            return jsonify({'error': 'Failed to load model'}), 500
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return jsonify({'error': str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# Background tasks
def cleanup_sessions():
    """Background task to clean up old sessions."""
    while True:
        time.sleep(60)  # Run every minute
        cleanup_old_sessions()


# Application startup
if __name__ == '__main__':
    # Get environment configuration
    env = os.environ.get('ENVIRONMENT', 'development')
    config = EnvironmentConfig.get_config(env)
    
    # Update Flask config
    app.config.update(config)
    
    # Initialize system
    initialize_system()
    
    # Start background tasks
    import threading
    cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
    cleanup_thread.start()
    
    # Run application
    logger.info(f"Starting PSL recognition server on {flask_config.HOST}:{flask_config.PORT}")
    socketio.run(
        app,
        host=flask_config.HOST,
        port=flask_config.PORT,
        debug=flask_config.DEBUG,
        use_reloader=False  # Disable reloader for production
    )