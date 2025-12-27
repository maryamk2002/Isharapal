#!/usr/bin/env python3
"""
Flask application V2 with WebSocket support for PSL recognition system.
ENHANCED: Fixed lagging, flickering, old predictions, and synchronization issues.

Key Improvements:
- Uses predictor_v2.py with optimized sliding window
- Faster response time (24-frame buffer for demo)
- Better stability filtering (no flickering)
- Synchronized keypoints visualization
- Auto-reset when hands disappear
- Cleaner session management
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
from config_v2 import (
    inference_config_v2,  # V2 inference settings
    filtering_config,  # Keypoint filtering settings
    recording_config,  # Recording management settings
    monitoring_config  # Performance monitoring settings
)
from models.model_manager import ModelManager
from inference.predictor_v2 import RealTimePredictorV2  # V2 Predictor
from utils.mediapipe_utils import MediaPipeProcessor, validate_landmarks
from utils.keypoint_filter import KeypointFilter  # NEW: Keypoint filtering
from utils.recording_manager import RecordingManager  # NEW: Recording automation
from utils.performance_monitor import PerformanceMonitor  # NEW: Performance tracking
from feedback_system import FeedbackDatabase  # Feedback system

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
MODELS_DIR = Path(__file__).resolve().parent / "saved_models" / "v2"
model_manager = ModelManager(models_dir=MODELS_DIR)
predictor = None
mp_processor = None
feedback_db = None  # Feedback database

# Session management (simplified)
sessions: Dict[str, Dict[str, Any]] = {}

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
    """Initialize the PSL recognition system with V2 predictor."""
    global predictor, mp_processor, feedback_db, global_performance_monitor
    
    try:
        logger.info("Initializing PSL recognition system V2...")
        
        # Initialize feedback database
        feedback_db = FeedbackDatabase()
        logger.info("[OK] Feedback system initialized")
        
        # Initialize global performance monitor
        global_performance_monitor = PerformanceMonitor(
            fps_window=monitoring_config.FPS_WINDOW_SIZE,
            inference_window=monitoring_config.INFERENCE_WINDOW_SIZE,
            auto_save_interval=monitoring_config.AUTO_SAVE_INTERVAL_SEC
        )
        logger.info("[OK] Performance monitor initialized")
        
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
            # Force CPU device for model loading
            import torch
            device = torch.device('cpu')
            model, config, labels = model_manager.load_model(best_model_name, device=device)
            if model is not None:
                config = config or {}
                if 'sequence_length' not in config and 'target_seq_len' in config:
                    config['sequence_length'] = config['target_seq_len']
                
                # Create V2 predictor with optimized settings
                predictor = RealTimePredictorV2(
                    model=model,
                    config=config,
                    labels=labels,
                    sliding_window_size=inference_config_v2.SLIDING_WINDOW_SIZE,
                    min_prediction_frames=inference_config_v2.MIN_FRAMES_FOR_PREDICTION,
                    stability_votes=inference_config_v2.STABILITY_VOTES,
                    stability_threshold=inference_config_v2.STABILITY_THRESHOLD,
                    min_confidence=inference_config_v2.MIN_CONFIDENCE,
                    reset_on_no_hands=True  # Auto-clear buffer when hands disappear
                )
                
                logger.info(f"[OK] Loaded model: {best_model_name}")
                logger.info(f"[OK] V2 Predictor initialized with {inference_config_v2.SLIDING_WINDOW_SIZE}-frame sliding window")
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


def extract_keypoints_from_frame(frame_data: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract keypoints from base64 encoded frame.
    
    Returns:
        Tuple of (keypoints_189, keypoints_for_viz) or (None, None) if failed
        - keypoints_189: Full 189-dim features for model
        - keypoints_for_viz: Hand landmarks for frontend visualization
    """
    try:
        if mp_processor is None:
            logger.error("MediaPipe processor not initialized")
            return None, None
        
        # Debug: Check frame_data format
        if not frame_data or not isinstance(frame_data, str):
            logger.error(f"Invalid frame_data type: {type(frame_data)}")
            return None, None
            
        if len(frame_data) < 100:
            logger.error(f"Frame data too short: {len(frame_data)} bytes, preview: {frame_data[:50]}")
            return None, None
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(frame_data.split(',')[-1])
        except Exception as e:
            logger.error(f"Base64 decode failed: {e}, data length: {len(frame_data)}")
            return None, None
            
        nparr = np.frombuffer(image_data, np.uint8)
        
        if len(nparr) == 0:
            logger.error("Empty numpy array after base64 decode")
            return None, None
            
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error(f"cv2.imdecode returned None. Array size: {len(nparr)}, dtype: {nparr.dtype}")
            return None, None
        
        # Extract landmarks
        landmarks = mp_processor.extract_landmarks(frame)
        
        if landmarks is not None and validate_landmarks(landmarks):
            # Full 189-dim features for model
            keypoints_189 = landmarks
            
            # First 126 features for visualization (actual hand landmarks)
            keypoints_for_viz = landmarks[:126]
            
            return keypoints_189, keypoints_for_viz
        else:
            return None, None
            
    except Exception as e:
        logger.error(f"Error extracting keypoints: {e}")
        return None, None


# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    global performance_stats
    
    performance_stats['total_connections'] += 1
    performance_stats['active_connections'] += 1
    
    session_id = request.sid
    
    # Initialize session with V2 predictor instance + new components
    sessions[session_id] = {
        'connected_at': time.time(),
        'last_activity': time.time(),
        'recognition_active': False,
        'predictor': None,
        'keypoint_filter': None,  # NEW: Keypoint filter
        'recording_manager': None,  # NEW: Recording manager
        'frames_processed': 0,
        'predictions_sent': 0,
        'consecutive_no_hands': 0,
        'last_landmarks': None,
        'repeated_landmarks_count': 0,
    }
    
    # Start performance monitoring for this session
    if global_performance_monitor:
        global_performance_monitor.start_session(session_id)
    
    logger.info(f"Client connected: {session_id}")
    emit('connected', {'session_id': session_id, 'status': 'ready'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    global performance_stats
    
    performance_stats['active_connections'] -= 1
    
    session_id = request.sid
    if session_id in sessions:
        # Clean up session predictor
        if 'predictor' in sessions[session_id] and sessions[session_id]['predictor'] is not None:
            sessions[session_id]['predictor'].clear_buffer()
        del sessions[session_id]
    
    logger.info(f"Client disconnected: {session_id}")


@socketio.on('start_recognition')
def handle_start_recognition(data):
    """Handle start recognition request."""
    session_id = request.sid
    
    if session_id not in sessions:
        emit('error', {'message': 'Session not found'})
        return
    
    # Initialize session-specific predictor
    if predictor is not None:
        # Create a fresh predictor instance for this session
        # (shares model weights but has independent buffer)
        sessions[session_id]['predictor'] = RealTimePredictorV2(
            model=predictor.model,
            config=predictor.config,
            labels=predictor.labels,
            device=predictor.device,
            sliding_window_size=inference_config_v2.SLIDING_WINDOW_SIZE,
            min_prediction_frames=inference_config_v2.MIN_FRAMES_FOR_PREDICTION,
            stability_votes=inference_config_v2.STABILITY_VOTES,
            stability_threshold=inference_config_v2.STABILITY_THRESHOLD,
            min_confidence=inference_config_v2.MIN_CONFIDENCE,
            reset_on_no_hands=True
        )
    else:
        sessions[session_id]['predictor'] = None
    
    # Initialize keypoint filter for this session
    if filtering_config.ENABLE_FILTERING:
        sessions[session_id]['keypoint_filter'] = KeypointFilter(
            window_size=filtering_config.MOVING_AVERAGE_WINDOW,
            jitter_threshold=filtering_config.JITTER_THRESHOLD,
            stuck_threshold_frames=filtering_config.STUCK_THRESHOLD_FRAMES,
            stuck_movement_threshold=filtering_config.STUCK_MOVEMENT_THRESHOLD
        )
    
    # Initialize recording manager for this session
    if recording_config.AUTO_STOP_ENABLED:
        sessions[session_id]['recording_manager'] = RecordingManager(
            idle_timeout_sec=recording_config.IDLE_TIMEOUT_SEC,
            min_confidence=recording_config.MIN_CONFIDENCE_FOR_RECORDING,
            auto_save=recording_config.AUTO_SAVE_SEGMENTS
        )
        sessions[session_id]['recording_manager'].start_session(session_id)
    
    sessions[session_id]['recognition_active'] = True
    sessions[session_id]['frames_processed'] = 0
    sessions[session_id]['predictions_sent'] = 0
    sessions[session_id]['last_activity'] = time.time()
    
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
    
    # Clear predictor buffer
    if sessions[session_id].get('predictor') is not None:
        sessions[session_id]['predictor'].clear_buffer()
    
    logger.info(f"Recognition stopped for session: {session_id}")
    emit('recognition_stopped', {'session_id': session_id})


@socketio.on('reset_recognition')
def handle_reset_recognition(data):
    """Handle reset recognition request - clears buffer."""
    session_id = request.sid
    
    if session_id not in sessions:
        emit('error', {'message': 'Session not found'})
        return
    
    # Clear predictor buffer
    if sessions[session_id].get('predictor') is not None:
        sessions[session_id]['predictor'].clear_buffer()
    
    sessions[session_id]['frames_processed'] = 0
    sessions[session_id]['predictions_sent'] = 0
    
    logger.info(f"Recognition reset for session: {session_id}")
    emit('recognition_reset', {
        'session_id': session_id,
        'message': 'Buffer cleared'
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


@socketio.on('feedback')
def handle_feedback(data):
    """
    Handle user feedback on predictions.
    Stores feedback in SQLite database for analysis and retraining.
    """
    global feedback_db
    
    session_id = request.sid
    
    if session_id not in sessions:
        emit('error', {'message': 'Session not found'})
        return
    
    try:
        # Extract feedback data
        label = data.get('label')
        is_correct = data.get('is_correct')
        metadata = data.get('metadata', {})
        
        if not label or is_correct is None:
            emit('error', {'message': 'Invalid feedback data'})
            return
        
        # Get confidence from metadata
        confidence = metadata.get('confidence', 0.0)
        
        # Store feedback
        if feedback_db:
            feedback_id = feedback_db.add_feedback(
                predicted_label=label,
                confidence=confidence,
                is_correct=is_correct,
                session_id=session_id,
                metadata=metadata
            )
            
            logger.info(
                f"Feedback received - Session: {session_id}, "
                f"Label: {label}, Correct: {is_correct}, "
                f"Confidence: {confidence:.3f}"
            )
            
            # Record feedback in performance monitor for accuracy tracking
            if global_performance_monitor:
                global_performance_monitor.record_feedback(label, is_correct)
            
            emit('feedback_received', {
                'feedback_id': feedback_id,
                'status': 'success',
                'message': 'Thank you for your feedback!'
            })
        else:
            logger.warning("Feedback database not initialized")
            emit('feedback_received', {
                'status': 'warning',
                'message': 'Feedback received but not stored'
            })
    
    except Exception as e:
        logger.error(f"Error handling feedback: {e}")
        emit('error', {'message': f'Failed to process feedback: {str(e)}'})


@socketio.on('frame_data')
def handle_frame_data(data):
    """
    Handle incoming frame data with V2 optimized prediction.
    
    Key improvements:
    - Faster response (32-frame buffer)
    - Better stability (no flickering)
    - Auto-reset on hand disappearance (no old predictions)
    - Synchronized keypoints
    """
    global performance_stats
    
    session_id = request.sid
    
    # Check if session still exists
    if session_id not in sessions:
        logger.warning(f"Frame received for non-existent session: {session_id}")
        return
    
    if not sessions[session_id].get('recognition_active', False):
        return

    if predictor is None or mp_processor is None:
        emit('frame_processed', {
            'session_id': session_id,
            'status': 'system_not_ready'
        })
        return
    
    # Get session predictor
    session_predictor = sessions[session_id].get('predictor')
    if session_predictor is None:
        emit('frame_processed', {
            'session_id': session_id,
            'status': 'predictor_not_initialized'
        })
        return
    
    try:
        start_time = time.time()
        current_timestamp = time.time()
        
        # Record frame sent (for FPS calculation)
        if global_performance_monitor:
            global_performance_monitor.record_frame_processed()
        
        # Extract keypoints
        keypoints_189, keypoints_for_viz = extract_keypoints_from_frame(data['frame'])
        
        # Get session components
        keypoint_filter = sessions[session_id].get('keypoint_filter')
        recording_manager = sessions[session_id].get('recording_manager')
        
        # Prepare response
        response = {
            'session_id': session_id,
            'timestamp': current_timestamp,
        }
        
        # Add keypoints for visualization if available
        if keypoints_for_viz is not None and inference_config.SEND_KEYPOINTS_TO_FRONTEND:
            response['keypoints'] = keypoints_for_viz.tolist()
        else:
            response['keypoints'] = None
        
        # If no hands detected
        if keypoints_189 is None:
            sessions[session_id]['consecutive_no_hands'] += 1
            
            # TOLERANCE - only clear buffer after 2 consecutive no-hand frames
            if sessions[session_id]['consecutive_no_hands'] >= 2:
                session_predictor.add_frame(None)
                sessions[session_id]['last_landmarks'] = None
                sessions[session_id]['repeated_landmarks_count'] = 0
                if keypoint_filter:
                    keypoint_filter.reset()
            
            # Update recording manager (no hands)
            if recording_manager:
                segment = recording_manager.on_no_hands(current_timestamp)
                if segment:
                    response['recording_segment'] = segment
                    response['recording_status'] = 'stopped'
                    logger.info(f"Recording segment completed: {segment['label']} ({segment['duration']:.1f}s)")
            
            response['status'] = 'no_hands'
            response['prediction'] = None
            response['keypoints'] = sessions[session_id].get('last_landmarks')  # Keep showing last landmarks
            emit('frame_processed', response)
            return
        
        # Reset no-hands counter
        sessions[session_id]['consecutive_no_hands'] = 0
        
        # Record landmarks extracted
        if global_performance_monitor:
            global_performance_monitor.record_landmarks_extracted()
        
        # Apply keypoint filtering (if enabled)
        filtered_keypoints = keypoints_189
        if keypoint_filter:
            keypoint_filter.add_keypoint(keypoints_189)
            filtered = keypoint_filter.get_filtered()
            
            # Check for stuck sequence
            if keypoint_filter.is_stuck():
                logger.warning(f"Stuck sequence detected for session {session_id}, resetting buffer")
                session_predictor.clear_buffer()
                keypoint_filter.reset()
                response['status'] = 'stuck_sequence_reset'
                response['message'] = 'Please move your hand'
                emit('frame_processed', response)
                return
            
            # Use filtered keypoints if available
            if filtered is not None:
                filtered_keypoints = filtered
        
        # Detect frozen/repeated frames (legacy check - kept for compatibility)
        if sessions[session_id]['last_landmarks'] is not None:
            last_lm = np.array(sessions[session_id]['last_landmarks'])
            if np.allclose(filtered_keypoints[:63], last_lm[:63], atol=0.001):
                sessions[session_id]['repeated_landmarks_count'] += 1
                if sessions[session_id]['repeated_landmarks_count'] > 10:
                    # Skip frozen frame
                    response['status'] = 'frozen_frame'
                    emit('frame_processed', response)
                    return
            else:
                sessions[session_id]['repeated_landmarks_count'] = 0
        
        sessions[session_id]['last_landmarks'] = filtered_keypoints.tolist()
        
        # Add frame to predictor
        session_predictor.add_frame(filtered_keypoints)
        
        sessions[session_id]['last_activity'] = time.time()
        sessions[session_id]['frames_processed'] += 1
        
        # Make prediction
        prediction_result = session_predictor.predict_current(return_all_predictions=True)
        
        if prediction_result and prediction_result.get('ready'):
            # Update performance stats
            prediction_time = time.time() - start_time
            prediction_time_ms = prediction_time * 1000
            
            performance_stats['total_predictions'] += 1
            performance_stats['successful_predictions'] += 1
            performance_stats['avg_prediction_time'] = (
                performance_stats['avg_prediction_time'] * 0.9 + prediction_time * 0.1
            )
            
            # Record in performance monitor
            if global_performance_monitor:
                global_performance_monitor.record_inference_time(prediction_time_ms)
            
            response['status'] = 'success'
            response['buffer_size'] = prediction_result['buffer_size']
            response['prediction_time_ms'] = prediction_result['prediction_time_ms']
            
            # Send prediction data
            # Only send stable predictions to avoid flickering
            if prediction_result.get('is_stable'):
                label = prediction_result['stable_prediction']
                confidence = prediction_result['stable_confidence']
                is_new = prediction_result['is_new']
                
                # ALWAYS include prediction in response (even if not "new")
                # Frontend will handle duplicate filtering
                response['prediction'] = {
                    'label': label,
                    'confidence': confidence,
                    'is_new': is_new,
                    'is_stable': True
                }
                
                # Record prediction in performance monitor
                if global_performance_monitor:
                    global_performance_monitor.record_prediction(label)
                
                # Update recording manager
                if recording_manager:
                    segment = recording_manager.on_prediction(label, confidence, current_timestamp)
                    if segment:
                        # Recording segment completed
                        response['recording_segment'] = segment
                        response['recording_status'] = 'stopped'
                        logger.info(f"Recording segment completed: {segment['label']} ({segment['duration']:.1f}s)")
                    else:
                        # Recording in progress
                        response['recording_status'] = recording_manager.get_recording_status()
                
                # Track new predictions (log only)
                if is_new:
                    sessions[session_id]['predictions_sent'] += 1
                    logger.info(
                        f"Session {session_id}: NEW prediction: "
                        f"{label} ({confidence:.3f})"
                    )
                else:
                    # Log continuing prediction
                    logger.debug(
                        f"Session {session_id}: Continuing prediction: "
                        f"{label} ({confidence:.3f})"
                    )
            else:
                # Not stable yet - send status but no prediction
                response['prediction'] = None
                response['status'] = 'collecting_stability'
                
                # Update recording manager with low confidence
                if recording_manager:
                    segment = recording_manager.on_no_hands(current_timestamp)
                    if segment:
                        response['recording_segment'] = segment
                        response['recording_status'] = 'stopped'
            
            # Optionally include all predictions (for debugging/visualization)
            if data.get('include_all_predictions', False):
                response['all_predictions'] = prediction_result.get('all_predictions', [])
        
        elif prediction_result is None and should_predict is False:
            # State machine is controlling flow - use its status
            response['status'] = response.get('state', 'collecting_frames')
        else:
            # Still collecting frames
            response['status'] = 'collecting_frames'
            response['buffer_size'] = prediction_result.get('buffer_size', 0)
            response['min_required'] = prediction_result.get('min_required', 32)
            response['prediction'] = None
        
        # Add performance metrics to response (emit every frame)
        if global_performance_monitor:
            metrics = global_performance_monitor.get_metrics_summary()
            response['performance'] = {
                'fps': metrics['fps'],
                'avg_inference_ms': metrics['avg_inference_ms'],
                'landmark_detection_rate': metrics['landmark_detection_rate']
            }
            
            # Auto-save metrics periodically
            if global_performance_monitor.should_auto_save():
                global_performance_monitor.save_metrics()
        
        emit('frame_processed', response)
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        performance_stats['failed_predictions'] += 1
        emit('frame_processed', {
            'session_id': session_id,
            'status': 'error',
            'error': str(e)
        })


# REST API Endpoints
@app.route('/')
def index():
    """Serve the frontend application."""
    frontend_path = Path(__file__).resolve().parent.parent / "frontend"
    return send_from_directory(frontend_path, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from frontend directory."""
    frontend_path = Path(__file__).resolve().parent.parent / "frontend"
    return send_from_directory(frontend_path, path)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'system_ready': predictor is not None and mp_processor is not None,
        'model_loaded': predictor is not None,
        'mediapipe_loaded': mp_processor is not None,
        'version': 'v2',
        'timestamp': time.time()
    })


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if predictor is None:
        return jsonify({'error': 'No model loaded'}), 404
    
    return jsonify({
        'model_info': predictor.get_model_info(),
        'version': 'v2'
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system performance statistics."""
    return jsonify({
        'performance': performance_stats,
        'sessions': {
            'active': len(sessions),
            'total': performance_stats['total_connections']
        },
        'version': 'v2',
        'timestamp': time.time()
    })


@app.route('/api/models/list', methods=['GET'])
def list_models():
    """List available models."""
    try:
        models = model_manager.list_models()
        best_model = model_manager.get_best_model()
        
        return jsonify({
            'models': models,
            'best_model': best_model,
            'count': len(models)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback/summary', methods=['GET'])
def feedback_summary():
    """Get feedback summary statistics."""
    if feedback_db is None:
        return jsonify({'error': 'Feedback system not initialized'}), 503
    
    try:
        summary = feedback_db.get_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback/statistics', methods=['GET'])
def feedback_statistics():
    """Get per-sign feedback statistics."""
    if feedback_db is None:
        return jsonify({'error': 'Feedback system not initialized'}), 503
    
    try:
        label = request.args.get('label')
        stats = feedback_db.get_statistics(label=label)
        return jsonify({'statistics': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback/confused-signs', methods=['GET'])
def feedback_confused_signs():
    """Get signs that are frequently confused."""
    if feedback_db is None:
        return jsonify({'error': 'Feedback system not initialized'}), 503
    
    try:
        min_errors = int(request.args.get('min_errors', 5))
        confused = feedback_db.get_confused_signs(min_errors=min_errors)
        
        result = [
            {'sign': sign, 'errors': errors, 'error_rate': rate}
            for sign, errors, rate in confused
        ]
        
        return jsonify({'confused_signs': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback/history', methods=['GET'])
def feedback_history():
    """Get feedback history."""
    if feedback_db is None:
        return jsonify({'error': 'Feedback system not initialized'}), 503
    
    try:
        limit = int(request.args.get('limit', 100))
        label = request.args.get('label')
        is_correct = request.args.get('is_correct')
        
        # Convert is_correct string to boolean
        if is_correct is not None:
            is_correct = is_correct.lower() in ('true', '1', 'yes')
        
        history = feedback_db.get_feedback_history(
            limit=limit,
            label=label,
            is_correct=is_correct
        )
        
        return jsonify({'history': history, 'count': len(history)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# PORT MANAGEMENT
# ==============================================================================
def check_and_free_port(port=5000):
    """Check if port is in use and free it if possible."""
    import socket
    import subprocess
    
    # Check if port is in use
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result == 0:
        # Port is in use, try to find and kill the process
        logger.warning(f"Port {port} is already in use. Attempting to free it...")
        try:
            # Find process using the port
            cmd = f'netstat -ano | findstr :{port} | findstr LISTENING'
            output = subprocess.check_output(cmd, shell=True, text=True)
            
            # Extract PID (last column)
            lines = output.strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) > 4:
                    pid = parts[-1]
                    logger.info(f"Killing process {pid} using port {port}")
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, check=False)
                    time.sleep(2)  # Wait for port to be freed
            
            logger.info(f"Port {port} freed successfully")
        except Exception as e:
            logger.error(f"Failed to free port {port}: {e}")
            logger.error(f"Please manually kill the process using: taskkill /F /PID <pid>")
            raise SystemExit(1)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # Check and free port if needed
    check_and_free_port(5000)
    
    # Initialize system
    initialize_system()
    
    # Print startup info
    print("\n" + "=" * 70)
    print("PSL RECOGNITION SYSTEM V2 - STARTING")
    print("=" * 70)
    print(f"[OK] Version: V2 (Enhanced)")
    print(f"[OK] Host: {flask_config.HOST}:{flask_config.PORT}")
    print(f"[OK] Model loaded: {predictor is not None}")
    print(f"[OK] MediaPipe loaded: {mp_processor is not None}")
    
    if predictor is not None:
        info = predictor.get_model_info()
        print(f"[OK] Model: {info['model_type']}")
        print(f"[OK] Classes: {info['num_classes']}")
        print(f"[OK] Sliding window: {info['sliding_window_size']} frames")
        print(f"[OK] Stability: {info['stability_threshold']}")
        print(f"[OK] Device: {info['device']}")
        print(f"[OK] Labels: {', '.join(info['labels'])}")
    
    print("\n" + "=" * 70)
    print("IMPROVEMENTS IN V2:")
    print("-" * 70)
    print("  • Faster response: 32-frame buffer (was 60)")
    print("  • No lagging: predictions start immediately after 32 frames")
    print("  • No flickering: improved stability filtering (3/5 votes)")
    print("  • No old predictions: auto-clear buffer when hands disappear")
    print("  • Synchronized keypoints: sent with predictions")
    print("=" * 70)
    print("\n[*] Server starting...\n")
    
    # Run server
    socketio.run(
        app,
        host=flask_config.HOST,
        port=flask_config.PORT,
        debug=flask_config.DEBUG,
        allow_unsafe_werkzeug=True  # For development only
    )

