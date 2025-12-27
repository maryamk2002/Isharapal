/**
 * Main Application Logic V2 for PSL Recognition System
 * Enhanced with feedback system, better UI updates, and optimized performance
 */

class PSLRecognitionApp {
    constructor() {
        this.isInitialized = false;
        this.isRecognitionActive = false;
        this.currentLanguage = 'urdu';
        this.settings = {
            sensitivity: 0.5,
            frameRate: 15,
            language: 'urdu'
        };
        this.frameIntervalMs = 1000 / this.settings.frameRate;
        this.lastFrameSentAt = 0;
        this.lastDisplayedPrediction = null;
        this.lastDisplayedConfidence = 0;
        
        // Components
        this.camera = null;
        this.websocket = null;
        this.visualizer = null;
        this.ui = null;
        this.recordingStatus = null;  // NEW: Recording status UI
        
        // DOM Elements
        this.elements = {};
        
        // Bind methods
        this.init = this.init.bind(this);
        this.startRecognition = this.startRecognition.bind(this);
        this.stopRecognition = this.stopRecognition.bind(this);
        this.resetSystem = this.resetSystem.bind(this);
        this.handleFeedback = this.handleFeedback.bind(this);
    }
    
    async init() {
        try {
            console.log('Initializing PSL Recognition System V2...');
            
            // Show loading overlay
            this.showLoadingOverlay();
            
            // Get DOM elements
            this.initDOMElements();
            
            // Initialize UI components
            this.ui = new UIManager();
            await this.ui.init();
            
            // Initialize camera
            this.camera = new CameraManager();
            
            // V2: No MediaPipe loading in frontend
            // this.camera.on('mediapipe_loading', (data) => {
            //     const progressEl = document.getElementById('loadingProgress');
            //     if (progressEl) {
            //         progressEl.textContent = data.message || 'Loading...';
            //     }
            // });
            
            this.updateLoadingMessage('Initializing camera...');
            const cameraReady = await this.camera.init();
            if (!cameraReady) {
                throw new Error('Camera initialization failed');
            }
            
            // Initialize WebSocket connection
            this.updateLoadingMessage('Connecting to backend...');
            this.websocket = new WebSocketManager();
            this.setupWebSocketCallbacks();
            await this.websocket.init();
            
            // Wait for connection
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Update UI with connection status
            if (this.websocket && this.websocket.connected) {
                console.log('✓ WebSocket V2 connected');
                this.ui.setConnectionStatus(true);
            }
            
            // Initialize visualizer
            this.updateLoadingMessage('Setting up visualization...');
            this.visualizer = new HandVisualizer();
            await this.visualizer.init();
            
            // Initialize recording status UI
            this.recordingStatus = new RecordingStatusUI();
            await this.recordingStatus.init();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Hide loading overlay
            this.hideLoadingOverlay();
            
            this.isInitialized = true;
            console.log('✓ PSL Recognition System V2 initialized successfully');
            
            // Show welcome notification
            this.showNotification('System ready! Click "Start" to begin.', 'success');
            
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showNotification(`Initialization failed: ${error.message}`, 'error');
            this.hideLoadingOverlay();
        }
    }
    
    initDOMElements() {
        this.elements = {
            // Buttons
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            resetBtn: document.getElementById('resetBtn'),
            feedbackCorrect: document.getElementById('feedbackCorrect'),
            feedbackIncorrect: document.getElementById('feedbackIncorrect'),
            
            // Display areas
            predictionDisplay: document.getElementById('predictionDisplay'),
            confidenceFill: document.getElementById('confidenceFill'),
            confidenceValue: document.getElementById('confidenceValue'),
            feedbackSection: document.getElementById('feedbackSection'),
            historyList: document.getElementById('historyList'),
            
            // Status displays
            systemStatusValue: document.getElementById('systemStatusValue'),
            signCountValue: document.getElementById('signCountValue'),
            fpsValue: document.getElementById('fpsValue'),
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            
            // Overlays
            loadingOverlay: document.getElementById('loadingOverlay'),
            notificationContainer: document.getElementById('notificationContainer')
        };
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ignore if user is typing in an input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }
            
            switch(e.code) {
                case 'Space':
                case 'Enter':
                    // Start recognition
                    if (!this.isRecognitionActive && this.isConnected) {
                        e.preventDefault();
                        this.handleStart();
                    }
                    break;
                    
                case 'Escape':
                    // Stop recognition
                    if (this.isRecognitionActive) {
                        e.preventDefault();
                        this.handleStop();
                    }
                    break;
                    
                case 'KeyR':
                    // Reset (Ctrl+R or just R)
                    if (!this.isRecognitionActive) {
                        e.preventDefault();
                        this.handleReset();
                    }
                    break;
                    
                case 'KeyC':
                    // Mark as Correct
                    if (this.lastDisplayedPrediction && this.elements.feedbackSection.style.display !== 'none') {
                        e.preventDefault();
                        this.handleFeedback(true);
                    }
                    break;
                    
                case 'KeyI':
                    // Mark as Incorrect
                    if (this.lastDisplayedPrediction && this.elements.feedbackSection.style.display !== 'none') {
                        e.preventDefault();
                        this.handleFeedback(false);
                    }
                    break;
                    
                case 'KeyH':
                    // Toggle history visibility (if implemented)
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.toggleHistory();
                    }
                    break;
            }
        });
        
        console.log('⌨️ Keyboard shortcuts enabled:');
        console.log('  Space/Enter → Start');
        console.log('  Esc → Stop');
        console.log('  R → Reset');
        console.log('  C → Mark Correct');
        console.log('  I → Mark Incorrect');
    }
    
    toggleHistory() {
        // Toggle history card visibility (optional feature)
        const historyCard = document.querySelector('.history-card');
        if (historyCard) {
            historyCard.style.display = historyCard.style.display === 'none' ? 'block' : 'none';
        }
    }
    
    setupWebSocketCallbacks() {
        // Connection status
        this.websocket.onConnectionChange = (connected) => {
            console.log(`Connection status changed: ${connected}`);
            this.ui.setConnectionStatus(connected);
            this.updateSystemStatus(connected ? 'Connected' : 'Disconnected');
        };
        
        // Prediction received - V2 ENHANCED
        this.websocket.onPredictionReceived = (prediction) => {
            this.handlePredictionReceived(prediction);
        };
        
        // Frame processed - V2 ENHANCED
        this.websocket.onFrameProcessed = (data) => {
            this.handleFrameProcessed(data);
        };
        
        // Error handling
        this.websocket.onError = (error) => {
            console.error('WebSocket error:', error);
            this.showNotification(error.message || 'An error occurred', 'error');
        };
    }
    
    setupEventListeners() {
        // Start button
        if (this.elements.startBtn) {
            this.elements.startBtn.addEventListener('click', this.startRecognition);
        }
        
        // Stop button
        if (this.elements.stopBtn) {
            this.elements.stopBtn.addEventListener('click', this.stopRecognition);
        }
        
        // Reset button
        if (this.elements.resetBtn) {
            this.elements.resetBtn.addEventListener('click', this.resetSystem);
        }
        
        // Feedback buttons
        if (this.elements.feedbackCorrect) {
            this.elements.feedbackCorrect.addEventListener('click', () => {
                this.handleFeedback(true);
            });
        }
        
        if (this.elements.feedbackIncorrect) {
            this.elements.feedbackIncorrect.addEventListener('click', () => {
                this.handleFeedback(false);
            });
        }
        
        // Keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Window resize
        window.addEventListener('resize', () => {
            if (this.visualizer) {
                this.visualizer.handleResize();
            }
        });
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ignore if user is typing in an input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }
            
            switch(e.code) {
                case 'Space':
                case 'Enter':
                    // Start recognition
                    if (!this.isRecognitionActive && this.websocket && this.websocket.connected) {
                        e.preventDefault();
                        this.startRecognition();
                    }
                    break;
                    
                case 'Escape':
                    // Stop recognition
                    if (this.isRecognitionActive) {
                        e.preventDefault();
                        this.stopRecognition();
                    }
                    break;
                    
                case 'KeyR':
                    // Reset (just R key)
                    if (!this.isRecognitionActive) {
                        e.preventDefault();
                        this.resetSystem();
                    }
                    break;
                    
                case 'KeyC':
                    // Mark as Correct
                    if (this.lastDisplayedPrediction && this.elements.feedbackSection && 
                        this.elements.feedbackSection.style.display !== 'none') {
                        e.preventDefault();
                        this.handleFeedback(true);
                    }
                    break;
                    
                case 'KeyI':
                    // Mark as Incorrect
                    if (this.lastDisplayedPrediction && this.elements.feedbackSection && 
                        this.elements.feedbackSection.style.display !== 'none') {
                        e.preventDefault();
                        this.handleFeedback(false);
                    }
                    break;
            }
        });
        
        console.log('⌨️ Keyboard shortcuts enabled:');
        console.log('  Space/Enter → Start Recognition');
        console.log('  Esc → Stop Recognition');
        console.log('  R → Reset System');
        console.log('  C → Mark Correct');
        console.log('  I → Mark Incorrect');
    }
    
    async startRecognition() {
        if (!this.isInitialized) {
            this.showNotification('System not initialized', 'error');
            return;
        }
        
        if (this.isRecognitionActive) {
            this.showNotification('Recognition already active', 'warning');
            return;
        }
        
        try {
            console.log('Starting recognition V2...');
            
            // Start WebSocket recognition
            if (!this.websocket.startRecognition()) {
                throw new Error('Failed to start recognition');
            }
            
            // Start camera (which automatically starts frame capture)
            const cameraStarted = await this.camera.start();
            if (!cameraStarted) {
                throw new Error('Failed to start camera');
            }
            
            // Listen to camera frames
            this.camera.on('frame', async (frame) => {
                await this.captureFrame(frame);
            });
            
            // Update UI
            this.isRecognitionActive = true;
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.updateSystemStatus('Recognizing...');
            
            // Clear previous prediction
            this.clearPredictionDisplay();
            
            console.log('✓ Recognition started');
            this.showNotification('Recognition started', 'success');
            
        } catch (error) {
            console.error('Failed to start recognition:', error);
            this.showNotification(`Failed to start: ${error.message}`, 'error');
        }
    }
    
    stopRecognition() {
        if (!this.isRecognitionActive) {
            return;
        }
        
        try {
            console.log('Stopping recognition...');
            
            // Stop camera
            this.camera.stop();
            
            // Remove frame listener
            this.camera.off('frame');
            
            // Stop WebSocket recognition
            this.websocket.stopRecognition();
            
            // Update UI
            this.isRecognitionActive = false;
            this.elements.startBtn.disabled = false;
            this.elements.stopBtn.disabled = true;
            this.updateSystemStatus('Stopped');
            
            // Hide feedback buttons
            this.hideFeedbackSection();
            
            console.log('✓ Recognition stopped');
            this.showNotification('Recognition stopped', 'info');
            
        } catch (error) {
            console.error('Failed to stop recognition:', error);
            this.showNotification(`Failed to stop: ${error.message}`, 'error');
        }
    }
    
    resetSystem() {
        try {
            console.log('Resetting system...');
            
            // Reset WebSocket
            if (this.websocket) {
                this.websocket.resetRecognition();
            }
            
            // Clear visualizer
            if (this.visualizer) {
                this.visualizer.clear();
            }
            
            // Clear UI
            this.clearPredictionDisplay();
            this.clearHistory();
            this.hideFeedbackSection();
            this.updateBufferStatus(0, 32);
            this.updateSystemStatus('Ready');
            
            console.log('✓ System reset');
            this.showNotification('System reset', 'info');
            
        } catch (error) {
            console.error('Failed to reset:', error);
            this.showNotification(`Failed to reset: ${error.message}`, 'error');
        }
    }
    
    async captureFrame(frame) {
        const now = Date.now();
        
        // Throttle frame sending
        if (now - this.lastFrameSentAt < this.frameIntervalMs) {
            return;
        }
        
        try {
            // Ensure video is ready
            if (!frame || frame.readyState !== 4) {
                return;
            }
            
            // Convert frame to base64
            const canvas = document.createElement('canvas');
            const width = frame.videoWidth || frame.width || 640;
            const height = frame.videoHeight || frame.height || 480;
            
            if (width === 0 || height === 0) {
                console.error('Invalid video dimensions:', width, height);
                return;
            }
            
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(frame, 0, 0, width, height);
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Validate frame data
            if (!frameData || frameData === 'data:,' || frameData.length < 100) {
                console.error('Invalid frame data generated:', frameData.substring(0, 50));
                return;
            }
            
            // Send to WebSocket
            this.websocket.sendFrame(frameData);
            this.lastFrameSentAt = now;
            
        } catch (error) {
            console.error('Error capturing frame:', error);
        }
    }
    
    handleFrameProcessed(data) {
        // Update buffer status
        if (data.buffer_size !== undefined) {
            const minRequired = data.min_required || 32;
            this.updateBufferStatus(data.buffer_size, minRequired);
        }
        
        // Update FPS
        const wsStatus = this.websocket.getConnectionStatus();
        this.updateFPS(wsStatus.currentFPS);
        
        // Update performance metrics (if available)
        if (data.performance) {
            this.updatePerformanceMetrics(data.performance);
        }
        
        // Update keypoints visualization
        if (data.keypoints) {
            this.visualizer.updateKeypoints(data.keypoints);
        } else {
            // No hands detected - clear keypoints
            this.visualizer.updateKeypoints(null);
        }
        
        // Handle recording status updates
        if (data.recording_status && this.recordingStatus) {
            this.recordingStatus.setStatus(data.recording_status);
        }
        
        // Handle completed recording segment
        if (data.recording_segment && this.recordingStatus) {
            const segment = data.recording_segment;
            this.recordingStatus.showSegment(
                segment.label,
                segment.confidence,
                segment.duration,
                segment.timestamp
            );
        }
        
        // Update system status based on frame processing status
        if (data.status === 'no_hands') {
            this.updateSystemStatus('No hands detected');
        } else if (data.status === 'collecting_frames') {
            this.updateSystemStatus('Collecting frames...');
        } else if (data.status === 'collecting_stability') {
            this.updateSystemStatus('Stabilizing...');
        } else if (data.status === 'stuck_sequence_reset') {
            this.updateSystemStatus('Stuck sequence reset');
            this.showNotification(data.message || 'Please move your hand', 'warning');
        } else if (data.status === 'success') {
            this.updateSystemStatus('Recognizing...');
        }
    }
    
    updatePerformanceMetrics(metrics) {
        /**
         * Update performance metrics display (FPS, inference time).
         * 
         * Args:
         *   metrics: Object with {fps, avg_inference_ms, landmark_detection_rate}
         */
        // Update FPS display (already handled by websocket status)
        if (metrics.fps !== undefined && this.elements.fpsValue) {
            this.elements.fpsValue.textContent = metrics.fps.toFixed(1);
        }
        
        // Update inference time (if there's a display for it)
        if (metrics.avg_inference_ms !== undefined) {
            const inferenceEl = document.getElementById('inferenceTime');
            if (inferenceEl) {
                inferenceEl.textContent = `${metrics.avg_inference_ms.toFixed(0)}ms`;
            }
        }
        
        // Update landmark detection rate (if there's a display for it)
        if (metrics.landmark_detection_rate !== undefined) {
            const detectionEl = document.getElementById('detectionRate');
            if (detectionEl) {
                detectionEl.textContent = `${(metrics.landmark_detection_rate * 100).toFixed(0)}%`;
            }
        }
    }
    
    handlePredictionReceived(prediction) {
        console.log('Prediction received:', prediction);
        
        // Update prediction display
        this.displayPrediction(prediction.label, prediction.confidence);
        
        // Add to history
        this.addToHistory(prediction);
        
        // Show feedback section
        this.showFeedbackSection();
        
        // Update visualizer
        if (this.visualizer) {
            this.visualizer.showPrediction(prediction.label, prediction.confidence);
        }
    }
    
    displayPrediction(label, confidence) {
        // Use UI manager's updatePrediction method
        if (this.ui) {
            this.ui.updatePrediction(label, confidence);
        }
        
        this.lastDisplayedPrediction = label;
        this.lastDisplayedConfidence = confidence;
    }
    
    clearPredictionDisplay() {
        // Use UI manager's clearPrediction method
        if (this.ui) {
            this.ui.clearPrediction();
        }
        
        this.lastDisplayedPrediction = null;
        this.lastDisplayedConfidence = 0;
    }
    
    showFeedbackSection() {
        if (this.elements.feedbackSection) {
            this.elements.feedbackSection.style.display = 'block';
        }
    }
    
    hideFeedbackSection() {
        if (this.elements.feedbackSection) {
            this.elements.feedbackSection.style.display = 'none';
        }
    }
    
    handleFeedback(isCorrect) {
        if (!this.lastDisplayedPrediction) {
            this.showNotification('No prediction to provide feedback for', 'warning');
            return;
        }
        
        console.log(`Feedback: ${this.lastDisplayedPrediction} = ${isCorrect ? 'Correct' : 'Incorrect'}`);
        
        // Send feedback to backend
        this.websocket.sendFeedback(this.lastDisplayedPrediction, isCorrect, {
            confidence: this.lastDisplayedConfidence,
            timestamp: Date.now()
        });
        
        // Show confirmation
        const message = isCorrect 
            ? `✓ Thank you! "${this.lastDisplayedPrediction}" marked as correct`
            : `✗ Thank you! "${this.lastDisplayedPrediction}" marked as incorrect`;
        this.showNotification(message, isCorrect ? 'success' : 'info');
        
        // Hide feedback buttons temporarily
        setTimeout(() => {
            this.hideFeedbackSection();
        }, 1000);
    }
    
    addToHistory(prediction) {
        if (!this.elements.historyList) return;
        
        // Remove "empty" message if present
        const emptyMessage = this.elements.historyList.querySelector('.history-empty');
        if (emptyMessage) {
            emptyMessage.remove();
        }
        
        // Create history item
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit'
        });
        
        historyItem.innerHTML = `
            <div>
                <div class="history-item-label">${prediction.label}</div>
                <div class="history-item-time">${timeString}</div>
            </div>
            <div class="history-item-confidence">${Math.round(prediction.confidence * 100)}%</div>
        `;
        
        // Add to top of list
        this.elements.historyList.insertBefore(historyItem, this.elements.historyList.firstChild);
        
        // Keep only last 10 items
        const items = this.elements.historyList.querySelectorAll('.history-item');
        if (items.length > 10) {
            items[items.length - 1].remove();
        }
    }
    
    clearHistory() {
        if (!this.elements.historyList) return;
        
        this.elements.historyList.innerHTML = `
            <div class="history-empty">
                <span class="urdu">ابھی تک کوئی نہیں</span>
                <span class="english">No signs yet</span>
            </div>
        `;
    }
    
    updateSystemStatus(status) {
        if (this.elements.systemStatus) {
            this.elements.systemStatus.textContent = status;
        }
    }
    
    updateBufferStatus(current, target) {
        if (this.elements.bufferStatus) {
            this.elements.bufferStatus.textContent = `${current}/${target}`;
        }
    }
    
    updateFPS(fps) {
        if (this.elements.fpsValue) {
            this.elements.fpsValue.textContent = Math.round(fps);
        }
    }
    
    showNotification(message, type = 'info') {
        if (!this.elements.notificationContainer) return;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        this.elements.notificationContainer.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 5000);
    }
    
    showLoadingOverlay() {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.style.display = 'flex';
        }
    }
    
    hideLoadingOverlay() {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.style.display = 'none';
        }
    }
    
    updateLoadingMessage(message) {
        const progressEl = document.getElementById('loadingProgress');
        if (progressEl) {
            progressEl.textContent = message;
        }
    }
    
    destroy() {
        // Stop recognition
        if (this.isRecognitionActive) {
            this.stopRecognition();
        }
        
        // Clean up components
        if (this.camera) {
            this.camera.destroy();
        }
        
        if (this.websocket) {
            this.websocket.destroy();
        }
        
        if (this.visualizer) {
            this.visualizer.destroy();
        }
        
        if (this.ui) {
            this.ui.destroy();
        }
        
        console.log('✓ App destroyed');
    }
    
    // Camera error handling
    handleCameraError(error) {
        console.error('Camera error:', error);
        let message = 'Camera error occurred.';
        
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            message = 'Camera access was denied. Please allow camera permissions in your browser settings.';
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            message = 'No camera device found. Please connect a camera and try again.';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            message = 'Camera is already in use by another application. Please close other apps using the camera.';
        } else if (error.name === 'OverconstrainedError') {
            message = 'Camera does not meet the required specifications.';
        } else if (error.name === 'SecurityError') {
            message = 'Camera access is blocked due to security settings.';
        }
        
        this.showCameraError(message);
    }
    
    showCameraError(message) {
        const modal = document.getElementById('cameraErrorModal');
        const messageEl = document.getElementById('cameraErrorMessage');
        const retryBtn = document.getElementById('retryCameraBtn');
        const closeBtn = document.getElementById('closeCameraErrorBtn');
        
        if (modal && messageEl) {
            messageEl.textContent = message;
            modal.style.display = 'flex';
            
            // Retry button
            if (retryBtn) {
                retryBtn.onclick = () => {
                    modal.style.display = 'none';
                    this.startRecognition();
                };
            }
            
            // Close button
            if (closeBtn) {
                closeBtn.onclick = () => {
                    modal.style.display = 'none';
                };
            }
        }
    }
    
    // Reconnection handling
    showReconnectingBanner(attempt = 0, maxAttempts = 3) {
        const banner = document.getElementById('reconnectingBanner');
        const attemptEl = document.getElementById('reconnectAttempt');
        const retryBtn = document.getElementById('manualReconnectBtn');
        
        if (banner) {
            banner.style.display = 'block';
            
            if (attemptEl && attempt > 0) {
                attemptEl.textContent = `(${attempt}/${maxAttempts})`;
            }
            
            if (retryBtn) {
                retryBtn.onclick = () => {
                    this.websocket.reconnect();
                };
            }
        }
    }
    
    hideReconnectingBanner() {
        const banner = document.getElementById('reconnectingBanner');
        if (banner) {
            banner.style.display = 'none';
        }
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PSLRecognitionApp;
}

