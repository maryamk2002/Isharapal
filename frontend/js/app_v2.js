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
        
        // Bound frame handler (to allow proper removal)
        this.boundFrameHandler = null;
        
        // Inactivity tracking
        this.inactivityTimeoutMs = 10 * 60 * 1000;  // 10 minutes in milliseconds
        this.inactivityCheckIntervalMs = 30 * 1000;  // Check every 30 seconds
        this.lastActivityTime = Date.now();
        this.inactivityCheckInterval = null;
        
        // Bind methods
        this.init = this.init.bind(this);
        this.startRecognition = this.startRecognition.bind(this);
        this.stopRecognition = this.stopRecognition.bind(this);
        this.resetSystem = this.resetSystem.bind(this);
        this.handleFeedback = this.handleFeedback.bind(this);
        this.recordActivity = this.recordActivity.bind(this);
        this.checkInactivity = this.checkInactivity.bind(this);
    }
    
    async init() {
        try {
            console.log('Initializing PSL Recognition System V2...');
            
            // Show loading overlay
            this.showLoadingOverlay();
            
            // Get DOM elements
            this.initDOMElements();
            
            // Hide feedback section initially (shown when prediction is made)
            this.hideFeedbackSection();
            
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
    
    setupWebSocketCallbacks() {
        // Connection status
        this.websocket.onConnectionChange = (connected) => {
            console.log(`Connection status changed: ${connected}`);
            this.ui.setConnectionStatus(connected);
            
            if (connected) {
                this.updateSystemStatus('Connected');
                
                // If recognition was active before disconnect, restart on backend
                if (this.isRecognitionActive) {
                    console.log('[App] Reconnected while recognition was active - restarting recognition on backend');
                    this.showNotification('Reconnected! Restarting recognition...', 'success');
                    
                    // Re-emit start_recognition to backend since session may have been lost
                    setTimeout(() => {
                        if (this.websocket && this.websocket.connected) {
                            this.websocket.socket.emit('start_recognition', {
                                sensitivity: this.settings.sensitivity
                            });
                            console.log('[App] Recognition restarted on backend after reconnection');
                            
                            // Re-enable buttons
                            if (this.elements.startBtn) {
                                this.elements.startBtn.disabled = true;
                            }
                            if (this.elements.stopBtn) {
                                this.elements.stopBtn.disabled = false;
                            }
                        }
                    }, 500);  // Small delay to ensure connection is stable
                }
            } else {
                this.updateSystemStatus('Disconnected - Reconnecting...');
                this.showNotification('Connection lost. Reconnecting...', 'warning');
                
                // DON'T stop recognition - let it resume on reconnect
                // Just update UI to show we're waiting
                if (this.elements.startBtn) {
                    this.elements.startBtn.disabled = true;
                }
                if (this.elements.stopBtn) {
                    this.elements.stopBtn.disabled = true;
                }
            }
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
    
    toggleHistory() {
        // Toggle history card visibility (optional feature)
        const historyCard = document.querySelector('.history-card');
        if (historyCard) {
            historyCard.style.display = historyCard.style.display === 'none' ? 'block' : 'none';
        }
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
            
            // RESET frontend state for fresh predictions
            this.lastDisplayedPrediction = null;
            this.lastDisplayedConfidence = 0;
            this.lastFrameSentAt = 0;  // CRITICAL: Reset throttle timer
            
            // Start WebSocket recognition
            if (!this.websocket.startRecognition()) {
                throw new Error('Failed to start recognition');
            }
            
            // Start camera (which automatically starts frame capture)
            const cameraStarted = await this.camera.start();
            if (!cameraStarted) {
                throw new Error('Failed to start camera');
            }
            
            // IMPORTANT: Set flag BEFORE adding frame listener
            this.isRecognitionActive = true;
            
            // IMPORTANT: Remove any existing landmark listener first to prevent accumulation
            if (this.boundFrameHandler) {
                this.camera.off('landmarks', this.boundFrameHandler);
            }
            
            // V3: Listen to landmarks from frontend MediaPipe (not raw frames)
            this.boundFrameHandler = async (data) => {
                await this.sendLandmarks(data);
            };
            
            // Listen to camera landmarks (processed by frontend MediaPipe)
            this.camera.on('landmarks', this.boundFrameHandler);
            
            // Start a debug heartbeat to detect if main loop is alive
            this.startDebugHeartbeat();
            
            // Start inactivity watchdog
            this.startInactivityWatchdog();
            
            // Hide any inactivity overlay
            this.hideInactivityOverlay();
            
            // Update UI
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
            
            // Remove landmarks listener properly
            if (this.boundFrameHandler) {
                this.camera.off('landmarks', this.boundFrameHandler);
                this.boundFrameHandler = null;
            }
            
            // Stop WebSocket recognition
            this.websocket.stopRecognition();
            
            // Stop debug heartbeat
            this.stopDebugHeartbeat();
            
            // Stop inactivity watchdog
            this.stopInactivityWatchdog();
            
            // RESET frontend state so next start is fresh
            this.lastDisplayedPrediction = null;
            this.lastDisplayedConfidence = 0;
            this.lastFrameSentAt = 0;  // Reset throttle timer for next start
            
            // Clear visualizer keypoints (remove stuck skeleton)
            if (this.visualizer) {
                this.visualizer.updateKeypoints(null);
                this.visualizer.clear();
            }
            
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
            
            // RESET frontend state
            this.lastDisplayedPrediction = null;
            this.lastDisplayedConfidence = 0;
            this.lastFrameSentAt = 0;  // Reset throttle timer
            
            // Reset WebSocket
            if (this.websocket) {
                this.websocket.resetRecognition();
            }
            
            // Clear visualizer (remove stuck keypoints)
            if (this.visualizer) {
                this.visualizer.updateKeypoints(null);
                this.visualizer.clear();
            }
            
            // Clear UI
            this.clearPredictionDisplay();
            this.clearHistory();
            this.hideFeedbackSection();
            this.updateBufferStatus(0, 45);
            this.updateSystemStatus('Ready');
            
            console.log('✓ System reset');
            this.showNotification('System reset', 'info');
            
        } catch (error) {
            console.error('Failed to reset:', error);
            this.showNotification(`Failed to reset: ${error.message}`, 'error');
        }
    }
    
    /**
     * V3: Send landmarks extracted by frontend MediaPipe.
     * Payload is ~2KB instead of ~150KB (98% reduction!)
     */
    async sendLandmarks(data) {
        const now = Date.now();
        
        // Throttle landmark sending (same as frame rate)
        if (now - this.lastFrameSentAt < this.frameIntervalMs) {
            return;
        }
        
        // Check if recognition is active
        if (!this.isRecognitionActive) {
            // Debug: log when recognition is not active
            console.warn('[App] Recognition not active, skipping landmark send');
            return;
        }
        
        // Debug: Check websocket connection state
        if (!this.websocket || !this.websocket.connected) {
            console.warn('[App] WebSocket not connected, skipping landmark send');
            return;
        }
        
        try {
            // Send landmarks to backend via WebSocket
            const sent = this.websocket.sendLandmarks(data);
            if (!sent) {
                console.warn('[App] sendLandmarks returned false');
            }
            this.lastFrameSentAt = now;
            
            // Update hand status in UI based on frontend detection
            if (this.ui) {
                this.ui.updateHandStatus(data.hasHands);
            }
            
            // Record activity when hands are detected (for inactivity tracking)
            if (data.hasHands) {
                this.recordActivity();
            }
            
            // Update visualizer with keypoints for skeleton drawing
            if (this.visualizer && data.landmarks) {
                // Send first 126 features for visualization (excluding padding)
                this.visualizer.updateKeypoints(data.landmarks.slice(0, 126));
            } else if (this.visualizer) {
                this.visualizer.updateKeypoints(null);
            }
            
        } catch (error) {
            console.error('Error sending landmarks:', error);
        }
    }
    
    // Legacy method kept for backwards compatibility
    async captureFrame(frame) {
        console.warn('captureFrame is deprecated in V3. Use sendLandmarks instead.');
    }
    
    handleFrameProcessed(data) {
        // Update buffer status
        if (data.buffer_size !== undefined) {
            const minRequired = data.min_required || 45;
            this.updateBufferStatus(data.buffer_size, minRequired);
        }
        
        // Update FPS
        const wsStatus = this.websocket.getConnectionStatus();
        this.updateFPS(wsStatus.currentFPS);
        
        // Update performance metrics (if available)
        if (data.performance) {
            this.updatePerformanceMetrics(data.performance);
        }
        
        // Update keypoints visualization and hand status
        if (data.keypoints) {
            this.visualizer.updateKeypoints(data.keypoints);
            if (this.ui) {
                this.ui.updateHandStatus(true);
            }
        } else {
            // No hands detected - clear keypoints
            this.visualizer.updateKeypoints(null);
            if (this.ui) {
                this.ui.updateHandStatus(false);
            }
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
            this.updateSystemStatus('No hands');
        } else if (data.status === 'collecting_frames') {
            this.updateSystemStatus('Collecting...');
        } else if (data.status === 'collecting_stability') {
            this.updateSystemStatus('Stabilizing...');
        } else if (data.status === 'low_confidence') {
            this.updateSystemStatus('Searching...');
            // Show searching state in UI
            if (this.ui) {
                this.ui.updatePrediction(null, data.prediction?.confidence || 0, [], false, true);
            }
        } else if (data.status === 'stuck_sequence_reset') {
            this.updateSystemStatus('Reset');
            this.showNotification(data.message || 'Please move your hand', 'warning');
        } else if (data.status === 'success') {
            this.updateSystemStatus('Active');
        }
        
        // Log top predictions for debugging (uncomment to enable)
        if (data.top_predictions && data.top_predictions.length > 0) {
            const topPreds = data.top_predictions.map(p => 
                `${p.label}: ${(p.confidence * 100).toFixed(1)}%`
            ).join(' | ');
            console.log(`Top 3: ${topPreds}`);
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
        
        if (isCorrect) {
            // Simple case: prediction was correct
            setTimeout(() => {
                try {
                    this.websocket.sendFeedback(this.lastDisplayedPrediction, true, {
                        confidence: this.lastDisplayedConfidence,
                        timestamp: Date.now()
                    });
                    
                    this.showNotification(`✓ Thank you! "${this.lastDisplayedPrediction}" marked as correct`, 'success');
                    console.log('Feedback sent successfully, camera should continue running');
                } catch (error) {
                    console.error('Error in feedback handler:', error);
                }
            }, 0);
            
            // Hide feedback buttons after 1.5 seconds
            setTimeout(() => {
                this.hideFeedbackSection();
            }, 1500);
        } else {
            // Incorrect prediction: show correction modal to get the correct label
            this.showCorrectionModal(this.lastDisplayedPrediction, this.lastDisplayedConfidence);
        }
    }
    
    // ================== CORRECTION MODAL ==================
    
    /**
     * Show the correction modal for incorrect predictions.
     * @param {string} incorrectLabel - The label that was incorrectly predicted
     * @param {number} confidence - The confidence of the incorrect prediction
     */
    showCorrectionModal(incorrectLabel, confidence) {
        const modal = document.getElementById('correctionModal');
        if (!modal) {
            console.error('Correction modal not found');
            return;
        }
        
        // Store for submission
        this._pendingCorrection = {
            incorrectLabel: incorrectLabel,
            confidence: confidence
        };
        
        // Update modal labels
        const labelEl = document.getElementById('incorrectPredictionLabel');
        const labelElEn = document.getElementById('incorrectPredictionLabelEn');
        if (labelEl) labelEl.textContent = incorrectLabel;
        if (labelElEn) labelElEn.textContent = incorrectLabel;
        
        // Populate sign list
        this.populateSignList();
        
        // Reset selection
        this._selectedCorrectLabel = null;
        document.getElementById('selectedSignDisplay').style.display = 'none';
        document.getElementById('submitCorrectionBtn').disabled = true;
        document.getElementById('signSearchInput').value = '';
        
        // Setup modal event listeners
        this.setupCorrectionModalListeners();
        
        // Show modal
        modal.style.display = 'flex';
    }
    
    /**
     * Hide the correction modal.
     */
    hideCorrectionModal() {
        const modal = document.getElementById('correctionModal');
        if (modal) {
            modal.style.display = 'none';
        }
        this._pendingCorrection = null;
        this._selectedCorrectLabel = null;
    }
    
    /**
     * Populate the sign list in the correction modal.
     */
    populateSignList() {
        const signList = document.getElementById('signList');
        if (!signList) return;
        
        // List of all 40 Urdu alphabet signs
        const allSigns = [
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
        ];
        
        signList.innerHTML = allSigns.map(sign => 
            `<div class="sign-list-item" data-sign="${sign}">${sign}</div>`
        ).join('');
        
        // Add click handlers
        signList.querySelectorAll('.sign-list-item').forEach(item => {
            item.addEventListener('click', () => {
                this.selectSign(item.dataset.sign);
            });
        });
    }
    
    /**
     * Select a sign from the list.
     * @param {string} sign - The selected sign label
     */
    selectSign(sign) {
        this._selectedCorrectLabel = sign;
        
        // Update UI
        const signList = document.getElementById('signList');
        signList.querySelectorAll('.sign-list-item').forEach(item => {
            item.classList.toggle('selected', item.dataset.sign === sign);
        });
        
        // Show selected display
        const display = document.getElementById('selectedSignDisplay');
        const label = document.getElementById('selectedSignLabel');
        if (display && label) {
            label.textContent = sign;
            display.style.display = 'block';
        }
        
        // Enable submit button
        document.getElementById('submitCorrectionBtn').disabled = false;
    }
    
    /**
     * Setup event listeners for the correction modal.
     */
    setupCorrectionModalListeners() {
        // Only setup once
        if (this._correctionListenersSetup) return;
        this._correctionListenersSetup = true;
        
        // Close button
        document.getElementById('closeCorrectionModal')?.addEventListener('click', () => {
            this.hideCorrectionModal();
        });
        
        // Cancel button
        document.getElementById('cancelCorrectionBtn')?.addEventListener('click', () => {
            this.hideCorrectionModal();
        });
        
        // Submit button
        document.getElementById('submitCorrectionBtn')?.addEventListener('click', () => {
            this.submitCorrection();
        });
        
        // Search input
        const searchInput = document.getElementById('signSearchInput');
        searchInput?.addEventListener('input', (e) => {
            this.filterSignList(e.target.value);
        });
        
        // Close on backdrop click
        document.getElementById('correctionModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'correctionModal') {
                this.hideCorrectionModal();
            }
        });
    }
    
    /**
     * Filter the sign list based on search input.
     * @param {string} query - Search query
     */
    filterSignList(query) {
        const signList = document.getElementById('signList');
        if (!signList) return;
        
        const lowerQuery = query.toLowerCase();
        signList.querySelectorAll('.sign-list-item').forEach(item => {
            const sign = item.dataset.sign.toLowerCase();
            item.classList.toggle('hidden', !sign.includes(lowerQuery));
        });
    }
    
    /**
     * Submit the correction with the correct label.
     */
    submitCorrection() {
        if (!this._selectedCorrectLabel || !this._pendingCorrection) {
            this.showNotification('Please select the correct sign', 'warning');
            return;
        }
        
        const { incorrectLabel, confidence } = this._pendingCorrection;
        const correctLabel = this._selectedCorrectLabel;
        
        console.log(`Submitting correction: predicted=${incorrectLabel}, correct=${correctLabel}`);
        
        // Send feedback with correction to backend
        this.websocket.sendFeedback(incorrectLabel, false, {
            confidence: confidence,
            timestamp: Date.now(),
            correct_label: correctLabel,  // This is the key addition
            save_sample: true  // Flag to save landmarks for retraining
        });
        
        // Show confirmation
        this.showNotification(
            `✓ Correction saved: "${incorrectLabel}" → "${correctLabel}"`,
            'success'
        );
        
        // Hide modal
        this.hideCorrectionModal();
        
        // Hide feedback section
        setTimeout(() => {
            this.hideFeedbackSection();
        }, 500);
    }
    
    addToHistory(prediction) {
        if (!this.elements.historyList) return;
        
        // Remove "empty" message if present (check both class variants)
        const emptyMessage = this.elements.historyList.querySelector('.history-empty, .history-empty-simple');
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
        if (this.elements.systemStatusValue) {
            this.elements.systemStatusValue.textContent = status;
        }
    }
    
    updateBufferStatus(current, target) {
        // Use UI manager's updateBuffer method
        if (this.ui) {
            this.ui.updateBuffer(current, target);
        }
    }
    
    updateFPS(fps) {
        if (this.ui) {
            this.ui.updateFPS(fps);
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
    
    startDebugHeartbeat() {
        this.stopDebugHeartbeat(); // Clear any existing
        this.lastHeartbeat = Date.now();
        this.debugHeartbeatInterval = setInterval(() => {
            const now = Date.now();
            console.log(`[Heartbeat] App alive at ${now}, isActive: ${this.isRecognitionActive}, camera: ${this.camera?.isActive}`);
            this.lastHeartbeat = now;
        }, 10000); // Every 10 seconds
    }
    
    stopDebugHeartbeat() {
        if (this.debugHeartbeatInterval) {
            clearInterval(this.debugHeartbeatInterval);
            this.debugHeartbeatInterval = null;
        }
    }
    
    // ================== INACTIVITY TRACKING ==================
    
    /**
     * Record user activity (resets inactivity timer).
     * Called when landmarks are detected or user interacts with UI.
     */
    recordActivity() {
        this.lastActivityTime = Date.now();
    }
    
    /**
     * Start the inactivity watchdog timer.
     * This checks periodically if the user has been inactive.
     */
    startInactivityWatchdog() {
        this.stopInactivityWatchdog();  // Clear any existing
        this.lastActivityTime = Date.now();  // Reset on start
        
        this.inactivityCheckInterval = setInterval(() => {
            this.checkInactivity();
        }, this.inactivityCheckIntervalMs);
        
        // Also listen for user interaction events
        this.setupActivityListeners();
        
        console.log(`[Inactivity] Watchdog started (timeout: ${this.inactivityTimeoutMs / 1000 / 60} minutes)`);
    }
    
    /**
     * Stop the inactivity watchdog timer.
     */
    stopInactivityWatchdog() {
        if (this.inactivityCheckInterval) {
            clearInterval(this.inactivityCheckInterval);
            this.inactivityCheckInterval = null;
        }
        
        // Remove activity listeners
        this.removeActivityListeners();
    }
    
    /**
     * Check if user has been inactive and trigger shutdown if needed.
     */
    checkInactivity() {
        if (!this.isRecognitionActive) {
            return;  // No need to check if not recognizing
        }
        
        const now = Date.now();
        const inactiveMs = now - this.lastActivityTime;
        
        if (inactiveMs >= this.inactivityTimeoutMs) {
            console.log(`[Inactivity] User inactive for ${(inactiveMs / 1000 / 60).toFixed(1)} minutes. Stopping recognition.`);
            this.handleInactivityTimeout();
        } else {
            // Log progress every check
            const remainingMs = this.inactivityTimeoutMs - inactiveMs;
            console.log(`[Inactivity] Active check: ${(inactiveMs / 1000).toFixed(0)}s since last activity, ${(remainingMs / 1000 / 60).toFixed(1)} min remaining`);
        }
    }
    
    /**
     * Handle inactivity timeout - stop camera and show overlay.
     */
    handleInactivityTimeout() {
        // Stop recognition
        this.stopRecognition();
        
        // Show inactivity overlay
        this.showInactivityOverlay();
        
        // Show notification
        this.showNotification('Stopped due to inactivity. Click "Start" to resume.', 'warning');
    }
    
    /**
     * Show the inactivity overlay.
     */
    showInactivityOverlay() {
        let overlay = document.getElementById('inactivityOverlay');
        
        if (!overlay) {
            // Create overlay if it doesn't exist
            overlay = document.createElement('div');
            overlay.id = 'inactivityOverlay';
            overlay.className = 'inactivity-overlay';
            overlay.innerHTML = `
                <div class="inactivity-content">
                    <div class="inactivity-icon">😴</div>
                    <h3 class="inactivity-title">
                        <span class="urdu">غیر فعالیت کی وجہ سے رک گیا</span>
                        <span class="english">Stopped due to inactivity</span>
                    </h3>
                    <p class="inactivity-message">
                        <span class="urdu">دوبارہ شروع کرنے کے لیے "شروع کریں" پر کلک کریں</span>
                        <span class="english">Click "Start Recognition" to resume</span>
                    </p>
                    <button class="btn btn-start" id="resumeFromInactivity">
                        <span class="btn-icon">▶</span>
                        <span class="btn-text">
                            <span class="urdu">دوبارہ شروع کریں</span>
                            <span class="english">Resume</span>
                        </span>
                    </button>
                </div>
            `;
            document.body.appendChild(overlay);
            
            // Add click handler for resume button
            document.getElementById('resumeFromInactivity').addEventListener('click', () => {
                this.hideInactivityOverlay();
                this.startRecognition();
            });
        }
        
        overlay.style.display = 'flex';
    }
    
    /**
     * Hide the inactivity overlay.
     */
    hideInactivityOverlay() {
        const overlay = document.getElementById('inactivityOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
    
    /**
     * Setup listeners for user activity events.
     */
    setupActivityListeners() {
        // These events indicate user is present
        this._activityHandler = () => this.recordActivity();
        
        document.addEventListener('mousemove', this._activityHandler);
        document.addEventListener('mousedown', this._activityHandler);
        document.addEventListener('keydown', this._activityHandler);
        document.addEventListener('touchstart', this._activityHandler);
        document.addEventListener('scroll', this._activityHandler);
    }
    
    /**
     * Remove activity listeners.
     */
    removeActivityListeners() {
        if (this._activityHandler) {
            document.removeEventListener('mousemove', this._activityHandler);
            document.removeEventListener('mousedown', this._activityHandler);
            document.removeEventListener('keydown', this._activityHandler);
            document.removeEventListener('touchstart', this._activityHandler);
            document.removeEventListener('scroll', this._activityHandler);
            this._activityHandler = null;
        }
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

