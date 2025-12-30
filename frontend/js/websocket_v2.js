/**
 * WebSocket Manager V2 for PSL Recognition System
 * Enhanced to work with backend V2 optimized predictor
 */

class WebSocketManager {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.sessionId = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        this.lastPrediction = null;
        this.predictionHistory = [];
        this.maxHistoryLength = 10;
        
        // Event callbacks
        this.onConnectionChange = null;
        this.onPredictionReceived = null;
        this.onFrameProcessed = null;
        this.onError = null;
        
        // Performance tracking
        this.framesSent = 0;
        this.framesProcessed = 0;
        this.lastFPSUpdate = Date.now();
        this.currentFPS = 0;
    }
    
    async init() {
        try {
            console.log('Initializing WebSocket connection V2...');
            
            // Connect to backend
            const serverUrl = 'http://localhost:5000';  // Change if needed
            this.socket = io(serverUrl, {
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionAttempts: 10,  // More attempts
                reconnectionDelay: 1000,   // Faster reconnect
                reconnectionDelayMax: 5000,
                timeout: 30000,            // Longer timeout
                pingTimeout: 120000,       // 2 minute ping timeout
                pingInterval: 10000        // 10 second ping interval
            });
            
            // Set up event listeners
            this.setupEventListeners();
            
            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Connection timeout'));
                }, 10000);
                
                this.socket.once('connected', (data) => {
                    clearTimeout(timeout);
                    console.log('✓ WebSocket V2 connected:', data);
                    this.connected = true;
                    this.sessionId = data.session_id;
                    resolve(true);
                });
                
                this.socket.once('connect_error', (error) => {
                    clearTimeout(timeout);
                    console.error('Connection error:', error);
                    reject(error);
                });
            });
            
        } catch (error) {
            console.error('WebSocket initialization failed:', error);
            throw error;
        }
    }
    
    setupEventListeners() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('Socket connected');
            this.connected = true;
            this.reconnectAttempts = 0;
            if (this.onConnectionChange) {
                this.onConnectionChange(true);
            }
        });
        
        this.socket.on('disconnect', (reason) => {
            console.log('Socket disconnected. Reason:', reason);
            this.connected = false;
            if (this.onConnectionChange) {
                this.onConnectionChange(false);
            }
            
            // Auto-reconnect for transient issues with progressive retry
            const reconnectableReasons = [
                'io server disconnect', 
                'ping timeout', 
                'transport close',
                'transport error'
            ];
            
            if (reconnectableReasons.includes(reason)) {
                this.attemptReconnect();
            }
        });
        
        this.socket.on('connected', (data) => {
            console.log('Session connected:', data);
            this.sessionId = data.session_id;
            this.connected = true;
            if (this.onConnectionChange) {
                this.onConnectionChange(true);
            }
        });
        
        // Recognition events
        this.socket.on('recognition_started', (data) => {
            console.log('Recognition started:', data);
        });
        
        this.socket.on('recognition_stopped', (data) => {
            console.log('Recognition stopped:', data);
        });
        
        this.socket.on('recognition_reset', (data) => {
            console.log('Recognition reset:', data);
            this.predictionHistory = [];
        });
        
        // Frame processing - V2 ENHANCED
        this.socket.on('frame_processed', (data) => {
            this.handleFrameProcessed(data);
        });
        
        // Error handling
        this.socket.on('error', (data) => {
            console.error('WebSocket error:', data);
            if (this.onError) {
                this.onError(data);
            }
        });
        
        // Reconnection
        this.socket.on('reconnect_attempt', (attemptNumber) => {
            console.log(`Reconnection attempt ${attemptNumber}...`);
            this.reconnectAttempts = attemptNumber;
        });
        
        this.socket.on('reconnect_failed', () => {
            console.error('Reconnection failed after maximum attempts');
            if (this.onError) {
                this.onError({message: 'Could not reconnect to server'});
            }
        });
        
        // Successful reconnection
        this.socket.on('reconnect', (attemptNumber) => {
            console.log(`✓ Reconnected successfully after ${attemptNumber} attempts`);
            this.connected = true;
            this.reconnectAttempts = 0;
            if (this.onConnectionChange) {
                this.onConnectionChange(true);
            }
        });
    }
    
    handleFrameProcessed(data) {
        this.framesProcessed++;
        
        // Update FPS
        const now = Date.now();
        if (now - this.lastFPSUpdate >= 1000) {
            this.currentFPS = this.framesProcessed;
            this.framesProcessed = 0;
            this.lastFPSUpdate = now;
        }
        
        // Trigger callback with all data
        if (this.onFrameProcessed) {
            this.onFrameProcessed(data);
        }
        
        // Handle prediction data - V2 ENHANCED
        if (data.prediction && data.prediction.is_stable) {
            const prediction = {
                label: data.prediction.label,
                confidence: data.prediction.confidence,
                timestamp: Date.now(),
                isNew: data.prediction.is_new,
                isStable: true
            };
            
            // Check if this is a different prediction from the last one we displayed
            const isDifferent = !this.lastPrediction || 
                                this.lastPrediction.label !== prediction.label;
            
            // Only trigger callback and add to history for NEW predictions
            if (data.prediction.is_new || isDifferent) {
                this.lastPrediction = prediction;
                
                // Add to history only for new predictions
                if (data.prediction.is_new) {
                    this.addToHistory(prediction);
                }
                
                // Trigger callback for UI update
                if (this.onPredictionReceived) {
                    this.onPredictionReceived(prediction);
                }
                
                console.log(`✓ ${data.prediction.is_new ? 'NEW' : 'UPDATED'} Prediction: ${prediction.label} (${(prediction.confidence * 100).toFixed(1)}%)`);
            }
        }
    }
    
    addToHistory(prediction) {
        this.predictionHistory.unshift(prediction);
        
        // Keep history limited
        if (this.predictionHistory.length > this.maxHistoryLength) {
            this.predictionHistory = this.predictionHistory.slice(0, this.maxHistoryLength);
        }
    }
    
    startRecognition() {
        if (!this.connected) {
            console.error('Cannot start recognition: not connected');
            return false;
        }
        
        console.log('Starting recognition...');
        this.socket.emit('start_recognition', {});
        this.framesSent = 0;
        this.framesProcessed = 0;
        this.lastPrediction = null;  // RESET: Allow first prediction to display
        this.predictionHistory = [];  // RESET: Clear old history
        
        // Start keep-alive interval to prevent timeout during long sessions
        this.startKeepAlive();
        
        return true;
    }
    
    startKeepAlive() {
        this.stopKeepAlive();  // Clear any existing
        
        // More aggressive keep-alive: 15 seconds (browsers throttle timers when hidden)
        this.keepAliveInterval = setInterval(() => {
            if (this.connected && this.socket) {
                this.socket.emit('ping_keep_alive', { timestamp: Date.now() });
            }
        }, 15000);  // Every 15 seconds
        
        // Setup visibility change handler to reconnect when tab becomes visible
        this.setupVisibilityHandler();
    }
    
    stopKeepAlive() {
        if (this.keepAliveInterval) {
            clearInterval(this.keepAliveInterval);
            this.keepAliveInterval = null;
        }
        this.removeVisibilityHandler();
    }
    
    /**
     * Progressive reconnection with exponential backoff
     */
    attemptReconnect(attempt = 1) {
        const maxAttempts = 5;
        const baseDelay = 1000;  // 1 second
        
        if (attempt > maxAttempts) {
            console.error(`❌ Reconnection failed after ${maxAttempts} attempts`);
            if (this.onError) {
                this.onError({ message: 'Could not reconnect to server. Please refresh the page.' });
            }
            return;
        }
        
        const delay = Math.min(baseDelay * Math.pow(2, attempt - 1), 10000);  // Max 10s
        console.log(`🔄 Attempting reconnection (attempt ${attempt}/${maxAttempts}) in ${delay}ms...`);
        
        setTimeout(() => {
            if (!this.connected && this.socket) {
                this.socket.connect();
                
                // Check if reconnection succeeded after 2 seconds
                setTimeout(() => {
                    if (!this.connected) {
                        this.attemptReconnect(attempt + 1);
                    } else {
                        console.log('✓ Reconnection successful!');
                    }
                }, 2000);
            }
        }, delay);
    }
    
    /**
     * Handle tab visibility changes - reconnect immediately when tab becomes visible
     */
    setupVisibilityHandler() {
        this._visibilityHandler = () => {
            if (document.visibilityState === 'visible') {
                console.log('[WebSocket] Tab visible - checking connection...');
                
                if (this.socket && !this.connected) {
                    console.log('[WebSocket] Not connected, forcing reconnect...');
                    this.socket.connect();
                } else if (this.socket && this.connected) {
                    // Send immediate ping to verify connection is alive
                    this.socket.emit('ping_keep_alive', { 
                        timestamp: Date.now(),
                        reason: 'visibility_check'
                    });
                }
            }
        };
        
        document.addEventListener('visibilitychange', this._visibilityHandler);
    }
    
    /**
     * Remove visibility handler
     */
    removeVisibilityHandler() {
        if (this._visibilityHandler) {
            document.removeEventListener('visibilitychange', this._visibilityHandler);
            this._visibilityHandler = null;
        }
    }
    
    stopRecognition() {
        if (!this.connected) {
            return false;
        }
        
        console.log('Stopping recognition...');
        this.socket.emit('stop_recognition', {});
        this.lastPrediction = null;  // RESET: Allow fresh predictions on restart
        
        // Stop keep-alive when recognition stops
        this.stopKeepAlive();
        
        return true;
    }
    
    resetRecognition() {
        if (!this.connected) {
            return false;
        }
        
        console.log('Resetting recognition...');
        this.socket.emit('reset_recognition', {});
        this.predictionHistory = [];
        this.lastPrediction = null;
        return true;
    }
    
    sendFrame(frameData) {
        if (!this.connected) {
            return false;
        }
        
        try {
            this.socket.emit('frame_data', {
                frame: frameData,
                include_all_predictions: false  // Set to true for debugging
            });
            this.framesSent++;
            return true;
        } catch (error) {
            console.error('Error sending frame:', error);
            return false;
        }
    }
    
    /**
     * V3: Send pre-extracted landmarks from frontend MediaPipe.
     * Payload: ~2KB JSON vs ~150KB Base64 image (98% reduction!)
     */
    sendLandmarks(data) {
        if (!this.connected) {
            // Log when not connected to debug
            if (this.framesSent % 30 === 0) {
                console.warn('[WS] Not connected, skipping landmark send');
            }
            return false;
        }
        
        try {
            this.socket.emit('landmark_data', {
                landmarks: data.landmarks,
                has_hands: data.hasHands,
                hand_count: data.handCount
            });
            this.framesSent++;
            
            // Debug: Log every 60 frames to confirm data is being sent
            if (this.framesSent % 60 === 0) {
                console.log(`[WS] Sent ${this.framesSent} landmark frames, hasHands: ${data.hasHands}`);
            }
            
            return true;
        } catch (error) {
            console.error('Error sending landmarks:', error);
            return false;
        }
    }
    
    sendFeedback(label, isCorrect, metadata = {}) {
        if (!this.connected) {
            return false;
        }
        
        try {
            this.socket.emit('feedback', {
                label: label,
                is_correct: isCorrect,
                timestamp: Date.now(),
                metadata: metadata
            });
            console.log(`Feedback sent: ${label} = ${isCorrect ? 'Correct' : 'Incorrect'}`);
            return true;
        } catch (error) {
            console.error('Error sending feedback:', error);
            return false;
        }
    }
    
    updateSettings(settings) {
        if (!this.connected) {
            return false;
        }
        
        this.socket.emit('update_settings', { settings });
        return true;
    }
    
    getConnectionStatus() {
        return {
            connected: this.connected,
            sessionId: this.sessionId,
            framesSent: this.framesSent,
            framesProcessed: this.framesProcessed,
            currentFPS: this.currentFPS
        };
    }
    
    getPredictionHistory() {
        return [...this.predictionHistory];
    }
    
    getLastPrediction() {
        return this.lastPrediction;
    }
    
    disconnect() {
        if (this.socket) {
            console.log('Disconnecting WebSocket...');
            this.socket.disconnect();
            this.connected = false;
            this.sessionId = null;
        }
    }
    
    destroy() {
        this.stopKeepAlive();
        this.disconnect();
        this.socket = null;
        this.onConnectionChange = null;
        this.onPredictionReceived = null;
        this.onFrameProcessed = null;
        this.onError = null;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketManager;
}

