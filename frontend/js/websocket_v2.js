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
                reconnectionAttempts: this.maxReconnectAttempts,
                reconnectionDelay: this.reconnectDelay,
                timeout: 10000
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
        
        this.socket.on('disconnect', () => {
            console.log('Socket disconnected');
            this.connected = false;
            if (this.onConnectionChange) {
                this.onConnectionChange(false);
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
            // Only process NEW stable predictions to avoid flickering
            if (data.prediction.is_new) {
                const prediction = {
                    label: data.prediction.label,
                    confidence: data.prediction.confidence,
                    timestamp: Date.now(),
                    isNew: true,
                    isStable: true
                };
                
                this.lastPrediction = prediction;
                
                // Add to history
                this.addToHistory(prediction);
                
                // Trigger callback
                if (this.onPredictionReceived) {
                    this.onPredictionReceived(prediction);
                }
                
                console.log(`✓ NEW Prediction: ${prediction.label} (${(prediction.confidence * 100).toFixed(1)}%)`);
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
        return true;
    }
    
    stopRecognition() {
        if (!this.connected) {
            return false;
        }
        
        console.log('Stopping recognition...');
        this.socket.emit('stop_recognition', {});
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

