/**
 * WebSocket manager for real-time communication with PSL recognition backend
 * Handles connection, frame streaming, and prediction receiving
 */

/**
 * Enhanced WebSocket Manager with robust error handling and reconnection logic
 * Handles real-time communication with the PSL recognition backend
 */
class WebSocketManager {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.isRecognitionActive = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;  // Increased for better reliability
        this.reconnectDelay = 1000;
        this.maxReconnectDelay = 30000;  // Max 30 seconds between retries
        this.sessionId = null;
        this.reconnectTimer = null;
        this.heartbeatTimer = null;
        
        // Event listeners
        this.listeners = new Map();
        
        // Performance monitoring
        this.stats = {
            framesSent: 0,
            predictionsReceived: 0,
            connectionTime: 0,
            lastPing: Date.now(),
            errors: 0,
            reconnects: 0
        };
        
        // Connection state
        this.connectionState = 'disconnected'; // 'connecting', 'connected', 'disconnecting', 'disconnected'
        
        console.log('✓ WebSocket Manager initialized');
    }
    
    async init() {
        try {
            console.log('Initializing WebSocket connection...');
            await this.connect();
            return true;
        } catch (error) {
            console.error('WebSocket initialization failed:', error);
            return false;
        }
    }
    
    async connect() {
        return new Promise((resolve, reject) => {
            try {
                // Validate Socket.IO library
                const ioClient = window.io;
                if (typeof ioClient !== 'function') {
                    const error = new Error('Socket.IO client library not loaded. Please check your internet connection.');
                    console.error(error);
                    reject(error);
                    return;
                }

                // Clean up existing socket
                if (this.socket) {
                    try {
                        if (typeof this.socket.removeAllListeners === 'function') {
                            this.socket.removeAllListeners();
                        }
                        this.socket.disconnect();
                    } catch (cleanupError) {
                        console.warn('Error cleaning up socket:', cleanupError);
                    }
                    this.socket = null;
                }

                // Update connection state
                this.connectionState = 'connecting';

                // Determine backend URL
                const url = window.location.protocol === 'file:' 
                    ? 'http://127.0.0.1:5000' 
                    : window.location.origin;
                console.log('Connecting to backend:', url);

                // Create socket connection with optimized settings
                this.socket = ioClient(url, {
                    path: '/socket.io',
                    reconnection: false,  // We handle reconnection manually
                    withCredentials: true,
                    transports: ['websocket', 'polling'],  // Try websocket first, fallback to polling
                    timeout: 10000,  // 10 second timeout
                    forceNew: true  // Force new connection
                });

                // Connection timeout handler
                const connectionTimeout = setTimeout(() => {
                    if (this.connectionState === 'connecting') {
                        console.error('Connection timeout');
                        this.socket.disconnect();
                        reject(new Error('Connection timeout after 10 seconds'));
                    }
                }, 10000);

                // Handle successful connection
                const handleConnect = () => {
                    clearTimeout(connectionTimeout);
                    console.log('✓ Socket.IO connected successfully');
                    this.isConnected = true;
                    this.connectionState = 'connected';
                    this.reconnectAttempts = 0;
                    this.stats.connectionTime = Date.now();
                    this.stats.reconnects++;
                    
                    // Start heartbeat
                    this.startHeartbeat();
                    
                    // Emit connected event immediately for UI
                    this.emit('connected', { sessionId: this.sessionId || this.socket.id });
                    
                    cleanup();
                    resolve();
                };

                // Handle connection errors
                const handleConnectError = (error) => {
                    clearTimeout(connectionTimeout);
                    console.error('Socket.IO connect error:', error);
                    this.connectionState = 'disconnected';
                    this.stats.errors++;
                    this.emit('error', { message: error.message || 'Connection failed', code: 'CONNECT_ERROR' });
                    
                    if (!this.isConnected) {
                        cleanup();
                        reject(error);
                    }
                };

                // Cleanup event listeners
                const cleanup = () => {
                    this.socket.off('connect', handleConnect);
                    this.socket.off('connect_error', handleConnectError);
                };

                // Attach event handlers
                this.socket.on('connect', handleConnect);
                this.socket.on('connect_error', handleConnectError);

                // Handle disconnection
                this.socket.on('disconnect', (reason) => {
                    console.log('Socket.IO disconnected:', reason);
                    this.isConnected = false;
                    this.isRecognitionActive = false;
                    this.connectionState = 'disconnected';
                    this.stopHeartbeat();
                    this.emit('disconnected', { reason });

                    // Auto-reconnect unless intentionally disconnected
                    if (reason !== 'io client disconnect' && this.reconnectAttempts < this.maxReconnectAttempts) {
                        console.log('Auto-reconnecting...');
                        this.attemptReconnect();
                    } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                        console.error('Max reconnection attempts reached');
                        this.emit('reconnect_failed', { reason: 'max_attempts' });
                    }
                });

                // Handle server events
                this.socket.on('connected', (data) => {
                    this.sessionId = data.session_id;
                    console.log('Server session ID:', this.sessionId);
                    this.emit('connected', data);
                });

                this.socket.on('prediction', (data) => {
                    this.stats.predictionsReceived++;
                    this.stats.lastPing = Date.now();  // Update last activity
                    this.emit('prediction', data);
                });

                this.socket.on('frame_processed', (data) => {
                    this.emit('frame_processed', data);
                });

                this.socket.on('recognition_started', (data) => {
                    console.log('Recognition started on server');
                    this.isRecognitionActive = true;
                    this.emit('recognition_started', data);
                });

                this.socket.on('recognition_stopped', (data) => {
                    console.log('Recognition stopped on server');
                    this.isRecognitionActive = false;
                    this.emit('recognition_stopped', data);
                });

                this.socket.on('error', (error) => {
                    console.error('Socket.IO server error:', error);
                    this.stats.errors++;
                    this.emit('error', { message: error.message || 'Server error', code: 'SERVER_ERROR' });
                });

                this.socket.on('system_status', (data) => {
                    this.emit('system_status', data);
                });

                // Handle connection warnings
                this.socket.on('connect_timeout', () => {
                    console.warn('Connection attempt timed out');
                    this.stats.errors++;
                });

                this.socket.on('reconnect_attempt', (attemptNumber) => {
                    console.log('Reconnection attempt:', attemptNumber);
                });

            } catch (error) {
                console.error('Failed to create Socket.IO connection:', error);
                this.connectionState = 'disconnected';
                this.stats.errors++;
                reject(error);
            }
        });
    }
    
    /**
     * Start heartbeat to detect connection issues
     */
    startHeartbeat() {
        this.stopHeartbeat();  // Clear any existing heartbeat
        
        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected && this.socket) {
                const timeSinceLastPing = Date.now() - this.stats.lastPing;
                
                // If no activity for 60 seconds, emit warning
                if (timeSinceLastPing > 60000) {
                    console.warn('No server activity for 60 seconds');
                    this.emit('connection_warning', { reason: 'no_activity', duration: timeSinceLastPing });
                }
                
                // Ping server to check connection
                this.socket.emit('ping', { timestamp: Date.now() });
            }
        }, 30000);  // Check every 30 seconds
    }
    
    /**
     * Stop heartbeat timer
     */
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }
    
    startRecognition() {
        if (!this.isConnected) {
            console.warn('WebSocket not connected, cannot start recognition');
            return false;
        }
        
        console.log('Starting recognition via WebSocket');
        this.socket.emit('start_recognition', {
            timestamp: Date.now()
        });
        return true;
    }
    
    stopRecognition() {
        if (!this.isConnected) {
            console.warn('WebSocket not connected, cannot stop recognition');
            return false;
        }
        
        console.log('Stopping recognition via WebSocket');
        this.isRecognitionActive = false;
        this.socket.emit('stop_recognition', {
            timestamp: Date.now()
        });
        return true;
    }
    
    resetRecognition() {
        if (!this.isConnected) {
            console.warn('WebSocket not connected, cannot reset recognition');
            return false;
        }
        
        console.log('Resetting recognition (clearing frame buffer) via WebSocket');
        this.socket.emit('reset_recognition', {
            timestamp: Date.now()
        });
        return true;
    }
    
    sendFrame(frameData) {
        if (!this.isConnected || !this.isRecognitionActive) {
            return false;
        }
        
        try {
            // Convert frame to base64 if needed
            let frameString;
            if (typeof frameData === 'string') {
                frameString = frameData;
            } else if (frameData instanceof HTMLCanvasElement) {
                frameString = frameData.toDataURL('image/jpeg', 0.8);
            } else if (frameData instanceof HTMLVideoElement) {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = frameData.videoWidth;
                canvas.height = frameData.videoHeight;
                ctx.drawImage(frameData, 0, 0);
                frameString = canvas.toDataURL('image/jpeg', 0.8);
            } else {
                console.warn('Unsupported frame data type');
                return false;
            }
            
            this.socket.emit('frame_data', {
                frame: frameString,
                timestamp: Date.now()
            });

            this.stats.framesSent++;
            return true;
            
        } catch (error) {
            console.error('Failed to send frame:', error);
            return false;
        }
    }
    
    getSessionInfo() {
        if (!this.isConnected) {
            return null;
        }
        
        this.socket.emit('get_session_info', {
            timestamp: Date.now()
        });
        return true;
    }
    
    getSystemStatus() {
        if (!this.isConnected) {
            return null;
        }
        
        this.socket.emit('get_system_status', {
            timestamp: Date.now()
        });
        return true;
    }
    
    updateSettings(settings) {
        if (!this.isConnected) {
            return false;
        }
        
        this.socket.emit('update_settings', {
            settings,
            timestamp: Date.now()
        });
        return true;
    }
    
    /**
     * Attempt to reconnect with exponential backoff
     */
    attemptReconnect() {
        // Clear any existing reconnect timer
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.emit('reconnect_failed', { 
                attempts: this.reconnectAttempts, 
                maxAttempts: this.maxReconnectAttempts 
            });
            return;
        }
        
        this.reconnectAttempts++;
        
        // Exponential backoff with jitter
        const baseDelay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        const jitter = Math.random() * 1000;  // Random 0-1000ms jitter
        const delay = Math.min(baseDelay + jitter, this.maxReconnectDelay);
        
        console.log(`Reconnecting in ${Math.round(delay/1000)}s (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.emit('reconnect_scheduled', { 
            delay, 
            attempt: this.reconnectAttempts, 
            maxAttempts: this.maxReconnectAttempts 
        });
        
        this.reconnectTimer = setTimeout(async () => {
            try {
                console.log(`Starting reconnection attempt ${this.reconnectAttempts}...`);
                await this.connect();
                console.log('✓ Reconnection successful!');
                this.emit('reconnected', { attempts: this.reconnectAttempts });
                this.reconnectAttempts = 0;  // Reset on success
            } catch (error) {
                console.error('Reconnection failed:', error.message);
                this.attemptReconnect();  // Try again
            }
        }, delay);
    }
    
    /**
     * Gracefully disconnect from server
     */
    disconnect() {
        console.log('Disconnecting WebSocket...');
        
        // Clear timers
        this.stopHeartbeat();
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        // Update state
        this.connectionState = 'disconnecting';
        
        // Stop recognition if active
        if (this.isRecognitionActive) {
            this.stopRecognition();
        }
        
        // Disconnect socket
        if (this.socket) {
            try {
                if (typeof this.socket.removeAllListeners === 'function') {
                    this.socket.removeAllListeners();
                }
                this.socket.disconnect();
            } catch (error) {
                console.warn('Error during socket disconnect:', error);
            }
            this.socket = null;
        }
        
        this.isConnected = false;
        this.isRecognitionActive = false;
        this.connectionState = 'disconnected';
        console.log('✓ WebSocket disconnected');
    }
    
    /**
     * Reset manager to initial state
     */
    reset() {
        console.log('Resetting WebSocket manager...');
        
        this.disconnect();
        this.reconnectAttempts = 0;
        this.sessionId = null;
        this.stats = {
            framesSent: 0,
            predictionsReceived: 0,
            connectionTime: 0,
            lastPing: Date.now(),
            errors: 0,
            reconnects: 0
        };
        console.log('✓ WebSocket manager reset');
    }
    
    // Event system
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }
    
    off(event, callback) {
        if (this.listeners.has(event)) {
            const callbacks = this.listeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }
    
    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }
    
    // Getters
    get connected() {
        return this.isConnected;
    }
    
    get recognitionActive() {
        return this.isRecognitionActive;
    }
    
    get connectionStats() {
        return {
            ...this.stats,
            uptime: this.isConnected ? Date.now() - this.stats.connectionTime : 0,
            reconnectAttempts: this.reconnectAttempts
        };
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketManager;
}


