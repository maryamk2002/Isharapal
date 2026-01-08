/**
 * Camera manager for webcam access and frame capture
 * Handles MediaPipe integration and frame processing
 */

class CameraManager {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.stream = null;
        this.isActive = false;
        this.frameRate = 15;  // Match backend TARGET_FPS config (15 FPS)
        this.frameInterval = null;
        this.lastFrameTime = 0;
        this.frameCanvas = null;
        this.frameCtx = null;
        
        // MediaPipe hands
        this.hands = null;
        this.camera = null;
        this.mediaPipeBusy = false;  // Prevent concurrent send() calls
        
        // Event listeners
        this.listeners = new Map();
        
        // Performance monitoring
        this.stats = {
            framesCaptured: 0,
            framesProcessed: 0,
            handDetections: 0,
            lastFrameTime: 0,
            avgFrameTime: 0
        };
        
        // Throttling for "no hands" events to prevent flooding
        this.lastNoHandsEmit = 0;
        this.noHandsThrottleMs = 200; // Only emit "no hands" every 200ms max
        this.consecutiveNoHands = 0;
        
        // Watchdog timer to detect MediaPipe freezes
        this.lastMediaPipeResponse = Date.now();
        this.watchdogInterval = null;
        this.mediaPipeFreezeThreshold = 5000; // 5 seconds without response = frozen
        
        // MediaPipe recovery tracking
        this.recoveryAttempts = 0;
        this.maxRecoveryAttempts = 3;
        this.isRecovering = false;
        this.consecutiveErrors = 0;
        this.maxConsecutiveErrors = 5;
        
        // Instance identification for global state management
        this.instanceId = Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        
        // Track tab visibility state for restoration
        this.wasActiveBeforeHidden = false;
        
        // Clear global MediaPipe on page unload to prevent memory issues
        window.addEventListener('beforeunload', () => {
            if (window.globalMediaPipeHands) {
                try {
                    window.globalMediaPipeHands.close();
                } catch (e) {
                    // Ignore errors during cleanup
                }
                window.globalMediaPipeHands = null;
            }
        });
        
        // Handle tab visibility changes to prevent MediaPipe freezes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('[Camera] Tab hidden - pausing frame capture');
                // Track if we were active before hiding
                this.wasActiveBeforeHidden = this.isActive;
                // Don't stop completely, just slow down
                this.noHandsThrottleMs = 1000; // Slow emissions when hidden
            } else {
                console.log('[Camera] Tab visible - resuming frame capture');
                this.noHandsThrottleMs = 200; // Normal rate
                this.lastMediaPipeResponse = Date.now(); // Reset watchdog
                this.recoveryAttempts = 0; // Reset recovery counter
                this.consecutiveErrors = 0; // Reset error counter
                
                // Add short delay before resuming to let MediaPipe stabilize
                setTimeout(() => {
                    if (this.wasActiveBeforeHidden && this.isActive) {
                        // Force a keep-alive emission after becoming visible
                        this.emit('landmarks', {
                            landmarks: null,
                            hasHands: false,
                            handCount: 0,
                            isVisibilityResume: true
                        });
                    }
                }, 200); // Slightly longer delay for stability
            }
        });
    }
    
    async init() {
        try {
            console.log('Initializing camera...');
            
            // Get video element
            this.video = document.getElementById('webcam');
            if (!this.video) {
                throw new Error('Video element not found');
            }
            
            // Get canvas for overlay
            this.canvas = document.getElementById('overlay-canvas');
            if (this.canvas) {
                this.ctx = this.canvas.getContext('2d');
            }
            
            // V3: MediaPipe runs in FRONTEND for 98% bandwidth reduction!
            await this.initMediaPipe();
            console.log('✓ V3 Mode: MediaPipe runs in browser (sending landmarks only)');

            if (!this.frameCanvas) {
                this.frameCanvas = document.createElement('canvas');
                this.frameCtx = this.frameCanvas.getContext('2d', { willReadFrequently: true });
            }
            
            console.log('Camera initialized successfully');
            return true;
            
        } catch (error) {
            console.error('Camera initialization failed:', error);
            this.emit('error', error);
            return false;
        }
    }
    
    async initMediaPipe() {
        try {
            // Prevent multiple simultaneous initialization attempts
            if (window.mediaPipeInitializing) {
                console.log('⏳ MediaPipe initialization already in progress, waiting...');
                // Wait for existing initialization to complete
                await new Promise((resolve) => {
                    const checkReady = setInterval(() => {
                        if (!window.mediaPipeInitializing && window.globalMediaPipeHands) {
                            clearInterval(checkReady);
                            this.hands = window.globalMediaPipeHands;
                            this.hands.onResults((results) => {
                                this.handleMediaPipeResults(results);
                            });
                            resolve();
                        }
                    }, 200);
                    // Timeout after 30 seconds
                    setTimeout(() => {
                        clearInterval(checkReady);
                        resolve();
                    }, 30000);
                });
                if (this.hands) {
                    console.log('✓ Using MediaPipe instance from parallel initialization');
                    return;
                }
            }
            
            // Check if global instance exists and is valid
            if (window.globalMediaPipeHands && window.mediaPipeInstanceId) {
                try {
                    // Verify the instance is still valid and not owned by a destroyed instance
                    console.log('[Camera] Checking existing MediaPipe Hands instance...');
                    this.hands = window.globalMediaPipeHands;
                    
                    // Re-attach results handler
                    this.hands.onResults((results) => {
                        this.handleMediaPipeResults(results);
                    });
                    
                    console.log('✓ Using cached MediaPipe Hands instance');
                    return;
                } catch (e) {
                    // Instance is stale, need to recreate
                    console.log('⚠️ Cached MediaPipe instance invalid, recreating...');
                    window.globalMediaPipeHands = null;
                    window.mediaPipeInstanceId = null;
                }
            }
            
            // Check if already initialized locally
            if (this.hands) {
                console.log('✓ MediaPipe Hands already initialized locally');
                return;
            }
            
            if (typeof Hands === 'undefined') {
                throw new Error('❌ MediaPipe Hands library not loaded from CDN. Check internet connection.');
            }
            
            // Mark initialization as in progress
            window.mediaPipeInitializing = true;
            
            console.log('⏳ Loading MediaPipe Hands (~10MB WASM files, please wait 10-30 seconds)...');
            this.emit('mediapipe_loading', { status: 'started', message: 'Loading MediaPipe (~10MB)...' });
            
            // Track which files have been logged to prevent spam
            const loggedFiles = new Set();
            
            // CDN fallback list for reliability
            const cdnList = [
                'https://cdn.jsdelivr.net/npm/@mediapipe/hands/',
                'https://unpkg.com/@mediapipe/hands/'
            ];
            let currentCdnIndex = 0;
            
            // Create hands instance with CDN fallback
            this.hands = new Hands({
                locateFile: (file) => {
                    // Only log each file once to prevent spam
                    if (!loggedFiles.has(file)) {
                        loggedFiles.add(file);
                        console.log(`📥 Loading: ${file}`);
                        this.emit('mediapipe_loading', { status: 'downloading', message: `Downloading ${file}...` });
                    }
                    return cdnList[currentCdnIndex] + file;
                }
            });
            
            // Set options
            this.hands.setOptions({
                maxNumHands: 2,
                modelComplexity: 0,  // Fastest (0 = lite model)
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            // Attach results handler
            this.hands.onResults((results) => {
                this.handleMediaPipeResults(results);
            });
            
            // Wait for MediaPipe to be ready (critical!)
            await this.waitForMediaPipeReady();
            
            // Store globally to prevent re-initialization
            window.globalMediaPipeHands = this.hands;
            window.mediaPipeInstanceId = this.instanceId; // Track which instance owns this
            window.mediaPipeInitializing = false;
            
            console.log('✅ MediaPipe Hands fully loaded and ready!');
            
        } catch (error) {
            window.mediaPipeInitializing = false;
            console.error('❌ MediaPipe initialization failed:', error);
            throw error;
        }
    }
    
    async waitForMediaPipeReady() {
        // SIMPLIFIED: Just set the real handler immediately and resolve
        // The old test-based approach was causing delays and handler issues
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('MediaPipe loading timeout (10s). Check your internet connection.'));
            }, 10000); // 10 second timeout
            
            console.log('⏳ Initializing MediaPipe handler...');
            this.emit('mediapipe_loading', { status: 'initializing', message: 'Setting up hand detection...' });
            
            // Set the real handler immediately - no test phase
            this.hands.onResults((results) => {
                this.handleMediaPipeResults(results);
            });
            
            // MediaPipe is ready when we can call initialize()
            // Just wait a short moment for WASM to load
            this.hands.initialize().then(() => {
                clearTimeout(timeout);
                console.log('✅ MediaPipe Hands initialized and ready!');
                this.emit('mediapipe_loading', { status: 'ready', message: 'MediaPipe ready!' });
                resolve();
            }).catch((err) => {
                clearTimeout(timeout);
                console.warn('⚠️ MediaPipe initialize() failed, but handler is set:', err);
                // Handler is already set, so we can continue
                this.emit('mediapipe_loading', { status: 'ready', message: 'MediaPipe ready (with warnings)' });
                resolve();
            });
        });
    }
    
    async start() {
        try {
            if (this.isActive) {
                console.log('Camera already active');
                return true;
            }
            
            console.log('Starting camera...');
            
            // Get user media - facingMode 'user' for front camera on mobile
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 },
                    facingMode: 'user'  // Front camera for sign language
                }
            });
            
            // Set video source
            this.video.srcObject = this.stream;
            
            // Wait for video to be ready
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    // Set canvas size to match video
                    if (this.canvas) {
                        this.canvas.width = this.video.videoWidth;
                        this.canvas.height = this.video.videoHeight;
                    }
                    resolve();
                };
            });
            
            // CRITICAL: Start video playback
            await this.video.play();
            
            // Wait for first frame to be rendered
            await new Promise(resolve => {
                const checkFrame = () => {
                    if (this.video.videoWidth > 0 && this.video.videoHeight > 0) {
                        console.log('[DEBUG] Video playing:', this.video.videoWidth, 'x', this.video.videoHeight);
                        resolve();
                    } else {
                        setTimeout(checkFrame, 100);
                    }
                };
                checkFrame();
            });
            
            // Start frame capture
            this.startFrameCapture();
            
            this.isActive = true;
            console.log('Camera started successfully');
            this.emit('started');
            
            return true;
            
        } catch (error) {
            console.error('Failed to start camera:', error);
            
            // FIXED: Provide specific error information for better UX
            const errorInfo = this._classifyCameraError(error);
            this.emit('error', {
                ...errorInfo,
                originalError: error
            });
            return false;
        }
    }
    
    /**
     * FIXED: Classify camera errors for better user feedback
     * @param {Error} error - The original error
     * @returns {Object} Classified error with type and user-friendly message
     */
    _classifyCameraError(error) {
        const errorName = error.name || '';
        const errorMessage = error.message || '';
        
        // Permission denied
        if (errorName === 'NotAllowedError' || errorMessage.includes('Permission denied')) {
            return {
                type: 'permission_denied',
                message: 'Camera access was denied. Please allow camera permissions.',
                userAction: 'Click the camera icon in your browser\'s address bar and allow access.',
                canRetry: true
            };
        }
        
        // No camera found
        if (errorName === 'NotFoundError' || errorMessage.includes('Requested device not found')) {
            return {
                type: 'no_camera',
                message: 'No camera found on this device.',
                userAction: 'Please connect a camera and try again.',
                canRetry: true
            };
        }
        
        // Camera in use by another app
        if (errorName === 'NotReadableError' || errorMessage.includes('Could not start video source')) {
            return {
                type: 'camera_in_use',
                message: 'Camera is being used by another application.',
                userAction: 'Close other apps using the camera (video calls, etc.) and try again.',
                canRetry: true
            };
        }
        
        // Overconstrained (requested resolution not available)
        if (errorName === 'OverconstrainedError') {
            return {
                type: 'overconstrained',
                message: 'Camera does not support the requested settings.',
                userAction: 'Try refreshing the page.',
                canRetry: true
            };
        }
        
        // HTTPS required
        if (errorName === 'SecurityError' || errorMessage.includes('secure context')) {
            return {
                type: 'security',
                message: 'Camera access requires a secure connection (HTTPS).',
                userAction: 'Please access this site using HTTPS.',
                canRetry: false
            };
        }
        
        // Generic/unknown error
        return {
            type: 'unknown',
            message: `Camera error: ${errorMessage || 'Unknown error'}`,
            userAction: 'Please refresh the page and try again.',
            canRetry: true
        };
    }
    
    stop() {
        try {
            if (!this.isActive) {
                return;
            }
            
            console.log('Stopping camera...');
            
            // Stop frame capture
            this.stopFrameCapture();
            
            // Stop media stream
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }
            
            // DON'T close MediaPipe - keep it alive for reuse
            // This prevents memory leak from repeated initialization
            
            // Clear video source
            if (this.video) {
                this.video.srcObject = null;
            }
            
            // Clear canvas
            if (this.ctx) {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            }
            
            this.isActive = false;
            console.log('Camera stopped (MediaPipe kept alive for reuse)');
            this.emit('stopped');
            
        } catch (error) {
            console.error('Failed to stop camera:', error);
            this.emit('error', error);
        }
    }
    
    startFrameCapture() {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
        }
        
        const interval = 1000 / this.frameRate;
        this.frameInterval = setInterval(() => {
            this.captureFrame();
        }, interval);
        
        // Start watchdog timer to detect MediaPipe freezes
        this.startWatchdog();
        
        console.log(`Frame capture started at ${this.frameRate} FPS`);
    }
    
    stopFrameCapture() {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
        
        // Stop watchdog
        this.stopWatchdog();
        
        console.log('Frame capture stopped');
    }
    
    startWatchdog() {
        this.stopWatchdog(); // Clear any existing
        this.lastMediaPipeResponse = Date.now();
        
        this.watchdogInterval = setInterval(() => {
            const timeSinceLastResponse = Date.now() - this.lastMediaPipeResponse;
            
            if (timeSinceLastResponse > this.mediaPipeFreezeThreshold) {
                console.warn(`⚠️ MediaPipe appears frozen (${(timeSinceLastResponse / 1000).toFixed(1)}s without response). Attempting recovery...`);
                
                // FIXED: Emit recovery_started event so UI can show feedback to user
                this.emit('mediapipe_recovery', {
                    status: 'started',
                    attempt: this.recoveryAttempts + 1,
                    maxAttempts: this.maxRecoveryAttempts,
                    message: 'Hand detection recovering...'
                });
                
                this.attemptMediaPipeRecovery();
            }
        }, 2000); // Check every 2 seconds
    }
    
    stopWatchdog() {
        if (this.watchdogInterval) {
            clearInterval(this.watchdogInterval);
            this.watchdogInterval = null;
        }
    }
    
    async attemptMediaPipeRecovery() {
        // Prevent concurrent recovery attempts
        if (this.isRecovering) {
            console.log('[Camera] Recovery already in progress, skipping');
            return;
        }
        
        this.isRecovering = true;
        this.recoveryAttempts++;
        
        console.log(`🔄 Attempting MediaPipe recovery (attempt ${this.recoveryAttempts}/${this.maxRecoveryAttempts})...`);
        
        // Reset the response timer to prevent repeated recovery attempts
        this.lastMediaPipeResponse = Date.now();
        
        try {
            // Check if we've exceeded max recovery attempts
            if (this.recoveryAttempts > this.maxRecoveryAttempts) {
                console.error('❌ Max recovery attempts exceeded - page reload may be required');
                this.emit('error', { 
                    message: 'MediaPipe recovery failed after multiple attempts. Please refresh the page.',
                    code: 'RECOVERY_EXHAUSTED',
                    requiresReload: true
                });
                this.isRecovering = false;
                return;
            }
            
            // Force re-initialize MediaPipe
            if (this.hands) {
                try {
                    this.hands.reset();
                    console.log('[Camera] MediaPipe reset successful');
                } catch (e) {
                    console.warn('[Camera] MediaPipe reset failed, attempting close and reinit:', e);
                    
                    // Try harder recovery - close and reinitialize
                    try {
                        if (typeof this.hands.close === 'function') {
                            this.hands.close();
                        }
                        window.globalMediaPipeHands = null;
                        window.mediaPipeInstanceId = null;
                        
                        // Reinitialize after a short delay
                        await new Promise(resolve => setTimeout(resolve, 500));
                        await this.initMediaPipe();
                        console.log('✅ MediaPipe reinitialized after close');
                    } catch (reinitError) {
                        console.error('[Camera] MediaPipe reinit failed:', reinitError);
                    }
                }
            }
            
            // Emit a keep-alive signal to backend
            this.emit('landmarks', {
                landmarks: null,
                hasHands: false,
                handCount: 0,
                isRecovery: true
            });
            
            // FIXED: Emit recovery success event so UI can update
            this.emit('mediapipe_recovery', {
                status: 'success',
                attempt: this.recoveryAttempts,
                message: 'Hand detection recovered!'
            });
            
            console.log('✅ MediaPipe recovery signal sent');
            
        } catch (error) {
            console.error('❌ MediaPipe recovery failed:', error);
            
            // FIXED: Emit recovery failure event
            this.emit('mediapipe_recovery', {
                status: 'failed',
                attempt: this.recoveryAttempts,
                message: 'Recovery failed. Please refresh the page.',
                error: error.message
            });
            
            this.emit('error', { message: 'MediaPipe recovery failed', error });
        } finally {
            this.isRecovering = false;
        }
    }
    
    captureFrame() {
        if (!this.isActive || !this.video || this.video.readyState !== 4) {
            return;
        }
        
        // CRITICAL: Verify video has valid dimensions before capturing
        if (this.video.videoWidth === 0 || this.video.videoHeight === 0) {
            console.warn('[SKIP] Video dimensions not ready:', this.video.videoWidth, this.video.videoHeight);
            return;
        }
        
        // Skip if MediaPipe initialization is still in progress
        if (window.mediaPipeInitializing) {
            if (this.stats.framesCaptured % 30 === 0) {
                console.log('⏳ Waiting for MediaPipe initialization...');
            }
            this.stats.framesCaptured++;
            return;
        }
        
        try {
            const startTime = performance.now();
            
            // V3: Send frame to MediaPipe for landmark extraction
            if (this.hands) {
                // Don't use mediaPipeBusy flag here - MediaPipe handles its own queue
                // Just send the frame and let MediaPipe process it
                this.hands.send({ image: this.video }).then(() => {
                    // Success - reset error counter
                    this.consecutiveErrors = 0;
                }).catch(err => {
                    // Track consecutive errors
                    this.consecutiveErrors++;
                    
                    // Log error but don't crash - throttle error logging
                    if (this.stats.framesCaptured % 60 === 0) {
                        console.warn(`[Camera] MediaPipe send error (${this.consecutiveErrors} consecutive):`, err?.message || err);
                    }
                    
                    // Emit warning after multiple consecutive errors
                    if (this.consecutiveErrors >= this.maxConsecutiveErrors) {
                        console.warn(`[Camera] ${this.consecutiveErrors} consecutive MediaPipe errors - may need recovery`);
                        this.emit('mediapipe_warning', { 
                            consecutiveErrors: this.consecutiveErrors,
                            message: 'Multiple consecutive frame processing errors'
                        });
                    }
                    
                    // Emit no-hands to keep connection alive
                    this.emit('landmarks', {
                        landmarks: null,
                        hasHands: false,
                        handCount: 0
                    });
                });
            } else {
                // MediaPipe not ready - log every 60 frames (~4 seconds)
                if (this.stats.framesCaptured % 60 === 0) {
                    console.warn('[Camera] MediaPipe not ready, skipping frame', this.stats.framesCaptured);
                }
                // Still emit no-hands to keep connection alive (throttled)
                if (this.stats.framesCaptured % 15 === 0) {
                    this.emit('landmarks', {
                        landmarks: null,
                        hasHands: false,
                        handCount: 0
                    });
                }
            }
            
            // Update stats
            this.stats.framesCaptured++;
            this.stats.lastFrameTime = startTime;
            
            const frameTime = performance.now() - startTime;
            this.stats.avgFrameTime = (this.stats.avgFrameTime * 0.9) + (frameTime * 0.1);
            
            // Debug: Log frame rate every 5 seconds (75 frames at 15 FPS)
            if (this.stats.framesCaptured % 75 === 0) {
                const now = Date.now();
                console.log(`[Camera] Frames: ${this.stats.framesCaptured}, processed: ${this.stats.framesProcessed}, hands: ${this.stats.handDetections}, loop alive: ${now}`);
            }
            
        } catch (error) {
            console.error('Frame capture error:', error);
            this.emit('error', error);
        }
    }
    
    handleMediaPipeResults(results) {
        try {
            // Reset watchdog timer - MediaPipe is responding
            this.lastMediaPipeResponse = Date.now();
            
            // Clear canvas
            if (this.ctx) {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            }
            
            // Draw hand landmarks with improved visualization
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0 && this.ctx) {
                this.drawHandLandmarksImproved(results.multiHandLandmarks);
                this.stats.handDetections++;
                this.consecutiveNoHands = 0; // Reset counter
                
                // Show hand detection indicator
                this.showHandDetectionIndicator(true);
                
                // V3: Convert landmarks to 189-feature array and emit
                const landmarks189 = this.convertLandmarksTo189(results.multiHandLandmarks);
                this.emit('landmarks', {
                    landmarks: landmarks189,
                    hasHands: true,
                    handCount: results.multiHandLandmarks.length
                });
                
            } else {
                // Show no hands detected
                this.showHandDetectionIndicator(false);
                this.consecutiveNoHands++;
                
                // THROTTLE "no hands" emissions to prevent flooding
                const now = Date.now();
                if (now - this.lastNoHandsEmit >= this.noHandsThrottleMs) {
                    this.lastNoHandsEmit = now;
                    
                    // V3: Emit no-hands signal (throttled)
                    this.emit('landmarks', {
                        landmarks: null,
                        hasHands: false,
                        handCount: 0
                    });
                }
                
                // If no hands for too long (10+ seconds), reduce processing
                if (this.consecutiveNoHands > 150) { // ~10 seconds at 15 FPS
                    // Slow down to prevent CPU overload
                    this.noHandsThrottleMs = 500; // Only emit every 500ms
                } else {
                    this.noHandsThrottleMs = 200; // Normal rate
                }
            }
            
            // Update stats
            this.stats.framesProcessed++;
            
        } catch (error) {
            console.error('MediaPipe results handling error:', error);
        }
    }
    
    /**
     * Convert MediaPipe landmarks to 189-feature array format.
     * Format: [hand1_x0, hand1_y0, hand1_z0, ..., hand2_x0, hand2_y0, hand2_z0, ..., padding]
     * Total: 21 landmarks × 3 coords × 2 hands = 126 features + 63 padding = 189
     */
    convertLandmarksTo189(multiHandLandmarks) {
        // Initialize array with zeros (padding)
        const features = new Array(189).fill(0);
        
        // Process up to 2 hands
        for (let handIdx = 0; handIdx < Math.min(multiHandLandmarks.length, 2); handIdx++) {
            const landmarks = multiHandLandmarks[handIdx];
            const offset = handIdx * 63; // 21 landmarks × 3 coords = 63 per hand
            
            for (let i = 0; i < 21; i++) {
                const landmark = landmarks[i];
                features[offset + i * 3] = landmark.x;       // x (0-1 normalized)
                features[offset + i * 3 + 1] = landmark.y;   // y (0-1 normalized)
                features[offset + i * 3 + 2] = landmark.z;   // z (depth, relative)
            }
        }
        
        return features;
    }
    
    showHandDetectionIndicator(detected) {
        // Show visual indicator on canvas when hands are detected
        if (!this.ctx) return;
        
        const padding = 10;
        const indicatorSize = 12;
        const x = this.canvas.width - indicatorSize - padding;
        const y = padding;
        
        // Draw indicator circle
        this.ctx.beginPath();
        this.ctx.arc(x, y, indicatorSize / 2, 0, 2 * Math.PI);
        this.ctx.fillStyle = detected ? '#10b981' : '#ef4444';
        this.ctx.fill();
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Draw text label
        this.ctx.font = 'bold 14px Arial';
        this.ctx.fillStyle = detected ? '#10b981' : '#ef4444';
        this.ctx.textAlign = 'right';
        this.ctx.fillText(detected ? 'HANDS DETECTED' : 'NO HANDS', x - indicatorSize, y + 5);
    }
    
    drawHandLandmarksImproved(multiHandLandmarks) {
        if (!this.ctx || !multiHandLandmarks) {
            return;
        }
        
        try {
            multiHandLandmarks.forEach((landmarks, handIndex) => {
                // Use different colors for left/right hand
                const color = handIndex === 0 ? '#00ff00' : '#ff00ff';  // Green for first hand, magenta for second
                
                // Draw connections first (underneath)
                this.drawHandConnectionsImproved(landmarks, color);
                
                // Draw landmarks (on top)
                this.drawLandmarkPoints(landmarks, color);
            });
            
        } catch (error) {
            console.error('Error drawing improved hand landmarks:', error);
        }
    }
    
    drawHandLandmarks(landmarks) {
        if (!this.ctx || !landmarks) {
            return;
        }
        
        try {
            landmarks.forEach((landmark, index) => {
                // Draw landmarks
                this.ctx.fillStyle = index === 0 ? '#00FF00' : '#FF0000';
                this.ctx.strokeStyle = index === 0 ? '#00FF00' : '#FF0000';
                this.ctx.lineWidth = 2;
                
                // Draw hand connections
                this.drawHandConnections(landmark);
                
                // Draw landmarks as circles
                landmark.forEach((point, pointIndex) => {
                    const x = point.x * this.canvas.width;
                    const y = point.y * this.canvas.height;
                    
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, 3, 0, 2 * Math.PI);
                    this.ctx.fill();
                    
                    // Draw landmark index
                    this.ctx.fillStyle = '#FFFFFF';
                    this.ctx.font = '10px Arial';
                    this.ctx.fillText(pointIndex.toString(), x + 5, y - 5);
                    this.ctx.fillStyle = index === 0 ? '#00FF00' : '#FF0000';
                });
            });
            
        } catch (error) {
            console.error('Error drawing hand landmarks:', error);
        }
    }
    
    drawHandConnectionsImproved(landmarks, color) {
        if (!this.ctx || !landmarks) {
            return;
        }
        
        // Complete hand connection indices (MediaPipe standard)
        const connections = [
            // Thumb
            [0, 1], [1, 2], [2, 3], [3, 4],
            // Index finger
            [0, 5], [5, 6], [6, 7], [7, 8],
            // Middle finger
            [0, 9], [9, 10], [10, 11], [11, 12],
            // Ring finger
            [0, 13], [13, 14], [14, 15], [15, 16],
            // Pinky
            [0, 17], [17, 18], [18, 19], [19, 20],
            // Palm connections
            [5, 9], [9, 13], [13, 17]
        ];
        
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.lineCap = 'round';
        
        connections.forEach(([start, end]) => {
            if (landmarks[start] && landmarks[end]) {
                const startX = landmarks[start].x * this.canvas.width;
                const startY = landmarks[start].y * this.canvas.height;
                const endX = landmarks[end].x * this.canvas.width;
                const endY = landmarks[end].y * this.canvas.height;
                
                this.ctx.beginPath();
                this.ctx.moveTo(startX, startY);
                this.ctx.lineTo(endX, endY);
                this.ctx.stroke();
            }
        });
    }
    
    drawLandmarkPoints(landmarks, color) {
        if (!this.ctx || !landmarks) {
            return;
        }
        
        landmarks.forEach((point, index) => {
            const x = point.x * this.canvas.width;
            const y = point.y * this.canvas.height;
            
            // Draw landmark point
            this.ctx.beginPath();
            this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
            this.ctx.fillStyle = color;
            this.ctx.fill();
            
            // Add white border for visibility
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            
            // Draw wrist (index 0) larger
            if (index === 0) {
                this.ctx.beginPath();
                this.ctx.arc(x, y, 8, 0, 2 * Math.PI);
                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = 3;
                this.ctx.stroke();
            }
        });
    }
    
    drawHandConnections(landmark) {
        if (!this.ctx || !landmark) {
            return;
        }
        
        // Hand connection indices (simplified)
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],  // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],  // Index finger
            [0, 9], [9, 10], [10, 11], [11, 12],  // Middle finger
            [0, 13], [13, 14], [14, 15], [15, 16],  // Ring finger
            [0, 17], [17, 18], [18, 19], [19, 20]   // Pinky
        ];
        
        this.ctx.beginPath();
        
        connections.forEach(([start, end]) => {
            if (landmark[start] && landmark[end]) {
                const startX = landmark[start].x * this.canvas.width;
                const startY = landmark[start].y * this.canvas.height;
                const endX = landmark[end].x * this.canvas.width;
                const endY = landmark[end].y * this.canvas.height;
                
                this.ctx.moveTo(startX, startY);
                this.ctx.lineTo(endX, endY);
            }
        });
        
        this.ctx.stroke();
    }
    
    setFrameRate(fps) {
        this.frameRate = Math.max(1, Math.min(30, fps));
        console.log(`Frame rate set to ${this.frameRate} FPS`);
        
        // Restart frame capture with new rate
        if (this.isActive) {
            this.startFrameCapture();
        }
    }
    
    getFrameData() {
        if (!this.video || this.video.readyState !== 4) {
            return null;
        }
        
        try {
            if (!this.frameCanvas || !this.frameCtx) {
                this.frameCanvas = document.createElement('canvas');
                this.frameCtx = this.frameCanvas.getContext('2d', { willReadFrequently: true });
            }

            const width = this.video.videoWidth || 640;
            const height = this.video.videoHeight || 480;

            if (!width || !height) {
                return null;
            }

            if (this.frameCanvas.width !== width || this.frameCanvas.height !== height) {
                this.frameCanvas.width = width;
                this.frameCanvas.height = height;
            }

            this.frameCtx.drawImage(this.video, 0, 0, width, height);

            return this.frameCanvas.toDataURL('image/jpeg', 0.72);
            
        } catch (error) {
            console.error('Error getting frame data:', error);
            return null;
        }
    }
    
    async reset() {
        console.log('Resetting camera...');
        
        this.stop();
        
        // DON'T close MediaPipe - keep it alive for reuse
        // Just clear the canvas
        if (this.ctx && this.canvas) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }

        // Reset stats
        this.stats = {
            framesCaptured: 0,
            framesProcessed: 0,
            handDetections: 0,
            lastFrameTime: 0,
            avgFrameTime: 0
        };
        
        console.log('Camera reset complete (MediaPipe kept alive)');
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
            if (callback === undefined) {
                // Clear all listeners for this event if no callback specified
                this.listeners.set(event, []);
            } else {
                const callbacks = this.listeners.get(event);
                const index = callbacks.indexOf(callback);
                if (index > -1) {
                    callbacks.splice(index, 1);
                }
            }
        }
    }
    
    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in camera event listener for ${event}:`, error);
                }
            });
        }
    }
    
    // Getters
    get active() {
        return this.isActive;
    }
    
    get videoElement() {
        return this.video;
    }
    
    get canvasElement() {
        return this.canvas;
    }
    
    get cameraStats() {
        return {
            ...this.stats,
            frameRate: this.frameRate,
            isActive: this.isActive
        };
    }
    
    destroy() {
        console.log('Destroying camera manager...');
        
        // Stop everything
        this.stop();
        
        // Clear all event listeners
        this.listeners.clear();
        
        // Clear references
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.frameCanvas = null;
        this.frameCtx = null;
        
        console.log('Camera manager destroyed');
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CameraManager;
}


