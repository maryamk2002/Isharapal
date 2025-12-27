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
        this.frameRate = 15;  // Increased to 15 FPS for continuous prediction
        this.frameInterval = null;
        this.lastFrameTime = 0;
        this.frameCanvas = null;
        this.frameCtx = null;
        
        // MediaPipe hands
        this.hands = null;
        this.camera = null;
        
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
            
            // V2: SKIP MediaPipe - backend handles it!
            // await this.initMediaPipe();
            console.log('✓ V2 Mode: Skipping MediaPipe (backend processes frames)');

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
            // CRITICAL: Global singleton to prevent multiple instances across page reloads
            if (window.globalMediaPipeHands) {
                console.log('✓ Using global MediaPipe Hands instance (already loaded)');
                this.hands = window.globalMediaPipeHands;
                
                // Re-attach results handler
                this.hands.onResults((results) => {
                    this.handleMediaPipeResults(results);
                });
                return;
            }
            
            // Check if already initialized locally
            if (this.hands) {
                console.log('✓ MediaPipe Hands already initialized locally');
                return;
            }
            
            if (typeof Hands === 'undefined') {
                throw new Error('❌ MediaPipe Hands library not loaded from CDN. Check internet connection.');
            }
            
            console.log('⏳ Loading MediaPipe Hands (~10MB WASM files, please wait 10-30 seconds)...');
            this.emit('mediapipe_loading', { status: 'started', message: 'Loading MediaPipe (~10MB)...' });
            
            // Create hands instance
            this.hands = new Hands({
                locateFile: (file) => {
                    console.log(`📥 Loading: ${file}`);
                    this.emit('mediapipe_loading', { status: 'downloading', message: `Downloading ${file}...` });
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
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
            
            console.log('✅ MediaPipe Hands fully loaded and ready!');
            
        } catch (error) {
            console.error('❌ MediaPipe initialization failed:', error);
            throw error;
        }
    }
    
    async waitForMediaPipeReady() {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('MediaPipe loading timeout (30s). Check your internet connection and try refreshing the page.'));
            }, 30000); // 30 second timeout
            
            // MediaPipe doesn't have a direct "ready" event, so we test it
            // by sending a small dummy image and waiting for the first result
            const testCanvas = document.createElement('canvas');
            testCanvas.width = 100;
            testCanvas.height = 100;
            const testCtx = testCanvas.getContext('2d');
            testCtx.fillStyle = '#000000';
            testCtx.fillRect(0, 0, 100, 100);
            
            console.log('⏳ Testing MediaPipe with dummy frame...');
            this.emit('mediapipe_loading', { status: 'testing', message: 'Testing MediaPipe initialization...' });
            
            // Listen for first result (means MediaPipe is ready)
            const testHandler = () => {
                clearTimeout(timeout);
                console.log('✓ MediaPipe responded to test frame - READY!');
                this.emit('mediapipe_loading', { status: 'ready', message: 'MediaPipe ready!' });
                resolve();
            };
            
            this.hands.onResults(testHandler);
            
            // Send test frame
            this.hands.send({ image: testCanvas }).then(() => {
                console.log('✓ Test frame sent to MediaPipe');
                // Wait a bit for response
                setTimeout(() => {
                    clearTimeout(timeout);
                    console.log('✓ MediaPipe appears ready (timeout not triggered)');
                    this.emit('mediapipe_loading', { status: 'ready', message: 'MediaPipe ready!' });
                    resolve();
                }, 2000);
            }).catch((err) => {
                clearTimeout(timeout);
                console.warn('⚠️ MediaPipe test failed but continuing:', err);
                this.emit('mediapipe_loading', { status: 'ready', message: 'MediaPipe ready (with warnings)' });
                resolve(); // Continue anyway
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
            
            // Get user media
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
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
            this.emit('error', error);
            return false;
        }
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
        
        console.log(`Frame capture started at ${this.frameRate} FPS`);
    }
    
    stopFrameCapture() {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
        
        console.log('Frame capture stopped');
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
        
        try {
            const startTime = performance.now();
            
            // V2: Skip MediaPipe processing (backend does it!)
            // if (this.hands) {
            //     this.hands.send({ image: this.video });
            // }
            
            // Emit raw frame data to backend
            this.emit('frame', this.video);
            
            // Update stats
            this.stats.framesCaptured++;
            this.stats.lastFrameTime = startTime;
            
            const frameTime = performance.now() - startTime;
            this.stats.avgFrameTime = (this.stats.avgFrameTime * 0.9) + (frameTime * 0.1);
            
        } catch (error) {
            console.error('Frame capture error:', error);
            this.emit('error', error);
        }
    }
    
    handleMediaPipeResults(results) {
        try {
            // Clear canvas
            if (this.ctx) {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            }
            
            // Draw hand landmarks with improved visualization
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0 && this.ctx) {
                this.drawHandLandmarksImproved(results.multiHandLandmarks);
                this.stats.handDetections++;
                
                // Show hand detection indicator
                this.showHandDetectionIndicator(true);
            } else {
                // Show no hands detected
                this.showHandDetectionIndicator(false);
            }
            
            // Emit processed results
            this.emit('mediapipe_results', results);
            
            // Update stats
            this.stats.framesProcessed++;
            
        } catch (error) {
            console.error('MediaPipe results handling error:', error);
        }
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
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CameraManager;
}


