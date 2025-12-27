/**
 * Main application logic for PSL Recognition System
 * Handles initialization, coordination, and user interactions
 */

class PSLRecognitionApp {
    constructor() {
        this.isInitialized = false;
        this.isRecognitionActive = false;
        this.currentLanguage = 'urdu';
        this.settings = {
            sensitivity: 0.5,
            frameRate: 15,  // Increased to 15 FPS for faster continuous prediction
            language: 'urdu'
        };
        this.frameIntervalMs = 1000 / this.settings.frameRate;
        this.lastFrameSentAt = 0;
        this.lastDisplayedPrediction = null;
        this.lastDisplayedConfidence = 0;
        
        // Initialize components
        this.camera = null;
        this.websocket = null;
        this.visualizer = null;
        this.ui = null;
        
        // Bind methods
        this.init = this.init.bind(this);
        this.startRecognition = this.startRecognition.bind(this);
        this.stopRecognition = this.stopRecognition.bind(this);
        this.resetSystem = this.resetSystem.bind(this);
        this.toggleSettings = this.toggleSettings.bind(this);
        this.updateSettings = this.updateSettings.bind(this);
        this.changeLanguage = this.changeLanguage.bind(this);
    }
    
    async init() {
        try {
            console.log('Initializing PSL Recognition System...');
            
            // Show loading overlay
            this.showLoadingOverlay();
            
            // Initialize UI components
            this.ui = new UIManager();
            await this.ui.init();
            
            // Initialize camera
            this.camera = new CameraManager();
            const cameraReady = await this.camera.init();
            if (!cameraReady) {
                throw new Error('Camera initialization failed');
            }
            
            // Initialize WebSocket connection
            this.websocket = new WebSocketManager();
            
            // Set up event listeners BEFORE connecting
            this.setupEventListeners();
            
            // Now connect
            await this.websocket.init();
            
            // Wait a moment for connection to establish
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Force enable start button if connected
            if (this.websocket && this.websocket.connected) {
                console.log('App: WebSocket connected, enabling start button');
                this.ui.setConnectionStatus(true);
            }
            
            // Initialize visualizer
            this.visualizer = new HandVisualizer();
            await this.visualizer.init();
            
            // Load settings
            this.loadSettings();
            
            // Hide loading overlay
            this.hideLoadingOverlay();
            
            this.isInitialized = true;
            console.log('PSL Recognition System initialized successfully');
            
            // Show welcome notification
            this.ui.showNotification(
                'System ready! Click "Start Recognition" to begin.',
                'success'
            );
            
        } catch (error) {
            console.error('Initialization failed:', error);
            this.ui.showNotification(
                `Initialization failed: ${error.message}`,
                'error'
            );
            this.hideLoadingOverlay();
        }
    }
    
    setupEventListeners() {
        // Camera controls
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const resetBtn = document.getElementById('resetBtn');
        
        if (startBtn) {
            startBtn.addEventListener('click', this.startRecognition);
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', this.stopRecognition);
        }
        
        if (resetBtn) {
            resetBtn.addEventListener('click', this.resetSystem);
        }
        
        // Settings
        const settingsToggle = document.getElementById('settingsToggle');
        const settingsClose = document.getElementById('settingsClose');
        const settingsPanel = document.getElementById('settingsPanel');
        
        if (settingsToggle) {
            settingsToggle.addEventListener('click', this.toggleSettings);
        }
        
        if (settingsClose) {
            settingsClose.addEventListener('click', this.toggleSettings);
        }
        
        // Settings controls
        const sensitivitySlider = document.getElementById('sensitivitySlider');
        const frameRateSelect = document.getElementById('frameRateSelect');
        const languageSelect = document.getElementById('languageSelect');
        
        if (sensitivitySlider) {
            sensitivitySlider.addEventListener('input', (e) => {
                this.settings.sensitivity = parseFloat(e.target.value);
                document.getElementById('sensitivityValue').textContent = e.target.value;
                this.updateSettings();
            });
        }
        
        if (frameRateSelect) {
            frameRateSelect.addEventListener('change', (e) => {
                this.settings.frameRate = parseInt(e.target.value);
                this.updateSettings();
            });
        }
        
        if (languageSelect) {
            languageSelect.addEventListener('change', (e) => {
                this.changeLanguage(e.target.value);
            });
        }
        
        // WebSocket events
        if (this.websocket) {
            this.websocket.on('connected', this.handleConnected.bind(this));
            this.websocket.on('disconnected', this.handleDisconnected.bind(this));
            this.websocket.on('prediction', this.handlePrediction.bind(this));
            this.websocket.on('frame_processed', this.handleFrameProcessed.bind(this));
            this.websocket.on('error', this.handleError.bind(this));
        }
        
        // Camera events
        if (this.camera) {
            this.camera.on('frame', this.handleFrame.bind(this));
            this.camera.on('error', this.handleCameraError.bind(this));
        }
        
        // Window events
        window.addEventListener('beforeunload', this.cleanup.bind(this));
        window.addEventListener('resize', this.handleResize.bind(this));
    }
    
    async startRecognition() {
        if (!this.isInitialized) {
            this.ui.showNotification('System not initialized', 'error');
            return;
        }
        
        if (this.isRecognitionActive) {
            return;
        }
        
        try {
            console.log('Starting recognition...');
            
            // Start camera
            await this.camera.start();
            
            // Start WebSocket recognition
            this.websocket.startRecognition();
            
            // Update UI
            this.ui.setRecognitionState(true);
            this.isRecognitionActive = true;
            this.lastFrameSentAt = 0;
            this.lastDisplayedPrediction = null;
            this.lastDisplayedConfidence = 0;
            
            this.ui.showNotification(
                'Recognition started! Show your hands to the camera.',
                'info'
            );
            
        } catch (error) {
            console.error('Failed to start recognition:', error);
            this.ui.showNotification(
                `Failed to start recognition: ${error.message}`,
                'error'
            );
        }
    }
    
    async stopRecognition() {
        if (!this.isRecognitionActive) {
            return;
        }
        
        try {
            console.log('Stopping recognition...');
            
            // Stop WebSocket recognition
            this.websocket.stopRecognition();
            
            // Update UI
            this.ui.setRecognitionState(false);
            this.isRecognitionActive = false;
            
            this.ui.showNotification('Recognition stopped', 'info');
            
        } catch (error) {
            console.error('Failed to stop recognition:', error);
            this.ui.showNotification(
                `Failed to stop recognition: ${error.message}`,
                'error'
            );
        }
    }
    
    async resetSystem() {
        try {
            console.log('Resetting system...');
            
            // Stop recognition if active
            if (this.isRecognitionActive) {
                await this.stopRecognition();
            }
            
            // Reset camera
            if (this.camera) {
                await this.camera.reset();
            }
            
            // Reset WebSocket
            if (this.websocket) {
                this.websocket.reset();
            }
            
            // Reset UI
            if (this.ui) {
                this.ui.reset();
            }
            
            // Reset visualizer
            if (this.visualizer) {
                this.visualizer.clear();
            }
            
            this.lastDisplayedPrediction = null;
            this.lastDisplayedConfidence = 0;
            
            // CRITICAL: Tell backend to clear frame buffer
            if (this.websocket && this.websocket.connected) {
                this.websocket.resetRecognition();
            }

            this.ui.showNotification('System reset - ready for new sign', 'success');
            
        } catch (error) {
            console.error('Failed to reset system:', error);
            this.ui.showNotification(
                `Failed to reset system: ${error.message}`,
                'error'
            );
        }
    }
    
    toggleSettings() {
        const settingsPanel = document.getElementById('settingsPanel');
        if (settingsPanel) {
            settingsPanel.classList.toggle('open');
        }
    }
    
    updateSettings() {
        // Update camera settings
        if (this.camera) {
            this.camera.setFrameRate(this.settings.frameRate);
        }
        
        // Update WebSocket settings
        if (this.websocket) {
            this.websocket.updateSettings(this.settings);
        }
        
        this.frameIntervalMs = 1000 / Math.max(1, this.settings.frameRate);

        // Save settings to localStorage
        localStorage.setItem('pslSettings', JSON.stringify(this.settings));
    }
    
    changeLanguage(language) {
        this.currentLanguage = language;
        this.settings.language = language;
        
        // Update UI language
        if (this.ui) {
            this.ui.setLanguage(language);
        }
        
        // Save settings
        this.updateSettings();
        
        this.ui.showNotification(
            `Language changed to ${language === 'urdu' ? 'اردو' : 'English'}`,
            'info'
        );
    }
    
    loadSettings() {
        try {
            const savedSettings = localStorage.getItem('pslSettings');
            if (savedSettings) {
                const settings = JSON.parse(savedSettings);
                this.settings = { ...this.settings, ...settings };
                
                // Update UI with loaded settings
                const sensitivitySlider = document.getElementById('sensitivitySlider');
                const frameRateSelect = document.getElementById('frameRateSelect');
                const languageSelect = document.getElementById('languageSelect');
                
                if (sensitivitySlider) {
                    sensitivitySlider.value = this.settings.sensitivity;
                    document.getElementById('sensitivityValue').textContent = this.settings.sensitivity;
                }
                
                if (frameRateSelect) {
                    frameRateSelect.value = this.settings.frameRate;
                }
                
                if (languageSelect) {
                    languageSelect.value = this.settings.language;
                }
                
                // Apply settings
                this.updateSettings();
                this.changeLanguage(this.settings.language);
            }
        } catch (error) {
            console.error('Failed to load settings:', error);
        }
    }
    
    // Event handlers
    handleConnected(data) {
        console.log('Connected to server');
        this.ui.setConnectionStatus(true);
        
        // Enable start button explicitly
        const startBtn = document.getElementById('startBtn');
        if (startBtn && !this.isRecognitionActive) {
            startBtn.disabled = false;
        }
    }
    
    handleDisconnected() {
        console.log('Disconnected from server');
        this.ui.setConnectionStatus(false);
    }
    
    handlePrediction(data) {
        // STABLE PREDICTION - Show only after majority voting confirms stability
        const isStable = data.is_stable || false;
        
        this.ui.updatePrediction(data.prediction, data.confidence, data.all_predictions, isStable);
        this.ui.updatePredictionList(data.all_predictions);
        
        // Update visualizer with prediction and keypoints
        if (this.visualizer) {
            this.visualizer.showPrediction(data.prediction, data.confidence);
            
            // Update hand skeleton with keypoints from backend
            if (data.keypoints && Array.isArray(data.keypoints)) {
                this.visualizer.updateKeypoints(data.keypoints);
            }
        }
        
        // Track last prediction for reference
        this.lastDisplayedPrediction = data.prediction;
        this.lastDisplayedConfidence = data.confidence;
        
        // Hide loading if showing
        this.ui.hideLoadingProgress();
    }
    
    handleFrameProcessed(data) {
        // ALWAYS update keypoints for continuous skeleton visualization
        if (this.visualizer && data.keypoints && Array.isArray(data.keypoints)) {
            this.visualizer.updateKeypoints(data.keypoints);
        }
        
        // Handle different statuses
        if (data.status === 'collecting_frames') {
            // Show initial frame collection progress
            const targetFrames = data.target_frames || 60;
            const percent = data.progress_percent || Math.round((data.frames_collected / targetFrames) * 100);
            this.ui.showLoadingProgress(percent);
        } else if (data.status === 'stabilizing') {
            // Show stabilization progress (show current top prediction while stabilizing)
            if (this.ui.showStabilizing) {
                this.ui.showStabilizing(data.history_size, data.threshold);
            }
            // Optionally show the current top prediction in a muted way
            if (data.current_top_prediction && this.ui.showTemporaryPrediction) {
                this.ui.showTemporaryPrediction(data.current_top_prediction, data.current_confidence);
            }
        } else if (data.status === 'stable_prediction_continues') {
            // Prediction is stable and continuing - keypoints already updated above
            this.ui.hideLoadingProgress();
        } else if (data.status === 'no_hands') {
            // No hands detected - clear skeleton
            if (this.visualizer) {
                this.visualizer.updateKeypoints(null);
            }
            this.ui.hideLoadingProgress();
        } else if (data.status === 'low_confidence') {
            // Low confidence but hands detected - skeleton still visible
            this.ui.hideLoadingProgress();
        } else {
            // Other statuses
            this.ui.hideLoadingProgress();
        }
    }
    
    handleFrame(frameData) {
        if (!this.isRecognitionActive || !this.websocket) {
            return;
        }

        // FAST MODE - No backpressure, just throttle by time
        const now = performance.now();
        if (now - this.lastFrameSentAt < this.frameIntervalMs) {
            return;
        }

        const payload = this.camera && typeof this.camera.getFrameData === 'function'
            ? this.camera.getFrameData()
            : null;

        if (!payload) {
            return;
        }

        const sent = this.websocket.sendFrame(payload);
        if (sent) {
            this.lastFrameSentAt = now;
        }
    }
    
    handleError(error) {
        console.error('WebSocket error:', error);
        this.ui.showNotification(`Connection error: ${error.message}`, 'error');
    }
    
    handleCameraError(error) {
        console.error('Camera error:', error);
        this.ui.showNotification(`Camera error: ${error.message}`, 'error');
    }
    
    handleResize() {
        // Handle window resize
        if (this.visualizer) {
            this.visualizer.handleResize();
        }
    }
    
    cleanup() {
        console.log('Cleaning up...');
        
        if (this.isRecognitionActive) {
            this.stopRecognition();
        }
        
        if (this.websocket) {
            this.websocket.disconnect();
        }
        
        if (this.camera) {
            this.camera.stop();
        }
    }
    
    showLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('hidden');
        }
    }
    
    hideLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    const app = new PSLRecognitionApp();
    await app.init();
    
    // Make app globally available for debugging
    window.pslApp = app;
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PSLRecognitionApp;
}
