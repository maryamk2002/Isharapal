/**
 * Browser-Only PSL Recognition Application
 * 
 * This replaces the WebSocket-based app_v2.js with a fully browser-based solution.
 * Uses ONNX Runtime Web for local inference - no backend required!
 * 
 * Features:
 * - MediaPipe hands detection (browser)
 * - ONNX model inference (browser)
 * - Feedback storage (LocalStorage/IndexedDB)
 * - Works offline after initial load
 */

class PSLRecognitionApp {
    constructor() {
        this.isInitialized = false;
        this.isRecognitionActive = false;
        this.currentLanguage = 'urdu';
        
        // ==========================================
        // OPTIMIZED Settings for Speed
        // ==========================================
        this.settings = {
            sensitivity: 0.5,
            frameRate: 18,  // INCREASED from 15 to 18 FPS for faster response
            language: 'urdu'
        };
        this.frameIntervalMs = 1000 / this.settings.frameRate;
        this.lastFrameSentAt = 0;
        this.lastDisplayedPrediction = null;
        this.lastDisplayedConfidence = 0;
        
        // NEW: Performance tracking
        this.performanceMetrics = {
            avgFrameTime: 0,
            avgPredictionTime: 0,
            droppedFrames: 0,
            qualityIssues: 0
        };
        
        // Components
        this.camera = null;
        this.predictor = null;  // ONNXPredictor (replaces WebSocket)
        this.feedbackManager = null;  // NEW: Browser-based feedback
        this.wordFormation = null;  // NEW: Word formation from letters
        this.visualizer = null;
        this.ui = null;
        
        // DOM Elements
        this.elements = {};
        
        // Bound handlers
        this.boundFrameHandler = null;
        this.predictionLoopInterval = null;
        
        // Current sequence for feedback (last landmarks)
        this.currentLandmarkSequence = null;
        
        // Bind methods
        this.init = this.init.bind(this);
        this.startRecognition = this.startRecognition.bind(this);
        this.stopRecognition = this.stopRecognition.bind(this);
        this.resetSystem = this.resetSystem.bind(this);
        this.handleFeedback = this.handleFeedback.bind(this);
    }
    
    async init() {
        try {
            console.log('Initializing PSL Recognition System (Browser-Only Mode)...');
            
            // Show loading overlay
            this.showLoadingOverlay();
            
            // Get DOM elements
            this.initDOMElements();
            
            // Hide feedback section initially
            this.hideFeedbackSection();
            
            // Initialize UI components
            this.ui = new UIManager();
            await this.ui.init();
            
            // Initialize camera (MediaPipe runs in browser)
            this.updateLoadingMessage('Initializing camera...');
            this.camera = new CameraManager();
            const cameraReady = await this.camera.init();
            if (!cameraReady) {
                throw new Error('Camera initialization failed');
            }
            
            // Initialize ONNX predictor (replaces WebSocket)
            // OPTIMIZED: Faster settings for better responsiveness
            this.updateLoadingMessage('Loading AI model (~11MB)...');
            this.predictor = new ONNXPredictor({
                // SPEED: Smaller buffer, faster predictions
                slidingWindowSize: 36,       // Was 45, faster buffer fill
                minPredictionFrames: 12,     // Was 32, predict after ~0.8s
                
                // SPEED: Faster stability (2/3 majority)
                stabilityVotes: 3,           // Was 5
                stabilityThreshold: 2,       // Was 3
                
                // ACCURACY: Balanced confidence
                minConfidence: 0.58,         // Was 0.55, slightly stricter
                lowConfidenceThreshold: 0.42,
                
                // ROBUSTNESS: Quality thresholds
                minHandVisibility: 0.6,
                maxJitterThreshold: 0.15
            });
            
            // Set up predictor callbacks
            this.predictor.onLoadProgress = (progress) => {
                this.updateLoadingMessage(progress.message);
            };
            
            this.predictor.onPrediction = (prediction) => {
                this.handleNewPrediction(prediction);
            };
            
            this.predictor.onError = (error) => {
                console.error('Predictor error:', error);
                this.showNotification(error.message, 'error');
            };
            
            // NEW: Quality issue handler - helps user know when hand detection is poor
            this.predictor.onQualityIssue = (quality) => {
                this.performanceMetrics.qualityIssues++;
                // Only show warning occasionally to avoid spam
                if (this.performanceMetrics.qualityIssues % 30 === 1) {
                    if (quality.reason === 'low_visibility') {
                        console.log('[Quality] Low hand visibility - ensure hand is fully in frame');
                    } else if (quality.reason === 'high_jitter') {
                        console.log('[Quality] Hand movement too fast - try slower gestures');
                    }
                }
            };
            
            const predictorReady = await this.predictor.init(
                'models/psl_model_v2.onnx',
                'models/psl_labels.json',
                'models/sign_thresholds.json'
            );
            
            if (!predictorReady) {
                throw new Error('Model loading failed');
            }
            
            // Initialize feedback manager
            this.updateLoadingMessage('Setting up feedback system...');
            this.feedbackManager = new FeedbackManager();
            await this.feedbackManager.init();
            
            // Set up feedback callbacks
            this.feedbackManager.onConfusionHint = (hint) => {
                this.showNotification(hint.hint, 'info');
            };
            
            // Initialize word formation module - OPTIMIZED for faster response
            this.updateLoadingMessage('Setting up word formation...');
            this.wordFormation = new WordFormation({
                pauseThresholdMs: 4000,  // FASTER: 4s instead of 5s for word completion
                minLetterGapMs: 1200,    // FASTER: 1.2s instead of 1.5s for same letter
                showPauseIndicator: true
            });
            
            // Set up word formation callbacks
            this.wordFormation.onLetterAdded = (data) => {
                this.updateCurrentWordDisplay(data.currentWord);
            };
            
            this.wordFormation.onWordComplete = (data) => {
                this.showNotification(`Word complete: ${data.word}`, 'success');
            };
            
            this.wordFormation.onSentenceUpdate = (data) => {
                this.updateSentenceDisplay(data);
            };
            
            this.wordFormation.onPauseProgress = (data) => {
                this.updatePauseIndicator(data);
            };
            
            await this.wordFormation.init('models/urdu_mapping.json');
            
            // Initialize visualizer
            this.updateLoadingMessage('Setting up visualization...');
            this.visualizer = new HandVisualizer();
            await this.visualizer.init();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Hide loading overlay
            this.hideLoadingOverlay();
            
            // Update connection status (always "connected" in browser mode)
            this.ui.setConnectionStatus(true);
            this.updateConnectionBadge(true);
            
            this.isInitialized = true;
            console.log('[OK] PSL Recognition System initialized (Browser-Only Mode)');
            
            // Show welcome notification with backend info
            const modelInfo = this.predictor.getModelInfo();
            const backendMsg = modelInfo.backendUsed === 'webgpu' 
                ? 'GPU accelerated! 🚀' 
                : 'Running on CPU';
            this.showNotification(`System ready! ${backendMsg} - All processing happens locally.`, 'success');
            
            // Log performance configuration
            console.log('[PERF] Configuration:', {
                backend: modelInfo.backendUsed,
                frameRate: this.settings.frameRate,
                slidingWindow: modelInfo.config.slidingWindowSize,
                minFrames: modelInfo.config.minPredictionFrames,
                stability: `${modelInfo.config.stabilityThreshold}/${modelInfo.config.stabilityVotes}`
            });
            
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
            connectionBadge: document.getElementById('connectionBadge'),
            
            // Overlays
            loadingOverlay: document.getElementById('loadingOverlay'),
            notificationContainer: document.getElementById('notificationContainer'),
            
            // Word Formation Elements
            sentenceDisplay: document.getElementById('sentenceDisplay'),
            currentWordDisplay: document.getElementById('currentWordDisplay'),
            pauseIndicator: document.getElementById('pauseIndicator'),
            pauseProgress: document.getElementById('pauseProgress'),
            pauseText: document.getElementById('pauseText'),
            copyTextBtn: document.getElementById('copyTextBtn'),
            clearTextBtn: document.getElementById('clearTextBtn'),
            addSpaceBtn: document.getElementById('addSpaceBtn'),
            deleteLetterBtn: document.getElementById('deleteLetterBtn'),
            deleteWordBtn: document.getElementById('deleteWordBtn')
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
        
        // Word Formation buttons
        if (this.elements.copyTextBtn) {
            this.elements.copyTextBtn.addEventListener('click', () => {
                this.copyTextToClipboard();
            });
        }
        
        if (this.elements.clearTextBtn) {
            this.elements.clearTextBtn.addEventListener('click', () => {
                this.clearWordFormation();
            });
        }
        
        if (this.elements.addSpaceBtn) {
            this.elements.addSpaceBtn.addEventListener('click', () => {
                if (this.wordFormation) {
                    this.wordFormation.addSpace();
                }
            });
        }
        
        if (this.elements.deleteLetterBtn) {
            this.elements.deleteLetterBtn.addEventListener('click', () => {
                if (this.wordFormation) {
                    this.wordFormation.deleteLastLetter();
                }
            });
        }
        
        if (this.elements.deleteWordBtn) {
            this.elements.deleteWordBtn.addEventListener('click', () => {
                if (this.wordFormation) {
                    this.wordFormation.deleteLastWord();
                }
            });
        }
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }
            
            switch(e.code) {
                case 'Space':
                case 'Enter':
                    if (!this.isRecognitionActive && this.isInitialized) {
                        e.preventDefault();
                        this.startRecognition();
                    }
                    break;
                    
                case 'Escape':
                    if (this.isRecognitionActive) {
                        e.preventDefault();
                        this.stopRecognition();
                    }
                    break;
                    
                case 'KeyR':
                    if (!this.isRecognitionActive) {
                        e.preventDefault();
                        this.resetSystem();
                    }
                    break;
                    
                case 'KeyC':
                    if (this.lastDisplayedPrediction && this.elements.feedbackSection && 
                        this.elements.feedbackSection.style.display !== 'none') {
                        e.preventDefault();
                        this.handleFeedback(true);
                    }
                    break;
                    
                case 'KeyI':
                    if (this.lastDisplayedPrediction && this.elements.feedbackSection && 
                        this.elements.feedbackSection.style.display !== 'none') {
                        e.preventDefault();
                        this.handleFeedback(false);
                    }
                    break;
                    
                case 'KeyE':
                    // Export feedback data (Ctrl+E)
                    if (e.ctrlKey && this.feedbackManager) {
                        e.preventDefault();
                        this.feedbackManager.downloadExport();
                        this.showNotification('Exporting feedback data...', 'info');
                    }
                    break;
            }
        });
        
        console.log('Keyboard shortcuts enabled:');
        console.log('  Space/Enter - Start Recognition');
        console.log('  Esc - Stop Recognition');
        console.log('  R - Reset System');
        console.log('  C - Mark Correct');
        console.log('  I - Mark Incorrect');
        console.log('  Ctrl+E - Export Feedback');
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
            console.log('Starting recognition (Browser-Only Mode)...');
            
            // Reset state
            this.lastDisplayedPrediction = null;
            this.lastDisplayedConfidence = 0;
            this.lastFrameSentAt = 0;
            this.predictor.clearBuffer();
            
            // Start camera
            const cameraStarted = await this.camera.start();
            if (!cameraStarted) {
                throw new Error('Failed to start camera');
            }
            
            this.isRecognitionActive = true;
            
            // Remove any existing landmark listener
            if (this.boundFrameHandler) {
                this.camera.off('landmarks', this.boundFrameHandler);
            }
            
            // Listen to landmarks from camera
            this.boundFrameHandler = (data) => {
                this.handleLandmarks(data);
            };
            this.camera.on('landmarks', this.boundFrameHandler);
            
            // Start prediction loop
            this.startPredictionLoop();
            
            // Update UI
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.updateSystemStatus('Recognizing...');
            
            // Clear previous prediction
            this.clearPredictionDisplay();
            
            console.log('[OK] Recognition started');
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
            
            // Remove landmarks listener
            if (this.boundFrameHandler) {
                this.camera.off('landmarks', this.boundFrameHandler);
                this.boundFrameHandler = null;
            }
            
            // Stop prediction loop
            this.stopPredictionLoop();
            
            // Clear predictor buffer
            this.predictor.clearBuffer();
            
            // Reset state
            this.lastDisplayedPrediction = null;
            this.lastDisplayedConfidence = 0;
            
            // Clear visualizer
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
            
            console.log('[OK] Recognition stopped');
            this.showNotification('Recognition stopped', 'info');
            
        } catch (error) {
            console.error('Failed to stop recognition:', error);
            this.showNotification(`Failed to stop: ${error.message}`, 'error');
        }
    }
    
    resetSystem() {
        try {
            console.log('Resetting system...');
            
            // Reset state
            this.lastDisplayedPrediction = null;
            this.lastDisplayedConfidence = 0;
            this.currentLandmarkSequence = null;
            
            // Clear predictor
            if (this.predictor) {
                this.predictor.clearBuffer();
            }
            
            // Clear visualizer
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
            
            // Clear word formation
            if (this.wordFormation) {
                this.wordFormation.clearAll();
            }
            
            console.log('[OK] System reset');
            this.showNotification('System reset', 'info');
            
        } catch (error) {
            console.error('Failed to reset:', error);
            this.showNotification(`Failed to reset: ${error.message}`, 'error');
        }
    }
    
    /**
     * Handle landmarks from camera (replaces sendLandmarks in app_v2.js)
     */
    handleLandmarks(data) {
        const now = Date.now();
        
        // Throttle frame processing
        if (now - this.lastFrameSentAt < this.frameIntervalMs) {
            return;
        }
        
        if (!this.isRecognitionActive) {
            return;
        }
        
        // Defensive check: ensure predictor is available
        if (!this.predictor || !this.predictor.isReady) {
            return;
        }
        
        try {
            this.lastFrameSentAt = now;
            
            // Update hand status
            if (this.ui) {
                this.ui.updateHandStatus(data.hasHands);
            }
            
            // Add frame to predictor buffer
            if (data.hasHands && data.landmarks) {
                this.predictor.addFrame(data.landmarks);
                this.currentLandmarkSequence = this.predictor.getCurrentSequence();
                
                // Update visualizer
                if (this.visualizer) {
                    this.visualizer.updateKeypoints(data.landmarks.slice(0, 126));
                }
            } else {
                // No hands - clear visualizer
                if (this.visualizer) {
                    this.visualizer.updateKeypoints(null);
                }
            }
            
        } catch (error) {
            console.error('Error handling landmarks:', error);
        }
    }
    
    /**
     * Start the prediction loop (runs at ~12Hz for faster response)
     */
    startPredictionLoop() {
        this.stopPredictionLoop();  // Clear any existing
        
        // OPTIMIZED: Faster prediction rate (12Hz instead of 10Hz)
        const predictionInterval = 83;  // ~12 predictions per second (was 100ms)
        
        this.predictionLoopInterval = setInterval(async () => {
            if (!this.isRecognitionActive) {
                return;
            }
            
            try {
                const startTime = performance.now();
                const result = await this.predictor.predict(true);
                const predTime = performance.now() - startTime;
                
                // Track prediction performance
                this.performanceMetrics.avgPredictionTime = 
                    this.performanceMetrics.avgPredictionTime * 0.9 + predTime * 0.1;
                
                this.handlePredictionResult(result);
                
            } catch (error) {
                console.error('Prediction loop error:', error);
            }
            
        }, predictionInterval);
        
        console.log('Prediction loop started at 12Hz');
    }
    
    /**
     * Stop the prediction loop
     */
    stopPredictionLoop() {
        if (this.predictionLoopInterval) {
            clearInterval(this.predictionLoopInterval);
            this.predictionLoopInterval = null;
        }
    }
    
    /**
     * Handle prediction result from ONNX predictor
     * OPTIMIZED: Better status handling and confidence display
     */
    handlePredictionResult(result) {
        // Update buffer status
        if (result.bufferSize !== undefined) {
            this.updateBufferStatus(result.bufferSize, this.predictor.config.slidingWindowSize);
        }
        
        // IMPROVED: More informative status messages
        if (result.status === 'collecting_frames') {
            const progress = Math.round((result.bufferSize / result.minRequired) * 100);
            this.updateSystemStatus(`Collecting... ${progress}%`);
        } else if (result.status === 'low_confidence') {
            this.updateSystemStatus('Searching...');
            if (this.ui) {
                this.ui.updatePrediction(null, result.confidence || 0, [], false, true);
            }
            // Track low confidence predictions
            this.performanceMetrics.droppedFrames++;
        } else if (result.status === 'collecting_stability') {
            // NEW: Show top prediction while stabilizing (gives user feedback)
            this.updateSystemStatus('Stabilizing...');
            if (result.prediction && this.ui) {
                // Show prediction with lower opacity or different style to indicate it's not final
                this.ui.updatePrediction(result.prediction, result.confidence, result.allPredictions || [], false, false);
            }
        } else if (result.status === 'success' && result.isStable) {
            this.updateSystemStatus('Active');
        } else if (result.status === 'error') {
            this.updateSystemStatus('Error');
            console.error('Prediction error:', result.error);
        }
        
        // Update FPS (use prediction time for more accurate reading)
        if (result.predictionTimeMs) {
            const inferenceRate = 1000 / result.predictionTimeMs;
            // Blend with actual camera FPS for display
            const displayFps = Math.min(inferenceRate, this.settings.frameRate);
            this.updateFPS(Math.round(displayFps));
        }
    }
    
    /**
     * Handle new stable prediction (triggered by predictor callback)
     */
    handleNewPrediction(prediction) {
        console.log('New prediction:', prediction);
        
        // Update prediction display
        this.displayPrediction(prediction.label, prediction.confidence);
        
        // Add to history
        this.addToHistory(prediction);
        
        // Send to word formation module
        if (this.wordFormation) {
            this.wordFormation.processPrediction({
                label: prediction.label,
                confidence: prediction.confidence,
                timestamp: Date.now()
            });
        }
        
        // Show feedback section
        this.showFeedbackSection();
        
        // Update visualizer
        if (this.visualizer) {
            this.visualizer.showPrediction(prediction.label, prediction.confidence);
        }
        
        // Update sign count
        const stats = this.predictor.getStats();
        if (this.elements.signCountValue) {
            this.elements.signCountValue.textContent = stats.successfulPredictions;
        }
    }
    
    displayPrediction(label, confidence) {
        if (this.ui) {
            this.ui.updatePrediction(label, confidence);
        }
        
        this.lastDisplayedPrediction = label;
        this.lastDisplayedConfidence = confidence;
    }
    
    clearPredictionDisplay() {
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
    
    // ================== WORD FORMATION ==================
    
    /**
     * Update the current word being formed display
     */
    updateCurrentWordDisplay(currentWord) {
        if (this.elements.currentWordDisplay) {
            this.elements.currentWordDisplay.textContent = currentWord || '-';
        }
    }
    
    /**
     * Update the sentence display with completed words and current word
     */
    updateSentenceDisplay(data) {
        if (this.elements.sentenceDisplay) {
            if (data.fullText) {
                this.elements.sentenceDisplay.innerHTML = `
                    <span class="completed-words">${data.sentence || ''}</span>
                    ${data.sentence && data.currentWord ? ' ' : ''}
                    <span class="forming-word">${data.currentWord || ''}</span>
                `;
            } else {
                this.elements.sentenceDisplay.innerHTML = '<span class="sentence-placeholder">الفاظ یہاں ظاہر ہوں گے</span>';
            }
        }
        
        if (this.elements.currentWordDisplay) {
            this.elements.currentWordDisplay.textContent = data.currentWord || '-';
        }
    }
    
    /**
     * Update the pause progress indicator
     */
    updatePauseIndicator(data) {
        const indicator = this.elements.pauseIndicator;
        const progressBar = this.elements.pauseProgress;
        const text = this.elements.pauseText;
        
        if (!indicator || !progressBar) return;
        
        if (data.progress > 0 && data.progress < 1) {
            indicator.classList.add('active');
            progressBar.style.width = `${data.progress * 100}%`;
            
            if (text) {
                const remaining = Math.ceil(data.remainingMs / 1000);
                text.textContent = `${remaining}s to complete word`;
            }
        } else {
            indicator.classList.remove('active');
            progressBar.style.width = '0%';
        }
    }
    
    /**
     * Copy the current sentence to clipboard
     */
    async copyTextToClipboard() {
        if (!this.wordFormation) return;
        
        const success = await this.wordFormation.copyToClipboard();
        if (success) {
            this.showNotification('Text copied to clipboard!', 'success');
        } else {
            this.showNotification('No text to copy or clipboard failed', 'warning');
        }
    }
    
    /**
     * Clear all word formation data
     */
    clearWordFormation() {
        if (this.wordFormation) {
            this.wordFormation.clearAll();
            this.showNotification('Text cleared', 'info');
        }
    }
    
    handleFeedback(isCorrect) {
        if (!this.lastDisplayedPrediction) {
            this.showNotification('No prediction to provide feedback for', 'warning');
            return;
        }
        
        console.log(`Feedback: ${this.lastDisplayedPrediction} = ${isCorrect ? 'Correct' : 'Incorrect'}`);
        
        if (isCorrect) {
            // Record correct feedback
            this.feedbackManager.recordCorrect(
                this.lastDisplayedPrediction,
                this.lastDisplayedConfidence,
                this.currentLandmarkSequence
            );
            
            this.showNotification(`Correct! "${this.lastDisplayedPrediction}" confirmed.`, 'success');
            
            // Hide feedback section after a moment
            setTimeout(() => {
                this.hideFeedbackSection();
            }, 1500);
            
        } else {
            // Show correction modal
            this.showCorrectionModal(this.lastDisplayedPrediction, this.lastDisplayedConfidence);
        }
    }
    
    // ================== CORRECTION MODAL ==================
    
    showCorrectionModal(incorrectLabel, confidence) {
        const modal = document.getElementById('correctionModal');
        if (!modal) {
            console.error('Correction modal not found');
            return;
        }
        
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
    
    hideCorrectionModal() {
        const modal = document.getElementById('correctionModal');
        if (modal) {
            modal.style.display = 'none';
        }
        this._pendingCorrection = null;
        this._selectedCorrectLabel = null;
    }
    
    populateSignList() {
        const signList = document.getElementById('signList');
        if (!signList) return;
        
        // Get labels from predictor
        const allSigns = this.predictor.labels || [];
        
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
    
    setupCorrectionModalListeners() {
        if (this._correctionListenersSetup) return;
        this._correctionListenersSetup = true;
        
        document.getElementById('closeCorrectionModal')?.addEventListener('click', () => {
            this.hideCorrectionModal();
        });
        
        document.getElementById('cancelCorrectionBtn')?.addEventListener('click', () => {
            this.hideCorrectionModal();
        });
        
        document.getElementById('submitCorrectionBtn')?.addEventListener('click', () => {
            this.submitCorrection();
        });
        
        const searchInput = document.getElementById('signSearchInput');
        searchInput?.addEventListener('input', (e) => {
            this.filterSignList(e.target.value);
        });
        
        document.getElementById('correctionModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'correctionModal') {
                this.hideCorrectionModal();
            }
        });
    }
    
    filterSignList(query) {
        const signList = document.getElementById('signList');
        if (!signList) return;
        
        const lowerQuery = query.toLowerCase();
        signList.querySelectorAll('.sign-list-item').forEach(item => {
            const sign = item.dataset.sign.toLowerCase();
            item.classList.toggle('hidden', !sign.includes(lowerQuery));
        });
    }
    
    submitCorrection() {
        if (!this._selectedCorrectLabel || !this._pendingCorrection) {
            this.showNotification('Please select the correct sign', 'warning');
            return;
        }
        
        const { incorrectLabel, confidence } = this._pendingCorrection;
        const correctLabel = this._selectedCorrectLabel;
        
        console.log(`Submitting correction: predicted=${incorrectLabel}, correct=${correctLabel}`);
        
        // Record feedback with correction
        this.feedbackManager.recordIncorrect(
            incorrectLabel,
            correctLabel,
            confidence,
            this.currentLandmarkSequence
        );
        
        // Show confirmation
        this.showNotification(
            `Correction saved: "${incorrectLabel}" -> "${correctLabel}"`,
            'success'
        );
        
        // Hide modal
        this.hideCorrectionModal();
        
        // Hide feedback section
        setTimeout(() => {
            this.hideFeedbackSection();
        }, 500);
    }
    
    // ================== HISTORY ==================
    
    // Store history items for feedback
    historyItems = [];
    historyIdCounter = 0;
    
    addToHistory(prediction) {
        if (!this.elements.historyList) return;
        
        // Remove "empty" message if present
        const emptyMessage = this.elements.historyList.querySelector('.history-empty, .history-empty-simple');
        if (emptyMessage) {
            emptyMessage.remove();
        }
        
        // Generate unique ID for this history item
        const itemId = `history-${++this.historyIdCounter}`;
        
        // Store prediction data for later feedback
        const historyData = {
            id: itemId,
            label: prediction.label,
            confidence: prediction.confidence,
            timestamp: Date.now(),
            landmarks: this.currentLandmarkSequence ? [...this.currentLandmarkSequence] : null,
            feedbackGiven: false
        };
        this.historyItems.unshift(historyData);
        
        // Keep only last 20 items in memory
        if (this.historyItems.length > 20) {
            this.historyItems.pop();
        }
        
        // Create history item element
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.id = itemId;
        historyItem.dataset.label = prediction.label;
        historyItem.dataset.confidence = prediction.confidence;
        
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit'
        });
        
        historyItem.innerHTML = `
            <div class="history-item-info">
                <div class="history-item-label">${prediction.label}</div>
                <div class="history-item-time">${timeString}</div>
            </div>
            <div class="history-item-actions">
                <span class="history-item-confidence">${Math.round(prediction.confidence * 100)}%</span>
                <div class="history-feedback-btns">
                    <button class="history-fb-btn history-fb-correct" title="Mark Correct" data-id="${itemId}">&#10003;</button>
                    <button class="history-fb-btn history-fb-incorrect" title="Mark Incorrect" data-id="${itemId}">&#10007;</button>
                </div>
            </div>
        `;
        
        // Add click handlers for feedback buttons
        const correctBtn = historyItem.querySelector('.history-fb-correct');
        const incorrectBtn = historyItem.querySelector('.history-fb-incorrect');
        
        correctBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.handleHistoryFeedback(itemId, true);
        });
        
        incorrectBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.handleHistoryFeedback(itemId, false);
        });
        
        // Add to top of list
        this.elements.historyList.insertBefore(historyItem, this.elements.historyList.firstChild);
        
        // Keep only last 10 visible items
        const items = this.elements.historyList.querySelectorAll('.history-item');
        if (items.length > 10) {
            items[items.length - 1].remove();
        }
    }
    
    /**
     * Handle feedback from history item
     */
    handleHistoryFeedback(itemId, isCorrect) {
        // Find the history data
        const historyData = this.historyItems.find(h => h.id === itemId);
        if (!historyData) {
            this.showNotification('History item not found', 'warning');
            return;
        }
        
        if (historyData.feedbackGiven) {
            this.showNotification('Feedback already given for this sign', 'info');
            return;
        }
        
        const historyElement = document.getElementById(itemId);
        
        if (isCorrect) {
            // Record correct feedback
            this.feedbackManager.recordCorrect(
                historyData.label,
                historyData.confidence,
                historyData.landmarks
            );
            
            // Mark as done in UI
            historyData.feedbackGiven = true;
            if (historyElement) {
                historyElement.classList.add('feedback-given', 'feedback-correct');
                const btns = historyElement.querySelector('.history-feedback-btns');
                if (btns) btns.innerHTML = '<span class="history-fb-done correct">&#10003;</span>';
            }
            
            this.showNotification(`"${historyData.label}" marked as correct`, 'success');
            
        } else {
            // Show correction modal for this history item
            this._pendingHistoryCorrection = historyData;
            this.showCorrectionModal(historyData.label, historyData.confidence);
        }
    }
    
    /**
     * Override submitCorrection to handle history corrections too
     */
    submitCorrectionOriginal = this.submitCorrection;
    
    submitCorrection() {
        if (!this._selectedCorrectLabel) {
            this.showNotification('Please select the correct sign', 'warning');
            return;
        }
        
        // Check if this is a history correction
        if (this._pendingHistoryCorrection) {
            const historyData = this._pendingHistoryCorrection;
            const correctLabel = this._selectedCorrectLabel;
            
            console.log(`History correction: ${historyData.label} -> ${correctLabel}`);
            
            // Record feedback
            this.feedbackManager.recordIncorrect(
                historyData.label,
                correctLabel,
                historyData.confidence,
                historyData.landmarks
            );
            
            // Mark as done in UI
            historyData.feedbackGiven = true;
            const historyElement = document.getElementById(historyData.id);
            if (historyElement) {
                historyElement.classList.add('feedback-given', 'feedback-incorrect');
                const btns = historyElement.querySelector('.history-feedback-btns');
                if (btns) btns.innerHTML = '<span class="history-fb-done incorrect">&#10007;</span>';
            }
            
            this.showNotification(
                `Correction saved: "${historyData.label}" -> "${correctLabel}"`,
                'success'
            );
            
            // Clear pending
            this._pendingHistoryCorrection = null;
            this.hideCorrectionModal();
            return;
        }
        
        // Regular correction (current prediction)
        if (!this._pendingCorrection) {
            this.showNotification('No pending correction', 'warning');
            return;
        }
        
        const { incorrectLabel, confidence } = this._pendingCorrection;
        const correctLabel = this._selectedCorrectLabel;
        
        console.log(`Submitting correction: predicted=${incorrectLabel}, correct=${correctLabel}`);
        
        // Record feedback with correction
        this.feedbackManager.recordIncorrect(
            incorrectLabel,
            correctLabel,
            confidence,
            this.currentLandmarkSequence
        );
        
        // Show confirmation
        this.showNotification(
            `Correction saved: "${incorrectLabel}" -> "${correctLabel}"`,
            'success'
        );
        
        // Hide modal
        this.hideCorrectionModal();
        
        // Hide feedback section
        setTimeout(() => {
            this.hideFeedbackSection();
        }, 500);
    }
    
    clearHistory() {
        if (!this.elements.historyList) return;
        
        this.elements.historyList.innerHTML = `
            <div class="history-empty-simple">
                <span class="english">No signs yet</span>
            </div>
        `;
    }
    
    // ================== STATUS UPDATES ==================
    
    updateSystemStatus(status) {
        if (this.elements.systemStatusValue) {
            this.elements.systemStatusValue.textContent = status;
        }
    }
    
    updateBufferStatus(current, target) {
        if (this.ui) {
            this.ui.updateBuffer(current, target);
        }
    }
    
    updateFPS(fps) {
        if (this.elements.fpsValue) {
            this.elements.fpsValue.textContent = Math.round(fps);
        }
    }
    
    updateConnectionBadge(connected) {
        // Always show as "connected" in browser mode (no server needed)
        if (this.elements.statusDot) {
            this.elements.statusDot.className = 'badge-dot connected';
        }
        if (this.elements.statusText) {
            this.elements.statusText.textContent = 'Browser Mode';
        }
        if (this.elements.connectionBadge) {
            this.elements.connectionBadge.classList.add('browser-mode');
        }
    }
    
    // ================== NOTIFICATIONS ==================
    
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
    
    // ================== LOADING OVERLAY ==================
    
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
    
    // ================== CLEANUP ==================
    
    destroy() {
        // Stop recognition
        if (this.isRecognitionActive) {
            this.stopRecognition();
        }
        
        // Clean up components
        if (this.camera) {
            this.camera.destroy();
        }
        
        if (this.predictor) {
            this.predictor.destroy();
        }
        
        if (this.feedbackManager) {
            this.feedbackManager.destroy();
        }
        
        if (this.visualizer) {
            this.visualizer.destroy();
        }
        
        if (this.ui) {
            this.ui.destroy();
        }
        
        console.log('[OK] App destroyed');
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PSLRecognitionApp;
}

