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

// DEBUG flag - set to false for production to reduce console noise
const PSL_DEBUG = false;

// Debug logger helper
function debugLog(...args) {
    if (PSL_DEBUG) {
        console.log(...args);
    }
}

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
            language: 'urdu',
            // Audio/Visual settings
            soundEnabled: true,
            ttsEnabled: false,
            darkMode: false
        };
        
        // Load saved preferences from localStorage
        this.loadUserPreferences();
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
        
        // CRITICAL FIX: Adaptive FPS for slow devices
        this.adaptiveFPS = {
            enabled: true,
            minFPS: 10,
            maxFPS: 18,
            currentFPS: 18,
            slowFrameCount: 0,
            slowFrameThreshold: 10,  // Reduce FPS after 10 slow frames
            lastAdjustTime: 0,
            adjustCooldownMs: 5000   // Don't adjust more than every 5s
        };
        
        // Components
        this.camera = null;
        this.predictor = null;  // ONNXPredictor (replaces WebSocket)
        this.feedbackManager = null;  // NEW: Browser-based feedback
        this.wordFormation = null;  // NEW: Word formation from letters
        this.visualizer = null;
        this.ui = null;
        
        // Analytics components
        this.sessionTracker = null;
        this.analyticsPanel = null;
        
        // Performance & Debug (NEW!)
        this.performanceMetrics = null;
        this.modelVersionChecker = null;
        this.sessionRecorder = null;
        this.offlineChecker = null;
        
        // Innovation modules
        this.practiceMode = null;
        this.disambiguator = null;
        this.wordShortcuts = null;
        this.pipMode = null;
        
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
            
            // Set up MediaPipe error handler
            this.camera.on('error', (error) => {
                this.handleCameraError(error);
            });
            
            // Set up MediaPipe warning handler
            this.camera.on('mediapipe_warning', (data) => {
                this.showMediaPipeWarning(data);
            });
            
            // Set up MediaPipe loading handler
            this.camera.on('mediapipe_loading', (data) => {
                this.updateLoadingMessage(data.message);
            });
            
            // FIXED: Set up MediaPipe recovery handler to show user feedback
            this.camera.on('mediapipe_recovery', (data) => {
                this.handleMediaPipeRecovery(data);
            });
            
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
                // Add letter to history with feedback options
                this.addLetterToHistory({
                    urduChar: data.letter,
                    romanized: data.romanized,
                    confidence: this.lastDisplayedConfidence || 0.85
                });
            };
            
            this.wordFormation.onWordComplete = (data) => {
                this.showNotification(`Word complete: ${data.word}`, 'success');
                this.playSound('word');
                
                // Speak the completed word if TTS is enabled
                if (this.settings.ttsEnabled && data.word) {
                    this.speak(data.word, 'ur-PK');
                }
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
            
            // Initialize analytics
            this.updateLoadingMessage('Setting up analytics...');
            this.sessionTracker = new SessionTracker();
            this.analyticsPanel = new AnalyticsPanel({
                sessionTracker: this.sessionTracker,
                feedbackManager: this.feedbackManager,
                updateIntervalMs: 1000
            });
            this.analyticsPanel.init();
            
            // Initialize Performance & Debug modules (NEW!)
            this.updateLoadingMessage('Setting up performance monitoring...');
            this.performanceMetrics = new PerformanceMetrics();
            this.performanceMetrics.init();
            
            this.modelVersionChecker = new ModelVersionChecker();
            await this.modelVersionChecker.init();
            
            this.sessionRecorder = new SessionRecorder();
            
            this.offlineChecker = new OfflineStatusChecker();
            this.offlineChecker.init();
            this.offlineChecker.onStatusChange = (status) => this.updateOfflineStatus(status);
            
            // Initial status updates
            this.updateModelVersionUI();
            this.updateOfflineStatusUI();
            
            // Initialize Innovation Modules
            this.updateLoadingMessage('Loading innovation features...');
            await this.initInnovationModules();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Apply saved user preferences (theme, sound, TTS)
            this.applyUserPreferences();
            
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
            
            // Error Overlays (Production-Ready)
            mediapipeErrorOverlay: document.getElementById('mediapipeErrorOverlay'),
            mediapipeErrorMessage: document.getElementById('mediapipeErrorMessage'),
            mediapipeRetryBtn: document.getElementById('mediapipeRetryBtn'),
            modelStatusOverlay: document.getElementById('modelStatusOverlay'),
            modelStatusText: document.getElementById('modelStatusText'),
            modelStatusProgress: document.getElementById('modelStatusProgress'),
            modelRetryBtn: document.getElementById('modelRetryBtn'),
            
            // Camera Error Modal
            cameraErrorModal: document.getElementById('cameraErrorModal'),
            cameraErrorMessage: document.getElementById('cameraErrorMessage'),
            retryCameraBtn: document.getElementById('retryCameraBtn'),
            
            // Performance Metrics (NEW!)
            perfFps: document.getElementById('perfFps'),
            perfInference: document.getElementById('perfInference'),
            perfMediapipe: document.getElementById('perfMediapipe'),
            perfMemory: document.getElementById('perfMemory'),
            perfChartCanvas: document.getElementById('perfChartCanvas'),
            
            // Offline & Model Status (NEW!)
            offlineIndicator: document.getElementById('offlineIndicator'),
            offlineStatus: document.getElementById('offlineStatus'),
            modelIndicator: document.getElementById('modelIndicator'),
            modelVersion: document.getElementById('modelVersion'),
            updateAvailable: document.getElementById('updateAvailable'),
            updateModelBtn: document.getElementById('updateModelBtn'),
            
            // Session Recording (NEW!)
            recordToggleBtn: document.getElementById('recordToggleBtn'),
            recordIcon: document.getElementById('recordIcon'),
            recordText: document.getElementById('recordText'),
            recordingTime: document.getElementById('recordingTime'),
            recordingFrames: document.getElementById('recordingFrames'),
            exportRecordingBtn: document.getElementById('exportRecordingBtn'),
            
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
            deleteWordBtn: document.getElementById('deleteWordBtn'),
            
            // Letter History Elements
            letterHistoryList: document.getElementById('letterHistoryList'),
            clearLettersBtn: document.getElementById('clearLettersBtn'),
            
            // Toggle Buttons (Theme, Sound, TTS)
            themeToggle: document.getElementById('themeToggle'),
            themeIcon: document.getElementById('themeIcon'),
            soundToggle: document.getElementById('soundToggle'),
            soundIcon: document.getElementById('soundIcon'),
            ttsToggle: document.getElementById('ttsToggle'),
            ttsIcon: document.getElementById('ttsIcon'),
            speakTextBtn: document.getElementById('speakTextBtn')
        };
        
        // Letter history tracking
        this.letterHistory = [];
        this.letterIdCounter = 0;
        
        // Debug: Log if letter history elements are found
        console.log('[DOM] letterHistoryList found:', !!this.elements.letterHistoryList);
        console.log('[DOM] clearLettersBtn found:', !!this.elements.clearLettersBtn);
    }
    
    /**
     * Initialize all innovation modules (Practice, Disambiguation, Shortcuts, PiP)
     */
    async initInnovationModules() {
        try {
            // 1. Practice Mode
            if (typeof PracticeMode !== 'undefined') {
                this.practiceMode = new PracticeMode({
                    imagesBasePath: 'assets/signs/',
                    manifestPath: 'assets/signs/manifest.json',
                    targetHoldTimeMs: 2000
                });
                await this.practiceMode.init();
                
                // Set weak signs from feedback data
                if (this.feedbackManager) {
                    const confusionMatrix = this.feedbackManager.getConfusionMatrix();
                    this.practiceMode.analyzeWeakSigns(confusionMatrix);
                }
                
                // Practice mode success callback
                this.practiceMode.onSuccess = (data) => {
                    this.showNotification(`✓ Correct! Streak: ${data.streak}`, 'success');
                    this.playSound('success');
                };
                
                console.log('[Innovation] Practice Mode initialized');
            }
            
            // 2. Disambiguation
            if (typeof SignDisambiguator !== 'undefined') {
                this.disambiguator = new SignDisambiguator({
                    imagesBasePath: 'assets/signs/',
                    ambiguityThreshold: 0.15,
                    maxAlternatives: 3,
                    autoDismissMs: 8000
                });
                this.disambiguator.init();
                
                // Update confusion pairs from feedback
                if (this.feedbackManager) {
                    const confusionMatrix = this.feedbackManager.getConfusionMatrix();
                    this.disambiguator.updateConfusionPairs(confusionMatrix);
                }
                
                // Selection callback
                this.disambiguator.onSelection = (data) => {
                    // User selected a different sign - use that instead
                    if (data.selected !== this.lastDisplayedPrediction) {
                        this.displayPrediction(data.selected, data.confidence);
                        if (this.wordFormation) {
                            this.wordFormation.processPrediction({
                                label: data.selected,
                                confidence: data.confidence,
                                timestamp: Date.now()
                            });
                        }
                    }
                };
                
                console.log('[Innovation] Disambiguation initialized');
            }
            
            // 3. Word Shortcuts
            if (typeof WordShortcuts !== 'undefined') {
                this.wordShortcuts = new WordShortcuts({
                    maxCustomShortcuts: 20
                });
                await this.wordShortcuts.init();
                
                // Word insert callback
                this.wordShortcuts.onWordInsert = (data) => {
                    if (this.wordFormation) {
                        // Insert the word directly into the sentence
                        this.wordFormation.insertWord(data.urdu);
                        this.showNotification(`Added: ${data.urdu}`, 'success');
                        this.playSound('word');
                    }
                };
                
                console.log('[Innovation] Word Shortcuts initialized');
            }
            
            // 4. PiP Mode
            if (typeof PiPMode !== 'undefined') {
                this.pipMode = new PiPMode({
                    defaultPosition: 'bottom-right'
                });
                // FIXED: Use correct element IDs from index_browser.html
                const videoEl = document.getElementById('webcam');
                const canvasEl = document.getElementById('overlay-canvas');
                this.pipMode.init(videoEl, canvasEl);
                
                // Toggle recognition callback
                this.pipMode.setToggleCallback(() => {
                    if (this.isRecognitionActive) {
                        this.stopRecognition();
                    } else {
                        this.startRecognition();
                    }
                });
                
                console.log('[Innovation] PiP Mode initialized');
            }
            
            console.log('[Innovation] All modules initialized successfully');
            
        } catch (error) {
            console.warn('[Innovation] Error initializing modules:', error);
            // Don't fail - innovation modules are optional enhancements
        }
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
        
        // Window resize - OPTIMIZED: Debounced to prevent excessive calls
        let resizeTimeout = null;
        window.addEventListener('resize', () => {
            // Clear previous timeout
            if (resizeTimeout) {
                clearTimeout(resizeTimeout);
            }
            // Debounce: Only handle resize after 150ms of no resize events
            resizeTimeout = setTimeout(() => {
                if (this.visualizer) {
                    this.visualizer.handleResize();
                }
            }, 150);
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
        
        // Clear letter history button
        if (this.elements.clearLettersBtn) {
            this.elements.clearLettersBtn.addEventListener('click', () => {
                this.clearLetterHistory();
            });
        }
        
        // CRITICAL FIX: Event delegation for history items (prevents memory leak)
        if (this.elements.historyList) {
            this.elements.historyList.addEventListener('click', (e) => {
                const btn = e.target.closest('.history-fb-btn');
                if (!btn) return;
                e.stopPropagation();
                const itemId = btn.dataset.id;
                const isCorrect = btn.classList.contains('history-fb-correct');
                this.handleHistoryFeedback(itemId, isCorrect);
            });
        }
        
        // CRITICAL FIX: Event delegation for letter history (prevents memory leak)
        if (this.elements.letterHistoryList) {
            this.elements.letterHistoryList.addEventListener('click', (e) => {
                const btn = e.target.closest('.letter-fb-btn');
                if (!btn) return;
                e.stopPropagation();
                const letterId = btn.dataset.id;
                const isCorrect = btn.classList.contains('letter-fb-correct');
                this.handleLetterFeedback(letterId, isCorrect);
            });
        }
        
        // Theme Toggle
        if (this.elements.themeToggle) {
            this.elements.themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
        
        // Sound Toggle
        if (this.elements.soundToggle) {
            this.elements.soundToggle.addEventListener('click', () => {
                this.toggleSound();
            });
        }
        
        // TTS Toggle
        if (this.elements.ttsToggle) {
            this.elements.ttsToggle.addEventListener('click', () => {
                this.toggleTTS();
            });
        }
        
        // Speak Text Button
        if (this.elements.speakTextBtn) {
            this.elements.speakTextBtn.addEventListener('click', () => {
                this.speakCurrentText();
            });
        }
        
        // Analytics Toggle
        const analyticsToggle = document.getElementById('analyticsToggle');
        const analyticsClose = document.getElementById('analyticsClose');
        
        if (analyticsToggle) {
            analyticsToggle.addEventListener('click', () => {
                if (this.analyticsPanel) {
                    this.analyticsPanel.toggle();
                    analyticsToggle.classList.toggle('active', this.analyticsPanel.isVisible);
                }
            });
        }
        
        if (analyticsClose) {
            analyticsClose.addEventListener('click', () => {
                if (this.analyticsPanel) {
                    this.analyticsPanel.hide();
                    if (analyticsToggle) {
                        analyticsToggle.classList.remove('active');
                    }
                }
            });
        }
        
        // Innovation Module Toggles
        this.setupInnovationListeners();
        
        // Error Recovery Buttons
        this.setupErrorRecoveryListeners();
    }
    
    /**
     * Set up event listeners for error recovery buttons
     */
    setupErrorRecoveryListeners() {
        // MediaPipe retry button
        if (this.elements.mediapipeRetryBtn) {
            this.elements.mediapipeRetryBtn.addEventListener('click', () => {
                this.retryMediaPipe();
            });
        }
        
        // Model retry button
        if (this.elements.modelRetryBtn) {
            this.elements.modelRetryBtn.addEventListener('click', () => {
                this.retryModelLoading();
            });
        }
        
        // Camera retry button
        if (this.elements.retryCameraBtn) {
            this.elements.retryCameraBtn.addEventListener('click', () => {
                this.retryCamera();
            });
        }
        
        // Close camera error modal by clicking outside
        if (this.elements.cameraErrorModal) {
            this.elements.cameraErrorModal.addEventListener('click', (e) => {
                if (e.target === this.elements.cameraErrorModal) {
                    this.hideCameraErrorModal();
                }
            });
        }
        
        // Recording controls (NEW!)
        if (this.elements.recordToggleBtn) {
            this.elements.recordToggleBtn.addEventListener('click', () => {
                this.toggleRecording();
            });
        }
        
        if (this.elements.exportRecordingBtn) {
            this.elements.exportRecordingBtn.addEventListener('click', () => {
                this.exportRecording();
            });
        }
        
        // Model update button (NEW!)
        if (this.elements.updateModelBtn) {
            this.elements.updateModelBtn.addEventListener('click', () => {
                this.showNotification('Model updates coming soon!', 'info');
            });
        }
    }
    
    /**
     * Set up event listeners for innovation modules
     */
    setupInnovationListeners() {
        // Practice Mode Toggle
        const practiceToggle = document.getElementById('practiceToggle');
        const practiceClose = document.getElementById('practiceClose');
        const practiceSkip = document.getElementById('practiceSkip');
        const practiceWeakSigns = document.getElementById('practiceWeakSigns');
        const practiceRandom = document.getElementById('practiceRandom');
        
        if (practiceToggle) {
            practiceToggle.addEventListener('click', () => {
                if (this.practiceMode) {
                    this.practiceMode.toggle();
                    practiceToggle.classList.toggle('active', this.practiceMode.isActive);
                }
            });
        }
        
        if (practiceClose) {
            practiceClose.addEventListener('click', () => {
                if (this.practiceMode) {
                    this.practiceMode.stop();
                    if (practiceToggle) practiceToggle.classList.remove('active');
                }
            });
        }
        
        if (practiceSkip) {
            practiceSkip.addEventListener('click', () => {
                if (this.practiceMode) this.practiceMode.onSignSkip();
            });
        }
        
        if (practiceWeakSigns) {
            practiceWeakSigns.addEventListener('click', () => {
                if (this.practiceMode) this.practiceMode.selectWeakSign();
            });
        }
        
        if (practiceRandom) {
            practiceRandom.addEventListener('click', () => {
                if (this.practiceMode) this.practiceMode.selectRandomSign();
            });
        }
        
        // Word Shortcuts Toggle
        const shortcutsToggle = document.getElementById('shortcutsToggle');
        const shortcutsClose = document.getElementById('shortcutsClose');
        
        if (shortcutsToggle) {
            shortcutsToggle.addEventListener('click', () => {
                if (this.wordShortcuts) {
                    this.wordShortcuts.toggle();
                    shortcutsToggle.classList.toggle('active', this.wordShortcuts.isVisible);
                }
            });
        }
        
        if (shortcutsClose) {
            shortcutsClose.addEventListener('click', () => {
                if (this.wordShortcuts) {
                    this.wordShortcuts.hide();
                    if (shortcutsToggle) shortcutsToggle.classList.remove('active');
                }
            });
        }
        
        // PiP Mode Toggle
        const pipToggle = document.getElementById('pipToggle');
        
        if (pipToggle) {
            pipToggle.addEventListener('click', async () => {
                if (this.pipMode) {
                    await this.pipMode.toggle();
                    pipToggle.classList.toggle('active', this.pipMode.isActive);
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
            debugLog('Starting recognition (Browser-Only Mode)...');
            
            // Reset state
            this.lastDisplayedPrediction = null;
            this.lastDisplayedConfidence = 0;
            this.lastFrameSentAt = 0;
            this.predictor.clearBuffer();
            
            // Start session tracking for analytics
            if (this.sessionTracker) {
                this.sessionTracker.startSession();
            }
            
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
            
            // PERFORMANCE FIX: Start visualizer animation only when recognition is active
            if (this.visualizer && !this.visualizer.isActive) {
                this.visualizer.startAnimation();
            }
            
            // FIXED: Restart word formation pause detection when recognition starts
            if (this.wordFormation) {
                this.wordFormation.startPauseDetection();
            }
            
            // Update UI
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.updateSystemStatus('Recognizing...');
            
            // Clear previous prediction
            this.clearPredictionDisplay();
            
            debugLog('[OK] Recognition started');
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
            debugLog('Stopping recognition...');
            
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
            
            // Clear and STOP visualizer to save CPU
            if (this.visualizer) {
                this.visualizer.updateKeypoints(null);
                this.visualizer.clear();
                // PERFORMANCE FIX: Stop animation loop when not recognizing
                this.visualizer.stopAnimation();
            }
            
            // FIXED: Stop word formation pause detection to save resources
            if (this.wordFormation) {
                this.wordFormation.stopPauseDetection();
            }
            
            // Update UI
            this.isRecognitionActive = false;
            this.elements.startBtn.disabled = false;
            this.elements.stopBtn.disabled = true;
            this.updateSystemStatus('Stopped');
            
            // Hide feedback buttons
            this.hideFeedbackSection();
            
            debugLog('[OK] Recognition stopped');
            this.showNotification('Recognition stopped', 'info');
            
        } catch (error) {
            console.error('Failed to stop recognition:', error);
            this.showNotification(`Failed to stop: ${error.message}`, 'error');
        }
    }
    
    resetSystem() {
        try {
            debugLog('Resetting system...');
            
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
            
            // Clear letter history
            this.clearLetterHistory();
            
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
                // DEAD-ZONE FILTER: Skip if hand is in bottom 20% of screen (transitioning)
                // Wrist Y coordinate is at index 1 (x=0, y=1, z=2 for first landmark)
                const wristY = data.landmarks[1];
                if (wristY > 0.80) {
                    // Hand is in "dead zone" (bottom of screen) - likely transitioning
                    // Don't add to buffer to prevent phantom predictions
                    return;
                }
                
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
     * Start the prediction loop (OPTIMIZED: adaptive rate based on buffer state)
     * PERFORMANCE FIX: Uses setTimeout for proper async timing instead of rAF
     * This prevents main thread blocking and allows proper inference pacing
     */
    startPredictionLoop() {
        this.stopPredictionLoop();  // Clear any existing
        
        // PERFORMANCE FIX: Track state for smart skipping
        this._lastBufferSize = 0;
        this._lastPredictionTime = 0;
        this._consecutiveSkips = 0;
        this._predictionLoopRunning = true;
        
        // Use setTimeout-based loop for proper async pacing (NOT rAF)
        // This prevents prediction calls from blocking the main thread
        const runPredictionCycle = async () => {
            if (!this.isRecognitionActive || !this._predictionLoopRunning) {
                return;
            }
            
            // CRITICAL FIX: Skip inference when tab is hidden (browser throttles anyway)
            if (document.hidden) {
                // Schedule slower check when hidden
                this.predictionLoopTimeout = setTimeout(runPredictionCycle, 500);
                return;
            }
            
            const now = performance.now();
            
            // STABILITY FIX: Guard against null predictor
            if (!this.predictor || !this.predictor.isReady) {
                this.predictionLoopTimeout = setTimeout(runPredictionCycle, 200);
                return;
            }
            
            // Calculate adaptive interval based on buffer state
            const bufferSize = this.predictor.frameBuffer?.length || 0;
            const minFrames = this.predictor.config?.minPredictionFrames || 12;
            
            // Adaptive interval:
            // - 150ms when collecting (buffer < 50%)
            // - 100ms when almost ready (buffer 50-100%)
            // - 83ms when ready for predictions
            let targetInterval = 150;
            if (bufferSize >= minFrames) {
                targetInterval = 83;  // Ready for predictions
            } else if (bufferSize >= minFrames * 0.5) {
                targetInterval = 100; // Getting close
            }
            
            // Skip prediction if buffer hasn't changed and not ready
            if (bufferSize === this._lastBufferSize && bufferSize < minFrames) {
                this._consecutiveSkips++;
                // Only update UI occasionally when skipping
                if (this._consecutiveSkips % 5 === 0) {
                    this.updateBufferStatus(bufferSize, this.predictor.config.slidingWindowSize);
                }
                // Schedule next cycle with setTimeout (non-blocking)
                this.predictionLoopTimeout = setTimeout(runPredictionCycle, targetInterval);
                return;
            }
            
            this._lastBufferSize = bufferSize;
            this._lastPredictionTime = now;
            this._consecutiveSkips = 0;
            
            try {
                const startTime = performance.now();
                const result = await this.predictor.predict(true);
                const predTime = performance.now() - startTime;
                
                // Handle inference_busy status (overlapping call prevented)
                if (result.status === 'inference_busy') {
                    // Retry sooner since inference is still running
                    this.predictionLoopTimeout = setTimeout(runPredictionCycle, 50);
                    return;
                }
                
                // Track prediction performance
                this.performanceMetrics.avgPredictionTime = 
                    this.performanceMetrics.avgPredictionTime * 0.9 + predTime * 0.1;
                
                // CRITICAL: Adaptive FPS for slow devices
                this.checkAdaptiveFPS(predTime);
                
                this.handlePredictionResult(result);
                
            } catch (error) {
                console.error('Prediction loop error:', error);
            }
            
            // Schedule next cycle with setTimeout (allows main thread to breathe)
            if (this._predictionLoopRunning) {
                this.predictionLoopTimeout = setTimeout(runPredictionCycle, targetInterval);
            }
        };
        
        // Start the loop with setTimeout
        this.predictionLoopTimeout = setTimeout(runPredictionCycle, 50);
        
        console.log('[PERF] Prediction loop started with adaptive rate (83-150ms, setTimeout-based)');
    }
    
    /**
     * Stop the prediction loop
     */
    stopPredictionLoop() {
        // Mark loop as stopped first
        this._predictionLoopRunning = false;
        
        // Clear timeout-based loop
        if (this.predictionLoopTimeout) {
            clearTimeout(this.predictionLoopTimeout);
            this.predictionLoopTimeout = null;
        }
        
        // Legacy: Clear rAF-based loop if exists
        if (this.predictionLoopInterval) {
            cancelAnimationFrame(this.predictionLoopInterval);
            this.predictionLoopInterval = null;
        }
        
        // Reset tracking variables
        this._lastBufferSize = 0;
        this._lastPredictionTime = 0;
        this._consecutiveSkips = 0;
    }
    
    /**
     * CRITICAL: Check if device is struggling and adapt FPS accordingly
     * This prevents lag and freezes on low-end laptops
     */
    checkAdaptiveFPS(predictionTimeMs) {
        if (!this.adaptiveFPS.enabled) return;
        
        const now = Date.now();
        
        // A frame is "slow" if prediction takes > 150ms (should be <100ms ideally)
        const isSlowFrame = predictionTimeMs > 150;
        
        if (isSlowFrame) {
            this.adaptiveFPS.slowFrameCount++;
        } else {
            // Decay slow frame count over time
            this.adaptiveFPS.slowFrameCount = Math.max(0, this.adaptiveFPS.slowFrameCount - 0.5);
        }
        
        // Check if we should adjust FPS (with cooldown)
        if (now - this.adaptiveFPS.lastAdjustTime < this.adaptiveFPS.adjustCooldownMs) {
            return;
        }
        
        // Too many slow frames -> reduce FPS
        if (this.adaptiveFPS.slowFrameCount >= this.adaptiveFPS.slowFrameThreshold) {
            const newFPS = Math.max(this.adaptiveFPS.minFPS, this.adaptiveFPS.currentFPS - 2);
            if (newFPS !== this.adaptiveFPS.currentFPS) {
                console.warn(`[ADAPTIVE] Device struggling - reducing FPS: ${this.adaptiveFPS.currentFPS} -> ${newFPS}`);
                this.adaptiveFPS.currentFPS = newFPS;
                this.settings.frameRate = newFPS;
                this.frameIntervalMs = 1000 / newFPS;
                
                // Update camera frame rate too
                if (this.camera) {
                    this.camera.frameRate = newFPS;
                }
                
                this.adaptiveFPS.lastAdjustTime = now;
                this.adaptiveFPS.slowFrameCount = 0;
                
                this.showNotification(`Performance mode: ${newFPS} FPS`, 'warning');
            }
        }
        // Performing well -> try to increase FPS (gradual)
        else if (this.adaptiveFPS.slowFrameCount === 0 && 
                 this.adaptiveFPS.currentFPS < this.adaptiveFPS.maxFPS &&
                 predictionTimeMs < 80) {
            const newFPS = Math.min(this.adaptiveFPS.maxFPS, this.adaptiveFPS.currentFPS + 1);
            if (newFPS !== this.adaptiveFPS.currentFPS) {
                console.log(`[ADAPTIVE] Performance good - increasing FPS: ${this.adaptiveFPS.currentFPS} -> ${newFPS}`);
                this.adaptiveFPS.currentFPS = newFPS;
                this.settings.frameRate = newFPS;
                this.frameIntervalMs = 1000 / newFPS;
                
                if (this.camera) {
                    this.camera.frameRate = newFPS;
                }
                
                this.adaptiveFPS.lastAdjustTime = now;
            }
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
        if (result.status === 'inference_busy') {
            // Inference is still running - skip update, don't change UI
            return;
        } else if (result.status === 'collecting_frames') {
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
        
        // FIXED: Update FPS using actual camera stats, not inference rate
        if (this.camera && this.camera.cameraStats) {
            const cameraStats = this.camera.cameraStats;
            // Use actual frames processed by camera, not inference rate
            const actualFps = cameraStats.isActive ? this.settings.frameRate : 0;
            this.updateFPS(actualFps);
        }
        
        // Update performance metrics (NEW!)
        if (result.predictionTimeMs) {
            this.updatePerformanceMetrics(result.predictionTimeMs, result.mediapipeTimeMs || null);
        }
    }
    
    /**
     * Handle new stable prediction (triggered by predictor callback)
     */
    handleNewPrediction(prediction) {
        debugLog('New prediction:', prediction);
        
        // Track prediction in analytics
        if (this.sessionTracker) {
            this.sessionTracker.recordPrediction(prediction.label, prediction.confidence);
        }
        
        // Record for debugging if recording is active (NEW!)
        this.recordPredictionFrame(prediction, this.currentLandmarkSequence);
        
        // Check for disambiguation (low confidence or ambiguous predictions)
        if (this.disambiguator && prediction.allPredictions && prediction.allPredictions.length >= 2) {
            const wasAmbiguous = this.disambiguator.checkAmbiguity(prediction.allPredictions);
            if (wasAmbiguous) {
                // Disambiguation UI shown - wait for user selection
                console.log('[Disambiguation] Showing options for ambiguous prediction');
                return; // Don't process further until user selects
            }
        }
        
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
        
        // Feed to practice mode if active
        if (this.practiceMode && this.practiceMode.isActive) {
            this.practiceMode.processPrediction({
                label: prediction.label,
                confidence: prediction.confidence
            });
        }
        
        // Update PiP display if active
        if (this.pipMode && this.pipMode.isActive) {
            this.pipMode.updatePrediction(prediction.label, prediction.confidence);
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
        
        // Play sound effect for new letter recognition
        if (label !== this.lastDisplayedPrediction) {
            this.playSound('letter');
            
            // Auto-speak the sign if TTS is enabled
            this.speakSign(label);
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
        
        // Track feedback in analytics
        if (this.sessionTracker) {
            this.sessionTracker.recordFeedback(this.lastDisplayedPrediction, isCorrect);
        }
        
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
        // MEMORY FIX: Don't store landmarks - they're ~27KB per item and rarely used
        const historyData = {
            id: itemId,
            label: prediction.label,
            confidence: prediction.confidence,
            timestamp: Date.now(),
            landmarks: null,  // MEMORY: Removed landmark storage (saves ~27KB per item)
            feedbackGiven: false
        };
        this.historyItems.unshift(historyData);
        
        // MEMORY FIX: Keep only last 10 items in memory (reduced from 15)
        while (this.historyItems.length > 10) {
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
        
        // NOTE: Event handlers use delegation (setupEventListeners) - no individual listeners needed
        
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
        
        // Check if this is a letter history correction
        if (this._pendingLetterCorrection) {
            const letterData = this._pendingLetterCorrection;
            const correctLabel = this._selectedCorrectLabel;
            
            console.log(`Letter correction: ${letterData.romanized} -> ${correctLabel}`);
            
            // Record feedback
            this.feedbackManager.recordIncorrect(
                letterData.romanized,
                correctLabel,
                letterData.confidence,
                letterData.landmarks
            );
            
            // Mark as done in UI
            letterData.feedbackGiven = true;
            const letterElement = document.getElementById(letterData.id);
            if (letterElement) {
                letterElement.classList.add('feedback-given', 'feedback-incorrect');
                const btns = letterElement.querySelector('.letter-feedback-btns');
                if (btns) btns.innerHTML = '<span class="letter-fb-done incorrect">✗</span>';
            }
            
            this.showNotification(
                `Letter correction saved: "${letterData.romanized}" -> "${correctLabel}"`,
                'success'
            );
            
            // Clear pending
            this._pendingLetterCorrection = null;
            this.hideCorrectionModal();
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
    
    // ================== LETTER HISTORY ==================
    
    /**
     * Add a letter to the letter history with feedback buttons
     */
    addLetterToHistory(letterData) {
        console.log('[LetterHistory] addLetterToHistory called:', letterData);
        
        if (!this.elements.letterHistoryList) {
            console.error('[LetterHistory] letterHistoryList element NOT found! Trying to find it now...');
            // Try to find it again (in case DOM wasn't ready earlier)
            this.elements.letterHistoryList = document.getElementById('letterHistoryList');
            if (!this.elements.letterHistoryList) {
                console.error('[LetterHistory] Still not found. Check HTML for id="letterHistoryList"');
                return;
            }
            console.log('[LetterHistory] Found it on retry!');
        }
        
        // Remove empty message if present
        const emptyMessage = this.elements.letterHistoryList.querySelector('.letter-history-empty');
        if (emptyMessage) {
            emptyMessage.remove();
        }
        
        // Generate unique ID for this letter
        const letterId = `letter-${++this.letterIdCounter}`;
        
        // Store letter data
        // MEMORY FIX: Don't store full landmark sequences (large arrays) - save memory
        const letterRecord = {
            id: letterId,
            urduChar: letterData.urduChar,
            romanized: letterData.romanized,
            confidence: letterData.confidence,
            timestamp: Date.now(),
            landmarks: null,  // MEMORY: Removed landmark storage - too large
            feedbackGiven: false
        };
        this.letterHistory.push(letterRecord);
        
        // MEMORY FIX: Keep only last 20 letters in memory (reduced from 30)
        while (this.letterHistory.length > 20) {
            const oldest = this.letterHistory.shift();
            const oldElement = document.getElementById(oldest.id);
            if (oldElement) oldElement.remove();
        }
        
        // Create letter history item element
        const letterItem = document.createElement('div');
        letterItem.className = 'letter-history-item';
        letterItem.id = letterId;
        letterItem.dataset.romanized = letterData.romanized;
        
        const confidencePercent = Math.round(letterData.confidence * 100);
        
        letterItem.innerHTML = `
            <span class="letter-char">${letterData.urduChar}</span>
            <span class="letter-label">${letterData.romanized}</span>
            <span class="letter-confidence">${confidencePercent}%</span>
            <div class="letter-feedback-btns">
                <button class="letter-fb-btn letter-fb-correct" title="Correct" data-id="${letterId}">✓</button>
                <button class="letter-fb-btn letter-fb-incorrect" title="Incorrect" data-id="${letterId}">✗</button>
            </div>
        `;
        
        // NOTE: Event handlers use delegation (setupEventListeners) - no individual listeners needed
        
        // Add to the list (at the end for RTL display order)
        this.elements.letterHistoryList.appendChild(letterItem);
        
        // Scroll to show the newest letter
        this.elements.letterHistoryList.scrollLeft = this.elements.letterHistoryList.scrollWidth;
        
        console.log(`[LetterHistory] Added: ${letterData.romanized} -> ${letterData.urduChar}`);
    }
    
    /**
     * Handle feedback for a specific letter in history
     */
    handleLetterFeedback(letterId, isCorrect) {
        // Find the letter data
        const letterData = this.letterHistory.find(l => l.id === letterId);
        if (!letterData) {
            this.showNotification('Letter not found', 'warning');
            return;
        }
        
        if (letterData.feedbackGiven) {
            this.showNotification('Feedback already given for this letter', 'info');
            return;
        }
        
        const letterElement = document.getElementById(letterId);
        
        if (isCorrect) {
            // Record correct feedback
            this.feedbackManager.recordCorrect(
                letterData.romanized,
                letterData.confidence,
                letterData.landmarks
            );
            
            // Mark as done in UI
            letterData.feedbackGiven = true;
            if (letterElement) {
                letterElement.classList.add('feedback-given', 'feedback-correct');
                const btns = letterElement.querySelector('.letter-feedback-btns');
                if (btns) btns.innerHTML = '<span class="letter-fb-done correct">✓</span>';
            }
            
            this.showNotification(`"${letterData.urduChar}" marked correct`, 'success');
            
        } else {
            // Show correction modal for this letter
            this._pendingLetterCorrection = letterData;
            this.showCorrectionModal(letterData.romanized, letterData.confidence);
        }
    }
    
    /**
     * Clear all letter history
     */
    clearLetterHistory() {
        this.letterHistory = [];
        this.letterIdCounter = 0;
        
        if (this.elements.letterHistoryList) {
            this.elements.letterHistoryList.innerHTML = `
                <div class="letter-history-empty">
                    <span class="english">Letters will appear here</span>
                </div>
            `;
        }
        
        this.showNotification('Letter history cleared', 'info');
    }
    
    // ================== STATUS UPDATES ==================
    
    updateSystemStatus(status) {
        if (this.elements.systemStatusValue) {
            this.elements.systemStatusValue.textContent = status;
        }
    }
    
    updateBufferStatus(current, target) {
        // PERFORMANCE: Throttle buffer status updates (max 5 per second)
        const now = Date.now();
        if (this._lastBufferUIUpdate && now - this._lastBufferUIUpdate < 200) {
            return; // Skip this update
        }
        this._lastBufferUIUpdate = now;
        
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
    
    // ================== THEME, SOUND & TTS ==================
    
    /**
     * Load saved user preferences from localStorage
     */
    loadUserPreferences() {
        try {
            const saved = localStorage.getItem('psl_user_preferences');
            if (saved) {
                const prefs = JSON.parse(saved);
                this.settings.soundEnabled = prefs.soundEnabled ?? true;
                this.settings.ttsEnabled = prefs.ttsEnabled ?? false;
                this.settings.darkMode = prefs.darkMode ?? false;
            }
        } catch (e) {
            console.warn('Could not load user preferences:', e);
        }
    }
    
    /**
     * Save user preferences to localStorage
     * STABILITY FIX: Added quota protection
     */
    saveUserPreferences() {
        try {
            const data = JSON.stringify({
                soundEnabled: this.settings.soundEnabled,
                ttsEnabled: this.settings.ttsEnabled,
                darkMode: this.settings.darkMode
            });
            localStorage.setItem('psl_user_preferences', data);
        } catch (e) {
            console.warn('Could not save user preferences:', e);
            // Handle quota exceeded
            if (e.name === 'QuotaExceededError') {
                console.warn('LocalStorage full - attempting cleanup');
                try {
                    localStorage.removeItem('psl_feedback_history');
                    localStorage.removeItem('psl_session_data');
                } catch (cleanupErr) {
                    // Ignore cleanup errors
                }
            }
        }
    }
    
    /**
     * Apply saved preferences to UI after DOM is ready
     */
    applyUserPreferences() {
        // Apply dark mode
        if (this.settings.darkMode) {
            document.documentElement.setAttribute('data-theme', 'dark');
            if (this.elements.themeIcon) this.elements.themeIcon.textContent = '☀️';
            if (this.elements.themeToggle) this.elements.themeToggle.classList.add('active');
        }
        
        // Apply sound toggle state
        if (this.elements.soundToggle) {
            this.elements.soundToggle.classList.toggle('active', this.settings.soundEnabled);
            if (this.elements.soundIcon) {
                this.elements.soundIcon.textContent = this.settings.soundEnabled ? '🔊' : '🔇';
            }
        }
        
        // Apply TTS toggle state
        if (this.elements.ttsToggle) {
            this.elements.ttsToggle.classList.toggle('active', this.settings.ttsEnabled);
            if (this.elements.ttsIcon) {
                this.elements.ttsIcon.textContent = this.settings.ttsEnabled ? '🗣️' : '🤐';
            }
        }
    }
    
    /**
     * Toggle dark/light theme
     */
    toggleTheme() {
        this.settings.darkMode = !this.settings.darkMode;
        
        if (this.settings.darkMode) {
            document.documentElement.setAttribute('data-theme', 'dark');
            if (this.elements.themeIcon) this.elements.themeIcon.textContent = '☀️';
            if (this.elements.themeToggle) this.elements.themeToggle.classList.add('active');
        } else {
            document.documentElement.removeAttribute('data-theme');
            if (this.elements.themeIcon) this.elements.themeIcon.textContent = '🌙';
            if (this.elements.themeToggle) this.elements.themeToggle.classList.remove('active');
        }
        
        this.saveUserPreferences();
        this.playSound('click');
    }
    
    /**
     * Toggle sound effects on/off
     */
    toggleSound() {
        this.settings.soundEnabled = !this.settings.soundEnabled;
        
        if (this.elements.soundToggle) {
            this.elements.soundToggle.classList.toggle('active', this.settings.soundEnabled);
        }
        if (this.elements.soundIcon) {
            this.elements.soundIcon.textContent = this.settings.soundEnabled ? '🔊' : '🔇';
        }
        
        this.saveUserPreferences();
        
        // Play confirmation sound if enabling
        if (this.settings.soundEnabled) {
            this.playSound('click');
        }
    }
    
    /**
     * Toggle text-to-speech auto-speak on/off
     */
    toggleTTS() {
        this.settings.ttsEnabled = !this.settings.ttsEnabled;
        
        if (this.elements.ttsToggle) {
            this.elements.ttsToggle.classList.toggle('active', this.settings.ttsEnabled);
        }
        if (this.elements.ttsIcon) {
            this.elements.ttsIcon.textContent = this.settings.ttsEnabled ? '🗣️' : '🤐';
        }
        
        this.saveUserPreferences();
        this.playSound('click');
        
        // Announce state change
        if (this.settings.ttsEnabled) {
            this.speak('Auto-speak enabled');
        }
    }
    
    /**
     * Play a sound effect
     * @param {string} type - Type of sound: 'success', 'error', 'click', 'letter'
     */
    playSound(type) {
        if (!this.settings.soundEnabled) return;
        
        // Create audio context on demand (for mobile compatibility)
        if (!this._audioContext) {
            try {
                this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
            } catch (e) {
                console.warn('Web Audio API not supported');
                return;
            }
        }
        
        const ctx = this._audioContext;
        
        // Resume audio context if suspended (mobile browsers require this after user interaction)
        if (ctx.state === 'suspended') {
            ctx.resume().catch(e => console.warn('Could not resume audio context:', e));
        }
        const now = ctx.currentTime;
        
        // Create oscillator for simple sound effects
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        
        switch (type) {
            case 'success':
                // Pleasant ascending tone
                osc.type = 'sine';
                osc.frequency.setValueAtTime(440, now);
                osc.frequency.linearRampToValueAtTime(660, now + 0.1);
                gain.gain.setValueAtTime(0.15, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
                osc.start(now);
                osc.stop(now + 0.2);
                break;
                
            case 'error':
                // Low warning tone
                osc.type = 'sine';
                osc.frequency.setValueAtTime(200, now);
                osc.frequency.linearRampToValueAtTime(150, now + 0.15);
                gain.gain.setValueAtTime(0.15, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
                osc.start(now);
                osc.stop(now + 0.2);
                break;
                
            case 'click':
                // Quick click
                osc.type = 'sine';
                osc.frequency.setValueAtTime(800, now);
                gain.gain.setValueAtTime(0.1, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.05);
                osc.start(now);
                osc.stop(now + 0.05);
                break;
                
            case 'letter':
                // Soft notification for new letter
                osc.type = 'sine';
                osc.frequency.setValueAtTime(523, now); // C5
                gain.gain.setValueAtTime(0.08, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);
                osc.start(now);
                osc.stop(now + 0.1);
                break;
                
            case 'word':
                // Two-tone word completion sound
                osc.type = 'sine';
                osc.frequency.setValueAtTime(523, now);      // C5
                osc.frequency.setValueAtTime(659, now + 0.1); // E5
                gain.gain.setValueAtTime(0.1, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
                osc.start(now);
                osc.stop(now + 0.2);
                break;
        }
    }
    
    /**
     * Speak text using Web Speech API
     * @param {string} text - Text to speak
     * @param {string} lang - Language code (default: 'ur-PK' for Urdu)
     */
    speak(text, lang = 'en-US') {
        if (!text || !window.speechSynthesis) return;
        
        // Cancel any ongoing speech
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang;
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 0.8;
        
        window.speechSynthesis.speak(utterance);
    }
    
    /**
     * Speak the currently formed sentence/word
     */
    speakCurrentText() {
        if (!this.wordFormation) {
            this.showNotification('No text to speak', 'warning');
            return;
        }
        
        const text = this.wordFormation.getFullDisplayText();
        if (!text || text.trim() === '') {
            this.showNotification('No text to speak', 'warning');
            return;
        }
        
        // Speak in Urdu
        this.speak(text, 'ur-PK');
        this.showNotification('Speaking text...', 'info');
    }
    
    /**
     * Auto-speak a recognized sign (if TTS is enabled)
     * @param {string} sign - The recognized sign label
     */
    speakSign(sign) {
        if (!this.settings.ttsEnabled || !sign) return;
        
        // Speak the romanized label in English
        this.speak(sign, 'en-US');
    }
    
    // ================== ERROR HANDLING (Production-Ready) ==================
    
    /**
     * Handle camera errors (permission denied, not found, etc.)
     * FIXED: Now handles classified error objects from camera.js for better UX
     */
    handleCameraError(error) {
        console.error('[Camera Error]', error);
        
        // Check if this is a classified error from our improved camera.js
        if (error.type) {
            // Use the classified error information
            const message = error.message;
            const userAction = error.userAction;
            
            // Handle MediaPipe recovery exhausted
            if (error.type === 'mediapipe_recovery' || error.code === 'RECOVERY_EXHAUSTED') {
                this.showMediaPipeError(message || 'MediaPipe stopped responding');
                return;
            }
            
            // Show modal for actionable errors
            if (error.canRetry) {
                this.showCameraErrorModal(`${message}\n\n${userAction}`);
            } else {
                this.showNotification(message, 'error');
            }
            return;
        }
        
        // Legacy handling for unclassified errors
        let message = 'Camera error occurred. Please try again.';
        let showModal = false;
        
        if (error.message) {
            const errorMsg = error.message.toLowerCase();
            
            if (errorMsg.includes('permission') || errorMsg.includes('denied') || errorMsg.includes('notallowed')) {
                message = 'Camera access was denied. Please allow camera permissions to use sign language recognition.';
                showModal = true;
            } else if (errorMsg.includes('notfound') || errorMsg.includes('no camera')) {
                message = 'No camera found. Please connect a camera and try again.';
                showModal = true;
            } else if (errorMsg.includes('mediapipe') || error.code === 'RECOVERY_EXHAUSTED') {
                this.showMediaPipeError(error.message || 'MediaPipe stopped responding');
                return;
            }
        }
        
        if (showModal) {
            this.showCameraErrorModal(message);
        } else {
            this.showNotification(message, 'error');
        }
    }
    
    /**
     * Show camera error modal with browser-specific instructions
     */
    showCameraErrorModal(message) {
        if (this.elements.cameraErrorMessage) {
            this.elements.cameraErrorMessage.textContent = message;
        }
        
        if (this.elements.cameraErrorModal) {
            this.elements.cameraErrorModal.style.display = 'flex';
            
            // Detect browser and show appropriate instructions
            const ua = navigator.userAgent.toLowerCase();
            const isChrome = ua.includes('chrome') && !ua.includes('edg');
            const isEdge = ua.includes('edg');
            const isFirefox = ua.includes('firefox');
            const isSafari = ua.includes('safari') && !ua.includes('chrome');
            const isMobile = /android|iphone|ipad|ipod/.test(ua);
            
            // Show browser-specific instructions
            const chromeInst = document.getElementById('chromeInstructions');
            const firefoxInst = document.getElementById('firefoxInstructions');
            const safariInst = document.getElementById('safariInstructions');
            const mobileInst = document.getElementById('mobileInstructions');
            
            // Hide all first
            [chromeInst, firefoxInst, safariInst, mobileInst].forEach(el => {
                if (el) el.style.display = 'none';
            });
            
            // Show relevant one
            if (isMobile && mobileInst) {
                mobileInst.style.display = 'block';
            } else if (isSafari && safariInst) {
                safariInst.style.display = 'block';
            } else if (isFirefox && firefoxInst) {
                firefoxInst.style.display = 'block';
            } else if (chromeInst) {
                chromeInst.style.display = 'block';
            }
        }
    }
    
    /**
     * Hide camera error modal
     */
    hideCameraErrorModal() {
        if (this.elements.cameraErrorModal) {
            this.elements.cameraErrorModal.style.display = 'none';
        }
    }
    
    /**
     * Show MediaPipe error overlay on video
     */
    showMediaPipeError(message) {
        if (this.elements.mediapipeErrorOverlay) {
            this.elements.mediapipeErrorOverlay.style.display = 'flex';
        }
        if (this.elements.mediapipeErrorMessage) {
            this.elements.mediapipeErrorMessage.textContent = message;
        }
        this.updateSystemStatus('Error');
    }
    
    /**
     * Hide MediaPipe error overlay
     */
    hideMediaPipeError() {
        if (this.elements.mediapipeErrorOverlay) {
            this.elements.mediapipeErrorOverlay.style.display = 'none';
        }
    }
    
    /**
     * Show MediaPipe warning (non-blocking)
     */
    showMediaPipeWarning(data) {
        console.warn('[MediaPipe Warning]', data);
        
        // Show subtle warning after multiple errors
        if (data.consecutiveErrors >= 5) {
            this.showNotification('Hand detection experiencing issues. Try moving your hand.', 'warning');
        }
    }
    
    /**
     * FIXED: Handle MediaPipe recovery events to show user feedback
     */
    handleMediaPipeRecovery(data) {
        debugLog('[MediaPipe Recovery]', data);
        
        switch (data.status) {
            case 'started':
                // Show recovery in progress
                this.updateSystemStatus(`Recovering... (${data.attempt}/${data.maxAttempts})`);
                this.showNotification(data.message, 'warning');
                break;
                
            case 'success':
                // Recovery succeeded
                this.updateSystemStatus('Active');
                this.showNotification(data.message, 'success');
                break;
                
            case 'failed':
                // Recovery failed
                this.updateSystemStatus('Error');
                this.showNotification(data.message, 'error');
                // Show the MediaPipe error overlay for manual retry
                this.showMediaPipeError(data.message);
                break;
        }
    }
    
    /**
     * Show model loading status overlay
     */
    showModelStatus(message, showRetry = false) {
        if (this.elements.modelStatusOverlay) {
            this.elements.modelStatusOverlay.style.display = 'flex';
        }
        if (this.elements.modelStatusText) {
            this.elements.modelStatusText.textContent = message;
        }
        if (this.elements.modelRetryBtn) {
            this.elements.modelRetryBtn.style.display = showRetry ? 'inline-flex' : 'none';
        }
    }
    
    /**
     * Hide model status overlay
     */
    hideModelStatus() {
        if (this.elements.modelStatusOverlay) {
            this.elements.modelStatusOverlay.style.display = 'none';
        }
    }
    
    /**
     * Retry MediaPipe initialization
     */
    async retryMediaPipe() {
        this.hideMediaPipeError();
        this.showNotification('Retrying hand detection...', 'info');
        
        try {
            if (this.camera) {
                await this.camera.initMediaPipe();
                this.showNotification('Hand detection restored!', 'success');
            }
        } catch (error) {
            console.error('MediaPipe retry failed:', error);
            this.showMediaPipeError('Retry failed. Please refresh the page.');
        }
    }
    
    /**
     * Retry ONNX model loading
     */
    async retryModelLoading() {
        this.showModelStatus('Retrying model load...');
        
        try {
            const success = await this.predictor.init(
                'models/psl_model_v2.onnx',
                'models/psl_labels.json',
                'models/sign_thresholds.json'
            );
            
            if (success) {
                this.hideModelStatus();
                this.showNotification('AI model loaded successfully!', 'success');
            } else {
                this.showModelStatus('Model loading failed. Please refresh.', true);
            }
        } catch (error) {
            console.error('Model retry failed:', error);
            this.showModelStatus('Model loading failed: ' + error.message, true);
        }
    }
    
    /**
     * Retry camera access
     */
    async retryCamera() {
        this.hideCameraErrorModal();
        this.showNotification('Requesting camera access...', 'info');
        
        try {
            const cameraReady = await this.camera.init();
            if (cameraReady) {
                this.showNotification('Camera initialized!', 'success');
            } else {
                this.showCameraErrorModal('Camera initialization failed. Please check permissions.');
            }
        } catch (error) {
            console.error('Camera retry failed:', error);
            this.handleCameraError(error);
        }
    }
    
    // ================== PERFORMANCE METRICS & STATUS (NEW!) ==================
    
    /**
     * Update performance metrics UI
     * OPTIMIZED: Throttled DOM updates to reduce layout thrashing
     */
    updatePerformanceMetrics(inferenceTime = null, mediapipeTime = null) {
        if (!this.performanceMetrics) return;
        
        // Record frame for FPS (always record internally)
        this.performanceMetrics.recordFrame();
        
        // Record timing data
        if (inferenceTime !== null) {
            this.performanceMetrics.recordInferenceTime(inferenceTime);
        }
        if (mediapipeTime !== null) {
            this.performanceMetrics.recordMediapipeTime(mediapipeTime);
        }
        
        // PERFORMANCE: Throttle DOM updates to max 4 per second (250ms)
        const now = Date.now();
        if (!this._lastPerfUIUpdate || now - this._lastPerfUIUpdate >= 250) {
            this._lastPerfUIUpdate = now;
            
            // Update UI elements (batched)
            if (this.elements.perfFps) {
                const fps = this.performanceMetrics.fps;
                this.elements.perfFps.textContent = fps;
                this.elements.perfFps.className = `perf-value ${this.performanceMetrics.getFpsClass()}`;
            }
            
            if (this.elements.perfInference) {
                this.elements.perfInference.textContent = `${Math.round(this.performanceMetrics.currentInferenceTime)}ms`;
                this.elements.perfInference.className = `perf-value ${this.performanceMetrics.getInferenceClass()}`;
            }
            
            if (this.elements.perfMediapipe) {
                this.elements.perfMediapipe.textContent = `${Math.round(this.performanceMetrics.currentMediapipeTime)}ms`;
            }
            
            if (this.elements.perfMemory) {
                this.elements.perfMemory.textContent = this.performanceMetrics.getMemoryUsage();
            }
            
            // Update chart (less frequently - every 500ms)
            if (!this._lastChartUpdate || now - this._lastChartUpdate >= 500) {
                this._lastChartUpdate = now;
                this.performanceMetrics.renderChart();
            }
        }
    }
    
    /**
     * Update model version UI
     */
    updateModelVersionUI() {
        if (!this.modelVersionChecker) return;
        
        if (this.elements.modelIndicator) {
            this.elements.modelIndicator.className = 'status-indicator online';
        }
        
        if (this.elements.modelVersion) {
            this.elements.modelVersion.textContent = `Model: ${this.modelVersionChecker.getVersionString()}`;
        }
    }
    
    /**
     * Update offline status UI
     */
    updateOfflineStatusUI() {
        if (!this.offlineChecker) return;
        
        const status = this.offlineChecker.getStatus();
        
        if (this.elements.offlineIndicator) {
            this.elements.offlineIndicator.className = `status-indicator ${this.offlineChecker.getStatusClass()}`;
        }
        
        if (this.elements.offlineStatus) {
            this.elements.offlineStatus.textContent = this.offlineChecker.getStatusText();
        }
    }
    
    /**
     * Handle offline status change
     */
    updateOfflineStatus(status) {
        this.updateOfflineStatusUI();
        
        if (!status.isOnline && status.offlineReady) {
            this.showNotification('Working offline - all features available!', 'info');
        } else if (!status.isOnline && !status.offlineReady) {
            this.showNotification('No internet connection', 'warning');
        }
    }
    
    // ================== SESSION RECORDING (NEW!) ==================
    
    /**
     * Toggle session recording
     */
    toggleRecording() {
        if (!this.sessionRecorder) return;
        
        if (this.sessionRecorder.isRecording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }
    
    /**
     * Start session recording
     */
    startRecording() {
        if (!this.sessionRecorder) return;
        
        this.sessionRecorder.start();
        
        // Update UI
        if (this.elements.recordToggleBtn) {
            this.elements.recordToggleBtn.classList.add('recording');
        }
        if (this.elements.recordIcon) {
            this.elements.recordIcon.textContent = '⏹';
        }
        if (this.elements.recordText) {
            this.elements.recordText.textContent = 'Stop Recording';
        }
        if (this.elements.exportRecordingBtn) {
            this.elements.exportRecordingBtn.style.display = 'none';
        }
        
        // Start recording UI update interval
        this._recordingUpdateInterval = setInterval(() => {
            this.updateRecordingUI();
        }, 500);
        
        this.showNotification('Recording started - capturing predictions for debugging', 'info');
    }
    
    /**
     * Stop session recording
     */
    stopRecording() {
        if (!this.sessionRecorder) return;
        
        this.sessionRecorder.stop();
        
        // Clear update interval
        if (this._recordingUpdateInterval) {
            clearInterval(this._recordingUpdateInterval);
            this._recordingUpdateInterval = null;
        }
        
        // Update UI
        if (this.elements.recordToggleBtn) {
            this.elements.recordToggleBtn.classList.remove('recording');
        }
        if (this.elements.recordIcon) {
            this.elements.recordIcon.textContent = '⏺';
        }
        if (this.elements.recordText) {
            this.elements.recordText.textContent = 'Start Recording';
        }
        if (this.elements.exportRecordingBtn) {
            this.elements.exportRecordingBtn.style.display = 'inline-flex';
        }
        
        this.showNotification(`Recording stopped - ${this.sessionRecorder.recordedFrames.length} frames captured`, 'success');
    }
    
    /**
     * Update recording UI with current status
     */
    updateRecordingUI() {
        if (!this.sessionRecorder) return;
        
        if (this.elements.recordingTime) {
            this.elements.recordingTime.textContent = this.sessionRecorder.getDuration();
        }
        if (this.elements.recordingFrames) {
            this.elements.recordingFrames.textContent = `${this.sessionRecorder.recordedFrames.length} frames`;
        }
    }
    
    /**
     * Export recording to file
     */
    exportRecording() {
        if (!this.sessionRecorder) return;
        
        if (this.sessionRecorder.recordedFrames.length === 0) {
            this.showNotification('No recording data to export', 'warning');
            return;
        }
        
        this.sessionRecorder.downloadRecording();
        this.showNotification('Recording exported successfully', 'success');
    }
    
    /**
     * Record prediction frame if recording is active
     */
    recordPredictionFrame(prediction, landmarks = null) {
        if (!this.sessionRecorder || !this.sessionRecorder.isRecording) return;
        
        const perfMetrics = this.performanceMetrics ? this.performanceMetrics.getMetrics() : null;
        this.sessionRecorder.recordPrediction(prediction, landmarks, perfMetrics);
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
        
        if (this.analyticsPanel) {
            this.analyticsPanel.destroy();
        }
        
        console.log('[OK] App destroyed');
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PSLRecognitionApp;
}

