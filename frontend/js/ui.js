/**
 * UI Manager for PSL Recognition System
 * Handles all user interface updates, notifications, and interactions
 */

class UIManager {
    // Configuration constants - sync with backend config.py
    static MODEL_SEQUENCE_LENGTH = 60;
    static CRITICAL_ELEMENTS = ['statusDot', 'statusText', 'startBtn', 'stopBtn', 'predictionResult', 'confidenceFill', 'confidenceText'];
    static OPTIONAL_ELEMENTS = ['bufferFill', 'bufferCount', 'bufferStatusText', 'historyList', 'clearHistoryBtn', 'feedbackSection', 'feedbackCorrect', 'feedbackIncorrect', 'notificationContainer'];
    
    constructor() {
        this._currentLanguage = 'urdu';
        this.isRecognitionActive = false;
        this.isConnected = false;
        
        // UI elements
        this.elements = {};
        this.missingElements = [];
        
        // Notification system
        this.notifications = [];
        this.maxNotifications = 5;
    }
    
    async init() {
        try {
            console.log('Initializing UI Manager...');
            
            // Cache DOM elements
            this.cacheElements();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Set up initial state
            this.setConnectionStatus(false);
            this.setRecognitionState(false);
            this.updateBuffer(0, UIManager.MODEL_SEQUENCE_LENGTH);
            this.updateHandStatus(false);
            this.updateFPS(0);
            
            console.log('UI Manager initialized');
            return true;
            
        } catch (error) {
            console.error('UI Manager initialization failed:', error);
            return false;
        }
    }
    
    cacheElements() {
        this.elements = {
            // Status elements
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            connectionBadge: document.getElementById('connectionBadge'),
            
            // Control buttons
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            resetBtn: document.getElementById('resetBtn'),
            
            // Prediction display - Try both IDs for compatibility
            predictionResult: document.getElementById('predictionResult') || document.getElementById('predictionDisplay'),
            confidenceFill: document.getElementById('confidenceFill'),
            confidenceText: document.getElementById('confidenceText'),
            
            // Buffer elements
            bufferFill: document.getElementById('bufferFill'),
            bufferCount: document.getElementById('bufferCount'),
            bufferStatusText: document.getElementById('bufferStatusText'),
            
            // Status indicators
            handStatus: document.getElementById('handStatus'),
            fpsValue: document.getElementById('fpsValue'),
            systemStatusValue: document.getElementById('systemStatusValue'),
            signCountValue: document.getElementById('signCountValue'),
            
            // History
            historyList: document.getElementById('historyList'),
            clearHistoryBtn: document.getElementById('clearHistoryBtn'),
            
            // Feedback
            feedbackSection: document.getElementById('feedbackSection'),
            feedbackCorrect: document.getElementById('feedbackCorrect'),
            feedbackIncorrect: document.getElementById('feedbackIncorrect'),
            
            // Notifications - Try both IDs for compatibility
            notificationContainer: document.getElementById('notificationContainer') || document.getElementById('notifications')
        };
        
        // Validate critical elements and log warnings for missing ones
        this.missingElements = [];
        UIManager.CRITICAL_ELEMENTS.forEach(elementName => {
            if (!this.elements[elementName]) {
                this.missingElements.push(elementName);
                console.warn(`[UIManager] Critical element missing: ${elementName}`);
            }
        });
        
        // Log optional missing elements at debug level
        UIManager.OPTIONAL_ELEMENTS.forEach(elementName => {
            if (!this.elements[elementName]) {
                console.debug(`[UIManager] Optional element not found: ${elementName}`);
            }
        });
        
        if (this.missingElements.length > 0) {
            console.warn(`[UIManager] ${this.missingElements.length} critical elements missing - some UI features may not work`);
        } else {
            console.log('[UIManager] All critical elements cached successfully');
        }
        
        // Track stats
        this.stats = {
            signCount: 0,
            fps: 0
        };
    }
    
    setConnectionStatus(connected) {
        this.isConnected = connected;
        
        if (this.elements.statusDot) {
            this.elements.statusDot.classList.toggle('connected', connected);
        }
        
        if (this.elements.statusText) {
            this.elements.statusText.textContent = connected ? 'Connected' : 'Connecting...';
        }
        
        if (this.elements.connectionBadge) {
            this.elements.connectionBadge.classList.toggle('connected', connected);
        }
        
        if (this.elements.systemStatusValue) {
            this.elements.systemStatusValue.textContent = connected ? 'Ready' : 'Disconnected';
            this.elements.systemStatusValue.style.color = connected ? '#10b981' : '#ef4444';
        }
        
        // Update button states
        this.updateButtonStates();
    }
    
    setRecognitionState(active) {
        this.isRecognitionActive = active;
        
        try {
            if (this.elements.startBtn) {
                this.elements.startBtn.disabled = active || !this.isConnected;
            }
            
            if (this.elements.stopBtn) {
                this.elements.stopBtn.disabled = !active;
            }
            
            if (this.elements.resetBtn) {
                this.elements.resetBtn.disabled = active;
            }
            
            // Update button text
            this.updateButtonText();
        } catch (error) {
            console.error('[UIManager] Error in setRecognitionState:', error);
        }
    }
    
    updateButtonStates() {
        const canStart = this.isConnected && !this.isRecognitionActive;
        const canStop = this.isRecognitionActive;
        const canReset = !this.isRecognitionActive;
        
        if (this.elements.startBtn) {
            this.elements.startBtn.disabled = !canStart;
        }
        
        if (this.elements.stopBtn) {
            this.elements.stopBtn.disabled = !canStop;
        }
        
        if (this.elements.resetBtn) {
            this.elements.resetBtn.disabled = !canReset;
        }
    }
    
    updateButtonText() {
        if (this.elements.startBtn) {
            const urduText = this.isRecognitionActive ? 'شروع کریں' : 'شروع کریں';
            const englishText = this.isRecognitionActive ? 'Start Recognition' : 'Start Recognition';
            this.updateMixedText(this.elements.startBtn, urduText, englishText);
        }
        
        if (this.elements.stopBtn) {
            const urduText = 'روکیں';
            const englishText = 'Stop Recognition';
            this.updateMixedText(this.elements.stopBtn, urduText, englishText);
        }
        
        if (this.elements.resetBtn) {
            const urduText = 'ری سیٹ';
            const englishText = 'Reset';
            this.updateMixedText(this.elements.resetBtn, urduText, englishText);
        }
    }
    
    updatePrediction(prediction, confidence, allPredictions, isStable = false, isLowConfidence = false) {
        try {
            // Update prediction display with new structure
            if (this.elements.predictionResult) {
                if (isLowConfidence) {
                    // Show "Searching..." state for low confidence predictions
                    this.elements.predictionResult.innerHTML = `
                        <span class="placeholder searching">
                            <span class="urdu">تلاش جاری ہے...</span>
                            <span class="english">Searching...</span>
                        </span>
                    `;
                    // Remove detected class
                    this.elements.predictionResult.classList.remove('sign-detected');
                } else if (prediction) {
                    // POLISH: Check if this is a new prediction for animation
                    const isNewPrediction = this._lastPrediction !== prediction;
                    this._lastPrediction = prediction;
                    
                    this.elements.predictionResult.innerHTML = `
                        <span class="sign-text ${isNewPrediction ? 'sign-new' : ''}">${prediction}</span>
                    `;
                    
                    // POLISH: Add visual pulse effect for new detections
                    if (isNewPrediction) {
                        try {
                            this.elements.predictionResult.classList.add('sign-detected');
                            setTimeout(() => {
                                try {
                                    if (this.elements.predictionResult) {
                                        this.elements.predictionResult.classList.remove('sign-detected');
                                        // Also remove animation class after it completes
                                        const signText = this.elements.predictionResult.querySelector('.sign-text');
                                        if (signText) {
                                            signText.classList.remove('sign-new');
                                        }
                                    }
                                } catch (e) {
                                    // Element may have been removed during timeout
                                }
                            }, 400);
                        } catch (e) {
                            console.debug('[UIManager] Animation effect failed:', e);
                        }
                    }
                    
                    // Increment sign count only for new predictions
                    if (isNewPrediction) {
                        this.stats.signCount++;
                        if (this.elements.signCountValue) {
                            this.elements.signCountValue.textContent = this.stats.signCount;
                        }
                        
                        // Add to history
                        this.addToHistory(prediction, confidence);
                    }
                    
                } else {
                    this.elements.predictionResult.innerHTML = `
                        <span class="placeholder">
                            <span class="urdu">اشارہ دکھائیں</span>
                            <span class="english">Make a sign</span>
                        </span>
                    `;
                    // Reset last prediction when cleared
                    this._lastPrediction = null;
                }
            }
            
            // Update confidence bar and value
            if (confidence !== undefined) {
                const confidencePercent = Math.round(confidence * 100);
                
                if (this.elements.confidenceFill) {
                    this.elements.confidenceFill.style.width = `${confidencePercent}%`;
                }
                
                if (this.elements.confidenceText) {
                    this.elements.confidenceText.textContent = `${confidencePercent}%`;
                }
            } else {
                if (this.elements.confidenceFill) {
                    this.elements.confidenceFill.style.width = '0%';
                }
                if (this.elements.confidenceText) {
                    this.elements.confidenceText.textContent = '0%';
                }
            }
        } catch (error) {
            console.error('[UIManager] Error in updatePrediction:', error);
        }
    }
    
    clearPrediction() {
        try {
            // Clear prediction display to placeholder state
            if (this.elements.predictionResult) {
                this.elements.predictionResult.innerHTML = `
                    <span class="placeholder">
                        <span class="urdu">اشارہ دکھائیں</span>
                        <span class="english">Make a sign</span>
                    </span>
                `;
            }
            
            // Reset confidence bar
            if (this.elements.confidenceFill) {
                this.elements.confidenceFill.style.width = '0%';
            }
            if (this.elements.confidenceText) {
                this.elements.confidenceText.textContent = '0%';
            }
        } catch (error) {
            console.error('[UIManager] Error in clearPrediction:', error);
        }
    }
    
    updateBuffer(current, total) {
        try {
            // Use constant if total not provided
            const targetTotal = total || UIManager.MODEL_SEQUENCE_LENGTH;
            
            // Update buffer progress bar
            const percent = Math.min(100, (current / targetTotal) * 100);
            
            if (this.elements.bufferFill) {
                this.elements.bufferFill.style.width = `${percent}%`;
            }
            
            if (this.elements.bufferCount) {
                this.elements.bufferCount.textContent = `${current}/${targetTotal}`;
            }
            
            if (this.elements.bufferStatusText) {
                if (current === 0) {
                    this.elements.bufferStatusText.innerHTML = `<span id="bufferCount">0/${UIManager.MODEL_SEQUENCE_LENGTH}</span> frames`;
                } else {
                    this.elements.bufferStatusText.innerHTML = `<span id="bufferCount">${current}/${targetTotal}</span> frames`;
                }
            }
            
            // Update system status
            if (this.elements.systemStatusValue) {
                if (current >= targetTotal) {
                    this.elements.systemStatusValue.textContent = 'Processing';
                    this.elements.systemStatusValue.style.color = '#3b82f6';
                } else if (current > 0) {
                    this.elements.systemStatusValue.textContent = 'Collecting';
                    this.elements.systemStatusValue.style.color = '#f59e0b';
                }
            }
        } catch (error) {
            console.error('[UIManager] Error in updateBuffer:', error);
        }
    }
    
    updateHandStatus(hasHands) {
        try {
            if (this.elements.handStatus) {
                const label = this.elements.handStatus.querySelector('.status-label');
                if (hasHands) {
                    this.elements.handStatus.classList.add('hands-detected');
                    if (label) label.textContent = 'Hands detected';
                } else {
                    this.elements.handStatus.classList.remove('hands-detected');
                    if (label) label.textContent = 'No hands';
                }
            }
        } catch (error) {
            console.error('[UIManager] Error in updateHandStatus:', error);
        }
    }
    
    updateFPS(fps) {
        try {
            this.stats.fps = fps;
            if (this.elements.fpsValue) {
                this.elements.fpsValue.textContent = Math.round(fps);
            }
        } catch (error) {
            console.error('[UIManager] Error in updateFPS:', error);
        }
    }
    
    addToHistory(prediction, confidence) {
        // Early return if historyList element is missing
        if (!this.elements.historyList) {
            console.debug('[UIManager] historyList element not found, skipping addToHistory');
            return;
        }
        
        try {
            // Remove empty state - check both class names for compatibility
            const emptyState = this.elements.historyList.querySelector('.history-empty') || 
                               this.elements.historyList.querySelector('.history-empty-simple');
            if (emptyState) {
                emptyState.remove();
            }
            
            // Show clear button
            if (this.elements.clearHistoryBtn) {
                this.elements.clearHistoryBtn.style.display = 'block';
            }
            
            // Create history item
            const item = document.createElement('div');
            item.className = 'history-item';
            
            const timestamp = new Date().toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit'
            });
            
            item.innerHTML = `
                <div class="history-item-label">${prediction}</div>
                <div class="history-item-meta">
                    <span class="history-item-confidence">${Math.round(confidence * 100)}%</span>
                    <span class="history-item-time">${timestamp}</span>
                </div>
            `;
            
            // Add to top of list
            this.elements.historyList.insertBefore(item, this.elements.historyList.firstChild);
            
            // Limit history to 8 items for compact view
            const items = this.elements.historyList.querySelectorAll('.history-item');
            if (items.length > 8) {
                items[items.length - 1].remove();
            }
        } catch (error) {
            console.error('[UIManager] Error in addToHistory:', error);
        }
    }
    
    clearHistory() {
        // Early return if historyList element is missing
        if (!this.elements.historyList) {
            console.debug('[UIManager] historyList element not found, skipping clearHistory');
            return;
        }
        
        try {
            this.elements.historyList.innerHTML = `
                <div class="history-empty">
                    <span class="english">No signs yet</span>
                </div>
            `;
            
            // Hide clear button
            if (this.elements.clearHistoryBtn) {
                this.elements.clearHistoryBtn.style.display = 'none';
            }
        } catch (error) {
            console.error('[UIManager] Error in clearHistory:', error);
        }
    }
    
    showStabilizing(historySize, threshold) {
        try {
            // Show that we're gathering predictions for stability
            if (this.elements.predictionResult) {
                const progress = Math.min(100, (historySize / threshold) * 100);
                this.elements.predictionResult.innerHTML = `
                    <span class="placeholder stabilizing">
                        <span class="urdu">استحکام کر رہے ہیں... ${Math.round(progress)}%</span>
                        <span class="english">Stabilizing... ${Math.round(progress)}%</span>
                    </span>
                `;
            }
        } catch (error) {
            console.error('[UIManager] Error in showStabilizing:', error);
        }
    }
    
    showTemporaryPrediction(prediction, confidence) {
        try {
            // Show a muted/temporary prediction while stabilizing
            if (this.elements.predictionResult && prediction) {
                this.elements.predictionResult.innerHTML = `
                    <span class="placeholder stabilizing">
                        <span class="prediction-temp">${prediction}</span>
                        <span class="english" style="opacity: 0.7; font-size: 0.8em;">Confirming...</span>
                    </span>
                `;
            }
        } catch (error) {
            console.error('[UIManager] Error in showTemporaryPrediction:', error);
        }
    }
    
    updateProgress(current, target) {
        // HIDDEN - Progress tracking removed from UI for simplicity
        // Backend still tracks frames, just not displayed
    }
    
    showLoadingProgress(percent) {
        try {
            // Show a subtle loading indicator during initial frame collection
            if (this.elements.predictionResult) {
                const placeholder = this.elements.predictionResult.querySelector('.placeholder');
                if (placeholder && percent < 100) {
                    placeholder.innerHTML = `
                        <span class="urdu">فریم جمع کر رہے ہیں... ${percent}%</span>
                        <span class="english">Collecting frames... ${percent}%</span>
                    `;
                } else if (percent < 100) {
                    // No placeholder exists, create the loading state
                    this.elements.predictionResult.innerHTML = `
                        <span class="placeholder">
                            <span class="urdu">فریم جمع کر رہے ہیں... ${percent}%</span>
                            <span class="english">Collecting frames... ${percent}%</span>
                        </span>
                    `;
                }
            }
        } catch (error) {
            console.error('[UIManager] Error in showLoadingProgress:', error);
        }
    }
    
    hideLoadingProgress() {
        try {
            // Hide loading indicator - only if showing loading state
            if (this.elements.predictionResult) {
                const placeholder = this.elements.predictionResult.querySelector('.placeholder');
                if (placeholder && placeholder.textContent && placeholder.textContent.includes('Collecting')) {
                    placeholder.innerHTML = `
                        <span class="urdu">اشارہ دکھائیں</span>
                        <span class="english">Make a sign</span>
                    `;
                }
            }
        } catch (error) {
            console.error('[UIManager] Error in hideLoadingProgress:', error);
        }
    }
    
    updateDetectionStatus(status) {
        // HIDDEN - Detection status removed from UI for simplicity
        // Backend still detects hands, just not displayed
    }
    
    setLanguage(language) {
        this._currentLanguage = language;
        
        // Update all text elements
        this.updateAllTexts();
        
        // Update settings panel
        if (this.elements.languageSelect) {
            this.elements.languageSelect.value = language;
        }
    }
    
    updateAllTexts() {
        // Update button texts
        this.updateButtonText();
        
        // Update status texts
        this.setConnectionStatus(this.isConnected);
        this.updateDetectionStatus('no_hands');
        
        // Update prediction placeholder
        if (this.elements.predictionResult) {
            const placeholder = this.elements.predictionResult.querySelector('.placeholder');
            if (placeholder) {
                placeholder.innerHTML = `
                    <span class="urdu">یہاں اشارہ ظاہر ہوگا</span>
                    <span class="english">Sign will appear here</span>
                `;
            }
        }
    }
    
    updateMixedText(element, urduText, englishText) {
        if (!element) return;
        
        element.innerHTML = `
            <span class="urdu">${urduText}</span>
            <span class="english">${englishText}</span>
        `;
    }
    
    showNotification(message, type = 'info', duration = 5000) {
        const container = this.elements.notificationContainer;
        if (!container) {
            console.debug('[UIManager] Notification container not found, logging instead:', type, message);
            return;
        }
        
        try {
            const notification = this.createNotification(message, type);
            container.appendChild(notification);
                
            // Remove old notifications
            while (container.children.length > this.maxNotifications) {
                container.removeChild(container.firstChild);
            }
                
            // Auto-remove notification
            setTimeout(() => {
                try {
                    if (notification.parentNode) {
                        notification.style.opacity = '0';
                        notification.style.transform = 'translateX(100%)';
                        setTimeout(() => {
                            try {
                                if (notification.parentNode) {
                                    notification.parentNode.removeChild(notification);
                                }
                            } catch (e) {
                                // Element may have been removed already
                            }
                        }, 300);
                    }
                } catch (e) {
                    // Element may have been removed already
                }
            }, duration);
            
            // Store notification reference
            this.notifications.push(notification);
        } catch (error) {
            console.error('[UIManager] Error in showNotification:', error);
        }
    }
    
    createNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        // Icon based on type
        const icons = {
            success: '✓',
            error: '✗',
            warning: '⚠',
            info: 'ℹ'
        };
        
        const icon = icons[type] || icons.info;
        
        notification.innerHTML = `
            <div class="notification-icon">${icon}</div>
            <div class="notification-content">
                <div class="notification-message">${message}</div>
            </div>
        `;
        
        return notification;
    }
    
    clearNotifications() {
        if (this.elements.notificationContainer) {
            this.elements.notificationContainer.innerHTML = '';
        }
        this.notifications = [];
    }
    
    setupEventListeners() {
        // Clear history button
        if (this.elements.clearHistoryBtn) {
            this.elements.clearHistoryBtn.addEventListener('click', () => {
                this.clearHistory();
            });
        }
    }
    
    reset() {
        console.log('[UIManager] Resetting UI...');
        
        try {
            // Reset prediction display
            this.updatePrediction(null, 0, []);
            
            // Reset buffer using constant
            this.updateBuffer(0, UIManager.MODEL_SEQUENCE_LENGTH);
            
            // Reset hand status
            this.updateHandStatus(false);
            
            // Clear history
            this.clearHistory();
            
            // Reset stats
            this.stats.signCount = 0;
            if (this.elements.signCountValue) {
                this.elements.signCountValue.textContent = '0';
            }
            
            // Reset system status
            if (this.elements.systemStatusValue) {
                this.elements.systemStatusValue.textContent = 'Ready';
                this.elements.systemStatusValue.style.color = '#10b981';
            }
            
            // Clear notifications
            this.clearNotifications();
            
            // Reset button states
            this.setRecognitionState(false);
            
            console.log('[UIManager] UI reset complete');
        } catch (error) {
            console.error('[UIManager] Error in reset:', error);
        }
    }
    
    // Utility methods
    showLoading(message = 'Loading...') {
        try {
            const overlay = document.getElementById('loadingOverlay');
            if (overlay) {
                const loadingText = overlay.querySelector('.loading-text');
                if (loadingText) {
                    loadingText.innerHTML = `
                        <span class="urdu">${message}</span>
                        <span class="english">${message}</span>
                    `;
                }
                overlay.classList.remove('hidden');
            }
        } catch (error) {
            console.error('[UIManager] Error in showLoading:', error);
        }
    }
    
    hideLoading() {
        try {
            const overlay = document.getElementById('loadingOverlay');
            if (overlay) {
                overlay.classList.add('hidden');
            }
        } catch (error) {
            console.error('[UIManager] Error in hideLoading:', error);
        }
    }
    
    // Getters
    get currentLanguage() {
        return this._currentLanguage;
    }

    updatePredictionList(allPredictions) {
        // REMOVED - All predictions list hidden for cleaner UI
        // Only show the main prediction with highest confidence
    }
    
    get recognitionActive() {
        return this.isRecognitionActive;
    }
    
    get connected() {
        return this.isConnected;
    }
    
    destroy() {
        // Clean up event listeners and references
        console.log('Destroying UI Manager...');
        
        // Clear notifications
        this.clearNotifications();
        
        // Reset state
        this.isConnected = false;
        this.isRecognitionActive = false;
        
        // Clear element references
        this.elements = {};
        this.notifications = [];
        
        console.log('UI Manager destroyed');
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIManager;
}


