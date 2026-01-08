/**
 * Practice Mode Module for PSL Recognition System
 * 
 * Provides guided sign language practice with:
 * - Reference images for each sign
 * - Challenge mode with random signs
 * - Progress tracking and streaks
 * - Weak sign targeting based on confusion data
 */

class PracticeMode {
    constructor(options = {}) {
        this.isActive = false;
        this.currentSign = null;
        this.currentChallenge = null;
        
        // Configuration
        this.config = {
            imagesBasePath: options.imagesBasePath || 'assets/signs/',
            manifestPath: options.manifestPath || 'assets/signs/manifest.json',
            targetHoldTimeMs: options.targetHoldTimeMs || 2000,
            streakBonus: options.streakBonus || 5,
            ...options
        };
        
        // State
        this.manifest = null;
        this.signs = [];
        this.weakSigns = [];
        this.streak = 0;
        this.totalAttempts = 0;
        this.successfulAttempts = 0;
        this.practiceHistory = [];
        
        // Callbacks
        this.onSignChange = null;
        this.onSuccess = null;
        this.onFail = null;
        this.onStreakUpdate = null;
        this.onProgressUpdate = null;
        
        // DOM Elements
        this.elements = {};
        
        // Hold timer for current sign recognition
        this.holdStartTime = null;
        this.holdTimer = null;
    }
    
    /**
     * Initialize practice mode
     */
    async init() {
        try {
            // Load sign manifest
            await this.loadManifest();
            
            // Initialize DOM elements
            this.initDOMElements();
            
            // Load practice history from localStorage
            this.loadPracticeHistory();
            
            console.log(`[PracticeMode] Initialized with ${this.signs.length} signs`);
            return true;
            
        } catch (error) {
            console.error('[PracticeMode] Initialization failed:', error);
            return false;
        }
    }
    
    /**
     * Load sign manifest JSON
     */
    async loadManifest() {
        try {
            const response = await fetch(this.config.manifestPath);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            this.manifest = await response.json();
            this.signs = Object.keys(this.manifest.signs);
            
            console.log(`[PracticeMode] Loaded manifest with ${this.signs.length} signs`);
            
        } catch (error) {
            console.warn('[PracticeMode] Could not load manifest, using default signs');
            // Fallback to default Urdu alphabet signs
            this.signs = [
                "1-Hay", "2-Hay", "Ain", "Alif", "Alifmad", "Aray", "Bay", "Byeh",
                "Chay", "Cyeh", "Daal", "Dal", "Dochahay", "Fay", "Gaaf", "Ghain",
                "Hamza", "Jeem", "Kaf", "Khay", "Kiaf", "Lam", "Meem", "Nuun",
                "Nuungh", "Pay", "Ray", "Say", "Seen", "Sheen", "Suad", "Taay",
                "Tay", "Tuey", "Wao", "Zaal", "Zaey", "Zay", "Zuad", "Zuey"
            ];
            
            this.manifest = {
                signs: this.signs.reduce((acc, sign) => {
                    acc[sign] = { romanized: sign, webp: `${sign}.webp`, jpg: `${sign}.jpg` };
                    return acc;
                }, {})
            };
        }
    }
    
    /**
     * Initialize DOM element references
     */
    initDOMElements() {
        this.elements = {
            panel: document.getElementById('practicePanel'),
            toggleBtn: document.getElementById('practiceToggle'),
            closeBtn: document.getElementById('practiceClose'),
            header: document.querySelector('.practice-header'),
            
            // Reference display
            referenceImage: document.getElementById('practiceReferenceImage'),
            signLabel: document.getElementById('practiceSignLabel'),
            urduLabel: document.getElementById('practiceUrduLabel'),
            
            // Progress
            holdProgress: document.getElementById('practiceHoldProgress'),
            holdText: document.getElementById('practiceHoldText'),
            
            // Stats
            streakDisplay: document.getElementById('practiceStreak'),
            accuracyDisplay: document.getElementById('practiceAccuracy'),
            
            // Controls
            skipBtn: document.getElementById('practiceSkip'),
            nextBtn: document.getElementById('practiceNext'),
            weakSignsBtn: document.getElementById('practiceWeakSigns'),
            randomBtn: document.getElementById('practiceRandom')
        };
        
        // Make panel draggable
        this.initDraggable();
    }
    
    /**
     * Initialize draggable functionality for the practice panel
     */
    initDraggable() {
        if (!this.elements.panel || !this.elements.header) return;
        
        let isDragging = false;
        let startX, startY;
        let panelX = 0, panelY = 0;
        
        // Add cursor style to header
        this.elements.header.style.cursor = 'move';
        
        const onDragStart = (e) => {
            // Only drag from header, not buttons
            if (e.target.closest('button')) return;
            
            isDragging = true;
            const rect = this.elements.panel.getBoundingClientRect();
            
            // Get current position
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            
            startX = clientX - rect.left - rect.width / 2;
            startY = clientY - rect.top - rect.height / 2;
            
            this.elements.panel.style.transition = 'none';
            e.preventDefault();
        };
        
        const onDragMove = (e) => {
            if (!isDragging) return;
            
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            
            panelX = clientX - startX - window.innerWidth / 2;
            panelY = clientY - startY - window.innerHeight / 2;
            
            // Keep panel within viewport bounds
            const maxX = window.innerWidth / 2 - 100;
            const maxY = window.innerHeight / 2 - 50;
            panelX = Math.max(-maxX, Math.min(maxX, panelX));
            panelY = Math.max(-maxY, Math.min(maxY, panelY));
            
            this.elements.panel.style.transform = `translate(calc(-50% + ${panelX}px), calc(-50% + ${panelY}px)) scale(1)`;
        };
        
        const onDragEnd = () => {
            if (!isDragging) return;
            isDragging = false;
            this.elements.panel.style.transition = '';
        };
        
        // Mouse events
        this.elements.header.addEventListener('mousedown', onDragStart);
        document.addEventListener('mousemove', onDragMove);
        document.addEventListener('mouseup', onDragEnd);
        
        // Touch events
        this.elements.header.addEventListener('touchstart', onDragStart, { passive: false });
        document.addEventListener('touchmove', onDragMove, { passive: false });
        document.addEventListener('touchend', onDragEnd);
        
        // Store cleanup function
        this._cleanupDraggable = () => {
            this.elements.header?.removeEventListener('mousedown', onDragStart);
            document.removeEventListener('mousemove', onDragMove);
            document.removeEventListener('mouseup', onDragEnd);
            this.elements.header?.removeEventListener('touchstart', onDragStart);
            document.removeEventListener('touchmove', onDragMove);
            document.removeEventListener('touchend', onDragEnd);
        };
        
        // Reset position function
        this.resetPanelPosition = () => {
            panelX = 0;
            panelY = 0;
            if (this.elements.panel) {
                this.elements.panel.style.transform = 'translate(-50%, -50%) scale(1)';
            }
        };
    }
    
    /**
     * Start practice mode
     */
    start() {
        this.isActive = true;
        this.streak = 0;
        
        if (this.elements.panel) {
            this.elements.panel.classList.add('visible');
        }
        
        // Select first sign
        this.selectRandomSign();
        
        console.log('[PracticeMode] Started');
    }
    
    /**
     * Stop practice mode
     */
    stop() {
        this.isActive = false;
        this.currentSign = null;
        this.clearHoldTimer();
        
        if (this.elements.panel) {
            this.elements.panel.classList.remove('visible');
        }
        
        // Reset panel position to center for next time
        if (this.resetPanelPosition) {
            this.resetPanelPosition();
        }
        
        // Save progress
        this.savePracticeHistory();
        
        console.log('[PracticeMode] Stopped');
    }
    
    /**
     * Toggle practice mode visibility
     */
    toggle() {
        if (this.isActive) {
            this.stop();
        } else {
            this.start();
        }
    }
    
    /**
     * Select a random sign to practice
     */
    selectRandomSign() {
        if (this.signs.length === 0) return;
        
        // Avoid repeating the same sign
        let newSign;
        do {
            const idx = Math.floor(Math.random() * this.signs.length);
            newSign = this.signs[idx];
        } while (newSign === this.currentSign && this.signs.length > 1);
        
        this.setCurrentSign(newSign);
    }
    
    /**
     * Select a sign from weak signs list (commonly confused)
     */
    selectWeakSign() {
        if (this.weakSigns.length === 0) {
            // No weak signs, pick random
            this.selectRandomSign();
            return;
        }
        
        const idx = Math.floor(Math.random() * this.weakSigns.length);
        this.setCurrentSign(this.weakSigns[idx]);
    }
    
    /**
     * Set the current sign to practice
     */
    setCurrentSign(sign) {
        this.currentSign = sign;
        this.clearHoldTimer();
        
        // Get sign info
        const signInfo = this.manifest?.signs?.[sign] || { romanized: sign };
        
        // Update UI
        if (this.elements.signLabel) {
            this.elements.signLabel.textContent = signInfo.romanized || sign;
        }
        
        if (this.elements.urduLabel) {
            this.elements.urduLabel.textContent = signInfo.urdu || '';
        }
        
        // Update reference image
        if (this.elements.referenceImage) {
            const webpPath = `${this.config.imagesBasePath}${signInfo.webp || sign + '.webp'}`;
            const jpgPath = `${this.config.imagesBasePath}${signInfo.jpg || sign + '.jpg'}`;
            
            // Try WebP first, fallback to JPG
            this.elements.referenceImage.onerror = () => {
                this.elements.referenceImage.src = jpgPath;
            };
            this.elements.referenceImage.src = webpPath;
        }
        
        // Reset progress
        this.updateHoldProgress(0);
        
        // Notify callback
        if (this.onSignChange) {
            this.onSignChange(sign, signInfo);
        }
        
        console.log(`[PracticeMode] Now practicing: ${sign}`);
    }
    
    /**
     * Process incoming prediction from recognition system
     * @param {Object} prediction - { label, confidence }
     */
    processPrediction(prediction) {
        if (!this.isActive || !this.currentSign) return;
        
        const isMatch = prediction.label === this.currentSign;
        
        if (isMatch && prediction.confidence > 0.6) {
            // Start or continue hold timer
            if (!this.holdStartTime) {
                this.startHoldTimer();
            } else {
                this.updateHoldProgress();
            }
        } else {
            // Wrong sign or low confidence - reset hold timer
            if (this.holdStartTime) {
                this.clearHoldTimer();
                this.updateHoldProgress(0);
            }
        }
    }
    
    /**
     * Start hold timer for sign recognition
     * FIXED: Uses requestAnimationFrame for smoother animation and better battery efficiency
     * FIXED: Added race condition guard to prevent multiple concurrent timers
     */
    startHoldTimer() {
        // RACE CONDITION FIX: Don't start if already running
        if (this.holdTimerActive && this.holdStartTime) {
            return; // Timer already running, don't start another
        }
        
        this.holdStartTime = Date.now();
        this.holdTimerActive = true;
        
        const updateProgress = () => {
            if (!this.holdTimerActive || !this.holdStartTime) {
                return; // Timer was cleared, stop the loop
            }
            
            const elapsed = Date.now() - this.holdStartTime;
            const progress = Math.min(elapsed / this.config.targetHoldTimeMs, 1);
            
            this.updateHoldProgress(progress);
            
            if (progress >= 1) {
                this.onSignSuccess();
            } else {
                // Continue animation loop
                this.holdTimer = requestAnimationFrame(updateProgress);
            }
        };
        
        // Start the animation loop
        this.holdTimer = requestAnimationFrame(updateProgress);
    }
    
    /**
     * Clear hold timer
     * FIXED: Properly cancels requestAnimationFrame
     */
    clearHoldTimer() {
        this.holdTimerActive = false;
        if (this.holdTimer) {
            cancelAnimationFrame(this.holdTimer);
            this.holdTimer = null;
        }
        this.holdStartTime = null;
    }
    
    /**
     * Update hold progress UI
     */
    updateHoldProgress(progress = null) {
        if (progress === null && this.holdStartTime) {
            const elapsed = Date.now() - this.holdStartTime;
            progress = Math.min(elapsed / this.config.targetHoldTimeMs, 1);
        }
        
        progress = progress || 0;
        
        if (this.elements.holdProgress) {
            this.elements.holdProgress.style.width = `${progress * 100}%`;
        }
        
        if (this.elements.holdText) {
            if (progress > 0 && progress < 1) {
                const remaining = Math.ceil((1 - progress) * this.config.targetHoldTimeMs / 1000);
                this.elements.holdText.textContent = `Hold for ${remaining}s...`;
            } else if (progress >= 1) {
                this.elements.holdText.textContent = 'Success!';
            } else {
                this.elements.holdText.textContent = 'Show the sign';
            }
        }
    }
    
    /**
     * Handle successful sign recognition
     */
    onSignSuccess() {
        this.clearHoldTimer();
        
        this.totalAttempts++;
        this.successfulAttempts++;
        this.streak++;
        
        // Record in history
        this.practiceHistory.push({
            sign: this.currentSign,
            success: true,
            timestamp: Date.now()
        });
        
        // Update stats display
        this.updateStatsDisplay();
        
        // Notify callback
        if (this.onSuccess) {
            this.onSuccess({
                sign: this.currentSign,
                streak: this.streak,
                accuracy: this.getAccuracy()
            });
        }
        
        console.log(`[PracticeMode] Success! Streak: ${this.streak}`);
        
        // Auto-advance after short delay
        setTimeout(() => {
            this.selectRandomSign();
        }, 1500);
    }
    
    /**
     * Handle skipped sign
     */
    onSignSkip() {
        this.totalAttempts++;
        this.streak = 0;
        
        // Record in history
        this.practiceHistory.push({
            sign: this.currentSign,
            success: false,
            skipped: true,
            timestamp: Date.now()
        });
        
        // Update stats display
        this.updateStatsDisplay();
        
        // Notify callback
        if (this.onFail) {
            this.onFail({
                sign: this.currentSign,
                skipped: true
            });
        }
        
        // Next sign
        this.selectRandomSign();
    }
    
    /**
     * Update stats display
     */
    updateStatsDisplay() {
        if (this.elements.streakDisplay) {
            this.elements.streakDisplay.textContent = this.streak;
        }
        
        if (this.elements.accuracyDisplay) {
            this.elements.accuracyDisplay.textContent = `${this.getAccuracy()}%`;
        }
        
        if (this.onProgressUpdate) {
            this.onProgressUpdate({
                streak: this.streak,
                accuracy: this.getAccuracy(),
                total: this.totalAttempts,
                successful: this.successfulAttempts
            });
        }
    }
    
    /**
     * Get current accuracy percentage
     */
    getAccuracy() {
        if (this.totalAttempts === 0) return 0;
        return Math.round((this.successfulAttempts / this.totalAttempts) * 100);
    }
    
    /**
     * Set weak signs from feedback manager data
     * @param {Array} confusedSigns - Array of sign labels that user often confuses
     */
    setWeakSigns(confusedSigns) {
        this.weakSigns = confusedSigns.filter(sign => this.signs.includes(sign));
        console.log(`[PracticeMode] Set ${this.weakSigns.length} weak signs for targeted practice`);
    }
    
    /**
     * Get weak signs from confusion matrix
     * @param {Object} confusionMatrix - From feedback manager
     * @param {number} threshold - Minimum confusion count
     */
    analyzeWeakSigns(confusionMatrix, threshold = 3) {
        const weakSet = new Set();
        
        for (const predicted in confusionMatrix) {
            for (const actual in confusionMatrix[predicted]) {
                if (confusionMatrix[predicted][actual] >= threshold) {
                    weakSet.add(predicted);
                    weakSet.add(actual);
                }
            }
        }
        
        this.weakSigns = Array.from(weakSet);
        return this.weakSigns;
    }
    
    /**
     * Load practice history from localStorage
     */
    loadPracticeHistory() {
        try {
            const saved = localStorage.getItem('psl_practice_history');
            if (saved) {
                const data = JSON.parse(saved);
                this.practiceHistory = data.history || [];
                this.totalAttempts = data.totalAttempts || 0;
                this.successfulAttempts = data.successfulAttempts || 0;
            }
        } catch (e) {
            console.warn('[PracticeMode] Could not load practice history');
        }
    }
    
    /**
     * Save practice history to localStorage
     */
    savePracticeHistory() {
        try {
            // Keep only last 100 entries
            const historyToSave = this.practiceHistory.slice(-100);
            
            localStorage.setItem('psl_practice_history', JSON.stringify({
                history: historyToSave,
                totalAttempts: this.totalAttempts,
                successfulAttempts: this.successfulAttempts,
                lastUpdated: Date.now()
            }));
        } catch (e) {
            console.warn('[PracticeMode] Could not save practice history');
        }
    }
    
    /**
     * Get sign info including image URL
     */
    getSignInfo(sign) {
        const signData = this.manifest?.signs?.[sign];
        if (!signData) return null;
        
        return {
            romanized: signData.romanized || sign,
            urdu: signData.urdu || '',
            webpUrl: `${this.config.imagesBasePath}${signData.webp || sign + '.webp'}`,
            jpgUrl: `${this.config.imagesBasePath}${signData.jpg || sign + '.jpg'}`
        };
    }
    
    /**
     * Get all available signs
     */
    getAllSigns() {
        return [...this.signs];
    }
    
    /**
     * Destroy practice mode
     */
    destroy() {
        this.stop();
        this.savePracticeHistory();
        
        // Clean up draggable listeners
        if (this._cleanupDraggable) {
            this._cleanupDraggable();
        }
    }
}


// ================================================================================
// EXPORTS
// ================================================================================

if (typeof window !== 'undefined') {
    window.PracticeMode = PracticeMode;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PracticeMode };
}


