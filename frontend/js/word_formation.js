/**
 * Word Formation Module for PSL Recognition
 * 
 * Handles:
 * - Converting romanized labels to Urdu characters
 * - Pause detection (5 seconds = word complete)
 * - Letter deduplication (prevents repeated letters from holding)
 * - Word and sentence building
 */

class WordFormation {
    constructor(options = {}) {
        // Configuration
        this.config = {
            pauseThresholdMs: options.pauseThresholdMs || 5000,  // 5 seconds for word completion
            minLetterGapMs: options.minLetterGapMs || 1500,       // 1.5 seconds min gap for same letter
            maxWordsInSentence: options.maxWordsInSentence || 20,
            showPauseIndicator: options.showPauseIndicator !== false
        };
        
        // State
        this.urduMapping = {};
        this.isReady = false;
        
        // Current word being formed
        this.currentWord = [];
        this.currentWordRomanized = [];
        
        // Completed words in sentence
        this.completedWords = [];
        
        // Tracking for deduplication
        this.lastLetter = null;
        this.lastLetterTime = 0;
        
        // Pause detection
        this.lastPredictionTime = 0;
        this.pauseCheckInterval = null;
        this.isPaused = false;
        
        // Event callbacks
        this.onWordComplete = null;
        this.onSentenceUpdate = null;
        this.onLetterAdded = null;
        this.onPauseProgress = null;  // For visual countdown
        
        console.log('[WordFormation] Module created');
    }
    
    /**
     * Initialize the word formation module.
     * @param {string} mappingPath - Path to urdu_mapping.json
     */
    async init(mappingPath = 'models/urdu_mapping.json') {
        try {
            console.log('[WordFormation] Loading Urdu character mapping...');
            
            const response = await fetch(mappingPath);
            if (!response.ok) {
                throw new Error(`Failed to load mapping: ${response.status}`);
            }
            
            this.urduMapping = await response.json();
            
            console.log(`[WordFormation] Loaded ${Object.keys(this.urduMapping).length} character mappings`);
            
            // Start pause detection loop
            this.startPauseDetection();
            
            this.isReady = true;
            return true;
            
        } catch (error) {
            console.error('[WordFormation] Initialization failed:', error);
            return false;
        }
    }
    
    /**
     * Process a new prediction from the ONNX predictor.
     * @param {Object} prediction - { label, confidence, timestamp }
     */
    processPrediction(prediction) {
        if (!this.isReady) {
            console.warn('[WordFormation] Not ready yet');
            return;
        }
        
        const now = Date.now();
        const label = prediction.label;
        
        // Check for deduplication
        const timeSinceLastLetter = now - this.lastLetterTime;
        const isSameLetter = label === this.lastLetter;
        
        // Only add if: different letter OR enough time has passed
        if (isSameLetter && timeSinceLastLetter < this.config.minLetterGapMs) {
            // Skip - same letter too soon (user is holding the sign)
            this.lastPredictionTime = now;  // Still update for pause detection
            return;
        }
        
        // Get Urdu character
        const urduChar = this.urduMapping[label];
        if (!urduChar) {
            console.warn(`[WordFormation] No mapping for label: ${label}`);
            return;
        }
        
        // Add letter to current word
        this.currentWord.push(urduChar);
        this.currentWordRomanized.push(label);
        
        // Update tracking
        this.lastLetter = label;
        this.lastLetterTime = now;
        this.lastPredictionTime = now;
        this.isPaused = false;
        
        console.log(`[WordFormation] Added letter: ${label} -> ${urduChar}`);
        console.log(`[WordFormation] Current word: ${this.getCurrentWordUrdu()}`);
        
        // Emit letter added event
        if (this.onLetterAdded) {
            this.onLetterAdded({
                letter: urduChar,
                romanized: label,
                currentWord: this.getCurrentWordUrdu(),
                currentWordRomanized: this.currentWordRomanized.join('-')
            });
        }
        
        // Update sentence display
        this.emitSentenceUpdate();
    }
    
    /**
     * Start the pause detection loop.
     */
    startPauseDetection() {
        if (this.pauseCheckInterval) {
            clearInterval(this.pauseCheckInterval);
        }
        
        // Check every 100ms for smooth countdown
        this.pauseCheckInterval = setInterval(() => {
            this.checkPause();
        }, 100);
        
        console.log('[WordFormation] Pause detection started');
    }
    
    /**
     * Stop the pause detection loop.
     */
    stopPauseDetection() {
        if (this.pauseCheckInterval) {
            clearInterval(this.pauseCheckInterval);
            this.pauseCheckInterval = null;
        }
    }
    
    /**
     * Check if enough time has passed to complete the current word.
     */
    checkPause() {
        if (this.currentWord.length === 0) {
            // No word being formed
            return;
        }
        
        const now = Date.now();
        const timeSinceLastPrediction = now - this.lastPredictionTime;
        
        // Calculate pause progress (0 to 1)
        const pauseProgress = Math.min(1, timeSinceLastPrediction / this.config.pauseThresholdMs);
        
        // Emit progress for visual indicator
        if (this.config.showPauseIndicator && this.onPauseProgress) {
            this.onPauseProgress({
                progress: pauseProgress,
                remainingMs: Math.max(0, this.config.pauseThresholdMs - timeSinceLastPrediction),
                currentWord: this.getCurrentWordUrdu()
            });
        }
        
        // Check if pause threshold reached
        if (timeSinceLastPrediction >= this.config.pauseThresholdMs && !this.isPaused) {
            this.isPaused = true;
            this.completeCurrentWord();
        }
    }
    
    /**
     * Complete the current word and add to sentence.
     */
    completeCurrentWord() {
        if (this.currentWord.length === 0) {
            return;
        }
        
        const wordUrdu = this.getCurrentWordUrdu();
        const wordRomanized = this.currentWordRomanized.join('-');
        
        console.log(`[WordFormation] Word complete: ${wordUrdu} (${wordRomanized})`);
        
        // Add to completed words
        this.completedWords.push({
            urdu: wordUrdu,
            romanized: wordRomanized,
            letters: [...this.currentWord],
            timestamp: Date.now()
        });
        
        // Limit sentence length
        if (this.completedWords.length > this.config.maxWordsInSentence) {
            this.completedWords.shift();
        }
        
        // Emit word complete event
        if (this.onWordComplete) {
            this.onWordComplete({
                word: wordUrdu,
                romanized: wordRomanized,
                letters: [...this.currentWord]
            });
        }
        
        // Clear current word
        this.currentWord = [];
        this.currentWordRomanized = [];
        this.lastLetter = null;
        
        // Update sentence display
        this.emitSentenceUpdate();
    }
    
    /**
     * Force complete the current word (e.g., when user clicks a button).
     */
    forceCompleteWord() {
        if (this.currentWord.length > 0) {
            this.completeCurrentWord();
        }
    }
    
    /**
     * Add a space (complete current word without adding new letter).
     */
    addSpace() {
        this.forceCompleteWord();
    }
    
    /**
     * Delete the last letter from current word.
     */
    deleteLastLetter() {
        if (this.currentWord.length > 0) {
            const removed = this.currentWord.pop();
            this.currentWordRomanized.pop();
            
            console.log(`[WordFormation] Deleted letter: ${removed}`);
            
            // Reset last letter tracking
            if (this.currentWord.length > 0) {
                this.lastLetter = this.currentWordRomanized[this.currentWordRomanized.length - 1];
            } else {
                this.lastLetter = null;
            }
            
            this.emitSentenceUpdate();
        }
    }
    
    /**
     * Delete the last completed word.
     */
    deleteLastWord() {
        if (this.completedWords.length > 0) {
            const removed = this.completedWords.pop();
            console.log(`[WordFormation] Deleted word: ${removed.urdu}`);
            this.emitSentenceUpdate();
        }
    }
    
    /**
     * Clear all - current word and sentence.
     */
    clearAll() {
        this.currentWord = [];
        this.currentWordRomanized = [];
        this.completedWords = [];
        this.lastLetter = null;
        this.lastLetterTime = 0;
        this.lastPredictionTime = 0;
        this.isPaused = false;
        
        console.log('[WordFormation] Cleared all');
        this.emitSentenceUpdate();
    }
    
    /**
     * Get the current word being formed (Urdu).
     */
    getCurrentWordUrdu() {
        return this.currentWord.join('');
    }
    
    /**
     * Get all completed words as Urdu sentence.
     */
    getSentenceUrdu() {
        return this.completedWords.map(w => w.urdu).join(' ');
    }
    
    /**
     * Get full display text (sentence + current word).
     */
    getFullDisplayText() {
        const sentence = this.getSentenceUrdu();
        const current = this.getCurrentWordUrdu();
        
        if (sentence && current) {
            return sentence + ' ' + current;
        } else if (sentence) {
            return sentence;
        } else if (current) {
            return current;
        }
        return '';
    }
    
    /**
     * Emit sentence update event.
     */
    emitSentenceUpdate() {
        if (this.onSentenceUpdate) {
            this.onSentenceUpdate({
                currentWord: this.getCurrentWordUrdu(),
                currentWordRomanized: this.currentWordRomanized.join('-'),
                sentence: this.getSentenceUrdu(),
                fullText: this.getFullDisplayText(),
                wordCount: this.completedWords.length,
                letterCount: this.currentWord.length
            });
        }
    }
    
    /**
     * Copy sentence to clipboard.
     */
    async copyToClipboard() {
        const text = this.getFullDisplayText();
        if (!text) {
            return false;
        }
        
        try {
            await navigator.clipboard.writeText(text);
            console.log('[WordFormation] Copied to clipboard:', text);
            return true;
        } catch (error) {
            console.error('[WordFormation] Clipboard copy failed:', error);
            return false;
        }
    }
    
    /**
     * Get romanized label for an Urdu character.
     * @param {string} urduChar - Urdu character
     * @returns {string|null} Romanized label or null
     */
    getRomanizedLabel(urduChar) {
        for (const [label, char] of Object.entries(this.urduMapping)) {
            if (char === urduChar) {
                return label;
            }
        }
        return null;
    }
    
    /**
     * Get Urdu character for a romanized label.
     * @param {string} label - Romanized label (e.g., "Alif")
     * @returns {string|null} Urdu character or null
     */
    getUrduChar(label) {
        return this.urduMapping[label] || null;
    }
    
    /**
     * Cleanup when done.
     */
    destroy() {
        this.stopPauseDetection();
        this.clearAll();
        console.log('[WordFormation] Destroyed');
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.WordFormation = WordFormation;
}

