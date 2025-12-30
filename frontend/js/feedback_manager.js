/**
 * Feedback Manager for Browser-Only PSL Recognition
 * 
 * Handles feedback storage using LocalStorage and IndexedDB.
 * Provides functionality for:
 * - Storing feedback metadata (LocalStorage ~5MB)
 * - Storing landmark sequences for corrections (IndexedDB ~50MB+)
 * - Dynamic threshold adjustment per sign
 * - Confusion tracking and hints
 * - Export functionality for retraining data
 */

class FeedbackManager {
    constructor(options = {}) {
        // Storage keys
        this.FEEDBACK_META_KEY = 'psl_feedback_meta';
        this.FEEDBACK_STATS_KEY = 'psl_feedback_stats';
        this.THRESHOLD_ADJUSTMENTS_KEY = 'psl_threshold_adjustments';
        this.CONFUSION_MATRIX_KEY = 'psl_confusion_matrix';
        
        // IndexedDB config
        this.DB_NAME = 'PSLFeedbackDB';
        this.DB_VERSION = 1;
        this.SAMPLES_STORE = 'samples';
        
        // IndexedDB instance
        this.db = null;
        
        // Configuration
        this.config = {
            minFeedbackForAdjustment: options.minFeedbackForAdjustment || 5,
            thresholdAdjustmentStep: options.thresholdAdjustmentStep || 0.05,
            minThreshold: options.minThreshold || 0.4,
            maxThreshold: options.maxThreshold || 0.95,
            confusionHintThreshold: options.confusionHintThreshold || 2,  // Show hint after 2 confusions
        };
        
        // Current session stats
        this.sessionStats = {
            correct: 0,
            incorrect: 0,
            corrections: {}
        };
        
        // Event callbacks
        this.onFeedbackSaved = null;
        this.onThresholdAdjusted = null;
        this.onConfusionHint = null;
        
        // Cloud sync (optional Firebase integration)
        this.cloudSync = null;
        this.cloudSyncEnabled = options.enableCloudSync !== false;
    }
    
    /**
     * Initialize the feedback manager.
     */
    async init() {
        try {
            console.log('[FeedbackManager] Initializing...');
            
            // Initialize IndexedDB for sample storage
            await this._initIndexedDB();
            
            // Load existing stats
            this._loadStats();
            
            // Initialize cloud sync (if available)
            if (this.cloudSyncEnabled && typeof CloudSync !== 'undefined') {
                this.cloudSync = new CloudSync();
                const syncReady = await this.cloudSync.init();
                if (syncReady) {
                    console.log('[FeedbackManager] Cloud sync enabled');
                } else {
                    console.log('[FeedbackManager] Cloud sync not configured (local only)');
                }
            }
            
            console.log('[FeedbackManager] Initialized successfully');
            return true;
            
        } catch (error) {
            console.error('[FeedbackManager] Initialization failed:', error);
            return false;
        }
    }
    
    /**
     * Initialize IndexedDB for storing landmark samples.
     */
    async _initIndexedDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.DB_NAME, this.DB_VERSION);
            
            request.onerror = (event) => {
                console.error('[FeedbackManager] IndexedDB error:', event.target.error);
                // Continue without IndexedDB - localStorage will still work
                resolve();
            };
            
            request.onsuccess = (event) => {
                this.db = event.target.result;
                console.log('[FeedbackManager] IndexedDB opened');
                resolve();
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // Create samples store for landmark sequences
                if (!db.objectStoreNames.contains(this.SAMPLES_STORE)) {
                    const store = db.createObjectStore(this.SAMPLES_STORE, {
                        keyPath: 'id',
                        autoIncrement: true
                    });
                    
                    // Create indexes for querying
                    store.createIndex('predictedLabel', 'predictedLabel', { unique: false });
                    store.createIndex('correctLabel', 'correctLabel', { unique: false });
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                    store.createIndex('isCorrection', 'isCorrection', { unique: false });
                    
                    console.log('[FeedbackManager] IndexedDB store created');
                }
            };
        });
    }
    
    /**
     * Load existing statistics from localStorage.
     */
    _loadStats() {
        try {
            const stored = localStorage.getItem(this.FEEDBACK_STATS_KEY);
            if (stored) {
                const stats = JSON.parse(stored);
                console.log('[FeedbackManager] Loaded stats:', stats);
            }
        } catch (e) {
            console.warn('[FeedbackManager] Could not load stats:', e);
        }
    }
    
    /**
     * Record correct feedback for a prediction.
     * @param {string} label - The predicted label
     * @param {number} confidence - The confidence score
     * @param {Array} landmarks - Optional landmark sequence for sample storage
     */
    async recordCorrect(label, confidence, landmarks = null) {
        try {
            // Update session stats
            this.sessionStats.correct++;
            
            // Store metadata in localStorage
            const metadata = {
                type: 'correct',
                label: label,
                confidence: confidence,
                timestamp: Date.now()
            };
            
            this._appendToLocalStorage(this.FEEDBACK_META_KEY, metadata);
            
            // Update per-sign stats
            this._updateSignStats(label, true, confidence);
            
            // Optionally adjust threshold (if consistently correct at low confidence)
            this._checkThresholdAdjustment(label, true, confidence);
            
            console.log(`[FeedbackManager] Recorded CORRECT: ${label} (${(confidence * 100).toFixed(1)}%)`);
            
            // Add to cloud sync queue
            if (this.cloudSync && this.cloudSync.isInitialized) {
                this.cloudSync.addToQueue({
                    type: 'correct',
                    label: label,
                    predictedLabel: label,
                    confidence: confidence,
                    isCorrect: true,
                    timestamp: Date.now()
                });
            }
            
            if (this.onFeedbackSaved) {
                this.onFeedbackSaved({ type: 'correct', label, confidence });
            }
            
            return true;
            
        } catch (error) {
            console.error('[FeedbackManager] Error recording correct feedback:', error);
            return false;
        }
    }
    
    /**
     * Record incorrect feedback with correction.
     * @param {string} predictedLabel - What the model predicted
     * @param {string} correctLabel - What the user actually signed
     * @param {number} confidence - The confidence score of wrong prediction
     * @param {Array} landmarks - Landmark sequence for retraining (optional)
     */
    async recordIncorrect(predictedLabel, correctLabel, confidence, landmarks = null) {
        try {
            // Update session stats
            this.sessionStats.incorrect++;
            if (!this.sessionStats.corrections[predictedLabel]) {
                this.sessionStats.corrections[predictedLabel] = {};
            }
            if (!this.sessionStats.corrections[predictedLabel][correctLabel]) {
                this.sessionStats.corrections[predictedLabel][correctLabel] = 0;
            }
            this.sessionStats.corrections[predictedLabel][correctLabel]++;
            
            // Store metadata in localStorage
            const metadata = {
                type: 'incorrect',
                predictedLabel: predictedLabel,
                correctLabel: correctLabel,
                confidence: confidence,
                timestamp: Date.now()
            };
            
            this._appendToLocalStorage(this.FEEDBACK_META_KEY, metadata);
            
            // Update confusion matrix
            this._updateConfusionMatrix(predictedLabel, correctLabel);
            
            // Update per-sign stats
            this._updateSignStats(predictedLabel, false, confidence);
            
            // Check for confusion pattern and emit hint
            this._checkConfusionPattern(predictedLabel, correctLabel);
            
            // Adjust threshold for this sign
            this._checkThresholdAdjustment(predictedLabel, false, confidence);
            
            // Store landmark sample in IndexedDB (for retraining)
            if (landmarks && landmarks.length > 0 && this.db) {
                await this._storeSample({
                    predictedLabel: predictedLabel,
                    correctLabel: correctLabel,
                    confidence: confidence,
                    landmarks: landmarks,
                    timestamp: Date.now(),
                    isCorrection: true
                });
            }
            
            console.log(`[FeedbackManager] Recorded INCORRECT: ${predictedLabel} -> ${correctLabel}`);
            
            // Add to cloud sync queue
            if (this.cloudSync && this.cloudSync.isInitialized) {
                this.cloudSync.addToQueue({
                    type: 'incorrect',
                    predictedLabel: predictedLabel,
                    correctLabel: correctLabel,
                    confidence: confidence,
                    isCorrect: false,
                    timestamp: Date.now()
                });
            }
            
            if (this.onFeedbackSaved) {
                this.onFeedbackSaved({ 
                    type: 'incorrect', 
                    predictedLabel, 
                    correctLabel, 
                    confidence 
                });
            }
            
            return true;
            
        } catch (error) {
            console.error('[FeedbackManager] Error recording incorrect feedback:', error);
            return false;
        }
    }
    
    /**
     * Append item to localStorage array.
     */
    _appendToLocalStorage(key, item) {
        try {
            let data = [];
            const stored = localStorage.getItem(key);
            if (stored) {
                data = JSON.parse(stored);
            }
            
            data.push(item);
            
            // Keep only last 1000 entries to prevent overflow
            if (data.length > 1000) {
                data = data.slice(-1000);
            }
            
            localStorage.setItem(key, JSON.stringify(data));
            
        } catch (e) {
            console.warn('[FeedbackManager] LocalStorage write failed:', e);
            // If quota exceeded, try to clear old data
            if (e.name === 'QuotaExceededError') {
                this._clearOldData();
            }
        }
    }
    
    /**
     * Store sample in IndexedDB.
     */
    async _storeSample(sample) {
        if (!this.db) return;
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.SAMPLES_STORE], 'readwrite');
            const store = transaction.objectStore(this.SAMPLES_STORE);
            
            const request = store.add(sample);
            
            request.onsuccess = () => {
                console.log('[FeedbackManager] Sample stored in IndexedDB');
                resolve(true);
            };
            
            request.onerror = (event) => {
                console.error('[FeedbackManager] IndexedDB store failed:', event.target.error);
                resolve(false);
            };
        });
    }
    
    /**
     * Update per-sign statistics.
     */
    _updateSignStats(label, isCorrect, confidence) {
        try {
            let stats = {};
            const stored = localStorage.getItem(this.FEEDBACK_STATS_KEY);
            if (stored) {
                stats = JSON.parse(stored);
            }
            
            if (!stats[label]) {
                stats[label] = {
                    correct: 0,
                    incorrect: 0,
                    totalConfidence: 0,
                    count: 0
                };
            }
            
            if (isCorrect) {
                stats[label].correct++;
            } else {
                stats[label].incorrect++;
            }
            stats[label].totalConfidence += confidence;
            stats[label].count++;
            
            localStorage.setItem(this.FEEDBACK_STATS_KEY, JSON.stringify(stats));
            
        } catch (e) {
            console.warn('[FeedbackManager] Stats update failed:', e);
        }
    }
    
    /**
     * Update confusion matrix.
     */
    _updateConfusionMatrix(predicted, actual) {
        try {
            let matrix = {};
            const stored = localStorage.getItem(this.CONFUSION_MATRIX_KEY);
            if (stored) {
                matrix = JSON.parse(stored);
            }
            
            if (!matrix[predicted]) {
                matrix[predicted] = {};
            }
            if (!matrix[predicted][actual]) {
                matrix[predicted][actual] = 0;
            }
            
            matrix[predicted][actual]++;
            
            localStorage.setItem(this.CONFUSION_MATRIX_KEY, JSON.stringify(matrix));
            
        } catch (e) {
            console.warn('[FeedbackManager] Confusion matrix update failed:', e);
        }
    }
    
    /**
     * Check for confusion patterns and emit hint.
     */
    _checkConfusionPattern(predicted, actual) {
        try {
            const stored = localStorage.getItem(this.CONFUSION_MATRIX_KEY);
            if (!stored) return;
            
            const matrix = JSON.parse(stored);
            const confusionCount = matrix[predicted]?.[actual] || 0;
            
            if (confusionCount >= this.config.confusionHintThreshold) {
                const hint = `"${predicted}" is often confused with "${actual}". ` +
                    `Try to make the sign more distinct.`;
                
                console.log(`[FeedbackManager] Confusion hint: ${hint}`);
                
                if (this.onConfusionHint) {
                    this.onConfusionHint({
                        predicted: predicted,
                        actual: actual,
                        count: confusionCount,
                        hint: hint
                    });
                }
            }
            
        } catch (e) {
            console.warn('[FeedbackManager] Confusion check failed:', e);
        }
    }
    
    /**
     * Check and adjust threshold based on feedback.
     */
    _checkThresholdAdjustment(label, isCorrect, confidence) {
        try {
            let stats = {};
            const statsStored = localStorage.getItem(this.FEEDBACK_STATS_KEY);
            if (statsStored) {
                stats = JSON.parse(statsStored);
            }
            
            const signStats = stats[label];
            if (!signStats || signStats.count < this.config.minFeedbackForAdjustment) {
                return;  // Not enough data
            }
            
            const accuracy = signStats.correct / signStats.count;
            const avgConfidence = signStats.totalConfidence / signStats.count;
            
            // Load current threshold adjustments
            let adjustments = {};
            const adjStored = localStorage.getItem(this.THRESHOLD_ADJUSTMENTS_KEY);
            if (adjStored) {
                adjustments = JSON.parse(adjStored);
            }
            
            const currentAdjustment = adjustments[label] || 0;
            let newAdjustment = currentAdjustment;
            
            // If accuracy is low, increase threshold
            if (accuracy < 0.7) {
                newAdjustment = Math.min(
                    currentAdjustment + this.config.thresholdAdjustmentStep,
                    this.config.maxThreshold - 0.7  // Max adjustment
                );
            }
            // If accuracy is high and confidence is low, decrease threshold
            else if (accuracy > 0.9 && avgConfidence < 0.75) {
                newAdjustment = Math.max(
                    currentAdjustment - this.config.thresholdAdjustmentStep,
                    this.config.minThreshold - 0.7  // Min adjustment
                );
            }
            
            if (newAdjustment !== currentAdjustment) {
                adjustments[label] = newAdjustment;
                localStorage.setItem(this.THRESHOLD_ADJUSTMENTS_KEY, JSON.stringify(adjustments));
                
                console.log(`[FeedbackManager] Threshold adjusted for ${label}: ${(0.7 + newAdjustment).toFixed(2)}`);
                
                if (this.onThresholdAdjusted) {
                    this.onThresholdAdjusted({
                        label: label,
                        newThreshold: 0.7 + newAdjustment,
                        adjustment: newAdjustment,
                        accuracy: accuracy
                    });
                }
            }
            
        } catch (e) {
            console.warn('[FeedbackManager] Threshold adjustment failed:', e);
        }
    }
    
    /**
     * Get adjusted thresholds for all signs.
     * @returns {Object} Map of label -> adjusted threshold
     */
    getAdjustedThresholds() {
        try {
            let adjustments = {};
            const stored = localStorage.getItem(this.THRESHOLD_ADJUSTMENTS_KEY);
            if (stored) {
                adjustments = JSON.parse(stored);
            }
            
            const result = {};
            for (const label in adjustments) {
                result[label] = 0.7 + adjustments[label];  // Base threshold + adjustment
            }
            
            return result;
            
        } catch (e) {
            return {};
        }
    }
    
    /**
     * Get feedback statistics.
     */
    getStats() {
        try {
            let stats = {};
            const stored = localStorage.getItem(this.FEEDBACK_STATS_KEY);
            if (stored) {
                stats = JSON.parse(stored);
            }
            
            // Calculate totals
            let totalCorrect = 0;
            let totalIncorrect = 0;
            
            for (const label in stats) {
                totalCorrect += stats[label].correct;
                totalIncorrect += stats[label].incorrect;
            }
            
            return {
                perSign: stats,
                total: {
                    correct: totalCorrect,
                    incorrect: totalIncorrect,
                    accuracy: totalCorrect / (totalCorrect + totalIncorrect) || 0
                },
                session: this.sessionStats
            };
            
        } catch (e) {
            return {
                perSign: {},
                total: { correct: 0, incorrect: 0, accuracy: 0 },
                session: this.sessionStats
            };
        }
    }
    
    /**
     * Get confusion matrix.
     */
    getConfusionMatrix() {
        try {
            const stored = localStorage.getItem(this.CONFUSION_MATRIX_KEY);
            if (stored) {
                return JSON.parse(stored);
            }
            return {};
        } catch (e) {
            return {};
        }
    }
    
    /**
     * Export all correction samples for retraining.
     * @returns {Promise<Object>} Export data as JSON
     */
    async exportForRetraining() {
        if (!this.db) {
            console.warn('[FeedbackManager] IndexedDB not available for export');
            return null;
        }
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.SAMPLES_STORE], 'readonly');
            const store = transaction.objectStore(this.SAMPLES_STORE);
            const index = store.index('isCorrection');
            
            const request = index.getAll(true);  // Get all correction samples
            
            request.onsuccess = (event) => {
                const samples = event.target.result;
                
                const exportData = {
                    exportDate: new Date().toISOString(),
                    version: '1.0',
                    sampleCount: samples.length,
                    samples: samples.map(s => ({
                        predictedLabel: s.predictedLabel,
                        correctLabel: s.correctLabel,
                        confidence: s.confidence,
                        landmarks: s.landmarks,
                        timestamp: s.timestamp
                    }))
                };
                
                console.log(`[FeedbackManager] Exported ${samples.length} samples for retraining`);
                resolve(exportData);
            };
            
            request.onerror = (event) => {
                console.error('[FeedbackManager] Export failed:', event.target.error);
                reject(event.target.error);
            };
        });
    }
    
    /**
     * Download export data as JSON file.
     */
    async downloadExport() {
        try {
            const data = await this.exportForRetraining();
            if (!data) {
                console.warn('[FeedbackManager] No data to export');
                return false;
            }
            
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `psl_feedback_export_${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            console.log('[FeedbackManager] Export downloaded');
            return true;
            
        } catch (error) {
            console.error('[FeedbackManager] Download failed:', error);
            return false;
        }
    }
    
    /**
     * Clear old data to free up storage space.
     */
    _clearOldData() {
        try {
            // Keep only recent feedback metadata
            const stored = localStorage.getItem(this.FEEDBACK_META_KEY);
            if (stored) {
                let data = JSON.parse(stored);
                if (data.length > 500) {
                    data = data.slice(-500);
                    localStorage.setItem(this.FEEDBACK_META_KEY, JSON.stringify(data));
                    console.log('[FeedbackManager] Cleared old metadata');
                }
            }
        } catch (e) {
            console.warn('[FeedbackManager] Clear old data failed:', e);
        }
    }
    
    /**
     * Clear all feedback data.
     */
    async clearAll() {
        try {
            // Clear localStorage
            localStorage.removeItem(this.FEEDBACK_META_KEY);
            localStorage.removeItem(this.FEEDBACK_STATS_KEY);
            localStorage.removeItem(this.THRESHOLD_ADJUSTMENTS_KEY);
            localStorage.removeItem(this.CONFUSION_MATRIX_KEY);
            
            // Clear IndexedDB
            if (this.db) {
                const transaction = this.db.transaction([this.SAMPLES_STORE], 'readwrite');
                const store = transaction.objectStore(this.SAMPLES_STORE);
                store.clear();
            }
            
            // Reset session stats
            this.sessionStats = {
                correct: 0,
                incorrect: 0,
                corrections: {}
            };
            
            console.log('[FeedbackManager] All data cleared');
            return true;
            
        } catch (error) {
            console.error('[FeedbackManager] Clear failed:', error);
            return false;
        }
    }
    
    /**
     * Get sample count from IndexedDB.
     */
    async getSampleCount() {
        if (!this.db) return 0;
        
        return new Promise((resolve) => {
            const transaction = this.db.transaction([this.SAMPLES_STORE], 'readonly');
            const store = transaction.objectStore(this.SAMPLES_STORE);
            const request = store.count();
            
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => resolve(0);
        });
    }
    
    /**
     * Destroy the feedback manager.
     */
    destroy() {
        if (this.db) {
            this.db.close();
            this.db = null;
        }
        console.log('[FeedbackManager] Destroyed');
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FeedbackManager;
}

