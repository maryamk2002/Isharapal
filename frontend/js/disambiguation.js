/**
 * Sign Disambiguation Module for PSL Recognition System
 * 
 * Handles confusion between similar signs by:
 * - Showing "Did you mean...?" suggestions when confidence is ambiguous
 * - Displaying side-by-side comparison images
 * - Providing visual differentiation hints
 * - Learning from user selections to improve accuracy
 */

class SignDisambiguator {
    constructor(options = {}) {
        // Configuration
        this.config = {
            // When top 2 predictions are within this range, show disambiguation
            ambiguityThreshold: options.ambiguityThreshold || 0.15,
            // Minimum confidence for any prediction to be shown
            minConfidence: options.minConfidence || 0.35,
            // Maximum number of alternatives to show
            maxAlternatives: options.maxAlternatives || 3,
            // Images base path
            imagesBasePath: options.imagesBasePath || 'assets/signs/',
            // Auto-dismiss timeout (0 = no auto-dismiss)
            autoDismissMs: options.autoDismissMs || 8000,
            ...options
        };
        
        // State
        this.isVisible = false;
        this.currentPredictions = [];
        this.selectedSign = null;
        this.dismissTimer = null;
        
        // Known confusion pairs (loaded from feedback)
        this.confusionPairs = {};
        
        // CONTEXT-AWARE BIAS: Common Urdu letter combinations
        // If previous letter is X, letter Y is more likely than Z
        this.urduBigrams = {
            "Alif": ["Lam", "Bay", "Nuun"],      // الف often followed by لام, بے, نون
            "Lam": ["Alif", "Meem", "Bay"],       // لام often followed by الف, میم, بے
            "Meem": ["Nuun", "Alif", "Bay"],      // میم often followed by نون, الف, بے
            "Nuun": ["Alif", "Tay", "Bay"],       // نون often followed by الف, تے, بے
            "Bay": ["Alif", "Nuun", "Ray"],       // بے often followed by الف, نون, رے
            "Tay": ["Alif", "Bay", "Nuun"],       // تے often followed by الف, بے, نون
            "Ray": ["Alif", "Hay", "Nuun"],       // رے often followed by الف, ہے, نون
            "Seen": ["Tay", "Alif", "Nuun"],      // سین often followed by تے, الف, نون
            "Kaf": ["Alif", "Tay", "Ray"],        // کاف often followed by الف, تے, رے
            "Hay": ["Alif", "Nuun", "Bay"]        // ہے often followed by الف, نون, بے
        };
        
        // Visual hints for commonly confused signs
        this.differentiationHints = {
            "1-Hay_2-Hay": {
                urdu: "ہ کے لیے انگلیاں پھیلائیں، ھ کے لیے بند رکھیں",
                english: "Spread fingers for ہ, keep closed for ھ"
            },
            "Seen_Sheen": {
                urdu: "س کے لیے تین انگلیاں، ش کے لیے انگلیاں ہلائیں",
                english: "Three fingers still for س, wave for ش"
            },
            "Tay_Taay": {
                urdu: "ت کے لیے انگشت اوپر، ط کے لیے نیچے",
                english: "Point up for ت, down for ط"
            },
            "Dal_Daal": {
                urdu: "د سیدھا، ڈ گول حرکت",
                english: "Straight for د, circular motion for ڈ"
            },
            "Ray_Aray": {
                urdu: "ر سیدھا، ڑ ہلاتے ہوئے",
                english: "Steady for ر, flick for ڑ"
            },
            "Zay_Zaey": {
                urdu: "ز چھوٹی حرکت، ض بڑی حرکت",
                english: "Small motion for ز, larger for ض"
            },
            "Seen_Say": {
                urdu: "س تین انگلیاں، ث دو انگلیاں",
                english: "Three fingers for س, two for ث"
            },
            "Byeh_Cyeh": {
                urdu: "ے افقی، ی عمودی",
                english: "Horizontal for ے, vertical for ی"
            }
        };
        
        // Callbacks
        this.onSelection = null;
        this.onDismiss = null;
        
        // DOM Elements
        this.elements = {};
    }
    
    /**
     * Initialize the disambiguator
     */
    init() {
        this.initDOMElements();
        this.setupEventListeners();
        console.log('[Disambiguator] Initialized');
    }
    
    /**
     * Initialize DOM element references
     */
    initDOMElements() {
        this.elements = {
            container: document.getElementById('disambiguationContainer'),
            overlay: document.getElementById('disambiguationOverlay'),
            title: document.getElementById('disambiguationTitle'),
            hint: document.getElementById('disambiguationHint'),
            optionsGrid: document.getElementById('disambiguationOptions'),
            dismissBtn: document.getElementById('disambiguationDismiss')
        };
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        if (this.elements.dismissBtn) {
            this.elements.dismissBtn.addEventListener('click', () => this.dismiss());
        }
        
        if (this.elements.overlay) {
            this.elements.overlay.addEventListener('click', () => this.dismiss());
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (!this.isVisible) return;
            
            // Number keys 1-3 to select option
            if (e.key >= '1' && e.key <= '3') {
                const idx = parseInt(e.key) - 1;
                if (idx < this.currentPredictions.length) {
                    this.selectOption(this.currentPredictions[idx].label);
                }
            }
            
            // Escape to dismiss
            if (e.key === 'Escape') {
                this.dismiss();
            }
        });
    }
    
    /**
     * Apply context bias based on previous letter (Urdu bigram patterns)
     * @param {Array} predictions - Array of {label, confidence}
     * @param {string} previousLetter - The previous recognized letter
     * @returns {Array} Re-ranked predictions with context bias applied
     */
    applyContextBias(predictions, previousLetter) {
        if (!previousLetter || !predictions || predictions.length < 2) {
            return predictions;
        }
        
        const likelyFollowers = this.urduBigrams[previousLetter] || [];
        if (likelyFollowers.length === 0) return predictions;
        
        // Apply small confidence boost to likely followers
        const biasAmount = 0.05; // 5% boost for common combinations
        
        return predictions.map(pred => {
            if (likelyFollowers.includes(pred.label)) {
                return {
                    ...pred,
                    confidence: Math.min(pred.confidence + biasAmount, 1.0),
                    hasBias: true
                };
            }
            return pred;
        }).sort((a, b) => b.confidence - a.confidence);
    }
    
    /**
     * Check if predictions are ambiguous and show disambiguation UI
     * @param {Array} predictions - Array of {label, confidence} sorted by confidence
     * @param {string} previousLetter - Optional: previous letter for context bias
     * @returns {boolean} Whether disambiguation was triggered
     */
    checkAmbiguity(predictions, previousLetter = null) {
        if (!predictions || predictions.length < 2) return false;
        
        // Apply context bias if previous letter is known
        if (previousLetter) {
            predictions = this.applyContextBias(predictions, previousLetter);
        }
        
        const top1 = predictions[0];
        const top2 = predictions[1];
        
        // Check if top 2 are too close
        const confidenceDiff = top1.confidence - top2.confidence;
        
        // Also check absolute confidence levels
        if (top1.confidence < this.config.minConfidence) return false;
        
        // If difference is small AND both are reasonable, show disambiguation
        if (confidenceDiff <= this.config.ambiguityThreshold && 
            top2.confidence >= this.config.minConfidence) {
            
            // Get top N alternatives
            const alternatives = predictions
                .slice(0, this.config.maxAlternatives)
                .filter(p => p.confidence >= this.config.minConfidence);
            
            this.show(alternatives);
            return true;
        }
        
        return false;
    }
    
    /**
     * Show disambiguation UI with options
     * @param {Array} predictions - Array of {label, confidence}
     */
    show(predictions) {
        this.currentPredictions = predictions;
        this.isVisible = true;
        
        // Clear any existing timer
        this.clearDismissTimer();
        
        // Update UI
        this.renderOptions(predictions);
        this.showHint(predictions);
        
        // Show container
        if (this.elements.container) {
            this.elements.container.classList.add('visible');
        }
        
        // Set auto-dismiss timer
        if (this.config.autoDismissMs > 0) {
            this.dismissTimer = setTimeout(() => {
                this.dismiss();
            }, this.config.autoDismissMs);
        }
        
        console.log('[Disambiguator] Showing options:', predictions.map(p => p.label).join(', '));
    }
    
    /**
     * Render option cards
     */
    renderOptions(predictions) {
        if (!this.elements.optionsGrid) return;
        
        this.elements.optionsGrid.innerHTML = predictions.map((pred, idx) => {
            const confidence = Math.round(pred.confidence * 100);
            const webpUrl = `${this.config.imagesBasePath}${pred.label}.webp`;
            const jpgUrl = `${this.config.imagesBasePath}${pred.label}.jpg`;
            
            return `
                <div class="disambiguation-option" data-sign="${pred.label}" tabindex="0">
                    <div class="option-number">${idx + 1}</div>
                    <div class="option-image-container">
                        <picture>
                            <source srcset="${webpUrl}" type="image/webp">
                            <img src="${jpgUrl}" alt="${pred.label}" class="option-image" loading="eager">
                        </picture>
                    </div>
                    <div class="option-info">
                        <div class="option-label">${pred.label}</div>
                        <div class="option-confidence">${confidence}%</div>
                    </div>
                </div>
            `;
        }).join('');
        
        // Add click handlers to options
        this.elements.optionsGrid.querySelectorAll('.disambiguation-option').forEach(option => {
            option.addEventListener('click', () => {
                this.selectOption(option.dataset.sign);
            });
            
            option.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    this.selectOption(option.dataset.sign);
                }
            });
        });
    }
    
    /**
     * Show differentiation hint for confused pair
     */
    showHint(predictions) {
        if (!this.elements.hint) return;
        
        // Look for known confusion pair
        const labels = predictions.map(p => p.label);
        let hint = null;
        
        for (let i = 0; i < labels.length; i++) {
            for (let j = i + 1; j < labels.length; j++) {
                const key1 = `${labels[i]}_${labels[j]}`;
                const key2 = `${labels[j]}_${labels[i]}`;
                
                hint = this.differentiationHints[key1] || this.differentiationHints[key2];
                if (hint) break;
            }
            if (hint) break;
        }
        
        if (hint) {
            this.elements.hint.innerHTML = `
                <div class="hint-urdu">${hint.urdu}</div>
                <div class="hint-english">${hint.english}</div>
            `;
            this.elements.hint.style.display = 'block';
        } else {
            this.elements.hint.style.display = 'none';
        }
    }
    
    /**
     * Handle user selection of an option
     */
    selectOption(label) {
        this.selectedSign = label;
        this.clearDismissTimer();
        
        // Highlight selected option
        if (this.elements.optionsGrid) {
            this.elements.optionsGrid.querySelectorAll('.disambiguation-option').forEach(option => {
                option.classList.toggle('selected', option.dataset.sign === label);
            });
        }
        
        // Notify callback
        if (this.onSelection) {
            // Find the prediction data
            const prediction = this.currentPredictions.find(p => p.label === label);
            this.onSelection({
                selected: label,
                confidence: prediction?.confidence || 0,
                alternatives: this.currentPredictions.filter(p => p.label !== label),
                wasAmbiguous: true
            });
        }
        
        console.log(`[Disambiguator] User selected: ${label}`);
        
        // Hide after short delay
        setTimeout(() => this.hide(), 500);
    }
    
    /**
     * Dismiss without selection (use top prediction)
     */
    dismiss() {
        this.clearDismissTimer();
        
        if (this.onDismiss) {
            this.onDismiss({
                predictions: this.currentPredictions,
                autoSelected: this.currentPredictions[0]?.label
            });
        }
        
        this.hide();
    }
    
    /**
     * Hide the disambiguation UI
     */
    hide() {
        this.isVisible = false;
        this.currentPredictions = [];
        this.selectedSign = null;
        
        if (this.elements.container) {
            this.elements.container.classList.remove('visible');
        }
    }
    
    /**
     * Clear auto-dismiss timer
     */
    clearDismissTimer() {
        if (this.dismissTimer) {
            clearTimeout(this.dismissTimer);
            this.dismissTimer = null;
        }
    }
    
    /**
     * Update confusion pairs from feedback data
     * @param {Object} confusionMatrix - From feedback manager
     */
    updateConfusionPairs(confusionMatrix) {
        this.confusionPairs = {};
        
        for (const predicted in confusionMatrix) {
            for (const actual in confusionMatrix[predicted]) {
                if (predicted !== actual) {
                    const count = confusionMatrix[predicted][actual];
                    const key = [predicted, actual].sort().join('_');
                    this.confusionPairs[key] = (this.confusionPairs[key] || 0) + count;
                }
            }
        }
        
        console.log(`[Disambiguator] Updated ${Object.keys(this.confusionPairs).length} confusion pairs`);
    }
    
    /**
     * Add a custom differentiation hint
     * @param {string} sign1 - First sign
     * @param {string} sign2 - Second sign
     * @param {string} urduHint - Hint in Urdu
     * @param {string} englishHint - Hint in English
     */
    addHint(sign1, sign2, urduHint, englishHint) {
        const key = [sign1, sign2].sort().join('_');
        this.differentiationHints[key] = {
            urdu: urduHint,
            english: englishHint
        };
    }
    
    /**
     * Get all known confusion pairs
     */
    getConfusionPairs() {
        return { ...this.confusionPairs };
    }
    
    /**
     * Check if two signs are commonly confused
     */
    areCommlyConfused(sign1, sign2, threshold = 3) {
        const key = [sign1, sign2].sort().join('_');
        return (this.confusionPairs[key] || 0) >= threshold;
    }
    
    /**
     * Destroy the disambiguator
     */
    destroy() {
        this.hide();
        this.clearDismissTimer();
    }
}


// ================================================================================
// EXPORTS
// ================================================================================

if (typeof window !== 'undefined') {
    window.SignDisambiguator = SignDisambiguator;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SignDisambiguator };
}


