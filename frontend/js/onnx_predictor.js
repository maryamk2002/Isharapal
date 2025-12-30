/**
 * ONNX Predictor for Browser-Based PSL Recognition
 * 
 * This module handles model loading and inference using ONNX Runtime Web.
 * It replaces the WebSocket-based prediction with local browser inference.
 * 
 * Features:
 * - Load ONNX model and labels
 * - Sliding window buffer (matches Python predictor)
 * - Stability voting (3/5 votes)
 * - Per-sign confidence thresholds
 * - EMA smoothing for predictions
 */

class ONNXPredictor {
    constructor(options = {}) {
        // Model and session
        this.session = null;
        this.labels = [];
        this.signThresholds = {};
        this.modelInfo = null;
        this.isReady = false;
        
        // Configuration (match Python predictor_v2.py)
        this.config = {
            slidingWindowSize: options.slidingWindowSize || 45,  // Frames to buffer
            modelSequenceLength: 60,  // Model expects 60 frames
            minPredictionFrames: options.minPredictionFrames || 32,  // Min frames before predicting
            stabilityVotes: options.stabilityVotes || 5,  // Recent predictions to consider
            stabilityThreshold: options.stabilityThreshold || 3,  // Votes needed for stable
            minConfidence: options.minConfidence || 0.55,  // Default confidence threshold
            lowConfidenceThreshold: options.lowConfidenceThreshold || 0.45,  // "Searching" state
            emaAlpha: options.emaAlpha || 0.7,  // EMA smoothing (0.7 = 70% new, 30% old)
        };
        
        // Sliding window buffer
        this.frameBuffer = [];
        
        // Prediction history for stability voting
        this.predictionHistory = [];
        
        // Last stable prediction tracking
        this.lastStablePrediction = null;
        this.lastStableConfidence = 0;
        this.lastStableTime = null;
        
        // EMA smoothing
        this.emaProbabilities = null;
        this.lastRawPrediction = null;
        
        // Performance stats
        this.stats = {
            totalPredictions: 0,
            successfulPredictions: 0,
            avgPredictionTime: 0,
            avgConfidence: 0,
            bufferResets: 0
        };
        
        // Event callbacks
        this.onPrediction = null;
        this.onReady = null;
        this.onError = null;
        this.onLoadProgress = null;
    }
    
    /**
     * Initialize the predictor by loading model and labels.
     * @param {string} modelPath - Path to ONNX model file
     * @param {string} labelsPath - Path to labels JSON file
     * @param {string} thresholdsPath - Path to sign thresholds JSON file
     */
    async init(modelPath = 'models/psl_model_v2.onnx', 
               labelsPath = 'models/psl_labels.json',
               thresholdsPath = 'models/sign_thresholds.json') {
        try {
            console.log('[ONNXPredictor] Initializing...');
            this._emitProgress('Loading model files...', 0);
            
            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime Web (ort) not loaded. Add the CDN script tag.');
            }
            
            // Load labels first (small file)
            this._emitProgress('Loading labels...', 10);
            const labelsResponse = await fetch(labelsPath);
            if (!labelsResponse.ok) {
                throw new Error(`Failed to load labels: ${labelsResponse.statusText}`);
            }
            const labelsData = await labelsResponse.json();
            this.labels = labelsData.labels;
            this.modelInfo = labelsData.model_info || {};
            this.config.modelSequenceLength = labelsData.sequence_length || 60;
            console.log(`[ONNXPredictor] Loaded ${this.labels.length} labels`);
            
            // Load sign thresholds
            this._emitProgress('Loading thresholds...', 20);
            try {
                const thresholdsResponse = await fetch(thresholdsPath);
                if (thresholdsResponse.ok) {
                    this.signThresholds = await thresholdsResponse.json();
                    console.log(`[ONNXPredictor] Loaded ${Object.keys(this.signThresholds).length} sign thresholds`);
                }
            } catch (e) {
                console.warn('[ONNXPredictor] Could not load sign thresholds, using defaults');
            }
            
            // Load ONNX model (larger file, ~11MB)
            this._emitProgress('Loading ONNX model (~11MB)...', 30);
            console.log('[ONNXPredictor] Loading ONNX model...');
            
            // Configure ONNX Runtime
            const sessionOptions = {
                executionProviders: ['wasm'],  // Use WebAssembly backend
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                enableMemPattern: true
            };
            
            // Create inference session
            this.session = await ort.InferenceSession.create(modelPath, sessionOptions);
            console.log('[ONNXPredictor] ONNX model loaded successfully');
            
            // Log model input/output info
            const inputNames = this.session.inputNames;
            const outputNames = this.session.outputNames;
            console.log(`[ONNXPredictor] Input: ${inputNames}, Output: ${outputNames}`);
            
            this._emitProgress('Model ready!', 100);
            this.isReady = true;
            
            if (this.onReady) {
                this.onReady({
                    labels: this.labels,
                    numClasses: this.labels.length,
                    modelInfo: this.modelInfo
                });
            }
            
            console.log('[ONNXPredictor] Initialization complete');
            return true;
            
        } catch (error) {
            console.error('[ONNXPredictor] Initialization failed:', error);
            if (this.onError) {
                this.onError(error);
            }
            return false;
        }
    }
    
    /**
     * Add a frame to the sliding window buffer.
     * @param {Float32Array|Array} landmarks - 189-element landmark array
     */
    addFrame(landmarks) {
        if (!landmarks || landmarks.length === 0) {
            // No landmarks (no hands) - optionally clear buffer
            return;
        }
        
        // Ensure landmarks is a regular array
        const frame = Array.isArray(landmarks) ? landmarks : Array.from(landmarks);
        
        // Add to buffer
        this.frameBuffer.push(frame);
        
        // Keep buffer at sliding window size
        while (this.frameBuffer.length > this.config.slidingWindowSize) {
            this.frameBuffer.shift();
        }
    }
    
    /**
     * Make prediction on current sliding window.
     * @param {boolean} returnAllPredictions - Whether to return all class predictions
     * @returns {Object} Prediction result
     */
    async predict(returnAllPredictions = true) {
        if (!this.isReady || !this.session) {
            return {
                ready: false,
                status: 'not_initialized',
                error: 'Model not loaded'
            };
        }
        
        // Check if buffer has enough frames
        if (this.frameBuffer.length < this.config.minPredictionFrames) {
            return {
                ready: false,
                bufferSize: this.frameBuffer.length,
                minRequired: this.config.minPredictionFrames,
                status: 'collecting_frames'
            };
        }
        
        try {
            const startTime = performance.now();
            
            // Prepare sequence (pad to model's expected length)
            const sequence = this._prepareSequence(this.frameBuffer);
            
            // Create input tensor
            // Shape: [1, sequence_length, input_dim] = [1, 60, 189]
            const inputTensor = new ort.Tensor(
                'float32',
                new Float32Array(sequence.flat()),
                [1, this.config.modelSequenceLength, 189]
            );
            
            // Run inference
            const feeds = { 'input': inputTensor };
            const results = await this.session.run(feeds);
            
            // Get output probabilities
            const outputData = results['output'].data;
            
            // Apply softmax to get probabilities
            const rawProbabilities = this._softmax(Array.from(outputData));
            
            // Get raw prediction (before smoothing)
            const rawPredictedIdx = this._argmax(rawProbabilities);
            const rawPredictedLabel = this.labels[rawPredictedIdx];
            const rawConfidence = rawProbabilities[rawPredictedIdx];
            
            // Apply EMA smoothing (reset if label changed)
            if (this.lastRawPrediction !== null && rawPredictedLabel !== this.lastRawPrediction) {
                // New sign detected - reset EMA for faster response
                this.emaProbabilities = [...rawProbabilities];
            } else if (this.emaProbabilities === null) {
                this.emaProbabilities = [...rawProbabilities];
            } else {
                // EMA: new_ema = alpha * new_value + (1 - alpha) * old_ema
                for (let i = 0; i < rawProbabilities.length; i++) {
                    this.emaProbabilities[i] = 
                        this.config.emaAlpha * rawProbabilities[i] + 
                        (1 - this.config.emaAlpha) * this.emaProbabilities[i];
                }
            }
            
            this.lastRawPrediction = rawPredictedLabel;
            
            // Use smoothed probabilities for prediction
            const predictedIdx = this._argmax(this.emaProbabilities);
            const confidenceScore = this.emaProbabilities[predictedIdx];
            const predictedLabel = this.labels[predictedIdx];
            
            // Get all predictions if requested
            let allPredictions = [];
            if (returnAllPredictions) {
                allPredictions = this.labels.map((label, i) => ({
                    label: label,
                    confidence: this.emaProbabilities[i]
                })).sort((a, b) => b.confidence - a.confidence);
            }
            
            // Get per-sign threshold or use default
            const signThreshold = this.signThresholds[predictedLabel] || this.config.minConfidence;
            
            // Add to prediction history if confidence is sufficient
            if (confidenceScore >= signThreshold) {
                this.predictionHistory.push({
                    label: predictedLabel,
                    confidence: confidenceScore,
                    timestamp: Date.now()
                });
                
                // Keep only recent predictions
                while (this.predictionHistory.length > this.config.stabilityVotes) {
                    this.predictionHistory.shift();
                }
            }
            
            // Apply stability voting
            const stablePrediction = this._getStablePrediction();
            
            // Determine if this is a new stable prediction
            let isNew = false;
            if (stablePrediction !== null) {
                const [stableLabel, stableConfidence] = stablePrediction;
                
                const timeSinceLast = this.lastStableTime !== null 
                    ? (Date.now() - this.lastStableTime) / 1000 
                    : 999;
                const isDifferentLabel = stableLabel !== this.lastStablePrediction;
                const isTimeoutPassed = timeSinceLast > 30.0;  // 30 seconds
                
                isNew = isDifferentLabel || isTimeoutPassed;
                
                if (isNew) {
                    this.lastStablePrediction = stableLabel;
                    this.lastStableConfidence = stableConfidence;
                    this.lastStableTime = Date.now();
                    
                    // Clear history if different label
                    if (isDifferentLabel) {
                        this.predictionHistory = [];
                    }
                    
                    console.log(`[ONNXPredictor] NEW stable: ${stableLabel} (${(stableConfidence * 100).toFixed(1)}%)`);
                }
            }
            
            // Update statistics
            const predictionTime = performance.now() - startTime;
            this._updateStats(predictionTime, confidenceScore, true);
            
            // Determine status
            let status = 'success';
            if (confidenceScore < this.config.lowConfidenceThreshold) {
                status = 'low_confidence';
            } else if (stablePrediction === null) {
                status = 'collecting_stability';
            }
            
            // Prepare result
            const result = {
                ready: true,
                prediction: predictedLabel,
                confidence: confidenceScore,
                stablePrediction: stablePrediction ? stablePrediction[0] : null,
                stableConfidence: stablePrediction ? stablePrediction[1] : 0,
                isStable: stablePrediction !== null,
                isNew: isNew,
                isLowConfidence: confidenceScore < this.config.lowConfidenceThreshold,
                allPredictions: allPredictions,
                bufferSize: this.frameBuffer.length,
                predictionTimeMs: predictionTime,
                status: status
            };
            
            // Emit prediction event if new stable prediction
            if (isNew && this.onPrediction) {
                this.onPrediction({
                    label: stablePrediction[0],
                    confidence: stablePrediction[1],
                    timestamp: Date.now()
                });
            }
            
            return result;
            
        } catch (error) {
            console.error('[ONNXPredictor] Prediction failed:', error);
            this._updateStats(0, 0, false);
            return {
                ready: true,
                status: 'error',
                error: error.message,
                bufferSize: this.frameBuffer.length
            };
        }
    }
    
    /**
     * Prepare sequence to match model's expected length.
     * Pads with zeros at the beginning (preserves recent frames).
     */
    _prepareSequence(buffer) {
        const targetLength = this.config.modelSequenceLength;
        const currentLength = buffer.length;
        
        if (currentLength === targetLength) {
            return buffer;
        } else if (currentLength < targetLength) {
            // Pad with zeros at the beginning
            const paddingLength = targetLength - currentLength;
            const padding = Array(paddingLength).fill(null).map(() => new Array(189).fill(0));
            return [...padding, ...buffer];
        } else {
            // Truncate from the beginning (keep most recent frames)
            return buffer.slice(-targetLength);
        }
    }
    
    /**
     * Apply softmax to get probabilities.
     */
    _softmax(logits) {
        const maxLogit = Math.max(...logits);
        const exps = logits.map(x => Math.exp(x - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sumExps);
    }
    
    /**
     * Get index of maximum value.
     */
    _argmax(arr) {
        let maxIdx = 0;
        let maxVal = arr[0];
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Get stable prediction using majority voting.
     * @returns {Array|null} [label, confidence] if stable, null otherwise
     */
    _getStablePrediction() {
        if (this.predictionHistory.length < this.config.stabilityThreshold) {
            return null;
        }
        
        // Look at recent predictions
        const recentPredictions = this.predictionHistory.slice(-this.config.stabilityVotes);
        
        // Count occurrences of each label
        const labelCounts = {};
        const labelConfidences = {};
        
        for (const pred of recentPredictions) {
            const label = pred.label;
            if (!(label in labelCounts)) {
                labelCounts[label] = 0;
                labelConfidences[label] = [];
            }
            labelCounts[label]++;
            labelConfidences[label].push(pred.confidence);
        }
        
        // Find most frequent label
        let mostFrequentLabel = null;
        let maxCount = 0;
        for (const label in labelCounts) {
            if (labelCounts[label] > maxCount) {
                maxCount = labelCounts[label];
                mostFrequentLabel = label;
            }
        }
        
        // Check if it meets stability threshold
        if (maxCount >= this.config.stabilityThreshold) {
            const avgConfidence = labelConfidences[mostFrequentLabel].reduce((a, b) => a + b, 0) 
                / labelConfidences[mostFrequentLabel].length;
            return [mostFrequentLabel, avgConfidence];
        }
        
        return null;
    }
    
    /**
     * Update performance statistics.
     */
    _updateStats(predictionTime, confidence, success) {
        this.stats.totalPredictions++;
        
        if (success) {
            this.stats.successfulPredictions++;
            this.stats.avgPredictionTime = 
                this.stats.avgPredictionTime * 0.9 + predictionTime * 0.1;
            this.stats.avgConfidence = 
                this.stats.avgConfidence * 0.9 + confidence * 0.1;
        }
    }
    
    /**
     * Emit progress event.
     */
    _emitProgress(message, percent) {
        if (this.onLoadProgress) {
            this.onLoadProgress({ message, percent });
        }
    }
    
    /**
     * Clear the frame buffer and prediction state.
     */
    clearBuffer() {
        this.frameBuffer = [];
        this.predictionHistory = [];
        this.lastStablePrediction = null;
        this.lastStableConfidence = 0;
        this.lastStableTime = null;
        this.emaProbabilities = null;
        this.lastRawPrediction = null;
        this.stats.bufferResets++;
        console.log('[ONNXPredictor] Buffer cleared');
    }
    
    /**
     * Reset stable prediction tracking (keep buffer).
     */
    resetStablePrediction() {
        this.predictionHistory = [];
        this.lastStablePrediction = null;
        this.lastStableConfidence = 0;
        this.lastStableTime = null;
    }
    
    /**
     * Get current frame buffer as array.
     * Useful for saving feedback samples.
     */
    getCurrentSequence() {
        if (this.frameBuffer.length === 0) {
            return null;
        }
        return [...this.frameBuffer];
    }
    
    /**
     * Get performance statistics.
     */
    getStats() {
        return {
            ...this.stats,
            bufferSize: this.frameBuffer.length,
            historySize: this.predictionHistory.length,
            lastStablePrediction: this.lastStablePrediction,
            lastStableConfidence: this.lastStableConfidence,
            isReady: this.isReady
        };
    }
    
    /**
     * Get model information.
     */
    getModelInfo() {
        return {
            labels: this.labels,
            numClasses: this.labels.length,
            modelInfo: this.modelInfo,
            config: this.config,
            isReady: this.isReady
        };
    }
    
    /**
     * Destroy the predictor and free resources.
     */
    destroy() {
        if (this.session) {
            // ONNX Runtime doesn't have explicit destroy, but we can null the reference
            this.session = null;
        }
        this.clearBuffer();
        this.isReady = false;
        console.log('[ONNXPredictor] Destroyed');
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ONNXPredictor;
}

