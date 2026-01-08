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
        
        // ==========================================
        // OPTIMIZED Configuration for Speed + Accuracy
        // ==========================================
        this.config = {
            // SPEED: Reduced buffer for faster response (~0.8s instead of ~1.5s)
            slidingWindowSize: options.slidingWindowSize || 36,  // Was 45, now 36 (~2.4s at 15FPS)
            modelSequenceLength: 60,  // Model expects 60 frames (padded if needed)
            
            // SPEED: Start predicting earlier
            minPredictionFrames: options.minPredictionFrames || 12,  // Was 32, now 12 (~0.8s)
            
            // SPEED: Faster stability (2/3 majority instead of 3/5)
            stabilityVotes: options.stabilityVotes || 3,  // Was 5, now 3
            stabilityThreshold: options.stabilityThreshold || 2,  // Was 3, now 2
            
            // ACCURACY: Clear confidence zones to eliminate ambiguity
            // FIXED: Removed "gray zone" between low and min confidence
            // Zone 1: < 0.50 = Low confidence (rejected, show "Searching...")
            // Zone 2: >= 0.50 = Acceptable (add to stability voting)
            minConfidence: options.minConfidence || 0.50,  // FIXED: Lowered to match lowConfidenceThreshold
            lowConfidenceThreshold: options.lowConfidenceThreshold || 0.50,  // FIXED: Same as minConfidence (no gray zone)
            
            // ROBUSTNESS: More responsive EMA
            emaAlpha: options.emaAlpha || 0.75,  // Was 0.7, now 0.75 (more responsive)
            
            // NEW: Hand quality thresholds
            minHandVisibility: options.minHandVisibility || 0.6,  // Minimum 60% landmarks visible
            maxJitterThreshold: options.maxJitterThreshold || 0.15,  // Max frame-to-frame jitter
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
            bufferResets: 0,
            rejectedLowQuality: 0,  // NEW: Track rejected frames
            rejectedLowConfidence: 0  // NEW: Track rejected predictions
        };
        
        // NEW: Previous frame for jitter detection
        this.previousLandmarks = null;
        
        // Event callbacks
        this.onPrediction = null;
        this.onReady = null;
        this.onError = null;
        this.onLoadProgress = null;
        this.onQualityIssue = null;  // NEW: Callback for quality issues
    }
    
    /**
     * Initialize the predictor by loading model and labels.
     * Includes automatic retry logic for network failures.
     * 
     * @param {string} modelPath - Path to ONNX model file
     * @param {string} labelsPath - Path to labels JSON file
     * @param {string} thresholdsPath - Path to sign thresholds JSON file
     * @param {number} retryCount - Current retry attempt (internal use)
     */
    async init(modelPath = 'models/psl_model_v2.onnx', 
               labelsPath = 'models/psl_labels.json',
               thresholdsPath = 'models/sign_thresholds.json',
               retryCount = 0) {
        
        const MAX_RETRIES = 3;
        const RETRY_DELAY_MS = 2000;
        
        try {
            console.log(`[ONNXPredictor] Initializing${retryCount > 0 ? ` (attempt ${retryCount + 1}/${MAX_RETRIES + 1})` : ''}...`);
            this._emitProgress('Loading model files...', 0);
            
            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime Web (ort) not loaded. Add the CDN script tag.');
            }
            
            // Load labels first (small file)
            this._emitProgress('Loading labels...', 10);
            const labelsResponse = await this._fetchWithTimeout(labelsPath, 10000);
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
                const thresholdsResponse = await this._fetchWithTimeout(thresholdsPath, 10000);
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
            
            // ==========================================
            // OPTIMIZED: Try WebGPU first, fallback to WASM
            // Safari Note: WebGPU support in Safari 17+ only
            // ==========================================
            let executionProviders = ['wasm'];
            let backendUsed = 'wasm';
            
            // Detect Safari
            const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
            
            // Check for WebGPU support (significantly faster)
            if (typeof navigator !== 'undefined' && navigator.gpu) {
                try {
                    const adapter = await navigator.gpu.requestAdapter();
                    if (adapter) {
                        console.log('[ONNXPredictor] WebGPU available! Using GPU acceleration.');
                        executionProviders = ['webgpu', 'wasm'];  // WebGPU with WASM fallback
                        backendUsed = 'webgpu';
                        this._emitProgress('Using GPU acceleration...', 35);
                    }
                } catch (e) {
                    console.log('[ONNXPredictor] WebGPU check failed, using WASM:', e.message);
                    if (isSafari) {
                        console.info('[ONNXPredictor] Safari detected. WebGPU requires Safari 17+. Using WASM fallback.');
                    }
                }
            } else if (isSafari) {
                console.info('[ONNXPredictor] Safari without WebGPU detected. Using WASM backend.');
            }
            
            // Configure ONNX Runtime with optimizations
            const sessionOptions = {
                executionProviders: executionProviders,
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                enableMemPattern: true,
                // NEW: Additional optimizations
                executionMode: 'sequential',  // Better for single predictions
                logSeverityLevel: 3,  // Warnings only
            };
            
            // Create inference session
            this.session = await ort.InferenceSession.create(modelPath, sessionOptions);
            this.backendUsed = backendUsed;
            console.log(`[ONNXPredictor] ONNX model loaded successfully (backend: ${backendUsed})`);
            
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
                    modelInfo: this.modelInfo,
                    isSafari: isSafari
                });
            }
            
            console.log('[ONNXPredictor] Initialization complete');
            return true;
            
        } catch (error) {
            console.error(`[ONNXPredictor] Initialization failed:`, error);
            
            // Retry logic for network errors
            if (retryCount < MAX_RETRIES && this._isRetryableError(error)) {
                console.log(`[ONNXPredictor] Retrying in ${RETRY_DELAY_MS}ms...`);
                this._emitProgress(`Network error. Retrying (${retryCount + 1}/${MAX_RETRIES})...`, 0);
                
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY_MS));
                return this.init(modelPath, labelsPath, thresholdsPath, retryCount + 1);
            }
            
            if (this.onError) {
                this.onError(error);
            }
            return false;
        }
    }
    
    /**
     * Fetch with timeout for network requests
     * @private
     */
    async _fetchWithTimeout(url, timeoutMs = 30000) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        
        try {
            const response = await fetch(url, { signal: controller.signal });
            return response;
        } finally {
            clearTimeout(timeout);
        }
    }
    
    /**
     * Check if error is retryable (network errors)
     * @private
     */
    _isRetryableError(error) {
        const message = error.message?.toLowerCase() || '';
        return message.includes('network') ||
               message.includes('fetch') ||
               message.includes('timeout') ||
               message.includes('abort') ||
               message.includes('failed to fetch') ||
               error.name === 'AbortError';
    }
    
    /**
     * Add a frame to the sliding window buffer with quality validation.
     * @param {Float32Array|Array} landmarks - 189-element landmark array
     * @returns {Object} Status of frame addition
     */
    addFrame(landmarks) {
        if (!landmarks || landmarks.length === 0) {
            // No landmarks (no hands) - optionally clear buffer
            return { added: false, reason: 'no_landmarks' };
        }
        
        // Ensure landmarks is a regular array
        const frame = Array.isArray(landmarks) ? landmarks : Array.from(landmarks);
        
        // NEW: Quality validation
        const quality = this._validateFrameQuality(frame);
        if (!quality.valid) {
            this.stats.rejectedLowQuality++;
            if (this.onQualityIssue) {
                this.onQualityIssue(quality);
            }
            // Still add frame but flag it (don't reject completely to maintain flow)
            // Only skip if quality is very poor
            if (quality.visibility < 0.3) {
                return { added: false, reason: 'very_low_quality', quality };
            }
        }
        
        // Add to buffer
        this.frameBuffer.push(frame);
        
        // Store for next frame jitter check
        this.previousLandmarks = frame;
        
        // Keep buffer at sliding window size
        while (this.frameBuffer.length > this.config.slidingWindowSize) {
            this.frameBuffer.shift();
        }
        
        return { added: true, quality };
    }
    
    /**
     * NEW: Validate frame quality before adding to buffer.
     * Checks visibility (non-zero landmarks) and jitter.
     * @param {Array} frame - 189-element landmark array
     * @returns {Object} Quality assessment
     */
    _validateFrameQuality(frame) {
        // Check visibility: count non-zero landmarks (first 126 are actual hand data)
        const handLandmarks = frame.slice(0, 126);
        let nonZeroCount = 0;
        for (let i = 0; i < handLandmarks.length; i++) {
            if (Math.abs(handLandmarks[i]) > 0.001) {
                nonZeroCount++;
            }
        }
        const visibility = nonZeroCount / 126;
        
        // Check jitter: compare with previous frame
        let jitter = 0;
        if (this.previousLandmarks) {
            let totalDiff = 0;
            let validPairs = 0;
            for (let i = 0; i < 126; i += 3) {
                // Only check x,y coordinates (skip z)
                const dx = Math.abs(frame[i] - this.previousLandmarks[i]);
                const dy = Math.abs(frame[i + 1] - this.previousLandmarks[i + 1]);
                if (frame[i] !== 0 && this.previousLandmarks[i] !== 0) {
                    totalDiff += dx + dy;
                    validPairs++;
                }
            }
            jitter = validPairs > 0 ? totalDiff / validPairs : 0;
        }
        
        // Validate ranges: x,y should be 0-1
        let outOfRange = 0;
        for (let i = 0; i < 126; i += 3) {
            if (frame[i] < 0 || frame[i] > 1) outOfRange++;
            if (frame[i + 1] < 0 || frame[i + 1] > 1) outOfRange++;
        }
        const rangeValid = outOfRange < 5;  // Allow a few outliers
        
        const valid = visibility >= this.config.minHandVisibility && 
                      jitter <= this.config.maxJitterThreshold &&
                      rangeValid;
        
        return {
            valid,
            visibility,
            jitter,
            rangeValid,
            reason: !valid ? (
                visibility < this.config.minHandVisibility ? 'low_visibility' :
                jitter > this.config.maxJitterThreshold ? 'high_jitter' :
                'out_of_range'
            ) : null
        };
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
                    timestamp: Date.now(),
                    allPredictions: allPredictions  // FIXED: Include all predictions for disambiguation
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
        this.previousLandmarks = null;  // NEW: Clear jitter detection state
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
            isReady: this.isReady,
            backendUsed: this.backendUsed || 'wasm',
            stats: this.stats
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

