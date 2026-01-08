/**
 * Analytics Module for PSL Recognition System
 * 
 * Provides real-time session analytics including:
 * - Session duration and prediction counts
 * - Accuracy tracking based on user feedback
 * - Confidence trends over time
 * - Confusion matrix visualization
 * - Performance metrics (FPS, inference time, memory)
 * - Model version checking
 * - Session recording for debugging
 * - Export functionality
 */

// ================================================================================
// SESSION TRACKER CLASS
// ================================================================================

class SessionTracker {
    constructor() {
        this.sessionStartTime = null;
        this.totalPredictions = 0;
        this.confirmedCorrect = 0;
        this.correctedPredictions = 0;
        this.predictionHistory = [];
        this.confidenceHistory = [];
        this.signCounts = {};
        
        // Limits
        this.maxHistoryLength = 100;
        this.maxConfidencePoints = 50;
    }
    
    /**
     * Start a new session
     */
    startSession() {
        this.sessionStartTime = Date.now();
        this.totalPredictions = 0;
        this.confirmedCorrect = 0;
        this.correctedPredictions = 0;
        this.predictionHistory = [];
        this.confidenceHistory = [];
        this.signCounts = {};
        
        console.log('[Analytics] Session started');
    }
    
    /**
     * Record a new prediction
     * @param {string} label - Predicted sign label
     * @param {number} confidence - Confidence score (0-1)
     */
    recordPrediction(label, confidence) {
        if (!this.sessionStartTime) {
            this.startSession();
        }
        
        this.totalPredictions++;
        
        // Track sign counts
        if (!this.signCounts[label]) {
            this.signCounts[label] = 0;
        }
        this.signCounts[label]++;
        
        // Add to prediction history
        const entry = {
            label: label,
            confidence: confidence,
            timestamp: Date.now(),
            wasCorrect: null // Will be updated by feedback
        };
        this.predictionHistory.push(entry);
        
        // Trim history if needed
        if (this.predictionHistory.length > this.maxHistoryLength) {
            this.predictionHistory.shift();
        }
        
        // Add to confidence history for chart
        this.confidenceHistory.push({
            timestamp: Date.now(),
            confidence: confidence,
            label: label
        });
        
        // Trim confidence history
        if (this.confidenceHistory.length > this.maxConfidencePoints) {
            this.confidenceHistory.shift();
        }
    }
    
    /**
     * Record user feedback on a prediction
     * @param {string} label - The prediction that was evaluated
     * @param {boolean} isCorrect - Whether the prediction was correct
     */
    recordFeedback(label, isCorrect) {
        if (isCorrect) {
            this.confirmedCorrect++;
        } else {
            this.correctedPredictions++;
        }
        
        // Update the most recent matching prediction in history
        for (let i = this.predictionHistory.length - 1; i >= 0; i--) {
            if (this.predictionHistory[i].label === label && 
                this.predictionHistory[i].wasCorrect === null) {
                this.predictionHistory[i].wasCorrect = isCorrect;
                break;
            }
        }
    }
    
    /**
     * Get session duration as formatted string
     * @returns {string} Duration in mm:ss format
     */
    getSessionDuration() {
        if (!this.sessionStartTime) {
            return '00:00';
        }
        
        const elapsed = Date.now() - this.sessionStartTime;
        const seconds = Math.floor(elapsed / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    
    /**
     * Get accuracy rate as percentage
     * @returns {number} Accuracy percentage (0-100)
     */
    getAccuracyRate() {
        const totalFeedback = this.confirmedCorrect + this.correctedPredictions;
        if (totalFeedback === 0) {
            return 0;
        }
        return Math.round((this.confirmedCorrect / totalFeedback) * 100);
    }
    
    /**
     * Get average confidence as percentage
     * @returns {number} Average confidence (0-100)
     */
    getAverageConfidence() {
        if (this.confidenceHistory.length === 0) {
            return 0;
        }
        
        const sum = this.confidenceHistory.reduce((acc, item) => acc + item.confidence, 0);
        return Math.round((sum / this.confidenceHistory.length) * 100);
    }
    
    /**
     * Get confidence data for chart
     * @returns {Array} Array of {timestamp, confidence, label}
     */
    getConfidenceOverTime() {
        return [...this.confidenceHistory];
    }
    
    /**
     * Get unique signs practiced this session
     * @returns {number} Count of unique signs
     */
    getUniqueSignsCount() {
        return Object.keys(this.signCounts).length;
    }
    
    /**
     * Get most practiced signs
     * @param {number} limit - Number of signs to return
     * @returns {Array} Array of {label, count}
     */
    getMostPracticedSigns(limit = 5) {
        return Object.entries(this.signCounts)
            .map(([label, count]) => ({ label, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, limit);
    }
    
    /**
     * Get recent prediction history
     * @param {number} limit - Number of predictions to return
     * @returns {Array} Recent predictions
     */
    getRecentPredictions(limit = 10) {
        return this.predictionHistory.slice(-limit).reverse();
    }
    
    /**
     * Export session data as JSON object
     * @returns {Object} Complete session data
     */
    exportToJSON() {
        return {
            exportDate: new Date().toISOString(),
            sessionDuration: this.getSessionDuration(),
            sessionDurationMs: this.sessionStartTime ? Date.now() - this.sessionStartTime : 0,
            totalPredictions: this.totalPredictions,
            confirmedCorrect: this.confirmedCorrect,
            correctedPredictions: this.correctedPredictions,
            accuracyRate: this.getAccuracyRate(),
            averageConfidence: this.getAverageConfidence(),
            uniqueSignsPracticed: this.getUniqueSignsCount(),
            signCounts: this.signCounts,
            predictionHistory: this.predictionHistory,
            confidenceHistory: this.confidenceHistory
        };
    }
}


// ================================================================================
// ANALYTICS PANEL CLASS
// ================================================================================

class AnalyticsPanel {
    constructor(options = {}) {
        this.sessionTracker = options.sessionTracker || new SessionTracker();
        this.feedbackManager = options.feedbackManager || null;
        this.isVisible = false;
        this.updateInterval = null;
        this.updateIntervalMs = options.updateIntervalMs || 1000;
        
        // DOM Elements (will be set in init)
        this.elements = {};
    }
    
    /**
     * Initialize the analytics panel
     */
    init() {
        this.initDOMElements();
        this.setupEventListeners();
        console.log('[Analytics] Panel initialized');
    }
    
    /**
     * Initialize DOM element references
     */
    initDOMElements() {
        this.elements = {
            panel: document.getElementById('analyticsPanel'),
            toggleBtn: document.getElementById('analyticsToggle'),
            
            // Stats
            durationValue: document.getElementById('analyticsDuration'),
            predictionsValue: document.getElementById('analyticsPredictions'),
            accuracyValue: document.getElementById('analyticsAccuracy'),
            confidenceValue: document.getElementById('analyticsAvgConfidence'),
            
            // Chart
            confidenceChart: document.getElementById('confidenceChart'),
            
            // Confusion list
            confusionList: document.getElementById('confusionList'),
            
            // History
            historyList: document.getElementById('analyticsHistory'),
            
            // Mastery
            masteryProgress: document.getElementById('masteryProgress'),
            masteryText: document.getElementById('masteryText'),
            
            // Export
            exportBtn: document.getElementById('analyticsExport')
        };
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Toggle button
        if (this.elements.toggleBtn) {
            this.elements.toggleBtn.addEventListener('click', () => this.toggle());
        }
        
        // Export button
        if (this.elements.exportBtn) {
            this.elements.exportBtn.addEventListener('click', () => this.exportData());
        }
    }
    
    /**
     * Toggle panel visibility
     */
    toggle() {
        this.isVisible = !this.isVisible;
        
        if (this.elements.panel) {
            this.elements.panel.classList.toggle('visible', this.isVisible);
        }
        
        if (this.elements.toggleBtn) {
            this.elements.toggleBtn.classList.toggle('active', this.isVisible);
        }
        
        if (this.isVisible) {
            this.startUpdates();
            this.render();
        } else {
            this.stopUpdates();
        }
    }
    
    /**
     * Show the panel
     */
    show() {
        if (!this.isVisible) {
            this.toggle();
        }
    }
    
    /**
     * Hide the panel
     */
    hide() {
        if (this.isVisible) {
            this.toggle();
        }
    }
    
    /**
     * Start periodic updates
     */
    startUpdates() {
        this.stopUpdates();
        this.updateInterval = setInterval(() => {
            this.render();
        }, this.updateIntervalMs);
    }
    
    /**
     * Stop periodic updates
     */
    stopUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    /**
     * Render all analytics data
     */
    render() {
        this.renderStats();
        this.renderConfidenceChart();
        this.renderConfusionList();
        this.renderHistory();
        this.renderMastery();
    }
    
    /**
     * Render stat cards
     */
    renderStats() {
        if (this.elements.durationValue) {
            this.elements.durationValue.textContent = this.sessionTracker.getSessionDuration();
        }
        
        if (this.elements.predictionsValue) {
            this.elements.predictionsValue.textContent = this.sessionTracker.totalPredictions;
        }
        
        if (this.elements.accuracyValue) {
            const accuracy = this.sessionTracker.getAccuracyRate();
            this.elements.accuracyValue.textContent = `${accuracy}%`;
            this.elements.accuracyValue.className = 'stat-value ' + this.getAccuracyClass(accuracy);
        }
        
        if (this.elements.confidenceValue) {
            this.elements.confidenceValue.textContent = `${this.sessionTracker.getAverageConfidence()}%`;
        }
    }
    
    /**
     * Get CSS class based on accuracy
     */
    getAccuracyClass(accuracy) {
        if (accuracy >= 80) return 'accuracy-high';
        if (accuracy >= 60) return 'accuracy-medium';
        return 'accuracy-low';
    }
    
    /**
     * Render confidence chart (CSS-only bars)
     */
    renderConfidenceChart() {
        if (!this.elements.confidenceChart) return;
        
        const data = this.sessionTracker.getConfidenceOverTime();
        const maxPoints = 30;
        const recentData = data.slice(-maxPoints);
        
        if (recentData.length === 0) {
            this.elements.confidenceChart.innerHTML = `
                <div class="chart-empty">No data yet</div>
            `;
            return;
        }
        
        const bars = recentData.map(point => {
            const height = Math.round(point.confidence * 100);
            const colorClass = this.getConfidenceColorClass(point.confidence);
            return `<div class="chart-bar ${colorClass}" style="height: ${height}%" title="${point.label}: ${height}%"></div>`;
        }).join('');
        
        this.elements.confidenceChart.innerHTML = `
            <div class="chart-bars">${bars}</div>
            <div class="chart-baseline"></div>
        `;
    }
    
    /**
     * Get color class for confidence bar
     */
    getConfidenceColorClass(confidence) {
        if (confidence >= 0.8) return 'bar-high';
        if (confidence >= 0.6) return 'bar-medium';
        return 'bar-low';
    }
    
    /**
     * Render confusion list from feedback manager
     */
    renderConfusionList() {
        if (!this.elements.confusionList || !this.feedbackManager) {
            return;
        }
        
        const confusionMatrix = this.feedbackManager.getConfusionMatrix();
        
        // Convert to sorted array of pairs
        const pairs = [];
        for (const predicted in confusionMatrix) {
            for (const actual in confusionMatrix[predicted]) {
                if (predicted !== actual) {
                    pairs.push({
                        from: predicted,
                        to: actual,
                        count: confusionMatrix[predicted][actual]
                    });
                }
            }
        }
        
        pairs.sort((a, b) => b.count - a.count);
        const topPairs = pairs.slice(0, 5);
        
        if (topPairs.length === 0) {
            this.elements.confusionList.innerHTML = `
                <div class="confusion-empty">No confusions recorded</div>
            `;
            return;
        }
        
        const maxCount = topPairs[0].count;
        
        const html = topPairs.map(pair => {
            const barWidth = Math.round((pair.count / maxCount) * 100);
            return `
                <div class="confusion-item">
                    <span class="confusion-pair">${pair.from} ↔ ${pair.to}</span>
                    <div class="confusion-bar-container">
                        <div class="confusion-bar" style="width: ${barWidth}%"></div>
                    </div>
                    <span class="confusion-count">${pair.count}×</span>
                </div>
            `;
        }).join('');
        
        this.elements.confusionList.innerHTML = html;
    }
    
    /**
     * Render recent prediction history
     */
    renderHistory() {
        if (!this.elements.historyList) return;
        
        const recent = this.sessionTracker.getRecentPredictions(10);
        
        if (recent.length === 0) {
            this.elements.historyList.innerHTML = `
                <div class="history-empty">No predictions yet</div>
            `;
            return;
        }
        
        const html = recent.map(pred => {
            const confidencePercent = Math.round(pred.confidence * 100);
            let statusIcon = '';
            let statusClass = '';
            
            if (pred.wasCorrect === true) {
                statusIcon = '✓';
                statusClass = 'correct';
            } else if (pred.wasCorrect === false) {
                statusIcon = '✗';
                statusClass = 'incorrect';
            }
            
            return `
                <span class="history-item ${statusClass}" title="${pred.label}: ${confidencePercent}%">
                    ${pred.label}${statusIcon ? ` ${statusIcon}` : ''}
                </span>
            `;
        }).join(' → ');
        
        this.elements.historyList.innerHTML = html;
    }
    
    /**
     * Render mastery progress
     */
    renderMastery() {
        if (!this.elements.masteryProgress || !this.elements.masteryText) return;
        
        const uniqueSigns = this.sessionTracker.getUniqueSignsCount();
        const totalSigns = 40; // Total Urdu alphabet signs
        const percentage = Math.round((uniqueSigns / totalSigns) * 100);
        
        this.elements.masteryProgress.style.width = `${percentage}%`;
        this.elements.masteryText.textContent = `${uniqueSigns}/${totalSigns} signs practiced`;
    }
    
    /**
     * Export analytics data
     */
    exportData() {
        const sessionData = this.sessionTracker.exportToJSON();
        
        // Add confusion matrix if available
        if (this.feedbackManager) {
            sessionData.confusionMatrix = this.feedbackManager.getConfusionMatrix();
            sessionData.feedbackStats = this.feedbackManager.getStats();
        }
        
        // Create and download file
        const blob = new Blob([JSON.stringify(sessionData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const now = new Date();
        const dateStr = now.toISOString().slice(0, 10);
        const timeStr = now.toTimeString().slice(0, 5).replace(':', '-');
        const filename = `psl_session_${dateStr}_${timeStr}.json`;
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log(`[Analytics] Exported session data to ${filename}`);
    }
    
    /**
     * Destroy the panel
     */
    destroy() {
        this.stopUpdates();
    }
}


// ================================================================================
// EXPORTS
// ================================================================================

// ================================================================================
// PERFORMANCE METRICS TRACKER
// ================================================================================

class PerformanceMetrics {
    constructor() {
        this.fpsHistory = [];
        this.inferenceHistory = [];
        this.mediapipeHistory = [];
        this.maxHistoryLength = 60; // 60 data points for chart
        
        this.lastFrameTime = performance.now();
        this.frameCount = 0;
        this.fps = 0;
        this.lastFpsUpdate = 0;
        
        // Current metrics
        this.currentInferenceTime = 0;
        this.currentMediapipeTime = 0;
        this.memoryUsage = null;
        
        // Chart canvas
        this.chartCanvas = null;
        this.chartCtx = null;
    }
    
    /**
     * Initialize the performance metrics tracker
     */
    init() {
        this.chartCanvas = document.getElementById('perfChartCanvas');
        if (this.chartCanvas) {
            this.chartCtx = this.chartCanvas.getContext('2d');
        }
        console.log('[Performance] Metrics tracker initialized');
    }
    
    /**
     * Record a frame for FPS calculation
     */
    recordFrame() {
        const now = performance.now();
        this.frameCount++;
        
        // Update FPS every 500ms
        if (now - this.lastFpsUpdate >= 500) {
            this.fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
            this.frameCount = 0;
            this.lastFpsUpdate = now;
            
            // Add to history
            this.fpsHistory.push({
                time: now,
                value: this.fps
            });
            
            if (this.fpsHistory.length > this.maxHistoryLength) {
                this.fpsHistory.shift();
            }
        }
        
        this.lastFrameTime = now;
    }
    
    /**
     * Record inference time
     * @param {number} timeMs - Inference time in milliseconds
     */
    recordInferenceTime(timeMs) {
        this.currentInferenceTime = timeMs;
        this.inferenceHistory.push({
            time: performance.now(),
            value: timeMs
        });
        
        if (this.inferenceHistory.length > this.maxHistoryLength) {
            this.inferenceHistory.shift();
        }
    }
    
    /**
     * Record MediaPipe processing time
     * @param {number} timeMs - MediaPipe time in milliseconds
     */
    recordMediapipeTime(timeMs) {
        this.currentMediapipeTime = timeMs;
        this.mediapipeHistory.push({
            time: performance.now(),
            value: timeMs
        });
        
        if (this.mediapipeHistory.length > this.maxHistoryLength) {
            this.mediapipeHistory.shift();
        }
    }
    
    /**
     * Get memory usage if available
     * @returns {string} Formatted memory string
     */
    getMemoryUsage() {
        if (performance.memory) {
            const usedMB = Math.round(performance.memory.usedJSHeapSize / (1024 * 1024));
            const totalMB = Math.round(performance.memory.jsHeapSizeLimit / (1024 * 1024));
            this.memoryUsage = usedMB;
            return `${usedMB}MB`;
        }
        return '--';
    }
    
    /**
     * Get FPS class for styling
     * @returns {string} CSS class
     */
    getFpsClass() {
        if (this.fps >= 25) return 'good';
        if (this.fps >= 15) return 'ok';
        return 'bad';
    }
    
    /**
     * Get inference time class for styling
     * @returns {string} CSS class
     */
    getInferenceClass() {
        if (this.currentInferenceTime <= 50) return 'good';
        if (this.currentInferenceTime <= 100) return 'ok';
        return 'bad';
    }
    
    /**
     * Render the performance chart
     */
    renderChart() {
        if (!this.chartCtx || !this.chartCanvas) return;
        
        const ctx = this.chartCtx;
        const width = this.chartCanvas.width;
        const height = this.chartCanvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw background
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        ctx.fillStyle = isDark ? 'rgba(0, 0, 0, 0.2)' : 'rgba(255, 255, 255, 0.5)';
        ctx.fillRect(0, 0, width, height);
        
        // Draw grid lines
        ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = (height / 4) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // Draw FPS line (green)
        this.drawLine(ctx, this.fpsHistory, '#22c55e', 60, width, height);
        
        // Draw inference time line (blue)
        this.drawLine(ctx, this.inferenceHistory, '#3b82f6', 150, width, height);
        
        // Draw MediaPipe time line (purple)
        this.drawLine(ctx, this.mediapipeHistory, '#8b5cf6', 150, width, height);
        
        // Legend
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#22c55e';
        ctx.fillText('FPS', 5, 12);
        ctx.fillStyle = '#3b82f6';
        ctx.fillText('Inference', 35, 12);
        ctx.fillStyle = '#8b5cf6';
        ctx.fillText('MediaPipe', 85, 12);
    }
    
    /**
     * Draw a line on the chart
     */
    drawLine(ctx, data, color, maxValue, width, height) {
        if (data.length < 2) return;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const xStep = width / (this.maxHistoryLength - 1);
        
        data.forEach((point, i) => {
            const x = (data.length - 1 - (data.length - 1 - i)) * xStep;
            const normalizedValue = Math.min(point.value / maxValue, 1);
            const y = height - (normalizedValue * (height - 10)) - 5;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
    }
    
    /**
     * Get all metrics as object
     * @returns {Object} Current metrics
     */
    getMetrics() {
        return {
            fps: this.fps,
            inferenceTime: this.currentInferenceTime,
            mediapipeTime: this.currentMediapipeTime,
            memory: this.memoryUsage
        };
    }
}


// ================================================================================
// MODEL VERSION CHECKER
// ================================================================================

class ModelVersionChecker {
    constructor() {
        this.currentVersion = null;
        this.latestVersion = null;
        this.modelPath = 'models/psl_model_v2.onnx';
        this.versionPath = 'models/model_version.json';
        this.remoteVersionUrl = null; // Set this to check for updates
        
        // Callbacks
        this.onUpdateAvailable = null;
    }
    
    /**
     * Initialize and check model version
     */
    async init() {
        await this.loadCurrentVersion();
        console.log('[ModelVersion] Current version:', this.currentVersion);
    }
    
    /**
     * Load current model version from local file
     */
    async loadCurrentVersion() {
        try {
            const response = await fetch(this.versionPath);
            if (response.ok) {
                const data = await response.json();
                this.currentVersion = data.version;
                return data;
            }
        } catch (err) {
            // Version file doesn't exist, use default
            console.log('[ModelVersion] No version file, using default');
        }
        
        // Default version info
        this.currentVersion = '2.0.0';
        return {
            version: '2.0.0',
            date: '2026-01-08',
            signs: 40,
            accuracy: '85%'
        };
    }
    
    /**
     * Check for updates from remote URL
     * @returns {Promise<boolean>} True if update available
     */
    async checkForUpdates() {
        if (!this.remoteVersionUrl) {
            console.log('[ModelVersion] No remote URL configured');
            return false;
        }
        
        try {
            const response = await fetch(this.remoteVersionUrl, {
                cache: 'no-cache'
            });
            
            if (response.ok) {
                const data = await response.json();
                this.latestVersion = data.version;
                
                if (this.isNewerVersion(data.version, this.currentVersion)) {
                    console.log('[ModelVersion] Update available:', data.version);
                    if (this.onUpdateAvailable) {
                        this.onUpdateAvailable(data);
                    }
                    return true;
                }
            }
        } catch (err) {
            console.warn('[ModelVersion] Update check failed:', err);
        }
        
        return false;
    }
    
    /**
     * Compare semantic versions
     * @param {string} v1 - First version
     * @param {string} v2 - Second version
     * @returns {boolean} True if v1 > v2
     */
    isNewerVersion(v1, v2) {
        const parts1 = v1.split('.').map(Number);
        const parts2 = v2.split('.').map(Number);
        
        for (let i = 0; i < 3; i++) {
            if ((parts1[i] || 0) > (parts2[i] || 0)) return true;
            if ((parts1[i] || 0) < (parts2[i] || 0)) return false;
        }
        return false;
    }
    
    /**
     * Get version display string
     * @returns {string} Formatted version string
     */
    getVersionString() {
        return `v${this.currentVersion || '?.?.?'}`;
    }
}


// ================================================================================
// SESSION RECORDER FOR DEBUGGING
// ================================================================================

class SessionRecorder {
    constructor() {
        this.isRecording = false;
        this.recordedFrames = [];
        this.startTime = null;
        this.maxFrames = 1800; // 1 minute at 30fps
        
        // Callbacks
        this.onFrameRecorded = null;
    }
    
    /**
     * Start recording session
     */
    start() {
        this.isRecording = true;
        this.recordedFrames = [];
        this.startTime = Date.now();
        console.log('[Recorder] Recording started');
    }
    
    /**
     * Stop recording session
     */
    stop() {
        this.isRecording = false;
        console.log('[Recorder] Recording stopped. Frames:', this.recordedFrames.length);
        return this.recordedFrames;
    }
    
    /**
     * Record a frame with prediction data
     * @param {Object} data - Frame data to record
     */
    recordFrame(data) {
        if (!this.isRecording) return;
        
        const frame = {
            timestamp: Date.now() - this.startTime,
            frameNumber: this.recordedFrames.length + 1,
            ...data
        };
        
        this.recordedFrames.push(frame);
        
        // Prevent memory overflow
        if (this.recordedFrames.length > this.maxFrames) {
            this.recordedFrames.shift();
        }
        
        if (this.onFrameRecorded) {
            this.onFrameRecorded(frame);
        }
    }
    
    /**
     * Record prediction with full context
     * @param {Object} prediction - Prediction result
     * @param {Array} landmarks - Hand landmarks (optional)
     * @param {Object} performance - Performance metrics
     */
    recordPrediction(prediction, landmarks = null, performance = null) {
        this.recordFrame({
            type: 'prediction',
            label: prediction.label,
            confidence: prediction.confidence,
            urduLabel: prediction.urduLabel || null,
            landmarks: landmarks ? this.compressLandmarks(landmarks) : null,
            performance: performance || null
        });
    }
    
    /**
     * Compress landmarks for storage (reduce precision)
     * @param {Array} landmarks - MediaPipe landmarks
     * @returns {Array} Compressed landmarks
     */
    compressLandmarks(landmarks) {
        if (!landmarks || !Array.isArray(landmarks)) return null;
        
        return landmarks.map(hand => 
            hand.map(point => ({
                x: Math.round(point.x * 10000) / 10000,
                y: Math.round(point.y * 10000) / 10000,
                z: Math.round((point.z || 0) * 10000) / 10000
            }))
        );
    }
    
    /**
     * Get recording duration as formatted string
     * @returns {string} Duration in mm:ss
     */
    getDuration() {
        if (!this.startTime) return '00:00';
        
        const elapsed = Date.now() - this.startTime;
        const seconds = Math.floor(elapsed / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    
    /**
     * Export recording as JSON
     * @returns {Object} Recording data
     */
    exportRecording() {
        return {
            exportDate: new Date().toISOString(),
            recordingDuration: this.getDuration(),
            totalFrames: this.recordedFrames.length,
            frames: this.recordedFrames
        };
    }
    
    /**
     * Download recording as file
     */
    downloadRecording() {
        const data = this.exportRecording();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const now = new Date();
        const dateStr = now.toISOString().slice(0, 10);
        const timeStr = now.toTimeString().slice(0, 5).replace(':', '-');
        const filename = `psl_recording_${dateStr}_${timeStr}.json`;
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log(`[Recorder] Exported recording to ${filename}`);
    }
}


// ================================================================================
// OFFLINE STATUS CHECKER
// ================================================================================

class OfflineStatusChecker {
    constructor() {
        this.isOnline = navigator.onLine;
        this.serviceWorkerActive = false;
        this.cacheStatus = null;
        
        // Callbacks
        this.onStatusChange = null;
    }
    
    /**
     * Initialize the offline status checker
     */
    init() {
        // Listen for online/offline events
        window.addEventListener('online', () => this.handleOnlineChange(true));
        window.addEventListener('offline', () => this.handleOnlineChange(false));
        
        // Check service worker status
        this.checkServiceWorker();
        
        console.log('[Offline] Status checker initialized');
    }
    
    /**
     * Handle online status change
     * @param {boolean} isOnline
     */
    handleOnlineChange(isOnline) {
        this.isOnline = isOnline;
        console.log(`[Offline] Status changed: ${isOnline ? 'Online' : 'Offline'}`);
        
        if (this.onStatusChange) {
            this.onStatusChange(this.getStatus());
        }
    }
    
    /**
     * Check service worker status
     */
    async checkServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.ready;
                this.serviceWorkerActive = !!registration.active;
                
                // Get cache status
                if (navigator.serviceWorker.controller) {
                    const channel = new MessageChannel();
                    channel.port1.onmessage = (event) => {
                        this.cacheStatus = event.data;
                    };
                    
                    navigator.serviceWorker.controller.postMessage(
                        { type: 'GET_CACHE_STATUS' },
                        [channel.port2]
                    );
                }
            } catch (err) {
                console.warn('[Offline] Service worker check failed:', err);
            }
        }
    }
    
    /**
     * Get current status
     * @returns {Object} Status object
     */
    getStatus() {
        return {
            isOnline: this.isOnline,
            serviceWorkerActive: this.serviceWorkerActive,
            cacheStatus: this.cacheStatus,
            offlineReady: this.serviceWorkerActive
        };
    }
    
    /**
     * Get status text
     * @returns {string} Human-readable status
     */
    getStatusText() {
        if (this.serviceWorkerActive) {
            return this.isOnline ? 'Online (Offline Ready)' : 'Offline Mode';
        }
        return this.isOnline ? 'Online' : 'No Connection';
    }
    
    /**
     * Get status indicator class
     * @returns {string} CSS class
     */
    getStatusClass() {
        if (this.serviceWorkerActive) {
            return 'online';
        }
        return this.isOnline ? 'checking' : 'offline';
    }
}


// ================================================================================
// EXPORTS
// ================================================================================

if (typeof window !== 'undefined') {
    window.SessionTracker = SessionTracker;
    window.AnalyticsPanel = AnalyticsPanel;
    window.PerformanceMetrics = PerformanceMetrics;
    window.ModelVersionChecker = ModelVersionChecker;
    window.SessionRecorder = SessionRecorder;
    window.OfflineStatusChecker = OfflineStatusChecker;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { 
        SessionTracker, 
        AnalyticsPanel,
        PerformanceMetrics,
        ModelVersionChecker,
        SessionRecorder,
        OfflineStatusChecker
    };
}

