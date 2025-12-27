/**
 * Recording Status UI Manager for PSL Recognition System
 * Displays recording status, segments, and provides visual feedback
 */

class RecordingStatusUI {
    constructor() {
        this.statusElement = null;
        this.statusDot = null;
        this.statusText = null;
        this.segmentsContainer = null;
        this.currentStatus = 'idle'; // 'idle', 'recording', 'stopped'
        this.segments = [];
        this.maxSegmentsDisplay = 10;
        
        // Bind methods
        this.init = this.init.bind(this);
        this.setStatus = this.setStatus.bind(this);
        this.showSegment = this.showSegment.bind(this);
        this.clearSegments = this.clearSegments.bind(this);
    }
    
    init() {
        try {
            console.log('Initializing recording status UI...');
            
            // Get or create status indicator
            this.statusElement = document.getElementById('recordingStatus');
            if (!this.statusElement) {
                this.statusElement = this.createStatusIndicator();
            }
            
            this.statusDot = this.statusElement.querySelector('.recording-dot');
            this.statusText = this.statusElement.querySelector('.recording-text');
            
            // Get or create segments container
            this.segmentsContainer = document.getElementById('recordingSegments');
            if (!this.segmentsContainer) {
                this.segmentsContainer = this.createSegmentsContainer();
            }
            
            // Set initial status
            this.setStatus('idle', 'Ready');
            
            console.log('Recording status UI initialized');
            return true;
            
        } catch (error) {
            console.error('Recording status UI initialization failed:', error);
            return false;
        }
    }
    
    createStatusIndicator() {
        /**
         * Create recording status indicator element
         * Returns: DOM element
         */
        const container = document.createElement('div');
        container.id = 'recordingStatus';
        container.className = 'recording-status';
        container.innerHTML = `
            <span class="recording-dot"></span>
            <span class="recording-text">Ready</span>
        `;
        
        // Insert into status card or create new container
        const statusCard = document.querySelector('.status-card');
        if (statusCard) {
            const statusItem = document.createElement('div');
            statusItem.className = 'status-item';
            statusItem.appendChild(container);
            statusCard.appendChild(statusItem);
        } else {
            document.body.appendChild(container);
        }
        
        return container;
    }
    
    createSegmentsContainer() {
        /**
         * Create segments display container
         * Returns: DOM element
         */
        const container = document.createElement('div');
        container.id = 'recordingSegments';
        container.className = 'recording-segments';
        container.innerHTML = `
            <h4 class="segments-title">Recording Segments</h4>
            <div class="segments-list"></div>
        `;
        
        // Insert into history card or create new container
        const historyCard = document.querySelector('.history-card');
        if (historyCard) {
            historyCard.appendChild(container);
        } else {
            document.body.appendChild(container);
        }
        
        return container;
    }
    
    setStatus(status, message = '') {
        /**
         * Update recording status
         * 
         * Args:
         *   status: 'idle', 'recording', or 'stopped'
         *   message: Optional status message
         */
        this.currentStatus = status;
        
        if (!this.statusElement || !this.statusDot || !this.statusText) {
            return;
        }
        
        // Remove all status classes
        this.statusDot.classList.remove('status-idle', 'status-recording', 'status-stopped');
        
        // Add appropriate class and set text
        switch (status) {
            case 'recording':
                this.statusDot.classList.add('status-recording');
                this.statusText.textContent = message || 'Recording...';
                this.statusText.style.color = '#27ae60'; // Green
                break;
                
            case 'stopped':
                this.statusDot.classList.add('status-stopped');
                this.statusText.textContent = message || 'Stopped (Idle timeout)';
                this.statusText.style.color = '#e74c3c'; // Red
                
                // Auto-reset to idle after 2 seconds
                setTimeout(() => {
                    if (this.currentStatus === 'stopped') {
                        this.setStatus('idle', 'Ready');
                    }
                }, 2000);
                break;
                
            case 'idle':
            default:
                this.statusDot.classList.add('status-idle');
                this.statusText.textContent = message || 'Ready';
                this.statusText.style.color = '#95a5a6'; // Gray
                break;
        }
    }
    
    showSegment(label, confidence, duration, timestamp = null) {
        /**
         * Display a completed recording segment
         * 
         * Args:
         *   label: Sign label
         *   confidence: Prediction confidence (0-1)
         *   duration: Segment duration in seconds
         *   timestamp: Optional timestamp
         */
        if (!this.segmentsContainer) {
            return;
        }
        
        // Create segment object
        const segment = {
            label: label,
            confidence: confidence,
            duration: duration,
            timestamp: timestamp || new Date().toISOString(),
            id: Date.now()
        };
        
        // Add to segments array
        this.segments.unshift(segment);
        
        // Limit number of displayed segments
        if (this.segments.length > this.maxSegmentsDisplay) {
            this.segments = this.segments.slice(0, this.maxSegmentsDisplay);
        }
        
        // Render segments
        this.renderSegments();
        
        console.log(`Recording segment: ${label} (${confidence.toFixed(2)}, ${duration.toFixed(1)}s)`);
    }
    
    renderSegments() {
        /**
         * Render all segments to the DOM
         */
        const segmentsList = this.segmentsContainer.querySelector('.segments-list');
        if (!segmentsList) {
            return;
        }
        
        // Clear existing segments
        segmentsList.innerHTML = '';
        
        if (this.segments.length === 0) {
            segmentsList.innerHTML = '<div class="segments-empty">No segments yet</div>';
            return;
        }
        
        // Create segment elements
        this.segments.forEach((segment, index) => {
            const segmentEl = document.createElement('div');
            segmentEl.className = 'segment-item';
            segmentEl.innerHTML = `
                <div class="segment-header">
                    <span class="segment-label">${segment.label}</span>
                    <span class="segment-confidence">${(segment.confidence * 100).toFixed(0)}%</span>
                </div>
                <div class="segment-footer">
                    <span class="segment-duration">${segment.duration.toFixed(1)}s</span>
                    <span class="segment-time">${this.formatTime(segment.timestamp)}</span>
                </div>
            `;
            
            // Add fade-in animation
            segmentEl.style.animation = 'fadeIn 0.3s ease-in';
            
            segmentsList.appendChild(segmentEl);
        });
    }
    
    clearSegments() {
        /**
         * Clear all displayed segments
         */
        this.segments = [];
        this.renderSegments();
        console.log('Recording segments cleared');
    }
    
    formatTime(isoString) {
        /**
         * Format ISO timestamp to readable time
         * 
         * Args:
         *   isoString: ISO 8601 timestamp
         * 
         * Returns:
         *   Formatted time string (HH:MM:SS)
         */
        try {
            const date = new Date(isoString);
            return date.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false
            });
        } catch (error) {
            return '';
        }
    }
    
    getSegments() {
        /**
         * Get all recorded segments
         * 
         * Returns:
         *   Array of segment objects
         */
        return this.segments;
    }
    
    exportSegments() {
        /**
         * Export segments as JSON
         * 
         * Returns:
         *   JSON string of segments
         */
        return JSON.stringify({
            segments: this.segments,
            total_segments: this.segments.length,
            total_duration: this.segments.reduce((sum, seg) => sum + seg.duration, 0),
            exported_at: new Date().toISOString()
        }, null, 2);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RecordingStatusUI;
}

