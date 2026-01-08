/**
 * Picture-in-Picture (PiP) Floating Mode for PSL Recognition System
 * 
 * Creates a floating mini window with:
 * - Compact camera view
 * - Current prediction display
 * - Minimal controls
 * - Draggable positioning
 * - Works while using other apps/tabs
 */

class PiPMode {
    constructor(options = {}) {
        // Configuration
        this.config = {
            defaultWidth: options.defaultWidth || 320,
            defaultHeight: options.defaultHeight || 280,
            minWidth: options.minWidth || 200,
            minHeight: options.minHeight || 180,
            cornerPadding: options.cornerPadding || 16,
            defaultPosition: options.defaultPosition || 'bottom-right',
            ...options
        };
        
        // State
        this.isActive = false;
        this.isPiPSupported = false;
        this.pipWindow = null;
        this.floatingContainer = null;
        
        // Dragging state
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        
        // References to main app elements
        this.videoElement = null;
        this.canvasElement = null;
        
        // Callbacks
        this.onActivate = null;
        this.onDeactivate = null;
        this.onPrediction = null;
        
        // Current prediction for display
        this.currentPrediction = null;
        this.currentConfidence = 0;
    }
    
    /**
     * Initialize PiP mode
     */
    init(videoElement, canvasElement) {
        this.videoElement = videoElement;
        this.canvasElement = canvasElement;
        
        // Check for PiP API support
        this.isPiPSupported = 'pictureInPictureEnabled' in document && 
                              document.pictureInPictureEnabled;
        
        // Create floating container as fallback
        this.createFloatingContainer();
        
        console.log(`[PiP] Initialized. Native PiP support: ${this.isPiPSupported}`);
        return true;
    }
    
    /**
     * Create floating container (fallback for non-PiP browsers)
     */
    createFloatingContainer() {
        // Check if already exists
        if (document.getElementById('pipFloatingContainer')) {
            this.floatingContainer = document.getElementById('pipFloatingContainer');
            return;
        }
        
        // Create container
        const container = document.createElement('div');
        container.id = 'pipFloatingContainer';
        container.className = 'pip-floating-container';
        container.innerHTML = `
            <div class="pip-header">
                <span class="pip-title">ISHARAPAL</span>
                <div class="pip-controls">
                    <button class="pip-btn pip-minimize" title="Minimize">−</button>
                    <button class="pip-btn pip-close" title="Close">×</button>
                </div>
            </div>
            <div class="pip-video-wrapper">
                <video id="pipVideo" autoplay muted playsinline></video>
                <canvas id="pipCanvas"></canvas>
            </div>
            <div class="pip-prediction-display">
                <div class="pip-prediction-label" id="pipPredictionLabel">-</div>
                <div class="pip-confidence-bar">
                    <div class="pip-confidence-fill" id="pipConfidenceFill"></div>
                </div>
                <div class="pip-confidence-text" id="pipConfidenceText">0%</div>
            </div>
            <div class="pip-footer">
                <button class="pip-action-btn pip-toggle-recognition" id="pipToggleBtn">
                    <span class="pip-action-icon">▶</span>
                </button>
            </div>
        `;
        
        document.body.appendChild(container);
        this.floatingContainer = container;
        
        // Apply initial styles
        this.applyFloatingStyles();
        
        // Setup event listeners
        this.setupFloatingListeners();
        
        // Position in default corner
        this.positionInCorner(this.config.defaultPosition);
    }
    
    /**
     * Apply CSS styles for floating container
     */
    applyFloatingStyles() {
        // Check if styles already exist
        if (document.getElementById('pipFloatingStyles')) return;
        
        const styles = document.createElement('style');
        styles.id = 'pipFloatingStyles';
        styles.textContent = `
            .pip-floating-container {
                position: fixed;
                width: ${this.config.defaultWidth}px;
                background: rgba(15, 20, 25, 0.95);
                backdrop-filter: blur(12px);
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.1);
                z-index: 10000;
                overflow: hidden;
                display: none;
                flex-direction: column;
                font-family: system-ui, -apple-system, sans-serif;
                user-select: none;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .pip-floating-container.visible {
                display: flex;
            }
            
            .pip-floating-container:hover {
                box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(78, 205, 196, 0.3);
            }
            
            .pip-floating-container.dragging {
                transform: scale(1.02);
                box-shadow: 0 16px 64px rgba(0, 0, 0, 0.6);
                cursor: grabbing;
            }
            
            .pip-floating-container.minimized {
                width: auto !important;
                height: auto !important;
            }
            
            .pip-floating-container.minimized .pip-video-wrapper,
            .pip-floating-container.minimized .pip-prediction-display,
            .pip-floating-container.minimized .pip-footer {
                display: none;
            }
            
            .pip-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 8px 12px;
                background: linear-gradient(135deg, rgba(78, 205, 196, 0.2), rgba(78, 205, 196, 0.05));
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                cursor: grab;
            }
            
            .pip-header:active {
                cursor: grabbing;
            }
            
            .pip-title {
                font-size: 12px;
                font-weight: 600;
                color: #4ecdc4;
                letter-spacing: 0.5px;
            }
            
            .pip-controls {
                display: flex;
                gap: 4px;
            }
            
            .pip-btn {
                width: 20px;
                height: 20px;
                border: none;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.1);
                color: #fff;
                font-size: 14px;
                line-height: 1;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
            }
            
            .pip-btn:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            
            .pip-close:hover {
                background: #ff5f57;
            }
            
            .pip-minimize:hover {
                background: #febc2e;
                color: #000;
            }
            
            .pip-video-wrapper {
                position: relative;
                width: 100%;
                aspect-ratio: 4/3;
                background: #000;
            }
            
            .pip-video-wrapper video,
            .pip-video-wrapper canvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            
            .pip-video-wrapper canvas {
                pointer-events: none;
            }
            
            .pip-prediction-display {
                padding: 12px;
                text-align: center;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .pip-prediction-label {
                font-size: 24px;
                font-weight: 700;
                color: #4ecdc4;
                margin-bottom: 8px;
                text-shadow: 0 2px 8px rgba(78, 205, 196, 0.3);
            }
            
            .pip-confidence-bar {
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
                overflow: hidden;
                margin-bottom: 4px;
            }
            
            .pip-confidence-fill {
                height: 100%;
                width: 0%;
                background: linear-gradient(90deg, #4ecdc4, #7ee8e0);
                border-radius: 3px;
                transition: width 0.3s ease;
            }
            
            .pip-confidence-text {
                font-size: 11px;
                color: rgba(255, 255, 255, 0.6);
            }
            
            .pip-footer {
                display: flex;
                justify-content: center;
                padding: 8px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .pip-action-btn {
                padding: 6px 16px;
                border: none;
                border-radius: 20px;
                background: linear-gradient(135deg, #4ecdc4, #3db8b0);
                color: #fff;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 6px;
                transition: all 0.2s ease;
            }
            
            .pip-action-btn:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 12px rgba(78, 205, 196, 0.4);
            }
            
            .pip-action-icon {
                font-size: 10px;
            }
            
            /* Resize handle */
            .pip-resize-handle {
                position: absolute;
                bottom: 0;
                right: 0;
                width: 16px;
                height: 16px;
                cursor: se-resize;
                background: linear-gradient(135deg, transparent 50%, rgba(255,255,255,0.2) 50%);
                border-radius: 0 0 16px 0;
            }
        `;
        
        document.head.appendChild(styles);
    }
    
    /**
     * Setup event listeners for floating container
     */
    setupFloatingListeners() {
        if (!this.floatingContainer) return;
        
        const header = this.floatingContainer.querySelector('.pip-header');
        const closeBtn = this.floatingContainer.querySelector('.pip-close');
        const minimizeBtn = this.floatingContainer.querySelector('.pip-minimize');
        const toggleBtn = this.floatingContainer.querySelector('#pipToggleBtn');
        
        // Dragging
        if (header) {
            header.addEventListener('mousedown', (e) => this.startDrag(e));
            header.addEventListener('touchstart', (e) => this.startDrag(e), { passive: false });
        }
        
        document.addEventListener('mousemove', (e) => this.onDrag(e));
        document.addEventListener('touchmove', (e) => this.onDrag(e), { passive: false });
        document.addEventListener('mouseup', () => this.endDrag());
        document.addEventListener('touchend', () => this.endDrag());
        
        // Close button
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.deactivate());
        }
        
        // Minimize button
        if (minimizeBtn) {
            minimizeBtn.addEventListener('click', () => this.toggleMinimize());
        }
        
        // Toggle recognition button
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                if (this.onToggleRecognition) {
                    this.onToggleRecognition();
                }
            });
        }
    }
    
    /**
     * Start dragging
     */
    startDrag(e) {
        if (e.target.closest('.pip-btn')) return;
        
        e.preventDefault();
        this.isDragging = true;
        this.floatingContainer.classList.add('dragging');
        
        const clientX = e.clientX || e.touches?.[0]?.clientX || 0;
        const clientY = e.clientY || e.touches?.[0]?.clientY || 0;
        
        const rect = this.floatingContainer.getBoundingClientRect();
        this.dragOffset = {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }
    
    /**
     * Handle drag movement
     */
    onDrag(e) {
        if (!this.isDragging) return;
        
        e.preventDefault();
        
        const clientX = e.clientX || e.touches?.[0]?.clientX || 0;
        const clientY = e.clientY || e.touches?.[0]?.clientY || 0;
        
        let newX = clientX - this.dragOffset.x;
        let newY = clientY - this.dragOffset.y;
        
        // Keep within viewport
        const rect = this.floatingContainer.getBoundingClientRect();
        const maxX = window.innerWidth - rect.width;
        const maxY = window.innerHeight - rect.height;
        
        newX = Math.max(0, Math.min(newX, maxX));
        newY = Math.max(0, Math.min(newY, maxY));
        
        this.floatingContainer.style.left = `${newX}px`;
        this.floatingContainer.style.top = `${newY}px`;
        this.floatingContainer.style.right = 'auto';
        this.floatingContainer.style.bottom = 'auto';
    }
    
    /**
     * End dragging
     */
    endDrag() {
        if (!this.isDragging) return;
        
        this.isDragging = false;
        this.floatingContainer.classList.remove('dragging');
    }
    
    /**
     * Position in a corner
     */
    positionInCorner(corner) {
        const padding = this.config.cornerPadding;
        
        switch (corner) {
            case 'top-left':
                this.floatingContainer.style.top = `${padding}px`;
                this.floatingContainer.style.left = `${padding}px`;
                this.floatingContainer.style.right = 'auto';
                this.floatingContainer.style.bottom = 'auto';
                break;
            case 'top-right':
                this.floatingContainer.style.top = `${padding}px`;
                this.floatingContainer.style.right = `${padding}px`;
                this.floatingContainer.style.left = 'auto';
                this.floatingContainer.style.bottom = 'auto';
                break;
            case 'bottom-left':
                this.floatingContainer.style.bottom = `${padding}px`;
                this.floatingContainer.style.left = `${padding}px`;
                this.floatingContainer.style.top = 'auto';
                this.floatingContainer.style.right = 'auto';
                break;
            case 'bottom-right':
            default:
                this.floatingContainer.style.bottom = `${padding}px`;
                this.floatingContainer.style.right = `${padding}px`;
                this.floatingContainer.style.top = 'auto';
                this.floatingContainer.style.left = 'auto';
                break;
        }
    }
    
    /**
     * Toggle minimize state
     */
    toggleMinimize() {
        if (this.floatingContainer) {
            this.floatingContainer.classList.toggle('minimized');
        }
    }
    
    /**
     * Activate PiP mode
     */
    async activate() {
        // Try native PiP first
        if (this.isPiPSupported && this.videoElement) {
            try {
                // Check if video can enter PiP
                if (!document.pictureInPictureElement) {
                    await this.videoElement.requestPictureInPicture();
                    this.isActive = true;
                    
                    // Listen for exit
                    this.videoElement.addEventListener('leavepictureinpicture', () => {
                        this.isActive = false;
                        if (this.onDeactivate) {
                            this.onDeactivate();
                        }
                    }, { once: true });
                    
                    if (this.onActivate) {
                        this.onActivate('native');
                    }
                    
                    console.log('[PiP] Native PiP activated');
                    return true;
                }
            } catch (error) {
                console.warn('[PiP] Native PiP failed, using floating fallback:', error);
            }
        }
        
        // Fallback to floating container
        return this.activateFloating();
    }
    
    /**
     * Activate floating mode (fallback)
     */
    activateFloating() {
        if (!this.floatingContainer) {
            this.createFloatingContainer();
        }
        
        // Clone video stream to PiP video element
        const pipVideo = this.floatingContainer.querySelector('#pipVideo');
        if (pipVideo && this.videoElement && this.videoElement.srcObject) {
            pipVideo.srcObject = this.videoElement.srcObject;
        }
        
        // Show container
        this.floatingContainer.classList.add('visible');
        this.isActive = true;
        
        if (this.onActivate) {
            this.onActivate('floating');
        }
        
        console.log('[PiP] Floating mode activated');
        return true;
    }
    
    /**
     * Deactivate PiP mode
     */
    async deactivate() {
        // Exit native PiP if active
        if (document.pictureInPictureElement) {
            try {
                await document.exitPictureInPicture();
            } catch (error) {
                console.warn('[PiP] Error exiting native PiP:', error);
            }
        }
        
        // Hide floating container
        if (this.floatingContainer) {
            this.floatingContainer.classList.remove('visible');
        }
        
        this.isActive = false;
        
        if (this.onDeactivate) {
            this.onDeactivate();
        }
        
        console.log('[PiP] Deactivated');
    }
    
    /**
     * Toggle PiP mode
     */
    toggle() {
        if (this.isActive) {
            return this.deactivate();
        } else {
            return this.activate();
        }
    }
    
    /**
     * Update prediction display in floating container
     */
    updatePrediction(label, confidence) {
        this.currentPrediction = label;
        this.currentConfidence = confidence;
        
        if (!this.floatingContainer) return;
        
        const labelEl = this.floatingContainer.querySelector('#pipPredictionLabel');
        const fillEl = this.floatingContainer.querySelector('#pipConfidenceFill');
        const textEl = this.floatingContainer.querySelector('#pipConfidenceText');
        
        if (labelEl) {
            labelEl.textContent = label || '-';
        }
        
        if (fillEl) {
            fillEl.style.width = `${Math.round(confidence * 100)}%`;
        }
        
        if (textEl) {
            textEl.textContent = `${Math.round(confidence * 100)}%`;
        }
    }
    
    /**
     * Update canvas overlay in floating container
     */
    updateCanvas(sourceCanvas) {
        if (!this.floatingContainer) return;
        
        const pipCanvas = this.floatingContainer.querySelector('#pipCanvas');
        if (!pipCanvas || !sourceCanvas) return;
        
        const ctx = pipCanvas.getContext('2d');
        
        // Match dimensions
        if (pipCanvas.width !== pipCanvas.clientWidth) {
            pipCanvas.width = pipCanvas.clientWidth;
            pipCanvas.height = pipCanvas.clientHeight;
        }
        
        // Draw scaled copy of source canvas
        ctx.clearRect(0, 0, pipCanvas.width, pipCanvas.height);
        ctx.drawImage(sourceCanvas, 0, 0, pipCanvas.width, pipCanvas.height);
    }
    
    /**
     * Check if PiP is currently active
     */
    isActiveMode() {
        return this.isActive;
    }
    
    /**
     * Set toggle recognition callback
     */
    setToggleCallback(callback) {
        this.onToggleRecognition = callback;
    }
    
    /**
     * Destroy PiP mode
     */
    destroy() {
        this.deactivate();
        
        if (this.floatingContainer) {
            this.floatingContainer.remove();
            this.floatingContainer = null;
        }
        
        const styles = document.getElementById('pipFloatingStyles');
        if (styles) {
            styles.remove();
        }
    }
}


// ================================================================================
// EXPORTS
// ================================================================================

if (typeof window !== 'undefined') {
    window.PiPMode = PiPMode;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PiPMode };
}


