/**
 * Hand visualization manager for PSL Recognition System
 * Handles drawing hand landmarks and prediction overlays
 */

class HandVisualizer {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.isActive = false;
        this.isDestroyed = false; // Prevent operations after destroy
        this.currentPrediction = null;
        this.predictionConfidence = 0;
        this.currentKeypoints = null;
        this.lastValidKeypoints = null;
        this.keypointsMissingStart = null;
        
        // Animation
        this.animationId = null;
        this.lastFrameTime = 0;
        this.fps = 30;
        this.frameInterval = 1000 / this.fps;
        
        // FPS Counter
        this.fpsHistory = [];
        this.currentFPS = 0;
        this.lastFPSUpdate = Date.now();
        this.frameCount = 0;
        
        // Visual effects
        this.effects = {
            predictionPulse: false,
            confidenceGlow: false,
            landmarkHighlight: false
        };
    }
    
    async init() {
        try {
            console.log('Initializing hand visualizer...');
            
            // Get canvas element
            this.canvas = document.getElementById('overlay-canvas');
            if (!this.canvas) {
                throw new Error('Overlay canvas not found');
            }
            
            this.ctx = this.canvas.getContext('2d');
            
            // Set up canvas properties
            this.ctx.lineCap = 'round';
            this.ctx.lineJoin = 'round';
            
            // Start animation loop
            this.startAnimation();
            
            console.log('Hand visualizer initialized');
            return true;
            
        } catch (error) {
            console.error('Hand visualizer initialization failed:', error);
            return false;
        }
    }
    
    startAnimation() {
        if (this.animationId) {
            return;
        }
        
        const animate = (currentTime) => {
            if (currentTime - this.lastFrameTime >= this.frameInterval) {
                this.draw();
                this.lastFrameTime = currentTime;
            }
            
            this.animationId = requestAnimationFrame(animate);
        };
        
        this.animationId = requestAnimationFrame(animate);
        this.isActive = true;
    }
    
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.isActive = false;
    }
    
    draw() {
        // Prevent operations after destroy
        if (this.isDestroyed) {
            return;
        }
        
        if (!this.ctx || !this.canvas) {
            return;
        }
        
        // Check canvas has valid dimensions
        if (this.canvas.width === 0 || this.canvas.height === 0) {
            return;
        }
        
        try {
            // Clear canvas
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Update FPS
            this.updateFPS();
            
            // Draw hand skeleton from backend keypoints
            if (this.currentKeypoints) {
                this.drawBackendKeypoints(this.currentKeypoints);
            }
            
            // Draw prediction overlay
            if (this.currentPrediction) {
                this.drawPredictionOverlay();
            }
            
            // Draw confidence indicator
            if (this.predictionConfidence > 0) {
                this.drawConfidenceIndicator();
            }
        } catch (error) {
            console.error('[Visualizer] Error in draw:', error);
        }
    }
    
    drawPredictionOverlay() {
        if (!this.currentPrediction) {
            return;
        }
        
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        // Create gradient for prediction text
        const gradient = this.ctx.createLinearGradient(0, 0, this.canvas.width, 0);
        gradient.addColorStop(0, '#3498db');
        gradient.addColorStop(1, '#2980b9');
        
        // Draw prediction background
        this.ctx.fillStyle = 'rgba(52, 152, 219, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, 60);
        
        // Draw prediction text
        this.ctx.fillStyle = gradient;
        this.ctx.font = 'bold 24px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        // Add pulse effect
        if (this.effects.predictionPulse) {
            const pulse = Math.sin(Date.now() / 200) * 0.1 + 1;
            this.ctx.scale(pulse, pulse);
        }
        
        this.ctx.fillText(this.currentPrediction, centerX, 30);
        
        // Reset transform
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
    
    drawConfidenceIndicator() {
        const indicatorWidth = 200;
        const indicatorHeight = 8;
        const x = (this.canvas.width - indicatorWidth) / 2;
        const y = this.canvas.height - 30;
        
        // Draw background
        this.ctx.fillStyle = 'rgba(236, 240, 241, 0.8)';
        this.ctx.fillRect(x, y, indicatorWidth, indicatorHeight);
        
        // Draw confidence fill
        const fillWidth = (indicatorWidth * this.predictionConfidence);
        
        // Create confidence gradient
        const gradient = this.ctx.createLinearGradient(x, y, x + fillWidth, y);
        if (this.predictionConfidence < 0.3) {
            gradient.addColorStop(0, '#e74c3c');
            gradient.addColorStop(1, '#c0392b');
        } else if (this.predictionConfidence < 0.7) {
            gradient.addColorStop(0, '#f39c12');
            gradient.addColorStop(1, '#e67e22');
        } else {
            gradient.addColorStop(0, '#27ae60');
            gradient.addColorStop(1, '#2ecc71');
        }
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(x, y, fillWidth, indicatorHeight);
        
        // Add glow effect
        if (this.effects.confidenceGlow) {
            this.ctx.shadowColor = gradient;
            this.ctx.shadowBlur = 10;
            this.ctx.fillRect(x, y, fillWidth, indicatorHeight);
            this.ctx.shadowBlur = 0;
        }
        
        // Draw confidence text
        this.ctx.fillStyle = '#2c3e50';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            `${Math.round(this.predictionConfidence * 100)}%`,
            x + indicatorWidth / 2,
            y - 5
        );
    }
    
    showPrediction(prediction, confidence) {
        this.currentPrediction = prediction;
        this.predictionConfidence = confidence;
        
        // Trigger pulse effect
        this.effects.predictionPulse = true;
        setTimeout(() => {
            this.effects.predictionPulse = false;
        }, 1000);
        
        // Trigger confidence glow
        if (confidence > 0.8) {
            this.effects.confidenceGlow = true;
            setTimeout(() => {
                this.effects.confidenceGlow = false;
            }, 500);
        }
    }
    
    /**
     * Update keypoints from backend for skeleton visualization
     * DEMO OPTIMIZED: Cache last valid landmarks, delayed clear
     */
    updateKeypoints(keypoints) {
        if (keypoints && keypoints.length >= 63) {
            // Valid keypoints received
            this.currentKeypoints = keypoints;
            this.lastValidKeypoints = keypoints;
            this.keypointsMissingStart = null;
        } else {
            // No keypoints - use cached if within tolerance
            if (!this.keypointsMissingStart) {
                this.keypointsMissingStart = Date.now();
            }
            
            const timeSinceMissing = Date.now() - this.keypointsMissingStart;
            if (timeSinceMissing < 400) {
                // Keep showing last valid landmarks for 400ms
                this.currentKeypoints = this.lastValidKeypoints;
            } else {
                // Clear after timeout
                this.currentKeypoints = null;
            }
        }
    }
    
    /**
     * Draw hand skeleton from backend keypoints (flat array format)
     */
    drawBackendKeypoints(keypoints) {
        // Null/undefined check
        if (!keypoints || !Array.isArray(keypoints)) {
            return;
        }
        
        if (keypoints.length < 63) {
            return;  // Need at least 1 hand (21 landmarks × 3 coords = 63 values)
        }
        
        try {
            // Hand connection indices (MediaPipe standard)
            const connections = [
                // Thumb
                [0, 1], [1, 2], [2, 3], [3, 4],
                // Index finger
                [0, 5], [5, 6], [6, 7], [7, 8],
                // Middle finger
                [0, 9], [9, 10], [10, 11], [11, 12],
                // Ring finger
                [0, 13], [13, 14], [14, 15], [15, 16],
                // Pinky
                [0, 17], [17, 18], [18, 19], [19, 20],
                // Palm connections
                [5, 9], [9, 13], [13, 17]
            ];
            
            // Extract hands (each hand is 63 values: 21 landmarks × 3 coords)
            const hands = [];
            
            // First hand (landmarks 0-62)
            if (keypoints.length >= 63) {
                const hand1 = [];
                for (let i = 0; i < 63; i += 3) {
                    // Bounds check for array access
                    if (i + 2 < keypoints.length) {
                        hand1.push({
                            x: keypoints[i],
                            y: keypoints[i + 1],
                            z: keypoints[i + 2]
                        });
                    }
                }
                // Only add if not all zeros and has 21 landmarks
                if (hand1.length === 21 && hand1.some(p => p.x !== 0 || p.y !== 0)) {
                    hands.push({ landmarks: hand1, color: '#00ff00' });  // Green for first hand
                }
            }
            
            // Second hand (landmarks 63-125)
            if (keypoints.length >= 126) {
                const hand2 = [];
                for (let i = 63; i < 126; i += 3) {
                    // Bounds check for array access
                    if (i + 2 < keypoints.length) {
                        hand2.push({
                            x: keypoints[i],
                            y: keypoints[i + 1],
                            z: keypoints[i + 2]
                        });
                    }
                }
                // Only add if not all zeros and has 21 landmarks
                if (hand2.length === 21 && hand2.some(p => p.x !== 0 || p.y !== 0)) {
                    hands.push({ landmarks: hand2, color: '#ff00ff' });  // Magenta for second hand
                }
            }
            
            // Draw each hand
            hands.forEach(hand => {
                this.drawHandSkeleton(hand.landmarks, hand.color, connections);
            });
        } catch (error) {
            console.error('[Visualizer] Error in drawBackendKeypoints:', error);
        }
    }
    
    /**
     * Draw hand skeleton with connections and landmarks
     */
    drawHandSkeleton(landmarks, color, connections) {
        if (!landmarks || landmarks.length < 21) {
            return;
        }
        
        // Draw connections first (underneath)
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.lineCap = 'round';
        
        connections.forEach(([startIdx, endIdx]) => {
            const start = landmarks[startIdx];
            const end = landmarks[endIdx];
            
            if (start && end && start.x > 0 && start.y > 0 && end.x > 0 && end.y > 0) {
                const startX = start.x * this.canvas.width;
                const startY = start.y * this.canvas.height;
                const endX = end.x * this.canvas.width;
                const endY = end.y * this.canvas.height;
                
                this.ctx.beginPath();
                this.ctx.moveTo(startX, startY);
                this.ctx.lineTo(endX, endY);
                this.ctx.stroke();
            }
        });
        
        // Draw landmark points (on top of connections)
        landmarks.forEach((point, index) => {
            if (point.x > 0 && point.y > 0) {
                const x = point.x * this.canvas.width;
                const y = point.y * this.canvas.height;
                
                // Draw landmark circle
                this.ctx.beginPath();
                this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
                this.ctx.fillStyle = color;
                this.ctx.fill();
                
                // White border for visibility
                this.ctx.strokeStyle = '#ffffff';
                this.ctx.lineWidth = 2;
                this.ctx.stroke();
                
                // Make wrist (index 0) larger
                if (index === 0) {
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, 8, 0, 2 * Math.PI);
                    this.ctx.strokeStyle = color;
                    this.ctx.lineWidth = 3;
                    this.ctx.stroke();
                }
            }
        });
    }
    
    /**
     * Update FPS counter
     */
    updateFPS() {
        this.frameCount++;
        const now = Date.now();
        const elapsed = now - this.lastFPSUpdate;
        
        if (elapsed >= 1000) {
            this.currentFPS = Math.round((this.frameCount * 1000) / elapsed);
            this.frameCount = 0;
            this.lastFPSUpdate = now;
        }
    }
    
    /**
     * Draw FPS counter
     */
    drawFPSCounter() {
        const x = this.canvas.width - 70;
        const y = 20;
        
        // Draw background (smaller)
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(x - 5, y - 18, 65, 26);
        
        // Draw FPS text (smaller)
        const fpsColor = this.currentFPS >= 15 ? '#27ae60' : this.currentFPS >= 10 ? '#f39c12' : '#e74c3c';
        this.ctx.fillStyle = fpsColor;
        this.ctx.font = 'bold 14px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`FPS: ${this.currentFPS}`, x, y);
    }
    
    clearPrediction() {
        this.currentPrediction = null;
        this.predictionConfidence = 0;
    }
    
    drawHandLandmarks(landmarks, handedness = []) {
        if (!landmarks || landmarks.length === 0) {
            return;
        }
        
        landmarks.forEach((landmark, handIndex) => {
            const isLeftHand = handedness[handIndex]?.label === 'Left';
            const color = isLeftHand ? '#00FF00' : '#FF0000';
            
            // Draw hand connections
            this.drawHandConnections(landmark, color);
            
            // Draw landmarks
            this.drawLandmarks(landmark, color, handIndex);
        });
    }
    
    drawHandConnections(landmark, color) {
        if (!landmark || landmark.length < 21) {
            return;
        }
        
        // Hand connection indices
        const connections = [
            // Thumb
            [0, 1], [1, 2], [2, 3], [3, 4],
            // Index finger
            [0, 5], [5, 6], [6, 7], [7, 8],
            // Middle finger
            [0, 9], [9, 10], [10, 11], [11, 12],
            // Ring finger
            [0, 13], [13, 14], [14, 15], [15, 16],
            // Pinky
            [0, 17], [17, 18], [18, 19], [19, 20],
            // Palm
            [0, 5], [5, 9], [9, 13], [13, 17]
        ];
        
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        connections.forEach(([start, end]) => {
            if (landmark[start] && landmark[end]) {
                const startX = landmark[start].x * this.canvas.width;
                const startY = landmark[start].y * this.canvas.height;
                const endX = landmark[end].x * this.canvas.width;
                const endY = landmark[end].y * this.canvas.height;
                
                this.ctx.moveTo(startX, startY);
                this.ctx.lineTo(endX, endY);
            }
        });
        
        this.ctx.stroke();
    }
    
    drawLandmarks(landmark, color, handIndex) {
        if (!landmark || landmark.length < 21) {
            return;
        }
        
        landmark.forEach((point, index) => {
            const x = point.x * this.canvas.width;
            const y = point.y * this.canvas.height;
            
            // Draw landmark circle
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Draw landmark index for important points
            if (index % 4 === 0) {
                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.font = '10px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(index.toString(), x, y - 8);
            }
        });
    }
    
    drawFrameProgress(current, target) {
        const progress = current / target;
        const barWidth = 200;
        const barHeight = 4;
        const x = (this.canvas.width - barWidth) / 2;
        const y = this.canvas.height - 50;
        
        // Draw background
        this.ctx.fillStyle = 'rgba(236, 240, 241, 0.8)';
        this.ctx.fillRect(x, y, barWidth, barHeight);
        
        // Draw progress
        this.ctx.fillStyle = '#3498db';
        this.ctx.fillRect(x, y, barWidth * progress, barHeight);
        
        // Draw text
        this.ctx.fillStyle = '#2c3e50';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${current}/${target}`, x + barWidth / 2, y - 5);
    }
    
    drawDetectionStatus(status) {
        const x = 10;
        const y = 10;
        const width = 150;
        const height = 30;
        
        // Status colors
        const colors = {
            'no_hands': '#e74c3c',
            'detected': '#27ae60',
            'collecting_frames': '#f39c12',
            'processing': '#3498db'
        };
        
        const color = colors[status] || colors['no_hands'];
        
        // Draw background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(x, y, width, height);
        
        // Draw status indicator
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(x + 15, y + 15, 5, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Draw status text
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(status.replace('_', ' '), x + 25, y + 20);
    }
    
    handleResize() {
        if (!this.canvas) {
            return;
        }
        
        // Canvas size should match video size
        const video = document.getElementById('webcam');
        if (video && video.videoWidth && video.videoHeight) {
            this.canvas.width = video.videoWidth;
            this.canvas.height = video.videoHeight;
        }
    }
    
    clear() {
        if (this.ctx && this.canvas) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
        
        this.clearPrediction();
        this.currentKeypoints = null;
    }
    
    destroy() {
        console.log('[Visualizer] Destroying...');
        
        // Mark as destroyed first to prevent any further operations
        this.isDestroyed = true;
        
        // Stop animation
        this.stopAnimation();
        
        // Clear canvas
        this.clear();
        
        // Reset all state
        this.currentPrediction = null;
        this.predictionConfidence = 0;
        this.currentKeypoints = null;
        this.lastValidKeypoints = null;
        this.keypointsMissingStart = null;
        this.fpsHistory = [];
        this.currentFPS = 0;
        this.frameCount = 0;
        this.effects = {
            predictionPulse: false,
            confidenceGlow: false,
            landmarkHighlight: false
        };
        
        // Clear references
        this.canvas = null;
        this.ctx = null;
        this.isActive = false;
        
        console.log('[Visualizer] Destroyed');
    }
    
    // Getters
    get active() {
        return this.isActive;
    }
    
    get canvasElement() {
        return this.canvas;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HandVisualizer;
}









