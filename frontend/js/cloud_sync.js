/**
 * Cloud Sync Module for ISHARAPAL
 * 
 * Handles synchronizing feedback data to Firebase Firestore.
 * 
 * Features:
 * - Batch writes (collect locally, sync periodically)
 * - Rate limiting (max 100 writes per session)
 * - Offline-first (always save locally first)
 * - Anonymous device ID (no personal data)
 * - Automatic retry on failure
 */

class CloudSync {
    constructor(options = {}) {
        // Configuration
        this.config = {
            syncIntervalMs: options.syncIntervalMs || 5 * 60 * 1000,  // 5 minutes
            maxWritesPerSession: options.maxWritesPerSession || 100,
            batchSize: options.batchSize || 10,
            collectionName: options.collectionName || 'psl_feedback',
            retryDelayMs: options.retryDelayMs || 30000,  // 30 seconds
            appVersion: options.appVersion || '1.0.0'
        };
        
        // State
        this.isInitialized = false;
        this.isOnline = navigator.onLine;
        this.db = null;  // Firestore instance
        
        // Pending feedback queue (to be synced)
        this.pendingQueue = [];
        this.PENDING_QUEUE_KEY = 'psl_pending_sync';
        
        // Session stats
        this.sessionStats = {
            writes: 0,
            errors: 0,
            lastSync: null
        };
        
        // Device ID (anonymous, persisted locally)
        this.deviceId = this.getOrCreateDeviceId();
        
        // Sync interval
        this.syncInterval = null;
        
        // Event callbacks
        this.onSyncComplete = null;
        this.onSyncError = null;
        this.onConnectionChange = null;
        
        console.log('[CloudSync] Module created');
    }
    
    /**
     * Initialize the cloud sync module.
     * Requires Firebase to be loaded and configured.
     */
    async init() {
        try {
            console.log('[CloudSync] Initializing...');
            
            // Check if Firebase is configured
            if (!window.isFirebaseConfigured || !window.isFirebaseConfigured()) {
                console.warn('[CloudSync] Firebase not configured. Cloud sync disabled.');
                console.info('[CloudSync] To enable, edit js/firebase_config.js with your Firebase project details.');
                return false;
            }
            
            // Check if Firebase SDK is loaded
            if (typeof firebase === 'undefined') {
                console.warn('[CloudSync] Firebase SDK not loaded. Cloud sync disabled.');
                return false;
            }
            
            // Initialize Firebase (if not already)
            if (!firebase.apps.length) {
                const config = window.getFirebaseConfig();
                firebase.initializeApp(config);
                console.log('[CloudSync] Firebase app initialized');
            }
            
            // Get Firestore instance
            this.db = firebase.firestore();
            
            // Enable offline persistence (optional but recommended)
            try {
                await this.db.enablePersistence({ synchronizeTabs: true });
                console.log('[CloudSync] Firestore offline persistence enabled');
            } catch (err) {
                if (err.code === 'failed-precondition') {
                    console.warn('[CloudSync] Multiple tabs open, persistence limited to one tab');
                } else if (err.code === 'unimplemented') {
                    console.warn('[CloudSync] Browser does not support persistence');
                }
            }
            
            // Load pending queue from storage
            this.loadPendingQueue();
            
            // Set up online/offline listeners
            window.addEventListener('online', () => {
                this.isOnline = true;
                console.log('[CloudSync] Back online');
                if (this.onConnectionChange) this.onConnectionChange(true);
                // Trigger sync when back online
                this.sync();
            });
            
            window.addEventListener('offline', () => {
                this.isOnline = false;
                console.log('[CloudSync] Gone offline');
                if (this.onConnectionChange) this.onConnectionChange(false);
            });
            
            // Sync on page close/unload
            window.addEventListener('beforeunload', () => {
                this.savePendingQueue();
                // Try to sync remaining items
                if (this.pendingQueue.length > 0 && this.isOnline) {
                    // Use sendBeacon for reliable last-minute sync
                    this.syncBeacon();
                }
            });
            
            // Start periodic sync
            this.startPeriodicSync();
            
            this.isInitialized = true;
            console.log('[CloudSync] Initialized successfully');
            console.log(`[CloudSync] Device ID: ${this.deviceId}`);
            console.log(`[CloudSync] Pending items: ${this.pendingQueue.length}`);
            
            // Initial sync if there are pending items
            if (this.pendingQueue.length > 0 && this.isOnline) {
                setTimeout(() => this.sync(), 5000);  // Wait 5 seconds then sync
            }
            
            return true;
            
        } catch (error) {
            console.error('[CloudSync] Initialization failed:', error);
            return false;
        }
    }
    
    /**
     * Get or create anonymous device ID.
     */
    getOrCreateDeviceId() {
        const DEVICE_ID_KEY = 'psl_device_id';
        let deviceId = localStorage.getItem(DEVICE_ID_KEY);
        
        if (!deviceId) {
            // Generate random UUID-like string
            deviceId = 'anon_' + Date.now().toString(36) + '_' + 
                       Math.random().toString(36).substring(2, 10);
            localStorage.setItem(DEVICE_ID_KEY, deviceId);
        }
        
        return deviceId;
    }
    
    /**
     * Add feedback to the sync queue.
     * @param {Object} feedback - Feedback data to sync
     */
    addToQueue(feedback) {
        if (this.sessionStats.writes >= this.config.maxWritesPerSession) {
            console.warn('[CloudSync] Session write limit reached');
            return false;
        }
        
        const syncItem = {
            ...feedback,
            deviceId: this.deviceId,
            appVersion: this.config.appVersion,
            queuedAt: Date.now(),
            synced: false
        };
        
        this.pendingQueue.push(syncItem);
        this.savePendingQueue();
        
        console.log(`[CloudSync] Added to queue. Pending: ${this.pendingQueue.length}`);
        
        return true;
    }
    
    /**
     * Start periodic sync.
     */
    startPeriodicSync() {
        if (this.syncInterval) {
            clearInterval(this.syncInterval);
        }
        
        this.syncInterval = setInterval(() => {
            if (this.isOnline && this.pendingQueue.length > 0) {
                this.sync();
            }
        }, this.config.syncIntervalMs);
        
        console.log(`[CloudSync] Periodic sync started (every ${this.config.syncIntervalMs / 1000}s)`);
    }
    
    /**
     * Stop periodic sync.
     */
    stopPeriodicSync() {
        if (this.syncInterval) {
            clearInterval(this.syncInterval);
            this.syncInterval = null;
        }
    }
    
    /**
     * Sync pending items to Firestore.
     */
    async sync() {
        if (!this.isInitialized || !this.db) {
            console.warn('[CloudSync] Not initialized, skipping sync');
            return;
        }
        
        if (!this.isOnline) {
            console.log('[CloudSync] Offline, skipping sync');
            return;
        }
        
        if (this.pendingQueue.length === 0) {
            console.log('[CloudSync] Nothing to sync');
            return;
        }
        
        console.log(`[CloudSync] Syncing ${this.pendingQueue.length} items...`);
        
        try {
            const batch = this.db.batch();
            const itemsToSync = this.pendingQueue.slice(0, this.config.batchSize);
            
            for (const item of itemsToSync) {
                const docRef = this.db.collection(this.config.collectionName).doc();
                
                // Prepare data for Firestore
                const firestoreData = {
                    predictedLabel: item.predictedLabel || item.label,
                    correctLabel: item.correctLabel || null,
                    confidence: item.confidence,
                    isCorrect: item.isCorrect !== undefined ? item.isCorrect : (item.type === 'correct'),
                    timestamp: firebase.firestore.FieldValue.serverTimestamp(),
                    deviceId: item.deviceId,
                    appVersion: item.appVersion,
                    clientTimestamp: item.timestamp || item.queuedAt
                };
                
                batch.set(docRef, firestoreData);
            }
            
            // Commit batch
            await batch.commit();
            
            // Remove synced items from queue
            this.pendingQueue = this.pendingQueue.slice(itemsToSync.length);
            this.savePendingQueue();
            
            // Update stats
            this.sessionStats.writes += itemsToSync.length;
            this.sessionStats.lastSync = Date.now();
            
            console.log(`[CloudSync] Synced ${itemsToSync.length} items. Remaining: ${this.pendingQueue.length}`);
            
            if (this.onSyncComplete) {
                this.onSyncComplete({
                    synced: itemsToSync.length,
                    remaining: this.pendingQueue.length,
                    sessionWrites: this.sessionStats.writes
                });
            }
            
            // Continue syncing if more items remain
            if (this.pendingQueue.length > 0) {
                setTimeout(() => this.sync(), 1000);  // Wait 1 second between batches
            }
            
        } catch (error) {
            console.error('[CloudSync] Sync failed:', error);
            this.sessionStats.errors++;
            
            if (this.onSyncError) {
                this.onSyncError(error);
            }
            
            // Retry after delay
            setTimeout(() => this.sync(), this.config.retryDelayMs);
        }
    }
    
    /**
     * Use sendBeacon for last-minute sync (on page close).
     * Note: This is a fallback and may not work with Firestore directly.
     */
    syncBeacon() {
        // Firestore doesn't support sendBeacon directly
        // Just save the queue for next session
        this.savePendingQueue();
        console.log('[CloudSync] Queue saved for next session');
    }
    
    /**
     * Load pending queue from localStorage.
     */
    loadPendingQueue() {
        try {
            const stored = localStorage.getItem(this.PENDING_QUEUE_KEY);
            if (stored) {
                this.pendingQueue = JSON.parse(stored);
            }
        } catch (e) {
            console.warn('[CloudSync] Failed to load pending queue:', e);
            this.pendingQueue = [];
        }
    }
    
    /**
     * Save pending queue to localStorage.
     */
    savePendingQueue() {
        try {
            localStorage.setItem(this.PENDING_QUEUE_KEY, JSON.stringify(this.pendingQueue));
        } catch (e) {
            console.warn('[CloudSync] Failed to save pending queue:', e);
        }
    }
    
    /**
     * Get sync status.
     */
    getStatus() {
        return {
            initialized: this.isInitialized,
            online: this.isOnline,
            pendingItems: this.pendingQueue.length,
            sessionWrites: this.sessionStats.writes,
            sessionErrors: this.sessionStats.errors,
            lastSync: this.sessionStats.lastSync,
            deviceId: this.deviceId
        };
    }
    
    /**
     * Force sync now.
     */
    async forceSync() {
        if (!this.isInitialized) {
            console.warn('[CloudSync] Not initialized');
            return false;
        }
        
        await this.sync();
        return true;
    }
    
    /**
     * Clear pending queue (for testing/debugging).
     */
    clearQueue() {
        this.pendingQueue = [];
        this.savePendingQueue();
        console.log('[CloudSync] Queue cleared');
    }
    
    /**
     * Cleanup.
     */
    destroy() {
        this.stopPeriodicSync();
        this.savePendingQueue();
        console.log('[CloudSync] Destroyed');
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.CloudSync = CloudSync;
}

