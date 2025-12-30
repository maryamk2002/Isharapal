/**
 * Service Worker for ISHARAPAL Browser-Only Mode
 * 
 * Caches all necessary files for offline use including:
 * - ONNX model (~11MB)
 * - MediaPipe WASM files
 * - HTML, CSS, JS files
 */

const CACHE_NAME = 'isharapal-v2';
const CACHE_VERSION = '2.0.0';

// Files to cache for offline use
const STATIC_CACHE = [
    './',
    './index_browser.html',
    './css/main_v2.css',
    './css/urdu-fonts.css',
    './js/camera.js',
    './js/visualization.js',
    './js/ui.js',
    './js/onnx_predictor.js',
    './js/feedback_manager.js',
    './js/word_formation.js',
    './js/firebase_config.js',
    './js/cloud_sync.js',
    './js/app_browser.js',
    './models/psl_labels.json',
    './models/sign_thresholds.json',
    './models/urdu_mapping.json'
];

// Large files to cache separately (with progress tracking)
const LARGE_CACHE = [
    './models/psl_model_v2.onnx'
];

// External CDN resources (cached on first use)
const CDN_CACHE = [
    'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js',
    'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js',
    'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js',
    'https://www.gstatic.com/firebasejs/9.22.0/firebase-app-compat.js',
    'https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore-compat.js'
];

// Install event - cache static files
self.addEventListener('install', (event) => {
    console.log('[SW] Installing Service Worker v' + CACHE_VERSION);
    
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            console.log('[SW] Caching static files');
            
            // Cache static files first
            return cache.addAll(STATIC_CACHE).then(() => {
                console.log('[SW] Static files cached');
                
                // Cache large files (model) - don't fail install if this fails
                return cache.addAll(LARGE_CACHE).then(() => {
                    console.log('[SW] Model cached');
                }).catch((err) => {
                    console.warn('[SW] Model caching failed (will cache on first use):', err);
                });
            });
        }).then(() => {
            // Skip waiting to activate immediately
            return self.skipWaiting();
        })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('[SW] Activating Service Worker v' + CACHE_VERSION);
    
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('[SW] Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => {
            // Take control of all pages immediately
            return self.clients.claim();
        })
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    
    // Skip non-GET requests
    if (event.request.method !== 'GET') {
        return;
    }
    
    // Handle CDN requests (cache on first use)
    if (url.origin !== location.origin) {
        event.respondWith(
            caches.match(event.request).then((cachedResponse) => {
                if (cachedResponse) {
                    return cachedResponse;
                }
                
                return fetch(event.request).then((response) => {
                    // Cache CDN resources
                    if (response.ok && CDN_CACHE.some(cdn => event.request.url.includes(cdn.split('/').pop()))) {
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(event.request, responseClone);
                            console.log('[SW] Cached CDN resource:', url.pathname);
                        });
                    }
                    return response;
                }).catch(() => {
                    console.warn('[SW] CDN fetch failed:', url.href);
                    return cachedResponse;
                });
            })
        );
        return;
    }
    
    // Handle local requests with cache-first strategy
    event.respondWith(
        caches.match(event.request).then((cachedResponse) => {
            if (cachedResponse) {
                // Return cached response
                return cachedResponse;
            }
            
            // Not in cache, fetch from network
            return fetch(event.request).then((response) => {
                // Cache the response for future use
                if (response.ok) {
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, responseClone);
                        console.log('[SW] Cached:', url.pathname);
                    });
                }
                return response;
            }).catch((err) => {
                console.error('[SW] Fetch failed:', url.pathname, err);
                
                // Return offline fallback if available
                if (url.pathname.endsWith('.html') || url.pathname === '/') {
                    return caches.match('./index_browser.html');
                }
                
                throw err;
            });
        })
    );
});

// Message event - for cache updates and progress
self.addEventListener('message', (event) => {
    if (event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
    
    if (event.data.type === 'GET_CACHE_STATUS') {
        caches.open(CACHE_NAME).then((cache) => {
            cache.keys().then((keys) => {
                event.ports[0].postMessage({
                    cached: keys.length,
                    version: CACHE_VERSION
                });
            });
        });
    }
});

console.log('[SW] Service Worker loaded');

