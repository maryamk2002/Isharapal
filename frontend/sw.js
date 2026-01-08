/**
 * Service Worker for ISHARAPAL Browser-Only Mode
 * 
 * Caches all necessary files for offline use including:
 * - ONNX model (~11MB)
 * - MediaPipe WASM files
 * - HTML, CSS, JS files
 */

const CACHE_NAME = 'isharapal-v5';
const CACHE_VERSION = '5.0.0';  // UPDATED: Performance optimizations + sign images cache

// Files to cache for offline use
const STATIC_CACHE = [
    './',
    './index_browser.html',
    './css/main_v2.css',
    './css/urdu-fonts.css',
    './css/analytics.css',
    './css/innovations.css',
    './js/camera.js',
    './js/visualization.js',
    './js/ui.js',
    './js/onnx_predictor.js',
    './js/feedback_manager.js',
    './js/word_formation.js',
    './js/firebase_config.js',
    './js/cloud_sync.js',
    './js/analytics.js',
    './js/practice-mode.js',
    './js/disambiguation.js',      // ADDED: Missing innovation module
    './js/word-shortcuts.js',      // ADDED: Missing innovation module
    './js/pip-mode.js',            // ADDED: Missing innovation module
    './js/app_browser.js',
    './models/psl_labels.json',
    './models/sign_thresholds.json',
    './models/urdu_mapping.json',
    './models/model_version.json',
    './manifest.json'              // ADDED: PWA manifest
];

// Large files to cache separately (with progress tracking)
const LARGE_CACHE = [
    './models/psl_model_v2.onnx'
];

// Sign images for practice mode (cached for offline use)
const SIGN_IMAGES_CACHE = [
    './assets/signs/manifest.json',
    './assets/signs/1-Hay.webp', './assets/signs/1-Hay.jpg',
    './assets/signs/2-Hay.webp', './assets/signs/2-Hay.jpg',
    './assets/signs/Ain.webp', './assets/signs/Ain.jpg',
    './assets/signs/Alif.webp', './assets/signs/Alif.jpg',
    './assets/signs/Alifmad.webp', './assets/signs/Alifmad.jpg',
    './assets/signs/Aray.webp', './assets/signs/Aray.jpg',
    './assets/signs/Bay.webp', './assets/signs/Bay.jpg',
    './assets/signs/Byeh.webp', './assets/signs/Byeh.jpg',
    './assets/signs/Chay.webp', './assets/signs/Chay.jpg',
    './assets/signs/Cyeh.webp', './assets/signs/Cyeh.jpg',
    './assets/signs/Daal.webp', './assets/signs/Daal.jpg',
    './assets/signs/Dal.webp', './assets/signs/Dal.jpg',
    './assets/signs/Dochahay.webp', './assets/signs/Dochahay.jpg',
    './assets/signs/Fay.webp', './assets/signs/Fay.jpg',
    './assets/signs/Gaaf.webp', './assets/signs/Gaaf.jpg',
    './assets/signs/Ghain.webp', './assets/signs/Ghain.jpg',
    './assets/signs/Hamza.webp', './assets/signs/Hamza.jpg',
    './assets/signs/Jeem.webp', './assets/signs/Jeem.jpg',
    './assets/signs/Kaf.webp', './assets/signs/Kaf.jpg',
    './assets/signs/Khay.webp', './assets/signs/Khay.jpg',
    './assets/signs/Kiaf.webp', './assets/signs/Kiaf.jpg',
    './assets/signs/Lam.webp', './assets/signs/Lam.jpg',
    './assets/signs/Meem.webp', './assets/signs/Meem.jpg',
    './assets/signs/Nuun.webp', './assets/signs/Nuun.jpg',
    './assets/signs/Nuungh.webp', './assets/signs/Nuungh.jpg',
    './assets/signs/Pay.webp', './assets/signs/Pay.jpg',
    './assets/signs/Ray.webp', './assets/signs/Ray.jpg',
    './assets/signs/Say.webp', './assets/signs/Say.jpg',
    './assets/signs/Seen.webp', './assets/signs/Seen.jpg',
    './assets/signs/Sheen.webp', './assets/signs/Sheen.jpg',
    './assets/signs/Suad.webp', './assets/signs/Suad.jpg',
    './assets/signs/Taay.webp', './assets/signs/Taay.jpg',
    './assets/signs/Tay.webp', './assets/signs/Tay.jpg',
    './assets/signs/Tuey.webp', './assets/signs/Tuey.jpg',
    './assets/signs/Wao.webp', './assets/signs/Wao.jpg',
    './assets/signs/Zaal.webp', './assets/signs/Zaal.jpg',
    './assets/signs/Zaey.webp', './assets/signs/Zaey.jpg',
    './assets/signs/Zay.webp', './assets/signs/Zay.jpg',
    './assets/signs/Zuad.webp', './assets/signs/Zuad.jpg',
    './assets/signs/Zuey.webp', './assets/signs/Zuey.jpg'
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
                    
                    // Cache sign images for practice mode - don't fail if missing
                    return cache.addAll(SIGN_IMAGES_CACHE).then(() => {
                        console.log('[SW] Sign images cached for offline practice mode');
                    }).catch((err) => {
                        console.warn('[SW] Sign images caching failed (will cache on first use):', err);
                    });
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

