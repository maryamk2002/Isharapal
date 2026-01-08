/**
 * Firebase Configuration for ISHARAPAL
 * ✅ CONFIGURED AND READY TO USE
 * 
 * Project: isharapal-psl
 * Status: Active
 */

const FIREBASE_CONFIG = {
    apiKey: "AIzaSyBBfbPeaAlzZrW8f5i46SKKemX_zLbR-Fs",
    authDomain: "isharapal-psl.firebaseapp.com",
    projectId: "isharapal-psl",
    storageBucket: "isharapal-psl.firebasestorage.app",
    messagingSenderId: "545007445528",
    appId: "1:545007445528:web:31dfc22cd4566d6671856d"
};

/**
 * Check if Firebase is configured (has valid non-placeholder values)
 */
function isFirebaseConfigured() {
    // Check that config has real values (not placeholders)
    return FIREBASE_CONFIG.apiKey && 
           FIREBASE_CONFIG.apiKey !== "YOUR_API_KEY_HERE" &&
           FIREBASE_CONFIG.projectId && 
           FIREBASE_CONFIG.projectId !== "YOUR_PROJECT_ID" &&
           FIREBASE_CONFIG.apiKey.startsWith("AIza");
}

/**
 * Get the Firebase config object
 */
function getFirebaseConfig() {
    if (!isFirebaseConfigured()) {
        console.warn('[Firebase] Not configured. Edit firebase_config.js with your project details.');
        return null;
    }
    return FIREBASE_CONFIG;
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.FIREBASE_CONFIG = FIREBASE_CONFIG;
    window.isFirebaseConfigured = isFirebaseConfigured;
    window.getFirebaseConfig = getFirebaseConfig;
    
    // Log status on load
    console.log('[Firebase] Config loaded:', isFirebaseConfigured() ? '✅ Ready' : '❌ Not configured');
}

