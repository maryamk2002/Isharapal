/**
 * Firebase Configuration for ISHARAPAL
 * 
 * INSTRUCTIONS:
 * 1. Go to https://console.firebase.google.com/
 * 2. Click "Create a project" (or select existing)
 * 3. Name it something like "isharapal-psl"
 * 4. Disable Google Analytics (optional, not needed)
 * 5. Click "Create project"
 * 6. Once created, click the web icon (</>) to add a web app
 * 7. Register app with name "ISHARAPAL Web"
 * 8. Copy the firebaseConfig object below
 * 9. Go to Build > Firestore Database > Create database
 * 10. Choose "Start in test mode" (for development)
 * 11. Select a location close to you
 * 12. Click "Enable"
 * 
 * REPLACE THE PLACEHOLDER VALUES BELOW WITH YOUR FIREBASE CONFIG
 */

const FIREBASE_CONFIG = {
    // ============================================
    // PASTE YOUR FIREBASE CONFIG HERE
    // ============================================
    apiKey: "YOUR_API_KEY_HERE",
    authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT_ID.appspot.com",
    messagingSenderId: "YOUR_SENDER_ID",
    appId: "YOUR_APP_ID"
    // ============================================
};

/**
 * Check if Firebase is configured
 */
function isFirebaseConfigured() {
    return FIREBASE_CONFIG.apiKey !== "YOUR_API_KEY_HERE" &&
           FIREBASE_CONFIG.projectId !== "YOUR_PROJECT_ID";
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
}

