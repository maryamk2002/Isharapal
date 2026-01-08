/**
 * Word Shortcuts Module for PSL Recognition System
 * 
 * Enables quick word/phrase insertion:
 * - Common Urdu word shortcuts (سلام، شکریہ، etc.)
 * - User-defined custom shortcuts
 * - Gesture sequence detection for common patterns
 * - Quick phrase panel UI
 */

class WordShortcuts {
    constructor(options = {}) {
        // Configuration
        this.config = {
            maxCustomShortcuts: options.maxCustomShortcuts || 20,
            sequenceTimeoutMs: options.sequenceTimeoutMs || 2000,
            ...options
        };
        
        // Built-in common word shortcuts
        this.commonWords = [
            // Greetings
            { id: 'salam', urdu: 'سلام', romanized: 'Salam', category: 'greetings', emoji: '👋' },
            { id: 'shukria', urdu: 'شکریہ', romanized: 'Shukria', category: 'greetings', emoji: '🙏' },
            { id: 'khudahafiz', urdu: 'خدا حافظ', romanized: 'Khuda Hafiz', category: 'greetings', emoji: '👋' },
            { id: 'walaikum', urdu: 'وعلیکم', romanized: 'Walaikum', category: 'greetings', emoji: '🤝' },
            
            // Questions
            { id: 'kya', urdu: 'کیا', romanized: 'Kya', category: 'questions', emoji: '❓' },
            { id: 'kyun', urdu: 'کیوں', romanized: 'Kyun', category: 'questions', emoji: '🤔' },
            { id: 'kab', urdu: 'کب', romanized: 'Kab', category: 'questions', emoji: '⏰' },
            { id: 'kahan', urdu: 'کہاں', romanized: 'Kahan', category: 'questions', emoji: '📍' },
            { id: 'kaun', urdu: 'کون', romanized: 'Kaun', category: 'questions', emoji: '👤' },
            { id: 'kitna', urdu: 'کتنا', romanized: 'Kitna', category: 'questions', emoji: '🔢' },
            
            // Common words
            { id: 'haan', urdu: 'ہاں', romanized: 'Haan', category: 'common', emoji: '✅' },
            { id: 'nahi', urdu: 'نہیں', romanized: 'Nahi', category: 'common', emoji: '❌' },
            { id: 'acha', urdu: 'اچھا', romanized: 'Acha', category: 'common', emoji: '👍' },
            { id: 'theek', urdu: 'ٹھیک', romanized: 'Theek', category: 'common', emoji: '👌' },
            { id: 'mujhe', urdu: 'مجھے', romanized: 'Mujhe', category: 'common', emoji: '👆' },
            { id: 'aap', urdu: 'آپ', romanized: 'Aap', category: 'common', emoji: '🫵' },
            
            // Needs
            { id: 'pani', urdu: 'پانی', romanized: 'Pani', category: 'needs', emoji: '💧' },
            { id: 'khana', urdu: 'کھانا', romanized: 'Khana', category: 'needs', emoji: '🍽️' },
            { id: 'madad', urdu: 'مدد', romanized: 'Madad', category: 'needs', emoji: '🆘' },
            { id: 'dawa', urdu: 'دوا', romanized: 'Dawa', category: 'needs', emoji: '💊' },
            
            // Time
            { id: 'abhi', urdu: 'ابھی', romanized: 'Abhi', category: 'time', emoji: '⏱️' },
            { id: 'baad', urdu: 'بعد', romanized: 'Baad', category: 'time', emoji: '⏳' },
            { id: 'kal', urdu: 'کل', romanized: 'Kal', category: 'time', emoji: '📅' },
            { id: 'aaj', urdu: 'آج', romanized: 'Aaj', category: 'time', emoji: '📆' },
            
            // Feelings
            { id: 'khush', urdu: 'خوش', romanized: 'Khush', category: 'feelings', emoji: '😊' },
            { id: 'udas', urdu: 'اداس', romanized: 'Udas', category: 'feelings', emoji: '😢' },
            { id: 'thaka', urdu: 'تھکا', romanized: 'Thaka', category: 'feelings', emoji: '😴' },
            { id: 'bhook', urdu: 'بھوک', romanized: 'Bhook', category: 'feelings', emoji: '🍽️' },
            { id: 'pyaas', urdu: 'پیاس', romanized: 'Pyaas', category: 'feelings', emoji: '💧' }
        ];
        
        // User-defined custom shortcuts
        this.customShortcuts = [];
        
        // Categories for UI organization
        this.categories = [
            { id: 'greetings', nameUrdu: 'سلام', nameEnglish: 'Greetings', icon: '👋' },
            { id: 'questions', nameUrdu: 'سوالات', nameEnglish: 'Questions', icon: '❓' },
            { id: 'common', nameUrdu: 'عام', nameEnglish: 'Common', icon: '💬' },
            { id: 'needs', nameUrdu: 'ضروریات', nameEnglish: 'Needs', icon: '🙋' },
            { id: 'time', nameUrdu: 'وقت', nameEnglish: 'Time', icon: '⏰' },
            { id: 'feelings', nameUrdu: 'احساسات', nameEnglish: 'Feelings', icon: '😊' },
            { id: 'custom', nameUrdu: 'اپنے', nameEnglish: 'Custom', icon: '⭐' }
        ];
        
        // Gesture sequences for quick detection
        this.sequencePatterns = new Map();
        this.currentSequence = [];
        this.sequenceTimer = null;
        
        // Callbacks
        this.onWordInsert = null;
        this.onSequenceMatch = null;
        
        // DOM Elements
        this.elements = {};
        this.isVisible = false;
    }
    
    /**
     * Initialize the word shortcuts module
     */
    async init() {
        // Load custom shortcuts from localStorage
        this.loadCustomShortcuts();
        
        // Initialize DOM
        this.initDOMElements();
        this.setupEventListeners();
        
        console.log(`[WordShortcuts] Initialized with ${this.commonWords.length} common words, ${this.customShortcuts.length} custom`);
        return true;
    }
    
    /**
     * Initialize DOM element references
     */
    initDOMElements() {
        this.elements = {
            panel: document.getElementById('shortcutsPanel'),
            toggleBtn: document.getElementById('shortcutsToggle'),
            closeBtn: document.getElementById('shortcutsClose'),
            categoryTabs: document.getElementById('shortcutsCategoryTabs'),
            wordGrid: document.getElementById('shortcutsWordGrid'),
            searchInput: document.getElementById('shortcutsSearch'),
            addCustomBtn: document.getElementById('shortcutsAddCustom'),
            customModal: document.getElementById('customShortcutModal'),
            customUrduInput: document.getElementById('customUrduInput'),
            customRomanizedInput: document.getElementById('customRomanizedInput'),
            customSaveBtn: document.getElementById('customSaveBtn'),
            customCancelBtn: document.getElementById('customCancelBtn')
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
        
        // Close button
        if (this.elements.closeBtn) {
            this.elements.closeBtn.addEventListener('click', () => this.hide());
        }
        
        // Search input
        if (this.elements.searchInput) {
            this.elements.searchInput.addEventListener('input', (e) => {
                this.filterWords(e.target.value);
            });
        }
        
        // Add custom button
        if (this.elements.addCustomBtn) {
            this.elements.addCustomBtn.addEventListener('click', () => this.showCustomModal());
        }
        
        // Custom modal buttons
        if (this.elements.customSaveBtn) {
            this.elements.customSaveBtn.addEventListener('click', () => this.saveCustomShortcut());
        }
        
        if (this.elements.customCancelBtn) {
            this.elements.customCancelBtn.addEventListener('click', () => this.hideCustomModal());
        }
        
        // Keyboard shortcut (Ctrl+W to toggle)
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'w') {
                e.preventDefault();
                this.toggle();
            }
        });
    }
    
    /**
     * Show the shortcuts panel
     */
    show() {
        this.isVisible = true;
        
        if (this.elements.panel) {
            this.elements.panel.classList.add('visible');
        }
        
        // Render categories and words
        this.renderCategories();
        this.renderWords('all');
    }
    
    /**
     * Hide the shortcuts panel
     */
    hide() {
        this.isVisible = false;
        
        if (this.elements.panel) {
            this.elements.panel.classList.remove('visible');
        }
    }
    
    /**
     * Toggle panel visibility
     */
    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }
    
    /**
     * Render category tabs
     */
    renderCategories() {
        if (!this.elements.categoryTabs) return;
        
        const allTab = `
            <button class="category-tab active" data-category="all">
                <span class="category-icon">📋</span>
                <span class="category-name">All</span>
            </button>
        `;
        
        const categoryTabs = this.categories.map(cat => `
            <button class="category-tab" data-category="${cat.id}">
                <span class="category-icon">${cat.icon}</span>
                <span class="category-name">${cat.nameEnglish}</span>
            </button>
        `).join('');
        
        this.elements.categoryTabs.innerHTML = allTab + categoryTabs;
        
        // Add click handlers
        this.elements.categoryTabs.querySelectorAll('.category-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // Update active state
                this.elements.categoryTabs.querySelectorAll('.category-tab').forEach(t => {
                    t.classList.remove('active');
                });
                tab.classList.add('active');
                
                // Render words for category
                this.renderWords(tab.dataset.category);
            });
        });
    }
    
    /**
     * Render word grid
     * @param {string} category - Category to filter by, or 'all'
     */
    renderWords(category = 'all') {
        if (!this.elements.wordGrid) return;
        
        // Get filtered words
        let words = [...this.commonWords];
        
        // Add custom shortcuts
        if (category === 'all' || category === 'custom') {
            words = words.concat(this.customShortcuts.map(s => ({...s, category: 'custom'})));
        }
        
        // Filter by category
        if (category !== 'all') {
            words = words.filter(w => w.category === category);
        }
        
        // Render grid
        this.elements.wordGrid.innerHTML = words.map(word => `
            <button class="word-shortcut-btn" data-id="${word.id}" data-urdu="${word.urdu}">
                <span class="word-emoji">${word.emoji || '📝'}</span>
                <span class="word-urdu">${word.urdu}</span>
                <span class="word-romanized">${word.romanized}</span>
            </button>
        `).join('');
        
        // Add click handlers
        this.elements.wordGrid.querySelectorAll('.word-shortcut-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.insertWord(btn.dataset.urdu, btn.dataset.id);
            });
        });
    }
    
    /**
     * Filter words by search query
     */
    filterWords(query) {
        if (!this.elements.wordGrid) return;
        
        const lowerQuery = query.toLowerCase();
        
        this.elements.wordGrid.querySelectorAll('.word-shortcut-btn').forEach(btn => {
            const urdu = btn.dataset.urdu;
            const romanized = btn.querySelector('.word-romanized').textContent.toLowerCase();
            
            const matches = urdu.includes(query) || romanized.includes(lowerQuery);
            btn.style.display = matches ? '' : 'none';
        });
    }
    
    /**
     * Insert a word
     */
    insertWord(urdu, id) {
        console.log(`[WordShortcuts] Inserting word: ${urdu} (${id})`);
        
        if (this.onWordInsert) {
            this.onWordInsert({
                urdu: urdu,
                id: id
            });
        }
        
        // Hide panel after insertion
        this.hide();
    }
    
    /**
     * Show custom shortcut modal
     */
    showCustomModal() {
        if (this.elements.customModal) {
            this.elements.customModal.style.display = 'flex';
            
            // Clear inputs
            if (this.elements.customUrduInput) {
                this.elements.customUrduInput.value = '';
                this.elements.customUrduInput.focus();
            }
            if (this.elements.customRomanizedInput) {
                this.elements.customRomanizedInput.value = '';
            }
        }
    }
    
    /**
     * Hide custom shortcut modal
     */
    hideCustomModal() {
        if (this.elements.customModal) {
            this.elements.customModal.style.display = 'none';
        }
    }
    
    /**
     * Save custom shortcut
     */
    saveCustomShortcut() {
        const urdu = this.elements.customUrduInput?.value?.trim();
        const romanized = this.elements.customRomanizedInput?.value?.trim();
        
        if (!urdu) {
            alert('Please enter the Urdu word');
            return;
        }
        
        // Generate ID
        const id = `custom_${Date.now()}`;
        
        // Add to custom shortcuts
        this.customShortcuts.push({
            id: id,
            urdu: urdu,
            romanized: romanized || urdu,
            category: 'custom',
            emoji: '⭐'
        });
        
        // Enforce limit
        if (this.customShortcuts.length > this.config.maxCustomShortcuts) {
            this.customShortcuts.shift(); // Remove oldest
        }
        
        // Save to localStorage
        this.saveCustomShortcuts();
        
        // Hide modal
        this.hideCustomModal();
        
        // Refresh display
        this.renderWords('custom');
        
        console.log(`[WordShortcuts] Added custom shortcut: ${urdu}`);
    }
    
    /**
     * Delete a custom shortcut
     */
    deleteCustomShortcut(id) {
        const idx = this.customShortcuts.findIndex(s => s.id === id);
        if (idx !== -1) {
            this.customShortcuts.splice(idx, 1);
            this.saveCustomShortcuts();
            this.renderWords('custom');
        }
    }
    
    /**
     * Load custom shortcuts from localStorage
     */
    loadCustomShortcuts() {
        try {
            const saved = localStorage.getItem('psl_custom_shortcuts');
            if (saved) {
                this.customShortcuts = JSON.parse(saved);
            }
        } catch (e) {
            console.warn('[WordShortcuts] Could not load custom shortcuts');
        }
    }
    
    /**
     * Save custom shortcuts to localStorage
     */
    saveCustomShortcuts() {
        try {
            localStorage.setItem('psl_custom_shortcuts', JSON.stringify(this.customShortcuts));
        } catch (e) {
            console.warn('[WordShortcuts] Could not save custom shortcuts');
        }
    }
    
    /**
     * Add a gesture sequence pattern
     * @param {Array} sequence - Array of sign labels
     * @param {Object} word - Word to insert when sequence matches
     */
    addSequencePattern(sequence, word) {
        const key = sequence.join('→');
        this.sequencePatterns.set(key, word);
    }
    
    /**
     * Process a sign prediction for sequence detection
     * @param {string} label - Predicted sign label
     */
    processSignForSequence(label) {
        // Clear timeout
        if (this.sequenceTimer) {
            clearTimeout(this.sequenceTimer);
        }
        
        // Add to current sequence
        this.currentSequence.push(label);
        
        // Check for matches
        const key = this.currentSequence.join('→');
        const match = this.sequencePatterns.get(key);
        
        if (match) {
            // Found a match!
            if (this.onSequenceMatch) {
                this.onSequenceMatch({
                    sequence: [...this.currentSequence],
                    word: match
                });
            }
            
            // Clear sequence
            this.currentSequence = [];
            return match;
        }
        
        // Check if current sequence could lead to a match
        let hasPrefix = false;
        for (const [patternKey] of this.sequencePatterns) {
            if (patternKey.startsWith(key)) {
                hasPrefix = true;
                break;
            }
        }
        
        if (!hasPrefix) {
            // Current sequence won't lead to any match, reset
            this.currentSequence = [label];
        }
        
        // Set timeout to clear sequence
        this.sequenceTimer = setTimeout(() => {
            this.currentSequence = [];
        }, this.config.sequenceTimeoutMs);
        
        return null;
    }
    
    /**
     * Get all words (common + custom)
     */
    getAllWords() {
        return [...this.commonWords, ...this.customShortcuts];
    }
    
    /**
     * Get words by category
     */
    getWordsByCategory(category) {
        if (category === 'custom') {
            return [...this.customShortcuts];
        }
        return this.commonWords.filter(w => w.category === category);
    }
    
    /**
     * Search for words
     */
    searchWords(query) {
        const lowerQuery = query.toLowerCase();
        const allWords = this.getAllWords();
        
        return allWords.filter(w => 
            w.urdu.includes(query) ||
            w.romanized.toLowerCase().includes(lowerQuery)
        );
    }
    
    /**
     * Destroy the module
     */
    destroy() {
        this.hide();
        if (this.sequenceTimer) {
            clearTimeout(this.sequenceTimer);
        }
    }
}


// ================================================================================
// EXPORTS
// ================================================================================

if (typeof window !== 'undefined') {
    window.WordShortcuts = WordShortcuts;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WordShortcuts };
}


