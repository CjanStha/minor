// CafeLocate - Authentication Module
// Handles user registration and login with username/email and password

class AuthManager {
    constructor() {
        this.user = null;
        this.token = null;
        this.uiManager = null;
        this.init();
    }

    init() {
        // Check for existing session
        const savedToken = localStorage.getItem('auth_token');
        const savedUser = localStorage.getItem('user_data');
        const isGuest = localStorage.getItem('is_guest') === 'true';

        if (isGuest && savedUser) {
            this.user = JSON.parse(savedUser);
            this.token = null;
            this.showMapPage();
        } else if (savedToken && savedUser) {
            this.token = savedToken;
            this.user = JSON.parse(savedUser);
            this.showMapPage();
        }
    }

    /**
     * Set UI Manager dependency
     * @param {UIManager} uiManager - The UI manager instance
     */
    setUIManager(uiManager) {
        this.uiManager = uiManager;
    }

    /**
     * Login user with credentials
     * @param {string} username - Username or email
     * @param {string} password - User password
     * @returns {Promise<Object>} - Login result
     */
    async login(username, password) {
        try {
            const response = await fetch('http://localhost:8000/api/auth/login/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const data = await response.json();

            if (response.ok) {
                this.user = data.user;
                this.token = data.token;

                // Save to localStorage
                localStorage.setItem('auth_token', this.token);
                localStorage.setItem('user_data', JSON.stringify(this.user));
                localStorage.removeItem('is_guest');

                this.showMapPage();
                return { success: true, user: this.user };
            } else {
                return { success: false, message: data.detail || 'Login failed' };
            }
        } catch (error) {
            console.error('Login error:', error);
            return { success: false, message: 'Network error. Please try again.' };
        }
    }

    /**
     * Register new user
     * @param {string} username - Desired username
     * @param {string} email - User email
     * @param {string} password - User password
     * @returns {Promise<Object>} - Registration result
     */
    async register(username, email, password) {
        try {
            const response = await fetch('http://localhost:8000/api/auth/register/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email, password })
            });

            const data = await response.json();

            if (response.ok) {
                // Registration successful, but don't auto-login
                return { success: true, message: 'Registration successful' };
            } else {
                return { success: false, message: this.getErrorMessage(data) };
            }
        } catch (error) {
            console.error('Registration error:', error);
            return { success: false, message: 'Network error. Please try again.' };
        }
    }

    /**
     * Login as guest user
     * @returns {Promise<Object>} - Guest login result
     */
    async guestLogin() {
        this.user = {
            id: null,
            username: 'Guest User',
            email: '',
            is_guest: true
        };
        this.token = null; // No token for guests

        // Save guest status to localStorage
        localStorage.setItem('is_guest', 'true');
        localStorage.setItem('user_data', JSON.stringify(this.user));
        localStorage.removeItem('auth_token');

        this.showMapPage();
        return { success: true, user: this.user };
    }

    /**
     * Logout current user
     */
    async logout() {
        this.user = null;
        this.token = null;

        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
        localStorage.removeItem('is_guest');

        // Switch back to login page
        document.getElementById('map-page').classList.remove('active');
        document.getElementById('login-page').classList.add('active');

        // Clear map if it exists
        if (window.mapManager) {
            window.mapManager.clearMap();
        }

        return { success: true };
    }

    /**
     * Show map page and update UI
     */
    showMapPage() {
        document.getElementById('login-page').classList.remove('active');
        document.getElementById('map-page').classList.add('active');

        // Update user info in header via UI manager
        if (this.uiManager && this.user) {
            this.uiManager.updateUserDisplay(this.user.username);
        }

        // Initialize map if not already done
        if (window.mapManager) {
            window.mapManager.init();
        }
    }

    /**
     * Extract error message from API response
     * @param {Object} data - API response data
     * @returns {string} - Error message
     */
    getErrorMessage(data) {
        if (typeof data === 'string') return data;
        if (data.detail) return data.detail;
        if (data.message) return data.message;

        // Handle field-specific errors
        const errors = [];
        for (const [field, messages] of Object.entries(data)) {
            if (Array.isArray(messages)) {
                errors.push(`${field}: ${messages.join(', ')}`);
            } else {
                errors.push(`${field}: ${messages}`);
            }
        }

        return errors.length > 0 ? errors.join('; ') : 'Registration failed';
    }

    /**
     * Check if user is authenticated
     * @returns {boolean} - Authentication status
     */
    isAuthenticated() {
        return this.token !== null;
    }

    /**
     * Get authentication token
     * @returns {string|null} - Auth token
     */
    getToken() {
        return this.token;
    }

    /**
     * Get current user data
     * @returns {Object|null} - User data
     */
    getUser() {
        return this.user;
    }
}

// Initialize auth manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.authManager = new AuthManager();
});

                this.showMapPage();
            } else {
                alert(data.error || 'Registration failed');
            }
        } catch (error) {
            console.error('Registration failed:', error);
            alert('Registration failed. Please try again.');
        }
    }

    async login() {
        const username = document.getElementById('login-username').value;
        const password = document.getElementById('login-password').value;

        if (!username || !password) {
            alert('Please enter username/email and password');
            return;
        }

        try {
            const response = await fetch('http://localhost:8000/api/auth/login/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const data = await response.json();

            if (response.ok) {
                this.user = data.user;
                this.token = data.token;

                // Save to localStorage
                localStorage.setItem('auth_token', this.token);
                localStorage.setItem('user_data', JSON.stringify(this.user));

                this.showMapPage();
            } else {
                alert(data.error || 'Login failed');
            }
        } catch (error) {
            console.error('Login failed:', error);
            alert('Login failed. Please try again.');
        }
    }

    guestLogin() {
        // Set guest user data
        this.user = {
            id: null,
            username: 'Guest User',
            email: '',
            is_guest: true
        };
        this.token = null; // No token for guests

        // Save guest status to localStorage
        localStorage.setItem('is_guest', 'true');
        localStorage.setItem('user_data', JSON.stringify(this.user));

        this.showMapPage();
    }

    showRegisterForm() {
        document.getElementById('login-form-container').style.display = 'none';
        document.getElementById('register-form-container').style.display = 'block';
    }

    showLoginForm() {
        document.getElementById('register-form-container').style.display = 'none';
        document.getElementById('login-form-container').style.display = 'block';
    }

    showMapPage() {
        document.getElementById('login-page').classList.remove('active');
        document.getElementById('map-page').classList.add('active');

        // Update user info in header
        const userNameEl = document.getElementById('user-name');
        if (userNameEl && this.user) {
            userNameEl.textContent = this.user.username;
        }

        // Initialize map if not already done
        if (window.mapManager) {
            window.mapManager.init();
        }
    }

// Initialize auth manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.authManager = new AuthManager();

    // Set up dependencies when UI manager is ready
    if (window.uiManager) {
        window.authManager.setUIManager(window.uiManager);
        window.uiManager.setDependencies(window.authManager, window.mapManager, window.apiManager);
    }
});
