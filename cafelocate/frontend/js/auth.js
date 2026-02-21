// CafeLocate - Authentication Module
// Handles user registration and login with username/email and password

class AuthManager {
    constructor() {
        this.user = null;
        this.token = null;
    }

    /**
     * Check for existing session on page load
     */
    checkExistingSession() {
        const savedToken = localStorage.getItem('auth_token');
        const savedUser = localStorage.getItem('user_data');
        const isGuest = localStorage.getItem('is_guest') === 'true';

        if (isGuest && savedUser) {
            try {
                this.user = JSON.parse(savedUser);
                this.token = null;
                this.showMapPage();
            } catch (e) {
                this.clearStorage();
            }
        } else if (savedToken && savedUser) {
            try {
                this.token = savedToken;
                this.user = JSON.parse(savedUser);
                this.showMapPage();
            } catch (e) {
                this.clearStorage();
            }
        }
    }

    /**
     * Login user with credentials
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

                localStorage.setItem('auth_token', this.token);
                localStorage.setItem('user_data', JSON.stringify(this.user));
                localStorage.removeItem('is_guest');

                this.showMapPage();
                return { success: true, user: this.user };
            } else {
                const errorMsg = data.error || data.detail || data.message || 'Login failed';
                return { success: false, message: errorMsg };
            }
        } catch (error) {
            console.error('Login error:', error);
            return { success: false, message: 'Network error. Is the backend server running on port 8000?' };
        }
    }

    /**
     * Register new user
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
                return { success: true, message: 'Registration successful! Please login.' };
            } else {
                return { success: false, message: this._getErrorMessage(data) };
            }
        } catch (error) {
            console.error('Registration error:', error);
            return { success: false, message: 'Network error. Is the backend server running on port 8000?' };
        }
    }

    /**
     * Login as guest user
     */
    async guestLogin() {
        this.user = {
            id: null,
            username: 'Guest User',
            email: '',
            is_guest: true
        };
        this.token = null;

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
        this.clearStorage();

        document.getElementById('map-page').classList.remove('active');
        document.getElementById('login-page').classList.add('active');

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

        const userNameEl = document.getElementById('user-name');
        if (userNameEl && this.user) {
            userNameEl.textContent = this.user.username;
        }

        // Initialize map after page is visible
        setTimeout(() => {
            if (window.mapManager) {
                window.mapManager.init();
            }
        }, 100);
    }

    clearStorage() {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
        localStorage.removeItem('is_guest');
    }

    _getErrorMessage(data) {
        if (typeof data === 'string') return data;
        if (data.error) return data.error;
        if (data.detail) return data.detail;
        if (data.message) return data.message;

        const errors = [];
        for (const [field, messages] of Object.entries(data)) {
            if (Array.isArray(messages)) {
                errors.push(`${field}: ${messages.join(', ')}`);
            } else if (typeof messages === 'string') {
                errors.push(`${field}: ${messages}`);
            }
        }
        return errors.length > 0 ? errors.join('; ') : 'Registration failed';
    }

    isAuthenticated() {
        return this.token !== null;
    }

    getToken() {
        return this.token;
    }

    getUser() {
        return this.user;
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.authManager = new AuthManager();
    window.authManager.checkExistingSession();
});
