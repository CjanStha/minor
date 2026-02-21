// CafeLocate - Authentication Module
// Handles user registration and login with username/email and password

class AuthManager {
    constructor() {
        this.user = null;
        this.token = null;
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

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Registration form
        const registerForm = document.getElementById('register-form');
        if (registerForm) {
            registerForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.register();
            });
        }

        // Login form
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.login();
            });
        }

        // Switch between login and register
        const showRegisterLink = document.getElementById('show-register');
        const showLoginLink = document.getElementById('show-login');

        if (showRegisterLink) {
            showRegisterLink.addEventListener('click', (e) => {
                e.preventDefault();
                this.showRegisterForm();
            });
        }

        if (showLoginLink) {
            showLoginLink.addEventListener('click', (e) => {
                e.preventDefault();
                this.showLoginForm();
            });
        }

        // Guest login button
        const guestLoginBtn = document.getElementById('guest-login-btn');
        if (guestLoginBtn) {
            guestLoginBtn.addEventListener('click', () => this.guestLogin());
        }
    }

    async register() {
        const username = document.getElementById('register-username').value;
        const email = document.getElementById('register-email').value;
        const password = document.getElementById('register-password').value;

        if (!username || !email || !password) {
            alert('Please fill in all fields');
            return;
        }

        try {
            const response = await fetch('http://localhost:8000/api/auth/register/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email, password })
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

    logout() {
        this.user = null;
        this.token = null;

        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
        localStorage.removeItem('is_guest');

        document.getElementById('map-page').classList.remove('active');
        document.getElementById('login-page').classList.add('active');

        // Clear map if it exists
        if (window.mapManager) {
            window.mapManager.clearMap();
        }
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

// Initialize auth manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.authManager = new AuthManager();
});
