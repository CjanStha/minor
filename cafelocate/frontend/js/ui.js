/**
 * CafeLocate - UI Management Module
 * Handles all user interface interactions and event bindings
 */

class UIManager {
    constructor() {
        this.initializeEventListeners();
    }

    /**
     * Initialize all UI event listeners
     */
    initializeEventListeners() {
        this.initializePasswordToggles();
        this.initializeAuthSwitches();
        this.initializeFormSubmissions();
        this.initializeLogoutButton();
    }

    /**
     * Initialize password visibility toggle functionality
     */
    initializePasswordToggles() {
        const loginToggle = document.getElementById('login-password-toggle');
        const loginPassword = document.getElementById('login-password');
        if (loginToggle && loginPassword) {
            loginToggle.addEventListener('click', () => {
                this.togglePasswordVisibility(loginPassword, loginToggle);
            });
        }

        const registerToggle = document.getElementById('register-password-toggle');
        const registerPassword = document.getElementById('register-password');
        if (registerToggle && registerPassword) {
            registerToggle.addEventListener('click', () => {
                this.togglePasswordVisibility(registerPassword, registerToggle);
            });
        }
    }

    togglePasswordVisibility(passwordField, toggleButton) {
        const isVisible = passwordField.type === 'text';
        passwordField.type = isVisible ? 'password' : 'text';
        toggleButton.textContent = isVisible ? '👁️' : '🙈';
        toggleButton.title = isVisible ? 'Show password' : 'Hide password';
    }

    /**
     * Initialize authentication form switching
     */
    initializeAuthSwitches() {
        const showRegisterLink = document.getElementById('show-register');
        const showLoginLink = document.getElementById('show-login');
        const loginForm = document.getElementById('login-form-container');
        const registerForm = document.getElementById('register-form-container');

        if (showRegisterLink) {
            showRegisterLink.addEventListener('click', (e) => {
                e.preventDefault();
                if (loginForm) loginForm.style.display = 'none';
                if (registerForm) registerForm.style.display = 'block';
            });
        }

        if (showLoginLink) {
            showLoginLink.addEventListener('click', (e) => {
                e.preventDefault();
                if (registerForm) registerForm.style.display = 'none';
                if (loginForm) loginForm.style.display = 'block';
            });
        }
    }

    /**
     * Initialize logout button
     */
    initializeLogoutButton() {
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', async () => {
                logoutBtn.textContent = 'Logging out...';
                logoutBtn.disabled = true;
                if (window.authManager) {
                    await window.authManager.logout();
                }
                logoutBtn.textContent = 'Logout';
                logoutBtn.disabled = false;
            });
        }
    }

    /**
     * Initialize form submission handlers
     */
    initializeFormSubmissions() {
        // Login form
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleLogin();
            });
        }

        // Register form
        const registerForm = document.getElementById('register-form');
        if (registerForm) {
            registerForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleRegister();
            });
        }

        // Guest login button
        const guestBtn = document.getElementById('guest-login-btn');
        if (guestBtn) {
            guestBtn.addEventListener('click', () => {
                this.handleGuestLogin();
            });
        }
    }

    /**
     * Handle login form submission
     */
    async handleLogin() {
        if (!window.authManager) {
            this.showNotification('Auth system not ready. Please refresh.', 'error');
            return;
        }

        const usernameEl = document.getElementById('login-username');
        const passwordEl = document.getElementById('login-password');
        const username = usernameEl ? usernameEl.value.trim() : '';
        const password = passwordEl ? passwordEl.value : '';

        if (!username || !password) {
            this.showNotification('Please fill in all fields', 'warning');
            return;
        }

        const submitBtn = document.querySelector('#login-form button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Logging in...';
        }

        try {
            const result = await window.authManager.login(username, password);
            if (result.success) {
                this.showNotification('Login successful!', 'success');
            } else {
                this.showNotification(result.message || 'Login failed', 'error');
            }
        } catch (error) {
            this.showNotification('Login failed. Please try again.', 'error');
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Login';
            }
        }
    }

    /**
     * Handle register form submission
     */
    async handleRegister() {
        if (!window.authManager) {
            this.showNotification('Auth system not ready. Please refresh.', 'error');
            return;
        }

        const usernameEl = document.getElementById('register-username');
        const emailEl = document.getElementById('register-email');
        const passwordEl = document.getElementById('register-password');

        const username = usernameEl ? usernameEl.value.trim() : '';
        const email = emailEl ? emailEl.value.trim() : '';
        const password = passwordEl ? passwordEl.value : '';

        if (!username || !email || !password) {
            this.showNotification('Please fill in all fields', 'warning');
            return;
        }

        if (password.length < 6) {
            this.showNotification('Password must be at least 6 characters', 'warning');
            return;
        }

        const submitBtn = document.querySelector('#register-form button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Registering...';
        }

        try {
            const result = await window.authManager.register(username, email, password);
            if (result.success) {
                this.showNotification(result.message || 'Registration successful! Please login.', 'success');
                // Switch to login form
                document.getElementById('register-form-container').style.display = 'none';
                document.getElementById('login-form-container').style.display = 'block';
                // Pre-fill username
                const loginUsername = document.getElementById('login-username');
                if (loginUsername) loginUsername.value = username;
            } else {
                this.showNotification(result.message || 'Registration failed', 'error');
            }
        } catch (error) {
            this.showNotification('Registration failed. Please try again.', 'error');
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Register';
            }
        }
    }

    /**
     * Handle guest login
     */
    async handleGuestLogin() {
        if (!window.authManager) {
            this.showNotification('Auth system not ready. Please refresh.', 'error');
            return;
        }

        const guestBtn = document.getElementById('guest-login-btn');
        if (guestBtn) {
            guestBtn.disabled = true;
            guestBtn.textContent = 'Loading...';
        }

        try {
            const result = await window.authManager.guestLogin();
            if (result.success) {
                this.showNotification('Welcome! Browsing as guest.', 'info');
            }
        } catch (error) {
            this.showNotification('Guest login failed. Please try again.', 'error');
            if (guestBtn) {
                guestBtn.disabled = false;
                guestBtn.textContent = 'Continue as Guest';
            }
        }
    }

    /**
     * Show notification toast
     */
    showNotification(message, type = 'info') {
        // Remove existing notifications
        document.querySelectorAll('.cafelocate-notification').forEach(n => n.remove());

        const notification = document.createElement('div');
        notification.className = `cafelocate-notification notification-${type}`;

        const colors = {
            success: 'linear-gradient(135deg, #00cec9, #00b894)',
            error: 'linear-gradient(135deg, #e17055, #d63031)',
            warning: 'linear-gradient(135deg, #fdcb6e, #e17055)',
            info: 'linear-gradient(135deg, #74b9ff, #0984e3)'
        };

        const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };

        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            zIndex: '99999',
            minWidth: '280px',
            maxWidth: '450px',
            padding: '14px 18px',
            borderRadius: '12px',
            background: colors[type] || colors.info,
            color: 'white',
            boxShadow: '0 8px 32px rgba(0,0,0,0.25)',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            fontFamily: "'Inter', 'Segoe UI', sans-serif",
            fontSize: '0.95rem',
            fontWeight: '500',
            cursor: 'pointer',
            animation: 'none',
            transition: 'opacity 0.3s ease'
        });

        notification.innerHTML = `<span style="font-size:1.1rem">${icons[type] || icons.info}</span><span>${message}</span>`;

        document.body.appendChild(notification);

        const remove = () => {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        };

        notification.addEventListener('click', remove);
        setTimeout(remove, 5000);
    }

    updateUserDisplay(username) {
        const el = document.getElementById('user-name');
        if (el) el.textContent = username || 'Guest User';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.uiManager = new UIManager();
});
