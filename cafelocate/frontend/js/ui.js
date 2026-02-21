/**
 * CafeLocate - UI Management Module
 * Handles all user interface interactions and event bindings
 */

class UIManager {
    constructor() {
        this.authManager = null;
        this.mapManager = null;
        this.apiManager = null;
        this.initializeEventListeners();
    }

    /**
     * Initialize all UI event listeners
     */
    initializeEventListeners() {
        this.initializePasswordToggles();
        this.initializeAuthSwitches();
        this.initializeLogoutButton();
        this.initializeFormSubmissions();
    }

    /**
     * Initialize password visibility toggle functionality
     */
    initializePasswordToggles() {
        // Login password toggle
        const loginToggle = document.getElementById('login-password-toggle');
        const loginPassword = document.getElementById('login-password');

        if (loginToggle && loginPassword) {
            loginToggle.addEventListener('click', () => {
                this.togglePasswordVisibility(loginPassword, loginToggle);
            });
        }

        // Register password toggle
        const registerToggle = document.getElementById('register-password-toggle');
        const registerPassword = document.getElementById('register-password');

        if (registerToggle && registerPassword) {
            registerToggle.addEventListener('click', () => {
                this.togglePasswordVisibility(registerPassword, registerToggle);
            });
        }
    }

    /**
     * Toggle password field visibility
     * @param {HTMLInputElement} passwordField - The password input field
     * @param {HTMLButtonElement} toggleButton - The toggle button
     */
    togglePasswordVisibility(passwordField, toggleButton) {
        const isVisible = passwordField.type === 'text';

        if (isVisible) {
            passwordField.type = 'password';
            toggleButton.textContent = '👁️';
            toggleButton.title = 'Show password';
        } else {
            passwordField.type = 'text';
            toggleButton.textContent = '🙈';
            toggleButton.title = 'Hide password';
        }

        // Add visual feedback
        toggleButton.style.transform = 'scale(1.1)';
        setTimeout(() => {
            toggleButton.style.transform = 'scale(1)';
        }, 150);
    }

    /**
     * Initialize authentication form switching
     */
    initializeAuthSwitches() {
        const showRegisterLink = document.getElementById('show-register');
        const showLoginLink = document.getElementById('show-login');
        const loginForm = document.getElementById('login-form-container');
        const registerForm = document.getElementById('register-form-container');

        if (showRegisterLink && loginForm && registerForm) {
            showRegisterLink.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchAuthForm(loginForm, registerForm);
            });
        }

        if (showLoginLink && loginForm && registerForm) {
            showLoginLink.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchAuthForm(registerForm, loginForm);
            });
        }
    }

    /**
     * Switch between login and register forms with animation
     * @param {HTMLElement} hideForm - Form to hide
     * @param {HTMLElement} showForm - Form to show
     */
    switchAuthForm(hideForm, showForm) {
        hideForm.style.opacity = '0';
        hideForm.style.transform = 'translateX(-20px)';

        setTimeout(() => {
            hideForm.style.display = 'none';
            showForm.style.display = 'block';
            showForm.style.opacity = '0';
            showForm.style.transform = 'translateX(20px)';

            // Force reflow
            showForm.offsetHeight;

            showForm.style.opacity = '1';
            showForm.style.transform = 'translateX(0)';
        }, 300);
    }

    /**
     * Initialize logout button functionality
     */
    initializeLogoutButton() {
        const logoutBtn = document.getElementById('logout-btn');

        if (logoutBtn && this.authManager) {
            logoutBtn.addEventListener('click', () => {
                this.handleLogout();
            });
        }
    }

    /**
     * Handle logout process
     */
    async handleLogout() {
        if (!this.authManager) {
            console.error('AuthManager not initialized');
            return;
        }

        try {
            // Show loading state
            const logoutBtn = document.getElementById('logout-btn');
            const originalText = logoutBtn.textContent;
            logoutBtn.textContent = 'Logging out...';
            logoutBtn.disabled = true;

            await this.authManager.logout();

            // Reset button
            logoutBtn.textContent = originalText;
            logoutBtn.disabled = false;

        } catch (error) {
            console.error('Logout error:', error);
            // Reset button on error
            const logoutBtn = document.getElementById('logout-btn');
            logoutBtn.textContent = 'Logout';
            logoutBtn.disabled = false;

            this.showNotification('Logout failed. Please try again.', 'error');
        }
    }

    /**
     * Initialize form submission handlers
     */
    initializeFormSubmissions() {
        const loginForm = document.getElementById('login-form');
        const registerForm = document.getElementById('register-form');

        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleLogin();
            });
        }

        if (registerForm) {
            registerForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleRegister();
            });
        }

        // Guest login
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
        if (!this.authManager) {
            console.error('AuthManager not initialized');
            return;
        }

        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value;

        if (!username || !password) {
            this.showNotification('Please fill in all fields', 'warning');
            return;
        }

        try {
            this.showLoadingState('login-form', true);

            const result = await this.authManager.login(username, password);

            if (result.success) {
                this.showNotification('Login successful!', 'success');
                // Page switching will be handled by AuthManager
            } else {
                this.showNotification(result.message || 'Login failed', 'error');
            }

        } catch (error) {
            console.error('Login error:', error);
            this.showNotification('Login failed. Please try again.', 'error');
        } finally {
            this.showLoadingState('login-form', false);
        }
    }

    /**
     * Handle register form submission
     */
    async handleRegister() {
        if (!this.authManager) {
            console.error('AuthManager not initialized');
            return;
        }

        const username = document.getElementById('register-username').value.trim();
        const email = document.getElementById('register-email').value.trim();
        const password = document.getElementById('register-password').value;

        if (!username || !email || !password) {
            this.showNotification('Please fill in all fields', 'warning');
            return;
        }

        if (password.length < 6) {
            this.showNotification('Password must be at least 6 characters long', 'warning');
            return;
        }

        try {
            this.showLoadingState('register-form', true);

            const result = await this.authManager.register(username, email, password);

            if (result.success) {
                this.showNotification('Registration successful! Please login.', 'success');
                // Switch to login form
                document.getElementById('show-login').click();
            } else {
                this.showNotification(result.message || 'Registration failed', 'error');
            }

        } catch (error) {
            console.error('Registration error:', error);
            this.showNotification('Registration failed. Please try again.', 'error');
        } finally {
            this.showLoadingState('register-form', false);
        }
    }

    /**
     * Handle guest login
     */
    async handleGuestLogin() {
        if (!this.authManager) {
            console.error('AuthManager not initialized');
            return;
        }

        try {
            const result = await this.authManager.guestLogin();

            if (result.success) {
                this.showNotification('Welcome! You are now browsing as a guest.', 'info');
            } else {
                this.showNotification('Guest login failed', 'error');
            }

        } catch (error) {
            console.error('Guest login error:', error);
            this.showNotification('Guest login failed. Please try again.', 'error');
        }
    }

    /**
     * Show loading state for forms
     * @param {string} formId - The form ID
     * @param {boolean} loading - Whether to show loading state
     */
    showLoadingState(formId, loading) {
        const form = document.getElementById(formId);
        const submitBtn = form.querySelector('button[type="submit"]');

        if (loading) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
            submitBtn.style.opacity = '0.7';
        } else {
            submitBtn.disabled = false;
            submitBtn.textContent = formId === 'login-form' ? 'Login' : 'Register';
            submitBtn.style.opacity = '1';
        }
    }

    /**
     * Show notification to user
     * @param {string} message - The notification message
     * @param {string} type - The notification type (success, error, warning, info)
     */
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-message">${message}</span>
            </div>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Add CSS styles dynamically if not already present
        if (!document.getElementById('notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'notification-styles';
            styles.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 10000;
                    min-width: 300px;
                    max-width: 500px;
                    padding: 16px;
                    border-radius: 12px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255,255,255,0.2);
                    animation: slideInRight 0.3s ease-out;
                    font-family: 'Inter', sans-serif;
                }

                .notification-success {
                    background: linear-gradient(135deg, #00cec9, #00b894);
                    color: white;
                }

                .notification-error {
                    background: linear-gradient(135deg, #e17055, #e84393);
                    color: white;
                }

                .notification-warning {
                    background: linear-gradient(135deg, #fdcb6e, #e17055);
                    color: white;
                }

                .notification-info {
                    background: linear-gradient(135deg, #74b9ff, #0984e3);
                    color: white;
                }

                .notification-content {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }

                .notification-icon {
                    font-size: 1.2rem;
                    flex-shrink: 0;
                }

                .notification-message {
                    flex: 1;
                    font-weight: 500;
                }

                @keyframes slideInRight {
                    from {
                        opacity: 0;
                        transform: translateX(100%);
                    }
                    to {
                        opacity: 1;
                        transform: translateX(0);
                    }
                }

                @keyframes slideOutRight {
                    from {
                        opacity: 1;
                        transform: translateX(0);
                    }
                    to {
                        opacity: 0;
                        transform: translateX(100%);
                    }
                }
            `;
            document.head.appendChild(styles);
        }

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);

        // Click to dismiss
        notification.addEventListener('click', () => {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        });
    }

    /**
     * Get notification icon based on type
     * @param {string} type - The notification type
     * @returns {string} - The icon emoji
     */
    getNotificationIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }

    /**
     * Set dependencies (called by main app)
     * @param {AuthManager} authManager - The authentication manager
     * @param {MapManager} mapManager - The map manager
     * @param {APIManager} apiManager - The API manager
     */
    setDependencies(authManager, mapManager, apiManager) {
        this.authManager = authManager;
        this.mapManager = mapManager;
        this.apiManager = apiManager;

        // Re-initialize logout button now that authManager is available
        this.initializeLogoutButton();
    }

    /**
     * Update user display name
     * @param {string} username - The username to display
     */
    updateUserDisplay(username) {
        const userNameElement = document.getElementById('user-name');
        if (userNameElement) {
            userNameElement.textContent = username || 'Guest User';
        }
    }
}

// Initialize UI Manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.uiManager = new UIManager();

    // Set up dependencies when all managers are ready
    const checkDependencies = () => {
        if (window.authManager && window.mapManager && window.apiManager) {
            window.authManager.setUIManager(window.uiManager);
            window.uiManager.setDependencies(window.authManager, window.mapManager, window.apiManager);
        } else {
            // Check again in a short delay
            setTimeout(checkDependencies, 100);
        }
    };

    checkDependencies();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIManager;
}