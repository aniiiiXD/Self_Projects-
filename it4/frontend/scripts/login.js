const API_URL = 'http://localhost:3003/api/user';

        function showMessage(message, isError = false) {
            const messageElement = document.getElementById('login-message');
            messageElement.textContent = message;
            messageElement.className = `mt-4 text-sm ${isError ? 'text-red-600' : 'text-green-600'}`;
        }

        // Handle login form submission
        document.getElementById('login-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('login-email').value;
            const password = document.getElementById('login-password').value;
            const rememberMe = document.getElementById('remember-me').checked;

            try {
                const response = await fetch(`${API_URL}/signin`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ email, password }),
                });
                const data = await response.json();
                
                if (response.ok) {
                    localStorage.setItem('authToken', data.token);
                    console.log(data.token)
                    showMessage('Login successful! Redirecting...');
                    
                    // If remember me is checked, store the token with an expiration
                    if (rememberMe) {
                        const expirationDate = new Date();
                        expirationDate.setDate(expirationDate.getDate() + 30);
                        localStorage.setItem('tokenExpiration', expirationDate.toISOString());
                    }
                    
                    // Redirect to dashboard or home page
                    setTimeout(() => {
                        window.location.href = './profile.html'; // Update this to your dashboard URL
                    }, 1500);
                } else {
                    showMessage(data.message || 'Login failed. Please check your credentials.', true);
                }
            } catch (error) {
                console.error('Login error: ' , error);
                showMessage('Error during login. Please try again.', true);
            }
        });

        // Check token expiration on page load
        document.addEventListener('DOMContentLoaded', () => {
            const token = localStorage.getItem('authToken');
            const expiration = localStorage.getItem('tokenExpiration');
            
            if (token && expiration) {
                if (new Date(expiration) < new Date()) {
                    // Token has expired
                    localStorage.removeItem('authToken');
                    localStorage.removeItem('tokenExpiration');
                }
            }
        });