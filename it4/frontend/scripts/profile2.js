// Constants and Configuration
const API_URL = 'http://localhost:3003/api/user';

// Utility Functions
function showLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.classList.remove('hidden');
    }
}

function hideLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.classList.add('hidden');
    }
}

function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 px-6 py-3 rounded-md text-white transform transition-transform duration-300 translate-x-full`;
    notification.textContent = message;
    notification.style.backgroundColor = type === 'error' ? '#dc2626' : '#16a34a';
    
    document.body.appendChild(notification);
    
    // Show notification
    setTimeout(() => {
        notification.classList.remove('translate-x-full');
    }, 100);
    
    // Remove notification
    setTimeout(() => {
        notification.classList.add('translate-x-full');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Profile Management Functions
async function fetchProfile() {
    try {
        showLoading();
        const token = localStorage.getItem('authToken');
        console.log('Fetch Profile - Token:', token ? 'Present' : 'Missing');
        if (!token) {
            console.log('No token found, redirecting to login page.');
            window.location.href = './login.html';
            return;
        }

        const response = await fetch(`${API_URL}/profile`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        console.log('Fetch Profile - Response status:', response.status);
        if (!response.ok) {
            if (response.status === 401) {
                localStorage.removeItem('authToken');
                window.location.href = './login.html';
                return;
            }
            throw new Error('Failed to fetch profile');
        }

        const data = await response.json();
        console.log('Fetch Profile - Data:', data);
        if (data.success) {
            populateForm(data.profile);
        } else {
            showNotification(data.message, 'error');
        }
    } catch (error) {
        showNotification('Failed to load profile data', 'error');
        console.error('Fetch Profile - Error:', error);
    } finally {
        hideLoading();
    }
}

// Update Profile Function with more logging
async function updateProfile(profileData) {
    try {
        const token = localStorage.getItem('authToken');
        console.log('Update Profile - Token:', token ? 'Present' : 'Missing');

        if (!token) {
            console.log('No token found, redirecting to login page.');
            window.location.href = './login.html';
            return false;
        }

        console.log('Update Profile - Profile Data:', profileData);

        if (!profileData.studentId?.trim() || !profileData.fullName?.trim()) {
            console.log('Validation failed - Missing required fields');
            showNotification('Student ID and Full Name are required', 'error');
            return false;
        }

        const requestBody = {
            studentId: profileData.studentId.trim(),
            fullName: profileData.fullName.trim(),
            phoneNumber: profileData.phoneNumber?.trim() || '',
            CATscore: profileData.CATscore ? Number(profileData.CATscore) : null,
            gradSchool: profileData.gradSchool?.trim() || ''
        };

        console.log('Update Profile - Request body:', requestBody);

        const response = await fetch(`${API_URL}/profile`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        console.log('Update Profile - Response status:', response.status);
        const data = await response.json();
        console.log('Update Profile - Response data:', data);

        if (!data.success) {
            console.log('Update failed:', data.message);
            showNotification(data.message || 'Failed to update profile', 'error');
            return false;
        }

        showNotification('Profile updated successfully');
        return true;

    } catch (error) {
        console.error('Update Profile - Error details:', error);
        showNotification('Failed to update profile', 'error');
        return false;
    }
}

// In the form submit listener, add logs
function setupEventListeners() {
    const form = document.getElementById('profileForm');
    if (!form) {
        console.log('Form element not found');
        return;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log('Form submitted');
        showLoading();

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        console.log('Form data collected:', data);

        if (validateForm(data)) {
            console.log('Form validation passed');
            const updateSuccess = await updateProfile(data);
            console.log('Profile update success:', updateSuccess);
        } else {
            console.log('Form validation failed');
        }

        hideLoading();
    });

    // Add event listener for the Edit button
    const editButton = document.getElementById('editProfileBtn');
    const saveButton = document.getElementById('saveProfileBtn');
    
    if (editButton && saveButton) {
        editButton.addEventListener('click', () => {
            console.log('Edit button clicked');
            // Enable all form inputs
            Array.from(form.elements).forEach(input => {
                if (input.tagName !== 'BUTTON') {
                    console.log(`Enabling input: ${input.name}`);
                    input.disabled = false;
                }
            });
            editButton.classList.add('hidden');
            saveButton.classList.remove('hidden');
        });

        saveButton.addEventListener('click', (e) => {
            console.log('Save button clicked');
            form.dispatchEvent(new Event('submit')); // Trigger the form submit event
        });
    } else {
        console.log('Edit or Save button not found');
    }
}

// Logout Function
function logout() {
    localStorage.removeItem('authToken');
    window.location.href = './login.html';
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    fetchProfile();
});
