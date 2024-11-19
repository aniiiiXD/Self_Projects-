// API URL Constants
const API_URL = 'http://localhost:3003/api/user';

// Utility Functions
function showLoading() {
    document.getElementById('loadingOverlay')?.classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay')?.classList.add('hidden');
}

function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    if (!notification) return;
    
    notification.textContent = message;
    notification.style.backgroundColor = type === 'error' ? '#dc2626' : '#16a34a';
    notification.classList.remove('translate-x-full');
    
    setTimeout(() => {
        notification.classList.add('translate-x-full');
    }, 3000);
}

// Form Management
function populateForm(profileData) {
    const form = document.getElementById('profileForm');
    if (!form) return;

    // Map data to form fields
    const fields = ['fullName', 'phoneNumber', 'CATscore', 'gradSchool'];
    fields.forEach(field => {
        const input = form.elements[field];
        if (input && profileData[field] !== undefined) {
            input.value = profileData[field];
        }
    });
}

function getFormData() {
    const form = document.getElementById('profileForm');
    if (!form) return null;
    
    const formData = new FormData(form);
    return Object.fromEntries(formData.entries());
}

// API Functions
async function fetchProfile() {
    try {
        showLoading();
        const token = localStorage.getItem('authToken');
        
        if (!token) {
            window.location.href = './login.html';
            return;
        }

        const response = await fetch(`${API_URL}/profile`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        }); 

        if (response.status === 401) {
            localStorage.removeItem('authToken');
            window.location.href = './login.html';
            return;
        }

        const data = await response.json();
        if (data.success && data.profile) {
            populateForm(data.profile);
        } else {
            showNotification(data.message || 'Failed to load profile', 'error');
        }
    } catch (error) {
        console.error('Profile fetch error:', error);
        showNotification('Failed to load profile data', 'error');
    } finally {
        hideLoading();
    }
}

async function updateProfile(profileData) {
    try {
        showLoading();
        const token = localStorage.getItem('authToken');
        
        if (!token) {
            window.location.href = './login.html';
            return false;
        }

        const decodedToken = parseJwt(token);
        const studentId = decodedToken.id; // Extract student ID from token

        const response = await fetch(`${API_URL}/profile`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                studentId: studentId, // Use the extracted studentId
                fullName: profileData.fullName?.trim(),
                phoneNumber: profileData.phoneNumber?.trim(),
                CATscore: profileData.CATscore ? Number(profileData.CATscore) : null,
                gradSchool: profileData.gradSchool?.trim()
            })
        });

        const data = await response.json();
        
        if (data.success) {
            showNotification('Profile updated successfully');
            return true;
        } else {
            showNotification(data.message || 'Update failed', 'error');
            return false;
        }
    } catch (error) {
        console.error('Profile update error:', error);
        showNotification('Failed to update profile', 'error');
        return false;
    } finally {
        hideLoading();
    }
}

// Parse JWT token to extract studentId
function parseJwt(token) {
    try {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));

        return JSON.parse(jsonPayload);
    } catch (e) {
        console.error('Invalid token', e);
        return null;
    }
}

// Event Handlers
function setupFormControls() {
    const form = document.getElementById('profileForm');
    const editBtn = document.getElementById('editProfileBtn');
    const saveBtn = document.getElementById('saveProfileBtn');
    
    if (!form || !editBtn || !saveBtn) return;

    // Edit button handler
    editBtn.addEventListener('click', () => {
        Array.from(form.elements).forEach(input => {
            if (input.tagName !== 'BUTTON') {
                input.disabled = false;
            }
        });
        editBtn.classList.add('hidden');
        saveBtn.classList.remove('hidden');
    });

    // Save button handler
    saveBtn.addEventListener('click', async () => {
        const formData = getFormData();
        if (formData) {
            const success = await updateProfile(formData);
            if (success) {
                Array.from(form.elements).forEach(input => {
                    if (input.tagName !== 'BUTTON') {
                        input.disabled = true;
                    }
                });
                saveBtn.classList.add('hidden');
                editBtn.classList.remove('hidden');
            }
        }
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupFormControls();
    fetchProfile();
});

// Logout handler
window.logout = () => {
    localStorage.removeItem('authToken');
    window.location.href = './login.html';
};
