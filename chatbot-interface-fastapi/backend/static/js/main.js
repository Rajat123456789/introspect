// Main JavaScript file
document.addEventListener('DOMContentLoaded', function() {
    console.log('Introspect application initialized');
    
    // Check backend connection
    checkBackendConnection();
    
    // Initialize sidebar interactions
    initializeSidebar();
    
    // Initialize UI theme preferences
    initializeThemePreferences();
    
    // Add visual feedback to interactive elements
    addVisualFeedback();
});

// Function to check backend connection
async function checkBackendConnection() {
    try {
        const response = await fetch('/api/health');
        
        if (response.ok) {
            const data = await response.json();
            console.log('Backend health check:', data);
            
            // Update UI based on API status if needed
            if (data.api_status) {
                updateApiStatusUI(data.api_status);
            }
        } else {
            console.error('Backend health check failed:', response.status);
            showBackendError();
        }
    } catch (error) {
        console.error('Error checking backend connection:', error);
        showBackendError();
    }
}

// Function to update UI based on API status
function updateApiStatusUI(apiStatus) {
    const apiProviderSelect = document.getElementById('api-provider');
    
    if (apiProviderSelect) {
        // Get all options
        const options = apiProviderSelect.options;
        
        // Update OpenAI option
        for (let i = 0; i < options.length; i++) {
            if (options[i].value === 'openai') {
                options[i].disabled = !apiStatus.openai;
                if (!apiStatus.openai) {
                    options[i].text = 'OpenAI (API Key Missing)';
                }
            }
            
            if (options[i].value === 'gemini') {
                options[i].disabled = !apiStatus.gemini;
                if (!apiStatus.gemini) {
                    options[i].text = 'Gemini (API Key Missing)';
                }
            }
        }
        
        // If the selected option is disabled, select the first enabled option
        if (apiProviderSelect.selectedOptions[0].disabled) {
            for (let i = 0; i < options.length; i++) {
                if (!options[i].disabled) {
                    apiProviderSelect.selectedIndex = i;
                    break;
                }
            }
        }
    }
}

// Function to initialize sidebar interactions
function initializeSidebar() {
    const sidebar = document.querySelector('.sidebar');
    
    if (sidebar) {
        // Add hover effect to data sources
        const dataSources = document.querySelectorAll('.data-source');
        dataSources.forEach(source => {
            source.addEventListener('mouseenter', () => {
                source.classList.add('hover');
            });
            source.addEventListener('mouseleave', () => {
                source.classList.remove('hover');
            });
        });
        
        // Add animation for checkbox changes
        const checkboxes = sidebar.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                const source = this.closest('.data-source');
                source.classList.add('pulse');
                setTimeout(() => {
                    source.classList.remove('pulse');
                }, 500);
            });
        });
        
        // Add style for animations
        const style = document.createElement('style');
        style.textContent = `
            .data-source.hover {
                background-color: rgba(52, 152, 219, 0.05);
            }
            
            .data-source.pulse {
                animation: pulse-animation 0.5s;
            }
            
            @keyframes pulse-animation {
                0% {
                    background-color: rgba(52, 152, 219, 0.1);
                }
                50% {
                    background-color: rgba(52, 152, 219, 0.2);
                }
                100% {
                    background-color: rgba(52, 152, 219, 0.05);
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// Function to initialize theme preferences
function initializeThemePreferences() {
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
    }
}

// Function to add visual feedback to interactive elements
function addVisualFeedback() {
    // Add ripple effect to buttons
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            ripple.classList.add('ripple-effect');
            
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            
            ripple.style.width = ripple.style.height = `${size}px`;
            ripple.style.left = `${e.clientX - rect.left - size/2}px`;
            ripple.style.top = `${e.clientY - rect.top - size/2}px`;
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // Add CSS for ripple effect
    const style = document.createElement('style');
    style.textContent = `
        button {
            position: relative;
            overflow: hidden;
        }
        
        .ripple-effect {
            position: absolute;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.4);
            transform: scale(0);
            animation: ripple 0.6s linear;
            pointer-events: none;
        }
        
        @keyframes ripple {
            to {
                transform: scale(2);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
}

// Function to show backend connection error
function showBackendError() {
    // Create error notification
    const notification = document.createElement('div');
    notification.className = 'error-notification';
    notification.textContent = 'Unable to connect to backend server. Some features may not work.';
    
    // Style the notification
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.left = '50%';
    notification.style.transform = 'translateX(-50%)';
    notification.style.backgroundColor = 'var(--danger-color, #e74c3c)';
    notification.style.color = 'white';
    notification.style.padding = '10px 20px';
    notification.style.borderRadius = '4px';
    notification.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
    notification.style.zIndex = '9999';
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Animate in
    notification.style.opacity = '0';
    notification.style.transition = 'opacity 0.3s';
    setTimeout(() => {
        notification.style.opacity = '1';
    }, 10);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
} 