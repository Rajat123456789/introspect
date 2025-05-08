// Chat functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize chat interface
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const useRawData = document.getElementById('use-raw-data');
    const apiProviderSelect = document.getElementById('api-provider');
    
    // Chat boxes
    const chatbox1Messages = document.querySelector('#chatbox1 .chatbox-messages');
    const chatbox2Messages = document.querySelector('#chatbox2 .chatbox-messages');
    const chatbox3Messages = document.querySelector('#chatbox3 .chatbox-messages');
    
    // Add initial messages
    addBotMessage(chatbox1Messages, "Hello! I'm the Base Model without any special context or prompting. Ask me anything, and I'll respond with my default capabilities.");
    addBotMessage(chatbox2Messages, "Hello! I'm Health LLM, a model enhanced with healthcare domain knowledge. I specialize in medical and health-related information, providing reliable guidance about health conditions, treatments, and wellness practices.");
    addBotMessage(chatbox3Messages, "Hello! I'm the Introspective Assistant. I analyze your digital and health data to help you reflect on your behaviors and patterns. I can provide insights about your YouTube viewing, Spotify listening, and health metrics to encourage self-awareness and personal growth through thoughtful questions.");
    
    // Handle form submission
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Disable input and button while sending
        chatInput.disabled = true;
        sendButton.disabled = true;
        
        // Get selected API provider
        const apiProvider = apiProviderSelect.value;
        
        // Add user message to all chat boxes
        addUserMessage(chatbox1Messages, message);
        addUserMessage(chatbox2Messages, message);
        addUserMessage(chatbox3Messages, message);
        
        // Clear input
        chatInput.value = '';
        
        try {
            // Send message to all three model types
            await Promise.all([
                sendMessageToModel(message, 'base', apiProvider, useRawData.checked, chatbox1Messages),
                sendMessageToModel(message, 'health', apiProvider, useRawData.checked, chatbox2Messages),
                sendMessageToModel(message, 'introspect', apiProvider, useRawData.checked, chatbox3Messages)
            ]);
        } catch (error) {
            console.error('Error sending messages:', error);
            
            // Add error message to all chat boxes
            const errorMessage = 'Sorry, there was an error processing your request. Please try again.';
            addBotMessage(chatbox1Messages, errorMessage);
            addBotMessage(chatbox2Messages, errorMessage);
            addBotMessage(chatbox3Messages, errorMessage);
        } finally {
            // Re-enable input and button
            chatInput.disabled = false;
            sendButton.disabled = false;
            chatInput.focus();
        }
    });
    
    // Helper function to add user message to chat
    function addUserMessage(chatbox, text) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user';
        messageElement.textContent = text;
        chatbox.appendChild(messageElement);
        
        // Scroll to bottom
        chatbox.scrollTop = chatbox.scrollHeight;
    }
    
    // Helper function to add bot message to chat
    function addBotMessage(chatbox, text) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot';
        messageElement.textContent = text;
        chatbox.appendChild(messageElement);
        
        // Scroll to bottom
        chatbox.scrollTop = chatbox.scrollHeight;
    }
    
    // Function to send message to a specific model
    async function sendMessageToModel(message, modelType, apiProvider, useRawData, chatbox) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    model_type: modelType,
                    use_raw_data: useRawData,
                    api_provider: apiProvider
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Add bot response to chat
            addBotMessage(chatbox, data.message);
        } catch (error) {
            console.error(`Error with ${modelType} model:`, error);
            throw error;
        }
    }
}); 