import React, { useState, useEffect, useRef } from 'react';
import Layout from './components/Layout'
import ChatBox from './components/ChatBox';
import './App.css'
import API_ENDPOINTS from './config';
import { IoSend } from 'react-icons/io5';
import { HiOutlineAdjustments } from 'react-icons/hi';

// Create a custom interface for the ref we'll use to control chatboxes
interface ChatBoxRef {
  sendMessage: (text: string) => void;
}

// Define API provider type
type ApiProvider = 'openai' | 'gemini';

function App() {
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';
  const [commonInput, setCommonInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useRaw, setUseRaw] = useState(false);
  const [apiProvider, setApiProvider] = useState<ApiProvider>('openai');
  const [apiStatus, setApiStatus] = useState<{
    openai: boolean;
    gemini: boolean;
  }>({
    openai: false,
    gemini: false
  });
  
  // Create refs for each chatbox
  const chatBox1Ref = useRef<ChatBoxRef>(null);
  const chatBox2Ref = useRef<ChatBoxRef>(null);
  const chatBox3Ref = useRef<ChatBoxRef>(null);

  useEffect(() => {
    // Check if backend is connected on component mount
    const checkBackendConnection = async () => {
      try {
        console.log('Checking backend connection...');
        const response = await fetch(API_ENDPOINTS.HEALTH, { 
          method: 'GET',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          // Add more resilient fetch options
          mode: 'cors',
          cache: 'no-cache',
          credentials: 'same-origin',
          // Set a timeout to avoid long waits
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('Backend health check response:', data);
          setIsBackendConnected(true);
          
          // Set API status if available
          if (data.api_status) {
            setApiStatus(data.api_status);
          }
        } else {
          console.error('Backend health check failed:', response.status);
          setIsBackendConnected(false);
        }
      } catch (error) {
        console.error('Error checking backend connection:', error);
        setIsBackendConnected(false);
      }
    };
    
    checkBackendConnection();
  }, []);

  const handleSendToAll = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!commonInput.trim()) return;
    
    setIsLoading(true);
    
    // We'll include the useRaw flag in the message that gets sent
    const messageWithFlag = useRaw ? `[RAW_DATA] ${commonInput}` : commonInput;
    
    // Send message to all three chatbots
    if (chatBox1Ref.current) chatBox1Ref.current.sendMessage(messageWithFlag);
    if (chatBox2Ref.current) chatBox2Ref.current.sendMessage(messageWithFlag);
    if (chatBox3Ref.current) chatBox3Ref.current.sendMessage(messageWithFlag);
    
    // Clear input after sending
    setCommonInput('');
    
    // Set a timeout to ensure loading state is visible briefly
    setTimeout(() => {
      setIsLoading(false);
    }, 300);
  };

  // Handle API provider change
  const handleApiProviderChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setApiProvider(e.target.value as ApiProvider);
  };

  // Create three instances of the ChatBox component with different initial messages
  const chatBox1 = (
    <ChatBox 
      backendUrl={backendUrl}
      passLoadingState={() => {}}
      initialMessage="Hello! I'm the Base Model without any special context or prompting. Ask me anything, and I'll respond with my default capabilities."
      ref={chatBox1Ref}
      useRaw={useRaw}
      modelType="base"
      apiProvider={apiProvider}
    />
  );

  const chatBox2 = (
    <ChatBox 
      backendUrl={backendUrl}
      passLoadingState={() => {}}
      initialMessage="Hello! I'm Health LLM, a model enhanced with healthcare domain knowledge. I specialize in medical and health-related information, providing reliable guidance about health conditions, treatments, and wellness practices."
      ref={chatBox2Ref}
      useRaw={useRaw}
      modelType="health"
      apiProvider={apiProvider}
    />
  );

  const chatBox3 = (
    <ChatBox 
      backendUrl={backendUrl}
      passLoadingState={() => {}}
      initialMessage="Hello! I'm the Introspective Assistant. I analyze your digital and health data to help you reflect on your behaviors and patterns. I can provide insights about your YouTube viewing, Spotify listening, and health metrics to encourage self-awareness and personal growth through thoughtful questions."
      ref={chatBox3Ref}
      useRaw={useRaw}
      modelType="introspect"
      apiProvider={apiProvider}
    />
  );

  return (
    <>
      <Layout 
        chatBox1={chatBox1}
        chatBox2={chatBox2}
        chatBox3={chatBox3}
      />
      <div className="common-input-container">
        <div className="common-input-header">
          <h3>Send to All Models at Once</h3>
          <div className="settings-toggles">
            <label 
              className={`raw-toggle ${useRaw ? 'active' : ''}`} 
              title="When enabled, gets raw unprocessed data directly from the model"
            >
              <input
                type="checkbox"
                checked={useRaw}
                onChange={(e) => setUseRaw(e.target.checked)}
              />
              <span>Use Raw Data</span>
              <HiOutlineAdjustments className="settings-icon" />
            </label>
            
            {/* <div className="api-provider-selector">
              <select 
                value={apiProvider} 
                onChange={handleApiProviderChange}
                title="Select which AI provider to use"
              >
                <option value="openai" disabled={!apiStatus.openai}>OpenAI {!apiStatus.openai && '(API Key Missing)'}</option>
                <option value="gemini" disabled={!apiStatus.gemini}>Gemini {!apiStatus.gemini && '(API Key Missing)'}</option>
              </select>
            </div> */}
          </div>
        </div>
        <form onSubmit={handleSendToAll}>
          <input
            type="text"
            value={commonInput}
            onChange={(e) => setCommonInput(e.target.value)}
            placeholder="Type a message to send to all chatbots simultaneously..."
            disabled={isLoading}
          />
          <button 
            type="submit" 
            disabled={isLoading || !commonInput.trim()}
            className="common-send-button"
          >
            {isLoading ? 'Sending...' : <IoSend />}
          </button>
        </form>
      </div>
    </>
  );
}

export default App
