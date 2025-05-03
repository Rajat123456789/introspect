import React, { useState, useEffect, useRef, KeyboardEvent, forwardRef, useImperativeHandle } from 'react';
import { FiRefreshCw, FiSend } from 'react-icons/fi';
import { IoSend } from 'react-icons/io5';
import { HiOutlineDatabase, HiOutlineAdjustments, HiChevronDown } from 'react-icons/hi';
import API_ENDPOINTS from '../config';
import ReactMarkdown from 'react-markdown';
import './ChatBox.css';
import IntrospectDataPanel from './IntrospectDataPanel';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  paragraphs?: string[];
}

// Define API provider type
type ApiProvider = 'openai' | 'gemini';

interface ChatBoxProps {
  backendUrl: string;
  onChatStart?: () => void;
  onReset?: () => void;
  onParagraphsUpdate?: (paragraphs: string[]) => void;
  passLoadingState?: (isLoading: boolean) => void;
  initialMessage?: string;
  useRaw?: boolean;
  modelType?: 'base' | 'health' | 'introspect';
  apiProvider?: ApiProvider;
}

// Define the interface for the ref
export interface ChatBoxRef {
  sendMessage: (text: string) => void;
}

const ChatBox = forwardRef<ChatBoxRef, ChatBoxProps>(({ 
  backendUrl, 
  onChatStart, 
  onReset, 
  onParagraphsUpdate, 
  passLoadingState,
  initialMessage,
  useRaw = false,
  modelType = 'base',
  apiProvider = 'openai'
}, ref) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: getInitialMessage(modelType),
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(true);
  const [enterToSend, setEnterToSend] = useState(true);
  const [useYouTubeData, setUseYouTubeData] = useState(false);
  const [youtubeDataAvailable, setYoutubeDataAvailable] = useState(false);
  const [lastCheckedInsightsTime, setLastCheckedInsightsTime] = useState<number>(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [introspectData, setIntrospectData] = useState<any>(null);
  const [introspectInsights, setIntrospectInsights] = useState<any>(null);
  const [isDataPanelOpen, setIsDataPanelOpen] = useState(false);
  const [isYoutubeDropdownOpen, setIsYoutubeDropdownOpen] = useState(false);
  const youtubeDropdownRef = useRef<HTMLDivElement>(null);

  // Function to get appropriate initial message based on model type
  function getInitialMessage(modelType: string): string {
    if (initialMessage) return initialMessage;
    
    // Base message for each model type
    let message = "";
    switch(modelType) {
      case 'base':
        message = "Hello! I'm a helpful assistant designed to provide accurate and concise information. How can I assist you today?";
        break;
      case 'health':
        message = "Hello! I'm Health LLM, your AI medical assistant. I can help you understand health conditions, explain medical concepts, and guide you to better health literacy. Remember, I don't provide medical diagnosis or treatment advice - please consult healthcare professionals for specific medical concerns. How can I assist with your health-related questions today?";
        break;
      case 'introspect':
        message = "Hello! I'm your Introspective Assistant, designed to help you reflect on your digital and health activities. I can analyze patterns in your data to provide personalized insights and encourage meaningful self-reflection. What would you like to explore about your behaviors and habits today?";
        break;
      default:
        message = "Hello! How can I assist you today?";
    }
    
    return message;
  }

  // Expose the sendMessage method to parent components via ref
  useImperativeHandle(ref, () => ({
    sendMessage: (text: string) => {
      // Create a synthetic event object
      const syntheticEvent = {
        preventDefault: () => {}
      } as React.FormEvent;
      
      // Set the input message and then send it
      setInputMessage(text);
      
      // We need to use setTimeout to ensure state is updated before calling handleSendMessage
      setTimeout(() => {
        handleSendMessageWithText(syntheticEvent, text);
      }, 0);
    }
  }));

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      const container = messagesEndRef.current;
      container.scrollTop = container.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check backend connection on component mount
  useEffect(() => {
    const checkBackendConnection = async () => {
      try {
        console.log('Checking backend connection...');
        const response = await fetch(API_ENDPOINTS.HEALTH, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          // Add more resilient fetch options
          mode: 'cors',
          cache: 'no-cache',
          credentials: 'same-origin',
          // Set a timeout to avoid long waits
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });
        
        const data = await response.json();
        console.log('Backend health check response:', data);
        
        if (response.ok) {
          if (data.status === 'healthy') {
            console.log('Backend connection successful');
            setIsBackendConnected(true);
          } else if (data.status === 'degraded') {
            console.warn('Backend is in degraded state:', data.message);
            setIsBackendConnected(false);
            setMessages(prev => [...prev, {
              id: Date.now(),
              text: `${data.message}`,
              sender: 'bot',
              timestamp: new Date(),
            }]);
          }
        } else {
          console.error('Backend health check failed:', data);
          setIsBackendConnected(false);
          setMessages(prev => [...prev, {
            id: Date.now(),
            text: "I'm having trouble connecting to my backend service. Some features may not work properly.",
            sender: 'bot',
            timestamp: new Date(),
          }]);
        }
      } catch (error) {
        console.error('Error checking backend connection:', error);
        setIsBackendConnected(false);
        setMessages(prev => [...prev, {
          id: Date.now(),
          text: "I'm having trouble connecting to my backend service. Some features may not work properly.",
          sender: 'bot',
          timestamp: new Date(),
        }]);
      }
    };
    
    checkBackendConnection();
  }, []);

  // Fetch introspection data if model type is 'introspect'
  useEffect(() => {
    if (isBackendConnected) {
      const fetchIntrospectionData = async () => {
        try {
          console.log('Fetching introspection data...');
          let dataFetchFailed = false;
          let insightsFetchFailed = false;
          let youtubeDataFound = false;
          
          // Fetch raw data with improved error handling
          try {
            console.log('Attempting to fetch from endpoint:', API_ENDPOINTS.INTROSPECT_DATA);
            const rawDataResponse = await fetch(API_ENDPOINTS.INTROSPECT_DATA, {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
              },
              mode: 'cors',
              cache: 'no-cache',
              credentials: 'same-origin',
              // Set timeout to prevent hanging requests
              signal: AbortSignal.timeout(10000) // 10 second timeout
            });
            
            console.log('Raw data response status:', rawDataResponse.status);
            
            if (rawDataResponse.ok) {
              const rawData = await rawDataResponse.json();
              console.log('Introspection raw data fetched successfully');
              setIntrospectData(rawData);
              
              // Check if there's YouTube data
              if (rawData && rawData.youtube && rawData.youtube.recent_videos && rawData.youtube.recent_videos.length > 0) {
                youtubeDataFound = true;
                setYoutubeDataAvailable(true);
                setUseYouTubeData(true); // Automatically enable YouTube data when available
                console.log('YouTube data found:', rawData.youtube.recent_videos.length, 'videos');
              }
            } else {
              // Log more information about the error
              console.error(`Failed to fetch raw data: ${rawDataResponse.status} ${rawDataResponse.statusText}`);
              dataFetchFailed = true;
              
              try {
                // Try to get error message from response if possible
                const errorData = await rawDataResponse.text();
                console.error('Error response:', errorData);
              } catch (textError) {
                console.error('Could not parse error response text');
              }
            }
          } catch (error) {
            console.error('Error fetching introspection raw data:', error);
            dataFetchFailed = true;
          }
          
          // Fetch insights with improved error handling
          try {
            console.log('Attempting to fetch from endpoint:', API_ENDPOINTS.INTROSPECT_INSIGHTS);
            const insightsResponse = await fetch(API_ENDPOINTS.INTROSPECT_INSIGHTS, {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
              },
              mode: 'cors',
              cache: 'no-cache',
              credentials: 'same-origin',
              // Set timeout to prevent hanging requests
              signal: AbortSignal.timeout(10000) // 10 second timeout
            });
            
            console.log('Insights response status:', insightsResponse.status);
            
            if (insightsResponse.ok) {
              const insights = await insightsResponse.json();
              console.log('Introspection insights fetched successfully');
              setIntrospectInsights(insights);
              
              // Automatically provide insights about YouTube viewing if available
              if (youtubeDataFound && insights && insights.youtube) {
                // Check if we have YouTube data
                if (introspectData?.youtube?.recent_videos && introspectData.youtube.recent_videos.length > 0) {
                  // Only show this once when first loading
                  if (messages.length === 1) {
                    const recentVideo = introspectData.youtube.recent_videos[0];
                    const videoTitle = recentVideo.title || "a video";
                    const videoChannel = recentVideo.channel || "a channel";
                    
                    // Create a simple notification instead of detailed insights
                    const youtubeNotification = `I noticed you recently watched "${videoTitle}" from ${videoChannel}. I can provide insights about your YouTube viewing habits if you're interested.`;
                    
                    setMessages(prev => [...prev, {
                      id: Date.now(),
                      text: youtubeNotification,
                      sender: 'bot',
                      timestamp: new Date(),
                    }]);
                  }
                }
              }
            } else {
              // Log more information about the error
              console.error(`Failed to fetch insights: ${insightsResponse.status} ${insightsResponse.statusText}`);
              insightsFetchFailed = true;
              
              try {
                // Try to get error message from response if possible
                const errorData = await insightsResponse.text();
                console.error('Error response:', errorData);
              } catch (textError) {
                console.error('Could not parse error response text');
              }
            }
          } catch (error) {
            console.error('Error fetching introspection insights:', error);
            insightsFetchFailed = true;
          }
          
          // Notify user if both fetches failed and we're in introspect mode
          if (modelType === 'introspect' && dataFetchFailed && insightsFetchFailed) {
            setMessages(prev => [...prev, {
              id: Date.now(),
              text: "I couldn't find your personal context data. I'll continue without personalized insights. Please ensure your context files exist in the backend/content_for_prompt directory.",
              sender: 'bot',
              timestamp: new Date(),
            }]);
          }
          // Notify if only one failed and we're in introspect mode
          else if (modelType === 'introspect' && (dataFetchFailed || insightsFetchFailed)) {
            setMessages(prev => [...prev, {
              id: Date.now(),
              text: "I found some of your personal data but not all of it. Some personalized insights may be limited.",
              sender: 'bot',
              timestamp: new Date(),
            }]);
          }
        } catch (error) {
          console.error('Error in fetchIntrospectionData:', error);
          if (modelType === 'introspect') {
            setMessages(prev => [...prev, {
              id: Date.now(),
              text: "I encountered an error while loading your introspection data. Some personalized features may not work properly.",
              sender: 'bot',
              timestamp: new Date(),
            }]);
          }
        }
      };
      
      fetchIntrospectionData();
    }
  }, [isBackendConnected, backendUrl, messages, modelType]);

  // Check for new YouTube model insights periodically
  useEffect(() => {
    if (isBackendConnected) {
      const checkYouTubeModelInsights = async () => {
        // Skip this check if we just checked recently (within the last 10 seconds)
        const currentTime = Date.now();
        if (currentTime - lastCheckedInsightsTime < 10000) {
          return;
        }
        
        setLastCheckedInsightsTime(currentTime);
        
        try {
          // Fetch model insights specific to this model type
          const response = await fetch(`${API_ENDPOINTS.YOUTUBE_MODEL_INSIGHTS}?model_type=${modelType}`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            mode: 'cors',
            cache: 'no-cache',
            credentials: 'same-origin',
            signal: AbortSignal.timeout(5000) // 5 second timeout
          });
          
          if (response.ok) {
            const data = await response.json();
            
            if (data.status === 'success' && data.insight) {
              // Check if this insight is new (not already in the chat)
              const insightExists = messages.some(
                msg => msg.sender === 'bot' && msg.text === data.insight
              );
              
              if (!insightExists) {
                // Add the insight to the chat
                setMessages(prev => [...prev, {
                  id: Date.now(),
                  text: data.insight,
                  sender: 'bot',
                  timestamp: new Date(),
                }]);
              }
            }
          }
        } catch (error) {
          console.log('No new YouTube model insights available');
        }
      };
      
      // Check immediately on mount
      //checkYouTubeModelInsights();
      
      // Set up interval to check every 30 seconds
      const intervalId = setInterval(checkYouTubeModelInsights, 30000);
      
      return () => {
        clearInterval(intervalId);
      };
    }
  }, [isBackendConnected, modelType, messages, lastCheckedInsightsTime]);

  const resetConversation = async () => {
    setIsResetting(true);
    
    if (!isBackendConnected) {
      setMessages([
        {
          id: Date.now(),
          text: "I've reset our conversation locally. Note that I'm still having trouble connecting to my backend service, so some features may not work properly.",
          sender: 'bot',
          timestamp: new Date(),
        }
      ]);
      setIsResetting(false);
      if (onReset) onReset();
      return;
    }
    
    try {
      const response = await fetch(API_ENDPOINTS.CLEAR_HISTORY, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          session_id: 'default'
        })
      });

      const data = await response.json();
      
      if (response.ok) {
        setMessages([
          {
            id: Date.now(),
            text: modelType === 'base' 
              ? "I've reset our conversation. How can I help you today?"
              : modelType === 'health'
                ? "I've reset our conversation. How can I help you with your health-related questions today?"
                : "I've reset our conversation. How can I help you reflect on your digital and health activities today?",
            sender: 'bot',
            timestamp: new Date(),
          }
        ]);
        if (onReset) onReset();
      } else {
        console.error('Failed to reset conversation:', data.error);
        setMessages(prev => [...prev, {
          id: Date.now(),
          text: "Sorry, I couldn't reset our conversation. Please try again.",
          sender: 'bot',
          timestamp: new Date(),
        }]);
      }
    } catch (error) {
      console.error('Error resetting conversation:', error);
      setMessages(prev => [...prev, {
        id: Date.now(),
        text: "Sorry, I encountered an error while resetting our conversation. Please try again.",
        sender: 'bot',
        timestamp: new Date(),
      }]);
    } finally {
      setIsResetting(false);
    }
  };

  // This is a helper function for sending a message with explicit text
  const handleSendMessageWithText = async (e: React.FormEvent, text: string) => {
    e.preventDefault();
    if (text.trim()) {
      // Trigger chat start on first user message
      if (messages.length === 1 && onChatStart) {
        onChatStart();
      }

      // Check if this is a raw data request
      const isRawRequest = text.startsWith('[RAW_DATA]');
      const cleanText = isRawRequest ? text.replace('[RAW_DATA]', '').trim() : text;

      const newUserMessage: Message = {
        id: Date.now(),
        text: cleanText, // Display clean text to user without the raw flag
        sender: 'user',
        timestamp: new Date(),
      };
      setMessages([...messages, newUserMessage]);
      setInputMessage('');
      setIsLoading(true);
      if (passLoadingState) passLoadingState(true);

      if (!isBackendConnected) {
        const errorMessage: Message = {
          id: Date.now() + 1,
          text: "Sorry, I'm still having trouble connecting to my backend service. Please try again later or reset our conversation using the refresh button above.",
          sender: 'bot',
          timestamp: new Date(),
        };
        setMessages(prevMessages => [...prevMessages, errorMessage]);
        setIsLoading(false);
        if (passLoadingState) passLoadingState(false);
        return;
      }

      try {
        // Basic payload with message and session
        const payload: any = {
          message: cleanText,
          session_id: 'default',
          use_raw_data: useRaw || isRawRequest || useYouTubeData, // Add useYouTubeData to trigger raw data usage
          model_type: modelType,
          api_provider: apiProvider // Add the API provider to the payload
        };

        // Add model-specific context data
        if (modelType === 'introspect') {
          // Add introspection data if available
          if (introspectData) {
            payload.introspect_data = introspectData;
          }
          
          // Add introspection insights if available
          if (introspectInsights) {
            payload.introspect_insights = introspectInsights;
          }
        }

        console.log('Sending chat request with payload:', payload);
        const response = await fetch(API_ENDPOINTS.CHAT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          mode: 'cors',
          cache: 'no-cache',
          credentials: 'same-origin',
          body: JSON.stringify(payload)
        });

        const data = await response.json();
        console.log('Chat response received:', data);
        
        if (!response.ok) {
          if (response.status === 429) {
            throw new Error("I'm currently experiencing high demand. Please wait a few seconds and try your message again.");
          }
          throw new Error(data.error || data.message || `Error: ${response.statusText}`);
        }

        // Update paragraphs if they exist in the response
        if (data.paragraphs && Array.isArray(data.paragraphs) && onParagraphsUpdate) {
          console.log('Paragraphs received from API:', data.paragraphs);
          onParagraphsUpdate(data.paragraphs);
        }
        
        const newBotMessage: Message = {
          id: Date.now() + 1,
          text: data.message,
          sender: 'bot',
          timestamp: new Date(),
        };
        
        setMessages(prevMessages => [...prevMessages, newBotMessage]);
      } catch (error) {
        console.error('Error sending message:', error);
        
        const errorMessage: Message = {
          id: Date.now() + 1,
          text: error instanceof Error ? error.message : "Sorry, I encountered an error while processing your request. Please try again later.",
          sender: 'bot',
          timestamp: new Date(),
        };
        
        setMessages(prevMessages => [...prevMessages, errorMessage]);
      } finally {
        setIsLoading(false);
        if (passLoadingState) passLoadingState(false);
      }
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    handleSendMessageWithText(e, inputMessage);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter') {
      if (enterToSend && !e.shiftKey) {
        e.preventDefault();
        if (inputMessage.trim()) {
          handleSendMessage(e);
        }
      }
    }
  };

  // Add effect to handle outside clicks for the dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (youtubeDropdownRef.current && !youtubeDropdownRef.current.contains(event.target as Node)) {
        setIsYoutubeDropdownOpen(false);
      }
    }
    
    // Add event listener
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      // Remove event listener on cleanup
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [youtubeDropdownRef]);

  return (
    <div className="chatbox">
      <div className="chatbox-controls">
        <div className="controls-left">
          <label className="enter-toggle">
            <input
              type="checkbox"
              checked={enterToSend}
              onChange={(e) => setEnterToSend(e.target.checked)}
            />
            Enter to send
          </label>
          
          {youtubeDataAvailable && (
            <div className="youtube-controls" ref={youtubeDropdownRef}>
              <div 
                className="youtube-button"
                onClick={() => {
                  // Toggle dropdown when button is clicked
                  setIsYoutubeDropdownOpen(!isYoutubeDropdownOpen);
                }}
              >
                <HiOutlineDatabase className="youtube-icon" />
                <span>YouTube Data</span>
                {useYouTubeData && <span className="status-indicator active">ON</span>}
                {!useYouTubeData && <span className="status-indicator">OFF</span>}
                <HiChevronDown className={`dropdown-icon ${isYoutubeDropdownOpen ? 'open' : ''}`} />
              </div>
              
              {modelType === 'introspect' && (
                <div className={`youtube-dropdown ${isYoutubeDropdownOpen ? 'show' : ''}`}>
                  <div 
                    className="dropdown-item toggle-item"
                    onClick={(e) => {
                      e.stopPropagation();
                      setUseYouTubeData(!useYouTubeData);
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={useYouTubeData}
                      onChange={() => {}}
                      onClick={(e) => e.stopPropagation()}
                    />
                    <span>Use data in chat</span>
                  </div>
                  
                  <div 
                    className="dropdown-item"
                    onClick={async (e) => {
                      e.stopPropagation();
                      setIsYoutubeDropdownOpen(false); // Close dropdown after clicking
                      try {
                        setIsLoading(true);
                        const response = await fetch(API_ENDPOINTS.YOUTUBE_ANALYZE, {
                          method: 'POST',
                          headers: {
                            'Content-Type': 'application/json',
                          }
                        });
                        
                        if (response.ok) {
                          const data = await response.json();
                          if (data.model_insights && data.model_insights.introspect) {
                            setMessages(prev => [...prev, {
                              id: Date.now(),
                              text: data.model_insights.introspect,
                              sender: 'bot',
                              timestamp: new Date(),
                            }]);
                          } else {
                            setMessages(prev => [...prev, {
                              id: Date.now(),
                              text: "I've analyzed your YouTube viewing patterns and found some interesting insights. Let's discuss what your viewing habits might reveal about your current interests.",
                              sender: 'bot',
                              timestamp: new Date(),
                            }]);
                          }
                        } else {
                          console.error('Failed to analyze YouTube data');
                        }
                      } catch (error) {
                        console.error('Error analyzing YouTube data:', error);
                      } finally {
                        setIsLoading(false);
                      }
                    }}
                  >
                    <HiOutlineAdjustments className="dropdown-icon-item" />
                    <span>Analyze viewing patterns</span>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {!youtubeDataAvailable && (
            <div className="youtube-button disabled">
              <HiOutlineDatabase className="youtube-icon" />
              <span>No YouTube Data</span>
            </div>
          )}
          
          {/* Remove the previous YouTube toggle and analyze button */}
          {/* View Data button hidden for now */}
        </div>
        <button 
          className="reset-button" 
          onClick={resetConversation} 
          disabled={isResetting || isLoading}
          title="Reset conversation"
        >
          <FiRefreshCw className={isResetting ? 'spinning' : ''} />
          {isResetting ? 'Resetting...' : 'Reset'}
        </button>
      </div>
      <div className="messages-container" ref={messagesEndRef}>
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
          >
            <div className="message-content">
              {message.sender === 'user' ? (
                <p>{message.text}</p>
              ) : (
                <ReactMarkdown>{message.text}</ReactMarkdown>
              )}
            </div>
            <div className="message-timestamp">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message bot-message loading">
            <div className="message-content">
              <div className="loading-text-animation">
                <span className="loading-text-animation-text loading-text">AI is thinking</span>
                <div className="dot-animation">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      <form onSubmit={handleSendMessage} className="input-container">
        <textarea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            modelType === 'base' 
              ? `Ask me anything... ${enterToSend ? '(Press Enter to send, Shift+Enter for new line)' : ''}`
              : modelType === 'health'
                ? `Ask me about health conditions, treatments, or wellness... ${enterToSend ? '(Press Enter to send, Shift+Enter for new line)' : ''}`
                : `Ask me to help reflect on your digital and health activities... ${enterToSend ? '(Press Enter to send, Shift+Enter for new line)' : ''}`
          }
          className="message-input"
          disabled={isLoading || isResetting}
          rows={3}
        />
        <button 
          className="send-button" 
          onClick={handleSendMessage} 
          disabled={isLoading || isResetting || !inputMessage.trim()}
        >
          {isLoading ? (
            <div className="loading-text-animation button-loading">
              <span className="loading-text-animation-text  loading-text">Sending</span>
              <div className="dot-animation">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          ) : (
            <IoSend />
          )}
        </button>
      </form>
      
      <IntrospectDataPanel 
        isOpen={isDataPanelOpen} 
        onClose={() => setIsDataPanelOpen(false)} 
      />
    </div>
  );
});

export default ChatBox; 