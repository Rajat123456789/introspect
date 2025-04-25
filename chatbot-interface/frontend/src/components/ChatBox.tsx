import React, { useState, useEffect, useRef, KeyboardEvent } from 'react';
import { FiRefreshCw, FiSend } from 'react-icons/fi';
import API_ENDPOINTS from '../config';
import ReactMarkdown from 'react-markdown';
import './ChatBox.css';
import { IoSend } from 'react-icons/io5';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  paragraphs?: string[];
}

interface ChatBoxProps {
  backendUrl: string;
  onChatStart?: () => void;
  onReset?: () => void;
  onParagraphsUpdate?: (paragraphs: string[]) => void;
  recruiterId?: number;
  passLoadingState?: (isLoading: boolean) => void;
}

const ChatBox: React.FC<ChatBoxProps> = ({ 
  backendUrl, 
  onChatStart, 
  onReset, 
  onParagraphsUpdate, 
  recruiterId,
  passLoadingState
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I'm Helix, your AI outreach assistant for recruiters. I can help you create personalized cold outreach messages, craft follow-up emails, design outreach campaigns, and improve your existing messages. What type of recruiting message can I help you with today?",
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(true);
  const [enterToSend, setEnterToSend] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [recruiterContext, setRecruiterContext] = useState<any>(null);

  

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      const container = messagesEndRef.current;
      container.scrollTop = container.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch recruiter context if ID is available
  useEffect(() => {
    if (recruiterId && isBackendConnected) {
      const fetchRecruiterContext = async () => {
        try {
          console.log(`Fetching recruiter details for ID: ${recruiterId}`);
          const response = await fetch(`${API_ENDPOINTS.RECRUITER_PROFILE}/${recruiterId}`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
            credentials: 'include'
          });
          
          if (response.ok) {
            const data = await response.json();
            if (data.data) {
              console.log('Recruiter context fetched:', data.data);
              setRecruiterContext(data.data);
            }
          } else {
            console.error('Failed to fetch recruiter context');
          }
        } catch (error) {
          console.error('Error fetching recruiter context:', error);
        }
      };
      
      fetchRecruiterContext();
    }
  }, [recruiterId, isBackendConnected]);

  // Check backend connection on component mount
  useEffect(() => {
    const checkBackendConnection = async () => {
      try {
        const response = await fetch(API_ENDPOINTS.HEALTH, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include'
        });
        const data = await response.json();
        
        if (response.ok) {
          if (data.status === 'healthy') {
            setIsBackendConnected(true);
          } else if (data.status === 'degraded') {
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
        credentials: 'include',
        body: JSON.stringify({ 
          session_id: 'default',
          recruiter_id: recruiterId
        })
      });

      const data = await response.json();
      
      if (response.ok) {
        setMessages([
          {
            id: Date.now(),
            text: "I've reset our conversation. How can I help you with recruiting outreach messages today?",
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

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (inputMessage.trim()) {
      // Trigger chat start on first user message
      if (messages.length === 1 && onChatStart) {
        onChatStart();
      }

      const newUserMessage: Message = {
        id: Date.now(),
        text: inputMessage,
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
          message: inputMessage,
          session_id: 'default'
        };

        // Add recruiter_id if available
        if (recruiterId) {
          payload.recruiter_id = recruiterId;
          console.log(`Including recruiter ID ${recruiterId} in chat request`);
        } else {
          console.log('No recruiter ID available for chat request');
        }

        // Add any available recruiter context
        if (recruiterContext) {
          payload.recruiter_context = {
            name: recruiterContext.name,
            company: recruiterContext.company,
            role: recruiterContext.role,
            company_description: recruiterContext.company_description,
            industry: recruiterContext.industry
          };
          console.log('Including recruiter context in chat request:', payload.recruiter_context);
        }

        console.log('Sending chat request with payload:', payload);
        const response = await fetch(API_ENDPOINTS.CHAT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
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
          placeholder={`Ask me to help with cold outreach, follow-ups, or improving messages... ${enterToSend ? '(Press Enter to send, Shift+Enter for new line)' : ''}`}
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
    </div>
  );
};

export default ChatBox; 