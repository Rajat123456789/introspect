import React, { useState, useEffect } from 'react';
import Layout from './components/Layout'
import ChatBox from './components/ChatBox';
import ParagraphCards from './components/ParagraphCards';
import RecruiterProfileModal from './components/RecruiterProfile';
import './App.css'
import API_ENDPOINTS from './config';
import { FiCopy } from 'react-icons/fi';

interface WorkspaceProps {
  paragraphs: string[];
  onParagraphEdit: (index: number, newText: string) => void;
  onRequestAIEdit: (index: number, instruction: string) => void;
  isLoading: boolean;
}

// Define a compatible interface for use within ChatBox
interface RecruiterProfileData {
  id?: number;
  [key: string]: any;
}

const Workspace: React.FC<WorkspaceProps> = ({ 
  paragraphs, 
  onParagraphEdit,
  onRequestAIEdit,
  isLoading
}) => {
  const [showCopyNotification, setShowCopyNotification] = useState(false);
  const [copyButtonText, setCopyButtonText] = useState("Copy Message");
  const [isCopying, setIsCopying] = useState(false);

  if (paragraphs.length === 0) {
    return null;
  }

  const handleCopyMessage = () => {
    // Format the message properly for pasting into email or LinkedIn
    // - Each paragraph is separated by two newlines
    // - Remove any markdown formatting that might exist
    // - Remove any placeholder brackets that might confuse users
    const formattedMessage = paragraphs
      .map(para => 
        para
          .replace(/\[/g, '')  // Remove opening brackets for placeholders
          .replace(/\]/g, '')  // Remove closing brackets for placeholders
          .trim()
      )
      .join('\n\n');
    
    setIsCopying(true);
    setCopyButtonText("Copying...");
    
    navigator.clipboard.writeText(formattedMessage)
      .then(() => {
        // Show success in button and notification
        setCopyButtonText("Copied!");
        setShowCopyNotification(true);
        
        // Reset button after delay
        setTimeout(() => {
          setCopyButtonText("Copy Message");
          setIsCopying(false);
          setShowCopyNotification(false);
        }, 2000);
      })
      .catch(err => {
        console.error('Failed to copy message: ', err);
        setCopyButtonText("Failed to Copy");
        setIsCopying(false);
        
        // Reset button after delay
        setTimeout(() => {
          setCopyButtonText("Copy Message");
        }, 2000);
      });
  };

  return (
    <div className="workspace-content">
      <div className="workspace-header">
        <h2>Message Draft</h2>
        <button 
          className={`copy-message-button ${isCopying ? 'copying' : ''}`}
          onClick={handleCopyMessage}
          title="Copy the entire message to clipboard"
          disabled={isCopying}
        >
          <FiCopy style={{ marginRight: '6px' }} /> {copyButtonText}
        </button>
      </div>
      <ParagraphCards 
        paragraphs={paragraphs} 
        onParagraphEdit={onParagraphEdit} 
        onRequestAIEdit={onRequestAIEdit}
        isLoading={isLoading}
      />
      {showCopyNotification && (
        <div className="copy-notification">
          Message copied to clipboard!
        </div>
      )}
    </div>
  );
};

function App() {
  const [isChatStarted, setIsChatStarted] = useState(false);
  const [currentParagraphs, setCurrentParagraphs] = useState<string[]>([]);
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
  const [recruiterProfile, setRecruiterProfile] = useState<RecruiterProfileData | null>(null);
  const [recruiterId, setRecruiterId] = useState<number | undefined>(undefined);
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';
  const [isPassingLoadingState, setIsPassingLoadingState] = useState(false);

  useEffect(() => {
    // Check if backend is connected on component mount
    const checkBackendConnection = async () => {
      try {
        const response = await fetch(`${backendUrl}/api/health`, { 
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });
        setIsBackendConnected(response.ok);
      } catch (error) {
        setIsBackendConnected(false);
      }
    };
    
    checkBackendConnection();
  }, [backendUrl]);

  // Fetch recruiter profile on app initialization
  useEffect(() => {
    if (isBackendConnected) {
      const fetchRecruiterProfile = async () => {
        try {
          const response = await fetch(API_ENDPOINTS.RECRUITER_PROFILE, {
            credentials: 'include'
          });
          
          if (response.ok) {
            const data = await response.json();
            if (data.data) {
              setRecruiterProfile(data.data);
              setRecruiterId(data.data.id);
            }
          }
        } catch (error) {
          console.error('Error fetching recruiter profile:', error);
        }
      };
      
      fetchRecruiterProfile();
    }
  }, [isBackendConnected]);

  const handleChatStart = () => {
    setIsChatStarted(true);
  };

  const handleReset = () => {
    setIsChatStarted(false);
    setCurrentParagraphs([]);
  };

  const handleParagraphsUpdate = (paragraphs: string[]) => {
    setCurrentParagraphs(paragraphs);
  };

  const handleParagraphEdit = (index: number, newText: string) => {
    setCurrentParagraphs(prev => {
      const updated = [...prev];
      updated[index] = newText;
      return updated;
    });
  };

  const handleRequestAIEdit = async (index: number, instruction: string) => {
    if (!isBackendConnected || index >= currentParagraphs.length) return;
    
    setIsLoading(true);
    try {
      const currentParagraph = currentParagraphs[index];
      console.log(`Requesting AI edit for paragraph ${index} with instruction: ${instruction}`);
      console.log('Original paragraph:', currentParagraph);
      
      // Prepare a more descriptive message that includes the paragraph content
      const editMessage = `Please edit the following paragraph based on this instruction: "${instruction}"

Original paragraph:
"${currentParagraph}"

Please return ONLY the edited paragraph without explanations or additional text. Just give me the revised version without quotation marks or additional context.`;
      
      const payload = {
        message: editMessage,
        paragraph_index: index,
        refinement_instruction: instruction,
        session_id: 'default'
      };

      // Add recruiter_id if available
      if (recruiterId) {
        Object.assign(payload, { recruiter_id: recruiterId });
      }

      const response = await fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      console.log('AI edit response:', data);
      
      // Check if response was successful
      if (!response.ok) {
        console.error('Error from server:', data.error || data.message || response.statusText);
        return;
      }
      
      // Get the most relevant refined paragraph from the response
      const extractRefinedParagraph = () => {
        // Check if we have paragraphs array
        if (data.paragraphs && Array.isArray(data.paragraphs) && data.paragraphs.length > 0) {
          console.log('Extracted paragraphs:', data.paragraphs);
          
          // If the backend directly returned the updated paragraph in the right slot,
          // simply use the paragraph at the corresponding index
          if (index < data.paragraphs.length) {
            return data.paragraphs[index];
          }
          
          // Otherwise use the first paragraph (most likely the refined one)
          return data.paragraphs[0];
        }
        
        // If no paragraphs array, try to extract from message
        if (data.message) {
          console.log('Extracting from message:', data.message);
          
          // Try to find a section that starts with "Refined paragraph:" or similar
          const refinedSections = [
            /Refined paragraph:[\s\n]*(.*?)(?:\n\n|$)/s,
            /Edited paragraph:[\s\n]*(.*?)(?:\n\n|$)/s,
            /Updated paragraph:[\s\n]*(.*?)(?:\n\n|$)/s,
            /Here's the refined paragraph:[\s\n]*(.*?)(?:\n\n|$)/s,
            /Here's the edited version:[\s\n]*(.*?)(?:\n\n|$)/s,
            /Here is the edited paragraph:[\s\n]*(.*?)(?:\n\n|$)/s,
            /Here's my edit:[\s\n]*(.*?)(?:\n\n|$)/s,
            /I've edited it to:[\s\n]*(.*?)(?:\n\n|$)/s
          ];
          
          for (const pattern of refinedSections) {
            const match = data.message.match(pattern);
            if (match && match[1]) {
              console.log('Found refined section with pattern:', pattern);
              return match[1].trim();
            }
          }
          
          // If no specific section found, try to identify the edited paragraph by looking
          // for sections that don't look like explanations
          const paragraphs = data.message
            .split('\n\n')
            .map((p: string) => p.trim())
            .filter((p: string) => p.length > 0);
            
          if (paragraphs.length > 0) {
            // Skip paragraphs that look like explanations
            const explanationStarters = [
              /^I've/i, /^I have/i, /^I made/i, /^This edit/i, 
              /^Based on/i, /^Following/i, /^As requested/i,
              /^Here's why/i, /^The changes/i, /^In this edit/i
            ];
            
            // Find the first paragraph that doesn't match explanation patterns
            for (const para of paragraphs) {
              const isExplanation = explanationStarters.some(pattern => pattern.test(para));
              if (!isExplanation) {
                console.log('Found a non-explanation paragraph:', para);
                return para;
              }
            }
            
            // If all look like explanations, just use the first paragraph
            console.log('Using first paragraph from message as fallback');
            return paragraphs[0];
          }
        }
        
        // Fallback: if all else fails, return the original paragraph
        console.log('No valid content found, returning original paragraph');
        return currentParagraph;
      };
      
      const refinedParagraph = extractRefinedParagraph();
      console.log('Final refined paragraph:', refinedParagraph);
      
      // Only update if we have something different from the original
      if (refinedParagraph && refinedParagraph !== currentParagraph) {
        handleParagraphEdit(index, refinedParagraph);
      } else {
        console.warn('No changes detected in the refined paragraph');
      }
    } catch (error) {
      console.error('Error requesting AI edit:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleOpenProfile = () => {
    setIsProfileModalOpen(true);
  };

  const handleCloseProfile = () => {
    setIsProfileModalOpen(false);
    // Update recruiterId when profile modal is closed (profile might have been updated)
    setRecruiterId(recruiterProfile?.id);
  };

  const chatBox = (
    <ChatBox 
      backendUrl={backendUrl} 
      onChatStart={handleChatStart}
      onReset={handleReset}
      onParagraphsUpdate={handleParagraphsUpdate}
      recruiterId={recruiterId}
      passLoadingState={setIsPassingLoadingState}
    />
  );

  return (
    <>
      <Layout 
        isChatStarted={isChatStarted} 
        chatBox={chatBox}
        onOpenProfile={handleOpenProfile}
      >
        {!isChatStarted ? (
          <div className="workspace-content">
            <h1>Helix Outreach Assistant</h1>
            <div className="examples-column">
              <h2>Example Prompts</h2>
              <div className="prompt-cards">
                <div className="prompt-card">
                  <h3>Cold Outreach</h3>
                  <p>"Create a cold outreach message to Sarah Chen for a Lead Developer role at Acme Tech."</p>
                </div>
                <div className="prompt-card">
                  <h3>Follow-Up</h3>
                  <p>"Draft a follow-up to Michael Rodriguez for a Product Manager position. No response to first email."</p>
                </div>
                <div className="prompt-card">
                  <h3>Improve</h3>
                  <p>"Make this more personal: 'Dear candidate, I hope this email finds you well...'"</p>
                </div>
              </div>
            </div>
          </div>
        ) : 
          isPassingLoadingState ? (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <p>Loading...</p>
            </div>
          ) : (
          <Workspace 
            paragraphs={currentParagraphs}
            onParagraphEdit={handleParagraphEdit}
            onRequestAIEdit={handleRequestAIEdit}
            isLoading={isLoading}
          />
        )}
      </Layout>

      <RecruiterProfileModal
        open={isProfileModalOpen}
        onClose={handleCloseProfile}
        isBackendConnected={isBackendConnected}
        initialProfile={recruiterProfile}
      />
    </>
  );
}

export default App
