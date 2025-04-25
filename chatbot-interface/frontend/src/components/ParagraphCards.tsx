import React, { useState, useEffect, useRef } from 'react';
import './ParagraphCards.css';
import ReactMarkdown from 'react-markdown';

interface ParagraphCardsProps {
  paragraphs: string[];
  onParagraphEdit: (index: number, newText: string) => void;
  onRequestAIEdit?: (index: number, paragraph: string) => void;
  isLoading?: boolean;
}

const ParagraphCards: React.FC<ParagraphCardsProps> = ({ 
  paragraphs, 
  onParagraphEdit,
  onRequestAIEdit,
  isLoading = false
}) => {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState('');
  const [aiEditingIndex, setAiEditingIndex] = useState<number | null>(null);
  const [aiEditInstruction, setAiEditInstruction] = useState('');
  const [showAiEditForm, setShowAiEditForm] = useState(false);
  const [submittedIndex, setSubmittedIndex] = useState<number | null>(null);
  const [prevIsLoading, setPrevIsLoading] = useState(false);
  const [updatedParagraphIndex, setUpdatedParagraphIndex] = useState<number | null>(null);
  const prevParagraphs = useRef<string[]>([]);

  // Effect to detect paragraph updates
  useEffect(() => {
    if (prevParagraphs.current.length > 0) {
      // Find which paragraph has changed
      const changedIndex = paragraphs.findIndex(
        (text, index) => 
          index < prevParagraphs.current.length && 
          text !== prevParagraphs.current[index]
      );
      
      if (changedIndex !== -1) {
        setUpdatedParagraphIndex(changedIndex);
        // Reset animation after a delay
        setTimeout(() => {
          setUpdatedParagraphIndex(null);
        }, 2000);
      }
    }
    
    // Update previous paragraphs
    prevParagraphs.current = [...paragraphs];
  }, [paragraphs]);

  // Effect to reset submittedIndex when loading completes
  useEffect(() => {
    // If loading was true and now it's false, reset the submittedIndex
    if (prevIsLoading && !isLoading) {
      setSubmittedIndex(null);
    }
    // Update the previous loading state
    setPrevIsLoading(isLoading);
  }, [isLoading, prevIsLoading]);

  const handleEditStart = (index: number, text: string) => {
    setEditingIndex(index);
    setEditText(text);
    setAiEditingIndex(null);
    setShowAiEditForm(false);
  };

  const handleEditSave = (index: number) => {
    onParagraphEdit(index, editText);
    setEditingIndex(null);
  };

  const handleEditCancel = () => {
    setEditingIndex(null);
  };

  const handleAiEditStart = (index: number) => {
    setAiEditingIndex(index);
    setAiEditInstruction('');
    setShowAiEditForm(true);
    setEditingIndex(null);
  };

  const handleAiEditSubmit = (index: number) => {
    if (onRequestAIEdit && aiEditInstruction.trim()) {
      setSubmittedIndex(index);
      onRequestAIEdit(index, aiEditInstruction);
      setShowAiEditForm(false);
      setAiEditingIndex(null);
    }
  };

  const handleAiEditCancel = () => {
    setShowAiEditForm(false);
    setAiEditingIndex(null);
  };

  return (
    <div className="paragraph-cards">
      {paragraphs.map((paragraph, index) => (
        <div 
          key={index} 
          className={`paragraph-card ${updatedParagraphIndex === index ? 'paragraph-updated' : ''}`}
        >
          {editingIndex === index ? (
            <div className="edit-container">
              <textarea
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                className="edit-textarea"
                autoFocus
              />
              <div className="edit-actions">
                <button onClick={() => handleEditSave(index)} className="save-button">
                  Save
                </button>
                <button onClick={handleEditCancel} className="cancel-button">
                  Cancel
                </button>
              </div>
            </div>
          ) : aiEditingIndex === index && showAiEditForm ? (
            <div className="ai-edit-container">
              <div className="paragraph-content">
                <p><strong>Original:</strong></p>
                <p>{paragraph}</p>
              </div>
              <div className="ai-edit-form">
                <p><strong>How should the AI edit this?</strong></p>
                <textarea
                  value={aiEditInstruction}
                  onChange={(e) => setAiEditInstruction(e.target.value)}
                  className="ai-edit-textarea"
                  placeholder="Examples: Make it more personal, shorten it, focus more on value proposition, etc."
                  autoFocus
                  rows={3}
                />
                <div className="edit-actions">
                  <button 
                    onClick={() => handleAiEditSubmit(index)} 
                    className="save-button"
                    disabled={!aiEditInstruction.trim()}
                  >
                    Send to AI
                  </button>
                  <button onClick={handleAiEditCancel} className="cancel-button">
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="view-container">
              <div className="paragraph-content">
                {isLoading && submittedIndex === index ? (
                  <div className="loading-container">
                    <div className="loading-overlay">
                      <div className="loading-text">
                        <span>AI is editing</span>
                        <span className="dot-animation">
                          <span>.</span>
                          <span>.</span>
                          <span>.</span>
                        </span>
                      </div>
                    </div>
                    <div className="faded-text">{paragraph}</div>
                  </div>
                ) : (
                  <ReactMarkdown>{paragraph}</ReactMarkdown>
                )}
              </div>
              <div className="paragraph-actions">
                <button
                  onClick={() => handleEditStart(index, paragraph)}
                  className="edit-button"
                  disabled={isLoading && submittedIndex === index}
                >
                  Edit Manually
                </button>
                {onRequestAIEdit && (
                  <button
                    onClick={() => handleAiEditStart(index)}
                    className="ai-edit-button"
                    disabled={isLoading && submittedIndex === index}
                  >
                    Request AI Edit
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default ParagraphCards; 