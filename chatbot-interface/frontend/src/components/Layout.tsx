import React from 'react';
import ThemeToggle from './ThemeToggle';
import './Layout.css';
import { IconButton } from '@mui/material';

interface LayoutProps {
  children?: React.ReactNode;
  isChatStarted: boolean;
  chatBox: React.ReactNode;
  onOpenProfile?: () => void;
}

const Layout: React.FC<LayoutProps> = ({ children, isChatStarted, chatBox, onOpenProfile }) => {
  return (
    <div className="layout">
      <header className="app-header">
        <h1>Helix: The Outreach Assistant</h1>
        <p>Helping recruiters craft personalized, effective messages to candidates</p>
        <div className="header-controls">
          {onOpenProfile && (
            <div className="profile-toggle">
              <IconButton 
                onClick={onOpenProfile} 
                color="inherit" 
                size="small"
                aria-label="Update profile"
              >
                <span className="profile-icon">👤</span>
              </IconButton>
            </div>
          )}
          <ThemeToggle />
        </div>
      </header>
      <div className="main-content">
        <div className="chatbot-section">
          <h2>Helix Assistant</h2>
          {chatBox}
        </div>
        <div className={`workspace-section ${isChatStarted ? 'chat-active' : ''}`}>
          <h2>Workspace</h2>
          {children}
        </div>
      </div>
    </div>
  );
};

export default Layout; 