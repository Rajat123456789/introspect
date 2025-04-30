import React from 'react';
import ThemeToggle from './ThemeToggle';
import './Layout.css';

interface LayoutProps {
  chatBox1: React.ReactNode;
  chatBox2: React.ReactNode;
  chatBox3: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ chatBox1, chatBox2, chatBox3 }) => {
  return (
    <div className="layout">
      <header className="app-header">
        <h1>Introspect.AI: The Introspective Assistant</h1>
        <p>Helping you gain deeper insights into your behavior patterns across digital platforms and health activities</p>
        <div className="header-controls">
          <ThemeToggle />
        </div>
      </header>
      <div className="main-content three-chatbot-layout">
        <div className="chatbot-section">
          <h2>Base Model</h2>
          {chatBox1}
        </div>
        <div className="chatbot-section">
          <h2>HealthLLM</h2>
          {chatBox2}
        </div>
        <div className="chatbot-section">
          <h2>Introspective Assistant</h2>
          {chatBox3}
        </div>
      </div>
    </div>
  );
};

export default Layout; 