import React, { useState } from 'react';
import './IntrospectDataPanel.css';

interface IntrospectDataPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const IntrospectDataPanel: React.FC<IntrospectDataPanelProps> = ({
  isOpen,
  onClose
}) => {
  const [activeTab, setActiveTab] = useState<'youtube' | 'spotify' | 'health'>('youtube');

  if (!isOpen) return null;

  return (
    <div className="introspect-data-panel">
      <div className="panel-header">
        <h3>Your Data Context</h3>
        <button className="close-button" onClick={onClose}>×</button>
      </div>
      
      <div className="panel-tabs">
        <button 
          className={`tab-button ${activeTab === 'youtube' ? 'active' : ''}`}
          onClick={() => setActiveTab('youtube')}
        >
          YouTube
        </button>
        <button 
          className={`tab-button ${activeTab === 'spotify' ? 'active' : ''}`}
          onClick={() => setActiveTab('spotify')}
        >
          Spotify
        </button>
        <button 
          className={`tab-button ${activeTab === 'health' ? 'active' : ''}`}
          onClick={() => setActiveTab('health')}
        >
          Health
        </button>
      </div>
      
      <div className="panel-content">
        {activeTab === 'youtube' && (
          <div className="data-section">
            <h4>YouTube Activity</h4>
            <div className="data-item">
              <span className="label">Total Videos Watched:</span>
              <span className="value">1,247</span>
            </div>
            <div className="data-item">
              <span className="label">Average Daily Watch Time:</span>
              <span className="value">78.4 minutes</span>
            </div>
            <div className="data-item">
              <span className="label">Top Categories:</span>
              <span className="value">Entertainment, Education, Technology</span>
            </div>
            <div className="data-item">
              <span className="label">Most Active Time:</span>
              <span className="value">Evenings (35% of viewing)</span>
            </div>
            <div className="data-item">
              <span className="label">Escapism Index:</span>
              <span className="value">72/100</span>
            </div>
          </div>
        )}
        
        {activeTab === 'spotify' && (
          <div className="data-section">
            <h4>Spotify Activity</h4>
            <div className="data-item">
              <span className="label">Total Tracks Played:</span>
              <span className="value">3,648</span>
            </div>
            <div className="data-item">
              <span className="label">Average Daily Listening:</span>
              <span className="value">62.3 minutes</span>
            </div>
            <div className="data-item">
              <span className="label">Top Genres:</span>
              <span className="value">Indie Rock, Alternative, Lo-fi Beats</span>
            </div>
            <div className="data-item">
              <span className="label">Mood Distribution:</span>
              <span className="value">32% Upbeat, 28% Melancholic</span>
            </div>
            <div className="data-item">
              <span className="label">Focus Playlist Usage:</span>
              <span className="value">35% of listening time</span>
            </div>
          </div>
        )}
        
        {activeTab === 'health' && (
          <div className="data-section">
            <h4>Health Metrics</h4>
            <div className="data-item">
              <span className="label">Average Daily Steps:</span>
              <span className="value">8,450</span>
            </div>
            <div className="data-item">
              <span className="label">Weekly Workout Minutes:</span>
              <span className="value">185</span>
            </div>
            <div className="data-item">
              <span className="label">Average Sleep Duration:</span>
              <span className="value">6.7 hours</span>
            </div>
            <div className="data-item">
              <span className="label">Sleep Quality Score:</span>
              <span className="value">68/100</span>
            </div>
            <div className="data-item">
              <span className="label">Average Resting Heart Rate:</span>
              <span className="value">68 BPM</span>
            </div>
          </div>
        )}
      </div>
      
      <div className="panel-footer">
        <p className="privacy-note">This data is stored locally and is only used to provide personalized insights.</p>
      </div>
    </div>
  );
};

export default IntrospectDataPanel; 