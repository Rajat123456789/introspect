import React, { useState, useEffect } from 'react';
import { Button, Tooltip, Box, CircularProgress } from '@mui/material';
import RecruiterProfileModal from './RecruiterProfile';
import API_ENDPOINTS from '../config';

interface RecruiterProfile {
  id?: number;
  email: string;
  name: string;
  company: string;
  role: string;
  company_description?: string;
  industry?: string;
  company_size?: string;
  recruiting_focus?: string;
  outreach_preferences?: Record<string, any>;
}

const SetupProfile: React.FC = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [hasProfile, setHasProfile] = useState(false);
  const [isChecking, setIsChecking] = useState(true);
  const [profileData, setProfileData] = useState<RecruiterProfile | null>(null);

  useEffect(() => {
    checkBackendConnection();
    // Check connection every 30 seconds
    const interval = setInterval(checkBackendConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (isBackendConnected) {
      checkExistingProfile();
    }
  }, [isBackendConnected]);

  const checkExistingProfile = async () => {
    try {
      setIsChecking(true);
      const response = await fetch(API_ENDPOINTS.RECRUITER_PROFILE, {
        credentials: 'include'
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.data?.id) {
          setHasProfile(true);
          setProfileData(data.data);
        } else {
          setHasProfile(false);
          setProfileData(null);
        }
      } else {
        setHasProfile(false);
        setProfileData(null);
      }
    } catch (err) {
      setHasProfile(false);
      setProfileData(null);
    } finally {
      setIsChecking(false);
    }
  };

  const checkBackendConnection = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.HEALTH, {
        credentials: 'include'
      });
      setIsBackendConnected(response.ok);
    } catch (err) {
      setIsBackendConnected(false);
    }
  };

  const handleOpenModal = () => {
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    // Refresh profile status after modal closes
    if (isBackendConnected) {
      checkExistingProfile();
    }
  };

  const button = (
    <Button
      variant="contained"
      color="primary"
      onClick={handleOpenModal}
      disabled={!isBackendConnected || isChecking}
      sx={{ 
        minWidth: 200,
        mt: 2.5,
        fontSize: '1rem',
        padding: '10px 20px',
        '&:hover': {
          backgroundColor: 'primary.dark',
        },
      }}
    >
      {isChecking ? (
        <CircularProgress size={20} color="inherit" sx={{ mr: 1 }} />
      ) : hasProfile ? (
        'View Profile'
      ) : (
        'Setup Profile'
      )}
    </Button>
  );

  return (
    <>
      {isBackendConnected ? (
        button
      ) : (
        <Tooltip title="Backend server is not available" placement="top">
          <span>{button}</span>
        </Tooltip>
      )}
      
      <RecruiterProfileModal
        open={isModalOpen}
        onClose={handleCloseModal}
        isBackendConnected={isBackendConnected}
        initialProfile={profileData}
      />
    </>
  );
};

export default SetupProfile; 