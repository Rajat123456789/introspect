import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Paper,
  Snackbar,
  Alert,
  SelectChangeEvent,
  Modal,
  CircularProgress,
  Backdrop,
  useTheme,
  IconButton,
} from '@mui/material';
import API_ENDPOINTS from '../config';

export interface RecruiterProfile {
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

interface RecruiterProfileModalProps {
  open: boolean;
  onClose: () => void;
  isBackendConnected: boolean;
  initialProfile: { id?: number } | null;
  onProfileUpdate?: (profile: RecruiterProfile) => void;
}

const COMPANY_SIZES = [
  '1-10',
  '11-50',
  '51-200',
  '201-500',
  '501-1000',
  '1001-5000',
  '5000+'
];

const INDUSTRIES = [
  'Technology',
  'Healthcare',
  'Finance',
  'Education',
  'Manufacturing',
  'Retail',
  'Other'
];

const RecruiterProfileModal: React.FC<RecruiterProfileModalProps> = ({ 
  open, 
  onClose,
  isBackendConnected,
  initialProfile,
  onProfileUpdate
}) => {
  const theme = useTheme();
  
  const modalStyle = {
    position: 'absolute' as 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: '90%',
    maxWidth: 800,
    bgcolor: 'background.paper',
    boxShadow: theme.shadows[24],
    p: 4,
    maxHeight: '90vh',
    overflow: 'auto',
    borderRadius: 1,
  };

  const [profile, setProfile] = useState<RecruiterProfile>({
    email: '',
    name: '',
    company: '',
    role: '',
    company_description: '',
    industry: '',
    company_size: '',
    recruiting_focus: '',
    outreach_preferences: {}
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    if (open && isBackendConnected) {
      if (initialProfile) {
        setProfile(initialProfile as RecruiterProfile);
      } else {
        loadProfile();
      }
    }
  }, [open, isBackendConnected, initialProfile]);

  const loadProfile = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(API_ENDPOINTS.RECRUITER_PROFILE, {
        credentials: 'include'
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.data) {
          setProfile(data.data);
        }
      } else {
        const data = await response.json();
        setError(data.error || 'Failed to load profile');
      }
    } catch (err) {
      setError('Error loading profile');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      const method = profile.id ? 'PUT' : 'POST';
      console.log(`Saving recruiter profile with method ${method}:`, profile);
      
      const response = await fetch(API_ENDPOINTS.RECRUITER_PROFILE, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(profile)
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Profile saved successfully:', data);
        setSuccess('Profile saved successfully');
        
        const updatedProfile = { ...profile };
        
        if (!profile.id && data.id) {
          updatedProfile.id = data.id;
          setProfile(updatedProfile);
          console.log(`New profile created with ID: ${data.id}`);
        }
        
        // Notify parent component about the updated profile
        if (onProfileUpdate) {
          onProfileUpdate(updatedProfile);
        }
        
        // Close modal after successful save
        setTimeout(() => {
          onClose();
        }, 1500);
      } else {
        const data = await response.json().catch(() => null);
        const errorMessage = data?.error || data?.message || 'Failed to save profile';
        console.error('Profile save error:', errorMessage);
        setError(errorMessage);
      }
    } catch (err) {
      console.error('Error saving profile:', err);
      setError('Error saving profile');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTextChange = (field: keyof RecruiterProfile) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setProfile(prev => ({
      ...prev,
      [field]: e.target.value
    }));
  };

  const handleSelectChange = (field: keyof RecruiterProfile) => (
    e: SelectChangeEvent
  ) => {
    setProfile(prev => ({
      ...prev,
      [field]: e.target.value
    }));
  };

  if (!isBackendConnected) {
    return (
      <Modal
        open={open}
        onClose={onClose}
        closeAfterTransition
        slots={{ backdrop: Backdrop }}
        slotProps={{
          backdrop: {
            timeout: 500,
            sx: { bgcolor: 'rgba(0, 0, 0, 0.5)' }
          },
        }}
      >
        <Box sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 'auto',
          maxWidth: 400,
          bgcolor: 'background.paper',
          boxShadow: theme.shadows[24],
          p: 4,
          borderRadius: 1,
        }}>
          <Box sx={{ position: 'absolute', top: 10, right: 10 }}>
            <IconButton 
              onClick={onClose} 
              size="small"
              aria-label="Close"
              sx={{ 
                color: 'text.secondary',
                '&:hover': {
                  color: 'text.primary',
                },
                borderRadius: 0,
              }}
            >
              <span style={{ 
                fontSize: '24px', 
                lineHeight: '1',
                display: 'flex',
                alignItems: 'center'
              }}>×</span>
            </IconButton>
          </Box>
          <Typography variant="h6" gutterBottom color="text.primary">
            Cannot Access Profile
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 3 }}>
            The backend server is not available. Please try again later.
          </Typography>
        </Box>
      </Modal>
    );
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      closeAfterTransition
      slots={{ backdrop: Backdrop }}
      slotProps={{
        backdrop: {
          timeout: 500,
          sx: { bgcolor: 'rgba(0, 0, 0, 0.5)' }
        },
      }}
    >
      <Box sx={modalStyle}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5" component="h2" color="text.primary">
            Recruiter Profile
          </Typography>
          <IconButton 
            onClick={onClose} 
            size="small"
            aria-label="Close"
            sx={{ 
              color: 'text.secondary',
              '&:hover': {
                color: 'text.primary',
              },
              borderRadius: 0
            }}
          >
            <span style={{ 
              fontSize: '24px', 
              lineHeight: '1',
              display: 'flex',
              alignItems: 'center'
            }}>×</span>
          </IconButton>
        </Box>

        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <form onSubmit={handleSubmit}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <Box sx={{ display: 'flex', gap: 3 }}>
                <Box sx={{ flex: 1 }}>
                  <TextField
                    required
                    fullWidth
                    label="Email"
                    value={profile.email}
                    onChange={handleTextChange('email')}
                  />
                </Box>
                
                <Box sx={{ flex: 1 }}>
                  <TextField
                    required
                    fullWidth
                    label="Name"
                    value={profile.name}
                    onChange={handleTextChange('name')}
                  />
                </Box>
              </Box>
              
              <Box sx={{ display: 'flex', gap: 3 }}>
                <Box sx={{ flex: 1 }}>
                  <TextField
                    required
                    fullWidth
                    label="Company"
                    value={profile.company}
                    onChange={handleTextChange('company')}
                  />
                </Box>
                
                <Box sx={{ flex: 1 }}>
                  <TextField
                    required
                    fullWidth
                    label="Role"
                    value={profile.role}
                    onChange={handleTextChange('role')}
                  />
                </Box>
              </Box>
              
              <Box>
                <TextField
                  fullWidth
                  multiline
                  minRows={3}
                  maxRows={15}
                  label="Company Description"
                  value={profile.company_description}
                  onChange={handleTextChange('company_description')}
                  InputProps={{
                    sx: {
                      height: 'auto',
                      overflow: 'auto',
                      '& .MuiInputBase-input': {
                        overflow: 'auto',
                        resize: 'vertical'
                      }
                    }
                  }}
                />
              </Box>
              
              <Box sx={{ display: 'flex', gap: 3 }}>
                <Box sx={{ flex: 1 }}>
                  <FormControl fullWidth>
                    <InputLabel>Industry</InputLabel>
                    <Select
                      value={profile.industry || ''}
                      label="Industry"
                      onChange={handleSelectChange('industry')}
                    >
                      {INDUSTRIES.map(industry => (
                        <MenuItem key={industry} value={industry}>
                          {industry}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>
                
                <Box sx={{ flex: 1 }}>
                  <FormControl fullWidth>
                    <InputLabel>Company Size</InputLabel>
                    <Select
                      value={profile.company_size || ''}
                      label="Company Size"
                      onChange={handleSelectChange('company_size')}
                    >
                      {COMPANY_SIZES.map(size => (
                        <MenuItem key={size} value={size}>
                          {size} employees
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>
              </Box>
              
              <Box>
                <TextField
                  fullWidth
                  label="Recruiting Focus"
                  placeholder="e.g., Engineering, Product, Design"
                  value={profile.recruiting_focus}
                  onChange={handleTextChange('recruiting_focus')}
                />
              </Box>
            </Box>

            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
              <Button
                type="submit"
                variant="contained"
                disabled={isLoading}
              >
                {isLoading ? 'Saving...' : 'Save Profile'}
              </Button>
            </Box>
          </form>
        )}

        <Snackbar
          open={!!error}
          autoHideDuration={6000}
          onClose={() => setError(null)}
        >
          <Alert severity="error" onClose={() => setError(null)} variant="filled">
            {error}
          </Alert>
        </Snackbar>

        <Snackbar
          open={!!success}
          autoHideDuration={6000}
          onClose={() => setSuccess(null)}
        >
          <Alert severity="success" onClose={() => setSuccess(null)} variant="filled">
            {success}
          </Alert>
        </Snackbar>
      </Box>
    </Modal>
  );
};

export default RecruiterProfileModal; 