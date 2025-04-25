// API Configuration
export const API_BASE_URL = 'http://localhost:5000';

export const API_ENDPOINTS = {
  CHAT: `${API_BASE_URL}/api/chat`,
  CLEAR_HISTORY: `${API_BASE_URL}/api/clear_history`,
  HEALTH: `${API_BASE_URL}/api/health`,
  RECRUITER_PROFILE: `${API_BASE_URL}/api/recruiter/profile`
};

export default API_ENDPOINTS; 