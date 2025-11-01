import { BotResponse } from './types';

const API_BASE_URL = 'http://localhost:8000/api';

export const chatAPI = {
  sendMessage: async (message: string, sessionId?: string, currentState?: any): Promise<BotResponse> => {
    const response = await fetch(`${API_BASE_URL}/chat/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        current_state: currentState || {}
      }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to send message');
    }
    
    return response.json();
  },
};