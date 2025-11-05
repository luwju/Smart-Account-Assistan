// In your types.ts - UPDATE THIS
export interface BotResponse {
  type: string;           // 'message', 'menu', 'options'
  text: string;
  options?: ChatOption[];
  source?: string;        // 'knowledge_base', 'ml_model', etc.
  ml_confidence?: number;
  detected_intent?: string;
  current_menu?: string;
  // Remove the old nested structure
  // session_id?: string;    // ⛔ Remove this
  // current_state?: any;    // ⛔ Remove this  
  // bot_response?: any;     // ⛔ Remove this
}

export interface ChatMessage {
  text: string;
  isUser: boolean;
  type: string;
  options?: ChatOption[];
}

export interface ChatOption {
  value: string;
  label: string;
  menu?: string;  // Add if your backend uses this
}