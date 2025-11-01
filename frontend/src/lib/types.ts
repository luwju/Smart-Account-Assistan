export interface ChatMessage {
  text: string;
  isUser: boolean;
  options?: ChatOption[];
  type: 'message' | 'options';
}

export interface ChatOption {
  value: string;
  label: string;
}

export interface BotResponse {
  session_id: string;
  bot_response: {
    text: string;
    type: string;
    options?: ChatOption[];
  };
  current_state: any;
}