'use client';
import { useState, useRef, useEffect } from 'react';
import { ChatMessage, ChatOption } from '../../lib/types';
import { chatAPI } from '../../lib/api';
import MessageBubble from './MessageBubble';
import OptionButtons from './OptionButtons';

export default function ChatInterface({ onClose }: { onClose: () => void }) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      text: "Hi, welcome to Coop Bank!",
      isUser: false,
      type: 'options',
      options: [
        { value: 'ind', label: 'Individual (Ind)' },
        { value: 'join', label: 'Joint (Join)' },
        { value: 'cmp', label: 'Company (Cmp)' },
        { value: 'other', label: 'Other' }
      ]
    }
  ]);
  
  const [sessionId, setSessionId] = useState<string>('');
  const [currentState, setCurrentState] = useState<any>({});
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleOptionSelect = async (option: ChatOption) => {
    await handleSendMessage(option.value);
  };

  const handleSendMessage = async (message: string) => {
    if (!message.trim()) return;

    // Add user message to UI immediately
    const userMessage: ChatMessage = {
      text: message,
      isUser: true,
      type: 'message'
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Send to backend
      const response = await chatAPI.sendMessage(message, sessionId, currentState);
      
      setSessionId(response.session_id);
      setCurrentState(response.current_state);

      // Add bot response to UI
      const botMessage: ChatMessage = {
        text: response.bot_response.text,
        isUser: false,
        type: response.bot_response.type as 'message' | 'options',
        options: response.bot_response.options
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = {
        text: "Sorry, I'm having trouble connecting. Please try again.",
        isUser: false,
        type: 'message'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg">
      {/* Chat Header */}
      <div className="bg-green-600 text-white p-4 rounded-t-lg flex justify-between items-center">
        <div>
          <h3 className="font-bold text-lg">Coop Bank Assistant</h3>
          <p className="text-green-100 text-sm">We're here to help</p>
        </div>
        <button
          onClick={onClose}
          className="text-white hover:text-green-200 transition-colors text-xl"
          aria-label="Close chat"
        >
          ✕
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.map((message, index) => (
          <div key={index}>
            <MessageBubble message={message} />
            {message.options && !message.isUser && (
              <OptionButtons 
                options={message.options} 
                onSelect={handleOptionSelect}
                disabled={isLoading}
              />
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}