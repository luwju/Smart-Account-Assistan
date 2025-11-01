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
  const [showTextInput, setShowTextInput] = useState(false); // ADD THIS LINE
  const [inputValue, setInputValue] = useState(''); // ADD THIS LINE
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null); // ADD THIS LINE

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Focus on input when it appears
    if (showTextInput && inputRef.current) {
      inputRef.current.focus();
    }
  }, [showTextInput]); // ADD THIS EFFECT

  const handleOptionSelect = async (option: ChatOption) => {
    if (option.value === 'other') {
      // Show text input for "Other" option
      setShowTextInput(true);
      // Add a message prompting user to type
      const promptMessage: ChatMessage = {
        text: "Please type what you are looking for:",
        isUser: false,
        type: 'message'
      };
      setMessages(prev => [...prev, promptMessage]);
    } else {
      await handleSendMessage(option.value);
    }
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
    setShowTextInput(false); // Hide input after sending
    setInputValue(''); // Clear input

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

  // ADD THIS FUNCTION FOR TEXT INPUT SUBMISSION
  const handleTextInputSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      handleSendMessage(inputValue.trim());
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
        
        {/* ADD TEXT INPUT FIELD */}
        {showTextInput && (
          <form onSubmit={handleTextInputSubmit} className="mt-4">
            <div className="flex space-x-2">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Type your question here..."
                className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isLoading}
                className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Send
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Press Enter or click Send to submit your question
            </p>
          </form>
        )}
        
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