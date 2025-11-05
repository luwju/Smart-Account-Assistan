'use client';
import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, ChatOption } from '../../lib/types';
import { chatAPI } from '../../lib/api';
import MessageBubble from './MessageBubble';
import OptionButtons from './OptionButtons';

// 🎲 Random helper
const randomPick = (arr: string[]) => arr[Math.floor(Math.random() * arr.length)];

export default function ChatInterface({ onClose }: { onClose: () => void }) {
  // 🌟 Pre-defined messages
  const greetings = [
    "Hi there! 👋 Welcome to Coop Bank.",
    "Hello! 😊 How can I assist you today?",
    "Welcome to Coop Bank! 💼 Ready to open your account?",
    "Hey! 👋 Great to see you! Would you like to open an account?",
    "Good day! 🌟 How can Coop Bank help you today?"
  ];

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      text: randomPick(greetings),
      isUser: false,
      type: 'options',
      options: [
        { value: 'yes', label: 'Yes' },
        { value: 'no', label: 'No (Exit)' }
      ]
    }
  ]);

  const [sessionId, setSessionId] = useState<string>('');
  const [currentState, setCurrentState] = useState<any>({});
  const [isLoading, setIsLoading] = useState(false);
  const [showTextInput, setShowTextInput] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => scrollToBottom(), [messages]);
  useEffect(() => { if (showTextInput && inputRef.current) inputRef.current.focus(); }, [showTextInput]);

  // ✨ Handle option selection - FIXED
  const handleOptionSelect = async (option: ChatOption) => {
    const userMessage: ChatMessage = { text: option.label, isUser: true, type: 'message' };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await chatAPI.sendMessage(option.value, sessionId, currentState);
      
      console.log('🔧 Full backend response:', response); // Debug log
      
      // ✅ FIX: Handle "enable_text_input" option
      if (option.value === 'enable_text_input') {
        setShowTextInput(true);
        setIsLoading(false);
        return;
      }
      
      // ✅ FIX: Use nested bot_response structure
      const botMessage: ChatMessage = {
        text: response.bot_response.text,           // ✅ Nested access
        isUser: false,
        type: response.bot_response.type,           // ✅ Nested access
        options: response.bot_response.options      // ✅ Nested access
      };

      setMessages(prev => [...prev, botMessage]);
      
      // Update session and state
      if (response.session_id) {
        setSessionId(response.session_id);
      }
      if (response.current_state) {
        setCurrentState(response.current_state);
      }
      
    } catch (err) {
      console.error('❌ Chat error:', err);
      setMessages(prev => [
        ...prev,
        { text: "Sorry, something went wrong 😔", isUser: false, type: 'message' }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // ✉️ Send free text - FIXED
  const handleSendMessage = async (message: string) => {
    if (!message.trim()) return;
    const userMessage: ChatMessage = { text: message, isUser: true, type: 'message' };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setShowTextInput(false);
    setInputValue('');

    try {
      const response = await chatAPI.sendMessage(message, sessionId, currentState);
      
      console.log('🔧 Full backend response:', response); // Debug log
      
      // ✅ FIX: Use nested bot_response structure
      const botMessage: ChatMessage = {
        text: response.bot_response.text,           // ✅ Nested access
        isUser: false,
        type: response.bot_response.type,           // ✅ Nested access
        options: response.bot_response.options      // ✅ Nested access
      };

      setMessages(prev => [...prev, botMessage]);
      
      // Update session and state
      if (response.session_id) {
        setSessionId(response.session_id);
      }
      if (response.current_state) {
        setCurrentState(response.current_state);
      }
      
    } catch (err) {
      console.error('❌ Chat error:', err);
      setMessages(prev => [
        ...prev,
        { text: "Sorry, something went wrong 😔", isUser: false, type: 'message' }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTextInputSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) handleSendMessage(inputValue.trim());
  };

  // Add a manual text input button
  const showManualTextInput = () => {
    setShowTextInput(true);
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg">
      {/* Header */}
      <div className="bg-green-600 text-white p-4 rounded-t-lg flex justify-between items-center">
        <div>
          <h3 className="font-bold text-lg">Coop Bank Assistant</h3>
          <p className="text-green-100 text-sm">We're here to help</p>
        </div>
        <button onClick={onClose} className="text-white hover:text-green-200 text-xl" aria-label="Close chat">✕</button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.map((message, index) => (
          <div key={index}>
            <MessageBubble message={message} />
            {message.options && !message.isUser && (
              <OptionButtons options={message.options} onSelect={handleOptionSelect} disabled={isLoading} />
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start mt-2">
            <div className="bg-white rounded-lg p-3 border border-gray-200 shadow-sm">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-300"></div>
              </div>
            </div>
          </div>
        )}

        {/* Manual text input button */}
        {!showTextInput && (
          <div className="flex justify-start mt-2">
            <button
              onClick={showManualTextInput}
              className="bg-blue-500 text-white rounded-lg px-4 py-2 text-sm hover:bg-blue-600"
            >
              💬 Type a question
            </button>
          </div>
        )}

        {showTextInput && (
          <form onSubmit={handleTextInputSubmit} className="mt-4">
            <div className="flex space-x-2">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                className="flex-1 border border-gray-300 rounded-lg p-2"
                placeholder="Type your message..."
                autoFocus
              />
              <button type="submit" className="bg-green-600 text-white rounded-lg px-4">Send</button>
              <button 
                type="button" 
                onClick={() => setShowTextInput(false)}
                className="bg-gray-500 text-white rounded-lg px-4"
              >
                Cancel
              </button>
            </div>
          </form>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}