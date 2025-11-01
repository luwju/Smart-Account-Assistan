'use client';
import { useState } from 'react';
import ChatInterface from './ChatInterface';
import ChatIcon from '../icons/ChatIcon';

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Chat Interface Popup */}
      {isOpen && (
        <div className="absolute bottom-20 right-0 w-96 h-[600px] bg-white rounded-lg shadow-2xl border border-gray-200 animate-fade-in">
          <ChatInterface onClose={() => setIsOpen(false)} />
        </div>
      )}
      
      {/* Floating Chat Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-14 h-14 bg-green-600 hover:bg-green-700 rounded-full flex items-center justify-center shadow-lg transition-all duration-300 hover:scale-110"
        aria-label="Open chat"
      >
        <ChatIcon className="w-6 h-6 text-white" />
      </button>
    </div>
  );
}