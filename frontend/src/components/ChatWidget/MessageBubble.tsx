import { ChatMessage } from '../../lib/types';

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  return (
    <div className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          message.isUser
            ? 'bg-green-600 text-white rounded-br-none'
            : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none shadow-sm'
        }`}
      >
        <p className="text-sm whitespace-pre-wrap">{message.text}</p>
      </div>
    </div>
  );
}