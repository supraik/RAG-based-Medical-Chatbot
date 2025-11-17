// ===== FILE: components/ChatInterface.tsx ====
'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { haleAPI, ChatMessage } from '@/lib/api';

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: "Hello! I'm Dr. Hale, your AI medical assistant. How can I help you today?",
      timestamp: new Date(),
    },
  ]);

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage]);

  // ===========================
  // Send a Chat Message (Stream)
  // ===========================
  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setIsStreaming(true);
    setStreamingMessage('');

    let fullAnswer = ''; // ← Fix streaming issue

    try {
      await haleAPI.sendMessageStream(
        userMessage.content,

        // Token callback
        (token) => {
          fullAnswer += token;
          setStreamingMessage((prev) => prev + token);
        },

        // Complete callback
        (data) => {
          const assistant: ChatMessage = {
            role: 'assistant',
            content: fullAnswer,
            timestamp: new Date(),
          };

          setMessages((prev) => [...prev, assistant]);

          setStreamingMessage('');
          setIsStreaming(false);
          setIsLoading(false);
        },

        // Error callback
        (error) => {
          console.error('Streaming error:', error);
          setStreamingMessage('');
          setIsStreaming(false);
          setIsLoading(false);
        }
      );
    } catch (err) {
      console.error('Send error:', err);
      setStreamingMessage('');
      setIsStreaming(false);
      setIsLoading(false);
    }
  };

  // Enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((msg, i) => (
          <ChatBubble key={i} message={msg} />
        ))}

        {/* Live Streaming Bubble */}
        {isStreaming && streamingMessage && (
          <ChatBubble
            message={{
              role: 'assistant',
              content: streamingMessage,
              timestamp: new Date(),
            }}
          />
        )}

        {/* Typing Indicator */}
        {isLoading && !streamingMessage && (
          <div className="flex gap-2 items-center">
            <div className="bg-gray-800 rounded-2xl px-6 py-4">
              <div className="flex gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce"></div>
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Input Box */}
      <div className="p-6 border-t border-gray-800">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a medical question..."
            disabled={isLoading}
            className="flex-1 px-4 py-3 rounded-xl bg-gray-900 text-white border border-gray-700 focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          />

          <button
            onClick={handleSendMessage}
            disabled={!input.trim() || isLoading}
            className="px-6 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 text-white flex items-center gap-2 hover:scale-105 transition disabled:opacity-50"
          >
            {isLoading ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
          </button>
        </div>
      </div>
    </div>
  );
}

// =====================
// Chat Bubble Component
// =====================
function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[70%] rounded-2xl px-6 py-4 ${
          isUser
            ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white ml-auto'
            : 'bg-gray-800 text-white border border-gray-700'
        } shadow-lg`}
      >
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        <p className="text-xs mt-2 text-gray-400">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>
    </div>
  );
}
