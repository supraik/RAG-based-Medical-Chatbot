'use client';

import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Send, Settings, Activity, Zap, Clock, Brain, Moon, Sun, Menu, X, TrendingUp, MessageSquare, BarChart3, Sparkles, Loader2, Download } from 'lucide-react';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// API Client
class HaleAIAPI {
  private baseURL: string;
  private sessionId: string | null = null;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async sendMessage(message: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        session_id: this.sessionId,
      }),
    });

    if (!response.ok) throw new Error(`API error: ${response.statusText}`);
    const data = await response.json();
    this.sessionId = data.session_id;
    return data;
  }

  async sendMessageStream(
    message: string,
    onToken: (token: string) => void,
    onComplete: (data: any) => void,
    onError: (error: string) => void
  ): Promise<void> {
    try {
      const response = await fetch(`${this.baseURL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: this.sessionId }),
      });

      if (!response.ok) throw new Error(`Stream error: ${response.statusText}`);

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error('No reader available');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'token') onToken(data.content);
            else if (data.type === 'complete') {
              this.sessionId = data.session_id;
              onComplete(data);
            } else if (data.type === 'error') onError(data.content);
          }
        }
      }
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Unknown error');
    }
  }

  async getAnalytics(): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/analytics`);
    if (!response.ok) throw new Error('Failed to fetch analytics');
    return response.json();
  }

  async exportConversation(format: 'json' | 'txt' = 'json'): Promise<any> {
    if (!this.sessionId) throw new Error('No active session');
    const response = await fetch(
      `${this.baseURL}/api/conversations/${this.sessionId}/export?format=${format}`
    );
    if (!response.ok) throw new Error('Failed to export');
    return response.json();
  }
}

const api = new HaleAIAPI();

const HaleAIApp = () => {
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m Dr. Hale, your AI medical assistant. How can I help you today?', timestamp: new Date() }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [sources, setSources] = useState<Array<{page: string, content: string}>>([]);
  const [analytics, setAnalytics] = useState({
    accuracy: 94.2,
    avg_latency: 245,
    total_queries: 1247,
    active_users: 23,
    recent_accuracy: [],
    recent_latency: [],
    weekly_queries: []
  });
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage]);

  useEffect(() => {
    // Load analytics periodically
    const loadAnalytics = async () => {
      try {
        const data = await api.getAnalytics();
        setAnalytics(data);
      } catch (err) {
        console.error('Failed to load analytics:', err);
      }
    };
    
    loadAnalytics();
    const interval = setInterval(loadAnalytics, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setIsStreaming(true);
    setStreamingMessage('');

    try {
      await api.sendMessageStream(
        userMessage.content,
        (token) => setStreamingMessage(prev => prev + token),
        (data) => {
          const assistantMessage = {
            role: 'assistant',
            content: streamingMessage,
            timestamp: new Date()
          };
          setMessages(prev => [...prev, assistantMessage]);
          setSources(data.sources || []);
          setStreamingMessage('');
          setIsStreaming(false);
          setIsLoading(false);
        },
        (error) => {
          console.error('Stream error:', error);
          const errorMessage = {
            role: 'assistant',
            content: `Sorry, I encountered an error: ${error}. Please make sure the backend is running on http://localhost:8000`,
            timestamp: new Date()
          };
          setMessages(prev => [...prev, errorMessage]);
          setIsStreaming(false);
          setIsLoading(false);
          setStreamingMessage('');
        }
      );
    } catch (error) {
      console.error('Error:', error);
      setIsLoading(false);
      setIsStreaming(false);
    }
  };

  const handleExport = async (format: 'json' | 'txt') => {
    try {
      const data = await api.exportConversation(format);
      const blob = new Blob([format === 'json' ? JSON.stringify(data, null, 2) : data.content], 
        { type: format === 'json' ? 'application/json' : 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `haleai-conversation.${format}`;
      a.click();
    } catch (err) {
      console.error('Export failed:', err);
    }
  };

  const StatCard = ({ icon: Icon, title, value, trend, color }: {
    icon: React.ComponentType<{ size?: number; className?: string }>;
    title: string;
    value: string | number;
    trend?: string;
    color: string;
  }) => (
    <div className={`${darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200'} backdrop-blur-sm border rounded-2xl p-6 transition-all duration-300 hover:scale-105 hover:shadow-2xl group`}>
      <div className="flex items-start justify-between">
        <div>
          <p className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-600'} mb-1`}>{title}</p>
          <p className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{value}</p>
          {trend && (
            <p className="text-sm text-green-500 mt-2 flex items-center gap-1">
              <TrendingUp size={16} />
              {trend}
            </p>
          )}
        </div>
        <div className={`p-3 rounded-xl ${color} bg-opacity-20 group-hover:scale-110 transition-transform duration-300`}>
          <Icon className={`${color.replace('bg-', 'text-')} opacity-80`} size={24} />
        </div>
      </div>
    </div>
  );

  const ChatBubble = ({ message }: { message: { role: string; content: string; timestamp: Date } }) => {
    const isUser = message.role === 'user';
    return (
      <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-slideIn`}>
        <div className={`max-w-[70%] rounded-2xl px-6 py-4 ${
          isUser 
            ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white ml-auto' 
            : darkMode 
              ? 'bg-gray-800/80 text-white border border-gray-700' 
              : 'bg-gray-100 text-gray-900 border border-gray-200'
        } backdrop-blur-sm shadow-lg`}>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
          <p className={`text-xs mt-2 ${isUser ? 'text-blue-100' : darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white' : 'bg-gradient-to-br from-gray-50 via-white to-gray-50 text-gray-900'} transition-colors duration-500`}>
      <style jsx>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-slideIn {
          animation: slideIn 0.3s ease-out;
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        .animate-pulse-slow {
          animation: pulse 2s infinite;
        }
      `}</style>

      {/* Header */}
      <header className={`${darkMode ? 'bg-gray-900/50 border-gray-800' : 'bg-white/50 border-gray-200'} backdrop-blur-md border-b sticky top-0 z-50`}>
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="lg:hidden">
              {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center">
                <Sparkles className="text-white" size={20} />
              </div>
              <div>
                <h1 className="text-xl font-bold">HaleAI</h1>
                <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Medical Intelligence</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/20 border border-green-500/30">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse-slow"></div>
              <span className="text-sm text-green-500 font-medium">Online</span>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-xl ${darkMode ? 'bg-gray-800 hover:bg-gray-700' : 'bg-gray-100 hover:bg-gray-200'} transition-all duration-300`}
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex h-[calc(100vh-73px)]">
        {/* Sidebar */}
        <aside className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 fixed lg:static w-64 h-full ${darkMode ? 'bg-gray-900/50 border-gray-800' : 'bg-white/50 border-gray-200'} backdrop-blur-md border-r transition-transform duration-300 z-40`}>
          <nav className="p-4 space-y-2">
            {[
              { id: 'chat', icon: MessageSquare, label: 'Chat' },
              { id: 'dashboard', icon: BarChart3, label: 'Analytics' },
              { id: 'settings', icon: Settings, label: 'Settings' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => { setActiveTab(tab.id); setSidebarOpen(false); }}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                    : darkMode
                      ? 'hover:bg-gray-800 text-gray-400 hover:text-white'
                      : 'hover:bg-gray-100 text-gray-600 hover:text-gray-900'
                }`}
              >
                <tab.icon size={20} />
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </nav>
        </aside>

        {/* Main Area */}
        <main className="flex-1 overflow-auto">
          {activeTab === 'chat' && (
            <div className="max-w-5xl mx-auto h-full flex flex-col p-6">
              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto mb-6 space-y-4">
                {messages.map((msg, i) => (
                  <ChatBubble key={i} message={msg} />
                ))}
                
                {isStreaming && streamingMessage && (
                  <ChatBubble message={{ role: 'assistant', content: streamingMessage, timestamp: new Date() }} />
                )}

                {isLoading && !streamingMessage && (
                  <div className="flex gap-2 items-center">
                    <div className={`${darkMode ? 'bg-gray-800' : 'bg-gray-200'} rounded-2xl px-6 py-4`}>
                      <div className="flex gap-2">
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    </div>
                  </div>
                )}

                {sources.length > 0 && (
                  <div className={`${darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200'} border rounded-2xl p-4`}>
                    <h4 className="font-semibold mb-2">📚 Sources ({sources.length})</h4>
                    <div className="space-y-2">
                      {sources.map((src, i) => (
                        <div key={i} className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'} p-2 rounded-lg ${darkMode ? 'bg-gray-900/50' : 'bg-gray-50'}`}>
                          <span className="font-medium">Page {src.page}:</span> {src.content}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div ref={chatEndRef} />
              </div>

              {/* Input Area */}
              <div className={`${darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200'} backdrop-blur-sm border rounded-2xl p-4 shadow-2xl`}>
                <div className="flex gap-3 mb-3">
                  <input
                    ref={inputRef}
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                    placeholder="Ask a medical question..."
                    disabled={isLoading}
                    className={`flex-1 px-4 py-3 rounded-xl ${darkMode ? 'bg-gray-900 text-white border-gray-700' : 'bg-gray-50 text-gray-900 border-gray-200'} border focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all disabled:opacity-50`}
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={!input.trim() || isLoading}
                    className="px-6 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium hover:shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 flex items-center gap-2"
                  >
                    {isLoading ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
                  </button>
                </div>
                <div className="flex gap-2">
                  <button onClick={() => handleExport('json')} className="text-xs px-3 py-1 rounded-lg bg-gray-700 hover:bg-gray-600 flex items-center gap-1">
                    <Download size={14} /> Export JSON
                  </button>
                  <button onClick={() => handleExport('txt')} className="text-xs px-3 py-1 rounded-lg bg-gray-700 hover:bg-gray-600 flex items-center gap-1">
                    <Download size={14} /> Export TXT
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'dashboard' && (
            <div className="p-6 space-y-6">
              <h2 className="text-3xl font-bold mb-6">Analytics Dashboard</h2>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard icon={Brain} title="Model Accuracy" value={`${analytics.accuracy.toFixed(1)}%`} trend="+2.3%" color="bg-blue-500" />
                <StatCard icon={Clock} title="Avg Latency" value={`${Math.round(analytics.avg_latency)}ms`} trend="-12ms" color="bg-purple-500" />
                <StatCard icon={Activity} title="Total Queries" value={analytics.total_queries} trend="+156" color="bg-green-500" />
                <StatCard icon={Zap} title="Active Users" value={analytics.active_users} trend="+5" color="bg-orange-500" />
              </div>

              {analytics.recent_accuracy?.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className={`${darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200'} backdrop-blur-sm border rounded-2xl p-6`}>
                    <h3 className="text-lg font-semibold mb-4">Model Accuracy</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <AreaChart data={analytics.recent_accuracy}>
                        <defs>
                          <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#e5e7eb'} />
                        <XAxis dataKey="time" stroke={darkMode ? '#9ca3af' : '#6b7280'} />
                        <YAxis stroke={darkMode ? '#9ca3af' : '#6b7280'} />
                        <Tooltip contentStyle={{ backgroundColor: darkMode ? '#1f2937' : '#fff', border: 'none', borderRadius: '8px' }} />
                        <Area type="monotone" dataKey="value" stroke="#3b82f6" fillOpacity={1} fill="url(#colorAccuracy)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  <div className={`${darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200'} backdrop-blur-sm border rounded-2xl p-6`}>
                    <h3 className="text-lg font-semibold mb-4">Response Latency</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={analytics.recent_latency}>
                        <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#e5e7eb'} />
                        <XAxis dataKey="time" stroke={darkMode ? '#9ca3af' : '#6b7280'} />
                        <YAxis stroke={darkMode ? '#9ca3af' : '#6b7280'} />
                        <Tooltip contentStyle={{ backgroundColor: darkMode ? '#1f2937' : '#fff', border: 'none', borderRadius: '8px' }} />
                        <Line type="monotone" dataKey="value" stroke="#a855f7" strokeWidth={3} dot={{ fill: '#a855f7', r: 4 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="p-6 max-w-4xl mx-auto space-y-6">
              <h2 className="text-3xl font-bold mb-6">Settings</h2>
              
              <div className={`${darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200'} backdrop-blur-sm border rounded-2xl p-6 space-y-6`}>
                <div>
                  <h3 className="text-lg font-semibold mb-4">Backend Configuration</h3>
                  <p className="text-sm text-gray-400 mb-4">
                    Make sure your FastAPI backend is running on <code className="bg-gray-900 px-2 py-1 rounded">http://localhost:8000</code>
                  </p>
                  <div className="space-y-2 text-sm">
                    <p>✅ Chat endpoint: <code>/api/chat</code></p>
                    <p>✅ Streaming: <code>/api/chat/stream</code></p>
                    <p>✅ Analytics: <code>/api/analytics</code></p>
                    <p>✅ Export: <code>/api/conversations/export</code></p>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-4">Quick Start</h3>
                  <div className="bg-gray-900 p-4 rounded-lg text-sm font-mono space-y-2">
                    <p># Terminal 1: Start Backend</p>
                    <p className="text-green-400">cd backend && python backend_api.py</p>
                    <p className="mt-4"># Open this UI</p>
                    <p className="text-blue-400">http://localhost:3000</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default HaleAIApp;