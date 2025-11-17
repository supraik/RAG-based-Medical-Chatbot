// ===== FILE: lib/api.ts =====
// API client for HaleAI (FastAPI backend)

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export interface ChatResponse {
  answer: string;
  sources: Array<{
    content: string;
    page: string;
    source: string;
  }>;
  num_sources: number;
  status: string;
  session_id: string;
  metadata: Record<string, any>;
}

export interface AnalyticsData {
  accuracy: number;
  avg_latency: number;
  total_queries: number;
  active_users: number;
  recent_accuracy: Array<{ time: string; value: number }>;
  recent_latency: Array<{ time: string; value: number }>;
  weekly_queries: Array<{ day: string; count: number }>;
}

export class HaleAIAPI {
  private baseURL: string;
  private sessionId: string | null = null;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/api/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  // Non-streaming request (rarely used)
  async sendMessage(message: string): Promise<ChatResponse> {
    const response = await fetch(`${this.baseURL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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

  // Streaming API using Server-Sent Events over fetch()
  async sendMessageStream(
    message: string,
    onToken: (token: string) => void,
    onComplete: (data: any) => void,
    onError: (error: string) => void
  ): Promise<void> {
    try {
      const response = await fetch(`${this.baseURL}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          session_id: this.sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Stream error: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No readable stream available");
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;

          const data = JSON.parse(line.slice(6));

          if (data.type === "token") {
            onToken(data.content);
          } else if (data.type === "complete") {
            this.sessionId = data.session_id;
            onComplete(data);
          } else if (data.type === "error") {
            onError(data.content);
          }
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      onError(message);
    }
  }

  async getAnalytics(): Promise<AnalyticsData> {
    const response = await fetch(`${this.baseURL}/api/analytics`);
    if (!response.ok) throw new Error("Failed to fetch analytics");
    return response.json();
  }

  async exportConversation(
    sessionId: string,
    format: "json" | "txt" = "json"
  ): Promise<any> {
    const response = await fetch(
      `${this.baseURL}/api/conversations/${sessionId}/export?format=${format}`
    );
    if (!response.ok) throw new Error("Failed to export conversation");
    return response.json();
  }
}

export const haleAPI = new HaleAIAPI();
