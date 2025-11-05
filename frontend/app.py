"""
HaleAI Medical Chatbot – Gradio Frontend
Professional RAG-powered medical assistant with safety, sources, and export.
"""
import gradio as gr
import json
import time
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# --------------------------------------------------------------------- #
# Safe Document fallback (avoids langchain version issues)
# --------------------------------------------------------------------- #
try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content: str = "", metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# --------------------------------------------------------------------- #
# Add project root to path
# --------------------------------------------------------------------- #
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Custom CSS & JS
# --------------------------------------------------------------------- #
custom_css = """
:root {
    --primary-color: #2563eb;
    --secondary-color: #3b82f6;
    --accent-color: #60a5fa;
    --background: #f8fafc;
    --surface: #ffffff;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --border: #e2e8f0;
}

.dark {
    --primary-color: #3b82f6;
    --secondary-color: #2563eb;
    --background: #0f172a;
    --surface: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #334155;
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    background: var(--background) !important;
}

.header-container {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}

.header-title {
    color: white;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.chatbot {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05) !important;
    background: var(--surface) !important;
}

.message {
    padding: 1rem !important;
    margin: 0.5rem 0 !important;
    border-radius: 12px !important;
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.message.user {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
    margin-left: auto !important;
}

.message.bot {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
}

.input-container textarea {
    border-radius: 12px !important;
    border: 2px solid var(--border) !important;
    padding: 1rem !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.input-container textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

.primary-btn {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
}

.secondary-btn {
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 2px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.secondary-btn:hover {
    background: var(--background) !important;
    border-color: var(--primary-color) !important;
}

.settings-panel {
    background: var(--surface);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid var(--border);
    margin-top: 1rem;
}

.accordion {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 0.5rem !important;
}

.source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.source-card:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

.loading-indicator {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--border);
    border-top-color: var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    animation: slideInRight 0.3s ease-out;
    z-index: 1000;
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(100%); }
    to { opacity: 1; transform: translateX(0); }
}

@media (max-width: 768px) {
    .gradio-container { padding: 1rem !important; }
    .header-title { font-size: 1.5rem; }
    .message { padding: 0.75rem !important; }
}

.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
}

@media (prefers-contrast: high) {
    .message { border: 2px solid currentColor !important; }
}

@media (prefers-reduced-motion: reduce) {
    * { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
}
"""

custom_js = """
function() {
    const chatbot = document.querySelector('.chatbot');
    if (chatbot) {
        chatbot.scrollTop = chatbot.scrollHeight;
    }
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const sendBtn = document.querySelector('.primary-btn');
            if (sendBtn) sendBtn.click();
        }
    });
    const theme = localStorage.getItem('theme') || 'light';
    document.documentElement.classList.toggle('dark', theme === 'dark');
}
"""

# --------------------------------------------------------------------- #
# HaleAIChatbot – Safe, Robust, and User-Friendly
# --------------------------------------------------------------------- #
class HaleAIChatbot:
    def __init__(self):
        from chatbot import HaleAI  # <-- your HaleAI class

        self.conversation_history = []
        self.model_name = "Gemini-1.5-Pro"
        self.system_status = "Initializing"
        self.session_id = f"session_{int(time.time())}"
        self.backend: Optional[HaleAI] = None

        # --- Safe backend init ---
        try:
            logger.info("Initializing HaleAI backend...")
            self.backend = HaleAI()
            self.backend.connect()
            self.system_status = "Connected"
            logger.info("HaleAI backend ready")
        except Exception as e:
            logger.error("Backend init failed: %s", e, exc_info=True)
            print(f"Backend Error: {e}")
            import traceback
            traceback.print_exc()
            self.system_status = "Error"

    def format_timestamp(self) -> str:
        return datetime.now().strftime("%I:%M %p")

    def get_system_info(self) -> Dict:
        return {
            "model": self.model_name,
            "status": self.system_status,
            "vector_store": "Pinecone" if self.backend else "Unavailable",
            "embedding_model": "Gemini Embedding",
            "session_id": self.session_id
        }

    def format_sources(self, source_docs: List[Document]) -> str:
        if not source_docs:
            return "No sources found in knowledge base."
        sources = "### Reference Sources\n\n"
        for i, doc in enumerate(source_docs, 1):
            content = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
            sources += f"**Source {i}**\n"
            sources += f"- Content: {content}\n"
            if doc.metadata:
                for k, v in doc.metadata.items():
                    sources += f"- {k}: {v}\n"
            sources += "\n"
        return sources

    def process_message(
        self,
        message: str,
        history: List[Tuple[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Tuple[List[Tuple[str, str]], str, str]:
        if not message.strip():
            return history, "", ""

        # --- Backend unavailable ---
        if self.backend is None:
            err = (
                "Backend unavailable. "
                "Check API key, Pinecone index, and console for errors."
            )
            history.append((message, err))
            return history, "Error", f"Status: {self.system_status}"

        # --- Normal RAG query ---
        try:
            response = self.backend.query(message, history=self.conversation_history)
            answer = response["answer"]
            sources = self.format_sources(response["sources"])

            metadata = f"""
**Response Metadata**
- Timestamp: {self.format_timestamp()}
- Model: {self.model_name}
- Sources: {response["num_sources"]}
- Status: {response["status"]}
            """.strip()

            history.append((message, answer))
            self.conversation_history.append({
                "user": message,
                "bot": answer,
                "timestamp": self.format_timestamp(),
                "sources": sources
            })

            return history, sources, metadata

        except Exception as e:
            import traceback
            traceback.print_exc()
            err = f"Error: {e}"
            history.append((message, err))
            return history, "Error", f"Error: {e}"

    def export_conversation(self, format_type: str = "json") -> str:
        if format_type == "json":
            return json.dumps(self.conversation_history, indent=2, ensure_ascii=False)
        elif format_type == "txt":
            text = f"HaleAI Session: {self.session_id}\n\n"
            for msg in self.conversation_history:
                text += f"[{msg['timestamp']}] You: {msg['user']}\n"
                text += f"[{msg['timestamp']}] Dr. Hale: {msg['bot']}\n\n"
            return text
        return ""

    def clear_history(self):
        self.conversation_history = []
        return [], "", ""

# --------------------------------------------------------------------- #
# Gradio Interface
# --------------------------------------------------------------------- #
def create_interface():
    chatbot = HaleAIChatbot()

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"),
        css=custom_css,
        js=custom_js,
        title="HaleAI Medical Chatbot",
        analytics_enabled=False
    ) as demo:

        gr.Markdown(
            """
            <div class="header-title">
                HaleAI Medical Chatbot
                <span class="status-indicator"></span>
            </div>
            <p style="color: rgba(255,255,255,0.9); margin: 0;">
                AI-powered medical information assistant | Powered by Gemini + RAG
            </p>
            """,
            elem_classes="header-container"
        )

        with gr.Row():
            with gr.Column(scale=7):
                chat = gr.Chatbot(
                    height=600,
                    show_label=False,
                    elem_classes="chatbot",
                    avatar_images=(None, "Doctor")
                )
                with gr.Row():
                    with gr.Column(scale=9):
                        msg = gr.Textbox(
                            placeholder="Ask a medical question... (Enter to send)",
                            show_label=False,
                            lines=2,
                            elem_classes="input-container"
                        )
                    with gr.Column(scale=1):
                        send = gr.Button("Send", variant="primary", elem_classes="primary-btn")

                with gr.Row():
                    clear = gr.Button("Clear", size="sm", elem_classes="secondary-btn")
                    export_json = gr.Button("JSON", size="sm", elem_classes="secondary-btn")
                    export_txt = gr.Button("TXT", size="sm", elem_classes="secondary-btn")

            with gr.Column(scale=3):
                with gr.Accordion("Settings", open=False):
                    gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.7, step=0.1)
                    gr.Slider(label="Max Tokens", minimum=128, maximum=2048, value=512, step=128)

                with gr.Accordion("Sources", open=True):
                    sources_out = gr.Markdown("Ask a question to see sources.")

                with gr.Accordion("Metadata", open=False):
                    meta_out = gr.Markdown()

                with gr.Accordion("Help", open=False):
                    gr.Markdown("""
                    ### Tips
                    - Be specific
                    - Ask follow-ups
                    - Always consult a doctor
                    """)

        gr.Markdown("""
        ---
        **Medical Disclaimer**: This tool provides **general information only**. 
        It is **not medical advice**. Always consult a qualified healthcare provider.
        """)

        export_file = gr.File(visible=False)

        # --- Events ---
        def respond(m, h, t, tk):
            return chatbot.process_message(m, h, t, tk)

        msg.submit(respond, [msg, chat, gr.Slider(), gr.Slider()], [chat, sources_out, meta_out]).then(lambda: "", None, msg)
        send.click(respond, [msg, chat, gr.Slider(), gr.Slider()], [chat, sources_out, meta_out]).then(lambda: "", None, msg)

        clear.click(chatbot.clear_history, None, [chat, sources_out, meta_out])
        export_json.click(lambda: chatbot.export_conversation("json"), None, export_file)
        export_txt.click(lambda: chatbot.export_conversation("txt"), None, export_file)

    return demo

# --------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None
    )