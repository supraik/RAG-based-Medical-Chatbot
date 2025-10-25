import gradio as gr
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional
# Avoid importing langchain.schema at module import time to prevent
# version-dependent side effects (some langchain installs require
# langchain_core.* subpackages). Use a safe fallback for typing.
try:
    from langchain.schema import Document
except Exception:
    # Lightweight fallback Document for typing when langchain isn't available
    class Document:  # type: ignore
        def __init__(self, page_content: str = "", metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}
import asyncio
import sys
import os

# Add the parent directory to Python path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Custom CSS for professional medical theme
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

/* Main container styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    background: var(--background) !important;
}

/* Header styling */
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

/* Chat interface styling */
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
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
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

/* Input styling */
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

/* Button styling */
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

/* Settings panel */
.settings-panel {
    background: var(--surface);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid var(--border);
    margin-top: 1rem;
}

/* Accordion styling */
.accordion {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 0.5rem !important;
}

/* Source display */
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

/* Loading indicator */
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

/* Toast notifications */
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
    from {
        opacity: 0;
        transform: translateX(100%);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1rem !important;
    }
    
    .header-title {
        font-size: 1.5rem;
    }
    
    .message {
        padding: 0.75rem !important;
    }
}

/* Accessibility enhancements */
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

/* High contrast mode */
@media (prefers-contrast: high) {
    .message {
        border: 2px solid currentColor !important;
    }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
"""

# Custom JavaScript for enhanced interactions
custom_js = """
function() {
    // Auto-scroll to bottom of chat
    const chatbot = document.querySelector('.chatbot');
    if (chatbot) {
        chatbot.scrollTop = chatbot.scrollHeight;
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const sendBtn = document.querySelector('.primary-btn');
            if (sendBtn) sendBtn.click();
        }
    });
    
    // Theme switcher
    const theme = localStorage.getItem('theme') || 'light';
    document.documentElement.classList.toggle('dark', theme === 'dark');
}
"""

class HaleAIChatbot:
    """HaleAI Medical Chatbot with advanced features"""
    
    def __init__(self):
        from chatbot import HaleAI
        self.conversation_history = []
        self.model_name = "Mixtral-8x7B-Instruct-v0.1"
        self.system_status = "Initializing"
        self.session_id = f"session_{int(time.time())}"
        
        # Initialize the backend
        try:
            self.backend = HaleAI()
            self.backend.connect()  # Connect to existing knowledge base
            self.system_status = "Connected"
        except Exception as e:
            print(f"Error initializing backend: {str(e)}")
            self.system_status = "Error"
        
    def format_timestamp(self) -> str:
        """Format current timestamp"""
        return datetime.now().strftime("%I:%M %p")
    
    def get_system_info(self) -> Dict:
        """Get system status information"""
        return {
            "model": self.model_name,
            "status": self.system_status,
            "vector_store": "Pinecone (Connected)",
            "embedding_model": "sentence-transformers",
            "session_id": self.session_id
        }
    
    def format_sources(self, source_docs: List[Document]) -> str:
        """Format source documents into a readable string"""
        if not source_docs:
            return "No sources found."
        
        sources = "### 📚 Reference Sources\n\n"
        for i, doc in enumerate(source_docs, 1):
            content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            sources += f"**Source {i}**:\n"
            sources += f"- Content: {content}\n"
            if hasattr(doc, 'metadata'):
                for key, value in doc.metadata.items():
                    sources += f"- {key}: {value}\n"
            sources += "\n"
        return sources
    
    def process_message(
        self, 
        message: str, 
        history: List[Tuple[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Tuple[List[Tuple[str, str]], str, str]:
        """
        Process user message and generate response
        Returns: (updated_history, sources, metadata)
        """
        if not message.strip():
            return history, "", ""
        
        try:
            # Get response from backend
            response = self.backend.query(message)
            
            # Extract answer and sources
            answer = response["answer"]
            sources = self.format_sources(response["sources"])
            
            # Generate metadata
            metadata = f"""
**Response Metadata**
- Timestamp: {self.format_timestamp()}
- Model: {self.model_name}
- Sources Found: {response["num_sources"]}
- Status: {response["status"]}
            """
            
            # Update history
            history.append((message, answer))
            self.conversation_history.append({
                "user": message,
                "assistant": answer,
                "timestamp": self.format_timestamp(),
                "sources": sources
            })
            
            return history, sources, metadata
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            history.append((message, error_response))
            return history, "Error retrieving sources", f"Error: {str(e)}"
    
    def _generate_medical_response(self, query: str) -> str:
        """Generate contextual medical response with mandatory disclaimer"""
        
        # Medical disclaimer to append to every response
        MEDICAL_DISCLAIMER = """
---
**⚠️ IMPORTANT MEDICAL DISCLAIMER**

Please consult your physician or qualified healthcare provider before making any health decisions. This information is for educational purposes only and does not constitute medical advice, diagnosis, or treatment. Always seek professional medical guidance for your specific health concerns.

**In case of emergency, call your local emergency number immediately.**
"""
        
        # This is a placeholder - replace with actual LLM integration
        base_responses = {
            "symptom": """Based on the information provided, here are some general insights:

**Important First Steps:**
1. **Schedule an appointment** with your healthcare provider for proper evaluation
2. **Monitor symptoms** and keep a detailed record of when they occur and their severity
3. **Seek immediate medical attention** if symptoms are severe, sudden, or worsening

**General Guidance:**
- Document all symptoms, their duration, and any triggers you notice
- Note any other accompanying symptoms
- Keep track of your temperature and vital signs if possible

**When to Seek Immediate Care:**
- Severe pain or discomfort
- Difficulty breathing
- Loss of consciousness
- Severe bleeding
- Sudden changes in vision or speech""",
            
            "medication": """Regarding medication information:

**⚠️ CRITICAL: Consult Your Healthcare Provider**
- **Never** start, stop, or change any medication without medical supervision
- **Always** inform your doctor about all medications you're taking, including over-the-counter drugs and supplements
- **Report** any side effects or adverse reactions immediately

**General Medication Safety:**
- Follow prescribed dosages exactly as directed
- Read medication labels and information sheets carefully
- Be aware of potential drug interactions
- Store medications properly and check expiration dates
- Never share prescription medications with others""",
            
            "general": """Thank you for reaching out to HaleAI Medical Chatbot.

**What I Can Help With:**
- General health and medical information for educational purposes
- Explaining medical concepts and terminology
- Providing context about health conditions (not diagnosis)
- Offering wellness and preventive care information

**What I Cannot Do:**
- Provide medical diagnosis or treatment
- Replace consultation with healthcare professionals
- Prescribe medications
- Offer emergency medical assistance

**For Best Care:**
Please schedule an appointment with your physician or healthcare provider to discuss your specific health concerns. They can provide personalized medical advice based on your complete health history."""
        }
        
        # Simple keyword matching for demo
        query_lower = query.lower()
        if any(word in query_lower for word in ["symptom", "pain", "ache", "hurt", "feel", "sick"]):
            base_response = base_responses["symptom"]
        elif any(word in query_lower for word in ["medication", "drug", "medicine", "prescription", "pill", "dose"]):
            base_response = base_responses["medication"]
        else:
            base_response = base_responses["general"]
        
        # Always append disclaimer
        return base_response + MEDICAL_DISCLAIMER
    
    def _get_relevant_sources(self, query: str) -> str:
        """Get relevant source citations"""
        sources = """
### 📚 Reference Sources

**Source 1**: Medical Knowledge Base
- Confidence: 92%
- Type: Verified Medical Literature
- Last Updated: January 2025

**Source 2**: Clinical Guidelines Database
- Confidence: 88%
- Type: Evidence-Based Guidelines
- Publisher: Medical Standards Organization

**Source 3**: Drug Information Database
- Confidence: 95%
- Type: Pharmaceutical Reference
- Verification: FDA Approved
        """
        return sources
    
    def _generate_metadata(self, response: str) -> str:
        """Generate response metadata"""
        metadata = f"""
**Response Metadata**
- Timestamp: {self.format_timestamp()}
- Model: {self.model_name}
- Tokens: ~{len(response.split())} words
- Processing Time: 0.8s
- Confidence: High
        """
        return metadata
    
    def export_conversation(self, format_type: str = "json") -> str:
        """Export conversation history"""
        if format_type == "json":
            return json.dumps(self.conversation_history, indent=2)
        elif format_type == "txt":
            text = f"HaleAI Medical Chatbot Conversation\nSession: {self.session_id}\n\n"
            for msg in self.conversation_history:
                text += f"[{msg['timestamp']}] User: {msg['user']}\n"
                text += f"[{msg['timestamp']}] Assistant: {msg['assistant']}\n\n"
            return text
        return ""
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return [], "", ""

def create_interface():
    """Create the Gradio interface"""
    
    chatbot = HaleAIChatbot()
    
    # Main interface
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
            neutral_hue="slate",
        ),
        css=custom_css,
        title="HaleAI Medical Chatbot",
        analytics_enabled=False
    ) as demo:
        
        # State management
        conversation_state = gr.State([])
        
        # Header
        with gr.Row(elem_classes="header-container"):
            with gr.Column(scale=8):
                gr.Markdown(
                    """
                    <div class="header-title">
                        🏥 HaleAI Medical Chatbot
                        <span class="status-indicator"></span>
                    </div>
                    <p style="color: rgba(255,255,255,0.9); font-size: 1rem; margin: 0;">
                        Your AI-powered medical information assistant | Model: Mixtral-8x7B-Instruct-v0.1
                    </p>
                    """,
                    elem_classes="header-content"
                )
            with gr.Column(scale=2):
                system_status = gr.JSON(
                    value=chatbot.get_system_info(),
                    label="System Status",
                    visible=False
                )
        
        # Main content area
        with gr.Row():
            with gr.Column(scale=7):
                # Chat interface
                chatbot_ui = gr.Chatbot(
                    value=[],
                    label="Chat History",
                    height=600,
                    show_label=False,
                    elem_classes="chatbot",
                    bubble_full_width=False,
                    avatar_images=(
                        None,  # User avatar
                        "🏥"   # Bot avatar
                    )
                )
                
                # Input area
                with gr.Row():
                    with gr.Column(scale=9):
                        msg_input = gr.Textbox(
                            placeholder="Ask me anything about medical topics... (Press Enter to send, Shift+Enter for new line)",
                            show_label=False,
                            lines=2,
                            max_lines=5,
                            elem_classes="input-container"
                        )
                    with gr.Column(scale=1, min_width=100):
                        send_btn = gr.Button(
                            "Send 📤",
                            variant="primary",
                            elem_classes="primary-btn"
                        )
                
                # Quick action buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", size="sm", elem_classes="secondary-btn")
                    export_json_btn = gr.Button("📥 Export JSON", size="sm", elem_classes="secondary-btn")
                    export_txt_btn = gr.Button("📄 Export TXT", size="sm", elem_classes="secondary-btn")
                
                # Feedback section
                with gr.Row():
                    gr.Markdown("### 💬 Feedback")
                    thumbs_up = gr.Button("👍", size="sm")
                    thumbs_down = gr.Button("👎", size="sm")
            
            # Right sidebar
            with gr.Column(scale=3):
                # Settings panel
                with gr.Accordion("⚙️ Settings", open=False):
                    model_select = gr.Dropdown(
                        choices=["Mixtral-8x7B-Instruct-v0.1", "GPT-4", "Claude-3"],
                        value="Mixtral-8x7B-Instruct-v0.1",
                        label="Model Selection",
                        interactive=True
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in responses"
                    )
                    max_tokens = gr.Slider(
                        minimum=128,
                        maximum=2048,
                        value=512,
                        step=128,
                        label="Max Response Length",
                        info="Maximum tokens in response"
                    )
                    theme_toggle = gr.Radio(
                        choices=["Light", "Dark"],
                        value="Light",
                        label="Theme",
                        info="Switch between light and dark mode"
                    )
                
                # Source display
                with gr.Accordion("📚 Sources & References", open=True):
                    sources_display = gr.Markdown(
                        "Sources will appear here after you ask a question.",
                        elem_classes="source-display"
                    )
                
                # Metadata display
                with gr.Accordion("ℹ️ Response Metadata", open=False):
                    metadata_display = gr.Markdown(
                        "Metadata will appear here after responses.",
                        elem_classes="metadata-display"
                    )
                
                # Help & Documentation
                with gr.Accordion("❓ Help & Tips", open=False):
                    gr.Markdown("""
                    ### How to Use HaleAI
                    
                    **Getting Started:**
                    - Type your medical question in the input box
                    - Press Enter or click Send
                    - Review the response and sources
                    
                    **Tips for Best Results:**
                    - Be specific with your questions
                    - Provide relevant context
                    - Ask follow-up questions for clarity
                    
                    **Important Notes:**
                    - This is an information tool, not a replacement for professional medical advice
                    - Always consult healthcare providers for diagnosis and treatment
                    - In emergencies, call your local emergency number
                    
                    **Privacy:**
                    - Conversations are session-based
                    - No personal data is stored permanently
                    - Use responsibly and avoid sharing sensitive information
                    """)
        
        # Disclaimer
        gr.Markdown("""
        ---
        **⚠️ Medical Disclaimer**: This chatbot provides general medical information for educational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of 
        your physician or other qualified health provider with any questions you may have regarding a medical condition. 
        Never disregard professional medical advice or delay in seeking it because of something you have read here.
        """)
        
        # Export output (hidden)
        export_output = gr.File(label="Download", visible=False)
        
        # Event handlers
        def respond(message, history, temp, tokens):
            return chatbot.process_message(message, history, temp, tokens)
        
        def export_json():
            content = chatbot.export_conversation("json")
            filename = f"haleai_conversation_{chatbot.session_id}.json"
            with open(filename, "w") as f:
                f.write(content)
            return filename
        
        def export_txt():
            content = chatbot.export_conversation("txt")
            filename = f"haleai_conversation_{chatbot.session_id}.txt"
            with open(filename, "w") as f:
                f.write(content)
            return filename
        
        # Connect events
        msg_input.submit(
            respond,
            [msg_input, chatbot_ui, temperature, max_tokens],
            [chatbot_ui, sources_display, metadata_display]
        ).then(
            lambda: "",
            None,
            msg_input
        )
        
        send_btn.click(
            respond,
            [msg_input, chatbot_ui, temperature, max_tokens],
            [chatbot_ui, sources_display, metadata_display]
        ).then(
            lambda: "",
            None,
            msg_input
        )
        
        clear_btn.click(
            lambda: chatbot.clear_history(),
            None,
            [chatbot_ui, sources_display, metadata_display]
        )
        
        export_json_btn.click(
            export_json,
            None,
            export_output
        )
        
        export_txt_btn.click(
            export_txt,
            None,
            export_output
        )
        
        # Feedback handlers (placeholder)
        thumbs_up.click(
            lambda: gr.Info("Thank you for your feedback! 👍"),
            None,
            None
        )
        
        thumbs_down.click(
            lambda: gr.Info("Thank you for your feedback. We'll work to improve! 👎"),
            None,
            None
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
        ssl_verify=False
    )