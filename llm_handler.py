"""
LLM handler for HaleAI using Google Gemini
Manages Gemini API interactions with proper error handling and safety
"""
import logging
from typing import List, Dict, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content: str = "", metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from config import Config

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handles LLM operations via Google Gemini API"""
    
    def __init__(self):
        """Initialize Gemini API client"""
        self._configure_gemini()
        self._setup_model()
        logger.info(f"✅ Gemini model initialized: {Config.GEMINI_MODEL}")
    
    def _configure_gemini(self) -> None:
        """Configure Gemini API"""
        if not Config.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Get your API key from https://makersuite.google.com/app/apikey"
            )
        genai.configure(api_key=Config.GEMINI_API_KEY)
    
    def _setup_model(self) -> None:
        """Setup Gemini model with configuration"""
        generation_config = {
            "temperature": Config.GEMINI_TEMPERATURE,
            "top_p": Config.GEMINI_TOP_P,
            "top_k": Config.GEMINI_TOP_K,
            "max_output_tokens": Config.GEMINI_MAX_OUTPUT_TOKENS,
        }
        
        # Safety settings for medical content
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,  # Allow some medical content
        }
        
        self.model = genai.GenerativeModel(
            model_name=Config.GEMINI_MODEL,
            generation_config=generation_config,
            safety_settings=safety_settings if Config.ENABLE_SAFETY_FILTERS else None
        )
        
        # Create chat session for context management
        self.chat = None
    
    def generate(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate response from Gemini
        
        Args:
            prompt: User query
            context: Retrieved context from RAG
            system_instruction: System-level instruction
            
        Returns:
            Generated response text
        """
        try:
            # Build full prompt with context
            full_prompt = self._build_prompt(prompt, context, system_instruction)
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Handle safety filters
            if not response.text:
                if response.prompt_feedback.block_reason:
                    logger.warning(f"Content blocked: {response.prompt_feedback.block_reason}")
                    return self._get_blocked_response()
                return "I apologize, but I couldn't generate a response. Please rephrase your question."
            
            return self._post_process_response(response.text)
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return self._get_error_response(str(e))
    
    def generate_with_chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response maintaining conversation history
        
        Args:
            message: User message
            history: Previous conversation turns
            
        Returns:
            Generated response
        """
        try:
            # Initialize new chat if needed
            if self.chat is None or history is None:
                self.chat = self.model.start_chat(history=[])
            else:
                # Convert history to Gemini format
                gemini_history = []
                for turn in history:
                    gemini_history.extend([
                        {"role": "user", "parts": [turn["user"]]},
                        {"role": "model", "parts": [turn["bot"]]}
                    ])
                self.chat = self.model.start_chat(history=gemini_history)
            
            response = self.chat.send_message(message)
            return self._post_process_response(response.text)
        
        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}", exc_info=True)
            return self._get_error_response(str(e))
    
    def _build_prompt(
        self,
        query: str,
        context: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """Build complete prompt with system instruction and context"""
        
        default_system = """You are Dr. Hale, a knowledgeable and empathetic medical AI assistant. 
You provide accurate medical information based on verified sources while maintaining a professional 
yet approachable tone. Always prioritize patient safety and encourage consultation with healthcare 
professionals for serious concerns."""
        
        system = system_instruction or default_system
        
        prompt_parts = [system, "\n"]
        
        if context:
            prompt_parts.extend([
                "\n## Medical Reference Context",
                "Use ONLY the following verified medical information to answer the question:",
                context,
                "\n"
            ])
        
        prompt_parts.extend([
            "\n## Patient Question",
            query,
            "\n",
            "## Your Response",
            "Provide a clear, accurate response based on the reference context above. ",
            "Structure your answer professionally with relevant details. ",
            "If the information is not in the context, clearly state that and recommend ",
            "consulting a healthcare professional."
        ])
        
        return "\n".join(prompt_parts)
    
    def _post_process_response(self, text: str) -> str:
        """Clean and format the response"""
        # Remove common artifacts
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        
        # Ensure response isn't too long
        max_length = 1500
        if len(text) > max_length:
            # Find last complete sentence within limit
            truncated = text[:max_length]
            last_period = max(
                truncated.rfind('.'),
                truncated.rfind('!'),
                truncated.rfind('?')
            )
            if last_period > 0:
                text = text[:last_period + 1]
        
        return text
    
    def _get_blocked_response(self) -> str:
        """Response when content is blocked by safety filters"""
        return """I apologize, but I cannot provide information on this topic due to safety guidelines. 
        
For medical concerns, please:
- Contact your healthcare provider directly
- Call emergency services if urgent (911 in US)
- Visit a medical professional for proper evaluation

I'm here to help with general medical information within appropriate boundaries."""
    
    def _get_error_response(self, error: str) -> str:
        """Response when an error occurs"""
        return f"""I apologize, but I encountered a technical issue while processing your request.

Please try:
1. Rephrasing your question
2. Asking about a different topic
3. Trying again in a moment

If the issue persists, please contact support.

Technical details: {error[:100]}"""
    
    def format_context(self, retrieved_docs: List[Document]) -> str:
        """Format retrieved documents as structured context"""
        if not retrieved_docs:
            return "No relevant medical information found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.page_content.strip()
            
            # Add metadata if available
            source_info = ""
            if hasattr(doc, 'metadata') and doc.metadata:
                meta = doc.metadata
                page = meta.get('page', 'Unknown')
                source = meta.get('source', 'Unknown')
                source_info = f" [Page {page}, Source: {source}]"
            
            context_parts.append(f"### Reference {i}{source_info}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def test_connection(self) -> bool:
        """Test Gemini API connection"""
        try:
            response = self.model.generate_content("Hello, this is a test.")
            return bool(response.text)
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False