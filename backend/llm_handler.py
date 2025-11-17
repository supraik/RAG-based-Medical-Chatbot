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
    """Handles LLM operations via Google Gemini API with a Hugging Face fallback."""
    
    def __init__(self):
        """Initialize Gemini API client, fall back to Hugging Face if needed."""
        # Flags and placeholders for fallback
        self.use_hf: bool = False
        self.hf_pipeline = None
        self.model = None
        self.chat = None

        # Try Gemini first; if anything fails and HF fallback enabled, initialize HF
        try:
            self._configure_gemini()
            self._setup_model()
            logger.info(f"✅ Gemini model initialized: {Config.GEMINI_MODEL}")
        except Exception as e:
            logger.warning("Gemini initialization failed, attempting Hugging Face fallback: %s", e, exc_info=True)
            if getattr(Config, 'USE_HF_FALLBACK', False):
                try:
                    self._setup_hf_model()
                    self.use_hf = True
                    logger.info("✅ Hugging Face fallback initialized: %s", Config.HF_FALLBACK_MODEL)
                except Exception as hf_e:
                    logger.error("Hugging Face fallback failed: %s", hf_e, exc_info=True)
                    raise
            else:
                raise
    
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

    def _setup_hf_model(self) -> None:
        """Setup a Hugging Face pipeline as a fallback model (text2text)."""
        try:
            from transformers import pipeline
        except Exception as e:
            logger.error("transformers library is required for HF fallback: %s", e)
            raise

        model_name = getattr(Config, 'HF_FALLBACK_MODEL', 'google/flan-t5-small')
        # Use text2text-generation (works well with FLAN-style models)
        try:
            self.hf_pipeline = pipeline('text2text-generation', model=model_name, device=-1)
        except Exception:
            # Try text-generation as a fallback
            self.hf_pipeline = pipeline('text-generation', model=model_name, device=-1)

        # mark model as HF-backed
        self.model = None
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
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

            # Build full prompt with context and optional conversation history
            # If history is provided, format it into a compact context string
            history_text = ''
            if history:
                normalized = self._normalize_history_turns(history)
                history_text = self._format_history_for_prompt(normalized)

            # Merge retrieved context and history_text: history first, then context
            combined_context = None
            if history_text and context:
                combined_context = history_text + "\n\n" + context
            elif history_text:
                combined_context = history_text
            else:
                combined_context = context

            # Build final prompt
            full_prompt = self._build_prompt(prompt, combined_context, system_instruction)

            if self.use_hf and self.hf_pipeline is not None:
                return self._post_process_response(self._generate_hf(full_prompt))

            # Generate response via Gemini
            response = self.model.generate_content(full_prompt)

            # Handle safety filters
            if not response.text:
                if getattr(response, 'prompt_feedback', None) and response.prompt_feedback.block_reason:
                    logger.warning(f"Content blocked: {response.prompt_feedback.block_reason}")
                    return self._get_blocked_response()
                return "I apologize, but I couldn't generate a response. Please rephrase your question."

            return self._post_process_response(response.text)

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            # If Gemini failed at runtime and HF fallback is enabled, try HF
            if not self.use_hf and getattr(Config, 'USE_HF_FALLBACK', False):
                try:
                    self._setup_hf_model()
                    self.use_hf = True
                    return self._post_process_response(self._generate_hf(full_prompt))
                except Exception:
                    logger.exception("HF fallback failed after Gemini error")
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
            # Normalize history into a list of {'user':..., 'bot':...}
            normalized = self._normalize_history_turns(history)

            # If HF fallback is active, include normalized history in the prompt
            if self.use_hf and self.hf_pipeline is not None:
                history_text = self._format_history_for_prompt(normalized)
                full_prompt = self._build_prompt(message, history_text, None)
                return self._post_process_response(self._generate_hf(full_prompt))

            # Convert normalized history into Gemini chat history shape
            gemini_history = []
            for turn in normalized:
                user_text = turn.get('user', '') or ''
                bot_text = turn.get('bot', '') or ''
                if user_text:
                    gemini_history.append({"role": "user", "parts": [user_text]})
                if bot_text:
                    gemini_history.append({"role": "model", "parts": [bot_text]})

            # Initialize chat with history (empty list if none)
            self.chat = self.model.start_chat(history=gemini_history or [])

            response = self.chat.send_message(message)
            return self._post_process_response(getattr(response, 'text', '') or '')
        
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

    def _normalize_history_turns(self, history: Optional[List]) -> List[Dict[str, str]]:
        """Normalize various history formats into a list of dicts with 'user' and 'bot' keys.

        Supports:
        - List[Tuple[user, assistant]]
        - List[Dict] with keys 'user' and 'bot' or 'assistant'
        - None -> []
        """
        if not history:
            return []

        normalized: List[Dict[str, str]] = []
        for turn in history:
            try:
                if isinstance(turn, dict):
                    user = turn.get('user') or turn.get('speaker') or turn.get('question') or ''
                    bot = turn.get('bot') or turn.get('assistant') or turn.get('answer') or ''
                elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    user, bot = turn[0] or '', turn[1] or ''
                else:
                    # Unknown format, skip
                    continue
                normalized.append({'user': str(user), 'bot': str(bot)})
            except Exception:
                continue
        return normalized

    def _format_history_for_prompt(self, normalized_history: List[Dict[str, str]], max_turns: int = 6) -> str:
        """Format normalized history into a compact string to include in prompts for HF fallback.

        Keeps only the last `max_turns` turns (pairs), joins them as:
        User: ...\nAssistant: ...\n
        """
        if not normalized_history:
            return ''

        # keep most recent turns
        hist = normalized_history[-max_turns:]
        parts = []
        for turn in hist:
            u = turn.get('user', '').strip()
            b = turn.get('bot', '').strip()
            if u:
                parts.append(f"User: {u}")
            if b:
                parts.append(f"Assistant: {b}")
        return "\n".join(parts)
    
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
            if self.use_hf and self.hf_pipeline is not None:
                out = self.hf_pipeline("Hello, this is a test.", max_length=getattr(Config, 'HF_MAX_TOKENS', 32))
                text = None
                if isinstance(out, list) and out:
                    text = out[0].get('generated_text') or out[0].get('generated_text') or out[0].get('text')
                return bool(text)

            response = self.model.generate_content("Hello, this is a test.")
            return bool(getattr(response, 'text', None))
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def _generate_hf(self, prompt: str) -> str:
        """Generate text using the HF pipeline (text2text or text-generation)."""
        if self.hf_pipeline is None:
            raise RuntimeError("HF pipeline is not initialized")

        # transformers pipelines return a list of dicts
        out = self.hf_pipeline(prompt, max_length=getattr(Config, 'HF_MAX_TOKENS', 256), do_sample=False)
        if isinstance(out, list) and out:
            candidate = out[0]
            return candidate.get('generated_text') or candidate.get('text') or ''
        if isinstance(out, dict):
            return out.get('generated_text') or out.get('text') or ''
        return str(out)