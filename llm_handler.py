"""
LLM handler for HaleAI
Manages Mistral model via Hugging Face API
"""
import requests
import time
from typing import List
try:
    from langchain.schema import Document
except Exception:
    # lightweight fallback Document for typing when langchain package can't be imported
    class Document:  # type: ignore
        def __init__(self, page_content: str = "", metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}
from config import *


class LLMHandler:
    """Handles LLM operations via Hugging Face API"""
    
    def __init__(self):
        self.api_url = HF_API_URL
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        self._validate_token()
    
    def _validate_token(self):
        """Validate HF token is present"""
        if not HF_TOKEN:
            raise ValueError(
                "HF_TOKEN not found in environment variables. "
                "Please add it to your .env file. "
                "Get your token from https://huggingface.co/settings/tokens"
            )
        print("✅ Hugging Face API token validated")
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate response from Hugging Face API using InferenceClient
        
        Args:
            prompt: The input prompt
            max_retries: Number of retry attempts if model is loading
            
        Returns:
            Generated text response
        """
        # Use the Hugging Face InferenceClient (same approach as test_hf.py)
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

        for attempt in range(max_retries):
            try:
                # Use `max_tokens` (InferenceClient expects this parameter name)
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )

                # The InferenceClient returns a structured object similar to the
                # REST API response used in test_hf.py. Extract the assistant text.
                try:
                    return response.choices[0].message["content"]
                except Exception:
                    # Fallback: stringify the response
                    return str(response)

            except Exception as e:
                msg = str(e).lower()
                if "loading" in msg or "model is loading" in msg:
                    if attempt < max_retries - 1:
                        wait = 20
                        print(f"⏱️ Model is loading, retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait)
                        continue
                    return "Error: Model is taking too long to load. Please try again in a few minutes."

                print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
                return f"Error: {str(e)}"

        return "Error: Model is taking too long to load. Please try again in a few minutes."
    
    def format_context(self, retrieved_docs: List[Document]) -> str:
        """Format retrieved documents as context"""
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.page_content.strip()
            # Limit each source to avoid token overflow
            if len(content) > 300:
                content = content[:300] + "..."
            context_parts.append(f"[Source {i}] {content}")
        
        return "\n\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create optimized prompt for the LLM
        Format adapts based on the model being used
        """
        # Evidence-first system instruction: only answer from provided context
        # If the context does not contain evidence linking items (for example,
        # symptoms or diagnoses), the assistant must explicitly say it has no
        # evidence and avoid inventing causal relationships.
        prompt = f"""
<|system|>
You are a careful, conservative medical assistant. Only use information that is explicitly present in the provided "Medical Information" context. Do NOT infer causal links, associations, or diagnoses that are not supported by the context. If the context does not provide evidence for a claim or a link between conditions/symptoms, state: "I don't have evidence in the provided information to support that link." Always encourage the user to consult a qualified healthcare professional for diagnosis or emergencies.
</s>
<|user|>
Medical Information:
{context}

Question: {query}
</s>
<|assistant|>
"""

        return prompt
    
    def extract_answer(self, generated_text: str, original_prompt: str = None) -> str:
        """
        Clean and extract the answer from generated text
        
        Args:
            generated_text: The raw generated text from the API
            original_prompt: The original prompt (optional, for compatibility)
        
        Returns:
            Cleaned answer text
        """
        # The API with return_full_text=False only returns the generated part
        # So we don't need to remove the prompt
        
        # Remove instruction tags if present (supports multiple formats)
        answer = generated_text
        for tag in ["[/INST]", "<s>", "</s>", "<|assistant|>", "<|user|>", "<|system|>"]:
            answer = answer.replace(tag, "")
        answer = answer.strip()
        
        # Remove any remaining prompt artifacts
        if original_prompt and answer.startswith(original_prompt):
            answer = answer[len(original_prompt):].strip()
        
        # Split into sentences and clean
        sentences = answer.split('. ')
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                if not sentence.endswith('.') and not sentence.endswith('?') and not sentence.endswith('!'):
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        final_answer = ' '.join(cleaned_sentences).strip()
        
        # Limit length
        if len(final_answer) > 600:
            # Cut at last complete sentence within limit
            truncated = final_answer[:600]
            last_period = truncated.rfind('.')
            if last_period > 0:
                final_answer = truncated[:last_period + 1]
            else:
                final_answer = truncated + "..."
        
        return final_answer if final_answer else "I don't have sufficient information to answer this question."