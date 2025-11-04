"""
Main HaleAI Medical Chatbot class
Professional RAG-based medical Q&A system with Gemini integration
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document
    except Exception:
        class Document:
            def __init__(self, page_content: str = "", metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}

from data_processor import DataProcessor
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
from rag_processor import RAGProcessor
from config import Config

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class HaleAI:
    """HaleAI Medical Chatbot with RAG and Gemini"""
    
    SYSTEM_INSTRUCTION = """You are Dr. Hale, an expert medical AI assistant with extensive knowledge 
of medical conditions, treatments, and healthcare best practices.

Your role:
- Provide accurate, evidence-based medical information
- Explain complex medical concepts in clear, accessible language
- Cite specific sources from the medical knowledge base
- Maintain a professional yet empathetic tone
- Prioritize patient safety above all

Important guidelines:
1. ONLY use information from the provided medical references
2. Always cite page numbers when referencing information
3. For emergencies, immediately advise seeking emergency care
4. Encourage professional medical consultation for diagnosis
5. Never make definitive diagnoses or treatment recommendations
6. Be clear about limitations and uncertainties
7. Use plain language, avoiding unnecessary medical jargon

Format your responses clearly:
- Start with a direct answer to the question
- Provide relevant details from sources
- Include practical next steps
- End with appropriate medical disclaimers when needed"""

    def __init__(self):
        """Initialize all components of the chatbot"""
        logger.info("Initializing HaleAI Medical Chatbot...")
        
        try:
            self.data_processor = DataProcessor()
            self.vector_store = VectorStoreManager(self.data_processor.embeddings)
            self.llm = LLMHandler()
            self.rag = RAGProcessor()
            
            # Test connections
            if not self.llm.test_connection():
                raise RuntimeError("Failed to connect to Gemini API")
            
            logger.info("✅ HaleAI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HaleAI: {str(e)}", exc_info=True)
            raise

    def setup_knowledge_base(self, data_dir: str) -> None:
        """
        Process documents and load to Pinecone
        
        Args:
            data_dir: Directory containing medical documents
        """
        logger.info("Setting up knowledge base...")
        
        try:
            chunks = self.data_processor.process_documents(data_dir)
            self.vector_store.load_documents(chunks)
            logger.info("✅ Knowledge base setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup knowledge base: {str(e)}", exc_info=True)
            raise

    def connect(self) -> None:
        """Connect to the existing knowledge base in Pinecone"""
        logger.info("Connecting to knowledge base...")
        
        try:
            self.vector_store.connect_to_existing_index()
            logger.info("✅ Connected to knowledge base")
            
        except Exception as e:
            logger.error(f"Failed to connect to knowledge base: {str(e)}", exc_info=True)
            raise

    def query(
        self, 
        user_question: str, 
        history: Optional[List[Dict]] = None,
        use_reranking: bool = True
    ) -> Dict:
        """
        Process a user query using RAG with Gemini
        
        Args:
            user_question: The user's medical question
            history: List of past turns [{"user": ..., "bot": ...}]
            use_reranking: Whether to use reranking (default: True)
        
        Returns:
            Dict with answer, sources, confidence, and metadata
        """
        if history is None:
            history = []
        
        logger.info(f"Processing query: {user_question[:100]}...")
        start_time = datetime.now()
        
        try:
            # Validate input
            if not user_question or not user_question.strip():
                return self._create_error_response(
                    "Please provide a valid question.",
                    history
                )
            
            # 1. Retrieve initial candidates
            raw_docs = self.vector_store.retrieve_documents(user_question)
            logger.info(f"Retrieved {len(raw_docs)} initial candidates")
            
            if not raw_docs:
                return self._create_no_sources_response(history)
            
            # 2. Rerank to get best matches
            if use_reranking:
                best_docs = self.rag.rerank(
                    user_question, 
                    raw_docs, 
                    top_k=Config.RERANK_TOP_K
                )
                logger.info(f"Reranked to top {len(best_docs)} documents")
            else:
                best_docs = raw_docs[:Config.RERANK_TOP_K]
            
            # 3. Build context from top documents
            context = self._build_context(best_docs)
            
            # 4. Generate response with Gemini
            answer = self.llm.generate(
                prompt=user_question,
                context=context,
                system_instruction=self.SYSTEM_INSTRUCTION
            )
            
            # 5. Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 6. Build response
            response = {
                "question": user_question,
                "answer": answer,
                "sources": best_docs,
                "num_sources": len(best_docs),
                "status": "success",
                "history": history + [{"user": user_question, "bot": answer}],
                "metadata": {
                    "processing_time": f"{processing_time:.2f}s",
                    "model": Config.GEMINI_MODEL,
                    "reranking_used": use_reranking,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"✅ Query processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return self._create_error_response(
                f"An error occurred while processing your question: {str(e)}",
                history
            )
    
    def _build_context(self, docs: List[Document]) -> str:
        """Build formatted context from documents with citations"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            
            # Extract metadata
            page = doc.metadata.get('page', 'N/A')
            source = doc.metadata.get('source', 'Medical Reference')
            
            # Format with clear citation
            context_parts.append(
                f"[Reference {i} - Page {page}]:\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def _create_no_sources_response(self, history: List[Dict]) -> Dict:
        """Create response when no sources are found"""
        message = """I couldn't find relevant information in my medical knowledge base to answer your question accurately.

**What you can do:**
1. Try rephrasing your question with more specific terms
2. Consult with a healthcare professional for personalized advice
3. For emergencies, please call emergency services immediately

**Remember:** For any health concerns, it's always best to consult with a qualified healthcare provider who can evaluate your specific situation."""
        
        return {
            "answer": message,
            "sources": [],
            "num_sources": 0,
            "status": "no_sources",
            "history": history,
            "metadata": {
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _create_error_response(self, error_message: str, history: List[Dict]) -> Dict:
        """Create response for error conditions"""
        return {
            "answer": error_message,
            "sources": [],
            "num_sources": 0,
            "status": "error",
            "history": history,
            "metadata": {
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """
        Process multiple queries in batch
        
        Args:
            questions: List of questions to process
            
        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing {len(questions)} queries in batch...")
        results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            result = self.query(question)
            results.append(result)
        
        logger.info(f"✅ Batch processing complete: {len(results)} queries processed")
        return results
    
    def chat(self) -> None:
        """Interactive CLI chat interface"""
        print("\n" + "=" * 70)
        print("🏥 HaleAI Medical Chatbot - Interactive Mode")
        print("=" * 70)
        print("\nCommands:")
        print("  - Type 'quit' or 'exit' to end the session")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'help' for more information")
        print("\nNote: This AI provides information only. Always consult healthcare")
        print("professionals for medical advice, diagnosis, or treatment.\n")
        
        if not self.vector_store.retriever:
            print("❌ Knowledge base not connected. Please run setup first.")
            return
        
        history = []
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thank you for using HaleAI. Stay healthy!")
                    break
                
                if user_input.lower() == 'clear':
                    history = []
                    print("✅ Conversation history cleared")
                    continue
                
                if user_input.lower() == 'help':
                    self._print_help()
                    continue
                
                # Process query
                print("\n🤔 Thinking...")
                response = self.query(user_input, history)
                
                print(f"\n🏥 Dr. Hale:\n{response['answer']}\n")
                
                if response["status"] == "success":
                    print(f"📚 Based on {response['num_sources']} medical references")
                    print(f"⏱️  Response time: {response['metadata']['processing_time']}")
                
                history = response.get("history", history)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                print(f"\n❌ An error occurred: {str(e)}")
    
    def _print_help(self) -> None:
        """Print help information"""
        print("\n" + "=" * 70)
        print("📖 HaleAI Help")
        print("=" * 70)
        print("""
HaleAI is an AI-powered medical information assistant that uses:
- Retrieval-Augmented Generation (RAG) for accurate information
- Google Gemini for natural language understanding
- Medical knowledge base for evidence-based responses

How to ask questions:
- Be specific: "What are the symptoms of diabetes?"
- Provide context: "I have a persistent cough for 2 weeks"
- Ask follow-ups: "What are the treatment options?"

Important Disclaimers:
- This is an information tool, NOT a replacement for medical care
- Always consult healthcare professionals for diagnosis and treatment
- In emergencies, call emergency services immediately
- Do not delay seeking medical care based on information here

Commands:
- 'quit' or 'exit': End the session
- 'clear': Clear conversation history
- 'help': Show this help message

For technical support or feedback, please contact your administrator.
        """)
        print("=" * 70)