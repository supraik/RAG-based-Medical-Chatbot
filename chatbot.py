"""
Main HaleAI Medical Chatbot class
Orchestrates all components for RAG-based medical Q&A
"""
from typing import Dict, List, Optional

# --- Langchain Document Import with Fallbacks ---
try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document
    except Exception:
        class Document:  # type: ignore
            def __init__(self, page_content: str = "", metadata: dict | None = None):
                self.page_content = page_content
                self.metadata = metadata or {}

# --- Local Module Imports ---
from data_processor import DataProcessor
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
from rag_processor import RAGProcessor


class HaleAI:
    """HaleAI Medical Chatbot with RAG and Reranking capabilities"""

    def __init__(self):
        """Initializes all components of the chatbot."""
        print("\nInitializing HaleAI Medical Chatbot...")
        print("=" * 60)

        self.data_processor = DataProcessor()
        self.vector_store = VectorStoreManager(self.data_processor.embeddings)
        self.llm = LLMHandler()
        self.rag = RAGProcessor()

        print("=" * 60)
        print("HaleAI initialized successfully!\n")

    def setup_knowledge_base(self, data_dir: str):
        """Process documents and load to Pinecone."""
        print("\nSetting up knowledge base...")
        print("=" * 60)

        chunks = self.data_processor.process_documents(data_dir)
        self.vector_store.load_documents(chunks)

        print("=" * 60)
        print("Knowledge base setup complete!\n")

    def connect(self):
        """Connects to the existing knowledge base in Pinecone."""
        print("\nConnecting to knowledge base...")
        self.vector_store.connect_to_existing_index()
        print("Connected!\n")

    def query(self, user_question: str, history: Optional[List[Dict]] = None) -> Dict:
        """
        Processes a user query using RAG with reranking.

        Args:
            user_question: The user's medical question.
            history: List of past turns [{"user": ..., "bot": ...}]

        Returns:
            Dict with answer, sources, status, and updated history.
        """
        if history is None:
            history = []

        print(f"\nQuery: {user_question}")
        print("-" * 60)

        try:
            # 1. Retrieve initial candidates
            raw_docs = self.vector_store.retrieve_documents(user_question)
            print(f"Found {len(raw_docs)} initial candidates")

            # 2. Rerank to top 3
            best_docs = self.rag.rerank(user_question, raw_docs, top_k=3)
            print(f"Top {len(best_docs)} after reranking")

            if not best_docs:
                msg = "I couldn't find relevant information. Please consult a doctor."
                return {
                    "answer": msg,
                    "sources": [],
                    "num_sources": 0,
                    "status": "no_sources",
                    "history": history
                }

            # 3. Build context + history
            context = "\n\n".join(
                f"[Source Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content.strip()}"
                for doc in best_docs
            )

            history_text = ""
            if history:
                history_text = "\n".join(
                    f"User: {turn['user']}\nDr. Hale: {turn['bot']}"
                    for turn in history[-4:]  # Last 4 turns
                ) + "\n"

            prompt = f"""
You are Dr. Hale — a kind, trusted doctor who explains things simply.

Your job is to help people understand their health using only the medical book pages below.

---

CONVERSATION SO FAR:
{history_text}

---

BOOK SOURCES (Use these only):
{context}

---

PATIENT'S QUESTION:
{user_question}

---

ANSWER IN 3 SIMPLE PARTS:

1. **WHAT THIS MEANS**  
   → Explain in 1–2 short sentences using everyday words.  
   → Example: "This means you're vomiting blood. It can be serious."

2. **WHAT TO DO**  
   → Say clearly:  
      • "Go to the hospital now" (if urgent)  
      • "See a doctor soon" (if not urgent)  
      • "No clear info — please see a doctor" (if unsure)

3. **FROM THE BOOK**  
   → Show page numbers: [Page 123] [Page 456]  
   → Keep it short

RULES:
- Use simple words only (no "hematemesis" — say "vomiting blood")
- NEVER guess or add outside knowledge
- NEVER say "I think" or "maybe"
- Be kind and calm

ANSWER NOW:
"""

            # 4. Generate answer
            answer = self.llm.generate(prompt).strip()
            if "GALE" in answer:
                answer = answer.split("GALE")[0].strip()

            # 5. Update history
            new_history = history + [{"user": user_question, "bot": answer}]

            return {
                "question": user_question,
                "answer": answer,
                "sources": best_docs,
                "num_sources": len(best_docs),
                "status": "success",
                "history": new_history
            }

        except Exception as e:
            print(f"Error: {e}")
            return {
                "answer": "Error processing your question.",
                "sources": [],
                "status": "error",
                "history": history
            }

    def chat(self):
        """Interactive CLI chat."""
        print("\n" + "=" * 60)
        print("HaleAI Medical Chatbot - Interactive Mode")
        print("=" * 60)
        print("Type 'quit' to exit.\n")

        if not self.vector_store.retriever:
            print("Knowledge base not connected.")
            return

        history = []
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    break

                resp = self.query(user_input, history)
                print(f"\nDr. Hale: {resp['answer']}\n")
                if resp["status"] == "success":
                    print(f"Sources: {resp['num_sources']} pages")
                history = resp.get("history", history)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break