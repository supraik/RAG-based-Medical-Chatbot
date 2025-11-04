"""
Main entry point for HaleAI Medical Chatbot
Professional CLI interface with improved user experience
"""
import sys
import logging
from pathlib import Path
from typing import Optional

from chatbot import HaleAI
from config import Config

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class HaleAICLI:
    """Command-line interface for HaleAI"""
    
    def __init__(self):
        self.chatbot: Optional[HaleAI] = None
    
    def print_banner(self):
        """Print application banner"""
        print("\n" + "=" * 70)
        print("🏥 HaleAI Medical Chatbot")
        print("   Powered by Google Gemini & RAG Technology")
        print("=" * 70)
        print(f"Model: {Config.GEMINI_MODEL}")
        print(f"Vector Store: Pinecone ({Config.PINECONE_INDEX_NAME})")
        print("=" * 70 + "\n")
    
    def print_menu(self):
        """Print main menu"""
        print("\n📋 Main Menu")
        print("-" * 70)
        print("1. 🚀 Setup Knowledge Base (First-time setup)")
        print("2. 💬 Start Interactive Chat")
        print("3. ❓ Single Query Mode")
        print("4. 🧪 Run Test Queries")
        print("5. 🔍 Test Knowledge Base Connection")
        print("6. ℹ️  System Information")
        print("7. 🚪 Exit")
        print("-" * 70)
    
    def setup_knowledge_base(self):
        """Setup mode: Process documents and create knowledge base"""
        print("\n" + "=" * 70)
        print("🔧 KNOWLEDGE BASE SETUP")
        print("=" * 70)
        print("\nThis will:")
        print("  1. Process medical documents from the data directory")
        print("  2. Create embeddings for all content")
        print("  3. Upload to Pinecone vector database")
        print("\n⚠️  This only needs to be run once or when updating documents.")
        print(f"📁 Data directory: {Config.DATA_DIR}")
        
        # Check if data directory exists
        if not Config.DATA_DIR.exists():
            print(f"\n❌ Error: Data directory not found: {Config.DATA_DIR}")
            print("   Please create the directory and add your medical documents.")
            return
        
        # Check for PDF files
        pdf_files = list(Config.DATA_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"\n❌ Error: No PDF files found in {Config.DATA_DIR}")
            print("   Please add medical documents to process.")
            return
        
        print(f"\n📄 Found {len(pdf_files)} PDF file(s):")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")
        
        # Confirm setup
        print("\n" + "-" * 70)
        confirm = input("Continue with setup? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print("Setup cancelled.")
            return
        
        try:
            print("\n⏳ Initializing chatbot...")
            self.chatbot = HaleAI()
            
            print("\n📚 Processing documents...")
            self.chatbot.setup_knowledge_base(str(Config.DATA_DIR))
            
            print("\n" + "=" * 70)
            print("✅ Setup Complete!")
            print("=" * 70)
            print("You can now use the chatbot in normal mode (Option 2).\n")
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}", exc_info=True)
            print(f"\n❌ Setup failed: {str(e)}")
            print("Please check the logs for details.")
    
    def start_chat(self):
        """Interactive chat mode"""
        try:
            if self.chatbot is None:
                print("\n⏳ Initializing chatbot...")
                self.chatbot = HaleAI()
                self.chatbot.connect()
            
            self.chatbot.chat()
            
        except Exception as e:
            logger.error(f"Chat failed: {str(e)}", exc_info=True)
            print(f"\n❌ Failed to start chat: {str(e)}")
            print("Please ensure the knowledge base is set up (Option 1).")
    
    def single_query(self):
        """Single query mode"""
        try:
            if self.chatbot is None:
                print("\n⏳ Initializing chatbot...")
                self.chatbot = HaleAI()
                self.chatbot.connect()
            
            print("\n" + "=" * 70)
            print("❓ Single Query Mode")
            print("=" * 70)
            
            question = input("\n💬 Enter your medical question: ").strip()
            
            if not question:
                print("No question provided.")
                return
            
            print("\n🤔 Processing your question...")
            response = self.chatbot.query(question)
            
            print("\n" + "=" * 70)
            print("🏥 Dr. Hale's Response")
            print("=" * 70)
            print(f"\n{response['answer']}\n")
            
            if response["status"] == "success":
                print("-" * 70)
                print(f"📚 Sources: {response['num_sources']} medical references")
                print(f"⏱️  Processing time: {response['metadata']['processing_time']}")
                print(f"🤖 Model: {response['metadata']['model']}")
                
                # Show sources
                show_sources = input("\n📖 Show detailed sources? (yes/no): ").strip().lower()
                if show_sources == 'yes':
                    print("\n" + "=" * 70)
                    print("📚 Detailed Sources")
                    print("=" * 70)
                    for i, doc in enumerate(response['sources'], 1):
                        print(f"\n[Source {i}]")
                        print(f"Page: {doc.metadata.get('page', 'N/A')}")
                        print(f"Content: {doc.page_content[:200]}...")
                        print("-" * 70)
            
            print()
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            print(f"\n❌ Query failed: {str(e)}")
    
    def run_tests(self):
        """Test mode with predefined queries"""
        try:
            if self.chatbot is None:
                print("\n⏳ Initializing chatbot...")
                self.chatbot = HaleAI()
                self.chatbot.connect()
            
            print("\n" + "=" * 70)
            print("🧪 Test Mode - Running Predefined Queries")
            print("=" * 70)
            
            test_questions = [
                "What is diabetes and what are its symptoms?",
                "How is high blood pressure treated?",
                "What are the common side effects of antibiotics?",
                "What should I do if I have chest pain?",
                "How can I prevent the flu?"
            ]
            
            print(f"\n📝 Running {len(test_questions)} test queries...\n")
            
            results = self.chatbot.batch_query(test_questions)
            
            # Print summary
            print("\n" + "=" * 70)
            print("📊 Test Results Summary")
            print("=" * 70)
            
            success_count = sum(1 for r in results if r['status'] == 'success')
            
            for i, (question, result) in enumerate(zip(test_questions, results), 1):
                status_icon = "✅" if result['status'] == 'success' else "❌"
                print(f"\n{status_icon} Test {i}/{len(test_questions)}")
                print(f"Q: {question}")
                print(f"Status: {result['status']}")
                print(f"Sources: {result['num_sources']}")
                if result['status'] == 'success':
                    print(f"Time: {result['metadata']['processing_time']}")
                print(f"Answer: {result['answer'][:150]}...")
                print("-" * 70)
            
            print(f"\n✅ Tests passed: {success_count}/{len(test_questions)}")
            print()
            
        except Exception as e:
            logger.error(f"Tests failed: {str(e)}", exc_info=True)
            print(f"\n❌ Tests failed: {str(e)}")
    
    def test_connection(self):
        """Test knowledge base connection"""
        try:
            print("\n⏳ Testing connections...")
            
            if self.chatbot is None:
                self.chatbot = HaleAI()
            
            print("\n✅ Gemini API: Connected")
            
            self.chatbot.connect()
            print("✅ Pinecone Vector Store: Connected")
            
            # Test retrieval
            test_query = "medical information"
            docs = self.chatbot.vector_store.retrieve_documents(test_query)
            print(f"✅ Document Retrieval: Working ({len(docs)} documents found)")
            
            print("\n🎉 All systems operational!\n")
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}", exc_info=True)
            print(f"\n❌ Connection test failed: {str(e)}\n")
    
    def show_system_info(self):
        """Display system information"""
        print("\n" + "=" * 70)
        print("ℹ️  System Information")
        print("=" * 70)
        print(f"\n🤖 AI Model:")
        print(f"   Name: {Config.GEMINI_MODEL}")
        print(f"   Temperature: {Config.GEMINI_TEMPERATURE}")
        print(f"   Max Tokens: {Config.GEMINI_MAX_OUTPUT_TOKENS}")
        print(f"\n📊 Vector Store:")
        print(f"   Provider: Pinecone")
        print(f"   Index: {Config.PINECONE_INDEX_NAME}")
        print(f"   Dimension: {Config.PINECONE_DIMENSION}")
        print(f"   Region: {Config.PINECONE_REGION}")
        print(f"\n🔍 Embeddings:")
        print(f"   Model: {Config.EMBEDDING_MODEL}")
        print(f"\n⚙️  RAG Configuration:")
        print(f"   Initial Retrieval: {Config.RETRIEVER_K} documents")
        print(f"   After Reranking: {Config.RERANK_TOP_K} documents")
        print(f"   Chunk Size: {Config.CHUNK_SIZE} characters")
        print(f"   Chunk Overlap: {Config.CHUNK_OVERLAP} characters")
        print(f"\n📁 Data:")
        print(f"   Directory: {Config.DATA_DIR}")
        print("=" * 70 + "\n")
    
    def run(self):
        """Main application loop"""
        self.print_banner()
        
        while True:
            try:
                self.print_menu()
                choice = input("\n🔹 Enter your choice (1-7): ").strip()
                
                if choice == '1':
                    self.setup_knowledge_base()
                elif choice == '2':
                    self.start_chat()
                elif choice == '3':
                    self.single_query()
                elif choice == '4':
                    self.run_tests()
                elif choice == '5':
                    self.test_connection()
                elif choice == '6':
                    self.show_system_info()
                elif choice == '7':
                    print("\n👋 Thank you for using HaleAI. Stay healthy!")
                    break
                else:
                    print("\n❌ Invalid choice. Please enter 1-7.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                print(f"\n❌ An error occurred: {str(e)}")


def main():
    """Entry point"""
    try:
        cli = HaleAICLI()
        cli.run()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        print(f"\n❌ Application failed to start: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()