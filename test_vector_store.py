"""
Test suite for vector store operations
"""
import unittest
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from vector_store import VectorStoreManager
from config import *

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        self.vector_store = VectorStoreManager(self.embeddings)
        
    def test_index_connection(self):
        """Test if we can connect to the Pinecone index"""
        try:
            self.vector_store.connect_to_existing_index()
            self.assertIsNotNone(self.vector_store.vectorstore)
            self.assertIsNotNone(self.vector_store.retriever)
            print("✅ Successfully connected to Pinecone index")
        except Exception as e:
            self.fail(f"Failed to connect to index: {str(e)}")

    def test_document_upload_and_retrieval(self):
        """Test document upload and retrieval"""
        # Test documents
        test_docs = [
            Document(page_content="Acne is a skin condition that occurs when hair follicles become clogged with oil and dead skin cells."),
            Document(page_content="Depression is a mental health disorder characterized by persistent feelings of sadness and loss of interest.")
        ]
        
        try:
            print("\n📝 Testing document upload and retrieval...")
            
            # Upload test documents
            print("📤 Uploading test documents to vector store...")
            self.vector_store.load_documents(test_docs)
            print("✅ Successfully uploaded test documents")
            
            # Wait a moment for indexing
            import time
            time.sleep(2)
            
            # Test retrieval with exact content match
            # Connect to existing index first
            print("\n🔗 Connecting to existing index...")
            self.vector_store.connect_to_existing_index()
            
            print("\n🔍 Testing retrieval with exact content...")
            # Use more specific query for acne
            acne_docs = self.vector_store.retrieve_documents("skin condition acne clogged hair follicles oil")
            print(f"Retrieved {len(acne_docs)} documents for acne query")
            if len(acne_docs) > 0:
                print(f"First result: {acne_docs[0].page_content[:100]}")
            self.assertTrue(len(acne_docs) > 0, "No documents retrieved for acne query")
            self.assertIn("acne", acne_docs[0].page_content.lower(), "Retrieved document doesn't contain 'acne'")
            print("✅ Successfully retrieved documents about acne")
            
            # Test retrieval with different query
            print("\n🔍 Testing retrieval with depression query...")
            # Use more specific query for depression
            depression_docs = self.vector_store.retrieve_documents("mental health disorder depression sadness loss of interest")
            print(f"Retrieved {len(depression_docs)} documents for depression query")
            if len(depression_docs) > 0:
                print(f"First result: {depression_docs[0].page_content[:100]}")
            self.assertTrue(len(depression_docs) > 0, "No documents retrieved for depression query")
            self.assertIn("depression", depression_docs[0].page_content.lower(), "Retrieved document doesn't contain 'depression'")
            print("✅ Successfully retrieved documents about depression")
            
        except Exception as e:
            self.fail(f"Document upload/retrieval test failed: {str(e)}")

    def test_error_handling(self):
        """Test error handling scenarios"""
        # Test retrieval without initialization
        uninit_store = VectorStoreManager(self.embeddings)
        with self.assertRaises(ValueError):
            uninit_store.retrieve_documents("test query")
            print("✅ Successfully caught uninitialized retriever error")
        
        # Test with empty query
        self.vector_store.connect_to_existing_index()
        empty_results = self.vector_store.retrieve_documents("")
        self.assertIsInstance(empty_results, list)
        print("✅ Successfully handled empty query")

    def test_retrieval_consistency(self):
        """Test if retrieval results are consistent"""
        self.vector_store.connect_to_existing_index()
        
        # Make same query multiple times
        query = "What is acne?"
        results1 = self.vector_store.retrieve_documents(query)
        results2 = self.vector_store.retrieve_documents(query)
        
        # Check if number of results is consistent
        self.assertEqual(len(results1), len(results2))
        print("✅ Retrieval results are consistent in length")
        
        # Check if top result is consistent
        self.assertEqual(results1[0].page_content, results2[0].page_content)
        print("✅ Top retrieval result is consistent")

if __name__ == '__main__':
    unittest.main(verbosity=2)