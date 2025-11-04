"""
Data processing utilities for HaleAI
Enhanced PDF processing, chunking, and embeddings management
"""
import logging
from typing import List
from pathlib import Path
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Try multiple embeddings imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        from langchain.embeddings import HuggingFaceEmbeddings

# Document import
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

from config import Config

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles document processing, chunking, and embeddings"""
    
    def __init__(self):
        """Initialize data processor with embeddings"""
        logger.info("Initializing DataProcessor...")
        self.embeddings = self._load_embeddings()
        self.text_splitter = self._create_text_splitter()
        logger.info("✅ DataProcessor initialized")
    
    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """Load HuggingFace embeddings model"""
        logger.info(f"Loading embeddings model: {Config.EMBEDDING_MODEL}")
        
        try:
            model = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
            )
            logger.info("✅ Embeddings model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load embeddings model: {str(e)}")
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create text splitter with optimized settings for medical content"""
        return RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",     # Line breaks
                ". ",     # Sentences
                "? ",     # Questions
                "! ",     # Exclamations
                "; ",     # Semicolons
                ", ",     # Commas
                " ",      # Spaces
                ""        # Characters
            ],
            keep_separator=True
        )
    
    def process_documents(self, data_dir: str) -> List[Document]:
        """
        Process all PDF documents in the data directory
        
        Args:
            data_dir: Path to directory containing PDF files
            
        Returns:
            List of processed document chunks with metadata
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find all PDF files
        pdf_files = list(data_path.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {data_dir}")
        
        logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
        
        all_chunks = []
        
        # Process each PDF
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                chunks = self._process_single_pdf(pdf_path)
                all_chunks.extend(chunks)
                logger.info(f"✅ Processed {pdf_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {str(e)}")
                continue
        
        if not all_chunks:
            raise ValueError("No content was successfully processed from any PDF")
        
        logger.info(f"✅ Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _process_single_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of document chunks
        """
        logger.info(f"Loading PDF: {pdf_path.name}")
        
        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        if not pages:
            raise ValueError(f"No pages loaded from {pdf_path.name}")
        
        logger.info(f"Loaded {len(pages)} pages from {pdf_path.name}")
        
        # Process each page
        chunks = []
        
        for page_num, page in enumerate(pages, 1):
            # Clean text
            text = self._clean_text(page.page_content)
            
            if not text.strip():
                logger.warning(f"Empty page {page_num} in {pdf_path.name}")
                continue
            
            # Split into chunks
            split_chunks = self.text_splitter.split_text(text)
            
            # Create Document objects with metadata
            for chunk_idx, chunk_text in enumerate(split_chunks):
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk": chunk_idx,
                        "total_chunks_on_page": len(split_chunks)
                    }
                ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace("\x00", "")  # Null bytes
        text = text.replace("\uf0b7", "")  # Bullet points
        
        # Normalize line breaks
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        
        return text.strip()
    
    def process_directory(self, data_dir: str) -> List[Document]:
        """
        Alternative method: Process entire directory at once
        
        Args:
            data_dir: Path to directory containing PDFs
            
        Returns:
            List of processed documents
        """
        logger.info(f"Processing directory: {data_dir}")
        
        # Load all PDFs using DirectoryLoader
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        pages = loader.load()
        logger.info(f"Loaded {len(pages)} pages from all PDFs")
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(pages)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        test_embedding = self.embeddings.embed_query("test")
        return len(test_embedding)
    
    def test_embeddings(self) -> bool:
        """Test if embeddings are working correctly"""
        try:
            test_text = "This is a test sentence for medical embeddings."
            embedding = self.embeddings.embed_query(test_text)
            
            logger.info(f"✅ Embeddings test passed")
            logger.info(f"   Embedding dimension: {len(embedding)}")
            logger.info(f"   Sample values: {embedding[:5]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Embeddings test failed: {str(e)}", exc_info=True)
            return False