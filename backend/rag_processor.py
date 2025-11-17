"""
RAG Processing utilities for HaleAI
Handles document reranking and context optimization
"""
import logging
from typing import List, Tuple, Optional, Union, Dict
from sentence_transformers import CrossEncoder

try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content: str = "", metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

logger = logging.getLogger(__name__)


class RAGProcessor:
    """Handles RAG-specific processing including reranking"""
    
    def __init__(self, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize RAG processor with reranker
        
        Args:
            reranker_model: Name of the cross-encoder model for reranking
        """
        logger.info(f"Loading reranker model: {reranker_model}")
        
        try:
            self.reranker = CrossEncoder(
                reranker_model,
                max_length=512,
                device='cpu'  # Use CPU for compatibility
            )
            logger.info("✅ Reranker model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load reranker: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize reranker: {str(e)}")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
        return_scores: bool = False
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Rerank documents using cross-encoder for improved relevance
        
        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            return_scores: Whether to return scores with documents
            
        Returns:
            List of top-k reranked documents (optionally with scores)
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        if len(documents) <= top_k:
            logger.info(f"Document count ({len(documents)}) <= top_k ({top_k}), skipping rerank")
            return documents
        
        try:
            # Create query-document pairs for scoring
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Get relevance scores
            scores = self.reranker.predict(pairs)
            
            # Sort by score (descending)
            ranked_pairs = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Get top k
            top_pairs = ranked_pairs[:top_k]
            
            # Log reranking results
            logger.info(f"Reranked {len(documents)} documents to top {len(top_pairs)}")
            logger.debug(f"Top scores: {[f'{score:.4f}' for _, score in top_pairs[:3]]}")
            
            if return_scores:
                return top_pairs
            else:
                return [doc for doc, _ in top_pairs]
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}", exc_info=True)
            # Fallback: return original documents
            return documents[:top_k]
    
    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Rerank documents and return with detailed metadata
        
        Args:
            query: User query
            documents: List of documents
            top_k: Number of top documents
            
        Returns:
            List of dicts with document and relevance info
        """
        ranked = self.rerank(query, documents, top_k, return_scores=True)
        
        results = []
        for i, (doc, score) in enumerate(ranked, 1):
            results.append({
                "rank": i,
                "document": doc,
                "relevance_score": float(score),
                "page": doc.metadata.get('page', 'N/A'),
                "source": doc.metadata.get('source', 'Unknown'),
                "content_preview": doc.page_content[:200] + "..."
            })
        
        return results
    
    def filter_by_relevance_threshold(
        self,
        query: str,
        documents: List[Document],
        threshold: float = 0.5,
        min_docs: int = 1,
        max_docs: int = 5
    ) -> List[Document]:
        """
        Filter documents by relevance score threshold
        
        Args:
            query: User query
            documents: List of documents
            threshold: Minimum relevance score (0-1)
            min_docs: Minimum documents to return
            max_docs: Maximum documents to return
            
        Returns:
            Filtered list of relevant documents
        """
        if not documents:
            return []
        
        # Get scores
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Filter by threshold
        relevant = [
            (doc, score)
            for doc, score in zip(documents, scores)
            if score >= threshold
        ]
        
        # Sort by score
        relevant.sort(key=lambda x: x[1], reverse=True)
        
        # Apply min/max constraints
        if len(relevant) < min_docs:
            # If too few relevant docs, return top min_docs anyway
            all_sorted = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            return [doc for doc, _ in all_sorted[:min_docs]]
        
        # Return up to max_docs
        return [doc for doc, _ in relevant[:max_docs]]
    
    def deduplicate_documents(
        self,
        documents: List[Document],
        similarity_threshold: float = 0.9
    ) -> List[Document]:
        """
        Remove duplicate or highly similar documents
        
        Args:
            documents: List of documents
            similarity_threshold: Threshold for considering docs as duplicates
            
        Returns:
            Deduplicated list of documents
        """
        if len(documents) <= 1:
            return documents
        
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            # Simple deduplication based on content hash
            content_hash = hash(doc.page_content.strip()[:500])
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        removed = len(documents) - len(unique_docs)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate documents")
        
        return unique_docs
    
    def optimize_context_window(
        self,
        query: str,
        documents: List[Document],
        max_tokens: int = 3000,
        chars_per_token: int = 4
    ) -> List[Document]:
        """
        Optimize documents to fit within context window
        
        Args:
            query: User query
            documents: List of documents
            max_tokens: Maximum tokens allowed
            chars_per_token: Approximate characters per token
            
        Returns:
            Optimized list of documents that fit in context
        """
        max_chars = max_tokens * chars_per_token
        
        # Rerank first
        ranked_docs = self.rerank(query, documents, top_k=len(documents))
        
        # Add documents until we hit the limit
        selected_docs = []
        current_chars = 0
        
        for doc in ranked_docs:
            doc_chars = len(doc.page_content)
            
            if current_chars + doc_chars <= max_chars:
                selected_docs.append(doc)
                current_chars += doc_chars
            else:
                # Try to fit a truncated version
                remaining = max_chars - current_chars
                if remaining > 200:  # Minimum useful content
                    truncated = Document(
                        page_content=doc.page_content[:remaining],
                        metadata=doc.metadata
                    )
                    selected_docs.append(truncated)
                break
        
        logger.info(f"Optimized context: {len(selected_docs)} docs, ~{current_chars} chars")
        return selected_docs
    
    def test_reranker(self, query: str = "medical symptoms") -> bool:
        """Test if reranker is working correctly"""
        try:
            test_docs = [
                Document(page_content="Medical symptoms include fever, cough, and fatigue."),
                Document(page_content="The weather is sunny today."),
                Document(page_content="Common symptoms of illness are headache and body pain.")
            ]
            
            results = self.rerank(query, test_docs, top_k=2, return_scores=True)
            
            logger.info("✅ Reranker test passed")
            logger.info(f"   Top document: {results[0][0].page_content[:50]}...")
            logger.info(f"   Score: {results[0][1]:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Reranker test failed: {str(e)}", exc_info=True)
            return False