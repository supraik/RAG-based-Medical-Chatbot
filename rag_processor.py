# rag_processor.py
from sentence_transformers import CrossEncoder
from typing import List
from langchain.schema import Document

class RAGProcessor:
    def __init__(self):
        print("Loading reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        print("Reranker loaded")

    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]