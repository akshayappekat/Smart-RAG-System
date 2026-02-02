"""Production-grade hybrid retrieval combining semantic and lexical search."""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False

from ..config import config
from ..models.document import DocumentChunk, QueryResult
from ..embeddings.embedding_service import embedding_service
from .vector_store import vector_store

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25-based lexical retrieval for keyword matching."""
    
    def __init__(self):
        if not BM25_AVAILABLE:
            raise ImportError("BM25 not available. Install with: pip install rank-bm25")
        
        self.bm25 = None
        self.chunks = []
        self.chunk_tokens = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        import re
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    async def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks for BM25 search."""
        if not chunks:
            return
        
        self.chunks = chunks
        self.chunk_tokens = []
        
        # Tokenize all chunks
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            self.chunk_tokens.append(tokens)
        
        # Create BM25 index
        if self.chunk_tokens:
            self.bm25 = BM25Okapi(self.chunk_tokens)
            logger.info(f"Indexed {len(chunks)} chunks for BM25 search")
        else:
            logger.warning("No chunks to index for BM25")
    
    async def search(self, query: str, k: int = 10) -> List[QueryResult]:
        """Search using BM25."""
        if not self.bm25 or not self.chunks:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                chunk = self.chunks[idx]
                result = QueryResult(
                    chunk=chunk,
                    document_id=getattr(chunk, 'document_id', 'unknown'),
                    score=float(scores[idx]),
                    retrieval_method="lexical"
                )
                results.append(result)
        
        logger.info(f"BM25 search returned {len(results)} results")
        return results


class CrossEncoderReranker:
    """Cross-encoder based reranking for improved relevance."""
    
    def __init__(self, model_name: Optional[str] = None):
        if not CROSSENCODER_AVAILABLE:
            raise ImportError("CrossEncoder not available. Install with: pip install sentence-transformers")
        
        self.model_name = model_name or config.model.reranker_model
        self.model = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self.model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        return self.model
    
    async def rerank(self, query: str, results: List[QueryResult], top_k: int = None) -> List[QueryResult]:
        """Rerank results using cross-encoder."""
        if not results:
            return results
        
        if top_k is None:
            top_k = len(results)
        
        # Prepare query-document pairs
        pairs = []
        for result in results:
            pairs.append([query, result.chunk.content])
        
        # Get cross-encoder scores
        model = self._load_model()
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(pairs)
        )
        
        # Update results with new scores
        for result, score in zip(results, scores):
            result.score = float(score)
            result.retrieval_method = "reranked"
        
        # Sort by new scores and return top-k
        reranked_results = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        
        logger.info(f"Reranked {len(results)} results, returning top {len(reranked_results)}")
        return reranked_results


class HybridRetriever:
    """Production-grade hybrid retriever combining semantic, lexical, and reranking."""
    
    def __init__(self, enable_reranking: bool = True):
        self.bm25_retriever = BM25Retriever() if BM25_AVAILABLE else None
        self.reranker = CrossEncoderReranker() if enable_reranking and CROSSENCODER_AVAILABLE else None
        self.semantic_weight = config.retrieval.semantic_weight
        self.lexical_weight = config.retrieval.lexical_weight
        self.rerank_top_k = config.retrieval.rerank_top_k
        self.final_top_k = config.retrieval.final_top_k
        self.vector_store = None  # Will be set by RAG orchestrator
        
        logger.info(f"Initialized HybridRetriever (BM25: {self.bm25_retriever is not None}, "
                   f"Reranker: {self.reranker is not None})")
    
    async def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks for both semantic and lexical search."""
        if not chunks:
            return
        
        # Add document_id to chunks if not present
        for chunk in chunks:
            if not hasattr(chunk, 'document_id'):
                chunk.document_id = 'unknown'
        
        # Generate embeddings if not present
        chunks_without_embeddings = [chunk for chunk in chunks if chunk.embedding is None]
        if chunks_without_embeddings:
            logger.info(f"Generating embeddings for {len(chunks_without_embeddings)} chunks")
            await embedding_service.embed_chunks(chunks_without_embeddings)
        
        # Index in vector store
        if self.vector_store:
            await self.vector_store.add_chunks(chunks)
        else:
            logger.warning("Vector store not available - only BM25 search will work")
        
        # Index in BM25
        if self.bm25_retriever:
            await self.bm25_retriever.index_chunks(chunks)
        
        logger.info(f"Indexed {len(chunks)} chunks for hybrid retrieval")
    
    async def search(self, query: str, k: Optional[int] = None, 
                    filters: Optional[Dict[str, Any]] = None,
                    enable_reranking: bool = True) -> List[QueryResult]:
        """Perform hybrid search combining semantic and lexical retrieval."""
        if k is None:
            k = self.final_top_k
        
        # Get more results for reranking
        retrieval_k = self.rerank_top_k if enable_reranking and self.reranker else k
        
        # Semantic search
        semantic_results = []
        if self.vector_store:
            try:
                query_embedding = await embedding_service.embed_query(query)
                semantic_results = await self.vector_store.search(
                    query_embedding, 
                    k=retrieval_k, 
                    filters=filters
                )
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
        else:
            logger.warning("Vector store not available - skipping semantic search")
        
        # Lexical search
        lexical_results = []
        if self.bm25_retriever:
            try:
                lexical_results = await self.bm25_retriever.search(query, k=retrieval_k)
                
                # Apply filters to lexical results
                if filters:
                    filtered_lexical = []
                    for result in lexical_results:
                        chunk_metadata = {
                            "document_id": result.document_id,
                            "section_title": result.chunk.section_title,
                            "chunk_type": result.chunk.chunk_type.value,
                            "is_title": result.chunk.is_title,
                            "is_abstract": result.chunk.is_abstract,
                        }
                        
                        # Check if chunk matches filters
                        matches = True
                        for key, value in filters.items():
                            if key in chunk_metadata and str(chunk_metadata[key]) != str(value):
                                matches = False
                                break
                        
                        if matches:
                            filtered_lexical.append(result)
                    
                    lexical_results = filtered_lexical
                    
            except Exception as e:
                logger.error(f"Lexical search failed: {e}")
        
        # Combine results using weighted scoring
        combined_results = self._combine_results(semantic_results, lexical_results)
        
        # Rerank if enabled
        if enable_reranking and self.reranker and combined_results:
            try:
                combined_results = await self.reranker.rerank(
                    query, 
                    combined_results, 
                    top_k=k
                )
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                # Fall back to combined results without reranking
                combined_results = combined_results[:k]
        else:
            combined_results = combined_results[:k]
        
        logger.info(f"Hybrid search returned {len(combined_results)} results "
                   f"(semantic: {len(semantic_results)}, lexical: {len(lexical_results)})")
        
        return combined_results
    
    def _combine_results(self, semantic_results: List[QueryResult], 
                        lexical_results: List[QueryResult]) -> List[QueryResult]:
        """Combine semantic and lexical results with weighted scoring."""
        # Create a map of chunk_id -> results for deduplication
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result.chunk.id
            result_map[chunk_id] = result
            result.score = result.score * self.semantic_weight
        
        # Add or update with lexical results
        for result in lexical_results:
            chunk_id = result.chunk.id
            if chunk_id in result_map:
                # Combine scores
                existing_result = result_map[chunk_id]
                existing_result.score += result.score * self.lexical_weight
                existing_result.retrieval_method = "hybrid"
            else:
                # New result from lexical search
                result.score = result.score * self.lexical_weight
                result_map[chunk_id] = result
        
        # Sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        stats = {
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight,
            "rerank_top_k": self.rerank_top_k,
            "final_top_k": self.final_top_k,
            "bm25_available": self.bm25_retriever is not None,
            "reranker_available": self.reranker is not None,
        }
        
        # Add vector store stats
        if self.vector_store:
            vector_stats = await self.vector_store.get_stats()
            stats["vector_store"] = vector_stats
        
        # Add BM25 stats
        if self.bm25_retriever:
            stats["bm25_chunks"] = len(self.bm25_retriever.chunks)
        
        return stats


# Global hybrid retriever instance
hybrid_retriever = HybridRetriever()