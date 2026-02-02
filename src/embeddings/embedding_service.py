"""Production-grade embedding service with caching and batch processing."""

import asyncio
import hashlib
import pickle
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer
import torch

from ..config import config
from ..models.document import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """High-performance embedding service with intelligent caching."""
    
    def __init__(self, model_name: Optional[str] = None, enable_cache: bool = True, offline_mode: bool = False):
        self.model_name = model_name or config.model.embedding_model
        self.enable_cache = enable_cache and config.system.enable_caching
        self.offline_mode = offline_mode
        self.cache_dir = config.system.cache_dir / "embeddings"
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        if self.enable_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        if self.offline_mode:
            logger.info("EmbeddingService running in offline mode - using mock embeddings")
        else:
            logger.info(f"Initialized EmbeddingService with model: {self.model_name}, device: {self.device}")
    
    def _load_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self.offline_mode:
            return None
            
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Model loaded successfully. Dimension: {self.model.get_sentence_embedding_dimension()}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Falling back to offline mode")
                self.offline_mode = True
                return None
        return self.model
    
    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """Generate a mock embedding for offline mode."""
        import hashlib
        # Create deterministic embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hex to numbers and normalize
        numbers = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
        # Pad or truncate to 384 dimensions
        while len(numbers) < 384:
            numbers.extend(numbers[:384-len(numbers)])
        embedding = np.array(numbers[:384], dtype=np.float32)
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        content = f"{self.model_name}_{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache."""
        if not self.enable_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding {cache_key}: {e}")
        return None
    
    def _cache_embedding(self, cache_key: str, embedding: np.ndarray) -> None:
        """Cache embedding to disk."""
        if not self.enable_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to cache embedding {cache_key}: {e}")
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_embedding = self._get_cached_embedding(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate embedding
        if self.offline_mode:
            embedding = self._generate_mock_embedding(text)
        else:
            model = self._load_model()
            if model is None:
                # Fallback to mock if model loading failed
                embedding = self._generate_mock_embedding(text)
            else:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    self.executor, 
                    lambda: model.encode([text], convert_to_numpy=True)[0]
                )
        
        # Cache the result
        self._cache_embedding(cache_key, embedding)
        
        return embedding
    
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with batching."""
        if not texts:
            return []
        
        embeddings = []
        cached_count = 0
        
        # Check cache for all texts
        cache_keys = [self._get_cache_key(text) for text in texts]
        cached_embeddings = {}
        
        for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
            cached_embedding = self._get_cached_embedding(cache_key)
            if cached_embedding is not None:
                cached_embeddings[i] = cached_embedding
                cached_count += 1
        
        # Identify texts that need embedding
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            if i not in cached_embeddings:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        logger.info(f"Embedding batch: {len(texts)} total, {cached_count} cached, {len(texts_to_embed)} to compute")
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if texts_to_embed:
            model = self._load_model()
            
            # Process in batches
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i + batch_size]
                batch_indices = indices_to_embed[i:i + batch_size]
                
                # Run in thread pool
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    self.executor,
                    lambda: model.encode(batch_texts, convert_to_numpy=True)
                )
                
                # Cache new embeddings
                for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    cache_key = self._get_cache_key(text)
                    self._cache_embedding(cache_key, embedding)
                    new_embeddings.append((batch_indices[j], embedding))
        
        # Combine cached and new embeddings in correct order
        result_embeddings = [None] * len(texts)
        
        # Add cached embeddings
        for i, embedding in cached_embeddings.items():
            result_embeddings[i] = embedding
        
        # Add new embeddings
        for i, embedding in new_embeddings:
            result_embeddings[i] = embedding
        
        return result_embeddings
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for document chunks and update them in place."""
        if not chunks:
            return chunks
        
        # Extract texts from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.embed_texts(texts)
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()  # Convert to list for JSON serialization
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query (alias for embed_text)."""
        return await self.embed_text(query)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model."""
        if self.offline_mode:
            return 384  # Standard dimension for mock embeddings
        
        model = self._load_model()
        if model is None:
            return 384  # Fallback dimension
        return model.get_sentence_embedding_dimension()
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def compute_similarities(self, query_embedding: np.ndarray, 
                           embeddings: List[np.ndarray]) -> List[float]:
        """Compute similarities between query and multiple embeddings."""
        similarities = []
        for embedding in embeddings:
            similarity = self.compute_similarity(query_embedding, embedding)
            similarities.append(similarity)
        return similarities
    
    async def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if not self.enable_cache:
            return
        
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear embedding cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enable_cache or not self.cache_dir.exists():
            return {"enabled": False}
        
        cache_files = list(self.cache_dir.glob("*.npy"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "cached_embeddings": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
        }


# Global embedding service instance
embedding_service = EmbeddingService()