"""Production-grade vector database with ChromaDB and FAISS support."""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ..config import config
from ..models.document import DocumentChunk, QueryResult

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store."""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: np.ndarray, k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[QueryResult]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks from a document."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store for production use."""
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: Optional[Path] = None):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory or config.system.vector_db_path
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "RAG document chunks with embeddings"}
            )
            logger.info(f"Created new ChromaDB collection: {collection_name}")
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to ChromaDB."""
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                continue
            
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.content)
            
            # Prepare metadata (ChromaDB requires string values)
            metadata = {
                "document_id": getattr(chunk, 'document_id', 'unknown'),
                "section_title": chunk.section_title or "",
                "chunk_type": chunk.chunk_type.value,
                "page_number": str(chunk.page_number) if chunk.page_number else "",
                "word_count": str(chunk.word_count),
                "is_title": str(chunk.is_title),
                "is_abstract": str(chunk.is_abstract),
                "contains_citations": str(chunk.contains_citations),
            }
            metadatas.append(metadata)
        
        if not ids:
            logger.warning("No chunks with embeddings to add")
            return
        
        # Add to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(ids)} chunks to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            raise
    
    async def search(self, query_embedding: np.ndarray, k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[QueryResult]:
        """Search for similar chunks in ChromaDB."""
        try:
            # Convert numpy array to list for ChromaDB
            query_embedding_list = query_embedding.tolist()
            
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if isinstance(value, (str, int, float, bool)):
                        where_clause[key] = str(value)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to QueryResult objects
            query_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    # Reconstruct DocumentChunk
                    chunk = DocumentChunk(
                        id=chunk_id,
                        content=results["documents"][0][i],
                        section_title=results["metadatas"][0][i].get("section_title", ""),
                        chunk_type=results["metadatas"][0][i].get("chunk_type", "paragraph"),
                        page_number=int(results["metadatas"][0][i].get("page_number", 0)) or None,
                        word_count=int(results["metadatas"][0][i].get("word_count", 0)),
                        is_title=results["metadatas"][0][i].get("is_title", "False") == "True",
                        is_abstract=results["metadatas"][0][i].get("is_abstract", "False") == "True",
                        contains_citations=results["metadatas"][0][i].get("contains_citations", "False") == "True",
                    )
                    
                    # Create QueryResult
                    query_result = QueryResult(
                        chunk=chunk,
                        document_id=results["metadatas"][0][i].get("document_id", "unknown"),
                        score=1.0 - results["distances"][0][i],  # Convert distance to similarity
                        retrieval_method="semantic"
                    )
                    query_results.append(query_result)
            
            logger.info(f"ChromaDB search returned {len(query_results)} results")
            return query_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks from a document."""
        try:
            # Get all chunk IDs for the document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            else:
                logger.info(f"No chunks found for document {document_id}")
                
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        try:
            count = self.collection.count()
            return {
                "type": "ChromaDB",
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as e:
            logger.error(f"Failed to get ChromaDB stats: {e}")
            return {"type": "ChromaDB", "error": str(e)}


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for high-performance similarity search."""
    
    def __init__(self, dimension: int, index_type: str = "IVF", persist_directory: Optional[Path] = None):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.persist_directory = persist_directory or config.system.vector_db_path
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.index = None
        self.chunk_metadata = {}  # Store chunk metadata separately
        self.id_to_index = {}  # Map chunk IDs to FAISS indices
        self.index_to_id = {}  # Map FAISS indices to chunk IDs
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        index_path = self.persist_directory / "faiss.index"
        metadata_path = self.persist_directory / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load existing index
                self.index = faiss.read_index(str(index_path))
                
                import pickle
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunk_metadata = data.get("chunk_metadata", {})
                    self.id_to_index = data.get("id_to_index", {})
                    self.index_to_id = data.get("index_to_id", {})
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                return
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
        
        # Create new index
        if self.index_type == "IVF":
            # IVF index for large datasets
            quantizer = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, max(1, self.dimension // 4)))
        else:
            # Flat index for smaller datasets
            self.index = faiss.IndexFlatIP(self.dimension)
        
        logger.info(f"Created new FAISS {self.index_type} index")
    
    def _save_index(self):
        """Save index and metadata to disk."""
        try:
            index_path = self.persist_directory / "faiss.index"
            metadata_path = self.persist_directory / "metadata.pkl"
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            import pickle
            data = {
                "chunk_metadata": self.chunk_metadata,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info("FAISS index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to FAISS index."""
        if not chunks:
            return
        
        embeddings = []
        valid_chunks = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                continue
            
            embeddings.append(np.array(chunk.embedding, dtype=np.float32))
            valid_chunks.append(chunk)
        
        if not embeddings:
            logger.warning("No chunks with embeddings to add")
            return
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.vstack(embeddings)
        faiss.normalize_L2(embeddings_array)
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if len(embeddings) >= self.index.nlist:
                self.index.train(embeddings_array)
                logger.info("FAISS index trained")
            else:
                logger.warning(f"Not enough vectors to train IVF index (need {self.index.nlist}, got {len(embeddings)})")
        
        # Add vectors to index
        start_idx = self.index.ntotal
        self.index.add(embeddings_array)
        
        # Store metadata
        for i, chunk in enumerate(valid_chunks):
            faiss_idx = start_idx + i
            self.id_to_index[chunk.id] = faiss_idx
            self.index_to_id[faiss_idx] = chunk.id
            
            # Store chunk metadata
            self.chunk_metadata[chunk.id] = {
                "content": chunk.content,
                "document_id": getattr(chunk, 'document_id', 'unknown'),
                "section_title": chunk.section_title,
                "chunk_type": chunk.chunk_type.value,
                "page_number": chunk.page_number,
                "word_count": chunk.word_count,
                "is_title": chunk.is_title,
                "is_abstract": chunk.is_abstract,
                "contains_citations": chunk.contains_citations,
            }
        
        # Save to disk
        self._save_index()
        
        logger.info(f"Added {len(valid_chunks)} chunks to FAISS index")
    
    async def search(self, query_embedding: np.ndarray, k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[QueryResult]:
        """Search for similar chunks in FAISS index."""
        if self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            # Convert results to QueryResult objects
            query_results = []
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                chunk_id = self.index_to_id.get(idx)
                if not chunk_id or chunk_id not in self.chunk_metadata:
                    continue
                
                metadata = self.chunk_metadata[chunk_id]
                
                # Apply filters if specified
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key in metadata and str(metadata[key]) != str(value):
                            skip = True
                            break
                    if skip:
                        continue
                
                # Reconstruct DocumentChunk
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=metadata["content"],
                    section_title=metadata["section_title"],
                    chunk_type=metadata["chunk_type"],
                    page_number=metadata["page_number"],
                    word_count=metadata["word_count"],
                    is_title=metadata["is_title"],
                    is_abstract=metadata["is_abstract"],
                    contains_citations=metadata["contains_citations"],
                )
                
                # Create QueryResult
                query_result = QueryResult(
                    chunk=chunk,
                    document_id=metadata["document_id"],
                    score=float(score),
                    retrieval_method="semantic"
                )
                query_results.append(query_result)
            
            logger.info(f"FAISS search returned {len(query_results)} results")
            return query_results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks from a document (FAISS doesn't support deletion, so we mark as deleted)."""
        deleted_count = 0
        
        # Find and remove chunks for this document
        chunks_to_remove = []
        for chunk_id, metadata in self.chunk_metadata.items():
            if metadata.get("document_id") == document_id:
                chunks_to_remove.append(chunk_id)
        
        # Remove from metadata
        for chunk_id in chunks_to_remove:
            if chunk_id in self.chunk_metadata:
                del self.chunk_metadata[chunk_id]
                deleted_count += 1
            
            if chunk_id in self.id_to_index:
                faiss_idx = self.id_to_index[chunk_id]
                del self.id_to_index[chunk_id]
                if faiss_idx in self.index_to_id:
                    del self.index_to_id[faiss_idx]
        
        # Save updated metadata
        self._save_index()
        
        logger.info(f"Marked {deleted_count} chunks as deleted for document {document_id}")
        logger.warning("FAISS doesn't support true deletion. Consider rebuilding index periodically.")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get FAISS statistics."""
        return {
            "type": "FAISS",
            "index_type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_chunks": len(self.chunk_metadata),
            "is_trained": getattr(self.index, 'is_trained', True),
            "persist_directory": str(self.persist_directory),
        }


def create_vector_store(store_type: str = "chroma", **kwargs) -> VectorStore:
    """Factory function to create vector store instances."""
    if store_type.lower() == "chroma":
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        # Remove dimension parameter for ChromaDB as it doesn't need it
        chroma_kwargs = {k: v for k, v in kwargs.items() if k != "dimension"}
        return ChromaVectorStore(**chroma_kwargs)
    
    elif store_type.lower() == "faiss":
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        # Get embedding dimension from config or kwargs
        dimension = kwargs.get("dimension", config.model.embedding_dimension)
        return FAISSVectorStore(dimension=dimension, **kwargs)
    
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")


# Global vector store instance (will be initialized by the application)
vector_store: Optional[VectorStore] = None