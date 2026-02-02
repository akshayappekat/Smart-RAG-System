"""Production-grade RAG orchestrator for enterprise knowledge systems."""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any, AsyncGenerator
from pathlib import Path

from .config import config
from .models.document import Document, DocumentChunk, QueryResult, RAGResponse
from .processing.document_processor import DocumentProcessor
from .embeddings.embedding_service import embedding_service
from .retrieval.vector_store import create_vector_store, vector_store
from .retrieval.hybrid_retriever import hybrid_retriever
from .llm.providers import llm_manager
from .analytics import analytics_dashboard

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Production-grade RAG system orchestrator."""
    
    def __init__(self, vector_store_type: str = "chroma"):
        self.document_processor = DocumentProcessor()
        self.vector_store_type = vector_store_type
        self.knowledge_base = {}  # document_id -> Document mapping
        self.is_initialized = False
        
        logger.info(f"Initialized RAG Orchestrator with {vector_store_type} vector store")
    
    async def initialize(self) -> None:
        """Initialize the RAG system components."""
        if self.is_initialized:
            return
        
        try:
            # Initialize vector store
            global vector_store
            if vector_store is None:
                vector_store = create_vector_store(
                    store_type=self.vector_store_type,
                    dimension=embedding_service.get_embedding_dimension()
                )
                logger.info(f"Initialized {self.vector_store_type} vector store")
            
            # Update hybrid retriever with vector store
            from .retrieval.hybrid_retriever import hybrid_retriever
            hybrid_retriever.vector_store = vector_store
            
            # Test embedding service
            test_embedding = await embedding_service.embed_text("test")
            logger.info(f"Embedding service ready (dimension: {len(test_embedding)})")
            
            # Test LLM
            model_info = llm_manager.get_model_info()
            logger.info(f"LLM ready: {model_info}")
            
            self.is_initialized = True
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    async def ingest_document(self, file_path: Path, document_id: Optional[str] = None) -> Document:
        """Ingest a single document into the knowledge base."""
        await self.initialize()
        
        try:
            # Process document
            logger.info(f"Processing document: {file_path}")
            document = await self.document_processor.process_file(file_path)
            
            if not document.is_processed:
                raise ValueError(f"Failed to process document: {document.processing_errors}")
            
            # Set document ID
            if document_id:
                document.id = document_id
            
            # Add document_id to chunks
            for chunk in document.chunks:
                chunk.document_id = document.id
            
            # Index chunks for retrieval
            await hybrid_retriever.index_documents(document.chunks)
            
            # Store in knowledge base
            self.knowledge_base[document.id] = document
            
            logger.info(f"Successfully ingested document {document.id} with {len(document.chunks)} chunks")
            return document
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {e}")
            raise
    
    async def ingest_documents_batch(self, file_paths: List[Path], 
                                   progress_callback: Optional[callable] = None) -> List[Document]:
        """Ingest multiple documents in batch."""
        await self.initialize()
        
        try:
            # Process documents
            logger.info(f"Batch processing {len(file_paths)} documents")
            documents = await self.document_processor.process_files_batch(
                file_paths, progress_callback
            )
            
            # Filter successfully processed documents
            valid_documents = [doc for doc in documents if doc.is_processed]
            
            if not valid_documents:
                logger.warning("No documents were successfully processed")
                return documents
            
            # Prepare all chunks for indexing
            all_chunks = []
            for document in valid_documents:
                # Add document_id to chunks
                for chunk in document.chunks:
                    chunk.document_id = document.id
                all_chunks.extend(document.chunks)
            
            # Index all chunks at once for efficiency
            if all_chunks:
                await hybrid_retriever.index_documents(all_chunks)
            
            # Store in knowledge base
            for document in valid_documents:
                self.knowledge_base[document.id] = document
            
            logger.info(f"Successfully ingested {len(valid_documents)} documents "
                       f"with {len(all_chunks)} total chunks")
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to batch ingest documents: {e}")
            raise
    
    async def query(self, query: str, 
                   max_chunks: Optional[int] = None,
                   filters: Optional[Dict[str, Any]] = None,
                   enable_reranking: bool = True,
                   stream: bool = False) -> RAGResponse:
        """Process a query and generate response."""
        await self.initialize()
        
        start_time = time.time()
        reasoning_steps = []
        query_id = f"query_{int(time.time() * 1000)}"
        
        # Track timing for analytics
        retrieval_start = 0
        llm_start = 0
        embedding_time = 0
        
        try:
            # Step 1: Retrieve relevant chunks
            reasoning_steps.append("Retrieving relevant document chunks")
            logger.info(f"Processing query: {query[:100]}...")
            
            retrieval_start = time.time()
            k = max_chunks or config.retrieval.final_top_k
            retrieved_chunks = await hybrid_retriever.search(
                query, k=k, filters=filters, enable_reranking=enable_reranking
            )
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_chunks:
                reasoning_steps.append("No relevant chunks found")
                response = RAGResponse(
                    query=query,
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time,
                    reasoning_steps=reasoning_steps
                )
                
                # Record analytics
                analytics_dashboard.record_query_metrics(
                    query_id=query_id,
                    query_text=query,
                    processing_time=response.processing_time,
                    confidence_score=response.confidence_score,
                    sources_count=len(response.sources),
                    retrieval_time=retrieval_time,
                    llm_time=0,
                    embedding_time=embedding_time,
                    cache_hit=False
                )
                
                return response
            
            reasoning_steps.append(f"Found {len(retrieved_chunks)} relevant chunks")
            
            # Step 2: Prepare context
            reasoning_steps.append("Preparing context from retrieved chunks")
            context = self._prepare_context(retrieved_chunks)
            
            # Step 3: Generate response
            reasoning_steps.append("Generating response using LLM")
            
            messages = self._create_messages(query, context, retrieved_chunks)
            
            if stream:
                # For streaming, we need to handle differently
                response_generator = await llm_manager.generate(
                    messages, stream=True, use_cache=False
                )
                
                # Return a special streaming response
                response = RAGResponse(
                    query=query,
                    answer="",  # Will be filled by streaming
                    sources=retrieved_chunks,
                    confidence_score=self._calculate_confidence(retrieved_chunks),
                    processing_time=time.time() - start_time,
                    reasoning_steps=reasoning_steps
                )
                
                # Record analytics for streaming
                analytics_dashboard.record_query_metrics(
                    query_id=query_id,
                    query_text=query,
                    processing_time=response.processing_time,
                    confidence_score=response.confidence_score,
                    sources_count=len(response.sources),
                    retrieval_time=retrieval_time,
                    llm_time=0,  # Can't measure for streaming
                    embedding_time=embedding_time,
                    cache_hit=False
                )
                
                return response
            else:
                llm_start = time.time()
                answer = await llm_manager.generate(messages, use_cache=True)
                llm_time = time.time() - llm_start
                
                # Step 4: Calculate confidence
                confidence_score = self._calculate_confidence(retrieved_chunks)
                
                processing_time = time.time() - start_time
                reasoning_steps.append(f"Response generated in {processing_time:.2f}s")
                
                response = RAGResponse(
                    query=query,
                    answer=answer,
                    sources=retrieved_chunks,
                    confidence_score=confidence_score,
                    processing_time=processing_time,
                    reasoning_steps=reasoning_steps
                )
                
                # Record analytics
                analytics_dashboard.record_query_metrics(
                    query_id=query_id,
                    query_text=query,
                    processing_time=processing_time,
                    confidence_score=confidence_score,
                    sources_count=len(retrieved_chunks),
                    retrieval_time=retrieval_time,
                    llm_time=llm_time,
                    embedding_time=embedding_time,
                    cache_hit=False  # TODO: Implement cache hit detection
                )
                
                return response
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            reasoning_steps.append(f"Error: {str(e)}")
            
            response = RAGResponse(
                query=query,
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                reasoning_steps=reasoning_steps
            )
            
            # Record analytics for failed queries
            analytics_dashboard.record_query_metrics(
                query_id=query_id,
                query_text=query,
                processing_time=response.processing_time,
                confidence_score=response.confidence_score,
                sources_count=0,
                retrieval_time=retrieval_time,
                llm_time=0,
                embedding_time=embedding_time,
                cache_hit=False
            )
            
            return response
    
    async def query_stream(self, query: str, 
                          max_chunks: Optional[int] = None,
                          filters: Optional[Dict[str, Any]] = None,
                          enable_reranking: bool = True) -> AsyncGenerator[str, None]:
        """Stream query response."""
        await self.initialize()
        
        try:
            # Retrieve chunks (same as regular query)
            k = max_chunks or config.retrieval.final_top_k
            retrieved_chunks = await hybrid_retriever.search(
                query, k=k, filters=filters, enable_reranking=enable_reranking
            )
            
            if not retrieved_chunks:
                yield "I couldn't find any relevant information to answer your question."
                return
            
            # Prepare context and messages
            context = self._prepare_context(retrieved_chunks)
            messages = self._create_messages(query, context, retrieved_chunks)
            
            # Stream response
            response_generator = await llm_manager.generate(
                messages, stream=True, use_cache=False
            )
            
            async for chunk in response_generator:
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"Error: {str(e)}"
    
    def _prepare_context(self, retrieved_chunks: List[QueryResult]) -> str:
        """Prepare context from retrieved chunks."""
        context_parts = []
        
        for i, result in enumerate(retrieved_chunks):
            chunk = result.chunk
            
            # Add source information
            source_info = f"Source {i+1}"
            if chunk.section_title:
                source_info += f" ({chunk.section_title})"
            if result.document_id != "unknown":
                source_info += f" [Doc: {result.document_id}]"
            
            context_parts.append(f"{source_info}:\n{chunk.content}\n")
        
        return "\n".join(context_parts)
    
    def _create_messages(self, query: str, context: str, 
                        retrieved_chunks: List[QueryResult]) -> List[Dict[str, str]]:
        """Create messages for LLM."""
        # System prompt for RAG
        system_prompt = """You are an intelligent assistant that answers questions based on provided context from documents. 

Guidelines:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite sources when possible (e.g., "According to Source 1...")
4. Be precise and factual
5. If you find contradictions in the sources, mention them
6. For academic/research questions, maintain scholarly tone
7. If asked about multiple documents, compare and synthesize information

Context quality indicators:
- Higher numbered sources may be more relevant
- Pay attention to section titles for context
- Consider document types (research papers, reports, etc.)"""
        
        # Add document type context if available
        doc_types = set()
        for result in retrieved_chunks:
            if hasattr(result.chunk, 'document_id') and result.document_id in self.knowledge_base:
                doc = self.knowledge_base[result.document_id]
                if doc.metadata.document_type:
                    doc_types.add(doc.metadata.document_type.value)
        
        if doc_types:
            system_prompt += f"\n\nDocument types in context: {', '.join(doc_types)}"
        
        # User message with context and query
        user_message = f"""Context from relevant documents:

{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    
    def _calculate_confidence(self, retrieved_chunks: List[QueryResult]) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not retrieved_chunks:
            return 0.0
        
        # Factors for confidence calculation
        factors = []
        
        # Average retrieval score
        avg_score = sum(result.score for result in retrieved_chunks) / len(retrieved_chunks)
        factors.append(min(avg_score, 1.0))
        
        # Number of sources (more sources = higher confidence, up to a point)
        source_factor = min(len(retrieved_chunks) / 5.0, 1.0)
        factors.append(source_factor)
        
        # Diversity of sources (different documents)
        unique_docs = len(set(result.document_id for result in retrieved_chunks))
        diversity_factor = min(unique_docs / 3.0, 1.0)
        factors.append(diversity_factor)
        
        # Content quality indicators
        quality_indicators = 0
        total_chunks = len(retrieved_chunks)
        
        for result in retrieved_chunks:
            chunk = result.chunk
            if chunk.is_abstract or chunk.is_title:
                quality_indicators += 1
            if chunk.contains_citations:
                quality_indicators += 0.5
            if chunk.word_count > 100:  # Substantial content
                quality_indicators += 0.5
        
        quality_factor = min(quality_indicators / total_chunks, 1.0)
        factors.append(quality_factor)
        
        # Calculate weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # Prioritize retrieval score
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return min(confidence, 1.0)
    
    async def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document."""
        if document_id not in self.knowledge_base:
            return None
        
        document = self.knowledge_base[document_id]
        return document.get_summary_info()
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the knowledge base."""
        return [doc.get_summary_info() for doc in self.knowledge_base.values()]
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the knowledge base."""
        if document_id not in self.knowledge_base:
            return False
        
        try:
            # Remove from vector store
            if vector_store:
                await vector_store.delete_document(document_id)
            
            # Remove from knowledge base
            del self.knowledge_base[document_id]
            
            logger.info(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "documents": len(self.knowledge_base),
            "total_chunks": sum(len(doc.chunks) for doc in self.knowledge_base.values()),
            "is_initialized": self.is_initialized,
        }
        
        # Add component stats
        if self.is_initialized:
            stats["embedding_service"] = embedding_service.get_cache_stats()
            stats["llm_manager"] = llm_manager.get_cache_stats()
            stats["retrieval"] = await hybrid_retriever.get_stats()
            
            if vector_store:
                stats["vector_store"] = await vector_store.get_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time(),
        }
        
        try:
            # Test embedding service
            test_embedding = await embedding_service.embed_text("health check")
            health["components"]["embedding_service"] = {
                "status": "healthy",
                "dimension": len(test_embedding)
            }
        except Exception as e:
            health["components"]["embedding_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        try:
            # Test LLM
            model_info = llm_manager.get_model_info()
            health["components"]["llm"] = {
                "status": "healthy",
                "model": model_info
            }
        except Exception as e:
            health["components"]["llm"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        try:
            # Test vector store
            if vector_store:
                vector_stats = await vector_store.get_stats()
                health["components"]["vector_store"] = {
                    "status": "healthy",
                    "stats": vector_stats
                }
            else:
                health["components"]["vector_store"] = {
                    "status": "not_initialized"
                }
        except Exception as e:
            health["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        return health


# Global RAG orchestrator instance
rag_orchestrator = RAGOrchestrator()