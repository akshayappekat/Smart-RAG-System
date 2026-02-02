"""Production-grade FastAPI application for the RAG system."""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
import tempfile
import os

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from ..rag_orchestrator import rag_orchestrator
from ..agents.multi_agent_orchestrator import multi_agent_orchestrator
from ..memory.conversation_memory import conversation_memory
from ..evaluation.hallucination_detector import hallucination_detector
from ..config import config
from ..analytics import analytics_dashboard

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.system.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    use_multi_agent: bool = Field(True, description="Use multi-agent system")
    max_chunks: Optional[int] = Field(None, description="Maximum number of chunks to retrieve")
    max_tokens: Optional[int] = Field(1000, description="Maximum response tokens")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters for document search")
    enable_reranking: bool = Field(True, description="Enable cross-encoder reranking")
    include_reasoning: bool = Field(False, description="Include reasoning chain")
    detect_hallucination: bool = Field(True, description="Run hallucination detection")
    stream: bool = Field(False, description="Stream the response")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    reasoning_steps: List[str]
    session_id: Optional[str] = None
    system_type: str = "standard"
    hallucination_check: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}

class DocumentInfo(BaseModel):
    id: str
    title: Optional[str]
    authors: List[str]
    chunk_count: int
    word_count: int
    is_processed: bool
    is_embedded: bool
    created_at: str

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    processing_time: float
    errors: List[str]

class SystemStats(BaseModel):
    documents: int
    total_chunks: int
    is_initialized: bool
    components: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, Any]
    timestamp: float

class AnalyticsResponse(BaseModel):
    performance_summary: Dict[str, Any]
    query_patterns: Dict[str, Any]
    retrieval_analysis: Dict[str, Any]
    system_health: Dict[str, Any]

# Create FastAPI app
app = FastAPI(
    title="Advanced RAG System",
    description="Production-grade Retrieval-Augmented Generation system for enterprise knowledge",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    logger.info("Starting RAG system...")
    try:
        await rag_orchestrator.initialize()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Perform system health check."""
    try:
        health = await rag_orchestrator.health_check()
        return HealthResponse(**health)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System statistics
@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get comprehensive system statistics."""
    try:
        stats = await rag_orchestrator.get_system_stats()
        return SystemStats(**stats)
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the knowledge base."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = await rag_orchestrator.query(
            query=request.query,
            max_chunks=request.max_chunks,
            filters=request.filters,
            enable_reranking=request.enable_reranking,
            stream=False
        )
        
        # Convert sources to serializable format
        sources = []
        for source in response.sources:
            source_dict = {
                "chunk_id": source.chunk.id,
                "content": source.chunk.content,
                "document_id": source.document_id,
                "score": source.score,
                "retrieval_method": source.retrieval_method,
                "section_title": source.chunk.section_title,
                "chunk_type": source.chunk.chunk_type.value,
                "page_number": source.chunk.page_number,
                "word_count": source.chunk.word_count,
            }
            sources.append(source_dict)
        
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            sources=sources,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            reasoning_steps=response.reasoning_steps
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming query endpoint
@app.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """Query the knowledge base with streaming response."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        async def generate_response():
            async for chunk in rag_orchestrator.query_stream(
                query=request.query,
                max_chunks=request.max_chunks,
                filters=request.filters,
                enable_reranking=request.enable_reranking
            ):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        logger.error(f"Streaming query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document upload endpoint
@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: Optional[str] = None
):
    """Upload and process a document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    allowed_extensions = {'.pdf', '.docx', '.txt', '.html', '.md'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = Path(temp_file.name)
        
        start_time = time.time()
        
        # Process document
        document = await rag_orchestrator.ingest_document(
            file_path=temp_file_path,
            document_id=document_id
        )
        
        processing_time = time.time() - start_time
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return UploadResponse(
            document_id=document.id,
            filename=file.filename,
            status="processed" if document.is_processed else "failed",
            chunk_count=len(document.chunks),
            processing_time=processing_time,
            errors=document.processing_errors
        )
        
    except Exception as e:
        # Clean up temp file on error
        if 'temp_file_path' in locals() and temp_file_path.exists():
            os.unlink(temp_file_path)
        
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch upload endpoint
@app.post("/documents/upload/batch")
async def upload_documents_batch(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    temp_files = []
    try:
        # Save all files temporarily
        file_paths = []
        for file in files:
            if not file.filename:
                continue
            
            file_extension = Path(file.filename).suffix.lower()
            allowed_extensions = {'.pdf', '.docx', '.txt', '.html', '.md'}
            if file_extension not in allowed_extensions:
                continue
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = Path(temp_file.name)
                temp_files.append(temp_file_path)
                file_paths.append(temp_file_path)
        
        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid files to process")
        
        start_time = time.time()
        
        # Process documents in batch
        documents = await rag_orchestrator.ingest_documents_batch(file_paths)
        
        processing_time = time.time() - start_time
        
        # Prepare response
        results = []
        for i, (document, original_file) in enumerate(zip(documents, files)):
            if i < len(file_paths):  # Ensure we have a corresponding file
                results.append({
                    "document_id": document.id,
                    "filename": original_file.filename,
                    "status": "processed" if document.is_processed else "failed",
                    "chunk_count": len(document.chunks),
                    "errors": document.processing_errors
                })
        
        return {
            "total_files": len(files),
            "processed_files": len(results),
            "processing_time": processing_time,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if temp_file.exists():
                os.unlink(temp_file)

# List documents
@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all documents in the knowledge base."""
    try:
        documents = await rag_orchestrator.list_documents()
        return [DocumentInfo(**doc) for doc in documents]
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get document info
@app.get("/documents/{document_id}", response_model=DocumentInfo)
async def get_document_info(document_id: str):
    """Get information about a specific document."""
    try:
        doc_info = await rag_orchestrator.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentInfo(**doc_info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Delete document
@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the knowledge base."""
    try:
        success = await rag_orchestrator.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": f"Document {document_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/analytics/dashboard", response_model=AnalyticsResponse)
async def get_analytics_dashboard(hours: int = 24):
    """Get comprehensive analytics dashboard data."""
    try:
        performance_summary = analytics_dashboard.get_performance_summary(hours)
        query_patterns = analytics_dashboard.get_query_patterns(hours)
        retrieval_analysis = analytics_dashboard.get_retrieval_analysis(hours)
        system_health = analytics_dashboard.get_system_health_report()
        
        return AnalyticsResponse(
            performance_summary=performance_summary,
            query_patterns=query_patterns,
            retrieval_analysis=retrieval_analysis,
            system_health=system_health
        )
    except Exception as e:
        logger.error(f"Failed to get analytics dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/performance")
async def get_performance_metrics(hours: int = 24):
    """Get performance metrics for the specified time period."""
    try:
        return analytics_dashboard.get_performance_summary(hours)
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/queries")
async def get_query_analytics(hours: int = 24):
    """Get query pattern analysis."""
    try:
        return analytics_dashboard.get_query_patterns(hours)
    except Exception as e:
        logger.error(f"Failed to get query analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/retrieval")
async def get_retrieval_analytics(hours: int = 24):
    """Get retrieval performance analysis."""
    try:
        return analytics_dashboard.get_retrieval_analysis(hours)
    except Exception as e:
        logger.error(f"Failed to get retrieval analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/export")
async def export_analytics_data(format: str = "json"):
    """Export analytics data."""
    try:
        if format.lower() not in ["json"]:
            raise HTTPException(status_code=400, detail="Only JSON format is currently supported")
        
        data = analytics_dashboard.export_metrics(format)
        
        return StreamingResponse(
            iter([data]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=analytics_export.{format}"}
        )
    except Exception as e:
        logger.error(f"Failed to export analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Multi-Agent System Endpoints
@app.post("/multi-agent/query", response_model=QueryResponse)
async def multi_agent_query(request: QueryRequest):
    """Query using the multi-agent system."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        start_time = time.time()
        
        # Use multi-agent system
        response = await multi_agent_orchestrator.process_query(request.query)
        
        # Add to conversation if session provided
        if request.session_id:
            await conversation_memory.add_turn(
                request.session_id,
                request.query,
                response.final_answer,
                response.confidence,
                response.sources_used,
                response.total_execution_time
            )
        
        # Hallucination detection
        hallucination_result = None
        if request.detect_hallucination and response.success:
            sources = []
            for agent_response in response.agent_responses.values():
                if agent_response.success and isinstance(agent_response.result, dict):
                    if "sources" in agent_response.result:
                        sources.extend(agent_response.result["sources"])
            
            hallucination_result = await hallucination_detector.detect_hallucination(
                request.query, response.final_answer, sources
            )
        
        return QueryResponse(
            query=request.query,
            answer=response.final_answer,
            sources=[],  # Multi-agent sources are complex, simplified here
            confidence_score=response.confidence,
            processing_time=response.total_execution_time,
            reasoning_steps=response.reasoning_chain if request.include_reasoning else [],
            session_id=request.session_id,
            system_type="multi_agent",
            hallucination_check=hallucination_result.__dict__ if hallucination_result else None,
            metadata={
                "agents_used": list(response.agent_responses.keys()),
                "plan_complexity": response.execution_plan.estimated_complexity if response.execution_plan else 0,
                "total_tasks": len(response.execution_plan.sub_tasks) if response.execution_plan else 0
            }
        )
        
    except Exception as e:
        logger.error(f"Multi-agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation Management Endpoints
@app.post("/conversation/start")
async def start_conversation(user_id: Optional[str] = None):
    """Start new conversation session."""
    try:
        session_id = await conversation_memory.create_session(user_id=user_id)
        return {
            "session_id": session_id,
            "message": "Conversation session started successfully"
        }
    except Exception as e:
        logger.error(f"Conversation start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}/history")
async def get_conversation_history(session_id: str, max_turns: int = 10):
    """Get conversation history for session."""
    try:
        context = await conversation_memory.get_conversation_context(session_id, max_turns)
        return context
    except Exception as e:
        logger.error(f"Conversation history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation/{session_id}/end")
async def end_conversation(session_id: str):
    """End conversation session."""
    try:
        await conversation_memory.end_session(session_id)
        return {"message": f"Conversation session {session_id} ended successfully"}
    except Exception as e:
        logger.error(f"Conversation end failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Evaluation Endpoints
@app.post("/evaluation/hallucination")
async def evaluate_hallucination(
    query: str,
    response: str,
    sources: List[Dict[str, Any]]
):
    """Evaluate response for potential hallucinations."""
    try:
        result = await hallucination_detector.detect_hallucination(query, response, sources)
        return result.__dict__
    except Exception as e:
        logger.error(f"Hallucination evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/agents")
async def get_agent_stats():
    """Get multi-agent system statistics."""
    try:
        return multi_agent_orchestrator.get_system_stats()
    except Exception as e:
        logger.error(f"Agent stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Advanced RAG System",
        "version": "1.0.0",
        "description": "Production-grade Retrieval-Augmented Generation system",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "query": "/query",
            "query_stream": "/query/stream",
            "upload": "/documents/upload",
            "documents": "/documents",
            "analytics": "/analytics/dashboard",
            "docs": "/docs"
        }
    }

# Run the application
def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api.main:app",
        host=config.system.api_host,
        port=config.system.api_port,
        reload=False,
        log_level=config.system.log_level.lower()
    )

if __name__ == "__main__":
    run_server()