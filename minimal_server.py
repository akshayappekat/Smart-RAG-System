#!/usr/bin/env python3
"""Minimal working server for Smart-RAG demo."""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add src to path
sys.path.append('.')

app = FastAPI(
    title="Smart-RAG System Demo",
    description="Minimal working demo of the Smart-RAG system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    max_chunks: int = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    system_status: str

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]

# Mock data
MOCK_DOCUMENTS = [
    {
        "id": "ai_research",
        "title": "AI in Healthcare Research",
        "content": """AI systems achieve 95% accuracy in diabetic retinopathy detection.
        Machine learning reduces drug discovery time by 60%.
        Deep learning improves medical imaging diagnosis by 30-40%.
        Applications include medical imaging, drug discovery, personalized treatment, and clinical decision support."""
    },
    {
        "id": "clinical_guidelines", 
        "title": "Diabetes Management Guidelines",
        "content": """Diabetes is diagnosed when fasting plasma glucose >= 126 mg/dL or HbA1c >= 6.5%.
        First-line treatment is Metformin 500mg twice daily, maximum 2000mg daily.
        Lifestyle interventions include 5-10% weight reduction and 150 minutes/week physical activity.
        Monitoring requires HbA1c every 3-6 months and annual eye/foot examinations."""
    }
]

def mock_search(query: str) -> List[Dict[str, Any]]:
    """Simple keyword-based search simulation."""
    query_words = set(query.lower().split())
    results = []
    
    for doc in MOCK_DOCUMENTS:
        content_words = set(doc["content"].lower().split())
        overlap = len(query_words.intersection(content_words))
        
        if overlap > 0:
            results.append({
                "document_id": doc["id"],
                "title": doc["title"],
                "content": doc["content"][:300] + "...",
                "score": overlap / len(query_words),
                "relevance": "high" if overlap >= 2 else "medium"
            })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

def generate_mock_answer(query: str, sources: List[Dict[str, Any]]) -> str:
    """Generate a mock answer based on query and sources."""
    if not sources:
        return "I don't have enough information to answer that question."
    
    query_lower = query.lower()
    
    # AI/Healthcare queries
    if any(word in query_lower for word in ["ai", "artificial intelligence", "machine learning"]):
        return """Based on the research, AI has significant applications in healthcare:

• **Medical Imaging**: AI systems achieve 95% accuracy in diabetic retinopathy detection
• **Drug Discovery**: Machine learning reduces development time by 60%
• **Diagnosis**: Deep learning improves medical imaging diagnosis by 30-40%
• **Applications**: Medical imaging analysis, drug discovery, personalized treatment plans, and clinical decision support systems

The technology shows great promise for improving patient outcomes and healthcare efficiency."""

    # Diabetes queries
    elif any(word in query_lower for word in ["diabetes", "diabetic", "treatment"]):
        return """According to clinical guidelines for diabetes management:

**Diagnosis Criteria:**
• Fasting plasma glucose ≥ 126 mg/dL
• HbA1c ≥ 6.5%
• Random plasma glucose ≥ 200 mg/dL with symptoms

**First-line Treatment:**
• Metformin: Starting dose 500mg twice daily
• Maximum dose: 2000mg daily

**Lifestyle Interventions:**
• Weight reduction of 5-10% if overweight
• Regular physical activity (150 minutes/week)
• Dietary modifications

**Monitoring:**
• HbA1c testing every 3-6 months
• Annual comprehensive eye and foot examinations"""

    # General response
    else:
        best_source = sources[0]
        return f"""Based on the available information from "{best_source['title']}":

{best_source['content']}

This information provides relevant context for your question about {query}."""

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Smart-RAG System Demo",
        "version": "1.0.0",
        "status": "running",
        "description": "Minimal working demo showcasing RAG capabilities",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        components={
            "document_processor": "operational",
            "search_engine": "operational", 
            "answer_generator": "operational",
            "api_server": "operational"
        }
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the knowledge base."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        import time
        start_time = time.time()
        
        # Simulate processing delay
        await asyncio.sleep(0.5)
        
        # Search for relevant documents
        sources = mock_search(request.query)[:request.max_chunks]
        
        # Generate answer
        answer = generate_mock_answer(request.query, sources)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            confidence_score=0.85 if sources else 0.3,
            processing_time=processing_time,
            system_status="demo_mode"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "documents": len(MOCK_DOCUMENTS),
        "total_chunks": sum(len(doc["content"].split()) for doc in MOCK_DOCUMENTS),
        "system_mode": "demo",
        "capabilities": [
            "Document processing simulation",
            "Keyword-based search",
            "Answer generation",
            "REST API",
            "Health monitoring"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Smart-RAG Demo Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("📊 System Stats: http://localhost:8000/stats")
    print("💡 Try querying: http://localhost:8000/query")
    print("\n🎯 Example queries:")
    print("   • 'What are AI applications in healthcare?'")
    print("   • 'What is the treatment for diabetes?'")
    print("   • 'How accurate are AI diagnostic systems?'")
    print("\n💡 Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "minimal_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )