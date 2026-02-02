# API Documentation

## Overview
The Advanced Multi-Agent RAG System provides a comprehensive RESTful API built with FastAPI.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API uses API key authentication through environment variables. Future versions will support token-based authentication.

## Endpoints

### Health Check
```http
GET /health
```
Returns system health status and component availability.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "components": {
    "database": "healthy",
    "vector_store": "healthy",
    "llm": "healthy"
  }
}
```

### Standard RAG Query
```http
POST /query
```
Process a query using the standard RAG pipeline.

**Request Body:**
```json
{
  "query": "What are the main applications of AI in healthcare?",
  "filters": {
    "document_id": "optional_document_id"
  },
  "max_chunks": 5
}
```

**Response:**
```json
{
  "query": "What are the main applications of AI in healthcare?",
  "answer": "AI has several main applications in healthcare...",
  "sources": [
    {
      "chunk_id": "uuid",
      "content": "relevant content",
      "document_id": "doc_uuid",
      "score": 0.95
    }
  ],
  "confidence_score": 0.87,
  "processing_time": 2.34,
  "reasoning_steps": ["step1", "step2"]
}
```

### Multi-Agent Query
```http
POST /multi-agent/query
```
Process a complex query using the multi-agent system.

**Request Body:**
```json
{
  "query": "Compare treatment options and calculate cost-effectiveness",
  "use_tools": true,
  "max_iterations": 3
}
```

### Streaming Query
```http
POST /query/stream
```
Get real-time streaming responses.

**Response:** Server-Sent Events (SSE) stream

### Document Management
```http
POST /documents/upload
```
Upload and process documents.

```http
GET /documents
```
List all processed documents.

```http
DELETE /documents/{document_id}
```
Delete a specific document.

### Conversation Management
```http
POST /conversation/start
```
Start a new conversation session.

```http
GET /conversation/{session_id}/history
```
Get conversation history.

```http
POST /conversation/{session_id}/end
```
End a conversation session.

### Analytics
```http
GET /analytics/dashboard
```
Get comprehensive system analytics.

```http
GET /analytics/performance
```
Get performance metrics.

## Error Handling
All endpoints return appropriate HTTP status codes and error messages:

```json
{
  "error": "Error description",
  "detail": "Detailed error information",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Rate Limiting
- Default: 100 requests per minute per IP
- Authenticated users: 1000 requests per minute

## Interactive Documentation
Visit `http://localhost:8000/docs` for interactive API documentation with Swagger UI.