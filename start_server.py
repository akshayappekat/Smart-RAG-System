#!/usr/bin/env python3
"""
Startup script for the Advanced RAG System API server.
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Start the RAG system API server."""
    
    # Check if .env file exists
    if not Path(".env").exists():
        print("⚠️  Warning: .env file not found!")
        print("   Copy .env.example to .env and add your API keys")
        print("   Example: cp .env.example .env")
        return
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("❌ OpenAI API key not configured!")
        print("   Please set OPENAI_API_KEY in your .env file")
        return
    
    print("🚀 Starting Advanced RAG System API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("📊 System Stats: http://localhost:8000/stats")
    print("\n💡 Press Ctrl+C to stop the server")
    
    # Start the server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()