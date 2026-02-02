"""Main entry point for the Advanced RAG System."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import config
from .rag_orchestrator import rag_orchestrator

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.system.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_system():
    """Initialize the RAG system."""
    logger.info("Initializing Advanced RAG System...")
    try:
        await rag_orchestrator.initialize()
        logger.info("System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return False


async def ingest_sample_documents():
    """Ingest sample documents if available."""
    sample_dir = Path("sample_documents")
    if not sample_dir.exists():
        logger.info("No sample_documents directory found, skipping sample ingestion")
        return
    
    # Find supported document files
    supported_extensions = {'.pdf', '.docx', '.txt', '.html', '.md'}
    sample_files = []
    
    for ext in supported_extensions:
        sample_files.extend(sample_dir.glob(f"*{ext}"))
    
    if not sample_files:
        logger.info("No supported documents found in sample_documents/")
        return
    
    logger.info(f"Found {len(sample_files)} sample documents, ingesting...")
    
    try:
        documents = await rag_orchestrator.ingest_documents_batch(sample_files)
        successful = sum(1 for doc in documents if doc.is_processed)
        logger.info(f"Successfully ingested {successful}/{len(documents)} sample documents")
    except Exception as e:
        logger.error(f"Failed to ingest sample documents: {e}")


async def interactive_demo():
    """Run an interactive demo of the RAG system."""
    print("\n" + "="*60)
    print("Advanced RAG System - Interactive Demo")
    print("="*60)
    
    # Show system stats
    stats = await rag_orchestrator.get_system_stats()
    print(f"\nSystem Status:")
    print(f"  Documents: {stats['documents']}")
    print(f"  Total Chunks: {stats['total_chunks']}")
    print(f"  Initialized: {stats['is_initialized']}")
    
    if stats['documents'] == 0:
        print("\nNo documents in knowledge base. Please add documents first.")
        print("You can:")
        print("1. Place documents in 'sample_documents/' directory and restart")
        print("2. Use the API endpoints to upload documents")
        print("3. Use the web interface (if available)")
        return
    
    print(f"\nYou can now ask questions about the {stats['documents']} documents in the knowledge base.")
    print("Type 'quit' to exit, 'stats' for system statistics, or 'help' for commands.\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if query.lower() == 'stats':
                stats = await rag_orchestrator.get_system_stats()
                print(f"\nSystem Statistics:")
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
                print()
                continue
            
            if query.lower() == 'help':
                print("\nAvailable commands:")
                print("  quit/exit/q - Exit the demo")
                print("  stats - Show system statistics")
                print("  help - Show this help message")
                print("  Any other text - Ask a question about the documents\n")
                continue
            
            print("\nProcessing query...")
            
            # Process the query
            response = await rag_orchestrator.query(query)
            
            print(f"\nAnswer:")
            print(f"{response.answer}\n")
            
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Processing Time: {response.processing_time:.2f}s")
            print(f"Sources Used: {len(response.sources)}")
            
            if response.sources:
                print(f"\nTop Sources:")
                for i, source in enumerate(response.sources[:3]):
                    print(f"  {i+1}. Score: {source.score:.3f} | "
                          f"Doc: {source.document_id} | "
                          f"Section: {source.chunk.section_title or 'N/A'}")
                    print(f"     Preview: {source.chunk.content[:100]}...")
            
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")
            logger.error(f"Query error: {e}")


def run_server():
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.system.api_host,
        port=config.system.api_port,
        reload=False,
        log_level=config.system.log_level.lower()
    )

async def run_api_server():
    """Run the FastAPI server."""
    try:
        print("\nStarting API server...")
        print(f"Server will be available at: http://{config.system.api_host}:{config.system.api_port}")
        print(f"API Documentation: http://{config.system.api_host}:{config.system.api_port}/docs")
        print("\nPress Ctrl+C to stop the server")
        
        # Run uvicorn server in a separate thread to avoid event loop conflicts
        import threading
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Keep the main thread alive
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")
            
    except ImportError as e:
        logger.error(f"Failed to import API server: {e}")
        print("API server dependencies not available. Install with: pip install fastapi uvicorn")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")


async def main():
    """Main application entry point."""
    print("Advanced RAG System for Education & Enterprise")
    print("=" * 50)
    
    # Initialize system
    if not await initialize_system():
        print("Failed to initialize system. Exiting.")
        sys.exit(1)
    
    # Ingest sample documents
    await ingest_sample_documents()
    
    # Determine run mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "interactive"
    
    if mode == "api":
        await run_api_server()
    elif mode == "interactive":
        await interactive_demo()
    elif mode == "server":
        await run_api_server()
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes:")
        print("  interactive - Run interactive demo (default)")
        print("  api/server - Run API server")
        print("\nUsage: python -m src.main [mode]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)