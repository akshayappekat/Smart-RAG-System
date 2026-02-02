#!/usr/bin/env python3
"""Offline demo that works without internet connection or API keys."""

import sys
import asyncio
import os
from pathlib import Path

sys.path.append('.')

# Set offline mode
os.environ['OFFLINE_MODE'] = '1'

from src.rag_orchestrator import rag_orchestrator

async def offline_demo():
    """Run RAG system demo in offline mode."""
    print("🚀 Advanced RAG System - Offline Demo")
    print("=" * 50)
    print("📡 Running in OFFLINE mode (no internet required)")
    print("🤖 Using mock embeddings and LLM responses")
    print()
    
    try:
        # Initialize system in offline mode
        print("1. Initializing RAG system...")
        
        # Patch embedding service for offline mode
        from src.embeddings.embedding_service import embedding_service
        embedding_service.offline_mode = True
        
        await rag_orchestrator.initialize()
        print("✅ System initialized successfully!")
        
        # Check if sample documents exist
        sample_dir = Path("sample_documents")
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.md"))
            if sample_files:
                print(f"\n2. Found {len(sample_files)} sample documents, ingesting...")
                
                # Ingest sample documents
                documents = []
                for file_path in sample_files:
                    doc = await rag_orchestrator.ingest_document(file_path)
                    documents.append(doc)
                    print(f"   ✅ Processed: {file_path.name}")
                
                print(f"✅ Successfully ingested {len(documents)} documents")
            else:
                print("\n2. No sample documents found, creating test documents...")
                await create_test_documents()
        else:
            print("\n2. Creating sample documents...")
            await create_test_documents()
        
        # Get system stats
        print("\n3. System Statistics:")
        stats = await rag_orchestrator.get_system_stats()
        print(f"   📊 Documents: {stats['documents']}")
        print(f"   📄 Total Chunks: {stats['total_chunks']}")
        print(f"   🔧 System Status: {'✅ Ready' if stats['is_initialized'] else '❌ Not Ready'}")
        
        # Test queries
        print("\n4. Testing Query Processing:")
        test_queries = [
            "What are the main applications of AI in healthcare?",
            "What is the first-line treatment for diabetes?",
            "How accurate are AI systems in medical diagnosis?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            
            try:
                response = await rag_orchestrator.query(query)
                print(f"   ✅ Answer: {response.answer[:150]}...")
                print(f"   📊 Confidence: {response.confidence_score:.2f}")
                print(f"   ⏱️  Processing Time: {response.processing_time:.2f}s")
                print(f"   📚 Sources: {len(response.sources)}")
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        # Health check
        print("\n5. System Health Check:")
        health = await rag_orchestrator.health_check()
        print(f"   🏥 Overall Status: {health['status']}")
        
        for component, status in health['components'].items():
            status_icon = "✅" if status['status'] == 'healthy' else "⚠️"
            print(f"   {status_icon} {component}: {status['status']}")
        
        print("\n" + "=" * 50)
        print("🎉 Offline demo completed successfully!")
        print("\n📋 What was demonstrated:")
        print("   ✅ Document processing and chunking")
        print("   ✅ Mock embedding generation")
        print("   ✅ BM25 lexical search")
        print("   ✅ Mock LLM response generation")
        print("   ✅ System health monitoring")
        print("   ✅ Performance metrics")
        
        print("\n🌐 For full functionality:")
        print("   🔑 Set OPENAI_API_KEY for real LLM responses")
        print("   📡 Internet connection for embedding models")
        print("   🚀 Run: python -m src.main api (for REST API)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

async def create_test_documents():
    """Create test documents for the demo."""
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    # AI research document
    ai_content = """# AI in Healthcare Research

## Abstract
This research examines artificial intelligence applications in healthcare, focusing on machine learning and deep learning approaches for medical diagnosis and treatment optimization.

## Key Findings
- AI systems achieve 95% accuracy in diabetic retinopathy detection
- Machine learning reduces drug discovery time by 60%
- Deep learning improves medical imaging diagnosis by 30-40%

## Applications
1. Medical Imaging Analysis
2. Drug Discovery and Development
3. Personalized Treatment Plans
4. Clinical Decision Support Systems

## Challenges
- Data privacy and security
- Regulatory compliance
- Integration with existing systems
- Need for explainable AI models
"""
    
    # Clinical guidelines
    clinical_content = """# Diabetes Management Guidelines

## Diagnostic Criteria
Diabetes mellitus is diagnosed when:
- Fasting plasma glucose >= 126 mg/dL
- HbA1c >= 6.5%
- Random plasma glucose >= 200 mg/dL with symptoms

## Treatment Protocol
### First-line Treatment
- Metformin: Starting dose 500mg twice daily
- Maximum dose: 2000mg daily
- Monitor for gastrointestinal side effects

### Lifestyle Interventions
- Weight reduction of 5-10% if overweight
- Regular physical activity (150 minutes/week)
- Dietary modifications (Mediterranean or low-carb diet)

## Monitoring Requirements
- HbA1c testing every 3-6 months
- Annual comprehensive eye examination
- Annual foot examination
- Lipid profile assessment annually
"""
    
    # Write documents
    (sample_dir / "ai_research.md").write_text(ai_content, encoding='utf-8')
    (sample_dir / "clinical_guidelines.md").write_text(clinical_content, encoding='utf-8')
    
    print("   ✅ Created sample documents")

if __name__ == "__main__":
    try:
        asyncio.run(offline_demo())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)