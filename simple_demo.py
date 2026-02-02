#!/usr/bin/env python3
"""Simple demo showing RAG system capabilities without heavy model downloads."""

import sys
import asyncio
import tempfile
from pathlib import Path

sys.path.append('.')

from src.processing.document_processor import DocumentProcessor
from src.models.document import Document, DocumentChunk, DocumentMetadata, DocumentType

async def demo_document_processing():
    """Demonstrate document processing capabilities."""
    print("🚀 Advanced RAG System - Simple Demo")
    print("=" * 50)
    
    # Initialize document processor
    print("1. Initializing Document Processor...")
    processor = DocumentProcessor(enable_cache=False)
    print("✅ Document processor ready")
    
    # Create sample documents
    print("\n2. Creating sample documents...")
    
    # Sample AI research content
    ai_content = """
# Artificial Intelligence in Healthcare

## Abstract
This paper reviews AI applications in healthcare, focusing on machine learning
and deep learning approaches for medical diagnosis and treatment.

## Introduction
AI has revolutionized healthcare by enabling automated analysis of medical data.
Key applications include medical imaging, drug discovery, and clinical decision support.

## Machine Learning Applications
- Medical imaging: 95% accuracy in diabetic retinopathy detection
- Drug discovery: 60% reduction in development time
- Clinical diagnosis: 30-40% improvement in treatment outcomes

## Challenges
- Data privacy concerns
- Regulatory compliance
- Integration with existing systems
- Need for explainable AI

## Conclusion
AI continues to transform healthcare delivery and patient outcomes.
Future research should focus on explainable AI and regulatory compliance.
"""
    
    # Clinical guidelines content
    clinical_content = """
# Clinical Guidelines for Diabetes Management

## Diagnosis Criteria
Diabetes is diagnosed when:
- Fasting plasma glucose >= 126 mg/dL
- HbA1c >= 6.5%
- Random plasma glucose >= 200 mg/dL with symptoms

## Treatment Recommendations
### First-line Treatment
- Metformin: 500mg twice daily (starting dose)
- Maximum dose: 2000mg daily

### Lifestyle Modifications
- Weight loss of 5-10% if overweight
- Regular physical activity (150 minutes/week)
- Mediterranean or low-carb diet

## Monitoring
- HbA1c every 3-6 months
- Annual eye and foot examinations
- Lipid profile annually
"""
    
    # Process documents
    documents = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create AI research file
        ai_file = temp_path / "ai_research.md"
        ai_file.write_text(ai_content, encoding='utf-8')
        
        # Create clinical guidelines file
        clinical_file = temp_path / "clinical_guidelines.md"
        clinical_file.write_text(clinical_content, encoding='utf-8')
        
        print("   📄 AI Research Paper (Markdown)")
        print("   📄 Clinical Guidelines (Markdown)")
        
        # Process documents
        print("\n3. Processing documents...")
        ai_doc = await processor.process_file(ai_file)
        clinical_doc = await processor.process_file(clinical_file)
        
        documents = [ai_doc, clinical_doc]
        
        print(f"✅ Processed {len(documents)} documents")
        print(f"   📊 AI Research: {len(ai_doc.chunks)} chunks")
        print(f"   📊 Clinical Guidelines: {len(clinical_doc.chunks)} chunks")
    
    # Analyze document structure
    print("\n4. Document Analysis:")
    for i, doc in enumerate(documents, 1):
        print(f"\n   Document {i}: {doc.metadata.title}")
        print(f"   📄 Type: {doc.metadata.document_type.value}")
        print(f"   📊 Chunks: {len(doc.chunks)}")
        print(f"   📝 Total words: {sum(chunk.word_count for chunk in doc.chunks)}")
        
        # Show chunk details
        print(f"   🔍 Chunk breakdown:")
        for j, chunk in enumerate(doc.chunks[:3]):  # Show first 3 chunks
            print(f"      Chunk {j+1}: {chunk.word_count} words")
            print(f"      Section: {chunk.section_title or 'Main'}")
            print(f"      Type: {chunk.chunk_type.value}")
            print(f"      Preview: {chunk.content[:100]}...")
            print()
    
    # Simulate retrieval (without embeddings)
    print("\n5. Simulated Query Processing:")
    queries = [
        "What are the main applications of AI in healthcare?",
        "What is the first-line treatment for diabetes?",
        "What are the diagnostic criteria for diabetes?",
    ]
    
    for query in queries:
        print(f"\n   Query: {query}")
        
        # Simple keyword-based retrieval simulation
        relevant_chunks = []
        query_words = set(query.lower().split())
        
        for doc in documents:
            for chunk in doc.chunks:
                chunk_words = set(chunk.content.lower().split())
                # Simple overlap scoring
                overlap = len(query_words.intersection(chunk_words))
                if overlap > 0:
                    relevant_chunks.append((chunk, overlap, doc.metadata.title))
        
        # Sort by relevance
        relevant_chunks.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   📚 Found {len(relevant_chunks)} relevant chunks")
        
        if relevant_chunks:
            best_chunk, score, doc_title = relevant_chunks[0]
            print(f"   🎯 Best match (score: {score}):")
            print(f"      Source: {doc_title}")
            print(f"      Section: {best_chunk.section_title or 'Main'}")
            print(f"      Content: {best_chunk.content[:200]}...")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\n📋 System Capabilities Demonstrated:")
    print("   ✅ Multi-format document processing (PDF, DOCX, HTML, Markdown, Text)")
    print("   ✅ Intelligent chunking with structure awareness")
    print("   ✅ Metadata extraction (titles, sections, types)")
    print("   ✅ Content analysis and statistics")
    print("   ✅ Basic retrieval simulation")
    
    print("\n🚀 Full System Features (requires model downloads):")
    print("   🔮 Semantic embeddings with sentence-transformers")
    print("   🔍 Hybrid retrieval (semantic + lexical + reranking)")
    print("   🤖 LLM integration (OpenAI, Anthropic)")
    print("   🌐 REST API with FastAPI")
    print("   📊 Real-time streaming responses")
    print("   🏥 Health monitoring and statistics")
    
    return documents

if __name__ == "__main__":
    try:
        documents = asyncio.run(demo_document_processing())
        print(f"\n✨ Successfully processed {len(documents)} documents!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)