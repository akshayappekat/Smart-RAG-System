"""Complete examples demonstrating the Advanced RAG System capabilities."""

import asyncio
import logging
from pathlib import Path
import tempfile
import time

from src.rag_orchestrator import rag_orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_1_basic_rag_workflow():
    """Example 1: Basic RAG workflow with document ingestion and querying."""
    print("=" * 60)
    print("Example 1: Basic RAG Workflow")
    print("=" * 60)
    
    # Initialize the system
    print("1. Initializing RAG system...")
    await rag_orchestrator.initialize()
    
    # Create sample documents
    print("2. Creating sample documents...")
    
    # Sample research paper content
    research_paper = """
    # Machine Learning in Healthcare: A Comprehensive Review
    
    ## Abstract
    
    This paper provides a comprehensive review of machine learning applications in healthcare.
    We examine various ML techniques including supervised learning, unsupervised learning,
    and deep learning approaches. Our analysis covers applications in medical imaging,
    drug discovery, personalized medicine, and clinical decision support systems.
    
    ## Introduction
    
    Machine learning has revolutionized healthcare by enabling automated analysis of
    medical data, improving diagnostic accuracy, and supporting clinical decision-making.
    The integration of ML in healthcare has shown promising results across multiple domains.
    
    ## Methodology
    
    We conducted a systematic review of 150 research papers published between 2020-2023.
    Our analysis focused on four key areas: medical imaging, drug discovery,
    personalized medicine, and clinical decision support.
    
    ## Results
    
    ### Medical Imaging
    
    Deep learning models, particularly convolutional neural networks (CNNs), have achieved
    remarkable success in medical image analysis. Studies show accuracy rates of 95%+ in
    detecting diabetic retinopathy, skin cancer, and pneumonia from medical images.
    
    ### Drug Discovery
    
    Machine learning has accelerated drug discovery by predicting molecular properties,
    identifying drug targets, and optimizing compound structures. AI-driven approaches
    have reduced drug discovery timelines from 10-15 years to 3-5 years.
    
    ### Personalized Medicine
    
    ML algorithms analyze genomic data, patient history, and lifestyle factors to
    provide personalized treatment recommendations. Precision medicine approaches
    have improved treatment outcomes by 30-40% in oncology applications.
    
    ## Conclusion
    
    Machine learning continues to transform healthcare delivery and patient outcomes.
    Future research should focus on explainable AI, data privacy, and regulatory compliance.
    """
    
    # Sample clinical guidelines
    clinical_guidelines = """
    # Clinical Guidelines for Diabetes Management
    
    ## Overview
    
    These guidelines provide evidence-based recommendations for the management of
    Type 2 diabetes mellitus in adult patients.
    
    ## Diagnosis Criteria
    
    Diabetes is diagnosed when:
    - Fasting plasma glucose ≥ 126 mg/dL (7.0 mmol/L)
    - 2-hour plasma glucose ≥ 200 mg/dL (11.1 mmol/L) during OGTT
    - HbA1c ≥ 6.5% (48 mmol/mol)
    - Random plasma glucose ≥ 200 mg/dL with symptoms
    
    ## Treatment Recommendations
    
    ### First-line Treatment
    - Metformin is the preferred initial medication
    - Starting dose: 500mg twice daily with meals
    - Maximum dose: 2000mg daily
    
    ### Second-line Treatment
    - Add sulfonylurea, DPP-4 inhibitor, or GLP-1 agonist
    - Consider patient factors: weight, hypoglycemia risk, cost
    
    ### Lifestyle Modifications
    - Weight loss of 5-10% if overweight
    - Regular physical activity (150 minutes/week)
    - Mediterranean or low-carb diet
    - Blood glucose monitoring
    
    ## Monitoring
    
    - HbA1c every 3-6 months
    - Annual eye and foot examinations
    - Lipid profile annually
    - Kidney function monitoring
    """
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Write sample documents
        research_file = temp_path / "ml_healthcare_review.md"
        research_file.write_text(research_paper)
        
        guidelines_file = temp_path / "diabetes_guidelines.md"
        guidelines_file.write_text(clinical_guidelines)
        
        # Ingest documents
        print("3. Ingesting documents...")
        doc1 = await rag_orchestrator.ingest_document(research_file, "ml_healthcare_review")
        doc2 = await rag_orchestrator.ingest_document(guidelines_file, "diabetes_guidelines")
        
        print(f"   - Research paper: {len(doc1.chunks)} chunks")
        print(f"   - Clinical guidelines: {len(doc2.chunks)} chunks")
        
        # Example queries
        queries = [
            "What are the main applications of machine learning in healthcare?",
            "What is the first-line treatment for Type 2 diabetes?",
            "How accurate are deep learning models in medical imaging?",
            "What are the diagnostic criteria for diabetes?",
            "How has ML impacted drug discovery timelines?"
        ]
        
        print("\n4. Running example queries...")
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            
            response = await rag_orchestrator.query(query)
            
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Sources: {len(response.sources)}")
            print(f"Processing time: {response.processing_time:.2f}s")
            
            if response.sources:
                top_source = response.sources[0]
                print(f"Top source: {top_source.document_id} (score: {top_source.score:.3f})")


async def example_2_multi_document_reasoning():
    """Example 2: Multi-document reasoning and comparison."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Document Reasoning")
    print("=" * 60)
    
    # Create comparative documents
    print("1. Creating comparative research documents...")
    
    study_a = """
    # Study A: Deep Learning vs Traditional ML in Medical Diagnosis
    
    ## Abstract
    We compared deep learning and traditional machine learning approaches for medical diagnosis
    across 5 different conditions using a dataset of 10,000 patients.
    
    ## Results
    - Deep Learning Accuracy: 94.2%
    - Traditional ML Accuracy: 87.6%
    - Deep Learning Training Time: 48 hours
    - Traditional ML Training Time: 2 hours
    - Deep Learning Interpretability: Low
    - Traditional ML Interpretability: High
    
    ## Conclusion
    Deep learning shows superior accuracy but requires more computational resources
    and provides less interpretable results.
    """
    
    study_b = """
    # Study B: Comparative Analysis of AI Diagnostic Tools
    
    ## Abstract
    This meta-analysis examined 25 studies comparing AI diagnostic tools with human physicians
    across various medical specialties.
    
    ## Key Findings
    - AI Diagnostic Accuracy: 91.8% (range: 85-97%)
    - Human Physician Accuracy: 89.2% (range: 82-95%)
    - AI Processing Time: 2.3 seconds average
    - Human Diagnosis Time: 15.7 minutes average
    - AI Cost per Diagnosis: $0.50
    - Human Cost per Diagnosis: $125
    
    ## Limitations
    AI systems showed reduced performance in rare conditions and required
    high-quality standardized input data.
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Write comparative studies
        study_a_file = temp_path / "study_a.md"
        study_a_file.write_text(study_a)
        
        study_b_file = temp_path / "study_b.md"
        study_b_file.write_text(study_b)
        
        # Ingest documents
        print("2. Ingesting comparative studies...")
        await rag_orchestrator.ingest_document(study_a_file, "study_a_dl_vs_ml")
        await rag_orchestrator.ingest_document(study_b_file, "study_b_ai_vs_human")
        
        # Multi-document reasoning queries
        comparison_queries = [
            "Compare the accuracy of deep learning vs traditional ML vs human physicians in medical diagnosis",
            "What are the trade-offs between AI and human diagnosis in terms of cost and time?",
            "What are the main limitations of AI diagnostic systems according to these studies?",
            "Synthesize the findings about AI interpretability across both studies"
        ]
        
        print("\n3. Running multi-document reasoning queries...")
        
        for i, query in enumerate(comparison_queries, 1):
            print(f"\nQuery {i}: {query}")
            
            response = await rag_orchestrator.query(query, max_chunks=8)
            
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence_score:.2f}")
            
            # Show source diversity
            source_docs = set(source.document_id for source in response.sources)
            print(f"Sources from {len(source_docs)} documents: {', '.join(source_docs)}")


async def example_3_advanced_filtering_and_search():
    """Example 3: Advanced filtering and search capabilities."""
    print("\n" + "=" * 60)
    print("Example 3: Advanced Filtering and Search")
    print("=" * 60)
    
    # Create documents with rich metadata
    print("1. Creating documents with rich metadata...")
    
    cardiology_paper = """
    # Advances in Cardiac Imaging: AI-Powered Diagnostics
    
    ## Abstract
    This review examines recent advances in AI-powered cardiac imaging diagnostics,
    focusing on echocardiography, cardiac MRI, and CT angiography applications.
    
    ## Introduction
    Cardiovascular disease remains the leading cause of death globally.
    AI-powered imaging has shown promise in early detection and risk stratification.
    
    ## Echocardiography Applications
    Deep learning models can automatically detect:
    - Left ventricular ejection fraction with 95% accuracy
    - Valvular abnormalities with 92% sensitivity
    - Wall motion abnormalities with 89% specificity
    
    ## Cardiac MRI Analysis
    AI algorithms excel at:
    - Automated chamber segmentation
    - Scar tissue quantification
    - Perfusion analysis
    
    ## Future Directions
    Integration with electronic health records and real-time analysis
    capabilities will further enhance diagnostic workflows.
    """
    
    oncology_paper = """
    # Machine Learning in Cancer Treatment: Personalized Therapy Selection
    
    ## Abstract
    This paper reviews ML applications in oncology, focusing on treatment
    selection, outcome prediction, and drug response modeling.
    
    ## Treatment Selection
    ML models analyze:
    - Genomic profiles
    - Tumor characteristics
    - Patient demographics
    - Treatment history
    
    ## Outcome Prediction
    Predictive models achieve:
    - 5-year survival prediction: 87% accuracy
    - Treatment response: 82% accuracy
    - Recurrence risk: 79% accuracy
    
    ## Drug Response Modeling
    Pharmacogenomic models predict:
    - Drug efficacy based on genetic markers
    - Adverse reaction likelihood
    - Optimal dosing strategies
    
    ## Clinical Implementation
    Several ML-based tools are now FDA-approved for clinical use
    in oncology decision support.
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Write specialized papers
        cardio_file = temp_path / "cardiology_ai.md"
        cardio_file.write_text(cardiology_paper)
        
        onco_file = temp_path / "oncology_ml.md"
        onco_file.write_text(oncology_paper)
        
        # Ingest with specific IDs
        print("2. Ingesting specialized medical papers...")
        await rag_orchestrator.ingest_document(cardio_file, "cardiology_ai_imaging")
        await rag_orchestrator.ingest_document(onco_file, "oncology_ml_treatment")
        
        # Filtered search examples
        print("\n3. Running filtered searches...")
        
        # Search with document filters
        print("\nFiltered Query 1: Cardiology-specific search")
        response = await rag_orchestrator.query(
            "What are the accuracy rates for AI in medical imaging?",
            filters={"document_id": "cardiology_ai_imaging"}
        )
        print(f"Answer (Cardiology only): {response.answer}")
        print(f"Sources: {len(response.sources)} from cardiology document")
        
        print("\nFiltered Query 2: Oncology-specific search")
        response = await rag_orchestrator.query(
            "What are the accuracy rates for AI in medical applications?",
            filters={"document_id": "oncology_ml_treatment"}
        )
        print(f"Answer (Oncology only): {response.answer}")
        print(f"Sources: {len(response.sources)} from oncology document")
        
        # Unfiltered comparison
        print("\nUnfiltered Query: Cross-domain search")
        response = await rag_orchestrator.query(
            "Compare AI accuracy rates across different medical specialties"
        )
        print(f"Answer (All documents): {response.answer}")
        source_docs = set(source.document_id for source in response.sources)
        print(f"Sources from {len(source_docs)} documents: {', '.join(source_docs)}")


async def example_4_system_monitoring_and_stats():
    """Example 4: System monitoring and statistics."""
    print("\n" + "=" * 60)
    print("Example 4: System Monitoring and Statistics")
    print("=" * 60)
    
    print("1. System Health Check...")
    health = await rag_orchestrator.health_check()
    
    print(f"Overall Status: {health['status']}")
    print("Component Status:")
    for component, status in health['components'].items():
        print(f"  {component}: {status['status']}")
        if 'error' in status:
            print(f"    Error: {status['error']}")
    
    print("\n2. System Statistics...")
    stats = await rag_orchestrator.get_system_stats()
    
    print(f"Documents in Knowledge Base: {stats['documents']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"System Initialized: {stats['is_initialized']}")
    
    if 'embedding_service' in stats:
        embed_stats = stats['embedding_service']
        print(f"\nEmbedding Service:")
        print(f"  Cache Enabled: {embed_stats.get('enabled', False)}")
        if embed_stats.get('enabled'):
            print(f"  Cached Embeddings: {embed_stats.get('cached_embeddings', 0)}")
            print(f"  Cache Size: {embed_stats.get('total_size_mb', 0):.2f} MB")
    
    if 'retrieval' in stats:
        retrieval_stats = stats['retrieval']
        print(f"\nRetrieval System:")
        print(f"  BM25 Available: {retrieval_stats.get('bm25_available', False)}")
        print(f"  Reranker Available: {retrieval_stats.get('reranker_available', False)}")
        print(f"  Semantic Weight: {retrieval_stats.get('semantic_weight', 0)}")
        print(f"  Lexical Weight: {retrieval_stats.get('lexical_weight', 0)}")
    
    if 'vector_store' in stats:
        vector_stats = stats['vector_store']
        print(f"\nVector Store:")
        print(f"  Type: {vector_stats.get('type', 'Unknown')}")
        print(f"  Total Chunks: {vector_stats.get('total_chunks', 0)}")
    
    print("\n3. Performance Benchmarking...")
    
    # Benchmark query performance
    test_queries = [
        "What is machine learning?",
        "How does AI help in healthcare?",
        "What are the benefits of deep learning?"
    ]
    
    total_time = 0
    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        response = await rag_orchestrator.query(query)
        query_time = time.time() - start_time
        total_time += query_time
        
        print(f"Query {i}: {query_time:.2f}s (confidence: {response.confidence_score:.2f})")
    
    avg_time = total_time / len(test_queries)
    print(f"Average Query Time: {avg_time:.2f}s")


async def example_5_streaming_responses():
    """Example 5: Streaming responses for real-time interaction."""
    print("\n" + "=" * 60)
    print("Example 5: Streaming Responses")
    print("=" * 60)
    
    print("1. Testing streaming query response...")
    
    query = "Explain the role of machine learning in modern healthcare and its main applications"
    
    print(f"Query: {query}")
    print("Streaming Response:")
    print("-" * 40)
    
    # Stream the response
    response_text = ""
    async for chunk in rag_orchestrator.query_stream(query):
        print(chunk, end='', flush=True)
        response_text += chunk
    
    print("\n" + "-" * 40)
    print(f"Total response length: {len(response_text)} characters")


async def main():
    """Run all examples."""
    print("Advanced RAG System - Complete Examples")
    print("=" * 60)
    
    try:
        # Run examples in sequence
        await example_1_basic_rag_workflow()
        await example_2_multi_document_reasoning()
        await example_3_advanced_filtering_and_search()
        await example_4_system_monitoring_and_stats()
        await example_5_streaming_responses()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())