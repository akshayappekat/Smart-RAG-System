"""Examples of using the DocumentProcessor."""

import asyncio
from pathlib import Path
from src.processing.document_processor import DocumentProcessor


async def basic_usage_example():
    """Basic usage of document processor."""
    print("=== Basic Document Processing Example ===")
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Process a single file
    file_path = Path("sample_documents/research_paper.pdf")
    if file_path.exists():
        document = await processor.process_file(file_path)
        
        print(f"Document processed: {document.metadata.title}")
        print(f"Number of chunks: {len(document.chunks)}")
        print(f"Document type: {document.metadata.document_type}")
        print(f"Processing errors: {len(document.processing_errors)}")
        
        # Show first chunk
        if document.chunks:
            first_chunk = document.chunks[0]
            print(f"\nFirst chunk preview:")
            print(f"Section: {first_chunk.section_title}")
            print(f"Type: {first_chunk.chunk_type}")
            print(f"Content: {first_chunk.content[:200]}...")
    else:
        print("Sample file not found. Please add a PDF file to sample_documents/")


async def batch_processing_example():
    """Example of batch processing multiple files."""
    print("\n=== Batch Processing Example ===")
    
    processor = DocumentProcessor(max_workers=4)
    
    # Find all documents in a directory
    docs_dir = Path("sample_documents")
    if docs_dir.exists():
        file_paths = list(docs_dir.glob("*.*"))
        supported_extensions = {'.pdf', '.docx', '.txt', '.html', '.md'}
        file_paths = [f for f in file_paths if f.suffix.lower() in supported_extensions]
        
        if file_paths:
            print(f"Processing {len(file_paths)} files...")
            
            # Progress callback
            async def progress_callback(current, total, filename):
                print(f"Progress: {current}/{total} - {filename}")
            
            # Process all files
            documents = await processor.process_files_batch(file_paths, progress_callback)
            
            print(f"\nProcessing complete!")
            print(f"Successfully processed: {sum(1 for d in documents if d.is_processed)}")
            print(f"Failed: {sum(1 for d in documents if not d.is_processed)}")
            
            # Summary statistics
            total_chunks = sum(len(d.chunks) for d in documents)
            total_words = sum(sum(c.word_count for c in d.chunks) for d in documents)
            
            print(f"Total chunks created: {total_chunks}")
            print(f"Total words processed: {total_words}")
        else:
            print("No supported files found in sample_documents/")
    else:
        print("sample_documents/ directory not found")


async def advanced_features_example():
    """Example showcasing advanced features."""
    print("\n=== Advanced Features Example ===")
    
    # Initialize with custom settings
    processor = DocumentProcessor(
        max_workers=2,
        enable_cache=True
    )
    
    # Create a sample markdown file with front matter
    sample_md = Path("temp_sample.md")
    sample_content = """---
title: "Advanced Document Processing"
author: "AI Assistant"
keywords: ["NLP", "document processing", "RAG"]
description: "Demonstration of advanced document processing features"
---

# Advanced Document Processing

## Introduction

This document demonstrates the advanced features of our document processor.

## Key Features

### Intelligent Chunking
The processor uses intelligent chunking based on document structure.

### Metadata Extraction
Metadata is automatically extracted from various sources:
- Front matter in Markdown
- HTML meta tags
- PDF metadata
- Document structure analysis

### Citation Detection
The system can detect citations like [1] and (Smith et al., 2023).

## Performance Optimization

The processor includes several performance optimizations:
1. Async processing
2. Batch processing
3. Intelligent caching
4. Concurrent execution

## Conclusion

This demonstrates the comprehensive capabilities of the document processor.
"""
    
    sample_md.write_text(sample_content)
    
    try:
        # Process the sample file
        document = await processor.process_file(sample_md)
        
        print(f"Title: {document.metadata.title}")
        print(f"Author: {document.metadata.authors}")
        print(f"Keywords: {document.metadata.keywords}")
        print(f"Abstract: {document.metadata.abstract}")
        
        print(f"\nChunk Analysis:")
        for i, chunk in enumerate(document.chunks):
            print(f"Chunk {i+1}:")
            print(f"  Section: {chunk.section_title}")
            print(f"  Type: {chunk.chunk_type}")
            print(f"  Words: {chunk.word_count}")
            print(f"  Has citations: {chunk.contains_citations}")
            print(f"  Is title: {chunk.is_title}")
            print(f"  Content preview: {chunk.content[:100]}...")
            print()
        
        # Test caching
        print("Testing cache performance...")
        import time
        
        start_time = time.time()
        cached_doc = await processor.process_file(sample_md)
        cache_time = time.time() - start_time
        
        print(f"Cache retrieval time: {cache_time:.4f}s")
        print(f"Documents are identical: {document.id == cached_doc.id}")
        
    finally:
        # Clean up
        if sample_md.exists():
            sample_md.unlink()


async def error_handling_example():
    """Example of error handling and validation."""
    print("\n=== Error Handling Example ===")
    
    processor = DocumentProcessor()
    
    # Test various error conditions
    test_cases = [
        ("nonexistent_file.pdf", "File not found"),
        ("empty_file.txt", "Empty file"),
        ("unsupported_file.xyz", "Unsupported format"),
    ]
    
    # Create empty file for testing
    empty_file = Path("empty_file.txt")
    empty_file.write_text("")
    
    # Create unsupported file
    unsupported_file = Path("unsupported_file.xyz")
    unsupported_file.write_text("Some content")
    
    try:
        for file_path, expected_error in test_cases:
            print(f"\nTesting: {file_path}")
            document = await processor.process_file(Path(file_path))
            
            if document.processing_errors:
                print(f"  Error caught: {document.processing_errors[0]}")
                print(f"  Expected: {expected_error}")
                print(f"  Status: {'✓' if expected_error.lower() in document.processing_errors[0].lower() else '✗'}")
            else:
                print(f"  Unexpected success!")
    
    finally:
        # Clean up test files
        for file_path in [empty_file, unsupported_file]:
            if file_path.exists():
                file_path.unlink()


async def main():
    """Run all examples."""
    print("Document Processor Examples")
    print("=" * 50)
    
    await basic_usage_example()
    await batch_processing_example()
    await advanced_features_example()
    await error_handling_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())