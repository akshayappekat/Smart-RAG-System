"""Performance tests for document processor."""

import pytest
import asyncio
import time
from pathlib import Path
import tempfile

from src.processing.document_processor import DocumentProcessor


@pytest.mark.performance
class TestPerformance:
    """Performance benchmarks for document processing."""
    
    @pytest.fixture
    def processor(self):
        return DocumentProcessor(max_workers=4, enable_cache=False)
    
    @pytest.mark.asyncio
    async def test_large_file_processing_time(self, processor):
        """Test processing time for large files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a large text file (1MB)
            large_file = Path(temp_dir) / "large.txt"
            content = "This is a test sentence. " * 10000  # ~250KB
            large_file.write_text(content * 4)  # ~1MB
            
            start_time = time.time()
            document = await processor.process_file(large_file)
            processing_time = time.time() - start_time
            
            assert document.is_processed
            assert processing_time < 10.0  # Should process 1MB in under 10 seconds
            print(f"Large file processing time: {processing_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, processor):
        """Test batch processing performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple medium-sized files
            files = []
            for i in range(10):
                file_path = Path(temp_dir) / f"doc_{i}.txt"
                content = f"Document {i} content. " * 1000  # ~25KB each
                file_path.write_text(content)
                files.append(file_path)
            
            start_time = time.time()
            documents = await processor.process_files_batch(files)
            batch_time = time.time() - start_time
            
            # Process same files sequentially for comparison
            start_time = time.time()
            sequential_docs = []
            for file_path in files:
                doc = await processor.process_file(file_path)
                sequential_docs.append(doc)
            sequential_time = time.time() - start_time
            
            assert len(documents) == 10
            assert all(doc.is_processed for doc in documents)
            assert batch_time < sequential_time  # Batch should be faster
            
            print(f"Batch processing: {batch_time:.2f}s")
            print(f"Sequential processing: {sequential_time:.2f}s")
            print(f"Speedup: {sequential_time/batch_time:.2f}x")
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test caching performance improvement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create processor with caching enabled
            cached_processor = DocumentProcessor(enable_cache=True)
            cached_processor.cache_dir = Path(temp_dir) / "cache"
            cached_processor.cache_dir.mkdir()
            
            # Create test file
            test_file = Path(temp_dir) / "test.txt"
            content = "Test content. " * 1000
            test_file.write_text(content)
            
            # First processing (no cache)
            start_time = time.time()
            doc1 = await cached_processor.process_file(test_file)
            first_time = time.time() - start_time
            
            # Second processing (with cache)
            start_time = time.time()
            doc2 = await cached_processor.process_file(test_file)
            cached_time = time.time() - start_time
            
            assert doc1.is_processed
            assert doc2.is_processed
            assert cached_time < first_time  # Cache should be faster
            
            print(f"First processing: {first_time:.4f}s")
            print(f"Cached processing: {cached_time:.4f}s")
            print(f"Cache speedup: {first_time/cached_time:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])