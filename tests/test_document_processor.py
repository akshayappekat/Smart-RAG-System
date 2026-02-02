"""Tests for document processor."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import os

from src.processing.document_processor import DocumentProcessor
from src.models.document import Document, DocumentType, ChunkType


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a document processor instance."""
        return DocumentProcessor(max_workers=2, enable_cache=False)
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_get_document_type(self, processor):
        """Test document type detection."""
        assert processor._get_document_type(Path("test.pdf")) == DocumentType.PDF
        assert processor._get_document_type(Path("test.docx")) == DocumentType.DOCX
        assert processor._get_document_type(Path("test.txt")) == DocumentType.TEXT
        assert processor._get_document_type(Path("test.html")) == DocumentType.HTML
        assert processor._get_document_type(Path("test.md")) == DocumentType.MARKDOWN
        assert processor._get_document_type(Path("test.unknown")) == DocumentType.TEXT
    
    def test_clean_text(self, processor):
        """Test text cleaning functionality."""
        dirty_text = "  This   is    a   test  \n\n\n  with   extra   spaces  \n\n"
        clean_text = processor._clean_text(dirty_text)
        
        assert "   " not in clean_text
        assert clean_text.strip() == clean_text
        assert "\n\n\n" not in clean_text
    
    def test_extract_title_from_content(self, processor):
        """Test title extraction."""
        content = "--- Page 1 ---\nThis is a Great Title\n\nThis is the content..."
        title = processor._extract_title_from_content(content)
        assert title == "This is a Great Title"
    
    def test_extract_abstract_from_content(self, processor):
        """Test abstract extraction."""
        content = """
        Title
        
        Abstract:
        This is the abstract of the document.
        It contains important information.
        
        Introduction
        This is the introduction...
        """
        abstract = processor._extract_abstract_from_content(content)
        assert "This is the abstract" in abstract
    
    def test_is_section_header(self, processor):
        """Test section header detection."""
        assert processor._is_section_header("1. Introduction")
        assert processor._is_section_header("2.1 Methods")
        assert processor._is_section_header("ABSTRACT")
        assert processor._is_section_header("CONCLUSION AND FUTURE WORK")
        assert not processor._is_section_header("This is just a regular sentence.")
        assert not processor._is_section_header("")
    
    def test_contains_citations(self, processor):
        """Test citation detection."""
        text_with_citations = "This is supported by research [1] and Smith et al. (2023)."
        text_without_citations = "This is just regular text without any references."
        
        assert processor._contains_citations(text_with_citations)
        assert not processor._contains_citations(text_without_citations)
    
    def test_detect_section_type(self, processor):
        """Test section type detection."""
        assert processor._detect_section_type("Abstract", "content") == ChunkType.SECTION
        assert processor._detect_section_type("Table 1", "content") == ChunkType.TABLE
        assert processor._detect_section_type("Figure Caption", "content") == ChunkType.FIGURE_CAPTION
        assert processor._detect_section_type("Introduction", "content") == ChunkType.PARAGRAPH
    
    @pytest.mark.asyncio
    async def test_process_text_file(self, processor, temp_dir):
        """Test processing a text file."""
        # Create a test text file
        test_file = temp_dir / "test.txt"
        test_content = "This is a test document.\n\nIt has multiple paragraphs.\n\nAnd some more content."
        test_file.write_text(test_content)
        
        # Process the file
        document = await processor.process_file(test_file)
        
        assert document.is_processed
        assert document.metadata.document_type == DocumentType.TEXT
        assert document.metadata.title == "test"
        assert len(document.chunks) > 0
        assert document.content == processor._clean_text(test_content)
    
    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, processor):
        """Test processing a non-existent file."""
        document = await processor.process_file(Path("nonexistent.txt"))
        
        assert not document.is_processed
        assert len(document.processing_errors) > 0
        assert "File not found" in document.processing_errors[0]
    
    @pytest.mark.asyncio
    async def test_process_empty_file(self, processor, temp_dir):
        """Test processing an empty file."""
        test_file = temp_dir / "empty.txt"
        test_file.write_text("")
        
        document = await processor.process_file(test_file)
        
        assert not document.is_processed
        assert len(document.processing_errors) > 0
        assert "empty" in document.processing_errors[0].lower()
    
    @pytest.mark.asyncio
    async def test_process_files_batch(self, processor, temp_dir):
        """Test batch processing of files."""
        # Create multiple test files
        files = []
        for i in range(3):
            test_file = temp_dir / f"test_{i}.txt"
            test_file.write_text(f"This is test document {i}.\n\nWith some content.")
            files.append(test_file)
        
        # Process batch
        documents = await processor.process_files_batch(files)
        
        assert len(documents) == 3
        for doc in documents:
            assert doc.is_processed
            assert len(doc.chunks) > 0
    
    @pytest.mark.asyncio
    async def test_process_html_file(self, processor, temp_dir):
        """Test processing an HTML file."""
        test_file = temp_dir / "test.html"
        html_content = """
        <html>
        <head>
            <title>Test Document</title>
            <meta name="author" content="Test Author">
            <meta name="description" content="Test description">
        </head>
        <body>
            <h1>Main Title</h1>
            <p>This is a paragraph.</p>
            <h2>Section Title</h2>
            <p>Another paragraph with content.</p>
        </body>
        </html>
        """
        test_file.write_text(html_content)
        
        document = await processor.process_file(test_file)
        
        assert document.is_processed
        assert document.metadata.document_type == DocumentType.HTML
        assert document.metadata.title == "Test Document"
        assert "Test Author" in document.metadata.authors
        assert document.metadata.abstract == "Test description"
        assert "Main Title" in document.content
    
    @pytest.mark.asyncio
    async def test_process_markdown_file(self, processor, temp_dir):
        """Test processing a Markdown file."""
        test_file = temp_dir / "test.md"
        md_content = """---
title: Test Markdown Document
author: Test Author
description: This is a test markdown document
keywords: [test, markdown, document]
---

# Main Title

This is the introduction paragraph.

## Section 1

Content of section 1.

## Section 2

Content of section 2.
"""
        test_file.write_text(md_content)
        
        document = await processor.process_file(test_file)
        
        assert document.is_processed
        assert document.metadata.document_type == DocumentType.MARKDOWN
        assert document.metadata.title == "Test Markdown Document"
        assert "Test Author" in document.metadata.authors
        assert document.metadata.abstract == "This is a test markdown document"
        assert "test" in document.metadata.keywords
        assert "# Main Title" in document.content
    
    def test_split_into_sections(self, processor):
        """Test section splitting."""
        content = """
        Introduction
        This is the introduction.
        
        1. Methods
        This is the methods section.
        
        2. Results
        This is the results section.
        
        CONCLUSION
        This is the conclusion.
        """
        
        sections = processor._split_into_sections(content)
        
        assert len(sections) >= 3
        section_titles = [title for title, _ in sections]
        assert any("Methods" in title for title in section_titles)
        assert any("Results" in title for title in section_titles)
    
    @pytest.mark.asyncio
    async def test_create_intelligent_chunks(self, processor):
        """Test intelligent chunking."""
        content = "This is a test document. " * 100  # Long content
        metadata = Mock()
        
        chunks = await processor._create_intelligent_chunks(content, metadata)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        for chunk in chunks:
            assert len(chunk.content) <= processor.chunk_size + processor.chunk_overlap
            assert chunk.word_count > 0
            assert chunk.sentence_count > 0
    
    def test_post_process_chunks(self, processor):
        """Test chunk post-processing."""
        from src.models.document import DocumentChunk
        
        # Create test chunks
        chunks = [
            DocumentChunk(content="Short", section_title="Test"),  # Too short
            DocumentChunk(content="This is a longer chunk with sufficient content.", section_title="Test"),
            DocumentChunk(content="Another short", section_title="Test"),  # Should merge with previous
            DocumentChunk(content="This is a completely different section.", section_title="Different"),
        ]
        
        processed = processor._post_process_chunks(chunks)
        
        # Should have fewer chunks due to merging
        assert len(processed) < len(chunks)
    
    def test_cache_functionality(self, temp_dir):
        """Test caching functionality."""
        processor = DocumentProcessor(enable_cache=True)
        processor.cache_dir = temp_dir / "cache"
        processor.cache_dir.mkdir()
        
        # Test cache key generation
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        cache_key = processor._get_cache_key(test_file)
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length
        
        # Test caching and retrieval
        from src.models.document import Document, DocumentMetadata
        test_doc = Document(content="test", metadata=DocumentMetadata(title="Test"))
        
        processor._cache_document(cache_key, test_doc)
        retrieved_doc = processor._get_cached_document(cache_key)
        
        assert retrieved_doc is not None
        assert retrieved_doc.content == "test"
        assert retrieved_doc.metadata.title == "Test"


@pytest.mark.asyncio
async def test_integration_workflow():
    """Integration test for the complete workflow."""
    processor = DocumentProcessor(enable_cache=False)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files of different types
        files = []
        
        # Text file
        txt_file = temp_path / "document.txt"
        txt_file.write_text("This is a comprehensive test document.\n\nIt has multiple sections and paragraphs.")
        files.append(txt_file)
        
        # HTML file
        html_file = temp_path / "document.html"
        html_file.write_text("""
        <html><head><title>HTML Test</title></head>
        <body><h1>Title</h1><p>Content paragraph.</p></body></html>
        """)
        files.append(html_file)
        
        # Markdown file
        md_file = temp_path / "document.md"
        md_file.write_text("# Markdown Test\n\nThis is markdown content.\n\n## Section\n\nMore content.")
        files.append(md_file)
        
        # Process all files
        documents = await processor.process_files_batch(files)
        
        assert len(documents) == 3
        for doc in documents:
            assert doc.is_processed
            assert len(doc.chunks) > 0
            assert doc.metadata.document_type in [DocumentType.TEXT, DocumentType.HTML, DocumentType.MARKDOWN]


if __name__ == "__main__":
    pytest.main([__file__])