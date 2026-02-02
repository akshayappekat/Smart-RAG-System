"""Configuration management for the Advanced RAG system."""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for AI models."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_provider: str = "openai"  # openai, anthropic, local
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    temperature: float = 0.1

@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_chunks_per_query: int = 10
    semantic_weight: float = 0.7
    lexical_weight: float = 0.3
    rerank_top_k: int = 20
    final_top_k: int = 5

@dataclass
class SystemConfig:
    """Main system configuration."""
    data_dir: Path = Path("data")
    vector_db_path: Path = Path("data/vector_db")
    cache_dir: Path = Path("data/cache")
    log_level: str = "INFO"
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Performance
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.vector_db_path.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.retrieval = RetrievalConfig()
        self.system = SystemConfig()
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Model config
        if os.getenv("EMBEDDING_MODEL"):
            self.model.embedding_model = os.getenv("EMBEDDING_MODEL")
        if os.getenv("LLM_MODEL"):
            self.model.llm_model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_PROVIDER"):
            self.model.llm_provider = os.getenv("LLM_PROVIDER")
            
        # System config
        if os.getenv("LOG_LEVEL"):
            self.system.log_level = os.getenv("LOG_LEVEL")
        if os.getenv("API_PORT"):
            self.system.api_port = int(os.getenv("API_PORT"))

# Global configuration instance
config = Config()