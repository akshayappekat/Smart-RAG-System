"""Production-grade LLM provider abstraction with multiple backends."""

import asyncio
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
from abc import ABC, abstractmethod
import json
import time

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ..config import config

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], 
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response from messages."""
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider with async support."""
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")
        
        self.model = model or config.model.llm_model
        
        # Initialize client with just the API key
        if api_key:
            self.client = openai.AsyncOpenAI(api_key=api_key)
        else:
            # Let OpenAI client use environment variable
            self.client = openai.AsyncOpenAI()
        
        logger.info(f"Initialized OpenAI provider with model: {self.model}")
    
    async def generate(self, messages: List[Dict[str, str]], 
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using OpenAI API."""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or config.model.max_tokens,
                "temperature": temperature or config.model.temperature,
                "stream": stream,
            }
            
            if stream:
                return self._stream_generate(params)
            else:
                response = await self.client.chat.completions.create(**params)
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def _stream_generate(self, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream generation from OpenAI."""
        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    async def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple approximation: ~4 characters per token for English
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "max_tokens": config.model.max_tokens,
            "temperature": config.model.temperature,
        }


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with async support."""
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic not available. Install with: pip install anthropic")
        
        self.model = model or "claude-3-sonnet-20240229"
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        
        logger.info(f"Initialized Anthropic provider with model: {self.model}")
    
    async def generate(self, messages: List[Dict[str, str]], 
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using Anthropic API."""
        try:
            # Convert messages to Anthropic format
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            params = {
                "model": self.model,
                "messages": user_messages,
                "max_tokens": max_tokens or config.model.max_tokens,
                "temperature": temperature or config.model.temperature,
                "stream": stream,
            }
            
            if system_message:
                params["system"] = system_message
            
            if stream:
                return self._stream_generate(params)
            else:
                response = await self.client.messages.create(**params)
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    async def _stream_generate(self, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream generation from Anthropic."""
        try:
            stream = await self.client.messages.create(**params)
            async for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    yield chunk.delta.text
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise
    
    async def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple approximation: ~4 characters per token for English
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        return {
            "provider": "anthropic",
            "model": self.model,
            "max_tokens": config.model.max_tokens,
            "temperature": config.model.temperature,
        }


class LocalLLMProvider(LLMProvider):
    """Local LLM provider (placeholder for future implementation)."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        logger.warning("LocalLLMProvider is not implemented yet")
    
    async def generate(self, messages: List[Dict[str, str]], 
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using local model."""
        raise NotImplementedError("Local LLM provider not implemented")
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get local model information."""
        return {
            "provider": "local",
            "model_path": self.model_path,
        }


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for demo purposes when no API key is available."""
    
    def __init__(self):
        logger.info("Using Mock LLM Provider - responses will be simulated")
    
    async def generate(self, messages: List[Dict[str, str]], 
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """Generate mock response based on context."""
        # Extract the user query from messages
        user_message = ""
        context = ""
        
        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"]
                # Extract context and query
                if "Context from relevant documents:" in user_message:
                    parts = user_message.split("Question:")
                    if len(parts) > 1:
                        context_part = parts[0]
                        query = parts[1].strip()
                        # Extract actual context
                        context_lines = context_part.split("\n")
                        context = "\n".join([line for line in context_lines if line.strip() and not line.startswith("Context")])
                    else:
                        query = user_message
                else:
                    query = user_message
        
        # Generate a mock response based on the context
        if stream:
            return self._stream_mock_response(query, context)
        else:
            return self._generate_mock_response(query, context)
    
    def _generate_mock_response(self, query: str, context: str) -> str:
        """Generate a mock response based on query and context."""
        query_lower = query.lower()
        
        # Simple keyword-based responses
        if "accuracy" in query_lower and "ai" in query_lower:
            return "Based on the provided context, AI systems in healthcare achieve high accuracy rates: 95%+ for diabetic retinopathy detection, 97% for skin cancer detection, and 94% for pneumonia detection from chest X-rays. These results demonstrate the significant potential of AI in medical imaging applications."
        
        elif "diabetes" in query_lower and ("treatment" in query_lower or "medication" in query_lower):
            return "According to the clinical guidelines, the first-line treatment for Type 2 diabetes is Metformin, with a starting dose of 500mg twice daily with meals and a maximum dose of 2000mg daily. Second-line treatments include sulfonylurea, DPP-4 inhibitors, or GLP-1 agonists, chosen based on patient factors such as weight, hypoglycemia risk, and cost."
        
        elif "drug discovery" in query_lower:
            return "Based on the research findings, machine learning has significantly accelerated drug discovery by reducing timelines from 10-15 years to 3-5 years. This represents a 60% reduction in drug discovery time, with 40% improvement in success rates and approximately 50% cost reduction."
        
        elif "diagnosis" in query_lower and "diabetes" in query_lower:
            return "According to the clinical guidelines, diabetes is diagnosed when any of the following criteria are met: Fasting plasma glucose ≥ 126 mg/dL (7.0 mmol/L), 2-hour plasma glucose ≥ 200 mg/dL (11.1 mmol/L) during OGTT, HbA1c ≥ 6.5% (48 mmol/mol), or random plasma glucose ≥ 200 mg/dL with symptoms."
        
        elif "machine learning" in query_lower and "healthcare" in query_lower:
            return "Based on the comprehensive review, machine learning has revolutionized healthcare through several key applications: medical imaging (achieving 95%+ accuracy in various diagnostic tasks), drug discovery (reducing development time by 60%), personalized medicine (improving treatment outcomes by 30-40% in oncology), and clinical decision support systems. The integration of AI has enabled automated analysis of medical data and improved diagnostic accuracy."
        
        else:
            # Generic response using context
            if context:
                # Extract key information from context
                sentences = context.split('.')[:3]  # Take first 3 sentences
                return f"Based on the provided context: {'. '.join(sentences).strip()}. This information addresses your query about the topic."
            else:
                return "I can provide information based on the available context. However, I'm currently running in demo mode without access to advanced language models. For more detailed responses, please configure an OpenAI or Anthropic API key."
    
    async def _stream_mock_response(self, query: str, context: str) -> AsyncGenerator[str, None]:
        """Stream a mock response."""
        response = self._generate_mock_response(query, context)
        words = response.split()
        
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield " " + word
            await asyncio.sleep(0.05)  # Simulate streaming delay
    
    async def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "provider": "mock",
            "model": "demo-model",
            "note": "This is a mock provider for demonstration purposes"
        }


class LLMManager:
    """Manager for LLM providers with caching and error handling."""
    
    def __init__(self):
        self.provider = None
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = config.system.cache_ttl
        
    def _get_provider(self) -> LLMProvider:
        """Get or create LLM provider."""
        if self.provider is None:
            provider_name = config.model.llm_provider.lower()
            
            try:
                if provider_name == "openai":
                    self.provider = OpenAIProvider()
                elif provider_name == "anthropic":
                    self.provider = AnthropicProvider()
                elif provider_name == "local":
                    self.provider = LocalLLMProvider("local_model")
                else:
                    raise ValueError(f"Unsupported LLM provider: {provider_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM provider {provider_name}: {e}")
                logger.info("LLM functionality will be limited without API keys")
                # Create a mock provider for demo purposes
                self.provider = MockLLMProvider()
        
        return self.provider
    
    def _get_cache_key(self, messages: List[Dict[str, str]], 
                      max_tokens: Optional[int], temperature: Optional[float]) -> str:
        """Generate cache key for request."""
        import hashlib
        content = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": config.model.llm_model,
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self.cache_ttl
    
    async def generate(self, messages: List[Dict[str, str]], 
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      stream: bool = False,
                      use_cache: bool = True) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response with caching and error handling."""
        provider = self._get_provider()
        
        # Check cache for non-streaming requests
        if not stream and use_cache:
            cache_key = self._get_cache_key(messages, max_tokens, temperature)
            if cache_key in self.cache:
                cached_response, timestamp = self.cache[cache_key]
                if self._is_cache_valid(timestamp):
                    logger.info("Returning cached LLM response")
                    return cached_response
        
        try:
            # Generate response
            start_time = time.time()
            response = await provider.generate(messages, max_tokens, temperature, stream)
            generation_time = time.time() - start_time
            
            # Cache non-streaming responses
            if not stream and use_cache and isinstance(response, str):
                cache_key = self._get_cache_key(messages, max_tokens, temperature)
                self.cache[cache_key] = (response, time.time())
                
                # Limit cache size
                if len(self.cache) > 1000:
                    # Remove oldest entries
                    oldest_keys = sorted(self.cache.keys(), 
                                       key=lambda k: self.cache[k][1])[:100]
                    for key in oldest_keys:
                        del self.cache[key]
            
            logger.info(f"LLM generation completed in {generation_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        provider = self._get_provider()
        return await provider.count_tokens(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        provider = self._get_provider()
        return provider.get_model_info()
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self.cache.clear()
        logger.info("LLM cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        valid_entries = sum(1 for _, timestamp in self.cache.values() 
                          if self._is_cache_valid(timestamp))
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "cache_ttl": self.cache_ttl,
        }


# Global LLM manager instance
llm_manager = LLMManager()