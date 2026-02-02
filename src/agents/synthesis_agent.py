"""Synthesis agent for combining information and generating final responses."""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResponse
from ..llm.providers import llm_manager

import logging
logger = logging.getLogger(__name__)


@dataclass
class SynthesisInput:
    """Input data for synthesis agent."""
    original_query: str
    knowledge_base_results: Optional[Dict[str, Any]] = None
    tool_results: Optional[Dict[str, Any]] = None
    web_search_results: Optional[Dict[str, Any]] = None
    additional_context: Optional[Dict[str, Any]] = None


class SynthesisAgent(BaseAgent):
    """Agent responsible for synthesizing information and generating final responses."""
    
    def __init__(self):
        super().__init__(
            agent_id="synthesis_agent",
            name="Synthesis Agent",
            description="Combines information from multiple sources and generates coherent responses"
        )
        
        self.synthesis_prompt = """
You are a Synthesis Agent in an advanced multi-agent RAG system. Your role is to combine information from multiple sources and generate a comprehensive, accurate response.

Available Information Sources:
{sources_summary}

Original Query: {query}

Knowledge Base Results:
{kb_results}

Tool Results:
{tool_results}

Web Search Results:
{web_results}

Instructions:
1. Analyze all available information sources
2. Identify the most relevant and reliable information
3. Synthesize a comprehensive response that addresses the original query
4. Cite sources appropriately
5. Indicate confidence level and any limitations
6. Ensure factual accuracy and avoid hallucinations

Generate a response that is:
- Comprehensive and well-structured
- Factually accurate based on provided sources
- Properly cited with source attribution
- Clear about confidence levels and limitations
- Helpful and directly addresses the user's query

Response:
"""
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Synthesize information and generate final response."""
        start_time = time.time()
        
        try:
            # Parse synthesis input
            synthesis_input = self._parse_synthesis_input(task, context or {})
            
            # Generate synthesized response
            response = await self._synthesize_response(synthesis_input)
            
            execution_time = time.time() - start_time
            self._update_stats(execution_time)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                result=response,
                reasoning=[
                    "Analyzed available information sources",
                    "Synthesized comprehensive response",
                    "Applied source attribution and confidence scoring"
                ],
                confidence=response.get("confidence", 0.8),
                execution_time=execution_time,
                metadata={
                    "sources_used": response.get("sources_used", []),
                    "synthesis_method": "llm_based",
                    "response_length": len(response.get("answer", ""))
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Synthesis failed: {e}")
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                result=None,
                reasoning=[f"Synthesis failed: {str(e)}"],
                confidence=0.0,
                execution_time=execution_time
            )
    
    def _parse_synthesis_input(self, task: str, context: Dict[str, Any]) -> SynthesisInput:
        """Parse input data for synthesis."""
        
        # Extract original query
        original_query = context.get("original_query", task)
        
        # Extract results from different agents
        kb_results = context.get("knowledge_base_results")
        tool_results = context.get("tool_results") 
        web_results = context.get("web_search_results")
        additional_context = context.get("additional_context")
        
        return SynthesisInput(
            original_query=original_query,
            knowledge_base_results=kb_results,
            tool_results=tool_results,
            web_search_results=web_results,
            additional_context=additional_context
        )
    
    async def _synthesize_response(self, synthesis_input: SynthesisInput) -> Dict[str, Any]:
        """Generate synthesized response using LLM."""
        
        # Prepare sources summary
        sources_summary = []
        if synthesis_input.knowledge_base_results:
            sources_summary.append("Knowledge Base: Internal documents and data")
        if synthesis_input.tool_results:
            sources_summary.append("Tools: Calculations, code execution, conversions")
        if synthesis_input.web_search_results:
            sources_summary.append("Web Search: Current information from the internet")
        
        sources_text = ", ".join(sources_summary) if sources_summary else "No external sources"
        
        # Format knowledge base results
        kb_text = "None available"
        if synthesis_input.knowledge_base_results:
            kb_results = synthesis_input.knowledge_base_results
            if isinstance(kb_results, dict) and "sources" in kb_results:
                kb_sources = kb_results["sources"][:3]  # Limit to top 3 sources
                kb_text = "\n".join([
                    f"- {source.get('content', '')[:200]}..." 
                    for source in kb_sources
                ])
        
        # Format tool results
        tool_text = "None available"
        if synthesis_input.tool_results:
            tool_results = synthesis_input.tool_results
            if isinstance(tool_results, dict):
                if "result" in tool_results:
                    tool_text = str(tool_results["result"])[:300]
        
        # Format web search results
        web_text = "None available"
        if synthesis_input.web_search_results:
            web_results = synthesis_input.web_search_results
            if isinstance(web_results, dict) and "results" in web_results:
                web_sources = web_results["results"][:2]  # Limit to top 2 results
                web_text = "\n".join([
                    f"- {result.get('title', '')}: {result.get('snippet', '')[:150]}..."
                    for result in web_sources
                ])
        
        # Create synthesis prompt
        prompt = self.synthesis_prompt.format(
            sources_summary=sources_text,
            query=synthesis_input.original_query,
            kb_results=kb_text,
            tool_results=tool_text,
            web_results=web_text
        )
        
        # Generate response using LLM
        messages = [
            {"role": "system", "content": "You are an expert information synthesizer."},
            {"role": "user", "content": prompt}
        ]
        
        llm_response = await llm_manager.generate(messages, max_tokens=1500)
        
        # Calculate confidence based on available sources
        confidence = self._calculate_confidence(synthesis_input)
        
        # Identify sources used
        sources_used = []
        if synthesis_input.knowledge_base_results:
            sources_used.append("knowledge_base")
        if synthesis_input.tool_results:
            sources_used.append("tools")
        if synthesis_input.web_search_results:
            sources_used.append("web_search")
        
        return {
            "answer": llm_response,
            "confidence": confidence,
            "sources_used": sources_used,
            "synthesis_method": "multi_source_llm",
            "original_query": synthesis_input.original_query
        }
    
    def _calculate_confidence(self, synthesis_input: SynthesisInput) -> float:
        """Calculate confidence score based on available information."""
        
        confidence = 0.3  # Base confidence
        
        # Add confidence based on available sources
        if synthesis_input.knowledge_base_results:
            kb_results = synthesis_input.knowledge_base_results
            if isinstance(kb_results, dict) and "sources" in kb_results:
                # Higher confidence if we have relevant KB sources
                num_sources = len(kb_results["sources"])
                confidence += min(0.4, num_sources * 0.1)
        
        if synthesis_input.tool_results:
            tool_results = synthesis_input.tool_results
            if isinstance(tool_results, dict) and tool_results.get("success"):
                confidence += 0.2
        
        if synthesis_input.web_search_results:
            web_results = synthesis_input.web_search_results
            if isinstance(web_results, dict) and "results" in web_results:
                num_results = len(web_results["results"])
                confidence += min(0.2, num_results * 0.05)
        
        # Cap confidence at 0.95
        return min(0.95, confidence)
    
    async def _detect_hallucination(self, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect potential hallucinations in the response."""
        
        # Simple hallucination detection based on source alignment
        hallucination_indicators = {
            "unsupported_claims": [],
            "confidence_score": 0.8,
            "source_alignment": "good"
        }
        
        # Check for specific patterns that might indicate hallucination
        response_lower = response.lower()
        
        # Flag potential issues
        warning_phrases = [
            "i think", "i believe", "probably", "might be", "could be",
            "it seems", "appears to be", "likely", "possibly"
        ]
        
        uncertainty_count = sum(1 for phrase in warning_phrases if phrase in response_lower)
        
        if uncertainty_count > 3:
            hallucination_indicators["confidence_score"] = 0.6
            hallucination_indicators["source_alignment"] = "uncertain"
            hallucination_indicators["unsupported_claims"].append(
                "High uncertainty language detected"
            )
        
        return hallucination_indicators
    
    def get_capabilities(self) -> List[str]:
        """Return synthesis agent capabilities."""
        return [
            "information_synthesis",
            "multi_source_integration",
            "response_generation",
            "source_attribution",
            "confidence_scoring",
            "hallucination_detection"
        ]