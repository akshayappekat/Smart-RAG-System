"""Hallucination detection for RAG responses."""

import asyncio
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..llm.providers import llm_manager

logger = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """Result from hallucination detection."""
    is_hallucination: bool
    confidence_score: float
    detected_issues: List[str]
    source_alignment_score: float
    factual_consistency_score: float
    reasoning: List[str]
    metadata: Dict[str, Any]


class HallucinationDetector:
    """Detects potential hallucinations in RAG responses."""
    
    def __init__(self):
        self.detection_prompt = """
You are an expert fact-checker analyzing AI responses for potential hallucinations or inaccuracies.

Original Query: {query}

AI Response: {response}

Source Documents: {sources}

Analyze the AI response for:
1. Factual accuracy based on provided sources
2. Claims not supported by the sources
3. Contradictions with source material
4. Overly confident statements without evidence
5. Fabricated details or statistics

Respond with a JSON object:
{{
    "is_hallucination": boolean,
    "confidence": float (0-1),
    "issues": ["list of specific issues found"],
    "source_alignment": float (0-1),
    "factual_consistency": float (0-1),
    "reasoning": ["explanation of analysis"]
}}
"""
    
    async def detect_hallucination(self, query: str, response: str, 
                                 sources: List[Dict[str, Any]]) -> HallucinationResult:
        """Detect potential hallucinations in the response."""
        
        try:
            # Perform multiple detection methods
            llm_detection = await self._llm_based_detection(query, response, sources)
            rule_detection = self._rule_based_detection(response, sources)
            source_alignment = self._calculate_source_alignment(response, sources)
            
            # Combine results
            combined_result = self._combine_detection_results(
                llm_detection, rule_detection, source_alignment
            )
            
            logger.debug(f"Hallucination detection completed: {combined_result.is_hallucination}")
            return combined_result
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            
            # Return conservative result on error
            return HallucinationResult(
                is_hallucination=False,
                confidence_score=0.5,
                detected_issues=[f"Detection error: {str(e)}"],
                source_alignment_score=0.5,
                factual_consistency_score=0.5,
                reasoning=["Error in hallucination detection"],
                metadata={"error": str(e)}
            )
    
    async def _llm_based_detection(self, query: str, response: str, 
                                 sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to detect hallucinations."""
        
        # Prepare source text
        source_text = "\n\n".join([
            f"Source {i+1}: {source.get('content', '')[:500]}..."
            for i, source in enumerate(sources[:3])
        ])
        
        # Create detection prompt
        prompt = self.detection_prompt.format(
            query=query,
            response=response,
            sources=source_text
        )
        
        messages = [
            {"role": "system", "content": "You are an expert fact-checker."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            llm_response = await llm_manager.generate(messages, max_tokens=800)
            
            # Parse JSON response
            import json
            result = json.loads(llm_response)
            
            return {
                "is_hallucination": result.get("is_hallucination", False),
                "confidence": result.get("confidence", 0.5),
                "issues": result.get("issues", []),
                "source_alignment": result.get("source_alignment", 0.5),
                "factual_consistency": result.get("factual_consistency", 0.5),
                "reasoning": result.get("reasoning", [])
            }
            
        except Exception as e:
            logger.warning(f"LLM-based detection failed: {e}")
            return {
                "is_hallucination": False,
                "confidence": 0.5,
                "issues": [],
                "source_alignment": 0.5,
                "factual_consistency": 0.5,
                "reasoning": ["LLM detection failed"]
            }
    
    def _rule_based_detection(self, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rule-based hallucination detection."""
        
        issues = []
        confidence = 0.8
        
        # Check for overly confident language without sources
        confidence_patterns = [
            r'\b(definitely|certainly|absolutely|without doubt|guaranteed)\b',
            r'\b(always|never|all|none|every|no one)\b',
            r'\b(exactly|precisely|specifically)\s+\d+',
        ]
        
        for pattern in confidence_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                issues.append(f"Overly confident language detected: {matches}")
        
        # Check for specific numbers/statistics without source support
        number_patterns = r'\b\d+\.?\d*\s*%|\b\d+\.?\d*\s*(million|billion|thousand)\b'
        numbers = re.findall(number_patterns, response, re.IGNORECASE)
        
        if numbers and not self._numbers_supported_by_sources(numbers, sources):
            issues.append(f"Specific statistics without clear source support: {numbers}")
        
        # Check for temporal claims
        temporal_patterns = [
            r'\b(today|yesterday|last week|this year|recently|currently)\b',
            r'\b(latest|newest|most recent|up-to-date)\b'
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append("Temporal claims that may not be current")
                break
        
        # Check response length vs source content
        if len(response) > 1000 and len(sources) < 2:
            issues.append("Long response with limited source material")
        
        # Calculate rule-based confidence
        if len(issues) > 3:
            confidence = 0.3
        elif len(issues) > 1:
            confidence = 0.6
        
        return {
            "issues": issues,
            "confidence": confidence,
            "rule_based_score": 1.0 - (len(issues) * 0.2)
        }
    
    def _calculate_source_alignment(self, response: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate how well the response aligns with sources."""
        
        if not sources:
            return 0.3  # Low alignment if no sources
        
        response_words = set(response.lower().split())
        source_words = set()
        
        for source in sources:
            content = source.get('content', '')
            source_words.update(content.lower().split())
        
        if not source_words:
            return 0.3
        
        # Calculate word overlap
        overlap = len(response_words.intersection(source_words))
        total_response_words = len(response_words)
        
        if total_response_words == 0:
            return 0.3
        
        alignment_score = min(1.0, overlap / total_response_words * 2)
        return alignment_score
    
    def _numbers_supported_by_sources(self, numbers: List[str], sources: List[Dict[str, Any]]) -> bool:
        """Check if numbers in response are supported by sources."""
        
        source_text = " ".join([source.get('content', '') for source in sources])
        
        for number in numbers:
            if number.lower() in source_text.lower():
                return True
        
        return False
    
    def _combine_detection_results(self, llm_result: Dict[str, Any], 
                                 rule_result: Dict[str, Any], 
                                 source_alignment: float) -> HallucinationResult:
        """Combine results from different detection methods."""
        
        # Combine issues
        all_issues = []
        all_issues.extend(llm_result.get("issues", []))
        all_issues.extend(rule_result.get("issues", []))
        
        # Calculate combined confidence
        llm_confidence = llm_result.get("confidence", 0.5)
        rule_confidence = rule_result.get("confidence", 0.5)
        
        # Weight LLM result more heavily
        combined_confidence = (llm_confidence * 0.6 + rule_confidence * 0.4)
        
        # Determine if hallucination detected
        is_hallucination = (
            llm_result.get("is_hallucination", False) or
            len(all_issues) > 2 or
            source_alignment < 0.3
        )
        
        # Adjust confidence based on source alignment
        if source_alignment < 0.4:
            combined_confidence *= 0.8
        
        # Build reasoning
        reasoning = []
        reasoning.extend(llm_result.get("reasoning", []))
        reasoning.append(f"Source alignment score: {source_alignment:.2f}")
        reasoning.append(f"Rule-based issues found: {len(rule_result.get('issues', []))}")
        
        return HallucinationResult(
            is_hallucination=is_hallucination,
            confidence_score=combined_confidence,
            detected_issues=all_issues,
            source_alignment_score=source_alignment,
            factual_consistency_score=llm_result.get("factual_consistency", 0.7),
            reasoning=reasoning,
            metadata={
                "llm_detection": llm_result.get("is_hallucination", False),
                "rule_issues_count": len(rule_result.get("issues", [])),
                "source_count": len(sources) if sources else 0
            }
        )


# Global hallucination detector instance
hallucination_detector = HallucinationDetector()