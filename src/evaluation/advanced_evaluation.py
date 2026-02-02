"""Advanced evaluation framework with RAGAS integration and bias detection."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import json
from pathlib import Path

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not available. Install with: pip install ragas")

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Comprehensive evaluation result."""
    query: str
    answer: str
    ground_truth: Optional[str]
    contexts: List[str]
    
    # RAGAS metrics
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float
    context_recall_score: float
    answer_correctness_score: Optional[float] = None
    answer_similarity_score: Optional[float] = None
    
    # Bias metrics
    bias_score: float
    bias_categories: List[str]
    
    # Custom metrics
    hallucination_score: float
    confidence_score: float
    response_time: float
    
    # Metadata
    timestamp: datetime
    model_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "contexts": self.contexts,
            "faithfulness_score": self.faithfulness_score,
            "answer_relevancy_score": self.answer_relevancy_score,
            "context_precision_score": self.context_precision_score,
            "context_recall_score": self.context_recall_score,
            "answer_correctness_score": self.answer_correctness_score,
            "answer_similarity_score": self.answer_similarity_score,
            "bias_score": self.bias_score,
            "bias_categories": self.bias_categories,
            "hallucination_score": self.hallucination_score,
            "confidence_score": self.confidence_score,
            "response_time": self.response_time,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version
        }


class BiasDetector:
    """Advanced bias detection system."""
    
    def __init__(self):
        self.bias_patterns = {
            "gender": [
                r"\b(he|she|his|her|him|man|woman|male|female|boy|girl)\b",
                r"\b(masculine|feminine|manly|womanly)\b"
            ],
            "racial": [
                r"\b(black|white|asian|hispanic|latino|african|european)\b",
                r"\b(race|racial|ethnicity|ethnic)\b"
            ],
            "age": [
                r"\b(young|old|elderly|senior|teenager|child|adult)\b",
                r"\b(age|aged|aging)\b"
            ],
            "religious": [
                r"\b(christian|muslim|jewish|hindu|buddhist|atheist)\b",
                r"\b(religion|religious|faith|belief)\b"
            ],
            "socioeconomic": [
                r"\b(poor|rich|wealthy|poverty|income|class)\b",
                r"\b(education|educated|uneducated)\b"
            ]
        }
        
        self.bias_keywords = {
            "gender": ["stereotype", "assumption", "typical", "usually", "generally"],
            "racial": ["culture", "background", "origin", "heritage"],
            "age": ["generation", "experience", "maturity", "wisdom"],
            "religious": ["values", "morals", "beliefs", "practices"],
            "socioeconomic": ["status", "background", "opportunity", "access"]
        }
    
    def detect_bias(self, text: str) -> Tuple[float, List[str]]:
        """Detect bias in text and return score and categories."""
        import re
        
        detected_categories = []
        bias_indicators = 0
        total_words = len(text.split())
        
        for category, patterns in self.bias_patterns.items():
            category_matches = 0
            
            for pattern in patterns:
                matches = re.findall(pattern, text.lower())
                category_matches += len(matches)
            
            # Check for bias keywords in context
            if category_matches > 0:
                for keyword in self.bias_keywords[category]:
                    if keyword in text.lower():
                        category_matches += 1
            
            if category_matches > 0:
                detected_categories.append(category)
                bias_indicators += category_matches
        
        # Calculate bias score (0-1, where 1 is highly biased)
        bias_score = min(bias_indicators / max(total_words, 1), 1.0)
        
        return bias_score, detected_categories
    
    def analyze_bias_context(self, query: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """Analyze bias in query, answer, and contexts."""
        query_bias, query_categories = self.detect_bias(query)
        answer_bias, answer_categories = self.detect_bias(answer)
        
        context_biases = []
        context_categories = []
        for context in contexts:
            bias, categories = self.detect_bias(context)
            context_biases.append(bias)
            context_categories.extend(categories)
        
        return {
            "query_bias": query_bias,
            "query_bias_categories": query_categories,
            "answer_bias": answer_bias,
            "answer_bias_categories": answer_categories,
            "context_bias_avg": np.mean(context_biases) if context_biases else 0.0,
            "context_bias_categories": list(set(context_categories)),
            "overall_bias": (query_bias + answer_bias + np.mean(context_biases or [0])) / 3
        }


class AdvancedEvaluationFramework:
    """Comprehensive evaluation framework with RAGAS and bias detection."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("data/evaluation")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.bias_detector = BiasDetector()
        self.evaluation_history: List[EvaluationResult] = []
        
        # Load existing evaluation history
        self._load_evaluation_history()
    
    async def evaluate_rag_response(self,
                                  query: str,
                                  answer: str,
                                  contexts: List[str],
                                  ground_truth: Optional[str] = None,
                                  model_version: str = "unknown",
                                  response_time: float = 0.0) -> EvaluationResult:
        """Comprehensive evaluation of RAG response."""
        start_time = datetime.now()
        
        # RAGAS evaluation
        ragas_scores = {}
        if RAGAS_AVAILABLE and contexts:
            try:
                ragas_scores = await self._evaluate_with_ragas(
                    query, answer, contexts, ground_truth
                )
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed: {e}")
                ragas_scores = self._get_default_ragas_scores()
        else:
            ragas_scores = self._get_default_ragas_scores()
        
        # Bias detection
        bias_analysis = self.bias_detector.analyze_bias_context(query, answer, contexts)
        
        # Hallucination detection (simplified)
        hallucination_score = await self._detect_hallucination(answer, contexts)
        
        # Confidence estimation
        confidence_score = self._estimate_confidence(answer, contexts)
        
        # Create evaluation result
        result = EvaluationResult(
            query=query,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
            faithfulness_score=ragas_scores.get("faithfulness", 0.0),
            answer_relevancy_score=ragas_scores.get("answer_relevancy", 0.0),
            context_precision_score=ragas_scores.get("context_precision", 0.0),
            context_recall_score=ragas_scores.get("context_recall", 0.0),
            answer_correctness_score=ragas_scores.get("answer_correctness"),
            answer_similarity_score=ragas_scores.get("answer_similarity"),
            bias_score=bias_analysis["overall_bias"],
            bias_categories=bias_analysis["answer_bias_categories"],
            hallucination_score=hallucination_score,
            confidence_score=confidence_score,
            response_time=response_time,
            timestamp=start_time,
            model_version=model_version
        )
        
        # Store result
        self.evaluation_history.append(result)
        self._save_evaluation_result(result)
        
        logger.info(f"Evaluation completed: faithfulness={result.faithfulness_score:.3f}, "
                   f"relevancy={result.answer_relevancy_score:.3f}, "
                   f"bias={result.bias_score:.3f}")
        
        return result
    
    async def _evaluate_with_ragas(self,
                                 query: str,
                                 answer: str,
                                 contexts: List[str],
                                 ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate using RAGAS metrics."""
        if not RAGAS_AVAILABLE:
            return self._get_default_ragas_scores()
        
        # Prepare dataset
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        
        dataset = Dataset.from_dict(data)
        
        # Define metrics to evaluate
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        
        if ground_truth:
            metrics.extend([answer_correctness, answer_similarity])
        
        try:
            # Run evaluation
            result = evaluate(dataset, metrics=metrics)
            
            return {
                "faithfulness": result["faithfulness"],
                "answer_relevancy": result["answer_relevancy"],
                "context_precision": result["context_precision"],
                "context_recall": result["context_recall"],
                "answer_correctness": result.get("answer_correctness"),
                "answer_similarity": result.get("answer_similarity")
            }
        except Exception as e:
            logger.error(f"RAGAS evaluation error: {e}")
            return self._get_default_ragas_scores()
    
    def _get_default_ragas_scores(self) -> Dict[str, float]:
        """Get default scores when RAGAS is not available."""
        return {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "answer_correctness": None,
            "answer_similarity": None
        }
    
    async def _detect_hallucination(self, answer: str, contexts: List[str]) -> float:
        """Detect hallucination in the answer based on contexts."""
        if not contexts or not answer:
            return 1.0  # High hallucination if no context
        
        # Simple approach: check if answer contains information not in contexts
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for context in contexts:
            context_words.update(context.lower().split())
        
        # Calculate overlap
        overlap = len(answer_words.intersection(context_words))
        total_answer_words = len(answer_words)
        
        if total_answer_words == 0:
            return 1.0
        
        # Hallucination score: 1 - (overlap / total_answer_words)
        hallucination_score = 1.0 - (overlap / total_answer_words)
        return max(0.0, min(1.0, hallucination_score))
    
    def _estimate_confidence(self, answer: str, contexts: List[str]) -> float:
        """Estimate confidence in the answer."""
        if not answer or not contexts:
            return 0.0
        
        # Factors affecting confidence
        factors = []
        
        # Length factor (longer answers might be more confident)
        length_factor = min(len(answer.split()) / 50, 1.0)
        factors.append(length_factor)
        
        # Context relevance factor
        answer_words = set(answer.lower().split())
        context_relevance = 0.0
        
        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(answer_words.intersection(context_words))
            relevance = overlap / max(len(answer_words), 1)
            context_relevance = max(context_relevance, relevance)
        
        factors.append(context_relevance)
        
        # Uncertainty indicators (reduce confidence)
        uncertainty_words = ["maybe", "possibly", "might", "could", "uncertain", "unclear"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in answer.lower())
        uncertainty_factor = max(0.0, 1.0 - (uncertainty_count * 0.2))
        factors.append(uncertainty_factor)
        
        # Average all factors
        return sum(factors) / len(factors)
    
    def get_evaluation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get evaluation summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_evaluations = [
            eval_result for eval_result in self.evaluation_history 
            if eval_result.timestamp >= cutoff_time
        ]
        
        if not recent_evaluations:
            return {"message": "No evaluations in the specified period"}
        
        # Calculate averages
        avg_faithfulness = np.mean([e.faithfulness_score for e in recent_evaluations])
        avg_relevancy = np.mean([e.answer_relevancy_score for e in recent_evaluations])
        avg_precision = np.mean([e.context_precision_score for e in recent_evaluations])
        avg_recall = np.mean([e.context_recall_score for e in recent_evaluations])
        avg_bias = np.mean([e.bias_score for e in recent_evaluations])
        avg_hallucination = np.mean([e.hallucination_score for e in recent_evaluations])
        avg_confidence = np.mean([e.confidence_score for e in recent_evaluations])
        
        # Bias analysis
        all_bias_categories = []
        for eval_result in recent_evaluations:
            all_bias_categories.extend(eval_result.bias_categories)
        
        bias_category_counts = {}
        for category in all_bias_categories:
            bias_category_counts[category] = bias_category_counts.get(category, 0) + 1
        
        # Quality distribution
        high_quality = sum(1 for e in recent_evaluations 
                          if e.faithfulness_score > 0.8 and e.answer_relevancy_score > 0.8)
        medium_quality = sum(1 for e in recent_evaluations 
                           if 0.5 <= e.faithfulness_score <= 0.8 and 0.5 <= e.answer_relevancy_score <= 0.8)
        low_quality = len(recent_evaluations) - high_quality - medium_quality
        
        return {
            "period_hours": hours,
            "total_evaluations": len(recent_evaluations),
            "average_scores": {
                "faithfulness": avg_faithfulness,
                "answer_relevancy": avg_relevancy,
                "context_precision": avg_precision,
                "context_recall": avg_recall,
                "bias_score": avg_bias,
                "hallucination_score": avg_hallucination,
                "confidence_score": avg_confidence
            },
            "quality_distribution": {
                "high_quality": high_quality,
                "medium_quality": medium_quality,
                "low_quality": low_quality
            },
            "bias_analysis": {
                "average_bias_score": avg_bias,
                "bias_category_counts": bias_category_counts,
                "high_bias_responses": sum(1 for e in recent_evaluations if e.bias_score > 0.5)
            },
            "performance_alerts": self._generate_performance_alerts(recent_evaluations)
        }
    
    def _generate_performance_alerts(self, evaluations: List[EvaluationResult]) -> List[str]:
        """Generate performance alerts based on evaluation results."""
        alerts = []
        
        if not evaluations:
            return alerts
        
        # Check for low faithfulness
        low_faithfulness = [e for e in evaluations if e.faithfulness_score < 0.5]
        if len(low_faithfulness) > len(evaluations) * 0.2:  # More than 20%
            alerts.append(f"{len(low_faithfulness)} responses have low faithfulness scores")
        
        # Check for high bias
        high_bias = [e for e in evaluations if e.bias_score > 0.5]
        if high_bias:
            alerts.append(f"{len(high_bias)} responses show potential bias")
        
        # Check for high hallucination
        high_hallucination = [e for e in evaluations if e.hallucination_score > 0.7]
        if high_hallucination:
            alerts.append(f"{len(high_hallucination)} responses may contain hallucinations")
        
        # Check for low confidence
        low_confidence = [e for e in evaluations if e.confidence_score < 0.3]
        if len(low_confidence) > len(evaluations) * 0.3:  # More than 30%
            alerts.append(f"{len(low_confidence)} responses have low confidence scores")
        
        return alerts
    
    def export_evaluation_data(self, format: str = "json") -> str:
        """Export evaluation data for analysis."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_evaluations": len(self.evaluation_history),
            "evaluations": [eval_result.to_dict() for eval_result in self.evaluation_history]
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _save_evaluation_result(self, result: EvaluationResult) -> None:
        """Save individual evaluation result."""
        try:
            results_file = self.storage_dir / "evaluation_results.jsonl"
            with open(results_file, 'a') as f:
                f.write(json.dumps(result.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to save evaluation result: {e}")
    
    def _load_evaluation_history(self) -> None:
        """Load existing evaluation history."""
        try:
            results_file = self.storage_dir / "evaluation_results.jsonl"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        # Reconstruct EvaluationResult (simplified)
                        result = EvaluationResult(**data)
                        self.evaluation_history.append(result)
                
                logger.info(f"Loaded {len(self.evaluation_history)} evaluation results")
        except Exception as e:
            logger.error(f"Failed to load evaluation history: {e}")


# Global evaluation framework instance
evaluation_framework = AdvancedEvaluationFramework()