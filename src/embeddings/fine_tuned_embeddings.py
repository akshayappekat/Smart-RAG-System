"""Fine-tuned embedding service with domain adaptation capabilities."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class FineTunedEmbeddingService:
    """Advanced embedding service with fine-tuning capabilities."""
    
    def __init__(self, 
                 base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 fine_tuned_model_path: Optional[Path] = None,
                 domain: str = "general"):
        self.base_model_name = base_model
        self.domain = domain
        self.fine_tuned_model_path = fine_tuned_model_path or Path(f"models/fine_tuned_{domain}")
        
        # Load model (fine-tuned if available, otherwise base)
        if self.fine_tuned_model_path.exists():
            logger.info(f"Loading fine-tuned model from {self.fine_tuned_model_path}")
            self.model = SentenceTransformer(str(self.fine_tuned_model_path))
        else:
            logger.info(f"Loading base model: {base_model}")
            self.model = SentenceTransformer(base_model)
        
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        # Performance tracking
        self.embedding_cache = {}
        self.performance_metrics = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "avg_embedding_time": 0.0,
            "fine_tuning_history": []
        }
    
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts with batching and caching."""
        if not texts:
            return np.array([])
        
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[text_hash]))
                self.performance_metrics["cache_hits"] += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            start_time = asyncio.get_event_loop().time()
            
            # Process in batches
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.encode, batch
                )
                new_embeddings.extend(batch_embeddings)
            
            # Update cache
            for text, embedding in zip(uncached_texts, new_embeddings):
                self.embedding_cache[hash(text)] = embedding
            
            # Update performance metrics
            embedding_time = asyncio.get_event_loop().time() - start_time
            self.performance_metrics["total_embeddings"] += len(uncached_texts)
            self.performance_metrics["avg_embedding_time"] = (
                (self.performance_metrics["avg_embedding_time"] * 
                 (self.performance_metrics["total_embeddings"] - len(uncached_texts)) + 
                 embedding_time) / self.performance_metrics["total_embeddings"]
            )
        
        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # Place new embeddings
        for i, embedding in enumerate(new_embeddings):
            all_embeddings[uncached_indices[i]] = embedding
        
        return np.array(all_embeddings)
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        embeddings = await self.embed_texts([query])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    async def fine_tune_on_domain_data(self, 
                                     training_data: List[Tuple[str, str, float]],
                                     validation_data: Optional[List[Tuple[str, str, float]]] = None,
                                     epochs: int = 4,
                                     batch_size: int = 16,
                                     learning_rate: float = 2e-5) -> Dict[str, Any]:
        """Fine-tune the embedding model on domain-specific data.
        
        Args:
            training_data: List of (text1, text2, similarity_score) tuples
            validation_data: Optional validation data in same format
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting fine-tuning with {len(training_data)} training examples")
        
        # Prepare training examples
        train_examples = [
            InputExample(texts=[text1, text2], label=float(score))
            for text1, text2, score in training_data
        ]
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Prepare evaluator if validation data provided
        evaluator = None
        if validation_data:
            val_examples = [
                InputExample(texts=[text1, text2], label=float(score))
                for text1, text2, score in validation_data
            ]
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                val_examples, name='validation'
            )
        
        # Create output directory
        self.fine_tuned_model_path.mkdir(parents=True, exist_ok=True)
        
        # Fine-tune the model
        start_time = datetime.now()
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            evaluator=evaluator,
            evaluation_steps=len(train_dataloader) // 2,
            output_path=str(self.fine_tuned_model_path),
            save_best_model=True,
            optimizer_params={'lr': learning_rate}
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Record fine-tuning history
        training_record = {
            "timestamp": end_time.isoformat(),
            "domain": self.domain,
            "training_examples": len(training_data),
            "validation_examples": len(validation_data) if validation_data else 0,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_time_seconds": training_time,
            "model_path": str(self.fine_tuned_model_path)
        }
        
        self.performance_metrics["fine_tuning_history"].append(training_record)
        
        # Save training record
        history_file = self.fine_tuned_model_path / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.performance_metrics["fine_tuning_history"], f, indent=2)
        
        logger.info(f"Fine-tuning completed in {training_time:.2f} seconds")
        
        return training_record
    
    def create_training_data_from_queries(self, 
                                        queries: List[str],
                                        relevant_docs: List[List[str]],
                                        irrelevant_docs: List[List[str]]) -> List[Tuple[str, str, float]]:
        """Create training data from query-document pairs.
        
        Args:
            queries: List of queries
            relevant_docs: List of relevant documents for each query
            irrelevant_docs: List of irrelevant documents for each query
            
        Returns:
            Training data as (query, doc, similarity_score) tuples
        """
        training_data = []
        
        for i, query in enumerate(queries):
            # Positive examples (relevant documents)
            if i < len(relevant_docs):
                for doc in relevant_docs[i]:
                    training_data.append((query, doc, 1.0))
            
            # Negative examples (irrelevant documents)
            if i < len(irrelevant_docs):
                for doc in irrelevant_docs[i]:
                    training_data.append((query, doc, 0.0))
        
        logger.info(f"Created {len(training_data)} training examples from {len(queries)} queries")
        return training_data
    
    def evaluate_embedding_quality(self, 
                                 test_queries: List[str],
                                 test_docs: List[str],
                                 relevance_scores: List[List[float]]) -> Dict[str, float]:
        """Evaluate embedding quality using test data.
        
        Args:
            test_queries: List of test queries
            test_docs: List of test documents
            relevance_scores: Relevance scores for each query-doc pair
            
        Returns:
            Evaluation metrics
        """
        query_embeddings = self.model.encode(test_queries)
        doc_embeddings = self.model.encode(test_docs)
        
        # Calculate cosine similarities
        similarities = np.dot(query_embeddings, doc_embeddings.T)
        
        # Calculate metrics
        metrics = {}
        
        for i, query_sims in enumerate(similarities):
            if i < len(relevance_scores):
                # Calculate correlation with human relevance scores
                correlation = np.corrcoef(query_sims, relevance_scores[i])[0, 1]
                metrics[f"query_{i}_correlation"] = correlation if not np.isnan(correlation) else 0.0
        
        # Overall metrics
        if metrics:
            metrics["avg_correlation"] = np.mean(list(metrics.values()))
        
        return metrics
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get embedding service performance metrics."""
        cache_hit_rate = (
            self.performance_metrics["cache_hits"] / 
            max(self.performance_metrics["total_embeddings"], 1)
        )
        
        return {
            **self.performance_metrics,
            "cache_hit_rate": cache_hit_rate,
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.base_model_name,
            "domain": self.domain,
            "is_fine_tuned": self.fine_tuned_model_path.exists()
        }
    
    def save_model(self, path: Optional[Path] = None) -> None:
        """Save the current model."""
        save_path = path or self.fine_tuned_model_path
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: Path) -> None:
        """Load a saved model."""
        if path.exists():
            self.model = SentenceTransformer(str(path))
            self.fine_tuned_model_path = path
            logger.info(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model not found at {path}")


# Domain-specific embedding services
class HealthcareEmbeddingService(FineTunedEmbeddingService):
    """Healthcare domain-specific embedding service."""
    
    def __init__(self):
        super().__init__(
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            domain="healthcare"
        )
        
        # Healthcare-specific preprocessing
        self.medical_abbreviations = {
            "MI": "myocardial infarction",
            "HTN": "hypertension",
            "DM": "diabetes mellitus",
            "COPD": "chronic obstructive pulmonary disease",
            "CHF": "congestive heart failure",
            "CAD": "coronary artery disease"
        }
    
    def preprocess_medical_text(self, text: str) -> str:
        """Preprocess medical text by expanding abbreviations."""
        processed_text = text
        for abbrev, expansion in self.medical_abbreviations.items():
            processed_text = processed_text.replace(abbrev, expansion)
        return processed_text
    
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Override to include medical text preprocessing."""
        processed_texts = [self.preprocess_medical_text(text) for text in texts]
        return await super().embed_texts(processed_texts, batch_size)


class LegalEmbeddingService(FineTunedEmbeddingService):
    """Legal domain-specific embedding service."""
    
    def __init__(self):
        super().__init__(
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            domain="legal"
        )
        
        # Legal-specific preprocessing
        self.legal_terms = {
            "plaintiff": "the party who initiates a lawsuit",
            "defendant": "the party being sued or accused",
            "tort": "a wrongful act that causes harm",
            "liability": "legal responsibility for damages"
        }
    
    def preprocess_legal_text(self, text: str) -> str:
        """Preprocess legal text with domain-specific handling."""
        # Convert to lowercase for consistency
        processed_text = text.lower()
        
        # Handle legal citations (simplified)
        import re
        citation_pattern = r'\d+\s+[A-Z][a-z]+\.?\s+\d+'
        processed_text = re.sub(citation_pattern, '[CITATION]', processed_text)
        
        return processed_text
    
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Override to include legal text preprocessing."""
        processed_texts = [self.preprocess_legal_text(text) for text in texts]
        return await super().embed_texts(processed_texts, batch_size)