"""Advanced analytics dashboard for RAG system performance monitoring."""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    query_text: str
    timestamp: datetime
    processing_time: float
    confidence_score: float
    sources_count: int
    retrieval_time: float
    llm_time: float
    embedding_time: float
    cache_hit: bool
    user_feedback: Optional[float] = None  # 1-5 rating
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    timestamp: datetime
    total_queries: int
    avg_processing_time: float
    avg_confidence_score: float
    cache_hit_rate: float
    documents_count: int
    chunks_count: int
    memory_usage_mb: float
    disk_usage_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AnalyticsDashboard:
    """Advanced analytics dashboard for monitoring RAG system performance."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("data/analytics")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.query_metrics: List[QueryMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        
        # Performance thresholds
        self.performance_thresholds = {
            'processing_time_warning': 5.0,  # seconds
            'processing_time_critical': 10.0,
            'confidence_warning': 0.3,
            'confidence_critical': 0.1,
            'cache_hit_rate_warning': 0.5,
            'cache_hit_rate_critical': 0.3,
        }
        
        # Load existing metrics
        self._load_metrics()
    
    def record_query_metrics(self, 
                           query_id: str,
                           query_text: str,
                           processing_time: float,
                           confidence_score: float,
                           sources_count: int,
                           retrieval_time: float = 0.0,
                           llm_time: float = 0.0,
                           embedding_time: float = 0.0,
                           cache_hit: bool = False) -> None:
        """Record metrics for a query."""
        metrics = QueryMetrics(
            query_id=query_id,
            query_text=query_text,
            timestamp=datetime.now(),
            processing_time=processing_time,
            confidence_score=confidence_score,
            sources_count=sources_count,
            retrieval_time=retrieval_time,
            llm_time=llm_time,
            embedding_time=embedding_time,
            cache_hit=cache_hit
        )
        
        self.query_metrics.append(metrics)
        self._save_query_metrics()
        
        logger.info(f"Recorded query metrics: {query_id}, time: {processing_time:.2f}s, confidence: {confidence_score:.2f}")
    
    def record_system_metrics(self,
                            total_queries: int,
                            avg_processing_time: float,
                            avg_confidence_score: float,
                            cache_hit_rate: float,
                            documents_count: int,
                            chunks_count: int,
                            memory_usage_mb: float,
                            disk_usage_mb: float) -> None:
        """Record system-wide metrics."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            total_queries=total_queries,
            avg_processing_time=avg_processing_time,
            avg_confidence_score=avg_confidence_score,
            cache_hit_rate=cache_hit_rate,
            documents_count=documents_count,
            chunks_count=chunks_count,
            memory_usage_mb=memory_usage_mb,
            disk_usage_mb=disk_usage_mb
        )
        
        self.system_metrics.append(metrics)
        self._save_system_metrics()
        
        logger.info(f"Recorded system metrics: queries: {total_queries}, avg_time: {avg_processing_time:.2f}s")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_queries = [q for q in self.query_metrics if q.timestamp >= cutoff_time]
        
        if not recent_queries:
            return {
                "period_hours": hours,
                "total_queries": 0,
                "message": "No queries in the specified period"
            }
        
        # Calculate metrics
        processing_times = [q.processing_time for q in recent_queries]
        confidence_scores = [q.confidence_score for q in recent_queries]
        cache_hits = sum(1 for q in recent_queries if q.cache_hit)
        
        return {
            "period_hours": hours,
            "total_queries": len(recent_queries),
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "p95_processing_time": self._percentile(processing_times, 95),
            "avg_confidence_score": sum(confidence_scores) / len(confidence_scores),
            "min_confidence_score": min(confidence_scores),
            "max_confidence_score": max(confidence_scores),
            "cache_hit_rate": cache_hits / len(recent_queries),
            "queries_per_hour": len(recent_queries) / hours,
            "performance_alerts": self._get_performance_alerts(recent_queries)
        }
    
    def get_query_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze query patterns and trends."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_queries = [q for q in self.query_metrics if q.timestamp >= cutoff_time]
        
        if not recent_queries:
            return {"message": "No queries to analyze"}
        
        # Analyze query lengths
        query_lengths = [len(q.query_text.split()) for q in recent_queries]
        
        # Analyze query types (simple heuristics)
        question_queries = sum(1 for q in recent_queries if '?' in q.query_text)
        comparison_queries = sum(1 for q in recent_queries if any(word in q.query_text.lower() 
                                for word in ['compare', 'difference', 'versus', 'vs']))
        definition_queries = sum(1 for q in recent_queries if any(word in q.query_text.lower() 
                                for word in ['what is', 'define', 'definition']))
        
        # Time-based patterns
        hourly_distribution = {}
        for query in recent_queries:
            hour = query.timestamp.hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        return {
            "period_hours": hours,
            "total_queries": len(recent_queries),
            "avg_query_length_words": sum(query_lengths) / len(query_lengths),
            "query_types": {
                "questions": question_queries,
                "comparisons": comparison_queries,
                "definitions": definition_queries,
                "other": len(recent_queries) - question_queries - comparison_queries - definition_queries
            },
            "hourly_distribution": hourly_distribution,
            "peak_hour": max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None
        }
    
    def get_retrieval_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze retrieval performance and source usage."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_queries = [q for q in self.query_metrics if q.timestamp >= cutoff_time]
        
        if not recent_queries:
            return {"message": "No queries to analyze"}
        
        sources_counts = [q.sources_count for q in recent_queries]
        retrieval_times = [q.retrieval_time for q in recent_queries if q.retrieval_time > 0]
        
        return {
            "period_hours": hours,
            "total_queries": len(recent_queries),
            "avg_sources_per_query": sum(sources_counts) / len(sources_counts),
            "min_sources": min(sources_counts),
            "max_sources": max(sources_counts),
            "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
            "queries_with_no_sources": sum(1 for count in sources_counts if count == 0),
            "high_confidence_queries": sum(1 for q in recent_queries if q.confidence_score > 0.8),
            "low_confidence_queries": sum(1 for q in recent_queries if q.confidence_score < 0.3)
        }
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        if not self.system_metrics:
            return {"message": "No system metrics available"}
        
        latest_metrics = self.system_metrics[-1]
        
        # Calculate trends if we have multiple data points
        trends = {}
        if len(self.system_metrics) >= 2:
            prev_metrics = self.system_metrics[-2]
            trends = {
                "processing_time_trend": latest_metrics.avg_processing_time - prev_metrics.avg_processing_time,
                "confidence_trend": latest_metrics.avg_confidence_score - prev_metrics.avg_confidence_score,
                "cache_hit_rate_trend": latest_metrics.cache_hit_rate - prev_metrics.cache_hit_rate,
                "documents_growth": latest_metrics.documents_count - prev_metrics.documents_count,
                "chunks_growth": latest_metrics.chunks_count - prev_metrics.chunks_count
            }
        
        # Health status
        health_status = "healthy"
        alerts = []
        
        if latest_metrics.avg_processing_time > self.performance_thresholds['processing_time_critical']:
            health_status = "critical"
            alerts.append("Critical: Average processing time is very high")
        elif latest_metrics.avg_processing_time > self.performance_thresholds['processing_time_warning']:
            health_status = "warning"
            alerts.append("Warning: Average processing time is elevated")
        
        if latest_metrics.avg_confidence_score < self.performance_thresholds['confidence_critical']:
            health_status = "critical"
            alerts.append("Critical: Average confidence score is very low")
        elif latest_metrics.avg_confidence_score < self.performance_thresholds['confidence_warning']:
            if health_status != "critical":
                health_status = "warning"
            alerts.append("Warning: Average confidence score is low")
        
        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "health_status": health_status,
            "alerts": alerts,
            "current_metrics": latest_metrics.to_dict(),
            "trends": trends,
            "resource_usage": {
                "memory_mb": latest_metrics.memory_usage_mb,
                "disk_mb": latest_metrics.disk_usage_mb,
                "documents": latest_metrics.documents_count,
                "chunks": latest_metrics.chunks_count
            }
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "query_metrics": [q.to_dict() for q in self.query_metrics],
            "system_metrics": [s.to_dict() for s in self.system_metrics],
            "performance_thresholds": self.performance_thresholds
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a list of numbers."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _get_performance_alerts(self, queries: List[QueryMetrics]) -> List[str]:
        """Generate performance alerts based on recent queries."""
        alerts = []
        
        if not queries:
            return alerts
        
        # Check processing times
        slow_queries = [q for q in queries if q.processing_time > self.performance_thresholds['processing_time_warning']]
        if slow_queries:
            alerts.append(f"{len(slow_queries)} queries exceeded processing time warning threshold")
        
        # Check confidence scores
        low_confidence = [q for q in queries if q.confidence_score < self.performance_thresholds['confidence_warning']]
        if low_confidence:
            alerts.append(f"{len(low_confidence)} queries had low confidence scores")
        
        # Check cache hit rate
        cache_hits = sum(1 for q in queries if q.cache_hit)
        cache_hit_rate = cache_hits / len(queries)
        if cache_hit_rate < self.performance_thresholds['cache_hit_rate_warning']:
            alerts.append(f"Cache hit rate is low: {cache_hit_rate:.2%}")
        
        return alerts
    
    def _save_query_metrics(self) -> None:
        """Save query metrics to disk."""
        try:
            metrics_file = self.storage_dir / "query_metrics.json"
            data = [q.to_dict() for q in self.query_metrics[-1000:]]  # Keep last 1000 queries
            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save query metrics: {e}")
    
    def _save_system_metrics(self) -> None:
        """Save system metrics to disk."""
        try:
            metrics_file = self.storage_dir / "system_metrics.json"
            data = [s.to_dict() for s in self.system_metrics[-100:]]  # Keep last 100 system snapshots
            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save system metrics: {e}")
    
    def _load_metrics(self) -> None:
        """Load existing metrics from disk."""
        try:
            # Load query metrics
            query_file = self.storage_dir / "query_metrics.json"
            if query_file.exists():
                with open(query_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        self.query_metrics.append(QueryMetrics(**item))
            
            # Load system metrics
            system_file = self.storage_dir / "system_metrics.json"
            if system_file.exists():
                with open(system_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        self.system_metrics.append(SystemMetrics(**item))
            
            logger.info(f"Loaded {len(self.query_metrics)} query metrics and {len(self.system_metrics)} system metrics")
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")


# Global analytics dashboard instance
analytics_dashboard = AnalyticsDashboard()