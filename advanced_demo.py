#!/usr/bin/env python3
"""Advanced demo showcasing enterprise-grade RAG system capabilities."""

import sys
import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.append('.')

from src.rag_orchestrator import rag_orchestrator
from src.analytics import analytics_dashboard

class AdvancedRAGDemo:
    """Comprehensive demo of advanced RAG system features."""
    
    def __init__(self):
        self.demo_queries = [
            "What are the main applications of AI in healthcare?",
            "Compare the effectiveness of AI systems in medical diagnosis",
            "What is the first-line treatment for diabetes according to clinical guidelines?",
            "How accurate are AI systems in diabetic retinopathy detection?",
            "What are the challenges in implementing AI in healthcare?",
            "Explain the diagnostic criteria for diabetes mellitus"