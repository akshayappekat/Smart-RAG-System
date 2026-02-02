"""Planning agent for query decomposition and task orchestration."""

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
class QueryPlan:
    """Structured plan for query execution."""
    original_query: str
    sub_tasks: List[Dict[str, Any]]
    execution_order: List[str]
    required_agents: List[str]
    estimated_complexity: float
    reasoning: List[str]


class PlanningAgent(BaseAgent):
    """Agent responsible for query analysis and task decomposition."""
    
    def __init__(self):
        super().__init__(
            agent_id="planner",
            name="Planning Agent",
            description="Analyzes queries and creates execution plans"
        )
        
        self.planning_prompt = """
You are a Planning Agent in a multi-agent RAG system. Your job is to analyze user queries and create detailed execution plans.

Available Agents:
- retrieval_agent: Searches knowledge base for relevant information
- tool_agent: Uses external tools (web search, calculations, code execution)
- synthesis_agent: Combines information and generates final responses

For the given query, create a JSON plan with:
1. Sub-tasks that need to be completed
2. Which agents should handle each task
3. Execution order
4. Reasoning for the plan

Query: {query}

Respond with a JSON object containing:
{{
    "sub_tasks": [
        {{
            "task_id": "task_1",
            "description": "Search knowledge base for X",
            "agent": "retrieval_agent",
            "priority": 1,
            "dependencies": []
        }}
    ],
    "execution_order": ["task_1", "task_2"],
    "reasoning": ["Step 1 explanation", "Step 2 explanation"],
    "complexity": 0.7
}}
"""
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Create execution plan for the given query."""
        start_time = time.time()
        
        try:
            # Analyze query complexity and requirements
            plan = await self._create_plan(task, context or {})
            
            execution_time = time.time() - start_time
            self._update_stats(execution_time)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                result=plan,
                reasoning=plan.reasoning,
                confidence=0.9,
                execution_time=execution_time,
                metadata={"query_complexity": plan.estimated_complexity}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Planning failed: {e}")
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                result=None,
                reasoning=[f"Planning failed: {str(e)}"],
                confidence=0.0,
                execution_time=execution_time
            )
    
    async def _create_plan(self, query: str, context: Dict[str, Any]) -> QueryPlan:
        """Create detailed execution plan for the query."""
        
        # Use LLM to analyze query and create plan
        messages = [
            {"role": "system", "content": "You are an expert query planner."},
            {"role": "user", "content": self.planning_prompt.format(query=query)}
        ]
        
        response = await llm_manager.generate(messages, max_tokens=1000)
        
        try:
            # Parse LLM response as JSON
            plan_data = json.loads(response)
            
            # Create structured plan
            plan = QueryPlan(
                original_query=query,
                sub_tasks=plan_data.get("sub_tasks", []),
                execution_order=plan_data.get("execution_order", []),
                required_agents=list(set([task.get("agent") for task in plan_data.get("sub_tasks", [])])),
                estimated_complexity=plan_data.get("complexity", 0.5),
                reasoning=plan_data.get("reasoning", ["Plan created"])
            )
            
            logger.info(f"Created plan with {len(plan.sub_tasks)} tasks")
            return plan
            
        except json.JSONDecodeError:
            # Fallback to simple plan if JSON parsing fails
            logger.warning("Failed to parse LLM response, creating fallback plan")
            return self._create_fallback_plan(query)
    
    def _create_fallback_plan(self, query: str) -> QueryPlan:
        """Create a simple fallback plan when LLM planning fails."""
        
        # Determine if query needs external information
        needs_web_search = any(keyword in query.lower() for keyword in [
            "latest", "recent", "current", "today", "news", "price", "weather"
        ])
        
        # Determine if query needs calculations
        needs_calculation = any(keyword in query.lower() for keyword in [
            "calculate", "compute", "how much", "percentage", "cost", "price"
        ])
        
        sub_tasks = []
        execution_order = []
        
        # Always start with knowledge base search
        sub_tasks.append({
            "task_id": "kb_search",
            "description": f"Search knowledge base for: {query}",
            "agent": "retrieval_agent",
            "priority": 1,
            "dependencies": []
        })
        execution_order.append("kb_search")
        
        # Add web search if needed
        if needs_web_search:
            sub_tasks.append({
                "task_id": "web_search",
                "description": f"Search web for current information: {query}",
                "agent": "tool_agent",
                "priority": 2,
                "dependencies": []
            })
            execution_order.append("web_search")
        
        # Add calculation if needed
        if needs_calculation:
            sub_tasks.append({
                "task_id": "calculation",
                "description": f"Perform calculations for: {query}",
                "agent": "tool_agent",
                "priority": 2,
                "dependencies": []
            })
            execution_order.append("calculation")
        
        # Always end with synthesis
        sub_tasks.append({
            "task_id": "synthesis",
            "description": f"Synthesize final answer for: {query}",
            "agent": "synthesis_agent",
            "priority": 3,
            "dependencies": execution_order[:-1] if len(execution_order) > 1 else ["kb_search"]
        })
        execution_order.append("synthesis")
        
        return QueryPlan(
            original_query=query,
            sub_tasks=sub_tasks,
            execution_order=execution_order,
            required_agents=["retrieval_agent", "tool_agent", "synthesis_agent"],
            estimated_complexity=0.6,
            reasoning=[
                "Created fallback plan",
                f"Knowledge base search: Always included",
                f"Web search: {'Included' if needs_web_search else 'Not needed'}",
                f"Calculations: {'Included' if needs_calculation else 'Not needed'}"
            ]
        )
    
    def get_capabilities(self) -> List[str]:
        """Return planning agent capabilities."""
        return [
            "query_analysis",
            "task_decomposition", 
            "execution_planning",
            "complexity_estimation",
            "agent_coordination"
        ]