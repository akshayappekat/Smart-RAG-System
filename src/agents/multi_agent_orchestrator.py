"""Multi-agent orchestrator for coordinating agent execution."""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResponse
from .planning_agent import PlanningAgent, QueryPlan
from .tool_agent import ToolAgent
from .synthesis_agent import SynthesisAgent
from ..rag_orchestrator import rag_orchestrator

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentResponse:
    """Response from multi-agent system execution."""
    success: bool
    final_answer: str
    confidence: float
    execution_plan: Optional[QueryPlan]
    agent_responses: Dict[str, AgentResponse]
    total_execution_time: float
    reasoning_chain: List[str]
    sources_used: List[str]
    metadata: Dict[str, Any]


class MultiAgentOrchestrator:
    """Orchestrates multiple agents to handle complex queries."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_history: List[MultiAgentResponse] = []
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info("Multi-agent orchestrator initialized")
    
    def _initialize_agents(self):
        """Initialize all agents in the system."""
        
        # Core agents
        self.agents["planner"] = PlanningAgent()
        self.agents["tool_agent"] = ToolAgent()
        self.agents["synthesis_agent"] = SynthesisAgent()
        
        # Note: retrieval_agent is handled by existing rag_orchestrator
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> MultiAgentResponse:
        """Process query using multi-agent system."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing query with multi-agent system: {query[:100]}...")
            
            # Step 1: Planning
            planning_response = await self._execute_planning(query, context or {})
            
            if not planning_response.success:
                return self._create_error_response(
                    "Planning failed", 
                    time.time() - start_time,
                    {"planning_error": planning_response.reasoning}
                )
            
            execution_plan: QueryPlan = planning_response.result
            
            # Step 2: Execute plan
            agent_responses = await self._execute_plan(execution_plan, context or {})
            
            # Step 3: Synthesis
            synthesis_response = await self._execute_synthesis(
                query, execution_plan, agent_responses, context or {}
            )
            
            total_execution_time = time.time() - start_time
            
            # Create final response
            final_response = MultiAgentResponse(
                success=synthesis_response.success,
                final_answer=synthesis_response.result.get("answer", "") if synthesis_response.success else "Failed to generate response",
                confidence=synthesis_response.confidence,
                execution_plan=execution_plan,
                agent_responses=agent_responses,
                total_execution_time=total_execution_time,
                reasoning_chain=self._build_reasoning_chain(execution_plan, agent_responses),
                sources_used=synthesis_response.result.get("sources_used", []) if synthesis_response.success else [],
                metadata={
                    "query": query,
                    "plan_complexity": execution_plan.estimated_complexity,
                    "agents_used": list(agent_responses.keys()),
                    "total_tasks": len(execution_plan.sub_tasks)
                }
            )
            
            # Store in history
            self.execution_history.append(final_response)
            
            logger.info(f"Multi-agent processing completed in {total_execution_time:.2f}s")
            return final_response
            
        except Exception as e:
            total_execution_time = time.time() - start_time
            logger.error(f"Multi-agent processing failed: {e}")
            
            return self._create_error_response(
                f"Multi-agent processing failed: {str(e)}",
                total_execution_time,
                {"error": str(e)}
            )
    
    async def _execute_planning(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Execute planning phase."""
        planner = self.agents["planner"]
        return await planner.execute(query, context)
    
    async def _execute_plan(self, plan: QueryPlan, context: Dict[str, Any]) -> Dict[str, AgentResponse]:
        """Execute the planned tasks."""
        agent_responses = {}
        
        # Group tasks by dependencies
        task_groups = self._group_tasks_by_dependencies(plan)
        
        # Execute task groups in order
        for group in task_groups:
            # Execute tasks in parallel within each group
            group_tasks = []
            
            for task in group:
                task_id = task["task_id"]
                agent_name = task["agent"]
                task_description = task["description"]
                
                if agent_name == "retrieval_agent":
                    # Use existing RAG orchestrator for retrieval
                    group_tasks.append(
                        self._execute_retrieval_task(task_id, task_description, context)
                    )
                elif agent_name in self.agents:
                    # Use multi-agent system agents
                    group_tasks.append(
                        self._execute_agent_task(task_id, agent_name, task_description, context)
                    )
                else:
                    logger.warning(f"Unknown agent: {agent_name}")
            
            # Wait for all tasks in group to complete
            if group_tasks:
                group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(group_results):
                    task = group[i]
                    task_id = task["task_id"]
                    
                    if isinstance(result, Exception):
                        logger.error(f"Task {task_id} failed: {result}")
                        agent_responses[task_id] = AgentResponse(
                            agent_id=task["agent"],
                            success=False,
                            result=None,
                            reasoning=[f"Task failed: {str(result)}"],
                            confidence=0.0,
                            execution_time=0.0
                        )
                    else:
                        agent_responses[task_id] = result
        
        return agent_responses
    
    def _group_tasks_by_dependencies(self, plan: QueryPlan) -> List[List[Dict[str, Any]]]:
        """Group tasks by their dependencies for parallel execution."""
        
        # Simple implementation: group by priority
        task_groups = {}
        
        for task in plan.sub_tasks:
            priority = task.get("priority", 1)
            if priority not in task_groups:
                task_groups[priority] = []
            task_groups[priority].append(task)
        
        # Return groups in priority order
        return [task_groups[priority] for priority in sorted(task_groups.keys())]
    
    async def _execute_retrieval_task(self, task_id: str, description: str, context: Dict[str, Any]) -> AgentResponse:
        """Execute retrieval task using existing RAG orchestrator."""
        start_time = time.time()
        
        try:
            # Extract query from description
            query = description.replace("Search knowledge base for: ", "")
            
            # Use existing RAG orchestrator
            rag_response = await rag_orchestrator.query(query)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_id="retrieval_agent",
                success=True,
                result={
                    "answer": rag_response.answer,
                    "sources": rag_response.sources,
                    "confidence": rag_response.confidence_score
                },
                reasoning=["Retrieved information from knowledge base"],
                confidence=rag_response.confidence_score,
                execution_time=execution_time,
                metadata={"task_id": task_id}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Retrieval task {task_id} failed: {e}")
            
            return AgentResponse(
                agent_id="retrieval_agent",
                success=False,
                result=None,
                reasoning=[f"Retrieval failed: {str(e)}"],
                confidence=0.0,
                execution_time=execution_time,
                metadata={"task_id": task_id}
            )
    
    async def _execute_agent_task(self, task_id: str, agent_name: str, description: str, context: Dict[str, Any]) -> AgentResponse:
        """Execute task using specified agent."""
        
        agent = self.agents[agent_name]
        task_context = {**context, "task_id": task_id}
        
        response = await agent.execute(description, task_context)
        response.metadata = response.metadata or {}
        response.metadata["task_id"] = task_id
        
        return response
    
    async def _execute_synthesis(self, query: str, plan: QueryPlan, 
                               agent_responses: Dict[str, AgentResponse], 
                               context: Dict[str, Any]) -> AgentResponse:
        """Execute synthesis phase."""
        
        # Prepare synthesis context
        synthesis_context = {
            "original_query": query,
            **context
        }
        
        # Add results from different agents
        for task_id, response in agent_responses.items():
            if response.success and response.result:
                task = next((t for t in plan.sub_tasks if t["task_id"] == task_id), None)
                if task:
                    agent_name = task["agent"]
                    
                    if agent_name == "retrieval_agent":
                        synthesis_context["knowledge_base_results"] = response.result
                    elif agent_name == "tool_agent":
                        synthesis_context["tool_results"] = response.result
                    elif "web" in task["description"].lower():
                        synthesis_context["web_search_results"] = response.result
        
        # Execute synthesis
        synthesis_agent = self.agents["synthesis_agent"]
        return await synthesis_agent.execute(query, synthesis_context)
    
    def _build_reasoning_chain(self, plan: QueryPlan, agent_responses: Dict[str, AgentResponse]) -> List[str]:
        """Build reasoning chain from execution."""
        
        reasoning_chain = [
            f"Query analysis: {plan.reasoning[0] if plan.reasoning else 'Analyzed query'}",
            f"Execution plan: {len(plan.sub_tasks)} tasks identified"
        ]
        
        # Add agent reasoning
        for task_id, response in agent_responses.items():
            task = next((t for t in plan.sub_tasks if t["task_id"] == task_id), None)
            if task and response.success:
                reasoning_chain.append(
                    f"{task['agent']}: {response.reasoning[0] if response.reasoning else 'Executed successfully'}"
                )
        
        reasoning_chain.append("Synthesized final response from all sources")
        
        return reasoning_chain
    
    def _create_error_response(self, error_message: str, execution_time: float, 
                             metadata: Dict[str, Any]) -> MultiAgentResponse:
        """Create error response."""
        
        return MultiAgentResponse(
            success=False,
            final_answer=f"I apologize, but I encountered an error while processing your query: {error_message}",
            confidence=0.0,
            execution_plan=None,
            agent_responses={},
            total_execution_time=execution_time,
            reasoning_chain=[f"Error: {error_message}"],
            sources_used=[],
            metadata=metadata
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = agent.get_stats()
        
        return {
            "total_queries_processed": len(self.execution_history),
            "average_execution_time": (
                sum(r.total_execution_time for r in self.execution_history) / 
                len(self.execution_history) if self.execution_history else 0
            ),
            "success_rate": (
                sum(1 for r in self.execution_history if r.success) / 
                len(self.execution_history) if self.execution_history else 0
            ),
            "agent_statistics": agent_stats,
            "available_agents": list(self.agents.keys())
        }
    
    async def reset_system(self):
        """Reset the multi-agent system."""
        self.execution_history.clear()
        
        # Reset individual agents
        for agent in self.agents.values():
            agent.message_history.clear()
            agent.execution_count = 0
            agent.total_execution_time = 0.0
        
        logger.info("Multi-agent system reset completed")


# Global multi-agent orchestrator instance
multi_agent_orchestrator = MultiAgentOrchestrator()