"""Base agent class for the multi-agent RAG system."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender: str
    recipient: str
    content: str
    message_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentResponse:
    """Response structure from agent execution."""
    agent_id: str
    success: bool
    result: Any
    reasoning: List[str]
    confidence: float
    execution_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.message_history: List[AgentMessage] = []
        self.execution_count = 0
        self.total_execution_time = 0.0
        
        logger.info(f"Initialized agent: {self.name} ({self.agent_id})")
    
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute the agent's primary function."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        pass
    
    async def send_message(self, recipient: str, content: str, 
                          message_type: str = "info") -> AgentMessage:
        """Send message to another agent."""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            message_type=message_type,
            timestamp=datetime.now()
        )
        
        self.message_history.append(message)
        logger.debug(f"Agent {self.agent_id} sent message to {recipient}: {content[:100]}...")
        
        return message
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive and process message from another agent."""
        self.message_history.append(message)
        logger.debug(f"Agent {self.agent_id} received message from {message.sender}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        avg_execution_time = (
            self.total_execution_time / self.execution_count 
            if self.execution_count > 0 else 0
        )
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "message_count": len(self.message_history),
            "capabilities": self.get_capabilities()
        }
    
    def _update_stats(self, execution_time: float) -> None:
        """Update agent execution statistics."""
        self.execution_count += 1
        self.total_execution_time += execution_time
    
    def __str__(self) -> str:
        return f"Agent({self.name}:{self.agent_id})"
    
    def __repr__(self) -> str:
        return f"Agent(id='{self.agent_id}', name='{self.name}')"