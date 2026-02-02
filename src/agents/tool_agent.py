"""Tool agent for external tool usage (web search, calculations, code execution)."""

import asyncio
import time
import json
import math
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResponse

import logging
logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    success: bool
    result: Any
    execution_time: float
    error_message: Optional[str] = None


class ToolAgent(BaseAgent):
    """Agent responsible for using external tools and services."""
    
    def __init__(self):
        super().__init__(
            agent_id="tool_agent",
            name="Tool Agent", 
            description="Uses external tools for web search, calculations, and code execution"
        )
        
        self.available_tools = {
            "web_search": self._web_search,
            "calculator": self._calculator,
            "code_executor": self._code_executor,
            "unit_converter": self._unit_converter,
            "date_calculator": self._date_calculator
        }
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute tool-based task."""
        start_time = time.time()
        
        try:
            # Determine which tool to use
            tool_name = self._select_tool(task, context or {})
            
            # Execute the tool
            tool_result = await self._execute_tool(tool_name, task, context or {})
            
            execution_time = time.time() - start_time
            self._update_stats(execution_time)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=tool_result.success,
                result=tool_result.result,
                reasoning=[
                    f"Selected tool: {tool_name}",
                    f"Tool execution: {'Success' if tool_result.success else 'Failed'}",
                    f"Tool execution time: {tool_result.execution_time:.2f}s"
                ],
                confidence=0.8 if tool_result.success else 0.2,
                execution_time=execution_time,
                metadata={
                    "tool_used": tool_name,
                    "tool_success": tool_result.success,
                    "tool_execution_time": tool_result.execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool agent execution failed: {e}")
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                result=None,
                reasoning=[f"Tool execution failed: {str(e)}"],
                confidence=0.0,
                execution_time=execution_time
            )
    
    def _select_tool(self, task: str, context: Dict[str, Any]) -> str:
        """Select appropriate tool based on task description."""
        task_lower = task.lower()
        
        # Web search indicators
        if any(keyword in task_lower for keyword in [
            "search", "latest", "current", "news", "recent", "web", "internet"
        ]):
            return "web_search"
        
        # Calculator indicators
        if any(keyword in task_lower for keyword in [
            "calculate", "compute", "math", "add", "subtract", "multiply", "divide",
            "percentage", "percent", "sum", "total", "average"
        ]):
            return "calculator"
        
        # Code execution indicators
        if any(keyword in task_lower for keyword in [
            "code", "python", "execute", "run", "script", "program"
        ]):
            return "code_executor"
        
        # Unit conversion indicators
        if any(keyword in task_lower for keyword in [
            "convert", "conversion", "units", "meters", "feet", "celsius", "fahrenheit"
        ]):
            return "unit_converter"
        
        # Date calculation indicators
        if any(keyword in task_lower for keyword in [
            "date", "days", "weeks", "months", "years", "age", "duration"
        ]):
            return "date_calculator"
        
        # Default to web search
        return "web_search"
    
    async def _execute_tool(self, tool_name: str, task: str, context: Dict[str, Any]) -> ToolResult:
        """Execute the specified tool."""
        if tool_name not in self.available_tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                execution_time=0.0,
                error_message=f"Tool '{tool_name}' not available"
            )
        
        start_time = time.time()
        
        try:
            tool_function = self.available_tools[tool_name]
            result = await tool_function(task, context)
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool {tool_name} execution failed: {e}")
            
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _web_search(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web search (mock implementation)."""
        # In a real implementation, you would use Tavily, SerpAPI, or DuckDuckGo
        await asyncio.sleep(0.5)  # Simulate API call
        
        # Mock search results
        mock_results = [
            {
                "title": f"Search result for: {query}",
                "url": "https://example.com/result1",
                "snippet": f"This is a mock search result for the query '{query}'. In a real implementation, this would contain actual web search results from services like Tavily or SerpAPI.",
                "relevance_score": 0.9
            },
            {
                "title": f"Related information about: {query}",
                "url": "https://example.com/result2", 
                "snippet": f"Additional mock information related to '{query}'. Real implementation would provide current, relevant web content.",
                "relevance_score": 0.7
            }
        ]
        
        return {
            "query": query,
            "results": mock_results,
            "total_results": len(mock_results),
            "search_time": 0.5,
            "source": "mock_search_engine"
        }
    
    async def _calculator(self, expression: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mathematical calculations."""
        try:
            # Extract mathematical expressions from text
            math_expressions = re.findall(r'[\d+\-*/().\s]+', expression)
            
            results = []
            for expr in math_expressions:
                expr = expr.strip()
                if expr and len(expr) > 1:
                    try:
                        # Safe evaluation of mathematical expressions
                        result = eval(expr, {"__builtins__": {}, "math": math})
                        results.append({
                            "expression": expr,
                            "result": result,
                            "success": True
                        })
                    except:
                        results.append({
                            "expression": expr,
                            "result": None,
                            "success": False,
                            "error": "Invalid mathematical expression"
                        })
            
            # If no expressions found, try to extract numbers and perform basic operations
            if not results:
                numbers = re.findall(r'\d+\.?\d*', expression)
                if len(numbers) >= 2:
                    nums = [float(n) for n in numbers[:2]]
                    if "add" in expression.lower() or "sum" in expression.lower():
                        result = nums[0] + nums[1]
                        operation = "addition"
                    elif "subtract" in expression.lower():
                        result = nums[0] - nums[1]
                        operation = "subtraction"
                    elif "multiply" in expression.lower():
                        result = nums[0] * nums[1]
                        operation = "multiplication"
                    elif "divide" in expression.lower():
                        result = nums[0] / nums[1] if nums[1] != 0 else "Division by zero"
                        operation = "division"
                    else:
                        result = sum(nums)
                        operation = "sum"
                    
                    results.append({
                        "expression": f"{operation} of {numbers}",
                        "result": result,
                        "success": True
                    })
            
            return {
                "input": expression,
                "calculations": results,
                "total_calculations": len(results)
            }
            
        except Exception as e:
            return {
                "input": expression,
                "error": str(e),
                "success": False
            }
    
    async def _code_executor(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code safely (mock implementation)."""
        # In a real implementation, you would use a sandboxed environment
        await asyncio.sleep(0.2)  # Simulate execution time
        
        return {
            "code": code,
            "output": f"Mock execution result for code: {code[:50]}...",
            "success": True,
            "execution_time": 0.2,
            "note": "This is a mock implementation. Real code execution would require proper sandboxing."
        }
    
    async def _unit_converter(self, conversion_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert between different units."""
        # Simple unit conversions
        conversions = {
            # Length
            "meters_to_feet": lambda x: x * 3.28084,
            "feet_to_meters": lambda x: x / 3.28084,
            "km_to_miles": lambda x: x * 0.621371,
            "miles_to_km": lambda x: x / 0.621371,
            
            # Temperature
            "celsius_to_fahrenheit": lambda x: (x * 9/5) + 32,
            "fahrenheit_to_celsius": lambda x: (x - 32) * 5/9,
            
            # Weight
            "kg_to_pounds": lambda x: x * 2.20462,
            "pounds_to_kg": lambda x: x / 2.20462,
        }
        
        # Extract numbers from request
        numbers = re.findall(r'\d+\.?\d*', conversion_request)
        
        if numbers:
            value = float(numbers[0])
            request_lower = conversion_request.lower()
            
            for conversion_key, conversion_func in conversions.items():
                if any(unit in request_lower for unit in conversion_key.split('_')):
                    try:
                        result = conversion_func(value)
                        return {
                            "input": conversion_request,
                            "conversion": conversion_key,
                            "input_value": value,
                            "result": result,
                            "success": True
                        }
                    except:
                        pass
        
        return {
            "input": conversion_request,
            "error": "Could not determine conversion type or extract value",
            "success": False
        }
    
    async def _date_calculator(self, date_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate dates and durations."""
        from datetime import datetime, timedelta
        
        try:
            current_date = datetime.now()
            
            # Extract numbers from request
            numbers = re.findall(r'\d+', date_request)
            
            if numbers:
                value = int(numbers[0])
                request_lower = date_request.lower()
                
                if "days" in request_lower:
                    if "ago" in request_lower:
                        result_date = current_date - timedelta(days=value)
                        operation = f"{value} days ago"
                    else:
                        result_date = current_date + timedelta(days=value)
                        operation = f"{value} days from now"
                elif "weeks" in request_lower:
                    if "ago" in request_lower:
                        result_date = current_date - timedelta(weeks=value)
                        operation = f"{value} weeks ago"
                    else:
                        result_date = current_date + timedelta(weeks=value)
                        operation = f"{value} weeks from now"
                else:
                    result_date = current_date + timedelta(days=value)
                    operation = f"{value} days from now (default)"
                
                return {
                    "input": date_request,
                    "operation": operation,
                    "current_date": current_date.strftime("%Y-%m-%d"),
                    "result_date": result_date.strftime("%Y-%m-%d"),
                    "success": True
                }
        
        except Exception as e:
            pass
        
        return {
            "input": date_request,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "error": "Could not parse date calculation request",
            "success": False
        }
    
    def get_capabilities(self) -> List[str]:
        """Return tool agent capabilities."""
        return [
            "web_search",
            "mathematical_calculations", 
            "code_execution",
            "unit_conversion",
            "date_calculations",
            "external_api_integration"
        ]