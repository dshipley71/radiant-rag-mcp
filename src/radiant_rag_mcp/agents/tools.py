"""
Tool abstraction for agentic RAG pipeline.

Provides a unified interface for different tools that agents can use,
including retrieval, web search, calculator, and code execution.
"""

from __future__ import annotations

import ast
import logging
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of tools available to agents."""
    RETRIEVAL = "retrieval"
    WEB_SEARCH = "web_search"
    CALCULATOR = "calculator"
    CODE_EXECUTION = "code_execution"
    CUSTOM = "custom"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    tool_name: str
    tool_type: ToolType
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": str(self.output)[:1000] if self.output else None,
            "tool_name": self.tool_name,
            "tool_type": self.tool_type.value,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str, tool_type: ToolType):
        self.name = name
        self.description = description
        self.tool_type = tool_type
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool metadata to dictionary for LLM context."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.tool_type.value,
            "enabled": self.enabled,
        }


class CalculatorTool(BaseTool):
    """
    Calculator tool for mathematical expressions.
    
    Safely evaluates mathematical expressions without using eval().
    Supports: +, -, *, /, **, %, sqrt, abs, round
    """
    
    # Allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Allowed functions
    FUNCTIONS = {
        "sqrt": lambda x: x ** 0.5,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate mathematical expressions. Supports +, -, *, /, **, %, sqrt, abs, round, min, max.",
            tool_type=ToolType.CALCULATOR,
        )
    
    def execute(self, expression: str, **kwargs) -> ToolResult:
        """
        Evaluate a mathematical expression safely.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            ToolResult with computed value or error
        """
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate safely
            result = self._eval_node(tree.body)
            
            return ToolResult(
                success=True,
                output=result,
                tool_name=self.name,
                tool_type=self.tool_type,
                metadata={"expression": expression},
            )
        except Exception as e:
            logger.warning(f"Calculator error: {e}")
            return ToolResult(
                success=False,
                output=None,
                tool_name=self.name,
                tool_type=self.tool_type,
                error=str(e),
            )
    
    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)
        
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(operand)
        
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id.lower()
                if func_name in self.FUNCTIONS:
                    args = [self._eval_node(arg) for arg in node.args]
                    return self.FUNCTIONS[func_name](*args)
                raise ValueError(f"Unsupported function: {func_name}")
            raise ValueError("Complex function calls not supported")
        
        elif isinstance(node, ast.List):
            return [self._eval_node(elem) for elem in node.elts]
        
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elem) for elem in node.elts)
        
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")


class CodeExecutionTool(BaseTool):
    """
    Safe code execution tool for Python expressions.
    
    Executes simple Python code in a sandboxed environment.
    Primarily for string manipulation, list operations, and data transformations.
    """
    
    # Allowed built-in functions
    SAFE_BUILTINS = {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "sorted": sorted,
        "reversed": reversed,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "range": range,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "any": any,
        "all": all,
        "isinstance": isinstance,
        "type": type,
        "print": lambda *args: None,  # No-op print for safety
    }
    
    def __init__(self, timeout: float = 5.0):
        super().__init__(
            name="code_executor",
            description="Execute simple Python expressions for string manipulation, list operations, and data transformations.",
            tool_type=ToolType.CODE_EXECUTION,
        )
        self._timeout = timeout
    
    def execute(self, code: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            context: Optional context variables
            
        Returns:
            ToolResult with execution output or error
        """
        try:
            # Create safe execution environment
            safe_globals = {"__builtins__": self.SAFE_BUILTINS}
            safe_locals = dict(context or {})
            
            # Parse to check for dangerous constructs
            tree = ast.parse(code, mode='exec')
            self._validate_ast(tree)
            
            # Execute
            exec(compile(tree, '<string>', 'exec'), safe_globals, safe_locals)
            
            # Get result (last expression or 'result' variable)
            result = safe_locals.get('result', None)
            
            return ToolResult(
                success=True,
                output=result,
                tool_name=self.name,
                tool_type=self.tool_type,
                metadata={"code_length": len(code)},
            )
        except Exception as e:
            logger.warning(f"Code execution error: {e}")
            return ToolResult(
                success=False,
                output=None,
                tool_name=self.name,
                tool_type=self.tool_type,
                error=str(e),
            )
    
    def _validate_ast(self, tree: ast.AST) -> None:
        """Validate AST for dangerous constructs."""
        for node in ast.walk(tree):
            # Block imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Imports are not allowed")
            
            # Block attribute access on dangerous objects
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    raise ValueError(f"Private attribute access not allowed: {node.attr}")
            
            # Block exec, eval, compile
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('exec', 'eval', 'compile', 'open', '__import__'):
                        raise ValueError(f"Function not allowed: {node.func.id}")


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Provides tool discovery, selection, and execution coordination.
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_usage_stats: Dict[str, Dict[str, int]] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        self._tool_usage_stats[tool.name] = {"calls": 0, "successes": 0, "failures": 0}
        logger.debug(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            del self._tool_usage_stats[name]
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self, enabled_only: bool = True) -> List[BaseTool]:
        """List all registered tools."""
        tools = list(self._tools.values())
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return tools
    
    def get_tools_for_llm(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Get tool descriptions for LLM context."""
        return [t.to_dict() for t in self.list_tools(enabled_only)]
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                tool_name=tool_name,
                tool_type=ToolType.CUSTOM,
                error=f"Tool not found: {tool_name}",
            )
        
        if not tool.enabled:
            return ToolResult(
                success=False,
                output=None,
                tool_name=tool_name,
                tool_type=tool.tool_type,
                error=f"Tool is disabled: {tool_name}",
            )
        
        # Track usage
        self._tool_usage_stats[tool_name]["calls"] += 1
        
        result = tool.execute(**kwargs)
        
        if result.success:
            self._tool_usage_stats[tool_name]["successes"] += 1
        else:
            self._tool_usage_stats[tool_name]["failures"] += 1
        
        return result
    
    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get tool usage statistics."""
        return dict(self._tool_usage_stats)


class ToolSelector:
    """
    LLM-powered tool selector.
    
    Uses LLM to determine which tools to use for a given query.
    """
    
    def __init__(self, llm: "LLMClient", registry: ToolRegistry):
        self._llm = llm
        self._registry = registry
    
    def select_tools(
        self,
        query: str,
        context: Optional[str] = None,
        max_tools: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Select appropriate tools for a query.
        
        Args:
            query: User query
            context: Optional context
            max_tools: Maximum tools to select
            
        Returns:
            List of tool selections with arguments
        """
        tools = self._registry.get_tools_for_llm()
        
        if not tools:
            return []
        
        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']}"
            for t in tools
        ])
        
        system = f"""You are a tool selection agent.
Given a query, select which tools (if any) would be helpful.

Available tools:
{tools_desc}

Return a JSON array of tool selections:
[
  {{"tool": "tool_name", "reason": "why this tool helps", "args": {{"arg1": "value1"}}}}
]

If no tools are needed, return an empty array: []
Select at most {max_tools} tools."""

        user = f"Query: {query}"
        if context:
            user += f"\n\nContext: {context}"
        user += "\n\nReturn JSON array only."

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default=[],
            expected_type=list,
        )

        if not response.success or not isinstance(result, list):
            return []

        # Validate selections
        valid_tool_names = {t['name'] for t in tools}
        selections = []
        for item in result[:max_tools]:
            if isinstance(item, dict) and item.get('tool') in valid_tool_names:
                selections.append(item)

        return selections


def create_default_tool_registry() -> ToolRegistry:
    """Create a tool registry with default tools."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CodeExecutionTool())
    return registry
