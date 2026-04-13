"""
Agent registry for Radiant Agentic RAG.

Provides a centralized registry for discovering and managing agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol defining the minimal agent interface."""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's main function."""
        ...


@dataclass
class AgentMetadata:
    """Metadata about a registered agent."""

    name: str
    description: str
    category: str = "general"
    version: str = "1.0.0"
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class RegisteredAgent:
    """A registered agent with its callable and metadata."""

    metadata: AgentMetadata
    callable: Callable[..., Any]
    instance: Optional[Any] = None  # The agent instance if class-based

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the agent."""
        return self.callable(*args, **kwargs)


class AgentRegistry:
    """
    Central registry for RAG pipeline agents.

    Provides:
        - Agent registration and discovery
        - Metadata management
        - Category-based organization
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._agents: Dict[str, RegisteredAgent] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(
        self,
        name: str,
        description: str,
        callable: Callable[..., Any],
        instance: Optional[Any] = None,
        category: str = "general",
        version: str = "1.0.0",
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> RegisteredAgent:
        """
        Register an agent.

        Args:
            name: Unique agent name
            description: Human-readable description
            callable: The agent's callable (function or method)
            instance: Optional agent instance (for class-based agents)
            category: Agent category for organization
            version: Agent version string
            input_schema: JSON schema for inputs
            output_schema: JSON schema for outputs
            tags: List of tags for filtering

        Returns:
            The registered agent

        Raises:
            ValueError: If agent name already registered
        """
        if name in self._agents:
            raise ValueError(f"Agent already registered: {name}")

        metadata = AgentMetadata(
            name=name,
            description=description,
            category=category,
            version=version,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            tags=tags or [],
        )

        agent = RegisteredAgent(
            metadata=metadata,
            callable=callable,
            instance=instance,
        )

        self._agents[name] = agent

        # Add to category index
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

        logger.debug(f"Registered agent: {name} (category={category})")
        return agent

    def register_instance(
        self,
        instance: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "general",
        method_name: str = "run",
        **kwargs: Any,
    ) -> RegisteredAgent:
        """
        Register an agent from a class instance.

        Args:
            instance: Agent instance with a run method
            name: Agent name (defaults to class name)
            description: Description (defaults to class docstring)
            category: Agent category
            method_name: Name of the method to use as callable
            **kwargs: Additional metadata

        Returns:
            The registered agent
        """
        agent_name = name or instance.__class__.__name__
        agent_desc = description or instance.__class__.__doc__ or f"{agent_name} agent"

        # Get the callable method
        if not hasattr(instance, method_name):
            raise ValueError(f"Instance has no method '{method_name}'")

        callable_method = getattr(instance, method_name)

        return self.register(
            name=agent_name,
            description=agent_desc.strip(),
            callable=callable_method,
            instance=instance,
            category=category,
            **kwargs,
        )

    def unregister(self, name: str) -> bool:
        """
        Remove an agent from the registry.

        Args:
            name: Agent name to remove

        Returns:
            True if agent was removed
        """
        if name not in self._agents:
            return False

        agent = self._agents[name]
        category = agent.metadata.category

        del self._agents[name]

        # Remove from category index
        if category in self._categories:
            self._categories[category] = [
                n for n in self._categories[category] if n != name
            ]

        logger.debug(f"Unregistered agent: {name}")
        return True

    def get(self, name: str) -> RegisteredAgent:
        """
        Get a registered agent by name.

        Args:
            name: Agent name

        Returns:
            RegisteredAgent

        Raises:
            KeyError: If agent not found
        """
        if name not in self._agents:
            raise KeyError(f"Unknown agent: {name}")
        return self._agents[name]

    def get_optional(self, name: str) -> Optional[RegisteredAgent]:
        """
        Get a registered agent by name, or None if not found.

        Args:
            name: Agent name

        Returns:
            RegisteredAgent or None
        """
        return self._agents.get(name)

    def has(self, name: str) -> bool:
        """Check if an agent is registered."""
        return name in self._agents

    def list_agents(self, category: Optional[str] = None) -> List[RegisteredAgent]:
        """
        List registered agents.

        Args:
            category: Optional category filter

        Returns:
            List of registered agents
        """
        if category:
            names = self._categories.get(category, [])
            return [self._agents[n] for n in names if n in self._agents]

        return list(self._agents.values())

    def list_names(self, category: Optional[str] = None) -> List[str]:
        """
        List agent names.

        Args:
            category: Optional category filter

        Returns:
            List of agent names
        """
        if category:
            return list(self._categories.get(category, []))
        return list(self._agents.keys())

    def list_categories(self) -> List[str]:
        """List all agent categories."""
        return list(self._categories.keys())

    def find_by_tag(self, tag: str) -> List[RegisteredAgent]:
        """
        Find agents by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching agents
        """
        return [
            agent for agent in self._agents.values()
            if tag in agent.metadata.tags
        ]

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Invoke an agent by name.

        Args:
            name: Agent name
            *args: Positional arguments for agent
            **kwargs: Keyword arguments for agent

        Returns:
            Agent result
        """
        agent = self.get(name)
        return agent(*args, **kwargs)

    def get_metadata(self, name: str) -> AgentMetadata:
        """
        Get agent metadata.

        Args:
            name: Agent name

        Returns:
            AgentMetadata
        """
        return self.get(name).metadata

    def to_dict(self) -> Dict[str, Any]:
        """
        Export registry as dictionary.

        Returns:
            Dictionary representation of registry
        """
        return {
            "agents": {
                name: {
                    "name": agent.metadata.name,
                    "description": agent.metadata.description,
                    "category": agent.metadata.category,
                    "version": agent.metadata.version,
                    "tags": agent.metadata.tags,
                }
                for name, agent in self._agents.items()
            },
            "categories": dict(self._categories),
            "total_agents": len(self._agents),
        }

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __iter__(self):
        return iter(self._agents.values())


# Global registry instance (optional convenience)
_global_registry: Optional[AgentRegistry] = None


def get_global_registry() -> AgentRegistry:
    """Get or create the global agent registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def register_agent(
    name: str,
    description: str,
    category: str = "general",
    **kwargs: Any,
) -> Callable[[Callable], Callable]:
    """
    Decorator for registering a function as an agent.

    Usage:
        @register_agent("MyAgent", "Does something useful")
        def my_agent(query: str) -> str:
            return "result"
    """
    def decorator(fn: Callable) -> Callable:
        registry = get_global_registry()
        registry.register(
            name=name,
            description=description,
            callable=fn,
            category=category,
            **kwargs,
        )
        return fn
    return decorator
