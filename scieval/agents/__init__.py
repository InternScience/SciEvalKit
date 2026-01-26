"""
Agent module with lazy loading for optional dependencies.

This module supports multiple agent implementations, each with its own
optional dependencies. Agents are only imported when actually used,
preventing import errors for unavailable dependencies.
"""

import importlib
import sys
from typing import TYPE_CHECKING, Dict, Type

# Always available - base classes and records
from .base import AgentBase, EvalSample
from .records import EvalResult, StepResult, ToolCalling, TrajectoryStore

# Agent registry mapping agent names to their module paths
_AGENT_REGISTRY: Dict[str, str] = {
    "SmolAgentsAgent": ".smolagents",
    "EarthLinkAgent": ".EarthLink.earthlink",
    # Add more agents here as they are implemented
}


def __getattr__(name: str):
    """
    Lazy loading for agent classes.
    
    This allows importing agent classes without requiring their dependencies
    to be installed until the agent is actually used.
    
    Example:
        from scieval.agents import SmolAgentsAgent  # Only imports when accessed
    """
    if name in _AGENT_REGISTRY:
        module_path = _AGENT_REGISTRY[name]
        try:
            module = importlib.import_module(module_path, package=__name__)
            agent_class = getattr(module, name)
            # Cache the imported class in the module's namespace
            globals()[name] = agent_class
            return agent_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import {name}. "
                f"This agent requires additional dependencies that may not be installed. "
                f"Error: {e}"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    Show available attributes including lazily-loaded agents.
    """
    return list(globals().keys()) + list(_AGENT_REGISTRY.keys())


def get_available_agents() -> Dict[str, bool]:
    """
    Check which agents are available based on installed dependencies.
    
    Returns:
        Dict mapping agent names to availability status (True if dependencies are met)
    
    Example:
        >>> from scieval.agents import get_available_agents
        >>> available = get_available_agents()
        >>> print(available)
        {'SmolAgentsAgent': True, 'LangChainAgent': False}
    """
    available = {}
    for agent_name, module_path in _AGENT_REGISTRY.items():
        try:
            importlib.import_module(module_path, package=__name__)
            available[agent_name] = True
        except ImportError:
            available[agent_name] = False
    return available


def create_agent(agent_type: str, **kwargs) -> AgentBase:
    """
    Factory function to create agent instances dynamically.
    
    Args:
        agent_type: Name of the agent class (e.g., "SmolAgentsAgent")
        **kwargs: Arguments to pass to the agent constructor
    
    Returns:
        Instance of the requested agent
    
    Raises:
        ValueError: If agent_type is not registered
        ImportError: If agent dependencies are not installed
    
    Example:
        >>> from scieval.agents import create_agent
        >>> agent = create_agent("SmolAgentsAgent", model_version="gpt-4")
    """
    if agent_type not in _AGENT_REGISTRY:
        available = list(_AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available agents: {', '.join(available)}"
        )
    
    # Use __getattr__ to load the agent class
    agent_class = __getattr__(agent_type)
    return agent_class(**kwargs)


# Export base classes and utilities in __all__
# Agent classes are not included here as they are lazily loaded
__all__ = [
    # Base classes
    "AgentBase",
    "EvalSample",
    # Records
    "EvalResult",
    "StepResult",
    "ToolCalling",
    "TrajectoryStore",
    # Utility functions
    "get_available_agents",
    "create_agent",
]
