"""
CodeFuse Agent - A lightweight, high-performance AI programming assistant framework
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("cfuse")
except importlib.metadata.PackageNotFoundError:
    # Development mode fallback
    __version__ = "0.1.0"

from codefuse.llm import create_llm, Message, MessageRole, Tool, LLMResponse
from codefuse.tools import BaseTool, ToolDefinition, ToolParameter, ToolResult, ToolRegistry
from codefuse.core import (
    EnvironmentInfo,
    Session,
    AgentProfile,
    AgentProfileManager,
    ContextEngine,
    AgentLoop,
    AgentEvent,
)
from codefuse.config import Config

__all__ = [
    # LLM
    "create_llm",
    "Message",
    "MessageRole",
    "Tool",
    "LLMResponse",
    # Tools
    "BaseTool",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    # Core
    "EnvironmentInfo",
    "Session",
    "AgentProfile",
    "AgentProfileManager",
    "ContextEngine",
    "AgentLoop",
    "AgentEvent",
    # Config
    "Config",
]

