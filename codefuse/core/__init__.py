"""
Core Module - Agent loop, context engine (unified context and session management)
"""

from codefuse.core.environment import EnvironmentInfo
from codefuse.core.context_engine import ContextEngine
from codefuse.core.agent_config import AgentProfile, AgentProfileManager
from codefuse.core.read_tracker import ReadTracker
from codefuse.core.agent_loop import AgentLoop, AgentEvent
from codefuse.core.tool_executor import ToolExecutor
from codefuse.observability import (
    setup_logging,
    mainLogger,
    get_session_dir,
    close_all_loggers,
    MetricsCollector,
)

# For backward compatibility, Session is now an alias to ContextEngine
Session = ContextEngine

__all__ = [
    "EnvironmentInfo",
    "ContextEngine",
    "Session",  # Alias for backward compatibility
    "ReadTracker",
    "AgentProfile",
    "AgentProfileManager",
    "AgentLoop",
    "AgentEvent",
    "ToolExecutor",
    "setup_logging",
    "mainLogger",
    "get_session_dir",
    "close_all_loggers",
    "MetricsCollector",
]

