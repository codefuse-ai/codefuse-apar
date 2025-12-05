"""
Observability Module - Logging, Trajectory, and Metrics for CodeFuse Agent

This module provides comprehensive observability capabilities including:
- Debug logging using structlog (JSONL format)
- Trajectory recording (event stream in JSONL)
- LLM messages snapshots (JSON)
- Hierarchical metrics collection
- Session tracking and analysis
- HTTP server logging with rotation and cleanup
"""

# Logging exports
from .logging import (
    setup_logging,
    mainLogger,
    get_session_dir,
    close_all_loggers,
)

# HTTP logging exports
from .http_logger import (
    HTTPLogger,
    create_http_logger,
)

# Writer exports
from .trajectory import TrajectoryWriter
from .llm_messages import LLMMessagesWriter

# Metrics exports
from .metrics import (
    # Models
    ToolCallMetric,
    APICallMetric,
    PromptMetric,
    SessionMetric,
    # Trackers
    ToolCallTracker,
    APICallTracker,
    PromptTracker,
    # Collector
    MetricsCollector,
)

__all__ = [
    # Logging
    "setup_logging",
    "mainLogger",
    "get_session_dir",
    "close_all_loggers",
    # HTTP Logging
    "HTTPLogger",
    "create_http_logger",
    # Writers
    "TrajectoryWriter",
    "LLMMessagesWriter",
    # Metrics - Models
    "ToolCallMetric",
    "APICallMetric",
    "PromptMetric",
    "SessionMetric",
    # Metrics - Trackers
    "ToolCallTracker",
    "APICallTracker",
    "PromptTracker",
    # Metrics - Collector
    "MetricsCollector",
]

