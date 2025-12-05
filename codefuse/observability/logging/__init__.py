"""
Logging Module - Unified logging using structlog

Provides mainLogger for debug logging:
- mainLogger: Fine-grained debug logs (file only, append mode)

Trajectory and LLM messages are now handled by dedicated writers
in the observability module.

Usage:
    from codefuse.observability.logging import setup_logging, mainLogger
    
    setup_logging(session_id="session-123", verbose=True)
    
    # Simple logging with structured data
    mainLogger.info("tool executed", tool="read_file", duration=0.5)
    
    # With context binding
    request_logger = mainLogger.bind(request_id="req-123")
    request_logger.info("processing step", step=1)
"""

from .setup import (
    setup_logging,
    mainLogger,
    get_session_dir,
    close_all_loggers,
)

__all__ = [
    "setup_logging",
    "mainLogger",
    "get_session_dir",
    "close_all_loggers",
]

