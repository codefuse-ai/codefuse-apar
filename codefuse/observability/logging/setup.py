"""Unified logging configuration using structlog"""

import os
import logging
import structlog
from pathlib import Path
from typing import Optional
from .utils import path_to_slug

# State tracking
_logging_initialized = False
_session_dir: Optional[Path] = None

def _json_formatter(logger, method_name, event_dict):
    """Custom formatter that outputs clean JSON lines"""
    import json
    from datetime import datetime, timezone
    
    # Build JSON structure
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": event_dict.pop("level", "info"),
    }
    
    # Add logger name if present
    if "logger" in event_dict:
        log_data["logger"] = event_dict.pop("logger")
    
    # Add event/message
    if "event" in event_dict:
        log_data["message"] = event_dict.pop("event")
    
    # Add all remaining fields
    log_data.update(event_dict)
    
    return json.dumps(log_data, ensure_ascii=False)


# Configure standard logging backend with NullHandler (silent before setup)
stdlib_logger = logging.getLogger("codefuse.main")
stdlib_logger.addHandler(logging.NullHandler())
stdlib_logger.propagate = False
stdlib_logger.setLevel(logging.DEBUG)

# Configure structlog once at module load time
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _json_formatter,
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Create global logger instance (ready to use, silent before setup)
mainLogger = structlog.get_logger("codefuse.main")


def setup_logging(
    session_id: str,
    workspace_path: Optional[str] = None,
    logs_dir: str = "~/.cfuse/logs",
    verbose: bool = False,
) -> Path:
    """
    Setup file handler for logging
    
    Configures mainLogger for debug logs (file only, append mode).
    Trajectory and LLM messages are now handled by dedicated writers.
    
    Args:
        session_id: Unique session identifier
        workspace_path: Workspace path (default: cwd)
        logs_dir: Base logs directory
        verbose: Enable console output (currently unused, kept for compatibility)
        
    Returns:
        Session directory path
    """
    global _logging_initialized, _session_dir
    
    if _logging_initialized:
        return _session_dir
    
    # Prepare session directory
    workspace_path = workspace_path or os.getcwd()
    base_logs_dir = Path(logs_dir).expanduser()
    workspace_slug = path_to_slug(workspace_path)
    session_dir = base_logs_dir / workspace_slug / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    _session_dir = session_dir
    
    # Configure main logger: DEBUG level, file only
    main_logger = logging.getLogger("codefuse.main")
    main_logger.handlers.clear()  # Remove NullHandler
    main_handler = logging.FileHandler(session_dir / "main.log", mode='a', encoding='utf-8')
    main_handler.setLevel(logging.DEBUG)
    main_logger.addHandler(main_handler)
    
    _logging_initialized = True
    
    mainLogger.info(
        "Logging initialized",
        session_id=session_id,
        workspace=workspace_path,
        logs_dir=str(session_dir),
        verbose=verbose,
    )
    
    return session_dir


def get_session_dir() -> Optional[Path]:
    """Get the current session directory path"""
    return _session_dir


def close_all_loggers():
    """Close all logger handlers and flush buffers"""
    logger = logging.getLogger("codefuse.main")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
