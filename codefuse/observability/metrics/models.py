"""
Metrics Models - Data classes for hierarchical metrics
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ToolCallMetric:
    """Metrics for a single tool call"""
    tool_call_id: str
    tool_name: str
    start_time: str
    end_time: Optional[str] = None
    duration: Optional[float] = None  # seconds
    success: bool = True
    error: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


@dataclass
class APICallMetric:
    """Metrics for a single API call"""
    api_id: str
    start_time: str
    end_time: Optional[str] = None
    duration: Optional[float] = None  # seconds
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    success: bool = True
    error: Optional[str] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass
class PromptMetric:
    """Metrics for a single user prompt/query"""
    prompt_id: str
    user_query: str
    start_time: str
    end_time: Optional[str] = None
    duration: Optional[float] = None  # seconds
    iterations: int = 0
    api_calls: List[APICallMetric] = field(default_factory=list)
    tool_calls: List[ToolCallMetric] = field(default_factory=list)


@dataclass
class SessionMetric:
    """Metrics for entire session"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    duration: Optional[float] = None  # seconds
    total_prompts: int = 0
    prompts: List[PromptMetric] = field(default_factory=list)

