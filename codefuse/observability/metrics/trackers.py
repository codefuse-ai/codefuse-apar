"""
Metrics Trackers - Context managers for tracking execution metrics
"""

import time
from datetime import datetime, timezone
from typing import Optional

from .models import ToolCallMetric, APICallMetric, PromptMetric, SessionMetric


class ToolCallTracker:
    """Context manager for tracking tool call execution"""
    
    def __init__(self, metric: ToolCallMetric, parent_prompt: PromptMetric):
        self.metric = metric
        self.parent_prompt = parent_prompt
        self._start_time = time.time()
    
    def set_error(self, error: str):
        """Set error for this tool call"""
        self.metric.success = False
        self.metric.error = error
    
    def set_success(self, success: bool = True):
        """Set success status"""
        self.metric.success = success
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        self.metric.end_time = datetime.now(timezone.utc).isoformat()
        self.metric.duration = end - self._start_time
        
        if exc_type is not None:
            self.metric.success = False
            self.metric.error = str(exc_val)
        
        return False  # Don't suppress exceptions


class APICallTracker:
    """Context manager for tracking API call"""
    
    def __init__(self, metric: APICallMetric, parent_prompt: PromptMetric):
        self.metric = metric
        self.parent_prompt = parent_prompt
        self._start_time = time.time()
    
    def set_tokens(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cache_creation_tokens: Optional[int] = None,
        cache_read_tokens: Optional[int] = None,
    ):
        """Set token usage information"""
        self.metric.prompt_tokens = prompt_tokens
        self.metric.completion_tokens = completion_tokens
        self.metric.total_tokens = total_tokens
        self.metric.cache_creation_tokens = cache_creation_tokens
        self.metric.cache_read_tokens = cache_read_tokens
    
    def set_model(self, model: str):
        """Set model name"""
        self.metric.model = model
    
    def set_finish_reason(self, finish_reason: str):
        """Set finish reason"""
        self.metric.finish_reason = finish_reason
    
    def set_error(self, error: str):
        """Set error for this API call"""
        self.metric.success = False
        self.metric.error = error
    
    def set_success(self, success: bool = True):
        """Set success status"""
        self.metric.success = success
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        self.metric.end_time = datetime.now(timezone.utc).isoformat()
        self.metric.duration = end - self._start_time
        
        if exc_type is not None:
            self.metric.success = False
            self.metric.error = str(exc_val)
        
        return False  # Don't suppress exceptions


class PromptTracker:
    """Context manager for tracking a prompt/query"""
    
    def __init__(self, metric: PromptMetric, session: SessionMetric):
        self.metric = metric
        self.session = session
        self._start_time = time.time()
    
    def increment_iteration(self):
        """Increment iteration count"""
        self.metric.iterations += 1
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        self.metric.end_time = datetime.now(timezone.utc).isoformat()
        self.metric.duration = end - self._start_time
        
        return False  # Don't suppress exceptions

