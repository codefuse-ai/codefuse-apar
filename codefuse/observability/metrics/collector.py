"""
Metrics Collector - Collects hierarchical metrics for Agent sessions
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union, List
from contextlib import contextmanager
from dataclasses import asdict

from .models import (
    SessionMetric,
    PromptMetric,
    APICallMetric,
    ToolCallMetric,
)
from .trackers import (
    ToolCallTracker,
    APICallTracker,
    PromptTracker,
)


# Model name aliases - map various naming conventions to canonical names
# This allows different architectures/services to use different naming schemes
MODEL_ALIASES = {
    # Internal proxy service aliases
    "claude_sonnet4": "claude-sonnet-4",
    "claude_opus4": "claude-opus-4",
    "claude_haiku4": "claude-haiku-4",
    "claude_sonnet4_5": "claude-sonnet-4.5",
    "claude-haiku-4_5": "claude-haiku-4.5"
    
    # Add more aliases as needed, for example:
    # "claude-sonnet-4.0": "claude-sonnet-4",
    # "sonnet4": "claude-sonnet-4",
    # "opus4": "claude-opus-4",
}

# Anthropic Pricing Table (USD per 1M tokens)
# Source: https://docs.anthropic.com/en/docs/about-claude/models#model-comparison
ANTHROPIC_PRICING = {
    "claude-opus-4.5": {
        "input": 5.00,
        "output": 25.00,
        "cache_write_5m": 6.25,
        "cache_write_1h": 10.00,
        "cache_read": 0.50,
    },
    "claude-opus-4.1": {
        "input": 15.00,
        "output": 75.00,
        "cache_write_5m": 18.75,
        "cache_write_1h": 30.00,
        "cache_read": 1.50,
    },
    "claude-opus-4": {
        "input": 15.00,
        "output": 75.00,
        "cache_write_5m": 18.75,
        "cache_write_1h": 30.00,
        "cache_read": 1.50,
    },
    "claude-sonnet-4.5": {
        "input": 3.00,
        "output": 15.00,
        "cache_write_5m": 3.75,
        "cache_write_1h": 6.00,
        "cache_read": 0.30,
    },
    "claude-sonnet-4": {
        "input": 3.00,
        "output": 15.00,
        "cache_write_5m": 3.75,
        "cache_write_1h": 6.00,
        "cache_read": 0.30,
    },
    "claude-haiku-4.5": {
        "input": 1.00,
        "output": 5.00,
        "cache_write_5m": 1.25,
        "cache_write_1h": 2.00,
        "cache_read": 0.10,
    },
}


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
    model_name: Optional[str] = None,
    cache_ttl: str = "5m"
) -> Dict[str, float]:
    """
    Calculate API cost based on token usage
    
    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        cache_creation_tokens: Number of tokens written to cache
        cache_read_tokens: Number of tokens read from cache
        model_name: Model name (for pricing lookup)
        cache_ttl: Cache TTL ("5m" or "1h")
        
    Returns:
        Dictionary with cost breakdown
    """
    # Try to find pricing for the model
    pricing = None
    if model_name:
        # Step 1: Try exact match in aliases (using original model name)
        canonical_name = MODEL_ALIASES.get(model_name)
        if canonical_name:
            pricing = ANTHROPIC_PRICING.get(canonical_name)
        
        # Step 2: Try exact match in pricing table (normalize for this step)
        if not pricing:
            model_lower = model_name.lower().replace("_", "-")
            if model_lower in ANTHROPIC_PRICING:
                pricing = ANTHROPIC_PRICING[model_lower]
        
        # Step 3: Try partial match (fallback for unknown variants)
        if not pricing:
            model_lower = model_name.lower().replace("_", "-")
            for key in ANTHROPIC_PRICING:
                key_normalized = key.lower().replace("_", "-")
                if key_normalized in model_lower or model_lower in key_normalized:
                    pricing = ANTHROPIC_PRICING[key]
                    break
    
    # If no pricing found, return None
    if not pricing:
        return {
            "with_cache": None,
            "without_cache": None,
            "savings": None,
            "model_found": False,
        }
    
    # Calculate cost with cache
    cache_write_key = f"cache_write_{cache_ttl}"
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    cache_write_cost = (cache_creation_tokens / 1_000_000) * pricing[cache_write_key]
    cache_read_cost = (cache_read_tokens / 1_000_000) * pricing["cache_read"]
    
    cost_with_cache = input_cost + output_cost + cache_write_cost + cache_read_cost
    
    # Calculate cost without cache (all cache tokens treated as normal input)
    total_input_without_cache = prompt_tokens + cache_creation_tokens + cache_read_tokens
    cost_without_cache = (
        (total_input_without_cache / 1_000_000) * pricing["input"] +
        (completion_tokens / 1_000_000) * pricing["output"]
    )
    
    savings = cost_without_cache - cost_with_cache
    savings_percent = (savings / cost_without_cache * 100) if cost_without_cache > 0 else 0
    
    return {
        "with_cache": round(cost_with_cache, 6),
        "without_cache": round(cost_without_cache, 6),
        "savings": round(savings, 6),
        "savings_percent": round(savings_percent, 2),
        "breakdown": {
            "input": round(input_cost, 6),
            "output": round(output_cost, 6),
            "cache_write": round(cache_write_cost, 6),
            "cache_read": round(cache_read_cost, 6),
        },
        "model_found": True,
        "currency": "USD",
    }


class MetricsCollector:
    """
    Collects hierarchical metrics for Agent sessions
    
    Hierarchy:
        Session
        └── Prompt (user query)
            ├── API Call (LLM interaction)
            └── Tool Call (tool execution)
    """
    
    def __init__(self, session_id: str):
        """
        Initialize metrics collector
        
        Args:
            session_id: Session ID
        """
        self.session_metric = SessionMetric(
            session_id=session_id,
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        self._session_start_time = time.time()
        
        # Track current prompt (for nested API/Tool calls)
        self._current_prompt: Optional[PromptMetric] = None
    
    @contextmanager
    def track_prompt(self, user_query: Union[str, List[Any]]):
        """
        Track a user prompt/query
        
        Args:
            user_query: User's query (text string or list of content blocks for multimodal)
            
        Yields:
            PromptTracker for this prompt
        """
        # Serialize user_query for storage (handle both str and List[ContentBlock])
        if isinstance(user_query, str):
            serialized_query = user_query
        else:
            # For multimodal content, create a summary for metrics
            text_parts = []
            image_count = 0
            for block in user_query:
                if hasattr(block, 'type'):
                    if block.type == "text" and hasattr(block, 'text') and block.text:
                        text_parts.append(block.text)
                    elif block.type == "image_url":
                        image_count += 1
            
            text_summary = text_parts[0][:100] if text_parts else ""
            serialized_query = f"{text_summary}... [with {image_count} image(s)]" if image_count > 0 else text_summary
        
        prompt_metric = PromptMetric(
            prompt_id=str(uuid.uuid4()),
            user_query=serialized_query,
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        
        self.session_metric.prompts.append(prompt_metric)
        self.session_metric.total_prompts += 1
        
        # Set as current prompt
        prev_prompt = self._current_prompt
        self._current_prompt = prompt_metric
        
        tracker = PromptTracker(prompt_metric, self.session_metric)
        
        try:
            yield tracker
        finally:
            # Restore previous prompt
            self._current_prompt = prev_prompt
    
    @contextmanager
    def track_api_call(self):
        """
        Track an API call to LLM
        
        Yields:
            APICallTracker for this API call
        """
        if self._current_prompt is None:
            raise RuntimeError("track_api_call must be called within track_prompt context")
        
        api_metric = APICallMetric(
            api_id=str(uuid.uuid4()),
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        
        self._current_prompt.api_calls.append(api_metric)
        
        tracker = APICallTracker(api_metric, self._current_prompt)
        
        yield tracker
    
    @contextmanager
    def track_tool_call(self, tool_name: str, tool_call_id: str, arguments: Optional[Dict[str, Any]] = None):
        """
        Track a tool call execution
        
        Args:
            tool_name: Name of the tool
            tool_call_id: Tool call ID
            arguments: Tool arguments
            
        Yields:
            ToolCallTracker for this tool call
        """
        if self._current_prompt is None:
            raise RuntimeError("track_tool_call must be called within track_prompt context")
        
        tool_metric = ToolCallMetric(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            start_time=datetime.now(timezone.utc).isoformat(),
            arguments=arguments,
        )
        
        self._current_prompt.tool_calls.append(tool_metric)
        
        tracker = ToolCallTracker(tool_metric, self._current_prompt)
        
        yield tracker
    
    def end_session(self):
        """Mark session as ended"""
        end = time.time()
        self.session_metric.end_time = datetime.now(timezone.utc).isoformat()
        self.session_metric.duration = end - self._session_start_time
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary of all collected metrics
        
        Returns:
            Dictionary containing aggregated metrics and statistics
        """
        # Ensure session is marked as ended
        if self.session_metric.end_time is None:
            self.end_session()
        
        # Aggregate API call metrics
        all_api_calls = []
        for prompt in self.session_metric.prompts:
            all_api_calls.extend(prompt.api_calls)
        
        api_success_count = sum(1 for api in all_api_calls if api.success)
        api_total_count = len(all_api_calls)
        api_success_rate = (api_success_count / api_total_count * 100) if api_total_count > 0 else 0
        
        total_prompt_tokens = sum(api.prompt_tokens or 0 for api in all_api_calls)
        total_completion_tokens = sum(api.completion_tokens or 0 for api in all_api_calls)
        total_tokens = sum(api.total_tokens or 0 for api in all_api_calls)
        total_cache_read_tokens = sum(api.cache_read_tokens or 0 for api in all_api_calls)
        total_cache_creation_tokens = sum(api.cache_creation_tokens or 0 for api in all_api_calls)
        
        api_durations = [api.duration for api in all_api_calls if api.duration is not None]
        avg_api_duration = (sum(api_durations) / len(api_durations)) if api_durations else 0
        
        # Aggregate tool call metrics
        all_tool_calls = []
        for prompt in self.session_metric.prompts:
            all_tool_calls.extend(prompt.tool_calls)
        
        tool_success_count = sum(1 for tool in all_tool_calls if tool.success)
        tool_total_count = len(all_tool_calls)
        tool_success_rate = (tool_success_count / tool_total_count * 100) if tool_total_count > 0 else 0
        
        tool_durations = [tool.duration for tool in all_tool_calls if tool.duration is not None]
        avg_tool_duration = (sum(tool_durations) / len(tool_durations)) if tool_durations else 0
        
        # Tool call breakdown by name
        tool_call_breakdown = {}
        for tool in all_tool_calls:
            name = tool.tool_name
            if name not in tool_call_breakdown:
                tool_call_breakdown[name] = {
                    "count": 0,
                    "success": 0,
                    "failed": 0,
                }
            tool_call_breakdown[name]["count"] += 1
            if tool.success:
                tool_call_breakdown[name]["success"] += 1
            else:
                tool_call_breakdown[name]["failed"] += 1
        
        # Prompt-level metrics
        prompt_durations = [p.duration for p in self.session_metric.prompts if p.duration is not None]
        avg_prompt_duration = (sum(prompt_durations) / len(prompt_durations)) if prompt_durations else 0
        total_iterations = sum(p.iterations for p in self.session_metric.prompts)
        
        # Get model name from first successful API call
        model_name = None
        for api in all_api_calls:
            if api.model:
                model_name = api.model
                break
        
        # Calculate cost
        cost_info = calculate_cost(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cache_creation_tokens=total_cache_creation_tokens,
            cache_read_tokens=total_cache_read_tokens,
            model_name=model_name,
            cache_ttl="5m"  # Default to 5m cache TTL
        )
        
        summary = {
            "session": {
                "session_id": self.session_metric.session_id,
                "start_time": self.session_metric.start_time,
                "end_time": self.session_metric.end_time,
                "duration": self.session_metric.duration,
                "total_prompts": self.session_metric.total_prompts,
            },
            "prompts": {
                "total": self.session_metric.total_prompts,
                "total_iterations": total_iterations,
                "avg_duration": avg_prompt_duration,
            },
            "api_calls": {
                "total": api_total_count,
                "success": api_success_count,
                "failed": api_total_count - api_success_count,
                "success_rate": round(api_success_rate, 2),
                "avg_duration": round(avg_api_duration, 3),
                "model": model_name,
                "tokens": {
                    "prompt": total_prompt_tokens,
                    "completion": total_completion_tokens,
                    "total": total_tokens,
                    "cache_read": total_cache_read_tokens,
                    "cache_creation": total_cache_creation_tokens,
                },
                "cost": cost_info,
            },
            "tool_calls": {
                "total": tool_total_count,
                "success": tool_success_count,
                "failed": tool_total_count - tool_success_count,
                "success_rate": round(tool_success_rate, 2),
                "avg_duration": round(avg_tool_duration, 3),
                "breakdown_by_tool": tool_call_breakdown,
            },
            "detailed_prompts": [
                {
                    "prompt_id": p.prompt_id,
                    "user_query": p.user_query,
                    "duration": p.duration,
                    "iterations": p.iterations,
                    "api_calls_count": len(p.api_calls),
                    "tool_calls_count": len(p.tool_calls),
                }
                for p in self.session_metric.prompts
            ],
        }
        
        return summary
    
    def get_raw_metrics(self) -> Dict[str, Any]:
        """
        Get raw metrics data (all collected metrics)
        
        Returns:
            Complete metrics data as nested dictionaries
        """
        return asdict(self.session_metric)

