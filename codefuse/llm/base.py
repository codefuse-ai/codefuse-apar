"""
LLM Base Classes and Data Structures
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Iterator, Literal, Any, Dict
from abc import ABC, abstractmethod
from enum import Enum


class MessageRole(str, Enum):
    """Message role in conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ContentBlock:
    """Content block for multimodal messages"""
    type: str  # "text", "image_url", etc.
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    """Tool call from the model"""
    id: str
    type: str  # "function"
    function: Dict[str, str]  # {"name": str, "arguments": str (JSON)}


@dataclass
class Message:
    """Unified message format"""
    role: MessageRole
    content: Union[str, List[ContentBlock]]
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result: Dict[str, Any] = {"role": self.role.value}
        
        if isinstance(self.content, str):
            result["content"] = self.content
        else:
            result["content"] = [
                {k: v for k, v in block.__dict__.items() if v is not None}
                for block in self.content
            ]
        
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        
        return result


@dataclass
class Tool:
    """Tool definition for function calling"""
    type: str = "function"
    function: Dict[str, Any] = field(default_factory=dict)  # {"name", "description", "parameters"}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "type": self.type,
            "function": self.function
        }


@dataclass
class TokenUsage:
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Optional cache-related tokens (for providers that support it)
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    
    def __str__(self) -> str:
        base = f"Tokens(prompt={self.prompt_tokens}, completion={self.completion_tokens}, total={self.total_tokens}"
        if self.cache_read_input_tokens:
            base += f", cache_read={self.cache_read_input_tokens}"
        if self.cache_creation_input_tokens:
            base += f", cache_creation={self.cache_creation_input_tokens}"
        return base + ")"


@dataclass
class LLMResponse:
    """Unified LLM response format"""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[TokenUsage] = None
    model: str = ""
    finish_reason: str = ""  # "stop", "tool_calls", "length", "content_filter", etc.
    raw_response: Optional[Dict[str, Any]] = None  # Original response for debugging
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls"""
        return len(self.tool_calls) > 0


@dataclass
class LLMChunk:
    """Streaming chunk from LLM"""
    type: Literal["content", "tool_call", "done"]
    delta: str = ""  # Content delta
    tool_call: Optional[ToolCall] = None
    usage: Optional[TokenUsage] = None  # Only present in final "done" chunk
    finish_reason: str = ""


class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        parallel_tool_calls: bool = True,
        enable_thinking: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize LLM instance
        
        Args:
            model: Model identifier
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            parallel_tool_calls: Enable parallel tool calls (default: True)
            enable_thinking: Enable thinking mode for models that support it (default: False)
            top_k: Top-k sampling parameter (default: None)
            top_p: Nucleus sampling parameter (0-1, default: None)
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.parallel_tool_calls = parallel_tool_calls
        self.enable_thinking = enable_thinking
        self.top_k = top_k
        self.top_p = top_p
        self.extra_params = kwargs
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Iterator[LLMChunk]]:
        """
        Generate completion from messages
        
        Args:
            messages: List of conversation messages
            tools: Optional list of tools/functions available to the model
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stream: If True, return iterator of chunks; if False, return complete response
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse for non-streaming, Iterator[LLMChunk] for streaming
            
        Raises:
            RetryableError: For errors that should be retried (timeout, rate limit)
            LLMError: For other errors
        """
        pass
    
    @property
    def supports_prompt_caching(self) -> bool:
        """Whether this provider supports prompt caching"""
        return False
    
    @property
    def supports_parallel_tools(self) -> bool:
        """Whether this provider supports parallel tool calls"""
        return True
    
    @property
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses"""
        return True
    
    def format_messages_for_logging(
        self, 
        messages: List[Message],
        tools: Optional[List[Tool]] = None
    ) -> Dict[str, Any]:
        """
        Format messages and tools for logging purposes
        
        This method converts internal Message/Tool format to the provider's
        API format for logging. Override in subclasses to customize.
        
        Args:
            messages: List of messages
            tools: Optional list of tools
            
        Returns:
            Dict with 'messages' and optionally 'tools' in API format
        """
        # Default implementation: use Message.to_dict()
        result = {
            "messages": [msg.to_dict() for msg in messages]
        }
        
        if tools:
            result["tools"] = [tool.to_dict() for tool in tools]
        
        return result
    
    def _prepare_cache_control(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None
    ) -> List[Message]:
        """
        Automatically add prompt caching markers if supported.
        Override in subclasses for provider-specific caching.
        
        Args:
            messages: Original messages
            tools: Optional tools
            
        Returns:
            Messages with cache control markers added
        """
        # Default implementation: no modification
        return messages

