"""
OpenAI Compatible LLM Implementation

Supports OpenAI API and compatible providers (DeepSeek, etc.)
"""

import logging
from typing import List, Optional, Union, Iterator, Dict, Any

from openai import OpenAI, APIError as OpenAIAPIError, APITimeoutError, RateLimitError as OpenAIRateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

from codefuse.llm.base import (
    BaseLLM,
    Message,
    Tool,
    ToolCall,
    LLMResponse,
    LLMChunk,
    TokenUsage,
)
from codefuse.llm.retry import retry_on_failure
from codefuse.llm.exceptions import (
    TimeoutError,
    RateLimitError,
    APIError,
    AuthenticationError,
    ContextLengthExceededError,
    InvalidRequestError,
    ModelNotFoundError,
)

from codefuse.observability import mainLogger


class OpenAICompatibleLLM(BaseLLM):
    """
    OpenAI Compatible LLM implementation
    
    Supports:
    - OpenAI API
    - DeepSeek API
    - Any other OpenAI-compatible API
    """
    
    def __init__(self, **kwargs):
        """Initialize OpenAI compatible client"""
        super().__init__(**kwargs)
        
        # Create OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        mainLogger.info(
            f"Initialized OpenAI Compatible LLM: model={self.model}, "
            f"base_url={self.base_url or 'default'}"
        )
    
    @property
    def supports_prompt_caching(self) -> bool:
        """OpenAI and compatible providers handle caching automatically"""
        return True
    
    @retry_on_failure(max_retries=3)
    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        parallel_tool_calls: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Union[LLMResponse, Iterator[LLMChunk]]:
        """
        Generate completion from messages
        
        Args:
            messages: List of conversation messages
            tools: Optional tools for function calling
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            parallel_tool_calls: Override instance-level parallel_tool_calls setting
            top_k: Top-k sampling parameter (override instance-level setting)
            top_p: Nucleus sampling parameter (override instance-level setting)
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse or Iterator[LLMChunk]
        """
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)

        
        # Build request parameters
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": stream,
        }
        
        # Add max_tokens if specified
        if max_tokens is not None or self.max_tokens is not None:
            params["max_tokens"] = max_tokens if max_tokens is not None else self.max_tokens
        
        # Add top_p if specified (standard OpenAI parameter)
        if top_p is not None or self.top_p is not None:
            params["top_p"] = top_p if top_p is not None else self.top_p
        
        # Add tools if provided
        if tools:
            params["tools"] = [self._convert_tool(t) for t in tools]
            params["tool_choice"] = "auto"
            # Use override if provided, otherwise use instance setting
            params["parallel_tool_calls"] = parallel_tool_calls if parallel_tool_calls is not None else self.parallel_tool_calls
        else:
            params["parallel_tool_calls"] = False
        
        # Build extra_body for custom parameters
        extra_body: Dict[str, Any] = {}
        
        # Add thinking mode via chat_template_kwargs
        if self.enable_thinking:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": True
            }
            mainLogger.debug("Thinking mode enabled via chat_template_kwargs")
        else:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": False
            }
            mainLogger.debug("Thinking mode disabled via chat_template_kwargs")
        
        # Add top_k if specified (custom parameter via extra_body)
        final_top_k = top_k if top_k is not None else self.top_k
        if final_top_k is not None:
            extra_body["top_k"] = final_top_k
            mainLogger.debug(f"Top-k sampling enabled: {final_top_k}")
        
        params["extra_body"] = extra_body

        
        # Add any extra parameters
        params.update(kwargs)
        
        mainLogger.debug(f"Calling LLM with {len(openai_messages)} messages, stream={stream}")

        # import json
        # with open("./openai_messages.json", "w") as f:
        #     json.dump(params, f)
        #     input()
        
        try:
            if stream:
                return self._handle_stream(params)
            else:
                return self._handle_completion(params)
        except Exception as e:
            # Convert to our custom exception types
            raise self._convert_exception(e)
    
    def _handle_completion(self, params: Dict[str, Any]) -> LLMResponse:
        """Handle non-streaming completion"""
        response: ChatCompletion = self.client.chat.completions.create(**params)
        
        choice = response.choices[0]
        message: ChatCompletionMessage = choice.message
        
        # Extract tool calls
        tool_calls: List[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function={
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                ))
        
        # Extract token usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                # Some providers may include cache-related tokens
                cache_creation_input_tokens=getattr(response.usage, 'cache_creation_input_tokens', None),
                cache_read_input_tokens=getattr(response.usage, 'cache_read_input_tokens', None),
            )
            mainLogger.info(f"Completion finished: {usage}")
        
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            raw_response=response.model_dump()
        )
    
    def _handle_stream(self, params: Dict[str, Any]) -> Iterator[LLMChunk]:
        """Handle streaming completion"""
        stream = self.client.chat.completions.create(**params)
        
        # Accumulate tool calls across chunks
        accumulated_tool_calls: Dict[int, Dict[str, str]] = {}
        finish_reason = None
        final_usage = None
        
        for chunk in stream:
            # Check for usage in chunks without choices (final usage chunk from OpenAI)
            # Some providers send a final chunk with empty choices but real usage data
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    # Only update if we get non-zero usage (real usage data)
                    if chunk.usage.total_tokens > 0:
                        final_usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                            cache_creation_input_tokens=getattr(chunk.usage, 'cache_creation_input_tokens', None),
                            cache_read_input_tokens=getattr(chunk.usage, 'cache_read_input_tokens', None),
                        )
                        mainLogger.debug(f"Received final usage chunk: {final_usage}")
                continue
            
            choice = chunk.choices[0]
            delta = choice.delta
 
            # Handle content delta
            if hasattr(delta, 'content') and delta.content:
                yield LLMChunk(
                    type="content",
                    delta=delta.content
                )
            
            # Handle tool calls
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = max(tc_delta.index, choice.index) 
                    
                    # Initialize tool call if first chunk for this index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "type": tc_delta.type or "function",
                            "name": "",
                            "arguments": ""
                        }
                    
                    # Update accumulated data
                    if tc_delta.id:
                        accumulated_tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.type:
                        accumulated_tool_calls[idx]["type"] = tc_delta.type
                    if tc_delta.function:
                        if tc_delta.function.name:
                            accumulated_tool_calls[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            accumulated_tool_calls[idx]["arguments"] += tc_delta.function.arguments
            
            # Record finish reason when encountered, but don't yield done yet
            # (we may receive a final usage chunk after this)
            if hasattr(choice, 'finish_reason') and choice.finish_reason:
                finish_reason = choice.finish_reason
                mainLogger.debug(f"Received finish_reason: {finish_reason}")
                
                # Some providers include usage in the finish chunk
                # Only use it if final_usage hasn't been set yet
                if not final_usage and hasattr(chunk, 'usage') and chunk.usage:
                    if chunk.usage.total_tokens > 0:
                        final_usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                            cache_creation_input_tokens=getattr(chunk.usage, 'cache_creation_input_tokens', None),
                            cache_read_input_tokens=getattr(chunk.usage, 'cache_read_input_tokens', None),
                        )
        
        # After stream ends, yield accumulated tool calls
        for tc_data in accumulated_tool_calls.values():
            yield LLMChunk(
                type="tool_call",
                tool_call=ToolCall(
                    id=tc_data["id"],
                    type=tc_data["type"],
                    function={
                        "name": tc_data["name"],
                        "arguments": tc_data["arguments"]
                    }
                )
            )
        
        # Log final usage
        if final_usage:
            mainLogger.info(f"Streaming finished: {final_usage}")
        else:
            mainLogger.warning("Streaming finished without usage information")
        
        # Yield final done chunk
        yield LLMChunk(
            type="done",
            usage=final_usage,
            finish_reason=finish_reason or "stop"
        )
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal Message format to OpenAI format"""
        result = []
        for msg in messages:
            openai_msg: Dict[str, Any] = {"role": msg.role.value}
            
            # Handle content
            if isinstance(msg.content, str):
                openai_msg["content"] = msg.content
            else:
                # List of content blocks
                openai_msg["content"] = [
                    {k: v for k, v in block.__dict__.items() if v is not None}
                    for block in msg.content
                ]
            
            # Handle optional fields
            if msg.name:
                openai_msg["name"] = msg.name
            
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": tc.function
                    }
                    for tc in msg.tool_calls
                ]
            
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            
            result.append(openai_msg)
        
        return result
    
    def _convert_tool(self, tool: Tool) -> Dict[str, Any]:
        """Convert internal Tool format to OpenAI format"""
        return {
            "type": tool.type,
            "function": tool.function
        }
    
    def format_messages_for_logging(
        self, 
        messages: List[Message],
        tools: Optional[List[Tool]] = None
    ) -> Dict[str, Any]:
        """
        Format messages and tools in OpenAI API format for logging
        
        This uses the same conversion logic as the actual API calls,
        so logs show exactly what was sent to the API.
        """
        result = {
            "messages": self._convert_messages(messages)
        }
        
        if tools:
            result["tools"] = [self._convert_tool(tool) for tool in tools]
        
        return result
    
    def _convert_exception(self, e: Exception) -> Exception:
        """Convert OpenAI exceptions to our custom exception types"""
        error_str = str(e).lower()
        error_type = type(e).__name__
        
        mainLogger.debug(f"Converting exception: {error_type}: {e}")
        
        # Timeout errors
        if isinstance(e, APITimeoutError) or "timeout" in error_str:
            return TimeoutError(f"Request timeout: {e}", original_error=e)
        
        # Rate limit errors (429)
        if isinstance(e, OpenAIRateLimitError) or "429" in error_str or "rate limit" in error_str:
            # Try to extract retry_after from the exception
            retry_after = None
            if hasattr(e, 'response') and e.response:
                retry_after_header = e.response.headers.get('retry-after') or e.response.headers.get('Retry-After')
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except (ValueError, TypeError):
                        pass
            
            return RateLimitError(
                f"Rate limit exceeded: {e}",
                retry_after=retry_after,
                original_error=e
            )
        
        # Context length exceeded
        if "context" in error_str and ("length" in error_str or "token" in error_str or "maximum" in error_str):
            return ContextLengthExceededError(f"Context length exceeded: {e}")
        
        # Authentication errors (401, 403)
        if "401" in error_str or "403" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            return AuthenticationError(f"Authentication failed: {e}")
        
        # Invalid request (400)
        if "400" in error_str or "invalid" in error_str or "bad request" in error_str:
            return InvalidRequestError(f"Invalid request: {e}")
        
        # Model not found (404)
        if "404" in error_str or "not found" in error_str or "model" in error_str:
            return ModelNotFoundError(f"Model not found: {e}")
        
        # Generic API error
        if isinstance(e, OpenAIAPIError):
            status_code = getattr(e, 'status_code', None)
            return APIError(f"API error: {e}", status_code=status_code, original_error=e)
        
        # Return as-is if we don't have a specific mapping
        mainLogger.warning(f"Unmapped exception type: {error_type}")
        return APIError(f"Unexpected error: {e}", original_error=e)

