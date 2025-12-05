"""
LLM Factory - Create LLM instances based on provider
"""

from typing import Optional

from codefuse.llm.base import BaseLLM
from codefuse.llm.providers.openai_compatible import OpenAICompatibleLLM
from codefuse.observability import mainLogger


def create_llm(
    provider: str = "openai_compatible",
    model: str = "gpt-4o",
    api_key: str = "",
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: int = 60,
    parallel_tool_calls: bool = True,
    enable_thinking: bool = False,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> BaseLLM:
    """
    Factory function to create LLM instances
    
    Args:
        provider: LLM provider type
            - "openai_compatible": OpenAI API and compatible providers (default)
            - "anthropic": Anthropic Claude API
            - "gemini": Google Gemini API
        model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet", etc.)
        api_key: API key for authentication
        base_url: Base URL for API endpoint (for openai_compatible)
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        parallel_tool_calls: Enable parallel tool calls (default: True)
        enable_thinking: Enable thinking mode for models that support it (default: False)
        top_k: Top-k sampling parameter (default: None)
        top_p: Nucleus sampling parameter (0-1, default: None)
        session_id: Session ID for Anthropic provider (used for x-idealab-session-id header)
        **kwargs: Additional provider-specific parameters
        
    Returns:
        BaseLLM instance configured for the specified provider
        
    Raises:
        ValueError: If provider is not supported
        
    Examples:
        >>> # OpenAI
        >>> llm = create_llm(
        ...     provider="openai_compatible",
        ...     model="gpt-4o",
        ...     api_key="sk-..."
        ... )
        
        >>> # DeepSeek
        >>> llm = create_llm(
        ...     provider="openai_compatible",
        ...     model="deepseek-chat",
        ...     api_key="sk-...",
        ...     base_url="https://api.deepseek.com"
        ... )
        
        >>> # Anthropic (when implemented)
        >>> llm = create_llm(
        ...     provider="anthropic",
        ...     model="claude-3-5-sonnet-20241022",
        ...     api_key="sk-ant-..."
        ... )
    """
    provider = provider.lower().strip()
    
    mainLogger.info(f"Creating LLM: provider={provider}, model={model}")
    
    if provider == "anthropic":
        from codefuse.llm.providers.anthropic import AnthropicLLM
        return AnthropicLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            parallel_tool_calls=parallel_tool_calls,
            enable_thinking=enable_thinking,
            top_k=top_k,
            top_p=top_p,
            session_id=session_id,
            **kwargs
        )
    
    elif provider == "gemini":
        from codefuse.llm.providers.gemini import GeminiLLM
        return GeminiLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            parallel_tool_calls=parallel_tool_calls,
            enable_thinking=enable_thinking,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )
    
    elif provider in ("openai_compatible", "openai"):
        # Default to OpenAI Compatible for all unspecified providers
        return OpenAICompatibleLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            parallel_tool_calls=parallel_tool_calls,
            enable_thinking=enable_thinking,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )
    
    else:
        mainLogger.warning(
            f"Unknown provider '{provider}', defaulting to openai_compatible. "
            f"Supported providers: openai_compatible, anthropic, gemini"
        )
        return OpenAICompatibleLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            parallel_tool_calls=parallel_tool_calls,
            enable_thinking=enable_thinking,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )