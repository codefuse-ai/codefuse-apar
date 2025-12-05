"""
Anthropic LLM Implementation with KV Cache Support

This implementation extends OpenAICompatibleLLM to add Anthropic-specific
prompt caching capabilities using cache_control markers.
"""

from typing import List, Optional, Dict, Any

from codefuse.llm.base import Message, MessageRole
from codefuse.llm.providers.openai_compatible import OpenAICompatibleLLM
from codefuse.observability import mainLogger


class AnthropicLLM(OpenAICompatibleLLM):
    """
    Anthropic Claude LLM implementation with KV cache support
    
    This class extends OpenAICompatibleLLM and adds Anthropic-specific
    prompt caching by marking the last Tool message with cache_control.
    
    Caching Strategy:
    - If messages end with USER: No cache marker (new request, short context)
    - If messages end with TOOL: Add cache_control to last Tool message
    
    This allows caching of accumulated context during agent loops while
    keeping fresh user queries uncached.
    """
    
    def __init__(self, session_id: Optional[str] = None, **kwargs):
        """
        Initialize Anthropic client with OpenAI-compatible SDK
        
        Args:
            session_id: Session ID for x-idealab-session-id header (ensures requests hit same instance)
            **kwargs: Other parameters passed to OpenAICompatibleLLM
        """
        # Set default base_url for Anthropic if not provided
        # But keep user-provided base_url (for internal proxy services)
        if 'base_url' not in kwargs or kwargs['base_url'] is None:
            kwargs['base_url'] = "https://api.anthropic.com/v1"
        
        # Store session_id before calling parent __init__
        self._session_id = session_id
        
        super().__init__(**kwargs)
        
        # Recreate client with custom header if session_id is provided
        if session_id:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                default_headers={
                    'x-idealab-session-id': session_id
                }
            )
            mainLogger.info(
                f"Initialized Anthropic LLM with KV cache support: model={self.model}, "
                f"base_url={self.base_url}, session_id={session_id}"
            )
        else:
            mainLogger.info(
                f"Initialized Anthropic LLM with KV cache support: model={self.model}, base_url={self.base_url}"
            )
    
    @property
    def supports_prompt_caching(self) -> bool:
        """Anthropic has native prompt caching support"""
        return True
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal Message format to Anthropic format with cache control
        
        This method extends the parent's _convert_messages to add cache_control
        markers on the last Tool message (if messages end with TOOL role).
        
        Args:
            messages: List of internal Message objects
            
        Returns:
            List of message dictionaries in Anthropic API format
        """
        # First convert using parent's logic
        openai_messages = super()._convert_messages(messages)
        
        # Check if we should add cache control
        if not messages or len(messages) == 0:
            return openai_messages
        
        last_message = messages[-1]
        
        # Only add cache control if last message is TOOL
        if last_message.role != MessageRole.TOOL:
            mainLogger.debug(
                f"No cache control added: last message role is {last_message.role.value}"
            )
            return openai_messages
        
        # Add cache control to the last message (which is a Tool message)
        last_msg_dict = openai_messages[-1]
        
        # Convert content to array format with cache_control
        content = last_msg_dict.get("content", "")
        
        if isinstance(content, str):
            # Convert string content to content block array with cache_control
            # last_msg_dict["content"] = [
            #     {
            #         "type": "text",
            #         "text": content,
            #         "cache_control": {"type": "ephemeral"}
            #     }
            # ]
            last_msg_dict["cache_control"] = {"type": "ephemeral"}
            mainLogger.debug(
                "Added cache_control to last Tool message",
                tool_call_id=last_msg_dict.get("tool_call_id")
            )
        elif isinstance(content, list):
            # Content is already an array, add cache_control to last block
            if len(content) > 0:
                content[-1]["cache_control"] = {"type": "ephemeral"}
                mainLogger.debug(
                    "Added cache_control to last content block of Tool message",
                    tool_call_id=last_msg_dict.get("tool_call_id")
                )
        
        return openai_messages

