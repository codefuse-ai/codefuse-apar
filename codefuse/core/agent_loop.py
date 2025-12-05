"""
Agent Loop - Main agent execution loop with tool calling
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Literal, Any, Callable, Union

from codefuse.llm.base import BaseLLM, Message, ContentBlock
from codefuse.tools.registry import ToolRegistry
from codefuse.core.context_engine import ContextEngine
from codefuse.core.tool_executor import ToolExecutor
from codefuse.observability import MetricsCollector, mainLogger


@dataclass
class AgentEvent:
    """
    Event emitted by the agent loop
    
    Events allow streaming updates to the caller about agent progress.
    """
    type: Literal[
        "llm_start",                  # LLM call started
        "llm_chunk",                  # Streaming chunk from LLM
        "llm_done",                   # LLM call completed
        "tool_confirmation_required", # Tool needs user confirmation
        "tool_start",                 # Tool execution started
        "tool_done",                  # Tool execution completed
        "agent_done",                 # Agent loop completed
        "agent_think",                # Agent thinking/reasoning message
        "error",                      # Error occurred
    ]
    data: Any  # Type depends on event type


class AgentLoop:
    """
    Main agent execution loop
    
    Handles:
    - LLM interaction
    - Tool calling and execution
    - User confirmation for dangerous tools
    - Streaming output
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        context_engine: ContextEngine,
        max_iterations: int = 10,
        yolo_mode: bool = False,
        confirmation_callback: Optional[Callable[[str, str, dict], bool]] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        remote_tool_enabled: bool = False,
        remote_tool_url: Optional[str] = None,
        remote_tool_instance_id: Optional[str] = None,
        remote_tool_timeout: int = 60,
    ):
        """
        Initialize agent loop
        
        Args:
            llm: LLM instance
            tool_registry: Tool registry
            context_engine: Context engine
            max_iterations: Maximum number of iterations
            yolo_mode: If True, auto-confirm all tool executions
            confirmation_callback: Callback for tool confirmation
                                  (tool_name, tool_id, arguments) -> bool
            metrics_collector: Optional metrics collector for tracking performance
            remote_tool_enabled: If True, use remote tool execution
            remote_tool_url: URL of remote tool service
            remote_tool_instance_id: Instance ID for remote execution
            remote_tool_timeout: Timeout for remote tool calls in seconds
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.context_engine = context_engine
        self.max_iterations = max_iterations
        self.yolo_mode = yolo_mode
        self.confirmation_callback = confirmation_callback
        self.metrics_collector = metrics_collector
        
        # Initialize tool executor
        self.tool_executor = ToolExecutor(
            tool_registry=tool_registry,
            context_engine=context_engine,
            yolo_mode=yolo_mode,
            confirmation_callback=confirmation_callback,
            metrics_collector=metrics_collector,
            remote_enabled=remote_tool_enabled,
            remote_url=remote_tool_url,
            remote_instance_id=remote_tool_instance_id,
            remote_timeout=remote_tool_timeout,
        )
        
        mainLogger.info(
            "AgentLoop initialized",
            session_id=self.context_engine.session_id,
            max_iterations=max_iterations,
            yolo_mode=yolo_mode,
            temperature=llm.temperature if hasattr(llm, 'temperature') else None,
        )
    
    @property
    def session_id(self) -> str:
        """Get session ID from context engine"""
        return self.context_engine.session_id
    
    def _build_llm_done_event_data(self, llm_response) -> dict:
        """
        Build llm_done event data from LLM response
        
        Args:
            llm_response: LLM response object
            
        Returns:
            Dictionary with event data
        """
        return {
            "content": llm_response.content,
            "has_tool_calls": llm_response.has_tool_calls,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function,
                }
                for tc in llm_response.tool_calls
            ] if llm_response.tool_calls else [],
            "session_id": self.session_id,
        }
    
    def _record_llm_metrics(self, api_tracker, llm_response) -> None:
        """
        Record LLM call metrics
        
        Args:
            api_tracker: API tracker from metrics collector
            llm_response: LLM response object
        """
        if not api_tracker:
            return
        
        if llm_response.usage:
            api_tracker.set_tokens(
                prompt_tokens=llm_response.usage.prompt_tokens,
                completion_tokens=llm_response.usage.completion_tokens,
                total_tokens=llm_response.usage.total_tokens,
                cache_creation_tokens=llm_response.usage.cache_creation_input_tokens,
                cache_read_tokens=llm_response.usage.cache_read_input_tokens,
            )
        
        if llm_response.model:
            api_tracker.set_model(llm_response.model)
        
        if llm_response.finish_reason:
            api_tracker.set_finish_reason(llm_response.finish_reason)
    
    @staticmethod
    def _summarize_query(user_query: Union[str, List[ContentBlock]]) -> str:
        """
        Generate a short summary of user query for logging
        
        Args:
            user_query: User query (text or multimodal content blocks)
            
        Returns:
            Query summary string
        """
        if isinstance(user_query, str):
            return user_query[:100] if len(user_query) > 100 else user_query
        else:
            # Multimodal content
            text_parts = [
                block.text for block in user_query 
                if block.type == "text" and block.text
            ]
            image_count = sum(
                1 for block in user_query 
                if block.type == "image_url"
            )
            text_preview = text_parts[0][:50] if text_parts else ""
            return f"{text_preview}... [{image_count} image(s)]"
    
    def _call_llm(self, messages: List[Message], tools: List, stream: bool):
        """
        Unified LLM call interface with metrics tracking
        
        Args:
            messages: Messages to send
            tools: Tools available
            stream: Whether to stream responses
            
        Returns:
            If stream=False: LLMResponse
            If stream=True: Generator
        """
        if not stream:
            # Non-streaming: call directly and record metrics
            if self.metrics_collector:
                with self.metrics_collector.track_api_call() as api_tracker:
                    llm_response = self.llm.generate(
                        messages=messages,
                        tools=tools,
                        stream=False,
                    )
                    self._record_llm_metrics(api_tracker, llm_response)
            else:
                llm_response = self.llm.generate(
                    messages=messages,
                    tools=tools,
                    stream=False,
                )
            return llm_response
        else:
            # Streaming: return generator
            return self.llm.generate(
                messages=messages,
                tools=tools,
                stream=True,
            )
    
    def run(
        self,
        user_query: Union[str, List[ContentBlock]],
        stream: bool = False,
    ) -> Iterator[AgentEvent]:
        """
        Run the agent loop
        
        Args:
            user_query: User's query (text string or list of content blocks for multimodal)
            stream: Whether to stream LLM responses
            
        Yields:
            AgentEvent objects representing agent progress
        """
        # 1. Add user message to context
        self.context_engine.add_user_message(user_query)
        
        # 2. Log the user query using helper method
        query_summary = self._summarize_query(user_query)
        mainLogger.info(
            "Starting user query",
            session_id=self.session_id,
            prompt_id=self.context_engine.prompt_id,
            query_summary=query_summary
        )
        
        # 3. Setup metrics tracking
        prompt_tracker_ctx = None
        prompt_tracker = None
        if self.metrics_collector:
            prompt_tracker_ctx = self.metrics_collector.track_prompt(user_query)
            prompt_tracker = prompt_tracker_ctx.__enter__()
        
        try:
            iteration = 0
            final_response = ""
            
            # 4. Main iteration loop
            while iteration < self.max_iterations:
                iteration += 1
                mainLogger.info(
                    "Agent iteration",
                    iteration=iteration,
                    max_iterations=self.max_iterations,
                    session_id=self.session_id,
                )
                
                # Track iteration in metrics
                if prompt_tracker:
                    prompt_tracker.increment_iteration()
                
                # Get current messages and tools from context engine
                messages = self.context_engine.get_messages_for_llm()
                tools = self.context_engine.get_tools_for_llm()
                
                try:
                    # Send LLM start event
                    yield AgentEvent(
                        type="llm_start",
                        data={
                            "iteration": iteration,
                            "session_id": self.session_id,
                        }
                    )
                    
                    # Call LLM (simplified branch using helper methods)
                    if stream:
                        llm_response = yield from self._handle_streaming_llm(
                            messages=messages,
                            tools=tools,
                            session_id=self.session_id,
                        )
                    else:
                        # Non-streaming: use unified call method and event construction
                        llm_response = self._call_llm(messages, tools, stream=False)
                        yield AgentEvent(
                            type="llm_done",
                            data=self._build_llm_done_event_data(llm_response)
                        )
                    
                    # Add assistant message to context
                    self.context_engine.add_assistant_message(llm_response, iteration=iteration)
                    self.context_engine.write_llm_messages(self.llm)
                    
                    # Check if we have tool calls
                    if not llm_response.has_tool_calls:
                        # No tool calls - we're done
                        final_response = llm_response.content
                        break
                    
                    # Execute tool calls
                    for tool_call in llm_response.tool_calls:
                        result_events = self.tool_executor.execute_tool_call(
                            tool_call, self.session_id
                        )
                        for tool_event in result_events:
                            yield AgentEvent(type=tool_event.type, data=tool_event.data)
                    
                except Exception as e:
                    mainLogger.error(
                        "Error in agent loop iteration",
                        iteration=iteration,
                        error=str(e),
                        session_id=self.session_id,
                        exc_info=True,
                    )
                    yield AgentEvent(
                        type="error",
                        data={"error": str(e), "iteration": iteration}
                    )
                    break
            
            # 5. Check if we hit max iterations
            if iteration >= self.max_iterations and not final_response:
                final_response = "Maximum iterations reached. The task may not be complete."
                mainLogger.warning(
                    "Agent loop reached maximum iterations",
                    iterations=iteration,
                    session_id=self.session_id,
                )
            
            # 6. Write final snapshot and send completion event
            self.context_engine.write_llm_messages(self.llm)
            
            yield AgentEvent(
                type="agent_done",
                data={
                    "final_response": final_response,
                    "iterations": iteration,
                    "session_id": self.session_id,
                }
            )
        
        finally:
            if prompt_tracker_ctx:
                prompt_tracker_ctx.__exit__(None, None, None)
    
    def _handle_streaming_llm(
        self,
        messages: List[Message],
        tools: List,
        session_id: str,
    ) -> Iterator[AgentEvent]:
        """
        Handle streaming LLM response
        
        Args:
            messages: Messages to send
            tools: Tools available
            session_id: Session ID
            
        Yields:
            AgentEvent objects for streaming chunks
            
        Returns:
            LLMResponse (via final return statement)
        """
        from codefuse.llm.base import LLMResponse
        
        content_parts = []
        tool_calls = []
        usage = None
        finish_reason = ""
        
        # Track API call if metrics collector is available
        if self.metrics_collector:
            api_tracker_ctx = self.metrics_collector.track_api_call()
            api_tracker = api_tracker_ctx.__enter__()
        else:
            api_tracker_ctx = None
            api_tracker = None
        
        try:
            # Get streaming generator using unified method
            stream = self._call_llm(messages, tools, stream=True)
            
            # Process chunks
            for chunk in stream:
                if chunk.type == "content":
                    content_parts.append(chunk.delta)
                    yield AgentEvent(type="llm_chunk", data={"delta": chunk.delta})
                elif chunk.type == "tool_call":
                    tool_calls.append(chunk.tool_call)
                elif chunk.type == "done":
                    usage = chunk.usage
                    finish_reason = chunk.finish_reason
            
            # Reconstruct LLMResponse
            response = LLMResponse(
                content="".join(content_parts),
                tool_calls=tool_calls,
                usage=usage,
                model=self.llm.model,
                finish_reason=finish_reason,
            )
            
            # Use unified metrics recording
            self._record_llm_metrics(api_tracker, response)
            
            # Use unified event construction
            yield AgentEvent(
                type="llm_done",
                data=self._build_llm_done_event_data(response)
            )
            
            return response
        
        finally:
            if api_tracker_ctx:
                api_tracker_ctx.__exit__(None, None, None)
