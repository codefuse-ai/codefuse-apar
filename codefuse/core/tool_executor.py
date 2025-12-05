"""
Tool Executor - Handles tool call execution with confirmation and error handling
"""

import json
import time
from typing import Iterator, Optional, Callable, Any
from dataclasses import dataclass

from codefuse.llm.base import ToolCall
from codefuse.tools.registry import ToolRegistry
from codefuse.tools.base import ToolResult
from codefuse.core.context_engine import ContextEngine
from codefuse.observability import MetricsCollector, mainLogger


@dataclass
class ToolExecutionEvent:
    """
    Event emitted during tool execution
    
    Types:
        - tool_confirmation_required: Tool needs user confirmation
        - tool_start: Tool execution started
        - tool_done: Tool execution completed
    
    Data fields for tool_done:
        - result: Full result content for LLM
        - display: User-friendly display text for UI
    """
    type: str
    data: dict


class ToolExecutor:
    """
    Handles tool call execution with safety checks and user confirmation
    
    Responsibilities:
    - Validate tool existence
    - Parse tool arguments
    - Request user confirmation for dangerous operations
    - Execute tools with error handling
    - Record results in context engine
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        context_engine: ContextEngine,
        yolo_mode: bool = False,
        confirmation_callback: Optional[Callable[[str, str, dict], bool]] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        remote_enabled: bool = False,
        remote_url: Optional[str] = None,
        remote_instance_id: Optional[str] = None,
        remote_timeout: int = 60,
    ):
        """
        Initialize tool executor
        
        Args:
            tool_registry: Registry of available tools
            context_engine: Context engine for recording results
            yolo_mode: If True, auto-confirm all tool executions
            confirmation_callback: Callback for tool confirmation
                                  (tool_name, tool_id, arguments) -> bool
            metrics_collector: Optional metrics collector for tracking tool execution
            remote_enabled: If True, use remote tool execution
            remote_url: URL of remote tool service
            remote_instance_id: Instance ID for remote execution
            remote_timeout: Timeout for remote tool calls in seconds
        """
        self.tool_registry = tool_registry
        self.context_engine = context_engine
        self.yolo_mode = yolo_mode
        self.confirmation_callback = confirmation_callback
        self.metrics_collector = metrics_collector
        self.remote_enabled = remote_enabled
        self.remote_url = remote_url
        self.remote_instance_id = remote_instance_id
        self.remote_timeout = remote_timeout
        
        # Initialize remote tool executor if enabled
        self.remote_executor = None
        if self.remote_enabled and self.remote_url and self.remote_instance_id:
            from codefuse.core.remote_tool_executor import RemoteToolExecutor
            self.remote_executor = RemoteToolExecutor(
                url=self.remote_url,
                instance_id=self.remote_instance_id,
                timeout=self.remote_timeout,
            )
            mainLogger.info(
                "Remote tool execution enabled",
                url=self.remote_url,
                instance_id=self.remote_instance_id,
            )
        elif self.remote_enabled:
            mainLogger.warning(
                "Remote tool execution requested but not properly configured",
                remote_url=self.remote_url,
                remote_instance_id=self.remote_instance_id,
            )
    
    def execute_tool_call(
        self,
        tool_call: ToolCall,
        session_id: str,
    ) -> Iterator[ToolExecutionEvent]:
        """
        Execute a single tool call
        
        Args:
            tool_call: Tool call to execute
            session_id: Session ID
            
        Yields:
            ToolExecutionEvent objects for tool execution progress
        """
        tool_name = tool_call.function["name"]
        tool = self.tool_registry.get_tool(tool_name)
        
        # Step 1: Validate tool exists
        if tool is None:
            yield from self._handle_tool_not_found(
                tool_call.id, tool_name, session_id
            )
            return
        
        # Step 2: Parse arguments
        try:
            arguments = json.loads(tool_call.function["arguments"])
        except json.JSONDecodeError as e:
            yield from self._handle_invalid_arguments(
                tool_call.id, tool_name, e, session_id
            )
            return
        
        # Step 3: Check if confirmation is needed
        confirmed = True
        if tool.requires_confirmation and not self.yolo_mode:
            # Emit confirmation required event
            yield ToolExecutionEvent(
                type="tool_confirmation_required",
                data={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "session_id": session_id,
                }
            )
            
            # Get user confirmation
            confirmed = self._get_user_confirmation(tool_name, tool_call.id, arguments)
        
        # Step 4: Handle rejection
        if not confirmed:
            yield from self._handle_tool_rejection(
                tool_call.id, tool_name, arguments, session_id
            )
            return
        
        # Step 5: Execute tool
        yield from self._execute_and_record(
            tool, tool_call.id, tool_name, arguments, confirmed, session_id
        )
    
    def _handle_tool_not_found(
        self, tool_call_id: str, tool_name: str, session_id: str
    ) -> Iterator[ToolExecutionEvent]:
        """Handle case where tool is not found"""
        error_msg = f"Tool not found: {tool_name}"
        result = f"Error: {error_msg}"
        mainLogger.error("Tool not found", tool_name=tool_name, session_id=session_id)
        
        # Add error result to context
        self.context_engine.add_tool_result(tool_call_id, result)
        
        yield ToolExecutionEvent(
            type="tool_done",
            data={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result": result,
                "confirmed": False,
                "session_id": session_id,
            }
        )
    
    def _handle_invalid_arguments(
        self, tool_call_id: str, tool_name: str, error: Exception, session_id: str
    ) -> Iterator[ToolExecutionEvent]:
        """Handle case where tool arguments are invalid"""
        error_msg = f"Invalid tool arguments JSON: {error}"
        result = f"Error: {error_msg}"
        mainLogger.error(
            "Invalid tool arguments JSON",
            tool_name=tool_name,
            error=str(error),
            session_id=session_id,
        )
        
        # Sanitize the assistant message to avoid downstream 400 errors
        # This will:
        # 1. Convert tool_calls to text in the assistant message content
        # 2. Clear the tool_calls field to prevent VLLM/SGLang from validating invalid JSON
        # 3. Add a user message instructing the model to retry with valid JSON
        self.context_engine.sanitize_invalid_tool_call(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            error_message=str(error),
        )
        
        yield ToolExecutionEvent(
            type="tool_done",
            data={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result": result,
                "confirmed": False,
                "session_id": session_id,
            }
        )
    
    def _get_user_confirmation(self, tool_name: str, tool_call_id: str, arguments: dict) -> bool:
        """Get user confirmation for tool execution"""
        if self.confirmation_callback:
            return self.confirmation_callback(tool_name, tool_call_id, arguments)
        else:
            # No callback - default to rejecting
            mainLogger.warning(
                "Tool requires confirmation but no callback provided",
                tool_name=tool_name,
            )
            return False
    
    def _handle_tool_rejection(
        self, tool_call_id: str, tool_name: str, arguments: dict, session_id: str
    ) -> Iterator[ToolExecutionEvent]:
        """Handle case where user rejects tool execution"""
        result = f"Tool execution rejected by user: {tool_name}"
        mainLogger.info("Tool execution rejected by user", tool_name=tool_name, session_id=session_id)
        
        # Add rejection result to context (automatically writes to trajectory)
        self.context_engine.add_tool_result(
            tool_call_id=tool_call_id,
            result=result,
            tool_name=tool_name,
            arguments=arguments,
            success=False,
        )
        
        yield ToolExecutionEvent(
            type="tool_done",
            data={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result": result,
                "confirmed": False,
                "arguments": arguments,
                "session_id": session_id,
            }
        )
    
    def _execute_and_record(
        self, tool: Any, tool_call_id: str, tool_name: str, 
        arguments: dict, confirmed: bool, session_id: str
    ) -> Iterator[ToolExecutionEvent]:
        """Execute tool and record result"""
        # Emit tool start event
        yield ToolExecutionEvent(
            type="tool_start",
            data={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "session_id": session_id,
            }
        )
        
        # Track tool execution with metrics if available
        if self.metrics_collector:
            tool_tracker_ctx = self.metrics_collector.track_tool_call(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
            )
            tool_tracker = tool_tracker_ctx.__enter__()
        else:
            tool_tracker_ctx = None
            tool_tracker = None
        
        # Execute tool and track duration
        start_time = time.time()
        success = True
        
        try:
            # Check if remote execution is enabled and available
            if self.remote_executor:
                mainLogger.info(
                    "Using remote tool execution",
                    tool_name=tool_name,
                    session_id=session_id,
                )
                tool_result = self.remote_executor.execute(
                    tool_name=tool_name,
                    tool_args=arguments,
                    session_id=session_id,
                )
                
                # Check if remote execution indicated failure
                if "Error:" in tool_result.content or "❌" in tool_result.display:
                    success = False
            else:
                # Local execution
                mainLogger.info(
                    "Using local tool execution",
                    tool_name=tool_name,
                    session_id=session_id,
                )
                tool_result = tool.execute(**arguments)
                
                # Ensure result is ToolResult (backward compatibility)
                if isinstance(tool_result, str):
                    tool_result = ToolResult(content=tool_result)
            
            if success:
                mainLogger.info("Tool executed successfully", tool_name=tool_name, session_id=session_id)
                if tool_tracker:
                    tool_tracker.set_success(True)
            else:
                mainLogger.warning("Tool execution completed with errors", tool_name=tool_name, session_id=session_id)
                if tool_tracker:
                    tool_tracker.set_error("Tool execution reported failure")
                    
        except Exception as e:
            success = False
            error_msg = f"Tool execution error: {str(e)}"
            tool_result = ToolResult(content=error_msg, display=f"❌ Error: {str(e)}")
            mainLogger.error(
                "Tool execution failed",
                tool_name=tool_name,
                error=str(e),
                session_id=session_id,
                exc_info=True,
            )
            if tool_tracker:
                tool_tracker.set_error(str(e))
        finally:
            duration = time.time() - start_time
            if tool_tracker_ctx:
                tool_tracker_ctx.__exit__(None, None, None)
        
        # Add tool result to context (automatically writes to trajectory)
        self.context_engine.add_tool_result(
            tool_call_id=tool_call_id,
            result=tool_result.content,
            tool_name=tool_name,
            arguments=arguments,
            success=success,
            duration=duration,
        )
        
        # Emit tool done event (display for user)
        yield ToolExecutionEvent(
            type="tool_done",
            data={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result": tool_result.content,  # For logging/debugging
                "display": tool_result.display,  # For user display
                "confirmed": confirmed,
                "arguments": arguments,
                "session_id": session_id,
            }
        )

