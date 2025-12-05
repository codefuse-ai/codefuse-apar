"""
Context Engine - Unified context and message management

This module provides context management for agent execution:
- Maintain conversation messages for LLM
- Store environment and tool definitions
- Build system prompts
- Provide context for each LLM call
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING, Dict, Any, Union
from uuid import uuid4

from codefuse.llm.base import Message, MessageRole, LLMResponse, ToolCall, Tool as LLMTool, ContentBlock
from codefuse.core.environment import EnvironmentInfo
from codefuse.observability import mainLogger
from codefuse.core.agent_config import AgentProfile

if TYPE_CHECKING:
    from codefuse.tools.registry import ToolRegistry
    from codefuse.observability.trajectory import TrajectoryWriter
    from codefuse.observability.llm_messages import LLMMessagesWriter
    from codefuse.llm.base import BaseLLM


class ContextEngine:
    """
    Context and message management engine
    
    This class manages runtime context for agent execution:
    - Maintain messages for LLM (user, assistant, tool)
    - Store environment and tool definitions
    - Provide context for each LLM call
    """
    
    def __init__(
        self,
        environment: EnvironmentInfo,
        tool_registry: "ToolRegistry",
        agent_profile: AgentProfile,
        max_tokens: int = 100000,
        session_id: Optional[str] = None,
        workspace: Optional[str] = None,
        trajectory_writer: Optional["TrajectoryWriter"] = None,
        llm_messages_writer: Optional["LLMMessagesWriter"] = None,
        conversation_history: Optional[List[Message]] = None,
        available_tools: Optional[List[str]] = None,
    ):
        """
        Initialize context engine
        
        Args:
            environment: Environment information
            tool_registry: Tool registry containing all available tools
            agent_profile: Agent profile with system prompt
            max_tokens: Maximum context length (for future truncation)
            session_id: Unique session ID (generated if not provided)
            workspace: Working directory path
            trajectory_writer: Optional trajectory writer for event recording
            llm_messages_writer: Optional LLM messages writer for snapshots
            conversation_history: Previous messages in the conversation (optional)
            available_tools: List of tool names to make available (None = all)
        """
        self.max_tokens = max_tokens
        
        # Session metadata
        self.session_id = session_id or self._generate_session_id()
        self.workspace = workspace or Path.cwd().as_posix()
        self.created_at = datetime.now().isoformat()
        
        # Prompt tracking (each user query gets a unique prompt_id within the session)
        self._prompt_counter = 0
        self._current_prompt_id: Optional[str] = None
        
        # Runtime state
        self._messages: List[Message] = []
        self._environment = environment
        self._agent_system_prompt = agent_profile.system_prompt
        
        # Tool management
        self._tool_registry = tool_registry
        self._available_tools = available_tools
        
        # Observability writers
        self._trajectory_writer = trajectory_writer
        self._llm_messages_writer = llm_messages_writer
        
        # Track current iteration (for trajectory events)
        self._current_iteration = 0
        
        # Track final response (for session summary)
        self._final_response: Optional[str] = None
        
        # Build and add system prompt
        self._system_prompt = self._build_system_prompt()
        self._messages.append(Message(role=MessageRole.SYSTEM, content=self._system_prompt))
        
        # Add conversation history if provided
        if conversation_history:
            self._messages.extend(conversation_history)
        
        mainLogger.info(
            "ContextEngine initialized",
            session_id=self.session_id,
            max_tokens=max_tokens,
            tool_count=len(tool_registry.list_tool_names()),
        )
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}_{uuid4().hex[:8]}"
    
    @property
    def prompt_id(self) -> Optional[str]:
        """
        Get current prompt ID
        
        Returns:
            Current prompt ID, or None if no user message has been added yet
        """
        return self._current_prompt_id
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt including environment info and tool descriptions
        
        Returns:
            Complete system prompt string
        """
        sections = []
        
        # 1. Agent-specific prompt (if provided)
        if self._agent_system_prompt:
            sections.append(self._agent_system_prompt)
        
        # 2. Environment information
        if self._environment:
            sections.append(self._environment.to_context_string())
        
        return "\n\n".join(sections)
    
    def add_user_message(self, content: Union[str, List[ContentBlock]]):
        """
        Add a user message to the conversation
        
        This also generates a new prompt_id for tracking this query.
        
        Args:
            content: User message content (text string or list of content blocks for multimodal)
        """
        # Generate new prompt_id for this user query
        self._prompt_counter += 1
        self._current_prompt_id = f"prompt_{self._prompt_counter:03d}"
        
        # Reset iteration counter for new prompt
        self._current_iteration = 0
        
        self._messages.append(Message(role=MessageRole.USER, content=content))
        mainLogger.debug(
            "Added user message to context",
            session_id=self.session_id,
            prompt_id=self._current_prompt_id,
        )
        
        # Write to trajectory
        if self._trajectory_writer:
            # Serialize content for JSON (handle both str and List[ContentBlock])
            if isinstance(content, str):
                serialized_content = content
            else:
                # Convert ContentBlock objects to dictionaries
                serialized_content = [
                    {k: v for k, v in block.__dict__.items() if v is not None}
                    for block in content
                ]
            
            self._trajectory_writer.write({
                "event_type": "user_message",
                "session_id": self.session_id,
                "prompt_id": self._current_prompt_id,
                "role": "user",
                "content": serialized_content,
            })
    
    def add_assistant_message(self, response: LLMResponse, iteration: Optional[int] = None):
        """
        Add an assistant message from LLM response
        
        Args:
            response: LLM response object
            iteration: Current iteration number (auto-incremented if not provided)
        """
        # Update iteration
        if iteration is not None:
            self._current_iteration = iteration
        else:
            self._current_iteration += 1
        
        # Add assistant message to context
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            tool_calls=[
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=tc.function,
                )
                for tc in response.tool_calls
            ] if response.tool_calls else None,
        )
        self._messages.append(assistant_message)
        mainLogger.debug(
            "Added assistant message to context",
            session_id=self.session_id,
            prompt_id=self._current_prompt_id,
        )
        
        # Track final response (if no tool calls)
        if not response.tool_calls or len(response.tool_calls) == 0:
            self._final_response = response.content
        
        # Write to trajectory
        if self._trajectory_writer:
            event_data: Dict[str, Any] = {
                "event_type": "assistant_response",
                "session_id": self.session_id,
                "prompt_id": self._current_prompt_id,
                "iteration": self._current_iteration,
                "role": "assistant",
                "content": response.content,
            }
            
            # Add tool_calls if present
            if response.tool_calls:
                event_data["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": tc.function,  # function is already a dict
                    }
                    for tc in response.tool_calls
                ]
            
            # Add extra_data if available
            extra_data: Dict[str, Any] = {}
            if response.model:
                extra_data["model"] = response.model
            if response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage.cache_creation_input_tokens:
                    token_usage["cache_creation_tokens"] = response.usage.cache_creation_input_tokens
                if response.usage.cache_read_input_tokens:
                    token_usage["cache_read_tokens"] = response.usage.cache_read_input_tokens
                extra_data["token_usage"] = token_usage
            
            if extra_data:
                event_data["extra_data"] = extra_data
            
            self._trajectory_writer.write(event_data)
    
    def add_tool_result(
        self,
        tool_call_id: str,
        result: str,
        tool_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        success: bool = True,
        duration: Optional[float] = None,
    ):
        """
        Add a tool execution result to the conversation
        
        Args:
            tool_call_id: ID of the tool call
            result: Tool execution result
            tool_name: Name of the tool (for trajectory logging)
            arguments: Tool arguments (for trajectory logging)
            success: Whether the tool execution succeeded
            duration: Execution duration in seconds (for trajectory logging)
        """
        tool_message = Message(
            role=MessageRole.TOOL,
            content=result,
            tool_call_id=tool_call_id,
        )
        self._messages.append(tool_message)
        mainLogger.debug(
            "Added tool result to context",
            tool_call_id=tool_call_id,
            session_id=self.session_id,
            prompt_id=self._current_prompt_id,
        )
        
        # Write to trajectory
        if self._trajectory_writer:
            event_data: Dict[str, Any] = {
                "event_type": "tool_result",
                "session_id": self.session_id,
                "prompt_id": self._current_prompt_id,
                "iteration": self._current_iteration,
                "tool_call_id": tool_call_id,
                "role": "tool",
                "content": result,
                "tool_success": success,
            }
            
            if tool_name:
                event_data["tool_name"] = tool_name
            if arguments:
                event_data["arguments"] = arguments
            
            # Add extra_data if available
            if duration is not None:
                event_data["extra_data"] = {"duration": duration}
            
            self._trajectory_writer.write(event_data)
    
    def sanitize_invalid_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        error_message: str,
    ):
        """
        Handle invalid tool call by converting tool_calls to text content
        and adding a user message to prompt the model to fix the error.
        
        This method:
        1. Finds the assistant message containing the specific tool_call_id
        2. Converts tool_calls to text format (with <Invalid JSON format> for arguments)
        3. Clears the tool_calls field to avoid downstream validation errors
        4. Adds a user message instructing the model to fix the JSON format
        
        Args:
            tool_call_id: ID of the invalid tool call
            tool_name: Name of the tool that was called
            error_message: Error message describing the JSON parsing failure
        """
        import json
        
        # Find assistant message with target tool_call_id (search from end)
        target_idx = None
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id == tool_call_id:
                        target_idx = i
                        break
                if target_idx is not None:
                    break
        
        if target_idx is None:
            mainLogger.warning(
                "Could not find assistant message to sanitize",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                session_id=self.session_id,
            )
            return
        
        assistant_msg = self._messages[target_idx]
        mainLogger.info(
            "Start Sanitizing invalid tool call",
            session_id=self.session_id,
            target_index=target_idx,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
        
        # Convert tool_calls to text format
        tool_calls_text = "\n\nTool calls attempted:\n" + "\n".join(
            f"- Tool: {tc.function.get('name', 'unknown')}\n"
            f"  ID: {tc.id}\n"
            f"  Arguments: {'<Invalid JSON format>' if tc.id == tool_call_id else self._format_tool_args(tc)}"
            for tc in assistant_msg.tool_calls
        )
        
        # Update message: append tool_calls text and clear tool_calls field
        self._messages[target_idx] = Message(
            role=MessageRole.ASSISTANT,
            content=(assistant_msg.content or "") + tool_calls_text,
            tool_calls=None,
        )

        mainLogger.info("Sanitization completed", session_id=self.session_id, target_index=target_idx, tool_call_id=tool_call_id, tool_name=tool_name)
        print(self._trajectory_writer)
        
        # Record sanitization event
        if self._trajectory_writer:
            self._trajectory_writer.write({
                "event_type": "tool_call_sanitized",
                "session_id": self.session_id,
                "prompt_id": self._current_prompt_id,
                "assistant_mesages_before_sanitization": assistant_msg.to_dict(),
                "assistant_mesages_after_sanitization": self._messages[target_idx].content,
            })
        
        # Add correction instruction
        self.add_user_message(
            f"Error: The previous tool call had invalid JSON format in the arguments. "
            f"Tool '{tool_name}' (ID: {tool_call_id}) failed with error: {error_message}\n\n"
            f"Please retry the tool call with VALID JSON format. Ensure that:\n"
            f"- All strings are properly quoted\n"
            f"- All special characters are properly escaped\n"
            f"- The JSON structure is complete and well-formed\n"
            f"- All brackets and braces are properly matched\n\n"
            f"Continue with the task using correct JSON format."
        )
        
        mainLogger.debug("Sanitization completed", tool_call_id=tool_call_id, session_id=self.session_id)
    
    def _format_tool_args(self, tool_call: ToolCall) -> str:
        """Helper to format tool call arguments, handling invalid JSON gracefully."""
        try:
            args = json.loads(tool_call.function.get('arguments', '{}'))
            return json.dumps(args, indent=2)
        except json.JSONDecodeError:
            return "<Invalid JSON format>"
    
    def get_messages_for_llm(self) -> List[Message]:
        """
        Get current messages to send to LLM
        
        Returns:
            List of messages for LLM
        """
        return self._messages.copy()
    
    def get_tools_for_llm(self) -> List[LLMTool]:
        """
        Get current tools to send to LLM
        
        Returns:
            List of tools in LLM-compatible format
        """
        return self._tool_registry.get_tools_for_llm(self._available_tools)
    
    def write_session_start(self, agent_name: str, model: str, tools: List[str], temperature: Optional[float] = None):
        """
        Write session start event to trajectory
        
        Args:
            agent_name: Name of the agent
            model: LLM model name
            tools: List of available tool names
            temperature: Model temperature setting
        """
        if self._trajectory_writer:
            event_data = {
                "event_type": "session_start",
                "session_id": self.session_id,
                "agent": agent_name,
                "model": model,
                "tools": tools,
                "workdir": self.workspace,
            }
            if temperature is not None:
                event_data["temperature"] = temperature
            self._trajectory_writer.write(event_data)
    
    def write_session_summary(self, summary_data: Dict[str, Any]):
        """
        Write session summary to trajectory with enhancements
        
        Adds final_response and git_diff to the summary data.
        
        Args:
            summary_data: Summary data from MetricsCollector.generate_summary()
        """
        if self._trajectory_writer:
            # Add final response
            if self._final_response:
                summary_data["final_response"] = self._final_response
            
            # Add git diff information
            git_diff = EnvironmentInfo.get_git_diff_info(self.workspace)
            if git_diff:
                summary_data["git_diff"] = git_diff
            
            self._trajectory_writer.write_summary(summary_data)
    
    def write_llm_messages(self, llm_provider: "BaseLLM"):
        """
        Write current LLM messages snapshot
        
        This should be called after each assistant response to capture
        the latest conversation state.
        
        Args:
            llm_provider: LLM provider instance for formatting messages
        """
        if self._llm_messages_writer:
            try:
                messages = self.get_messages_for_llm()
                tools = self.get_tools_for_llm()
                formatted_data = llm_provider.format_messages_for_logging(messages, tools)
                formatted_data["session_id"] = self.session_id
                self._llm_messages_writer.write(formatted_data)
            except Exception as e:
                mainLogger.warning(
                    "Failed to write LLM messages",
                    error=str(e),
                    session_id=self.session_id,
                )
    
    def set_writers(
        self,
        trajectory_writer: Optional["TrajectoryWriter"] = None,
        llm_messages_writer: Optional["LLMMessagesWriter"] = None,
    ):
        """
        Set observability writers (can be called after initialization)
        
        Args:
            trajectory_writer: Trajectory writer instance
            llm_messages_writer: LLM messages writer instance
        """
        if trajectory_writer:
            self._trajectory_writer = trajectory_writer
        if llm_messages_writer:
            self._llm_messages_writer = llm_messages_writer
    
    @staticmethod
    def load_conversation_history(llm_messages_file: Path) -> List[Message]:
        """
        Load conversation history from llm_messages.json file
        
        This method reads a saved llm_messages.json file and extracts the
        conversation history (user and assistant messages, excluding system messages).
        
        Args:
            llm_messages_file: Path to the llm_messages.json file
            
        Returns:
            List of Message objects representing the conversation history
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        if not llm_messages_file.exists():
            raise FileNotFoundError(f"LLM messages file not found: {llm_messages_file}")
        
        try:
            with open(llm_messages_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract messages from the snapshot
            if 'messages' not in data:
                raise ValueError("Invalid llm_messages.json format: missing 'messages' key")
            
            messages_data = data['messages']
            conversation_history = []
            
            for msg_data in messages_data:
                role_str = msg_data.get('role')
                
                # Skip system messages - they will be rebuilt
                if role_str == 'system':
                    continue
                
                # Parse role
                try:
                    role = MessageRole(role_str)
                except ValueError:
                    mainLogger.warning(f"Unknown message role: {role_str}, skipping")
                    continue
                
                # Parse content
                content = msg_data.get('content', '')
                
                # Parse tool_calls if present
                tool_calls = None
                if 'tool_calls' in msg_data and msg_data['tool_calls']:
                    tool_calls = [
                        ToolCall(
                            id=tc.get('id', ''),
                            type=tc.get('type', 'function'),
                            function=tc.get('function', {}),
                        )
                        for tc in msg_data['tool_calls']
                    ]
                
                # Create message
                message = Message(
                    role=role,
                    content=content,
                    name=msg_data.get('name'),
                    tool_calls=tool_calls,
                    tool_call_id=msg_data.get('tool_call_id'),
                )
                conversation_history.append(message)
            
            mainLogger.info(
                "Loaded conversation history from llm_messages.json",
                message_count=len(conversation_history),
                file=str(llm_messages_file),
            )
            
            return conversation_history
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in llm_messages.json: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse llm_messages.json: {e}")
