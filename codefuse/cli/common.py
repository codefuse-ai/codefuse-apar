"""
Common utilities for CLI - shared between headless and interactive modes
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel

from codefuse import create_llm
from codefuse.config import Config
from codefuse.tools.registry import create_default_registry
from codefuse.core import (
    EnvironmentInfo,
    AgentProfileManager,
    ContextEngine,
    AgentLoop,
    ReadTracker,
    setup_logging,
    mainLogger,
    MetricsCollector,
)
from codefuse.observability.trajectory import TrajectoryWriter
from codefuse.observability.llm_messages import LLMMessagesWriter
from codefuse.observability.logging.utils import path_to_slug
from codefuse.llm.base import Message

console = Console()


def confirmation_callback(tool_name: str, tool_id: str, arguments: dict) -> bool:
    """
    Callback for tool execution confirmation
    
    Args:
        tool_name: Name of the tool
        tool_id: Tool call ID
        arguments: Tool arguments
        
    Returns:
        True if confirmed, False otherwise
    """
    console.print()
    console.print(Panel(
        f"[yellow]Tool:[/yellow] {tool_name}\n"
        f"[yellow]Arguments:[/yellow]\n{arguments}",
        title="⚠️  Tool Confirmation Required",
        border_style="yellow"
    ))
    
    return Confirm.ask("Confirm execution?", default=False)


def handle_list_agents(agent_manager: AgentProfileManager) -> None:
    """
    Handle --list-agents parameter
    
    Args:
        agent_manager: Agent profile manager instance
    """
    console.print("\n[bold]Available Agents:[/bold]\n")
    for agent_name in agent_manager.list_agents():
        info = agent_manager.get_agent_info(agent_name)
        console.print(Panel(info, border_style="blue"))


def check_and_load_existing_session(
    session_id: str,
    workspace_path: str,
    logs_dir: str,
) -> Optional[List[Message]]:
    """
    Check if a session already exists and load its conversation history
    
    Args:
        session_id: Session ID to check
        workspace_path: Workspace path
        logs_dir: Base logs directory
        
    Returns:
        List of Message objects if session exists, None otherwise
    """
    # Construct session directory path
    base_logs_dir = Path(logs_dir).expanduser()
    workspace_slug = path_to_slug(workspace_path)
    session_dir = base_logs_dir / workspace_slug / session_id
    llm_messages_file = session_dir / "llm_messages.json"
    
    # Check if session exists
    if not llm_messages_file.exists():
        return None
    
    # Try to load conversation history
    try:
        conversation_history = ContextEngine.load_conversation_history(llm_messages_file)
        console.print(f"[cyan]📂 Resuming session:[/cyan] {session_id}")
        console.print(f"[cyan]   Loaded {len(conversation_history)} messages from history[/cyan]\n")
        mainLogger.info(
            "Session resume: loaded conversation history",
            session_id=session_id,
            message_count=len(conversation_history),
        )
        return conversation_history
    except Exception as e:
        console.print(f"[yellow]⚠️  Warning: Failed to load session history:[/yellow] {e}")
        console.print(f"[yellow]   Starting fresh session with ID: {session_id}[/yellow]\n")
        mainLogger.warning(
            "Session resume failed, starting fresh",
            session_id=session_id,
            error=str(e),
        )
        return None


def _initialize_agent_profile(
    agent_manager: AgentProfileManager,
    agent_name: str,
    agent_profile: Optional[Any],
    llm_model: str,
) -> Tuple[Any, str]:
    """
    Initialize agent profile and determine model name
    
    Args:
        agent_manager: Agent profile manager instance
        agent_name: Name of the agent profile to use
        agent_profile: Optional pre-loaded AgentProfile (from --agent-file)
        llm_model: Model name from config
        
    Returns:
        Tuple of (agent_profile, model_name)
    """
    # Load agent profile (use provided one or load from manager)
    if agent_profile is None:
        agent_profile = agent_manager.get_agent(agent_name)
        if not agent_profile:
            console.print(f"[red]Error:[/red] Agent profile '{agent_name}' not found")
            console.print(f"Available agents: {', '.join(agent_manager.list_agents())}")
            raise ValueError(f"Agent profile not found: {agent_name}")
    
    # Determine model to use
    model_name = agent_profile.get_model_name(llm_model)
    
    return agent_profile, model_name


def _generate_session_id() -> str:
    """Generate a unique session ID"""
    from datetime import datetime
    from uuid import uuid4
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"session_{timestamp}_{uuid4().hex[:8]}"


def _setup_observability(
    session_id: str,
    cfg: Config,
    verbose: bool,
) -> Tuple[Path, MetricsCollector, "TrajectoryWriter", "LLMMessagesWriter"]:
    """
    Setup all observability components (logging, trajectory, metrics)
    
    Args:
        session_id: Session ID
        cfg: Configuration object
        verbose: Whether verbose logging is enabled
        
    Returns:
        Tuple of (session_dir, metrics_collector, trajectory_writer, llm_messages_writer)
    """
    # Setup unified logging with session_id
    # This must be done early so loggers are available for subsequent operations
    session_dir = setup_logging(
        session_id=session_id,
        logs_dir=cfg.logging.logs_dir,
        workspace_path=os.getcwd(),
        verbose=verbose,
    )
    
    # Initialize observability writers
    trajectory_writer = TrajectoryWriter(session_dir / "trajectory.jsonl")
    llm_messages_writer = LLMMessagesWriter(session_dir / "llm_messages.json")
    
    # Initialize metrics collector for tracking performance
    metrics_collector = MetricsCollector(session_id=session_id)
    
    return session_dir, metrics_collector, trajectory_writer, llm_messages_writer


def _initialize_llm(
    cfg: Config,
    model_name: str,
    session_id: str,
):
    """
    Initialize LLM instance
    
    Args:
        cfg: Configuration object
        model_name: Model name to use
        session_id: Session ID for tracking
        
    Returns:
        LLM instance
    """
    mainLogger.info("LLM initializing", model_name=model_name, enable_thinking=cfg.llm.enable_thinking)
    
    # Build kwargs, filtering out None values to avoid overriding factory defaults
    llm_kwargs = {
        "model": model_name,
        "session_id": session_id,  # For Anthropic x-idealab-session-id header
    }
    
    # Only add non-None values to avoid overriding factory defaults
    if cfg.llm.provider is not None:
        llm_kwargs["provider"] = cfg.llm.provider
    if cfg.llm.api_key is not None:
        llm_kwargs["api_key"] = cfg.llm.api_key
    if cfg.llm.base_url is not None:
        llm_kwargs["base_url"] = cfg.llm.base_url
    if cfg.llm.temperature is not None:
        llm_kwargs["temperature"] = cfg.llm.temperature
    if cfg.llm.max_tokens is not None:
        llm_kwargs["max_tokens"] = cfg.llm.max_tokens
    if cfg.llm.timeout is not None:
        llm_kwargs["timeout"] = cfg.llm.timeout
    if cfg.llm.parallel_tool_calls is not None:
        llm_kwargs["parallel_tool_calls"] = cfg.llm.parallel_tool_calls
    if cfg.llm.enable_thinking is not None:
        llm_kwargs["enable_thinking"] = cfg.llm.enable_thinking
    if cfg.llm.top_k is not None:
        llm_kwargs["top_k"] = cfg.llm.top_k
    if cfg.llm.top_p is not None:
        llm_kwargs["top_p"] = cfg.llm.top_p
    
    llm = create_llm(**llm_kwargs)
    
    return llm


def _initialize_tools(
    cfg: Config,
    read_tracker: ReadTracker,
    agent_profile: Any,
) -> Tuple[Any, List[str]]:
    """
    Initialize tool registry and get available tools
    
    Args:
        cfg: Configuration object
        read_tracker: Read tracker for file read tracking
        agent_profile: Agent profile instance
        
    Returns:
        Tuple of (tool_registry, available_tools)
    """
    # Initialize tool registry with workspace_root, read_tracker, and config
    workspace_root = Path(cfg.agent_config.workspace_root).expanduser().resolve()
    tool_registry = create_default_registry(
        workspace_root=workspace_root,
        read_tracker=read_tracker,
        config=cfg,
    )
    
    # Get available tools based on agent profile
    available_tools = agent_profile.get_tool_list(tool_registry.list_tool_names())
    
    return tool_registry, available_tools


def _initialize_agent_loop(
    llm,
    tool_registry,
    context_engine: ContextEngine,
    cfg: Config,
    metrics_collector: MetricsCollector,
) -> AgentLoop:
    """
    Create and initialize AgentLoop
    
    Args:
        llm: LLM instance
        tool_registry: Tool registry instance
        context_engine: Context engine instance
        cfg: Configuration object
        metrics_collector: Metrics collector instance
        
    Returns:
        AgentLoop instance
    """
    agent_loop = AgentLoop(
        llm=llm,
        tool_registry=tool_registry,
        context_engine=context_engine,
        max_iterations=cfg.agent_config.max_iterations,
        yolo_mode=cfg.agent_config.yolo,
        confirmation_callback=confirmation_callback if not cfg.agent_config.yolo else None,
        metrics_collector=metrics_collector,
        remote_tool_enabled=cfg.agent_config.remote_tool_enabled,
        remote_tool_url=cfg.agent_config.remote_tool_url,
        remote_tool_instance_id=cfg.agent_config.remote_tool_instance_id,
        remote_tool_timeout=cfg.agent_config.remote_tool_timeout,
    )
    
    return agent_loop


def initialize_agent_components(
    cfg: Config,
    agent_name: str,
    verbose: bool,
    agent_profile: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Initialize all agent components (shared logic for headless and interactive modes)
    
    This function orchestrates the initialization of all major components:
    - Agent profile and model selection
    - Session and context management
    - Observability (logging, metrics, trajectory)
    - LLM and tools
    - Agent execution loop
    
    Args:
        cfg: Configuration object
        agent_name: Name of the agent profile to use (ignored if agent_profile is provided)
        verbose: Whether verbose logging is enabled
        agent_profile: Optional pre-loaded AgentProfile (from --agent-file)
        session_id: Optional custom session ID (auto-generated if not provided)
        
    Returns:
        Dictionary containing all initialized components:
        - agent_manager: AgentProfileManager
        - agent_profile: AgentProfile
        - env_info: EnvironmentInfo
        - context_engine: ContextEngine
        - session_dir: Path to session directory
        - llm: LLM instance
        - tool_registry: ToolRegistry
        - agent_loop: AgentLoop
        - available_tools: List of tool names
        - model_name: Resolved model name
        - config: Config instance
        - metrics_collector: MetricsCollector instance
        - resumed_conversation: List of resumed Message objects (None if new session)
    """
    # 1. Initialize agent profile and determine model
    agent_manager = AgentProfileManager()
    agent_profile, model_name = _initialize_agent_profile(
        agent_manager=agent_manager,
        agent_name=agent_name,
        agent_profile=agent_profile,
        llm_model=cfg.llm.model,
    )
    
    # 2. Collect environment info
    env_info = EnvironmentInfo.collect()
    
    # 3. Determine session ID (use provided or generate new one)
    actual_session_id = session_id or _generate_session_id()
    
    # 4. Check if we should resume an existing session
    resumed_conversation = None
    if session_id:  # Only check for resume if session_id was explicitly provided
        resumed_conversation = check_and_load_existing_session(
            session_id=session_id,
            workspace_path=env_info.cwd,
            logs_dir=cfg.logging.logs_dir,
        )
    
    # 5. Create ReadTracker (independent, no dependencies)
    read_tracker = ReadTracker()
    
    # 6. Initialize tools (depends on read_tracker)
    tool_registry, available_tools = _initialize_tools(
        cfg=cfg,
        read_tracker=read_tracker,
        agent_profile=agent_profile,
    )
    
    # 7. Setup observability (logging, trajectory, metrics)
    session_dir, metrics_collector, trajectory_writer, llm_messages_writer = _setup_observability(
        session_id=actual_session_id,
        cfg=cfg,
        verbose=verbose,
    )
    
    # 8. Create ContextEngine (depends on tool_registry, env_info, agent_profile)
    context_engine = ContextEngine(
        environment=env_info,
        tool_registry=tool_registry,
        agent_profile=agent_profile,
        max_tokens=cfg.agent_config.max_context_tokens,
        session_id=actual_session_id,
        workspace=env_info.cwd,
        trajectory_writer=trajectory_writer,
        llm_messages_writer=llm_messages_writer,
        conversation_history=resumed_conversation,
        available_tools=available_tools,
    )
    
    # 9. Initialize LLM
    llm = _initialize_llm(
        cfg=cfg,
        model_name=model_name,
        session_id=actual_session_id,
    )
    
    # 10. Create agent loop
    agent_loop = _initialize_agent_loop(
        llm=llm,
        tool_registry=tool_registry,
        context_engine=context_engine,
        cfg=cfg,
        metrics_collector=metrics_collector,
    )
    
    # 11. Write session start event
    context_engine.write_session_start(
        agent_name=agent_profile.name,
        model=model_name,
        tools=available_tools,
        temperature=llm.temperature if hasattr(llm, 'temperature') else None,
    )
    
    # Return all components
    return {
        "agent_manager": agent_manager,
        "agent_profile": agent_profile,
        "env_info": env_info,
        "context_engine": context_engine,
        "session_dir": session_dir,
        "llm": llm,
        "tool_registry": tool_registry,
        "agent_loop": agent_loop,
        "available_tools": available_tools,
        "model_name": model_name,
        "config": cfg,
        "metrics_collector": metrics_collector,
        "resumed_conversation": resumed_conversation,
        "read_tracker": read_tracker,
    }

