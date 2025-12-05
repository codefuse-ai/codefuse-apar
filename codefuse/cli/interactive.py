"""
Interactive Mode - REPL for continuous conversation
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from codefuse.llm.base import Message, MessageRole
from codefuse.observability import mainLogger, get_session_dir, close_all_loggers

console = Console()


def run_interactive(
    components: Dict[str, Any],
    stream: bool = True,
):
    """
    Run agent in interactive mode (REPL)
    
    Args:
        components: Dictionary of initialized components from initialize_agent_components()
        stream: Whether to stream LLM responses
        save_session: Whether to save session information
    """
    # Unpack components
    agent_profile = components["agent_profile"]
    env_info = components["env_info"]
    agent_loop = components["agent_loop"]
    available_tools = components["available_tools"]
    session_dir = components["session_dir"]
    config = components["config"]
    model_name = components["model_name"]
    context_engine = components["context_engine"]
    metrics_collector = components["metrics_collector"]
    resumed_conversation = components["resumed_conversation"]
    
    # Display welcome message
    console.print()
    
    # Build session info
    session_info = f"Session ID: {context_engine.session_id}"
    if resumed_conversation:
        session_info += f"\n[cyan]Resumed with {len(resumed_conversation)} messages[/cyan]"
    
    console.print(Panel(
        f"[bold blue]CodeFuse Interactive Mode[/bold blue]\n\n"
        f"Agent: {agent_profile.name}\n"
        f"Model: {model_name}\n"
        f"{session_info}\n\n"
        f"[dim]Type your message and press Enter to send.[/dim]\n"
        f"[dim]Special commands:[/dim]\n"
        f"  /exit, /quit - Exit the session\n"
        f"  /help - Show help information\n"
        f"  /clear - Clear conversation history\n"
        f"  /status - Show session status",
        border_style="blue"
    ))
    console.print()
    
    if config.agent_config.yolo:
        console.print("[yellow]⚡ YOLO mode enabled - auto-confirming all tools[/yellow]\n")
    
    # Initialize prompt session with history
    session = PromptSession(history=InMemoryHistory())
    
    # Conversation history (for context across multiple turns)
    # If resuming a session, start with the loaded history
    conversation_history: List[Message] = resumed_conversation if resumed_conversation else []
    
    mainLogger.info("Interactive mode started", session_id=context_engine.session_id)
    
    # REPL loop
    while True:
        try:
            # Get user input
            user_input = session.prompt("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.startswith("/"):
                if user_input in ["/exit", "/quit"]:
                    console.print("\n[yellow]Exiting interactive mode...[/yellow]")
                    break
                
                elif user_input == "/help":
                    _show_help()
                    continue
                
                elif user_input == "/clear":
                    conversation_history.clear()
                    # Note: This only clears local conversation history
                    # ContextEngine messages are not cleared (would need session restart)
                    console.print("[green]✓ Local conversation history cleared[/green]")
                    console.print("[dim]Note: Full reset requires restarting the session[/dim]\n")
                    mainLogger.info("Conversation history cleared", session_id=context_engine.session_id)
                    continue
                
                elif user_input == "/status":
                    _show_status(components, conversation_history)
                    continue
                
                else:
                    console.print(f"[red]Unknown command:[/red] {user_input}")
                    console.print("[dim]Type /help for available commands[/dim]\n")
                    continue
            
            # User message will be logged by agent_loop automatically
            
            # Display thinking indicator
            console.print("\n[dim]Assistant:[/dim] ", end="")
            
            # Run agent loop
            final_response = ""
            current_content = ""
            current_tool_calls = []
            iterations = 1
            
            for event in agent_loop.run(
                user_query=user_input,
                stream=stream,
            ):
                if event.type == "llm_start":
                    iteration = event.data.get("iteration", 0)
                    if iteration > 1:
                        console.print(f"\n[dim]→ Iteration {iteration}[/dim]")
                
                elif event.type == "llm_chunk":
                    delta = event.data["delta"]
                    console.print(delta, end="")
                    current_content += delta
                
                elif event.type == "llm_done":
                    if not stream:
                        content = event.data["content"]
                        if content:
                            console.print(content)
                            current_content = content
                    else:
                        console.print()
                    
                    if "tool_calls" in event.data and event.data["tool_calls"]:
                        current_tool_calls = event.data["tool_calls"]
                
                elif event.type == "tool_start":
                    tool_name = event.data["tool_name"]
                    arguments = event.data.get("arguments", {})
                    args_str = _format_tool_arguments(arguments)
                    console.print(f"\n[cyan]🔧 Executing tool:[/cyan] {tool_name}{args_str}")
                
                elif event.type == "tool_done":
                    tool_name = event.data["tool_name"]
                    tool_call_id = event.data.get("tool_call_id")
                    arguments = event.data.get("arguments", {})
                    display = event.data.get("display", event.data.get("result", ""))
                    confirmed = event.data.get("confirmed", True)
                    
                    if not confirmed:
                        console.print(f"[yellow]⚠️  Tool rejected:[/yellow] {tool_name}")
                    else:
                        # Use display field (user-friendly) instead of result (LLM content)
                        console.print(f"[cyan]{display}[/cyan]")
                    
                    # Tool results are logged by tool_executor automatically
                
                elif event.type == "agent_done":
                    final_response = event.data["final_response"]
                    iterations = event.data["iterations"]
                    
                    # Save assistant message to trajectory
                    assistant_message = {
                        "role": "assistant",
                        "content": final_response or current_content,
                        "timestamp": datetime.now().isoformat(),
                    }
                    if current_tool_calls:
                        assistant_message["tool_calls"] = current_tool_calls
                    # Assistant messages are logged by agent_loop automatically
                    
                    # Update conversation history for next turn
                    conversation_history.append(Message(
                        role=MessageRole.USER,
                        content=user_input,
                    ))
                    conversation_history.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=final_response or current_content,
                    ))
                    
                    console.print()
                
                elif event.type == "error":
                    error = event.data["error"]
                    console.print(f"\n[red]Error:[/red] {error}")
            
            # Reset for next turn
            current_content = ""
            current_tool_calls = []
        
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Use /exit or /quit to exit[/yellow]\n")
            continue
        
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}\n")
            mainLogger.error("Interactive loop error", error=str(e), exc_info=True)
            continue
    
    # Generate and save metrics summary
    summary = metrics_collector.generate_summary()
    
    # Write summary to trajectory
    context_engine.write_session_summary(summary)
    
    mainLogger.info("Interactive mode completed", status="success")
    
    # Display session summary
    console.print()
    console.print(Panel(
        f"[bold]Session Summary[/bold]\n\n"
        f"[green]Total Prompts:[/green] {summary['session']['total_prompts']}\n"
        f"[green]Total Iterations:[/green] {summary['prompts']['total_iterations']}\n"
        f"[green]API Calls:[/green] {summary['api_calls']['total']} "
        f"({summary['api_calls']['success_rate']}% success)\n"
        f"[green]Total Tokens:[/green] {summary['api_calls']['tokens']['total']:,}\n"
        f"  • Prompt: {summary['api_calls']['tokens']['prompt']:,}\n"
        f"  • Completion: {summary['api_calls']['tokens']['completion']:,}\n"
        f"  • Cache Read: {summary['api_calls']['tokens']['cache_read']:,}\n"
        f"[green]Tool Calls:[/green] {summary['tool_calls']['total']} "
        f"({summary['tool_calls']['success_rate']}% success)\n"
        f"[green]Session Duration:[/green] {summary['session']['duration']:.2f}s",
        title="[bold]Performance Metrics[/bold]",
        border_style="cyan"
    ))
    
    # Display session info
    console.print(f"\n[dim]Session logs:[/dim] {get_session_dir()}")
    
    # Close all loggers
    close_all_loggers()


def _format_tool_arguments(arguments: Dict[str, Any], max_length: int = 100) -> str:
    """
    Format tool arguments for display, with truncation if too long
    
    Args:
        arguments: Tool arguments dictionary
        max_length: Maximum length before truncation
        
    Returns:
        Formatted string representation of arguments
    """
    if not arguments:
        return ""
    
    # Convert arguments to JSON string
    args_json = json.dumps(arguments, ensure_ascii=False)
    
    # If short enough, return as-is
    if len(args_json) <= max_length:
        return f" [dim]{args_json}[/dim]"
    
    # Truncate and add ellipsis
    truncated = args_json[:max_length] + "..."
    return f" [dim]{truncated}[/dim]"


def _show_help():
    """Show help information"""
    console.print()
    console.print(Panel(
        "[bold]Interactive Mode Commands[/bold]\n\n"
        "/exit, /quit - Exit interactive mode\n"
        "/help - Show this help message\n"
        "/clear - Clear conversation history\n"
        "/status - Show current session status\n\n"
        "[dim]Just type your message to chat with the assistant.[/dim]",
        border_style="blue",
        title="Help"
    ))
    console.print()


def _show_status(components: Dict[str, Any], conversation_history: List[Message]):
    """Show current session status"""
    agent_profile = components["agent_profile"]
    model_name = components["model_name"]
    context_engine = components["context_engine"]
    config = components["config"]
    
    console.print()
    console.print(Panel(
        f"[bold]Session Status[/bold]\n\n"
        f"Session ID: {context_engine.session_id}\n"
        f"Agent: {agent_profile.name}\n"
        f"Model: {model_name}\n"
        f"Conversation Turns: {len(conversation_history) // 2}\n"
        f"Max Iterations: {config.agent_config.max_iterations}\n"
        f"YOLO Mode: {'Enabled' if config.agent_config.yolo else 'Disabled'}\n"
        f"Logs: {get_session_dir()}",
        border_style="blue",
        title="Status"
    ))
    console.print()

