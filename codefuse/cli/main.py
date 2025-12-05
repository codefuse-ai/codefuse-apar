"""
Main CLI Entry Point - Unified command-line interface
"""

import sys
import json
import click
from rich.console import Console

from codefuse.config import Config
from codefuse.core import AgentProfileManager, AgentProfile
from codefuse.cli.common import initialize_agent_components, handle_list_agents
from codefuse.cli.headless import run_headless
from codefuse.cli.interactive import run_interactive

console = Console()


@click.command()
@click.option(
    "-p", "--prompt",
    help="User prompt/query (if provided, runs in headless mode)"
)
@click.option(
    "-pp", "--prompt-file",
    type=click.Path(exists=True),
    help="Read prompt from file (mutually exclusive with -p)"
)
@click.option(
    "--agent",
    default="default",
    help="Agent profile to use (default: default)"
)
@click.option(
    "--agent-file",
    type=click.Path(exists=True),
    help="Load agent profile from Markdown file (overrides --agent)"
)
@click.option(
    "--provider",
    help="LLM provider (openai_compatible, anthropic, gemini)"
)
@click.option(
    "--model",
    help="Override model name"
)
@click.option(
    "--api-key",
    help="API key (or use environment variable)"
)
@click.option(
    "--base-url",
    help="Base URL for API endpoint"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose logging"
)
@click.option(
    "--logs-dir",
    help="Base directory for logs (default: ~/.cfuse/logs)"
)
@click.option(
    "--max-iterations",
    type=int,
    help="Maximum agent iterations (default: 200)"
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Enable/disable streaming output (default: enabled)"
)
@click.option(
    "--yolo",
    is_flag=True,
    help="YOLO mode: auto-confirm all tool executions"
)
@click.option(
    "--list-agents",
    is_flag=True,
    help="List available agent profiles and exit"
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--save-session",
    is_flag=True,
    help="Save session trajectory to file"
)
@click.option(
    "--temperature",
    type=float,
    help="Model temperature (0.0-2.0, default: 0.0)"
)
@click.option(
    "--top-p",
    type=float,
    help="Nucleus sampling parameter (0.0-1.0)"
)
@click.option(
    "--top-k",
    type=int,
    help="Top-k sampling parameter"
)
@click.option(
    "--parallel-tool-calls/--no-parallel-tool-calls",
    default=None,
    help="Enable/disable parallel tool calls (default: enabled)"
)
@click.option(
    "--think",
    is_flag=True,
    help="Enable thinking mode for models that support it"
)
@click.option(
    "--session-id",
    help="Custom session ID (auto-generated if not provided)"
)
@click.option(
    "--http",
    is_flag=True,
    help="Enable HTTP server mode for external tool execution"
)
@click.option(
    "--port",
    type=int,
    default=8080,
    help="Port for HTTP server mode (default: 8080)"
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host address for HTTP server mode (default: 0.0.0.0, use 127.0.0.1 for localhost only)"
)
@click.option(
    "--remote-tool-enabled",
    is_flag=True,
    help="Enable remote tool execution via HTTP"
)
@click.option(
    "--remote-tool-url",
    help="URL of the remote tool service"
)
@click.option(
    "--remote-tool-instance-id",
    help="Instance ID for remote tool execution"
)
@click.option(
    "--remote-tool-timeout",
    type=int,
    help="Timeout for remote tool calls in seconds (default: 60)"
)
@click.option(
    "--image-url",
    multiple=True,
    help="Image URL (can be specified multiple times, supports HTTP/HTTPS or base64 data URI)"
)
@click.option(
    "--image-url-file",
    type=click.Path(exists=True),
    help="Read image URLs from JSON file (should contain a list of URLs)"
)
def main(
    prompt: str,
    prompt_file: str,
    agent: str,
    agent_file: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    verbose: bool,
    logs_dir: str,
    max_iterations: int,
    stream: bool,
    yolo: bool,
    list_agents: bool,
    config: str,
    save_session: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    parallel_tool_calls: bool,
    think: bool,
    session_id: str,
    http: bool,
    port: int,
    host: str,
    remote_tool_enabled: bool,
    remote_tool_url: str,
    remote_tool_instance_id: str,
    remote_tool_timeout: int,
    image_url: tuple,
    image_url_file: str,
):
    """
    CodeFuse Agent - AI-powered coding assistant
    
    Run in headless mode with -p/--prompt or -pp/--prompt-file, interactive mode, or HTTP server mode.
    
    \b
    Examples:
        # Headless mode
        cfuse -p "Read README.md and summarize it"
        
        # Read prompt from file
        cfuse -pp prompt.txt
        
        # Interactive mode
        cfuse
        
        # HTTP server mode (listen on all interfaces)
        cfuse --http --port 8080
        
        # HTTP server mode (localhost only)
        cfuse --http --port 8080 --host 127.0.0.1
        
        # Resume an existing session (loads conversation history)
        cfuse --session-id session_20241029_123456_abc123def
        
        # YOLO mode (auto-confirm all tools)
        cfuse -p "Create a hello.py file" --yolo
        
        # Use specific agent
        cfuse -p "Debug this error" --agent debugger
        
        # Load agent from file
        cfuse -p "Help me with this task" --agent-file ./my_agent.md
        
        # List available agents
        cfuse --list-agents
    """
    
    try:
        # Check for mutually exclusive parameters
        if prompt and prompt_file:
            console.print("[red]Error:[/red] Cannot use both -p/--prompt and -pp/--prompt-file at the same time")
            sys.exit(1)
        
        # Validate image_url usage
        if (image_url or image_url_file) and not (prompt or prompt_file):
            console.print("[red]Error:[/red] --image-url/--image-url-file requires -p/--prompt or -pp/--prompt-file")
            sys.exit(1)
        
        # If prompt-file is provided, read the file content
        if prompt_file:
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                if not prompt:
                    console.print(f"[red]Error:[/red] Prompt file '{prompt_file}' is empty")
                    sys.exit(1)
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to read prompt file '{prompt_file}': {e}")
                sys.exit(1)
        
        # If image-url-file is provided, read and parse the JSON file
        image_urls_from_file = []
        if image_url_file:
            try:
                with open(image_url_file, 'r', encoding='utf-8') as f:
                    image_urls_from_file = json.load(f)
                
                # Validate that it's a list
                if not isinstance(image_urls_from_file, list):
                    console.print(f"[red]Error:[/red] Image URL file '{image_url_file}' must contain a JSON list")
                    sys.exit(1)
                
                # Validate that all elements are strings
                if not all(isinstance(url, str) for url in image_urls_from_file):
                    console.print(f"[red]Error:[/red] All elements in image URL file must be strings")
                    sys.exit(1)
                
                if not image_urls_from_file:
                    console.print(f"[yellow]Warning:[/yellow] Image URL file '{image_url_file}' is empty")
                
            except json.JSONDecodeError as e:
                console.print(f"[red]Error:[/red] Failed to parse JSON from '{image_url_file}': {e}")
                sys.exit(1)
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to read image URL file '{image_url_file}': {e}")
                sys.exit(1)
        
        # Merge image URLs from both sources (CLI args and file)
        all_image_urls = list(image_url) + image_urls_from_file
        
        # Handle HTTP server mode
        if http:
            from codefuse.cli.http_server import run_http_server
            
            # HTTP mode doesn't require API key for LLM
            # Only needs minimal config
            cfg = Config.load(config)
            
            # Merge CLI args (None filtering is handled by Config.merge_with_cli_args)
            cli_args = {
                "verbose": verbose,
                "logs_dir": logs_dir,
            }
            
            cfg = Config.merge_with_cli_args(cfg, **cli_args)
            
            # Start HTTP server
            run_http_server(cfg, host, port)
            return
        
        # Handle --list-agents early (no need to initialize components)
        if list_agents:
            agent_manager = AgentProfileManager()
            handle_list_agents(agent_manager)
            return
        
        # Load configuration
        cfg = Config.load(config)
        
        # Merge CLI arguments (None filtering is handled by Config.merge_with_cli_args)
        cli_args = {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "parallel_tool_calls": parallel_tool_calls,
            "enable_thinking": think,
            "max_iterations": max_iterations,
            "yolo": yolo,
            "agent": agent,
            "verbose": verbose,
            "logs_dir": logs_dir,
            "remote_tool_enabled": remote_tool_enabled,
            "remote_tool_url": remote_tool_url,
            "remote_tool_instance_id": remote_tool_instance_id,
            "remote_tool_timeout": remote_tool_timeout,
        }
        
        cfg = Config.merge_with_cli_args(cfg, **cli_args)
        
        # Validate configuration
        validation_errors = cfg.validate()
        if validation_errors:
            console.print("[red]Configuration Errors:[/red]")
            for error in validation_errors:
                console.print(f"  - {error}")
            sys.exit(1)
        
        # Handle --agent-file: load agent profile from file
        loaded_agent_profile = None
        if agent_file:
            try:
                console.print(f"[cyan]Loading agent from file:[/cyan] {agent_file}")
                loaded_agent_profile = AgentProfile.from_markdown(agent_file)
                console.print(f"[green]✓ Agent loaded:[/green] {loaded_agent_profile.name}")
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to load agent from file '{agent_file}'")
                console.print(f"[red]Reason:[/red] {str(e)}")
                if verbose:
                    import traceback
                    console.print(traceback.format_exc())
                sys.exit(1)
        
        # Initialize all components once (shared by both modes)
        components = initialize_agent_components(
            cfg=cfg,
            agent_name=agent,
            agent_profile=loaded_agent_profile,
            verbose=cfg.logging.verbose,
            session_id=session_id,
        )
        
        # Route to appropriate mode based on presence of prompt
        if prompt:
            # Headless mode: single prompt execution
            run_headless(
                prompt=prompt,
                components=components,
                stream=stream,
                image_urls=tuple(all_image_urls),
            )
        else:
            # Interactive mode: REPL
            run_interactive(
                components=components,
                stream=stream,
            )
    
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

