"""
Headless Mode - Single-prompt execution
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Union
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from codefuse.llm.base import ContentBlock
from codefuse.observability import mainLogger, close_all_loggers

console = Console()


def run_headless(
    prompt: str,
    components: Dict[str, Any],
    stream: bool = True,
    image_urls: tuple = tuple(),
):
    """
    Run agent in headless mode (single prompt execution)
    
    Args:
        prompt: User prompt/query
        components: Dictionary of initialized components from initialize_agent_components()
        stream: Whether to stream LLM responses
        image_urls: Optional tuple of image URLs to include in the prompt
    """
    # Unpack components
    agent_profile = components["agent_profile"]
    env_info = components["env_info"]
    agent_loop = components["agent_loop"]
    available_tools = components["available_tools"]
    session_dir = components["session_dir"]
    config = components["config"]
    model_name = components["model_name"]
    metrics_collector = components["metrics_collector"]
    context_engine = components["context_engine"]
    resumed_conversation = components["resumed_conversation"]
    
    # Build user query content (text + optional images)
    user_query: Union[str, List[ContentBlock]]
    if image_urls:
        # Build multimodal content
        content_blocks: List[ContentBlock] = []
        
        # Add text block
        if prompt:
            content_blocks.append(ContentBlock(type="text", text=prompt))
        
        # Add image blocks
        for url in image_urls:
            content_blocks.append(ContentBlock(
                type="image_url",
                image_url={"url": url}
            ))
        
        user_query = content_blocks
    else:
        # Pure text content
        user_query = prompt
    
    # User message will be logged by agent_loop automatically
    
    # Run agent loop
    mainLogger.info("Agent loop starting", session_id=context_engine.session_id)
    
    final_response = ""
    current_content = ""
    current_tool_calls = []  # Track tool calls for the current response
    iterations = 1
    
    for event in agent_loop.run(
        user_query=user_query,
        stream=stream,
    ):
        if event.type == "llm_done":
            if not stream:
                # Non-streaming: save content
                content = event.data["content"]
                if content:
                    current_content = content
            
            # Check if there are tool calls in the response
            if "tool_calls" in event.data and event.data["tool_calls"]:
                current_tool_calls = event.data["tool_calls"]
        
        elif event.type == "tool_done":
            tool_name = event.data["tool_name"]
            tool_call_id = event.data.get("tool_call_id")
            result = event.data["result"]
            confirmed = event.data.get("confirmed", True)
            
                # Tool results are logged by tool_executor automatically
        
        elif event.type == "agent_done":
            final_response = event.data["final_response"]
            iterations = event.data["iterations"]
            
            # Save final assistant message to trajectory
            assistant_message = {
                "role": "assistant",
                "content": final_response or current_content,
                "timestamp": datetime.now().isoformat(),
            }
            if current_tool_calls:
                assistant_message["tool_calls"] = current_tool_calls
            # Assistant messages are logged by agent_loop automatically
            
            # Only output the final response content
            console.print(final_response or current_content)
        
        elif event.type == "error":
            error = event.data["error"]
            console.print(f"[red]Error:[/red] {error}")
    
    # Generate and save metrics summary
    summary = metrics_collector.generate_summary()
    
    # Write summary to trajectory
    context_engine.write_session_summary(summary)
    
    mainLogger.info("Agent loop completed", status="success")
    
    # Close all loggers
    close_all_loggers()
