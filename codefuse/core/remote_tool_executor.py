"""
Remote Tool Executor - Handles tool execution via HTTP requests
"""

import json
import time
from typing import Dict, Any, Optional

import requests

from codefuse.tools.base import ToolResult
from codefuse.observability import mainLogger


class RemoteToolExecutor:
    """
    Executes tools remotely via HTTP POST requests
    
    This executor sends tool calls to a remote service and receives
    the execution results over HTTP.
    """
    
    def __init__(
        self,
        url: str,
        instance_id: str,
        timeout: int = 60,
    ):
        """
        Initialize remote tool executor
        
        Args:
            url: Remote tool service URL
            instance_id: Instance ID for the remote execution environment
            timeout: Timeout for HTTP requests in seconds
        """
        self.url = url
        self.instance_id = instance_id
        self.timeout = timeout
        
        mainLogger.info(
            "RemoteToolExecutor initialized",
            url=url,
            instance_id=instance_id,
            timeout=timeout,
        )
    
    def execute(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        session_id: str,
    ) -> ToolResult:
        """
        Execute a tool remotely
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            session_id: Session ID for logging
            
        Returns:
            ToolResult containing the execution result
        """
        # Construct request payload
        payload = {
            "instance_id": self.instance_id,
            "toolName": tool_name,
            "toolArgs": tool_args,
        }
        
        mainLogger.info(
            "Sending remote tool call",
            tool_name=tool_name,
            instance_id=self.instance_id,
            url=self.url,
            payload=payload,
            session_id=session_id,
        )
        
        start_time = time.time()
        
        try:
            # Send POST request
            response = requests.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            
            response_time = time.time() - start_time
            
            mainLogger.info(
                "Received remote tool response",
                tool_name=tool_name,
                status_code=response.status_code,
                response_time_seconds=round(response_time, 2),
                session_id=session_id,
            )
            
            # Check HTTP status code
            if response.status_code != 200:
                error_msg = f"Remote tool call failed with status {response.status_code}"
                mainLogger.error(
                    "Remote tool call HTTP error",
                    tool_name=tool_name,
                    status_code=response.status_code,
                    response_text=response.text[:500],  # Log first 500 chars
                    session_id=session_id,
                )
                return ToolResult(
                    content=f"Error: {error_msg}\nResponse: {response.text}",
                    display=f"❌ Remote tool call failed (HTTP {response.status_code})",
                )
            
            # Parse JSON response
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                mainLogger.error(
                    "Failed to parse remote tool response JSON",
                    tool_name=tool_name,
                    error=str(e),
                    response_text=response.text[:500],
                    session_id=session_id,
                    exc_info=True,
                )
                return ToolResult(
                    content=f"Error: Failed to parse JSON response: {str(e)}",
                    display=f"❌ Invalid JSON response from remote tool",
                )
            
            # Validate response structure
            if "response" not in response_data:
                mainLogger.error(
                    "Invalid remote tool response structure: missing 'response' field",
                    tool_name=tool_name,
                    response_data=response_data,
                    session_id=session_id,
                )
                return ToolResult(
                    content=f"Error: Invalid response structure: {json.dumps(response_data)}",
                    display=f"❌ Invalid response format from remote tool",
                )
            
            response_inner = response_data["response"]
            
            # Extract result and success flag
            result_content = response_inner.get("result", "")
            success = response_inner.get("success", False)
            
            mainLogger.info(
                "Remote tool execution completed",
                tool_name=tool_name,
                success=success,
                result_length=len(result_content),
                session_id=session_id,
            )
            
            # Return result
            if success:
                return ToolResult(
                    content=result_content,
                    display=f"✓ Remote tool '{tool_name}' executed successfully",
                )
            else:
                # Tool executed but reported failure
                mainLogger.warning(
                    "Remote tool execution reported failure",
                    tool_name=tool_name,
                    result=result_content[:500],
                    session_id=session_id,
                )
                return ToolResult(
                    content=result_content,
                    display=f"⚠ Remote tool '{tool_name}' completed with errors",
                )
        
        except requests.exceptions.Timeout:
            error_msg = f"Remote tool call timed out after {self.timeout} seconds"
            mainLogger.error(
                "Remote tool call timeout",
                tool_name=tool_name,
                timeout=self.timeout,
                session_id=session_id,
            )
            return ToolResult(
                content=f"Error: {error_msg}",
                display=f"❌ Remote tool call timed out",
            )
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            mainLogger.error(
                "Remote tool call connection error",
                tool_name=tool_name,
                error=str(e),
                url=self.url,
                session_id=session_id,
                exc_info=True,
            )
            return ToolResult(
                content=f"Error: {error_msg}",
                display=f"❌ Failed to connect to remote tool service",
            )
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            mainLogger.error(
                "Remote tool call request error",
                tool_name=tool_name,
                error=str(e),
                session_id=session_id,
                exc_info=True,
            )
            return ToolResult(
                content=f"Error: {error_msg}",
                display=f"❌ Remote tool call failed",
            )
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            mainLogger.error(
                "Remote tool call unexpected error",
                tool_name=tool_name,
                error=str(e),
                session_id=session_id,
                exc_info=True,
            )
            return ToolResult(
                content=f"Error: {error_msg}",
                display=f"❌ Unexpected error in remote tool call",
            )

