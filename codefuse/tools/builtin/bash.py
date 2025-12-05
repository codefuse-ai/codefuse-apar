"""
Bash Tool - Execute bash commands in a persistent shell session
"""

import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, List, Tuple
from queue import Queue, Empty

from codefuse.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult
from codefuse.observability import mainLogger


# Constants
DEFAULT_TIMEOUT = 30
SHELL_PROMPT_MARKER = "___CFUSE_PROMPT_MARKER___"
COMMAND_END_MARKER = "___CFUSE_CMD_END___"


class ShellSession:
    """
    Manages a persistent shell session for executing commands
    
    Features:
    - Persistent bash process with shared environment
    - Non-blocking output reading
    - Working directory tracking
    - Proper cleanup on exit
    """
    
    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Initialize shell session
        
        Args:
            workspace_root: Initial working directory for the shell
        """
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._cwd = self._workspace_root
        self._process: Optional[subprocess.Popen] = None
        self._output_queue: Queue = Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        
        self._start_shell()
    
    def _start_shell(self):
        """Start the persistent shell process"""
        try:
            # Start bash in interactive mode with no startup files
            self._process = subprocess.Popen(
                ['/bin/bash', '--norc', '--noprofile'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self._workspace_root),
                text=True,
                bufsize=1,  # Line buffered
            )
            
            # Start output reader thread
            self._running = True
            self._reader_thread = threading.Thread(
                target=self._read_output,
                daemon=True
            )
            self._reader_thread.start()
            
            # Set custom prompt and disable command history
            self._send_raw(f'export PS1="{SHELL_PROMPT_MARKER}"\n')
            self._send_raw('unset HISTFILE\n')
            self._send_raw('set +o history\n')
            
            # Clear initial output
            time.sleep(0.1)
            self._drain_queue()
            
            mainLogger.info(f"Shell session started in {self._workspace_root}")
            
        except Exception as e:
            mainLogger.error(f"Failed to start shell session: {e}", exc_info=True)
            raise RuntimeError(f"Failed to start shell session: {e}")
    
    def _read_output(self):
        """Read output from shell process (runs in separate thread)"""
        if not self._process or not self._process.stdout:
            return
        
        try:
            for line in iter(self._process.stdout.readline, ''):
                if not line:
                    break
                self._output_queue.put(line)
        except Exception as e:
            mainLogger.debug(f"Output reader thread error: {e}")
        finally:
            self._running = False
    
    def _send_raw(self, command: str):
        """Send raw command to shell"""
        if self._process and self._process.stdin:
            self._process.stdin.write(command)
            self._process.stdin.flush()
    
    def _drain_queue(self) -> str:
        """Drain all output from queue"""
        output = []
        while True:
            try:
                line = self._output_queue.get_nowait()
                output.append(line)
            except Empty:
                break
        return ''.join(output)
    
    def execute_command(
        self,
        command: str,
        timeout: int = DEFAULT_TIMEOUT
    ) -> Tuple[str, int, bool]:
        """
        Execute command in the shell session
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (output, exit_code, timed_out)
        """
        if not self._process or self._process.poll() is not None:
            raise RuntimeError("Shell session is not running")
        
        # Clear any pending output
        self._drain_queue()
        
        # Send command with exit code capture and end marker
        command_with_marker = (
            f'{command}\n'
            f'echo "EXIT_CODE=$?"\n'
            f'echo "{COMMAND_END_MARKER}"\n'
        )
        self._send_raw(command_with_marker)
        
        # Collect output with timeout
        output_lines = []
        exit_code = 0
        timed_out = False
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                timed_out = True
                break
            
            try:
                line = self._output_queue.get(timeout=0.1)
                
                # Check for end marker
                if COMMAND_END_MARKER in line:
                    break
                
                # Extract exit code
                if line.startswith('EXIT_CODE='):
                    try:
                        exit_code = int(line.split('=')[1].strip())
                    except (ValueError, IndexError):
                        pass
                    continue
                
                # Skip prompt markers
                if SHELL_PROMPT_MARKER in line:
                    continue
                
                output_lines.append(line)
                
            except Empty:
                continue
        
        output = ''.join(output_lines).rstrip('\n')
        
        # Update working directory if cd command was successful
        if command.strip().startswith('cd ') and exit_code == 0:
            self._update_cwd()
        
        return output, exit_code, timed_out
    
    def _update_cwd(self):
        """Update the current working directory tracking"""
        try:
            # Get current directory from shell
            self._drain_queue()
            self._send_raw('pwd\n')
            self._send_raw(f'echo "{COMMAND_END_MARKER}"\n')
            
            # Read output
            output_lines = []
            start_time = time.time()
            while time.time() - start_time < 2:
                try:
                    line = self._output_queue.get(timeout=0.1)
                    if COMMAND_END_MARKER in line:
                        break
                    if SHELL_PROMPT_MARKER not in line:
                        output_lines.append(line)
                except Empty:
                    continue
            
            if output_lines:
                cwd_str = ''.join(output_lines).strip()
                if cwd_str:
                    self._cwd = Path(cwd_str)
                    mainLogger.debug(f"Updated CWD to: {self._cwd}")
        except Exception as e:
            mainLogger.debug(f"Failed to update CWD: {e}")
    
    def get_cwd(self) -> Path:
        """Get current working directory"""
        return self._cwd
    
    def cleanup(self):
        """Cleanup shell session"""
        self._running = False
        
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception as e:
                mainLogger.debug(f"Error during shell cleanup: {e}")
                try:
                    self._process.kill()
                except:
                    pass
        
        mainLogger.info("Shell session cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


class BashTool(BaseTool):
    """
    Tool for executing bash commands in a persistent shell session
    
    Features:
    - Persistent shell environment (env vars, cwd persist across commands)
    - Command filtering with allowed/disallowed patterns
    - Timeout control with helpful error messages
    - Safety checks and user confirmation
    """
    
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        timeout: int = DEFAULT_TIMEOUT,
        allowed_commands: Optional[List[str]] = None,
        disallowed_commands: Optional[List[str]] = None,
    ):
        """
        Initialize BashTool
        
        Args:
            workspace_root: Workspace root directory
            timeout: Default timeout for commands in seconds
            allowed_commands: List of regex patterns for auto-approved commands
            disallowed_commands: List of regex patterns for auto-rejected commands
        """
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._timeout = timeout
        self._shell_session = ShellSession(workspace_root=self._workspace_root)
        
        # Compile regex patterns
        self._allowed_patterns = [
            re.compile(pattern) for pattern in (allowed_commands or [])
        ]
        self._disallowed_patterns = [
            re.compile(pattern) for pattern in (disallowed_commands or [])
        ]
        
        # Track current command for confirmation logic
        self._current_command: Optional[str] = None
        self._current_requires_confirmation = True
    
    @property
    def definition(self) -> ToolDefinition:
        """Define the bash tool"""
        return ToolDefinition(
            name="bash",
            description=(
                "Executes a given bash command.\n\n"
                "Important:\n"
                "- VERY IMPORTANT: You MUST avoid using search commands like `find` and `grep`. "
                "Instead use grep, glob, to search. You MUST avoid read tools like "
                "`cat`, `head`, `tail`, and use read_file to read files.\n"
                "- When issuing multiple commands, use the ';' or '&&' operator to separate them. "
                "DO NOT use newlines (newlines are ok in quoted strings).\n"
                "- IMPORTANT: All commands share the same shell session. Shell state (environment variables, "
                "virtual environments, current directory, etc.) persist between commands. For example, if you "
                "set an environment variable as part of a command, the environment variable will persist for "
                "subsequent commands.\n"
                "- Try to maintain your current working directory throughout the session by using absolute paths "
                "and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.\n"
                "  <good-example>\n"
                "  pytest /foo/bar/tests\n"
                "  </good-example>\n"
                "  <bad-example>\n"
                "  cd /foo/bar && pytest tests\n"
                "  </bad-example>\n"
                "- DO NOT USE GIT COMMAND UNLESS USER EXPLICITLY REQUESTS IT"
            ),
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="The bash command to execute",
                    required=True,
                ),
            ],
            requires_confirmation=True,  # Default, but can be overridden by filters
        )
    
    @property
    def requires_confirmation(self) -> bool:
        """Check if current command requires confirmation"""
        return self._current_requires_confirmation
    
    def _check_command_filter(self, command: str) -> Tuple[str, bool, Optional[str]]:
        """
        Check command against allowed/disallowed patterns
        
        Args:
            command: Command to check
            
        Returns:
            Tuple of (status, requires_confirmation, reason)
            status: "allowed", "disallowed", or "neutral"
        """
        # Check disallowed patterns first (highest priority)
        for pattern in self._disallowed_patterns:
            if pattern.search(command):
                return "disallowed", False, f"Command matches disallowed pattern: {pattern.pattern}"
        
        # Check allowed patterns
        for pattern in self._allowed_patterns:
            if pattern.search(command):
                return "allowed", False, f"Command matches allowed pattern: {pattern.pattern}"
        
        # Neutral - requires normal confirmation
        return "neutral", True, None
    
    def execute(
        self,
        command: str,
        **kwargs
    ) -> ToolResult:
        """
        Execute the bash command
        
        Args:
            command: Bash command to execute
            
        Returns:
            ToolResult with:
                - content: Command output and status for LLM
                - display: User-friendly summary
        """
        try:
            # Check command filter
            status, requires_conf, reason = self._check_command_filter(command)
            
            # Store for requires_confirmation property
            self._current_command = command
            self._current_requires_confirmation = requires_conf
            
            # Handle disallowed commands
            if status == "disallowed":
                error_msg = f"Command rejected by policy: {reason}"
                mainLogger.warning(error_msg, command=command)
                return ToolResult(
                    content=f"Error: {error_msg}",
                    display=f"❌ Command rejected by policy"
                )
            
            # Log command execution
            cwd = self._shell_session.get_cwd()
            mainLogger.info(f"Executing bash command in {cwd}: {command}")
            
            # Execute command
            output, exit_code, timed_out = self._shell_session.execute_command(
                command,
                timeout=self._timeout
            )
            
            # Handle timeout
            if timed_out:
                timeout_msg = (
                    f"Command timed out after {self._timeout} seconds.\n\n"
                    f"Possible reasons:\n"
                    f"1. The command is taking longer than {self._timeout}s to complete\n"
                    f"2. You provided an interactive command that's waiting for input\n"
                    f"3. You provided a background command that doesn't terminate\n\n"
                    f"Consider:\n"
                    f"- Using a non-interactive version of the command\n"
                    f"- Breaking the task into smaller commands\n"
                    f"- Avoiding background processes"
                )
                mainLogger.warning(f"Command timed out: {command}")
                return ToolResult(
                    content=f"Error: {timeout_msg}",
                    display=f"⏱️ Command timed out ({self._timeout}s)"
                )
            
            # Format result
            if exit_code == 0:
                result_content = f"Command executed successfully.\n\nOutput:\n{output}" if output else "Command executed successfully (no output)."
                result_display = f"✓ Command executed (exit code: 0)"
            else:
                result_content = f"Command failed with exit code {exit_code}.\n\nOutput:\n{output}" if output else f"Command failed with exit code {exit_code} (no output)."
                result_display = f"❌ Command failed (exit code: {exit_code})"
            
            mainLogger.info(f"Command completed with exit code {exit_code}")
            
            return ToolResult(
                content=result_content,
                display=result_display
            )
            
        except RuntimeError as e:
            error_msg = f"Shell session error: {str(e)}"
            mainLogger.error(error_msg, exc_info=True)
            return ToolResult(
                content=f"Error: {error_msg}",
                display=f"❌ Shell error"
            )
        except Exception as e:
            error_msg = f"Unexpected error executing command: {str(e)}"
            mainLogger.error(error_msg, exc_info=True)
            return ToolResult(
                content=f"Error: {error_msg}",
                display=f"❌ Error: {str(e)}"
            )
    
    def cleanup(self):
        """Cleanup shell session"""
        if self._shell_session:
            self._shell_session.cleanup()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

