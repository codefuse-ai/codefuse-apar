"""
Read File Tool - Read file contents from the workspace
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from codefuse.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult
from codefuse.tools.builtin.filesystem_base import FileSystemToolMixin, MAX_TOKENS, MAX_FILE_SIZE_BYTES
from codefuse.observability import mainLogger

if TYPE_CHECKING:
    from codefuse.core.read_tracker import ReadTracker


# Read-specific limits
DEFAULT_MAX_LINES = 1000  # Default maximum lines to read if no range specified


class ReadFileTool(FileSystemToolMixin, BaseTool):
    """
    Tool for reading file contents
    
    Features:
    - Read entire file or specific line range
    - Safety checks for file existence and readability
    - Prevents reading binary files
    - File size and token limits
    - Workspace root directory restriction
    - Line number formatting for LLM context
    """
    
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        read_tracker: Optional["ReadTracker"] = None,
    ):
        """
        Initialize ReadFileTool
        
        Args:
            workspace_root: Workspace root directory to restrict file access.
                          Defaults to current working directory.
            read_tracker: Optional read tracker for tracking read files.
        """
        super().__init__(workspace_root=workspace_root)
        self._read_tracker = read_tracker
    
    @property
    def definition(self) -> ToolDefinition:
        """Define the read_file tool"""
        return ToolDefinition(
            name="read_file",
            description=(
                "Reads a file from the local filesystem. You can access any file directly by using this tool.\n\n"
                "Important:\n"
                "- The path parameter MUST be an absolute path, not a relative path\n"
                "- By default, it reads up to 1000 lines starting from the beginning of the file\n"
                "- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters\n"
                "- Results are returned with line numbers starting at 1\n"
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Absolute path to the file to read",
                    required=True,
                ),
                ToolParameter(
                    name="start_line",
                    type="number",
                    description="Starting line number",
                    required=False,
                ),
                ToolParameter(
                    name="end_line",
                    type="number",
                    description="Ending line number",
                    required=False,
                ),
            ],
            requires_confirmation=False,  # Reading is safe
        )
    
    def _check_file_size(self, file_path: Path, has_pagination: bool) -> Optional[str]:
        """
        Check if file size exceeds limit
        
        Args:
            file_path: Path to the file
            has_pagination: Whether pagination parameters are provided
            
        Returns:
            Error message if file is too large and no pagination, None otherwise
        """
        file_size = file_path.stat().st_size
        
        # If file is too large and no pagination is provided
        if file_size > MAX_FILE_SIZE_BYTES and not has_pagination:
            size_kb = file_size / 1024
            max_kb = MAX_FILE_SIZE_BYTES / 1024
            return (
                f"File size ({size_kb:.1f}KB) exceeds maximum ({max_kb:.0f}KB). "
                f"Please use start_line and end_line parameters to read specific portions."
            )
        return None
    
    
    def execute(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute the read_file tool
        
        Args:
            path: Absolute path to the file to read
            start_line: Optional starting line (1-indexed)
            end_line: Optional ending line (1-indexed)
            
        Returns:
            ToolResult with:
                - content: Formatted file contents with line numbers for LLM
                - display: Summary message for user
        """
        try:
            # Step 1: Check if path is absolute
            if error := self._check_absolute_path(path):
                return ToolResult(
                    content=f"Error: {error}",
                    display=f"❌ {error}"
                )
            
            # Resolve path
            file_path = Path(path).resolve()
            
            # Step 2: Check if within workspace
            if error := self._check_within_workspace(file_path):
                mainLogger.warning(f"File access outside workspace: {error}")
                return ToolResult(
                    content=f"Error: {error}",
                    display=f"❌ Access denied: outside workspace"
                )
            
            # Step 3: Check file existence
            if not file_path.exists():
                error_msg = f"File not found: {path}"
                mainLogger.error(error_msg)
                return ToolResult(
                    content=f"Error: {error_msg}",
                    display=f"❌ File not found"
                )
            
            # Step 4: Check it's a file
            if not file_path.is_file():
                error_msg = f"Path is not a file: {path}"
                mainLogger.error(error_msg)
                return ToolResult(
                    content=f"Error: {error_msg}",
                    display=f"❌ Not a file"
                )
            
            # Step 5: Check file size
            has_pagination = start_line is not None or end_line is not None
            if error := self._check_file_size(file_path, has_pagination):
                mainLogger.warning(f"File too large: {error}")
                return ToolResult(
                    content=f"Error: {error}",
                    display=f"❌ File too large (>256KB)"
                )
            
            # Step 6: Read file contents with encoding fallback
            try:
                file_content, encoding = self._read_with_encoding_fallback(file_path)
                lines = file_content.splitlines(keepends=True)
                mainLogger.debug(f"Successfully read file with encoding: {encoding}")
            except UnicodeDecodeError as e:
                error_msg = f"Cannot read file (encoding error): {path}"
                mainLogger.error(f"{error_msg}: {e}")
                return ToolResult(
                    content=f"Error: {error_msg}",
                    display=f"❌ Encoding error"
                )
            
            # Step 7: Handle line range
            start_idx = (start_line - 1) if start_line else 0
            
            # Determine end index
            if end_line is not None:
                end_idx = end_line
            else:
                # Default: read up to DEFAULT_MAX_LINES from start
                end_idx = start_idx + DEFAULT_MAX_LINES
            
            # Cap at actual file length
            end_idx = min(end_idx, len(lines))
            
            # Validate line numbers
            if start_idx < 0 or start_idx >= len(lines):
                error_msg = f"Invalid start_line {start_line} (file has {len(lines)} lines)"
                return ToolResult(
                    content=f"Error: {error_msg}",
                    display=f"❌ {error_msg}"
                )
            if end_idx < start_idx:
                error_msg = f"Invalid end_line {end_line} (must be >= start_line)"
                return ToolResult(
                    content=f"Error: {error_msg}",
                    display=f"❌ {error_msg}"
                )
            
            selected_lines = lines[start_idx:end_idx]
            content = ''.join(selected_lines)
            actual_start_line = start_line or 1
            actual_end_line = actual_start_line + len(selected_lines) - 1
            
            # Check if file was truncated
            was_truncated = end_idx < len(lines) and end_line is None
            
            # Step 8: Check token limit
            if error := self._check_token_limit(content, MAX_TOKENS):
                mainLogger.warning(f"Token limit exceeded: {error}")
                return ToolResult(
                    content=f"Error: {error}",
                    display=f"❌ Content too large (>{MAX_TOKENS:,} tokens)"
                )
            
            # Step 9: Format content with line numbers
            formatted_content = self._format_with_line_numbers(content, actual_start_line)
            
            # Add truncation warning if file was truncated
            if was_truncated:
                truncation_note = (
                    f"\n\n<system-reminder>"
                    f"Note: File has {len(lines)} total lines, but only showing lines {actual_start_line}-{actual_end_line} "
                    f"(default limit: {DEFAULT_MAX_LINES} lines). "
                    f"Use start_line and end_line parameters to read other portions of the file."
                    f"</system-reminder>"
                )
                formatted_content += truncation_note
            
            # Step 10: Prepare display message
            num_lines = len(selected_lines)
            line_range = f"lines {actual_start_line}-{actual_end_line}"
            
            if was_truncated:
                display_msg = f"✓ Read {line_range} ({num_lines}/{len(lines)} lines)"
            else:
                display_msg = f"✓ Read {line_range} ({num_lines} lines)"
            
            mainLogger.info(f"Read {file_path} ({num_lines} lines, total: {len(lines)})")
            
            # Mark file as read (for edit tool validation)
            if self._read_tracker:
                self._read_tracker.mark_as_read(str(file_path))
            
            return ToolResult(
                content=formatted_content,
                display=display_msg
            )
            
        except PermissionError as e:
            error_msg = f"Permission denied reading file: {path}"
            mainLogger.error(f"{error_msg}: {e}")
            return ToolResult(
                content=f"Error: {error_msg}",
                display=f"❌ Permission denied"
            )
        except Exception as e:
            error_msg = f"Unexpected error reading file: {path}"
            mainLogger.error(f"{error_msg}: {e}", exc_info=True)
            return ToolResult(
                content=f"Error: {error_msg} - {str(e)}",
                display=f"❌ Error: {str(e)}"
            )

