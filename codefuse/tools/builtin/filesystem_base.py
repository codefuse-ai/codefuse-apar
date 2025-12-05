"""
Filesystem Tool Base - Common functionality for file system tools
"""

from pathlib import Path
from typing import Optional, Tuple, List

from codefuse.tools.base import ToolResult
from codefuse.observability import mainLogger


# File size and token limits
MAX_FILE_SIZE_BYTES = 256 * 1024  # 256KB
MAX_TOKENS = 25000  # Maximum tokens allowed in file content


class FileSystemToolMixin:
    """
    Mixin class providing common functionality for file system tools
    
    This mixin provides:
    - Path safety checks (absolute path, workspace restriction)
    - Content size limits (token estimation and checking)
    - Error handling utilities
    """
    
    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Initialize the file system tool mixin
        
        Args:
            workspace_root: Workspace root directory to restrict file access.
                          Defaults to current working directory.
        """
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        mainLogger.debug(f"FileSystemToolMixin initialized with workspace_root: {self._workspace_root}")
    
    def _check_absolute_path(self, path: str) -> Optional[str]:
        """
        Check if path is absolute
        
        Args:
            path: Path to check
            
        Returns:
            Error message if path is not absolute, None otherwise
        """
        if not Path(path).is_absolute():
            return f"Path must be absolute, but got relative path: {path}"
        return None
    
    def _check_within_workspace(self, file_path: Path) -> Optional[str]:
        """
        Check if file is within workspace root directory
        
        Args:
            file_path: Path to check (must be resolved)
            
        Returns:
            Error message if file is outside workspace, None otherwise
        """
        try:
            file_path.relative_to(self._workspace_root)
            return None
        except ValueError:
            return (
                f"Path must be within workspace root ({self._workspace_root}), "
                f"but got: {file_path}"
            )
    
    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a path to absolute form
        
        Args:
            path: Path to resolve (can be absolute or relative)
            
        Returns:
            Resolved absolute Path object
        """
        return Path(path).expanduser().resolve()
    
    def _estimate_tokens(self, content: str) -> int:
        """
        Estimate token count for content
        
        Uses rough estimation: characters / 4
        
        Args:
            content: Text content to estimate
            
        Returns:
            Estimated token count
        """
        return len(content) // 4
    
    def _check_token_limit(self, content: str, max_tokens: int = MAX_TOKENS) -> Optional[str]:
        """
        Check if content exceeds token limit
        
        Args:
            content: Content to check
            max_tokens: Maximum allowed tokens (default: MAX_TOKENS)
            
        Returns:
            Error message if content exceeds limit, None otherwise
        """
        token_count = self._estimate_tokens(content)
        
        if token_count > max_tokens:
            return (
                f"Content ({token_count:,} tokens) exceeds maximum ({max_tokens:,} tokens). "
                f"Please reduce the content size."
            )
        return None
    
    def _create_error_result(self, error_msg: str, display_msg: str) -> ToolResult:
        """
        Create a standardized error result
        
        Args:
            error_msg: Detailed error message for LLM
            display_msg: User-friendly error message for display
            
        Returns:
            ToolResult with error information
        """
        return ToolResult(
            content=f"Error: {error_msg}",
            display=f"❌ {display_msg}"
        )
    
    def _read_with_encoding_fallback(self, file_path: Path) -> Tuple[str, str]:
        """
        Read file with multiple encoding fallbacks
        
        This method tries multiple encodings to read a file, falling back
        to more permissive options if the preferred encoding fails.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Tuple of (file_content, encoding_used)
            
        Raises:
            UnicodeDecodeError: If all encoding attempts fail
        """
        encodings = [
            ("utf-8", None),
            ("latin-1", None),
            ("utf-8", "replace"),
        ]
        
        last_exception = None
        for encoding, errors in encodings:
            try:
                content = file_path.read_text(encoding=encoding, errors=errors)
                return content, encoding
            except UnicodeDecodeError as e:
                last_exception = e
                continue
        
        # All encodings failed
        raise UnicodeDecodeError(
            "all",
            b"",
            0,
            1,
            f"Failed to decode file with all attempted encodings: {last_exception}"
        )
    
    def _find_occurrence_lines(self, content: str, search_string: str) -> List[int]:
        """
        Find line numbers where search_string starts
        
        This method handles both single-line and multi-line search strings.
        For multi-line strings, it returns the line number where each occurrence starts.
        
        Args:
            content: File content to search in
            search_string: String to search for (can be multi-line)
            
        Returns:
            List of line numbers (1-indexed) where search_string starts
        """
        occurrence_lines = []
        start_pos = 0
        
        # Find all occurrences in the content
        while True:
            pos = content.find(search_string, start_pos)
            if pos == -1:
                break
            
            # Count line number by counting newlines before this position
            line_num = content[:pos].count('\n') + 1
            occurrence_lines.append(line_num)
            
            # Move to next position
            start_pos = pos + 1
        
        return occurrence_lines
    
    def _format_with_line_numbers(self, content: str, start_line: int = 1) -> str:
        """
        Format content with line numbers
        
        Format: LINE_NUMBER→LINE_CONTENT
        Line numbers are right-aligned to 6 characters
        
        Args:
            content: Text content to format
            start_line: Starting line number (1-indexed)
            
        Returns:
            Formatted content with line numbers
        """
        if not content:
            return content
        
        lines = content.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line_num = start_line + i
            # Right-align line number to 6 characters
            formatted_lines.append(f"{line_num:6d}→{line}")
        
        return '\n'.join(formatted_lines)

