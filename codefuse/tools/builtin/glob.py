"""
Glob Tool - Fast file pattern matching tool
"""

import glob as glob_lib
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from codefuse.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult
from codefuse.tools.builtin.filesystem_base import FileSystemToolMixin
from codefuse.tools.builtin.list_directory import DEFAULT_IGNORE_PATTERNS
from codefuse.observability import mainLogger


# Result limit
MAX_FILES = 100


@dataclass
class GlobResult:
    """Result structure for glob operations"""
    files: List[str]  # Absolute file paths
    total_found: int  # Total files found before truncation
    truncated: bool  # Whether results were truncated


class GlobTool(FileSystemToolMixin, BaseTool):
    """
    Tool for finding files by glob pattern
    
    Features:
    - Fast file pattern matching
    - Supports standard glob patterns (*, ?, **, [...])
    - Returns matching file paths sorted by modification time (newest first)
    - Automatically ignores common build/cache directories
    - Result limit to prevent excessive output
    """
    
    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Initialize GlobTool
        
        Args:
            workspace_root: Workspace root directory to restrict searches.
                          Defaults to current working directory.
        """
        super().__init__(workspace_root=workspace_root)
    
    @property
    def definition(self) -> ToolDefinition:
        """Define the glob tool"""
        return ToolDefinition(
            name="glob",
            description=(
                "Find files by name patterns using glob syntax. Handles codebases of any size efficiently.\n\n"
                "- Accepts standard glob patterns such as:\n"
                "*.py - match files in current directory\n"
                "**/*.js - search all subdirectories recursively\n"
                "src/**/*.ts - limit search to specific path\n"
                "test_*.py - match files with prefix\n\n"
                "- Notes:\n"
                "Limits results to 100 file paths.\n"
                "Supports parallel calls with different patterns(Recommended to use this tool in a batch of patterns to find files that are potentially useful)."
            ).strip(),
            parameters=[
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="The glob pattern to match files against",
                    required=True,
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description=(
                        "The directory to search in. If not specified, the workspace root will be used. "
                        "IMPORTANT: Omit this field to use the default directory. Must be an absolute path if provided."
                    ),
                    required=False,
                ),
            ],
            requires_confirmation=False,  # Searching is safe
        )
    
    def _should_ignore(self, file_path: Path) -> bool:
        """
        Check if a file should be ignored based on default patterns
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be ignored
        """
        # Check against all default ignore patterns
        for pattern in DEFAULT_IGNORE_PATTERNS:
            # Check if pattern matches any part of the path
            for part in file_path.parts:
                if glob_lib.fnmatch.fnmatch(part, pattern):
                    return True
            
            # Also check the full path string
            if glob_lib.fnmatch.fnmatch(str(file_path), pattern):
                return True
        
        return False
    
    def _execute_glob(self, pattern: str, search_path: Path) -> List[Path]:
        """
        Execute glob search with pattern
        
        Args:
            pattern: Glob pattern to match
            search_path: Directory to search in
            
        Returns:
            List of matching file paths
        """
        # Determine if pattern contains recursive wildcard
        has_recursive = '**' in pattern
        
        # Build the full pattern path
        full_pattern = str(search_path / pattern)
        
        # Execute glob
        matches = glob_lib.glob(full_pattern, recursive=has_recursive)
        
        # Convert to Path objects and filter
        result_paths: List[Path] = []
        for match in matches:
            path = Path(match).resolve()
            
            # Only include files (not directories)
            if not path.is_file():
                continue
            
            # Skip ignored paths
            if self._should_ignore(path):
                continue
            
            result_paths.append(path)
        
        return result_paths
    
    def _sort_by_mtime(self, file_paths: List[Path]) -> List[Path]:
        """
        Sort files by modification time (newest first), then by filename
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Sorted list of file paths
        """
        if not file_paths:
            return file_paths
        
        try:
            # Get file stats for all files
            file_stats = []
            for file_path in file_paths:
                try:
                    mtime = file_path.stat().st_mtime if file_path.exists() else 0
                    file_stats.append((file_path, mtime))
                except Exception:
                    # If we can't stat the file, use mtime = 0
                    file_stats.append((file_path, 0))
            
            # Sort by modification time (newest first), then by filename (alphabetical)
            file_stats.sort(key=lambda x: (-x[1], str(x[0])))
            
            return [fp for fp, _ in file_stats]
        except Exception as e:
            mainLogger.warning(f"Error sorting files by mtime: {e}")
            return file_paths
    
    def _apply_limit(self, file_paths: List[Path], limit: int = MAX_FILES) -> GlobResult:
        """
        Apply result limit
        
        Args:
            file_paths: List of file paths
            limit: Maximum number of files to return
            
        Returns:
            GlobResult with limited files and truncation info
        """
        total_found = len(file_paths)
        truncated = total_found > limit
        limited_files = file_paths[:limit] if truncated else file_paths
        
        # Convert to absolute path strings
        absolute_paths = [str(fp.resolve()) for fp in limited_files]
        
        return GlobResult(
            files=absolute_paths,
            total_found=total_found,
            truncated=truncated,
        )
    
    def execute(
        self,
        pattern: str,
        path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute the glob tool
        
        Args:
            pattern: Glob pattern to match files against
            path: Optional directory to search in (defaults to workspace_root)
            
        Returns:
            ToolResult with:
                - content: File paths (one per line) for LLM
                - display: Summary message for user
        """
        try:
            # Step 1: Validate pattern
            if not pattern or not isinstance(pattern, str):
                return self._create_error_result(
                    "Pattern is required and must be a non-empty string",
                    "Invalid pattern"
                )
            
            # Step 2: Resolve search path
            if path:
                # Check if path is absolute (required by our convention)
                if not Path(path).is_absolute():
                    return self._create_error_result(
                        f"Path must be absolute, but got relative path: {path}",
                        "Path must be absolute"
                    )
                
                search_path = self._resolve_path(path)
                
                # Check if within workspace
                if error := self._check_within_workspace(search_path):
                    mainLogger.warning(f"Glob search outside workspace: {error}")
                    return self._create_error_result(error, "Access denied: outside workspace")
            else:
                search_path = self._workspace_root
            
            # Check if path exists
            if not search_path.exists():
                return self._create_error_result(
                    f"Path not found: {search_path}",
                    "Path not found"
                )
            
            # Check if it's a directory
            if not search_path.is_dir():
                return self._create_error_result(
                    f"Path is not a directory: {search_path}",
                    "Not a directory"
                )
            
            # Step 3: Execute glob search
            mainLogger.info(f"Executing glob search: pattern='{pattern}', path={search_path}")
            matched_files = self._execute_glob(pattern, search_path)
            
            # Step 4: Sort by modification time
            sorted_files = self._sort_by_mtime(matched_files)
            
            # Step 5: Apply limit
            result = self._apply_limit(sorted_files, MAX_FILES)
            
            # Step 6: Format output
            if result.total_found == 0:
                content = "No files found"
                display = "No files found"
            else:
                # Join file paths with newlines
                content = '\n'.join(result.files)
                
                # Add truncation message if needed
                if result.truncated:
                    content += (
                        f"\n\n(Results are truncated. Found {result.total_found} files, "
                        f"showing first {len(result.files)}. "
                        f"Consider using a more specific path or pattern.)"
                    )
                
                # Display message
                num_files = len(result.files)
                if result.truncated:
                    display = f"✓ Found {num_files} files (truncated from {result.total_found})"
                else:
                    display = f"✓ Found {num_files} file{'s' if num_files != 1 else ''}"
            
            mainLogger.info(
                f"Glob search complete: pattern='{pattern}', "
                f"found={result.total_found}, returned={len(result.files)}, "
                f"truncated={result.truncated}"
            )
            
            return ToolResult(content=content, display=display)
            
        except Exception as e:
            error_msg = f"Unexpected error during glob search: {str(e)}"
            mainLogger.error(error_msg, exc_info=True)
            return self._create_error_result(error_msg, f"Error: {str(e)}")

