"""
Grep Tool - Search files using ripgrep
"""

import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from codefuse.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult
from codefuse.tools.builtin.filesystem_base import FileSystemToolMixin
from codefuse.tools.utils.ripgrep import execute_ripgrep
from codefuse.observability import mainLogger


# Output limits to prevent overwhelming responses
MAX_OUTPUT_LINES = 20000
MAX_OUTPUT_CHARS = 20000


@dataclass
class GrepSearchResult:
    """Result structure for grep operations"""
    mode: str  # 'content', 'files_with_matches', or 'count'
    lines: List[str]  # Raw output lines from ripgrep
    num_files: int = 0
    num_matches: int = 0
    num_lines: int = 0


class GrepTool(FileSystemToolMixin, BaseTool):
    """
    Tool for searching files using ripgrep
    
    Features:
    - Fast regex search across files using ripgrep
    - Multiple output modes: content, files_with_matches, count
    - Context control (-A/-B/-C for surrounding lines)
    - File filtering (glob patterns, file types)
    - Respects .gitignore by default
    - Output size limits
    """
    
    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Initialize GrepTool
        
        Args:
            workspace_root: Workspace root directory to restrict searches.
                          Defaults to current working directory.
        """
        super().__init__(workspace_root=workspace_root)
    
    @property
    def definition(self) -> ToolDefinition:
        """Define the grep tool"""
        return ToolDefinition(
            name="grep",
            description=(
                "Search file contents using regex patterns. Built on ripgrep for high performance.\n\n"
                "- Best for exact symbol/string searches across the codebase. Prefer this over terminal grep/rg.\n"
                "- Supports full regex: \"log.*Error\", \"function\\s+\\w+\". Escape special chars for exact matches: \"functionCall\\(\\)\".\n"
                "- Pattern syntax follows ripgrep - literal braces need escaping: interface\\{\\} to match interface{}.\n\n"
                "- Output Modes:\n"
                "files_with_matches - shows only file paths (default)\n"
                "content - shows matching lines with context (auto-enables line numbers)\n"
                "count - shows match counts per file\n\n"
                "- Notes:\n"
                "Use 'type' or 'glob' parameters only when certain of file type needed.\n"
                "Avoid overly broad patterns like '--glob *' which may bypass filters and slow down search.\n"
                "Supports parallel calls with different patterns(Recommended to use this tool in a batch of patterns to find files that are potentially useful)."
            ).strip(),
            parameters=[
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="The regular expression pattern to search for in file contents (rg --regexp)",
                    required=True,
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="File or directory to search in (rg pattern -- PATH). Defaults to Cursor workspace roots.",
                    required=False,
                ),
                ToolParameter(
                    name="glob",
                    type="string",
                    description="Glob pattern (rg --glob GLOB -- PATH) to filter files (e.g. \"*.js\", \"*.{ts,tsx}\").",
                    required=False,
                ),
                ToolParameter(
                    name="output_mode",
                    type="string",
                    description=(
                        "Output mode: \"content\" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), "
                        "\"files_with_matches\" shows file paths (supports head_limit), "
                        "\"count\" shows match counts (supports head_limit). Defaults to \"files_with_matches\"."
                    ),
                    required=False,
                    enum=["content", "files_with_matches", "count"],
                ),
                ToolParameter(
                    name="-B",
                    type="number",
                    description="Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise.",
                    required=False,
                ),
                ToolParameter(
                    name="-A",
                    type="number",
                    description="Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise.",
                    required=False,
                ),
                ToolParameter(
                    name="-C",
                    type="number",
                    description="Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise.",
                    required=False,
                ),
                ToolParameter(
                    name="-n",
                    type="boolean",
                    description="Show line numbers in output (rg -n). Defaults to true. Requires output_mode: \"content\", ignored otherwise.",
                    required=False,
                ),
                ToolParameter(
                    name="-i",
                    type="boolean",
                    description="Case insensitive search (rg -i) Defaults to false",
                    required=False,
                ),
                ToolParameter(
                    name="type",
                    type="string",
                    description="File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than glob for standard file types.",
                    required=False,
                ),
                ToolParameter(
                    name="head_limit",
                    type="number",
                    description=(
                        "Limit output to first N lines/entries, equivalent to \"| head -N\". "
                        "Works across all output modes: content (limits output lines), files_with_matches (limits file paths), "
                        "count (limits count entries). When unspecified, shows all ripgrep results."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="multiline",
                    type="boolean",
                    description="Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.",
                    required=False,
                ),
            ],
            requires_confirmation=False,  # Searching is safe
        )
    
    def _build_ripgrep_args(
        self,
        pattern: str,
        output_mode: str = "content",
        before_context: Optional[int] = None,
        after_context: Optional[int] = None,
        context: Optional[int] = None,
        show_line_numbers: Optional[bool] = None,
        case_insensitive: Optional[bool] = None,
        file_type: Optional[str] = None,
        glob: Optional[str] = None,
        multiline: Optional[bool] = None,
    ) -> List[str]:
        """
        Build ripgrep command arguments
        
        Args:
            pattern: Regex pattern to search
            output_mode: Output mode (content/files_with_matches/count)
            before_context: Lines before match (-B)
            after_context: Lines after match (-A)
            context: Lines before and after match (-C)
            show_line_numbers: Show line numbers (-n)
            case_insensitive: Case insensitive search (-i)
            file_type: File type filter (--type)
            glob: Glob pattern filter (--glob)
            multiline: Enable multiline mode (-U --multiline-dotall)
            
        Returns:
            List of ripgrep arguments
        """
        args: List[str] = []
        
        # Enable multiline mode if requested
        if multiline:
            args.extend(['-U', '--multiline-dotall'])
        
        # Case insensitive search
        if case_insensitive:
            args.append('-i')
        
        # Output mode specific flags
        if output_mode == 'files_with_matches':
            args.append('-l')  # List files with matches
        elif output_mode == 'count':
            args.append('-c')  # Count matches per file
        
        # Line numbers for content mode
        if show_line_numbers and output_mode == 'content':
            args.append('-n')
        
        # Context options for content mode
        if output_mode == 'content':
            if context is not None:
                args.extend(['-C', str(context)])
            else:
                if before_context is not None:
                    args.extend(['-B', str(before_context)])
                if after_context is not None:
                    args.extend(['-A', str(after_context)])
        
        # Handle patterns that start with dash
        if pattern.startswith('-'):
            args.extend(['-e', pattern])
        else:
            args.append(pattern)
        
        # File type filter
        if file_type:
            args.extend(['--type', file_type])
        
        # Glob patterns
        if glob:
            # Split glob patterns by whitespace and commas
            glob_patterns = self._parse_glob_patterns(glob)
            for pattern in glob_patterns:
                args.extend(['--glob', pattern])
        
        return args
    
    def _parse_glob_patterns(self, glob: str) -> List[str]:
        """
        Parse glob patterns, handling complex patterns with braces
        
        Args:
            glob: Glob pattern string (may contain multiple patterns)
            
        Returns:
            List of individual glob patterns
        """
        patterns: List[str] = []
        parts = glob.split()
        
        for part in parts:
            if '{' in part and '}' in part:
                # Keep brace patterns intact
                patterns.append(part)
            else:
                # Split by comma
                patterns.extend([p.strip() for p in part.split(',') if p.strip()])
        
        return [p for p in patterns if p]
    
    def _apply_head_limit(self, lines: List[str], head_limit: Optional[int]) -> List[str]:
        """
        Apply head limit to output lines
        
        Args:
            lines: Output lines
            head_limit: Maximum number of lines to return
            
        Returns:
            Limited list of lines
        """
        if head_limit is not None and head_limit >= 0:
            return lines[:head_limit]
        return lines
    
    def _sort_files_by_mtime(self, file_paths: List[str]) -> List[str]:
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
                    path = Path(file_path)
                    mtime = path.stat().st_mtime if path.exists() else 0
                    file_stats.append((file_path, mtime))
                except Exception:
                    # If we can't stat the file, use mtime = 0
                    file_stats.append((file_path, 0))
            
            # Sort by modification time (newest first), then by filename (alphabetical)
            file_stats.sort(key=lambda x: (-x[1], x[0]))
            
            return [fp for fp, _ in file_stats]
        except Exception as e:
            mainLogger.warning(f"Error sorting files by mtime: {e}")
            return file_paths
    
    def _parse_ripgrep_output(
        self,
        lines: List[str],
        output_mode: str
    ) -> GrepSearchResult:
        """
        Parse ripgrep output based on mode
        
        Args:
            lines: Output lines from ripgrep
            output_mode: Output mode (content/files_with_matches/count)
            
        Returns:
            GrepSearchResult object
        """
        if output_mode == 'content':
            return GrepSearchResult(
                mode='content',
                lines=lines,
                num_lines=len(lines),
            )
        
        elif output_mode == 'count':
            total_matches = 0
            file_count = 0
            
            for line in lines:
                # Count format: "filepath:count"
                colon_index = line.rfind(':')
                if colon_index > 0:
                    count_str = line[colon_index + 1:]
                    try:
                        count = int(count_str)
                        total_matches += count
                        file_count += 1
                    except ValueError:
                        pass
            
            return GrepSearchResult(
                mode='count',
                lines=lines,
                num_matches=total_matches,
                num_files=file_count,
            )
        
        else:  # files_with_matches
            # Sort files by modification time (newest first), then by name
            sorted_files = self._sort_files_by_mtime(lines)
            return GrepSearchResult(
                mode='files_with_matches',
                lines=sorted_files,
                num_files=len(sorted_files),
            )
    
    def _apply_output_limit(self, content: str) -> str:
        """
        Apply final output truncation to prevent overwhelming responses
        
        Args:
            content: Content to potentially truncate
            
        Returns:
            Truncated content if necessary
        """
        if len(content) <= MAX_OUTPUT_CHARS:
            return content
        
        truncated = content[:MAX_OUTPUT_CHARS]
        extra_lines = content[MAX_OUTPUT_CHARS:].count('\n')
        return f"{truncated}\n\n... [{extra_lines} lines truncated] ..."
    
    def _format_result(self, result: GrepSearchResult) -> ToolResult:
        """
        Format the final result for return
        
        Args:
            result: Grep search result
            
        Returns:
            ToolResult with content and display
        """
        if result.mode == 'content':
            line_count = result.num_lines
            content = '\n'.join(result.lines) if result.lines else 'No matches found'
            llm_content = self._apply_output_limit(content)
            display = f"✓ Found {line_count} matching line{'s' if line_count != 1 else ''}"
            
            return ToolResult(content=llm_content, display=display)
        
        elif result.mode == 'count':
            match_count = result.num_matches
            file_count = result.num_files
            
            content_lines = result.lines if result.lines else ['No matches found']
            summary = (
                f"\n\nFound {match_count} total {'occurrence' if match_count == 1 else 'occurrences'} "
                f"across {file_count} {'file' if file_count == 1 else 'files'}."
            )
            full_content = '\n'.join(content_lines) + summary
            llm_content = self._apply_output_limit(full_content)
            
            display = (
                f"✓ Found {match_count} match{'es' if match_count != 1 else ''} "
                f"in {file_count} file{'s' if file_count != 1 else ''}"
            )
            
            return ToolResult(content=llm_content, display=display)
        
        else:  # files_with_matches
            file_count = result.num_files
            
            if file_count == 0:
                return ToolResult(
                    content='No files found',
                    display='No matching files found'
                )
            
            file_list = '\n'.join(result.lines)
            full_content = f"Found {file_count} file{'s' if file_count != 1 else ''}:\n{file_list}"
            llm_content = self._apply_output_limit(full_content)
            
            display = f"✓ Found {file_count} matching file{'s' if file_count != 1 else ''}"
            
            return ToolResult(content=llm_content, display=display)
    
    def execute(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        output_mode: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute the grep tool
        
        Args:
            pattern: Regular expression pattern to search for
            path: Optional path to search in (defaults to workspace_root)
            glob: Optional glob pattern to filter files
            output_mode: Output mode (content/files_with_matches/count)
            **kwargs: Additional parameters (-A, -B, -C, -n, -i, type, head_limit, multiline)
            
        Returns:
            ToolResult with:
                - content: Search results for LLM
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
                    mainLogger.warning(f"Grep search outside workspace: {error}")
                    return self._create_error_result(error, "Access denied: outside workspace")
            else:
                search_path = self._workspace_root
            
            # Check if path exists
            if not search_path.exists():
                return self._create_error_result(
                    f"Path not found: {search_path}",
                    "Path not found"
                )
            
            # Step 3: Parse parameters
            output_mode = output_mode or 'files_with_matches'
            before_context = kwargs.get('-B')
            after_context = kwargs.get('-A')
            context = kwargs.get('-C')
            show_line_numbers = kwargs.get('-n', True)
            case_insensitive = kwargs.get('-i', False)
            file_type = kwargs.get('type')
            head_limit = kwargs.get('head_limit')
            multiline = kwargs.get('multiline', False)
            
            # Step 4: Validate context options only work with content mode
            context_options = ['-A', '-B', '-C', '-n']
            has_context_options = any(kwargs.get(opt) is not None for opt in context_options)
            if has_context_options and output_mode != 'content':
                return self._create_error_result(
                    "Context options (-A, -B, -C, -n) can only be used with output_mode: 'content'",
                    "Invalid parameters for output mode"
                )
            
            # Step 5: Validate -C doesn't conflict with -A/-B
            if context is not None and (before_context is not None or after_context is not None):
                return self._create_error_result(
                    "Cannot use -C with -A or -B (use either -C alone or -A/-B combination)",
                    "Conflicting context parameters"
                )
            
            # Step 6: Build ripgrep arguments
            rg_args = self._build_ripgrep_args(
                pattern=pattern,
                output_mode=output_mode,
                before_context=before_context,
                after_context=after_context,
                context=context,
                show_line_numbers=show_line_numbers,
                case_insensitive=case_insensitive,
                file_type=file_type,
                glob=glob,
                multiline=multiline,
            )
            
            # Step 7: Execute ripgrep
            mainLogger.info(f"Executing grep search: pattern='{pattern}', path={search_path}, mode={output_mode}")
            output_lines = execute_ripgrep(rg_args, str(search_path))
            
            # Step 8: Apply head limit
            limited_lines = self._apply_head_limit(output_lines, head_limit)
            
            # Step 9: Parse output
            result = self._parse_ripgrep_output(limited_lines, output_mode)
            
            # Step 10: Format and return result
            return self._format_result(result)
            
        except RuntimeError as e:
            # Ripgrep not found or other runtime error
            error_msg = str(e)
            mainLogger.error(f"Grep tool error: {error_msg}")
            return self._create_error_result(error_msg, "Grep error")
        except Exception as e:
            error_msg = f"Unexpected error during grep search: {str(e)}"
            mainLogger.error(error_msg, exc_info=True)
            return self._create_error_result(error_msg, f"Error: {str(e)}")

