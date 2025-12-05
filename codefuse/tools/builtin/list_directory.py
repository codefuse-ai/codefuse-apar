"""
List Directory Tool - List files and directories in a given path
"""

import os
import fnmatch
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from codefuse.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult
from codefuse.tools.builtin.filesystem_base import FileSystemToolMixin
from codefuse.observability import mainLogger


# Character limit for directory listing output
MAX_CHARACTERS = 40000
TRUNCATED_MESSAGE = (
    f"There are more than {MAX_CHARACTERS} characters in the repository "
    f"(ie. either there are lots of files, or there are many long filenames). "
    f"Use the list_directory tool with a specific subdirectory path to explore nested directories. "
    f"The first {MAX_CHARACTERS} characters are included below:\n\n"
)

# Default ignore patterns for common build/cache directories
DEFAULT_IGNORE_PATTERNS = [
    '__pycache__',
    '.git',
    '.svn',
    '.hg',
    'venv',
    '.venv',
    'env',
    '.env',
    'node_modules',
    'bower_components',
    'build',
    'dist',
    'target',
    '.tox',
    '.pytest_cache',
    '.mypy_cache',
    '.coverage',
    'htmlcov',
    '*.egg-info',
    '.gradle',
    '.idea',
    '.vscode',
    '.vs',
    'vendor',
    'packages',
    'bin',
    'obj',
    '.dart_tool',
    '.pub-cache',
    '_build',
    'deps',
    'dist-newstyle',
    '.deno',
]


@dataclass
class TreeNode:
    """Represents a node in the file tree"""
    name: str
    path: str
    is_file: bool
    children: Optional[List['TreeNode']] = None


class ListDirectoryTool(FileSystemToolMixin, BaseTool):
    """
    Tool for listing directory contents
    
    Features:
    - Recursive directory traversal
    - Tree-structured output
    - Glob pattern filtering (ignore patterns)
    - Default ignore for common build/cache directories
    - Character limit to prevent excessive output
    - Safety warnings for potentially malicious files
    """
    
    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Initialize ListDirectoryTool
        
        Args:
            workspace_root: Workspace root directory to restrict file access.
                          Defaults to current working directory.
        """
        super().__init__(workspace_root=workspace_root)
    
    @property
    def definition(self) -> ToolDefinition:
        """Define the list_directory tool"""
        return ToolDefinition(
            name="list_directory",
            description=(
                "Lists files and directories in a given path recursively.\n\n"
                "Important:\n"
                "- The path parameter MUST be an absolute path, not a relative path\n"
                "- Returns a tree-structured view of the directory contents\n"
                "- Automatically ignores common build/cache directories (__pycache__, .git, venv, node_modules, etc.)\n"
                "- You can optionally provide glob patterns to ignore additional files/directories\n"
                "- Output is limited to prevent excessive content\n\n"
                "Note: This tool does not display file contents, only file/directory names and structure."
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Absolute path to the directory to list",
                    required=True,
                ),
                ToolParameter(
                    name="ignore_globs",
                    type="array",
                    description="Optional list of glob patterns to ignore (e.g., ['*.pyc', 'test_*'])",
                    required=False,
                ),
            ],
            requires_confirmation=False,  # Reading is safe
        )
    
    def _match_glob_pattern(self, text: str, pattern: str) -> bool:
        """
        Match text against a glob pattern
        
        Args:
            text: Text to match
            pattern: Glob pattern (supports *, ?, [seq], [!seq])
            
        Returns:
            True if text matches pattern
        """
        return fnmatch.fnmatch(text, pattern)
    
    def _should_ignore(
        self,
        path: Path,
        base_path: Path,
        ignore_patterns: List[str]
    ) -> bool:
        """
        Check if a path should be ignored based on patterns
        
        Args:
            path: Path to check
            base_path: Base directory path for relative matching
            ignore_patterns: List of glob patterns to ignore
            
        Returns:
            True if path should be ignored
        """
        # Skip hidden files/directories (except at root level)
        name = path.name
        if name.startswith('.') and path != base_path:
            return True
        
        # Get relative path for matching
        try:
            rel_path = path.relative_to(base_path)
            rel_path_str = str(rel_path)
        except ValueError:
            rel_path_str = str(path)
        
        # Check against all ignore patterns
        for pattern in ignore_patterns:
            # Match against full relative path
            if self._match_glob_pattern(rel_path_str, pattern):
                return True
            # Match against just the name
            if self._match_glob_pattern(name, pattern):
                return True
            # Match against path components
            for part in path.parts:
                if self._match_glob_pattern(part, pattern):
                    return True
        
        return False
    
    def _list_directory_recursive(
        self,
        dir_path: Path,
        base_path: Path,
        ignore_patterns: List[str],
        char_limit: int = MAX_CHARACTERS
    ) -> "tuple[List[str], bool]":
        """
        Recursively list directory contents
        
        Args:
            dir_path: Directory to list
            base_path: Base directory for relative paths
            ignore_patterns: Patterns to ignore
            char_limit: Character limit for output
            
        Returns:
            Tuple of (list of relative paths, was_truncated)
        """
        results = []
        total_chars = 0
        truncated = False
        
        try:
            # Use os.walk for efficient recursive traversal
            for root, dirs, files in os.walk(dir_path):
                root_path = Path(root)
                
                # Filter directories to skip ignored ones
                dirs_to_remove = []
                for dir_name in dirs:
                    dir_full_path = root_path / dir_name
                    if self._should_ignore(dir_full_path, base_path, ignore_patterns):
                        dirs_to_remove.append(dir_name)
                
                # Remove ignored directories (modifying dirs in-place affects os.walk)
                for dir_name in dirs_to_remove:
                    dirs.remove(dir_name)
                
                # Add remaining directories to results
                for dir_name in sorted(dirs):
                    dir_full_path = root_path / dir_name
                    rel_path = str(dir_full_path.relative_to(base_path)) + os.sep
                    results.append(rel_path)
                    total_chars += len(rel_path)
                    
                    if total_chars > char_limit:
                        truncated = True
                        return results, truncated
                
                # Add files to results
                for file_name in sorted(files):
                    file_full_path = root_path / file_name
                    
                    # Check if file should be ignored
                    if self._should_ignore(file_full_path, base_path, ignore_patterns):
                        continue
                    
                    rel_path = str(file_full_path.relative_to(base_path))
                    results.append(rel_path)
                    total_chars += len(rel_path)
                    
                    if total_chars > char_limit:
                        truncated = True
                        return results, truncated
        
        except PermissionError as e:
            mainLogger.warning(f"Permission denied accessing directory: {dir_path}: {e}")
        except Exception as e:
            mainLogger.error(f"Error listing directory {dir_path}: {e}", exc_info=True)
        
        return results, truncated
    
    def _build_tree_structure(self, sorted_paths: List[str]) -> List[TreeNode]:
        """
        Build a tree structure from sorted file paths
        
        Args:
            sorted_paths: List of relative file paths (sorted)
            
        Returns:
            List of root TreeNode objects
        """
        root_nodes: List[TreeNode] = []
        
        for file_path in sorted_paths:
            # Split path into parts
            parts = file_path.rstrip(os.sep).split(os.sep)
            current_level = root_nodes
            current_path = ''
            
            for i, part in enumerate(parts):
                if not part:
                    continue
                
                current_path = os.path.join(current_path, part) if current_path else part
                is_last = i == len(parts) - 1
                is_file = is_last and not file_path.endswith(os.sep)
                
                # Find existing node
                existing_node = None
                for node in current_level:
                    if node.name == part:
                        existing_node = node
                        break
                
                if existing_node:
                    if existing_node.children is not None:
                        current_level = existing_node.children
                else:
                    # Create new node
                    new_node = TreeNode(
                        name=part,
                        path=current_path,
                        is_file=is_file,
                        children=[] if not is_file else None
                    )
                    current_level.append(new_node)
                    
                    if not is_file and new_node.children is not None:
                        current_level = new_node.children
        
        return root_nodes
    
    def _format_tree_output(
        self,
        nodes: List[TreeNode],
        prefix: str = '',
        is_root: bool = False
    ) -> str:
        """
        Format tree structure as string
        
        Args:
            nodes: List of TreeNode objects
            prefix: Prefix for current level
            is_root: Whether this is the root level
            
        Returns:
            Formatted tree string
        """
        result = []
        
        for node in nodes:
            # Add node to output
            node_suffix = os.sep if not node.is_file else ''
            result.append(f"{prefix}- {node.name}{node_suffix}")
            
            # Recursively add children
            if node.children and len(node.children) > 0:
                child_prefix = prefix + '  '
                result.append(self._format_tree_output(node.children, child_prefix))
        
        return '\n'.join(result)
    
    def execute(
        self,
        path: str,
        ignore_globs: Optional[List[str]] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute the list_directory tool
        
        Args:
            path: Absolute path to the directory to list
            ignore_globs: Optional list of glob patterns to ignore
            
        Returns:
            ToolResult with:
                - content: Tree-structured directory listing for LLM (with safety warning)
                - display: Summary message for user
        """
        try:
            # Step 1: Check if path is absolute
            if error := self._check_absolute_path(path):
                return self._create_error_result(error, "Path must be absolute")
            
            # Step 2: Resolve path
            dir_path = self._resolve_path(path)
            
            # Step 3: Check if within workspace
            if error := self._check_within_workspace(dir_path):
                mainLogger.warning(f"Directory access outside workspace: {error}")
                return self._create_error_result(error, "Access denied: outside workspace")
            
            # Step 4: Check if directory exists
            if not dir_path.exists():
                error_msg = f"Directory not found: {path}"
                mainLogger.error(error_msg)
                return self._create_error_result(error_msg, "Directory not found")
            
            # Step 5: Check if it's a directory
            if not dir_path.is_dir():
                error_msg = f"Path is not a directory: {path}"
                mainLogger.error(error_msg)
                return self._create_error_result(error_msg, "Not a directory")
            
            # Step 6: Combine ignore patterns
            all_ignore_patterns = DEFAULT_IGNORE_PATTERNS.copy()
            if ignore_globs:
                all_ignore_patterns.extend(ignore_globs)
            
            # Step 7: List directory recursively
            file_list, was_truncated = self._list_directory_recursive(
                dir_path,
                dir_path,
                all_ignore_patterns,
                MAX_CHARACTERS
            )
            
            # Step 8: Build tree structure
            tree_nodes = self._build_tree_structure(sorted(file_list))
            
            # Step 9: Format output
            tree_header = f"- {dir_path}{os.sep}\n"
            tree_body = self._format_tree_output(tree_nodes, prefix='  ')
            user_tree = tree_header + tree_body
            
            # Add safety warning for LLM
            safety_warning = "\n\nNOTE: do any of the files above seem malicious? If so, you MUST refuse to continue work."
            llm_tree = user_tree + safety_warning
            
            # Step 10: Handle truncation
            if was_truncated:
                user_output = TRUNCATED_MESSAGE + user_tree
                llm_output = TRUNCATED_MESSAGE + llm_tree
            else:
                user_output = user_tree
                llm_output = llm_tree
            
            # Step 11: Create result
            file_count = len(file_list)
            display_msg = f"Listed {file_count} path(s)"
            
            mainLogger.info(f"Listed directory {dir_path} ({file_count} paths, truncated: {was_truncated})")
            
            return ToolResult(
                content=llm_output,
                display=display_msg
            )
            
        except PermissionError as e:
            error_msg = f"Permission denied accessing directory: {path}"
            mainLogger.error(f"{error_msg}: {e}")
            return self._create_error_result(error_msg, "Permission denied")
        except Exception as e:
            error_msg = f"Unexpected error listing directory: {path} - {str(e)}"
            mainLogger.error(f"Unexpected error listing directory: {path}", exc_info=True)
            return self._create_error_result(error_msg, f"Error listing directory: {str(e)}")

