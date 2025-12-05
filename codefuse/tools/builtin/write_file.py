"""
Write File Tool - Write or modify file contents in the workspace
"""

from pathlib import Path
from typing import Optional

from codefuse.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult
from codefuse.tools.builtin.filesystem_base import FileSystemToolMixin, MAX_TOKENS
from codefuse.observability import mainLogger


class WriteFileTool(FileSystemToolMixin, BaseTool):
    """
    Tool for writing file contents
    
    Features:
    - Create new files or overwrite existing files
    - Safety checks for path validity and workspace restriction
    - Content size validation
    - Requires user confirmation (unless in YOLO mode)
    """
    
    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Initialize WriteFileTool
        
        Args:
            workspace_root: Workspace root directory to restrict file access.
                          Defaults to current working directory.
        """
        super().__init__(workspace_root=workspace_root)
    
    @property
    def definition(self) -> ToolDefinition:
        """Define the write_file tool"""
        return ToolDefinition(
            name="write_file",
            description=(
                "Write content to a file in the workspace (creates or overwrites).\n\n"
                "Important:\n"
                "- The path parameter MUST be an absolute path, not a relative path\n"
                "- File must be within the workspace root directory\n"
                "- Content size is limited to prevent excessive file sizes"
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Absolute path to the file to write",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file",
                    required=True,
                ),
            ],
            requires_confirmation=True,  # Writing is dangerous!
        )
    
    def execute(
        self,
        path: str,
        content: str,
        **kwargs
    ) -> ToolResult:
        """
        Execute the write_file tool
        
        Args:
            path: Absolute path to the file to write
            content: Content to write to the file
            
        Returns:
            ToolResult with:
                - content: Detailed success/error message for LLM
                - display: User-friendly summary for UI
        """
        try:
            # Step 1: Check if path is absolute
            if error := self._check_absolute_path(path):
                return self._create_error_result(error, "Path must be absolute")
            
            # Step 2: Resolve path
            file_path = self._resolve_path(path)
            
            # Step 3: Check if within workspace
            if error := self._check_within_workspace(file_path):
                mainLogger.warning(f"File write outside workspace: {error}")
                return self._create_error_result(error, "Access denied: outside workspace")
            
            # Step 4: Check content size limit
            if error := self._check_token_limit(content, MAX_TOKENS):
                mainLogger.warning(f"Content too large: {error}")
                return self._create_error_result(error, f"Content too large (>{MAX_TOKENS:,} tokens)")
            
            # Step 5: Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Step 6: Check if file exists (for logging)
            file_existed = file_path.exists()
            
            # Step 7: Write content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Step 8: Calculate stats and return result
            lines = content.count('\n') + 1
            chars = len(content)
            
            action = "Updated" if file_existed else "Created"
            mainLogger.info(f"{action} {file_path} ({lines} lines, {chars} characters)")
            
            result_content = f"Successfully {action.lower()} file: {path} ({lines} lines, {chars} characters)"
            result_display = f"✓ {action} {path} ({lines} lines)"
            
            return ToolResult(content=result_content, display=result_display)
            
        except PermissionError as e:
            error_msg = f"Permission denied writing file: {path}"
            mainLogger.error(f"{error_msg}: {e}")
            return self._create_error_result(error_msg, f"Permission denied: {path}")
        except Exception as e:
            error_msg = f"Unexpected error writing file: {path} - {str(e)}"
            mainLogger.error(f"Unexpected error writing file: {path}", exc_info=True)
            return self._create_error_result(error_msg, f"Error writing {path}: {str(e)}")

