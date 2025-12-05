"""
Edit File Tool - Perform exact string replacements in files
"""

from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING

from codefuse.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult
from codefuse.tools.builtin.filesystem_base import FileSystemToolMixin, MAX_TOKENS
from codefuse.observability import mainLogger

if TYPE_CHECKING:
    from codefuse.core.read_tracker import ReadTracker


# Edit-specific constants
CONTEXT_LINES = 4  # Number of lines to show before/after edit for confirmation


class EditFileTool(FileSystemToolMixin, BaseTool):
    """
    Tool for editing file contents with exact string replacement
    
    Features:
    - Requires file to be read before editing (safety check)
    - Exact string matching with uniqueness validation
    - Support for replace_all mode (rename variables, etc.)
    - Shows edit snippet for confirmation
    - Workspace restriction and safety checks
    """
    
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        read_tracker: Optional["ReadTracker"] = None,
    ):
        """
        Initialize EditFileTool
        
        Args:
            workspace_root: Workspace root directory to restrict file access.
                          Defaults to current working directory.
            read_tracker: Read tracker for validation that file was read before editing.
        """
        super().__init__(workspace_root=workspace_root)
        self._read_tracker = read_tracker
    
    @property
    def definition(self) -> ToolDefinition:
        """Define the edit_file tool"""
        return ToolDefinition(
            name="edit_file",
            description=(
                "Performs exact string replacements in files.\n\n"
                "Usage:\n"
                "- You MUST use read_file tool at least once before editing. "
                "This tool will error if you attempt an edit without reading the file.\n"
                "- When editing text from read_file output, ensure you preserve the exact indentation "
                "(tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: "
                "spaces + line number + → + content. Everything after the → is the actual file content to match. "
                "Never include any part of the line number prefix in old_string or new_string.\n"
                "- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n"
                "- The edit will FAIL if old_string is not unique in the file. Either provide a larger string "
                "with more surrounding context to make it unique or use replace_all to change every instance.\n"
                "- Use replace_all for replacing and renaming strings across the file. "
                "This parameter is useful if you want to rename a variable for instance.\n\n"
                "Important:\n"
                "- The file_path parameter MUST be an absolute path, not a relative path\n"
                "- old_string must match the file content exactly (including whitespace)\n"
                "- new_string must be different from old_string"
            ),
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="The absolute path to the file to modify",
                    required=True,
                ),
                ToolParameter(
                    name="old_string",
                    type="string",
                    description="The text to replace",
                    required=True,
                ),
                ToolParameter(
                    name="new_string",
                    type="string",
                    description="The text to replace it with (must be different from old_string)",
                    required=True,
                ),
                ToolParameter(
                    name="replace_all",
                    type="boolean",
                    description="Replace all occurrences of old_string (default false)",
                    required=False,
                ),
            ],
            requires_confirmation=True,  # Editing is dangerous!
        )
    
    def _generate_edit_snippet(
        self,
        content: str,
        replacement_line: int,
        new_content: str,
        context_lines: int = CONTEXT_LINES
    ) -> Tuple[str, int]:
        """
        Generate a snippet showing the edited region with context
        
        Args:
            content: New file content (after replacement)
            replacement_line: Line number where replacement started (0-indexed)
            new_content: The new string that was inserted
            context_lines: Number of context lines to show before/after
            
        Returns:
            Tuple of (formatted_snippet, start_line_number)
        """
        lines = content.split('\n')
        num_new_lines = new_content.count('\n')
        
        # Calculate snippet range
        start_line = max(0, replacement_line - context_lines)
        end_line = min(len(lines), replacement_line + num_new_lines + 1 + context_lines)
        
        snippet_lines = lines[start_line:end_line]
        snippet_content = '\n'.join(snippet_lines)
        
        # Format with line numbers using inherited method (1-indexed)
        formatted_snippet = self._format_with_line_numbers(snippet_content, start_line + 1)
        
        return formatted_snippet, start_line + 1
    
    def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs
    ) -> ToolResult:
        """
        Execute the edit_file tool
        
        Args:
            file_path: Absolute path to the file to edit
            old_string: Text to replace
            new_string: Replacement text
            replace_all: If True, replace all occurrences; if False, only unique occurrences
            
        Returns:
            ToolResult with:
                - content: Detailed edit confirmation with snippet for LLM
                - display: User-friendly summary for UI
        """
        try:
            # Step 1: Check if path is absolute
            if error := self._check_absolute_path(file_path):
                return self._create_error_result(error, "Path must be absolute")
            
            # Step 2: Resolve path
            resolved_path = self._resolve_path(file_path)
            
            # Step 3: Check if within workspace
            if error := self._check_within_workspace(resolved_path):
                mainLogger.warning(f"File edit outside workspace: {error}")
                return self._create_error_result(error, "Access denied: outside workspace")
            
            # Step 4: Check file existence
            if not resolved_path.exists():
                error_msg = f"File not found: {file_path}"
                mainLogger.error(error_msg)
                return self._create_error_result(error_msg, "File not found")
            
            # Step 5: Check it's a file
            if not resolved_path.is_file():
                error_msg = f"Path is not a file: {file_path}"
                mainLogger.error(error_msg)
                return self._create_error_result(error_msg, "Not a file")
            
            # Step 6: Check if file was read
            if self._read_tracker and not self._read_tracker.is_read(str(resolved_path)):
                error_msg = (
                    f"File has not been read yet: {file_path}. "
                    f"You must use read_file tool at least once before editing."
                )
                mainLogger.warning(error_msg)
                return self._create_error_result(
                    error_msg,
                    "Must read file before editing"
                )
            
            # Step 7: Read file with encoding fallback
            try:
                file_content, encoding = self._read_with_encoding_fallback(resolved_path)
            except UnicodeDecodeError as e:
                error_msg = f"Cannot read file (encoding error): {file_path}"
                mainLogger.error(f"{error_msg}: {e}")
                return self._create_error_result(error_msg, "File encoding error")
            
            # Step 8: Normalize tabs
            file_content = file_content.expandtabs()
            old_string = old_string.expandtabs()
            new_string = new_string.expandtabs()
            
            # Step 9: Check if old_string == new_string
            if old_string == new_string:
                error_msg = f"old_string is identical to new_string. No replacement needed."
                mainLogger.info(error_msg)
                return self._create_error_result(error_msg, "No changes to make")
            
            # Step 10: Count occurrences
            occurrences = file_content.count(old_string)
            
            if occurrences == 0:
                error_msg = (
                    f"old_string not found in file. The string to replace does not appear "
                    f"verbatim in {file_path}. Make sure to match the exact content including "
                    f"whitespace and indentation."
                )
                mainLogger.warning(error_msg)
                return self._create_error_result(error_msg, "String not found")
            
            if occurrences > 1 and not replace_all:
                occurrence_lines = self._find_occurrence_lines(file_content, old_string)
                error_msg = (
                    f"Multiple occurrences of old_string found in lines {occurrence_lines}. "
                    f"Please ensure it is unique by providing more context, or set replace_all=True "
                    f"to replace all {occurrences} occurrences."
                )
                mainLogger.warning(error_msg)
                return self._create_error_result(
                    error_msg,
                    f"Not unique ({occurrences} occurrences)"
                )
            
            # Step 11: Perform replacement
            if replace_all:
                new_file_content = file_content.replace(old_string, new_string)
                num_replacements = occurrences
            else:
                # Replace only the first (and only) occurrence
                new_file_content = file_content.replace(old_string, new_string, 1)
                num_replacements = 1
            
            # Step 12: Check content size limit
            if error := self._check_token_limit(new_file_content, MAX_TOKENS):
                mainLogger.warning(f"New content too large: {error}")
                return self._create_error_result(error, f"Content too large (>{MAX_TOKENS:,} tokens)")
            
            # Step 13: Write new content
            try:
                resolved_path.write_text(new_file_content, encoding=encoding)
            except Exception as e:
                error_msg = f"Failed to write file: {file_path}"
                mainLogger.error(f"{error_msg}: {e}", exc_info=True)
                return self._create_error_result(error_msg, f"Write failed: {str(e)}")
            
            # Step 14: Generate edit snippet for confirmation
            replacement_line = file_content.split(old_string)[0].count('\n')
            snippet, snippet_start_line = self._generate_edit_snippet(
                new_file_content,
                replacement_line,
                new_string,
                CONTEXT_LINES
            )
            
            # Step 15: Prepare success message
            action = "all occurrences" if replace_all else "occurrence"
            mainLogger.info(
                f"Edited {resolved_path} ({num_replacements} {action} replaced)"
            )
            
            result_content = (
                f"Successfully edited {file_path}. "
                f"Replaced {num_replacements} {action} of old_string with new_string.\n\n"
                f"Here's a snippet of the edited file showing the changes (lines {snippet_start_line}-"
                f"{snippet_start_line + snippet.count(chr(10))}):\n"
                f"{snippet}\n\n"
                f"Review the changes and make sure they are as expected. "
                f"Edit the file again if necessary."
            )
            
            result_display = (
                f"✓ Edited {file_path} ({num_replacements} replacement{'s' if num_replacements > 1 else ''})"
            )
            
            return ToolResult(content=result_content, display=result_display)
            
        except PermissionError as e:
            error_msg = f"Permission denied editing file: {file_path}"
            mainLogger.error(f"{error_msg}: {e}")
            return self._create_error_result(error_msg, "Permission denied")
        except Exception as e:
            error_msg = f"Unexpected error editing file: {file_path}"
            mainLogger.error(f"{error_msg}: {e}", exc_info=True)
            return self._create_error_result(error_msg, f"Error: {str(e)}")

