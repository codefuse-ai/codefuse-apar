"""
Tool Registry - Manage available tools
"""

from pathlib import Path
from typing import Dict, Optional, List, Any

from codefuse.tools.base import BaseTool, ToolDefinition
from codefuse.llm.base import Tool as LLMTool
from codefuse.observability import mainLogger


class ToolRegistry:
    """
    Registry for managing available tools
    
    The registry maintains a collection of tools and provides methods
    to register, retrieve, and list tools.
    """
    
    def __init__(self):
        """Initialize empty tool registry"""
        self._tools: Dict[str, BaseTool] = {}
        mainLogger.info("Initialized empty ToolRegistry")
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool
        
        Args:
            tool: Tool instance to register
        """
        name = tool.definition.name
        if name in self._tools:
            mainLogger.warning("Tool already registered, overwriting", tool_name=name)
        
        self._tools[name] = tool
        mainLogger.info(
            "Registered tool",
            tool_name=name,
            requires_confirmation=tool.requires_confirmation,
        )
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance if found, None otherwise
        """
        return self._tools.get(name)
    
    def get_all_definitions(self) -> List[ToolDefinition]:
        """
        Get all tool definitions
        
        Returns:
            List of all registered tool definitions
        """
        return [tool.definition for tool in self._tools.values()]
    
    def get_tools_for_llm(self, tool_names: Optional[List[str]] = None) -> List[LLMTool]:
        """
        Get tools in LLM-compatible format
        
        Args:
            tool_names: Optional list of specific tool names to include.
                       If None, includes all tools.
        
        Returns:
            List of Tool objects compatible with LLM.generate()
        """
        definitions = self.get_all_definitions()
        
        # Filter by tool_names if specified
        if tool_names is not None:
            definitions = [d for d in definitions if d.name in tool_names]
        
        # Convert to LLM Tool format
        llm_tools = []
        for definition in definitions:
            openai_format = definition.to_openai_format()
            llm_tools.append(
                LLMTool(
                    type=openai_format["type"],
                    function=openai_format["function"]
                )
            )
        
        return llm_tools
    
    def list_tool_names(self) -> List[str]:
        """
        List all registered tool names
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        """Get number of registered tools"""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered"""
        return name in self._tools


def create_default_registry(
    workspace_root: Optional["Path"] = None,
    read_tracker: Optional[Any] = None,
    config: Optional[Any] = None,
) -> ToolRegistry:
    """
    Create a default tool registry with all built-in tools
    
    Args:
        workspace_root: Workspace root directory for file operations.
                       Defaults to current working directory.
        read_tracker: Optional read tracker for file read tracking (needed by ReadFileTool/EditFileTool).
        config: Optional configuration object for tool-specific settings.
    
    Returns:
        ToolRegistry with all built-in tools registered
    """
    from pathlib import Path
    from codefuse.tools.builtin import (
        ReadFileTool,
        WriteFileTool,
        EditFileTool,
        ListDirectoryTool,
        GrepTool,
        GlobTool,
        BashTool,
    )
    
    registry = ToolRegistry()
    
    # Resolve workspace_root
    workspace = (workspace_root or Path.cwd()).resolve()
    
    # Register built-in tools with workspace_root
    # ReadFileTool and EditFileTool need read_tracker for file read tracking
    registry.register(ReadFileTool(workspace_root=workspace, read_tracker=read_tracker))
    registry.register(WriteFileTool(workspace_root=workspace))
    registry.register(EditFileTool(workspace_root=workspace, read_tracker=read_tracker))
    # registry.register(ListDirectoryTool(workspace_root=workspace))
    registry.register(GrepTool(workspace_root=workspace))
    registry.register(GlobTool(workspace_root=workspace))
    
    # Register BashTool with configuration
    bash_timeout = config.agent_config.bash_timeout if config else 30
    bash_allowed = config.agent_config.bash_allowed_commands if config else []
    bash_disallowed = config.agent_config.bash_disallowed_commands if config else []
    
    registry.register(BashTool(
        workspace_root=workspace,
        timeout=bash_timeout,
        allowed_commands=bash_allowed,
        disallowed_commands=bash_disallowed,
    ))
    
    mainLogger.info(
        "Created default registry",
        tool_count=len(registry.list_tool_names()),
        workspace_root=str(workspace)
    )
    
    return registry

