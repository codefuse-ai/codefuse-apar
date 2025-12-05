"""
Base classes for tools
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ToolResult:
    """
    Result of a tool execution
    
    Attributes:
        content: Full result content for LLM (detailed, structured)
        display: User-friendly display text for interactive mode (concise, formatted)
    """
    content: str
    display: Optional[str] = None
    
    def __post_init__(self):
        """If display is not provided, use content as display"""
        if self.display is None:
            self.display = self.content
    
    def __str__(self) -> str:
        """String representation returns content for LLM"""
        return self.content


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            result["enum"] = self.enum
        return result


@dataclass
class ToolDefinition:
    """Definition of a tool"""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    requires_confirmation: bool = False  # Whether user confirmation is required
    
    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling format
        
        Returns:
            Dict compatible with OpenAI's tools API
        """
        # Build parameters schema
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_dict()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }


class BaseTool(ABC):
    """
    Abstract base class for all tools
    
    Tools are the actions that the agent can take in the environment.
    Each tool must define its interface and implement the execution logic.
    """
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """
        Get the tool definition
        
        Returns:
            ToolDefinition describing the tool's interface
        """
        pass
    
    @property
    def requires_confirmation(self) -> bool:
        """
        Check if this tool requires user confirmation before execution
        
        Dangerous operations (like writing files) should require confirmation
        unless running in YOLO mode.
        
        Returns:
            True if confirmation is required, False otherwise
        """
        return self.definition.requires_confirmation
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given arguments
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult containing:
                - content: Full result for LLM (detailed information)
                - display: User-friendly display text (concise summary)
            
        Raises:
            Exception: If tool execution fails
        """
        pass
    
    def validate_arguments(self, **kwargs) -> None:
        """
        Validate tool arguments before execution
        
        Args:
            **kwargs: Arguments to validate
            
        Raises:
            ValueError: If arguments are invalid
        """
        # Check required parameters
        for param in self.definition.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Missing required parameter: {param.name}")

