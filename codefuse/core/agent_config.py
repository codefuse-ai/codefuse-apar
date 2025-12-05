"""
Agent Configuration - Agent profiles and management
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

from codefuse.observability import mainLogger


@dataclass
class AgentProfile:
    """
    Agent profile defining behavior, tools, and model
    
    Loaded from Markdown files with YAML frontmatter
    """
    name: str
    description: str
    system_prompt: str
    tools: Optional[List[str]] = None  # None = inherit all tools
    model: Optional[str] = None  # None = use default model
    
    @classmethod
    def from_markdown(cls, path: str) -> "AgentProfile":
        """
        Load agent profile from Markdown file with YAML frontmatter
        
        Format:
        ```markdown
        ---
        name: agent-name
        description: Agent description
        tools: tool1, tool2, tool3  # Optional
        model: model-name  # Optional
        ---
        
        System prompt content...
        ```
        
        Args:
            path: Path to the Markdown file
            
        Returns:
            AgentProfile instance
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Agent profile not found: {path}")
        
        content = file_path.read_text(encoding='utf-8')
        
        # Parse YAML frontmatter and content
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(frontmatter_pattern, content, re.DOTALL)
        
        if not match:
            raise ValueError(f"Invalid agent profile format in {path} (missing frontmatter)")
        
        frontmatter_str = match.group(1)
        system_prompt = match.group(2).strip()
        
        # Parse frontmatter (simple YAML parsing)
        frontmatter = {}
        for line in frontmatter_str.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove comments
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                # Handle null/None values
                if value.lower() in ('null', 'none', ''):
                    value = None
                
                frontmatter[key] = value
        
        # Extract fields
        name = frontmatter.get('name')
        if not name:
            raise ValueError(f"Agent profile missing 'name' field in {path}")
        
        description = frontmatter.get('description', '')
        
        # Parse tools (comma-separated list or None)
        tools_str = frontmatter.get('tools')
        tools = None
        if tools_str:
            tools = [t.strip() for t in tools_str.split(',') if t.strip()]
        
        # Parse model
        model = frontmatter.get('model')
        if model and model.lower() in ('inherit', 'default'):
            model = None
        
        mainLogger.info("Loaded agent profile", name=name, path=str(path))
        
        return cls(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            model=model,
        )
    
    @classmethod
    def get_builtin_agent(cls) -> "AgentProfile":
        """
        Get the built-in default agent profile
        
        Returns:
            Default AgentProfile
        """
        return cls(
            name="default",
            description="Default coding assistant for general development tasks",
            system_prompt="""You are CodeFuse, an AI coding assistant designed to help developers with their coding tasks. You have access to tools that allow you to read and write files in the workspace.

Your approach:
1. Carefully analyze the user's request
2. Use available tools to gather necessary information
3. Propose clear, well-thought-out solutions
4. Execute changes carefully and verify results

When modifying files:
- Always read files before modifying them
- Make precise, targeted changes
- Explain what you're doing and why

Be concise, accurate, and helpful.""",
            tools=None,  # Inherits all available tools
            model=None,  # Uses default model from config
        )
    
    def get_tool_list(self, all_tools: List[str]) -> List[str]:
        """
        Get the list of tools available to this agent
        
        Args:
            all_tools: List of all available tool names
            
        Returns:
            List of tool names this agent can use
        """
        if self.tools is None:
            # Inherit all tools
            return all_tools
        else:
            # Return intersection of requested tools and available tools
            return [t for t in self.tools if t in all_tools]
    
    def get_model_name(
        self,
        default_model: str,
        model_aliases: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Get the model name to use for this agent
        
        Args:
            default_model: Default model name to use if not specified
            model_aliases: Optional mapping of aliases to model names
                          (e.g., {"sonnet": "claude-3-5-sonnet-20241022"})
        
        Returns:
            Resolved model name
        """
        if self.model is None:
            return default_model
        
        # Check if it's an alias
        if model_aliases and self.model in model_aliases:
            return model_aliases[self.model]
        
        # Return as-is
        return self.model


class AgentProfileManager:
    """
    Manager for agent profiles
    
    Loads built-in and user-defined agent profiles from disk.
    """
    
    def __init__(self, agent_dir: str = "~/.cfuse/agents"):
        """
        Initialize agent profile manager
        
        Args:
            agent_dir: Directory containing user-defined agent profiles
        """
        self.agent_dir = Path(agent_dir).expanduser()
        self._profiles: Dict[str, AgentProfile] = {}
        
        # Load built-in agent
        self._load_builtin_agent()
        
        # Load user agents
        self._load_user_agents()
        
        mainLogger.info("AgentProfileManager initialized", profile_count=len(self._profiles))
    
    def _load_builtin_agent(self) -> None:
        """Load the built-in default agent"""
        default_agent = AgentProfile.get_builtin_agent()
        self._profiles[default_agent.name] = default_agent
        mainLogger.debug("Loaded built-in default agent")
    
    def _load_user_agents(self) -> None:
        """Load user-defined agents from agent_dir"""
        if not self.agent_dir.exists():
            mainLogger.debug("Agent directory does not exist", agent_dir=str(self.agent_dir))
            return
        
        # Look for .md files
        for file_path in self.agent_dir.glob("*.md"):
            try:
                agent = AgentProfile.from_markdown(str(file_path))
                self._profiles[agent.name] = agent
                mainLogger.info("Loaded user agent", name=agent.name)
            except Exception as e:
                mainLogger.error("Failed to load agent", path=str(file_path), error=str(e))
    
    def get_agent(self, name: str) -> Optional[AgentProfile]:
        """
        Get an agent profile by name
        
        Args:
            name: Agent name
            
        Returns:
            AgentProfile if found, None otherwise
        """
        return self._profiles.get(name)
    
    def list_agents(self) -> List[str]:
        """
        List all available agent names
        
        Returns:
            List of agent names
        """
        return list(self._profiles.keys())
    
    def get_agent_info(self, name: str) -> Optional[str]:
        """
        Get human-readable information about an agent
        
        Args:
            name: Agent name
            
        Returns:
            Formatted string with agent info, or None if not found
        """
        agent = self.get_agent(name)
        if not agent:
            return None
        
        lines = [
            f"Agent: {agent.name}",
            f"Description: {agent.description}",
        ]
        
        if agent.model:
            lines.append(f"Model: {agent.model}")
        else:
            lines.append("Model: (inherits from config)")
        
        if agent.tools:
            lines.append(f"Tools: {', '.join(agent.tools)}")
        else:
            lines.append("Tools: (all available)")
        
        return "\n".join(lines)

