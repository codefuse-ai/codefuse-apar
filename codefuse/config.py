"""
Configuration Management
"""

import os
import copy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, List, Any
import yaml

from codefuse.observability import mainLogger


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    enable_thinking: Optional[bool] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None


@dataclass
class AgentConfig:
    """Agent configuration"""
    max_iterations: Optional[int] = None
    max_context_tokens: Optional[int] = None
    enable_tools: Optional[bool] = None
    yolo: Optional[bool] = None
    agent: Optional[str] = None
    workspace_root: Optional[str] = None
    bash_timeout: Optional[int] = None
    bash_allowed_commands: Optional[list] = None
    bash_disallowed_commands: Optional[list] = None
    remote_tool_enabled: Optional[bool] = None
    remote_tool_url: Optional[str] = None
    remote_tool_instance_id: Optional[str] = None
    remote_tool_timeout: Optional[int] = None


@dataclass
class LoggingConfig:
    """Logging configuration"""
    logs_dir: Optional[str] = None
    verbose: Optional[bool] = None


# Default values (centralized)
DEFAULTS = {
    "llm": {
        "provider": "openai_compatible",
        "model": "",
        "api_key": "",
        "base_url": "",
        "temperature": 0.0,
        "max_tokens": None,
        "timeout": 60,
        "parallel_tool_calls": True,
        "enable_thinking": False,
        "top_k": None,
        "top_p": None,
    },
    "agent_config": {
        "max_iterations": 200,
        "max_context_tokens": 100000,
        "enable_tools": True,
        "yolo": False,
        "agent": "default",
        "workspace_root": ".",
        "bash_timeout": 30,
        "bash_allowed_commands": [],
        "bash_disallowed_commands": [],
        "remote_tool_enabled": False,
        "remote_tool_url": "",
        "remote_tool_instance_id": "",
        "remote_tool_timeout": 60,
    },
    "logging": {
        "logs_dir": "~/.cfuse/logs",
        "verbose": False,
    },
}


# Environment variable mapping (only core configs)
ENV_MAPPING = [
    ("api_key", "llm", str, ["OPENAI_API_KEY"]),
    ("base_url", "llm", str, ["LLM_BASE_URL"]),
    ("model", "llm", str, ["LLM_MODEL"]),
    ("logs_dir", "logging", str, ["LOGS_DIR"]),
    ("verbose", "logging", bool, ["VERBOSE"]),
]


# Validation rules (section, field, check_function, error_message)
VALIDATIONS = [
    ('llm', 'temperature', lambda v: 0 <= v <= 2, "temperature must be 0-2"),
    ('llm', 'top_p', lambda v: 0 <= v <= 1, "top_p must be 0-1"),
    ('llm', 'top_k', lambda v: v > 0, "top_k must be positive"),
    ('llm', 'timeout', lambda v: v > 0, "timeout must be positive"),
    ('llm', 'max_tokens', lambda v: v > 0, "max_tokens must be positive"),
    ('agent_config', 'max_iterations', lambda v: v > 0, "max_iterations must be positive"),
    ('agent_config', 'bash_timeout', lambda v: v > 0, "bash_timeout must be positive"),
    ('agent_config', 'remote_tool_timeout', lambda v: v > 0, "remote_tool_timeout must be positive"),
]


def _get_env_value(env_vars: List[str], type_: type) -> Any:
    """Get first available environment variable and convert to type"""
    for env_var in env_vars:
        value = os.getenv(env_var)
        if value is not None:
            try:
                if type_ == bool:
                    return value.lower() in ('true', '1', 'yes')
                elif type_ == int:
                    return int(value)
                elif type_ == float:
                    return float(value)
                else:
                    return value
            except (ValueError, AttributeError) as e:
                mainLogger.warning(
                    "Failed to convert environment variable",
                    env_var=env_var,
                    value=value,
                    error=str(e)
                )
    return None


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand ${VAR} in strings"""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    elif isinstance(data, str):
        import re
        def replacer(match):
            var_name = match.group(1) or match.group(2)
            return os.getenv(var_name, match.group(0))
        return re.sub(r'\$\{([^}]+)\}|\$(\w+)', replacer, data)
    return data


@dataclass
class Config:
    """Main configuration"""
    llm: LLMConfig = None
    agent_config: AgentConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.llm is None:
            self.llm = LLMConfig()
        if self.agent_config is None:
            self.agent_config = AgentConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
    
    @classmethod
    def from_defaults(cls) -> "Config":
        """Create config from default values"""
        return cls(
            llm=LLMConfig(**DEFAULTS["llm"]),
            agent_config=AgentConfig(**DEFAULTS["agent_config"]),
            logging=LoggingConfig(**DEFAULTS["logging"]),
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> Optional["Config"]:
        """Load config from YAML file"""
        file_path = Path(path).expanduser()
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            data = _expand_env_vars(data)
            
            llm_data = data.get('llm', {})
            agent_data = data.get('agent_config', {}) or data.get('agent', {})  # Support both for backward compatibility
            logging_data = data.get('logging', {})
            
            mainLogger.info("Loaded configuration from file", path=str(path))
            return cls(
                llm=LLMConfig(**{k: v for k, v in llm_data.items() if k in {f.name for f in fields(LLMConfig)}}),
                agent_config=AgentConfig(**{k: v for k, v in agent_data.items() if k in {f.name for f in fields(AgentConfig)}}),
                logging=LoggingConfig(**{k: v for k, v in logging_data.items() if k in {f.name for f in fields(LoggingConfig)}}),
            )
        except Exception as e:
            mainLogger.error("Failed to load config from file", path=str(path), error=str(e))
            return None
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables"""
        cfg = cls()
        
        # Map 'agent' section in ENV_MAPPING to 'agent_config'
        for field_name, section, type_, env_vars in ENV_MAPPING:
            value = _get_env_value(env_vars, type_)
            if value is not None:
                section_name = 'agent_config' if section == 'agent' else section
                section_obj = getattr(cfg, section_name)
                setattr(section_obj, field_name, value)
        
        return cfg
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration: defaults → file → env
        Priority: defaults < file < env < cli (cli done via merge_with_cli_args)
        """
        # Start with defaults
        cfg = cls.from_defaults()
        
        # Try to load from file
        if config_path:
            file_cfg = cls.from_yaml(config_path)
        else:
            # Try default locations
            file_cfg = None
            for path in [".cfuse.yaml", "~/.cfuse.yaml", "~/.config/cfuse/config.yaml"]:
                file_cfg = cls.from_yaml(path)
                if file_cfg:
                    break
        
        if file_cfg:
            cfg = cls._merge(cfg, file_cfg)
        
        # Merge environment variables
        env_cfg = cls.from_env()
        cfg = cls._merge(cfg, env_cfg)
        
        return cfg
    
    @staticmethod
    def _merge(base: "Config", override: "Config") -> "Config":
        """Merge configs: non-None values in override take precedence"""
        result = copy.deepcopy(base)
        
        # Merge each section
        for section_name in ['llm', 'agent_config', 'logging']:
            base_section = getattr(result, section_name)
            override_section = getattr(override, section_name)
            
            for field in fields(base_section):
                override_value = getattr(override_section, field.name)
                if override_value is not None:
                    setattr(base_section, field.name, override_value)
        
        return result
    
    @classmethod
    def merge_with_cli_args(cls, config: "Config", **cli_args) -> "Config":
        """Merge CLI arguments (highest priority)"""
        result = copy.deepcopy(config)
        
        # Special mapping: CLI 'think' → config 'enable_thinking'
        if cli_args.get('think') is not None:
            result.llm.enable_thinking = cli_args['think']
        
        # Auto-match all other CLI args to config fields by name
        for key, value in cli_args.items():
            if value is None or key == 'think':
                continue
            
            # Try to find matching field in each section
            for section in [result.llm, result.agent_config, result.logging]:
                if hasattr(section, key):
                    setattr(section, key, value)
                    break
        
        return result
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        
        # Check required LLM parameters
        required_llm_fields = [
            ('api_key', 'LLM API key'),
            ('model', 'LLM model'),
            ('base_url', 'LLM base URL'),
        ]
        
        for field_name, display_name in required_llm_fields:
            value = getattr(self.llm, field_name)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                errors.append(
                    f"{display_name} is required. "
                    f"Set it via --{field_name.replace('_', '-')} flag, "
                    f"{field_name.upper().replace('MODEL', 'LLM_MODEL')} environment variable, "
                    f"or config file."
                )
        
        # Check value range validations
        for section_name, field, check, msg in VALIDATIONS:
            value = getattr(getattr(self, section_name), field)
            if value is not None and not check(value):
                errors.append(f"{msg}, got {value}")
        
        return errors
