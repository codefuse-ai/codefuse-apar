![](./images/codefuse_logo.png)
# 🚀 CodeFuse-Agent (CFuse)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**A lightweight, cleanly-architected agent framework designed for research and experimentation.**

CodeFuse-Agent is fully open-source and can be installed with a single `pip install` command, providing a complete yet minimal toolset for code-related tasks. We open-source CFuse to facilitate reproducible research and encourage further exploration of LLM-based coding agents.

## 🏆 SWE-bench Lite Results

| Configuration | Resolved |
|---------------|----------|
| CFuse + Claude Sonnet 4.5 (Single Attempt) | **61%** |
| CFuse + Trajectory-Aware Test-Time Scaling | **61.67%** |

We introduce **Trajectory-Aware Test-Time Scaling (TTS)**, a novel verification mechanism that aggregates self-generated test cases from multiple trajectories for cross-validation, achieving state-of-the-art results on SWE-bench Lite.

📄 **Technical Report**: [tech_report.md](tech_report.md)

## ✨ Features

### Configurable Agent Profiles

Agent behavior is defined through declarative Markdown profiles (system prompt, tools, model, etc.), enabling quick switching of system prompts and tool subsets without code changes.

### Dual Execution Modes

- **Local Mode**: Execute tool calls directly in the local environment
- **HTTP Mode**: Serve as a tool execution backend or delegate calls to remote sandboxes

This decoupling of agent decisions from environment execution makes CFuse suitable as scaffolding for RL training pipelines.

### Built-in Tools

Six essential tools for code exploration and modification:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with optional line range selection |
| `write_file` | Create or overwrite files |
| `edit_file` | Perform edits via search-and-replace |
| `grep` | Fast code search powered by ripgrep |
| `glob` | File discovery using glob patterns |
| `bash` | Execute shell commands with timeout control |

## 🏗️ Architecture

| Layer | Responsibility |
|-------|----------------|
| **Interaction** | Terminal UI / Headless / HTTP modes |
| **Agent Loop** | Core lifecycle: LLM interaction, tool dispatch, iteration control |
| **Context Engine** | Message history, environment context, compression, prompt assembly |
| **LLM Provider** | OpenAI-compatible API support |
| **Tool Execution** | 6 built-in tools + remote execution |
| **Observability** | Trajectory logs, execution metrics, cost tracking |

## 📦 Installation

```bash
pip install -e .
```

## 🔑 Configuration

### Required Environment Variables

CodeFuse-Agent requires three environment variables to be configured:

```bash
# Required: Your OpenAI API key (or compatible API key)
export OPENAI_API_KEY=your-api-key

# Required: The LLM model to use
export LLM_MODEL=gpt-4o

# Required: The API base URL
export LLM_BASE_URL=https://api.openai.com/v1
```

**Important Notes:**
- All three environment variables are **required** for the agent to function
- `OPENAI_API_KEY` is the only API key variable used 
- `LLM_BASE_URL` can be set to any OpenAI-compatible API endpoint
- `LLM_MODEL` should match the model name available on your API endpoint

### Configuration File (Optional)

You can optionally create a `.cfuse.yaml` configuration file in your project root or `~/.cfuse.yaml`:

```yaml
llm:
  provider: openai_compatible
  model: ${LLM_MODEL}           # Uses environment variable
  api_key: ${OPENAI_API_KEY}    # Uses environment variable
  base_url: ${LLM_BASE_URL}     # Uses environment variable
  temperature: 0.0
  max_tokens: null
  timeout: 60

agent_config:
  max_iterations: 200
  max_context_tokens: 128000
  enable_tools: true
  yolo: false
  agent: default
  workspace_root: .
  bash_timeout: 30

logging:
  logs_dir: ~/.cfuse/logs
  verbose: false
```

**Configuration Priority** (highest to lowest):
1. CLI arguments (`--model`, `--api-key`, `--base-url`, etc.)
2. Environment variables (`OPENAI_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`)
3. Configuration file (`.cfuse.yaml`)
4. Default values

## 🚀 Quick Start

### Interactive Mode

```bash
# Basic startup
cfuse

# Enable YOLO mode (auto-confirm all tool calls)
cfuse --yolo

# Start with specific workspace
cfuse --workspace-root /path/to/project
```

### Headless Mode

```bash
# Single query
cfuse -p "Read README.md and summarize it"

# Auto-execute without confirmation
cfuse -p "Analyze project structure" --yolo

# Complex task with more iterations
cfuse -p "Refactor the auth module" --yolo --max-iterations 300
```

### Using Different Models

```bash
# Use specific model
cfuse --model gpt-4o --api-key sk-xxx --base-url https://api.openai.com/v1

# Use local model (LM Studio, Ollama, etc.)
cfuse --model llama3 --api-key dummy --base-url http://localhost:1234/v1

# Adjust temperature (0.0 = deterministic, higher = creative)
cfuse -p "Fix bug in auth.py" --temperature 0.0 --yolo
```

### Logging and Debugging

Logs include `main.log`, `trajectory/`, and `llm_messages/` in `~/.cfuse/logs`

```bash
# Enable verbose logging
cfuse -p "Your task" --verbose --yolo

# Custom log directory
cfuse -p "Your task" --logs-dir ./my_logs --yolo
```

### Common Usage Patterns

```bash
# Bug fixing with verbose logs
cfuse -p "Fix the authentication bug" --workspace-root ./backend --verbose --yolo

# Code review with low temperature
cfuse -p "Review src/utils/parser.py" --temperature 0.1

# Long-running refactoring task
cfuse -p "Refactor database layer" --max-iterations 500 --yolo --logs-dir ./refactor_logs
```

## ⚙️ CLI Options

### Main Options

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --prompt TEXT` | User query (headless mode). If omitted, launches interactive mode. | `None` |
| `--yolo` | Auto-confirm all tool calls without prompting | `False` |
| `--workspace-root PATH` | Working directory for the agent | `.` |
| `--agent TEXT` | Agent profile (`default`, `swe`, or path to `.md` file) | `default` |
| `--max-iterations INT` | Maximum agent loop iterations | `200` |

### Model Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--model TEXT` | LLM model name | `$LLM_MODEL` |
| `--api-key TEXT` | API key for authentication | `$OPENAI_API_KEY` |
| `--base-url TEXT` | API base URL | `$LLM_BASE_URL` |
| `--temperature FLOAT` | Model temperature (0.0-2.0, lower = more deterministic) | `0.0` |
| `--max-tokens INT` | Maximum tokens in response | `null` |
| `--timeout INT` | API request timeout (seconds) | `60` |

### Logging

| Option | Description | Default |
|--------|-------------|---------|
| `--logs-dir PATH` | Log directory path | `~/.cfuse/logs` |
| `-v, --verbose` | Enable verbose logging | `False` |
| `--stream / --no-stream` | Enable/disable streaming output | `True` |

### Other Options

| Option | Description |
|--------|-------------|
| `--config PATH` | Path to YAML configuration file |
| `--bash-timeout INT` | Timeout for bash commands (seconds, default: 30) |
| `--max-context-tokens INT` | Maximum context window size (default: 128000) |
| `--enable-tools / --no-tools` | Enable/disable tool execution |
| `--http` | Launch HTTP server mode |
| `--http-port INT` | HTTP server port (default: 8000) |
| `--help` | Show help message |

**Configuration Priority:** CLI args > Environment variables > Config file > Defaults

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
