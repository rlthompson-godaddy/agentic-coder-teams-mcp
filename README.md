<div align="center">

# agentic-coder-teams-mcp

Multi-backend MCP server for orchestrating teams of agentic coding agents.

</div>

https://github.com/user-attachments/assets/531ada0a-6c36-45cd-8144-a092bb9f9a19

## What is this?

Claude Code has a built-in [agent teams](https://code.claude.com/docs/en/agent-teams) feature that lets multiple Claude Code instances coordinate as a team with shared task lists, inter-agent messaging, and tmux-based spawning. But the protocol is internal and tightly coupled to Claude Code's own tooling.

This MCP server reimplements that protocol as a standalone [Model Context Protocol](https://modelcontextprotocol.io/) server with one major addition: **pluggable backend support for 17 agentic coding CLIs**. Any MCP client can use it to spawn and coordinate heterogeneous teams of coding agents across different tools and providers.

## Supported backends

The server auto-discovers which backends are available based on binaries found on your `PATH`:

| Backend | CLI binary | Description |
|---------|-----------|-------------|
| `claude-code` | `claude` | [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (default) |
| `codex` | `codex` | [OpenAI Codex CLI](https://github.com/openai/codex) |
| `gemini` | `gemini` | [Gemini CLI](https://github.com/google-gemini/gemini-cli) |
| `opencode` | `opencode` | [OpenCode](https://opencode.ai) (multi-provider) |
| `aider` | `aider` | [Aider](https://aider.chat) |
| `copilot` | `copilot` | [GitHub Copilot CLI](https://github.com/github/copilot-cli) |
| `auggie` | `auggie` | [Augment Code](https://www.augmentcode.com/) |
| `goose` | `goose` | [Goose](https://github.com/block/goose) |
| `qwen` | `qwen` | [Qwen Chat CLI](https://github.com/QwenLM) |
| `vibe` | `vibe` | [Vibe](https://github.com/thevibe-ai/vibe) |
| `kimi` | `kimi` | [Kimi CLI](https://kimi.ai) |
| `amp` | `amp` | [Amp](https://amp.dev) |
| `rovodev` | `rovodev` | [Rovo Dev](https://www.atlassian.com/software/rovo) |
| `llxprt` | `llxprt` | [LLXpert](https://llxpert.ai) |
| `coder` | `coder` | [Coder](https://coder.com) |
| `claudish` | `claudish` | [Claudish](https://github.com/claudish-dev/claudish) (multi-provider) |
| `happy` | `happy` | [Happy](https://happy.dev) |

Third-party backends can register via Python [entry points](https://packaging.python.org/en/latest/specifications/entry-points/) using the `claude_teams.backends` group.

## Install

### Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "claude-teams": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/rlthompson-godaddy/agentic-coder-teams-mcp", "claude-teams"]
    }
  }
}
```

### OpenCode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "claude-teams": {
      "type": "local",
      "command": ["uvx", "--from", "git+https://github.com/rlthompson-godaddy/agentic-coder-teams-mcp", "claude-teams"],
      "enabled": true
    }
  }
}
```

### Any MCP client

The server speaks standard MCP over stdio. Point your client at:

```
uvx --from git+https://github.com/rlthompson-godaddy/agentic-coder-teams-mcp claude-teams
```

## Requirements

- **Python 3.12+**
- **[tmux](https://github.com/tmux/tmux)** (agents spawn in tmux panes)
- At least one supported agentic CLI on your `PATH` (e.g., `claude`, `codex`, `gemini`)

## MCP tools

### Team management

| Tool | Description |
|------|-------------|
| `team_create` | Create a new agent team with a name and description. One team per server session. |
| `team_delete` | Delete a team and all its data. Fails if teammates are still active. |
| `read_config` | Read team configuration and member list. |
| `list_backends` | List all available backends and their supported models. |

### Agent lifecycle

| Tool | Description |
|------|-------------|
| `spawn_teammate` | Spawn a coding agent in a tmux pane. Specify the backend, model, and prompt. |
| `health_check` | Check if a spawned agent's tmux pane is still alive. |
| `force_kill_teammate` | Forcibly kill a teammate's tmux pane and remove from team. |
| `process_shutdown_approved` | Cleanly remove a teammate after graceful shutdown approval. |

### Messaging

| Tool | Description |
|------|-------------|
| `send_message` | Send direct messages, broadcasts, or shutdown/plan-approval responses. |
| `read_inbox` | Read messages from an agent's inbox (with optional unread-only filter). |
| `poll_inbox` | Long-poll an inbox for new messages (blocks up to 30 seconds). |

### Task tracking

| Tool | Description |
|------|-------------|
| `task_create` | Create a new task with auto-incrementing ID. |
| `task_update` | Update task status, owner, dependencies, or metadata. |
| `task_list` | List all tasks for a team. |
| `task_get` | Get full details of a specific task. |

## CLI

The package also provides a `claude-teams` CLI built with [Typer](https://typer.tiangolo.com/) for inspecting and managing teams from the terminal:

```
claude-teams serve       # Start the MCP server
claude-teams backends    # List available backends
claude-teams config TEAM # Show team config
claude-teams status TEAM # Show member status table
claude-teams inbox TEAM  # Read an agent's inbox
claude-teams health TEAM # Health-check all agents
claude-teams kill TEAM   # Kill a specific agent
```

All commands support `--json` for machine-readable output.

## How it works

### Spawning

Teammates launch as separate processes in tmux panes via `tmux split-window`. Each agent gets:
- A unique agent ID (`name@team`)
- An assigned color from a rotating palette
- Backend-specific CLI flags and environment variables
- Its initial prompt delivered to its inbox

### Messaging

JSON-based inboxes stored under `~/.claude/teams/<team>/inboxes/`. File locking via `fcntl.flock()` prevents corruption from concurrent reads and writes. Supports direct messages, broadcasts, and structured messages for shutdown approval and plan review.

### Task tracking

JSON task files stored under `~/.claude/tasks/<team>/`. Tasks support:
- Status progression: `pending` -> `in_progress` -> `completed`
- Ownership assignment to specific agents
- Dependency graphs (`blocks` / `blockedBy`) with cycle detection
- Arbitrary metadata

### Concurrency safety

- **Config writes**: Atomic via `tempfile.mkstemp()` + `os.replace()` to prevent partial reads
- **Inbox operations**: Guarded by `fcntl.flock()` file locks
- **Task operations**: Guarded by `fcntl.flock()` file locks with validation-then-write phasing

### Backend architecture

Backends implement a `Backend` protocol providing:
- **Lifecycle**: `spawn`, `health_check`, `kill`, `graceful_shutdown`
- **Interactivity**: `capture`, `send`, `wait_idle`, `execute_in_pane`
- **Model resolution**: Map generic tiers (`fast`, `balanced`, `powerful`) to backend-specific model IDs

A `BaseBackend` class provides shared tmux lifecycle management via [`claude-code-tools`](https://pypi.org/project/claude-code-tools/). Concrete backends only need to implement `build_command`, `build_env`, and model resolution.

## Storage layout

```
~/.claude/
├── teams/<team-name>/
│   ├── config.json          # Team config + member list
│   └── inboxes/
│       ├── team-lead.json   # Lead agent inbox
│       ├── worker-1.json    # Teammate inboxes
│       └── .lock
└── tasks/<team-name>/
    ├── 1.json               # Task files (auto-incrementing IDs)
    ├── 2.json
    └── .lock
```

## Development

### Setup

```bash
git clone https://github.com/rlthompson-godaddy/agentic-coder-teams-mcp.git
cd agentic-coder-teams-mcp
uv sync
```

### Running tests

```bash
uv run pytest                           # Run all 504 tests
uv run pytest --cov=claude_teams        # With coverage (94%)
uv run pytest tests/test_tasks.py -v    # Single module
```

### Linting and type checking

```bash
uv run ruff format                      # Format
uv run ruff check                       # Lint
uv run ty check                         # Type check (Astral's ty)
```

### Adding a backend

1. Create `src/claude_teams/backends/your_backend.py` inheriting from `BaseBackend`
2. Implement `build_command()`, `build_env()`, `supported_models()`, `default_model()`, and `resolve_model()`
3. Add the entry to `_BUILTIN_BACKENDS` in `registry.py`
4. Add tests in `tests/test_backends/test_your_backend.py`

Or register externally via the `claude_teams.backends` entry point group in your package's `pyproject.toml`.

## Acknowledgments

This project stands on the shoulders of giants. The original implementation and protocol reverse-engineering was done by [Victor](https://github.com/cs50victor) in [claude-code-teams-mcp](https://github.com/cs50victor/claude-code-teams-mcp), based on his [deep dive into Claude Code's internals](https://gist.github.com/cs50victor/0a7081e6824c135b4bdc28b566e1c719). His work cracking open the agent teams protocol and building the first standalone MCP server for it made everything here possible. Thank you, Victor.

## License

[MIT](./LICENSE)
