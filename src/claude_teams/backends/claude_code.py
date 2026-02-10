from claude_teams.backends.base import BaseBackend, SpawnRequest


class ClaudeCodeBackend(BaseBackend):
    """Backend adapter for Claude Code CLI."""

    _name = "claude-code"
    _binary_name = "claude"

    @property
    def is_interactive(self) -> bool:
        """Claude Code runs as an interactive MCP client with native team messaging.

        Returns:
            bool: Always True.
        """
        return True

    _MODEL_MAP: dict[str, str] = {
        "fast": "haiku",
        "balanced": "sonnet",
        "powerful": "opus",
        "haiku": "haiku",
        "sonnet": "sonnet",
        "opus": "opus",
    }

    def supported_models(self) -> list[str]:
        """Return supported Claude Code model short-names.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return ["haiku", "sonnet", "opus"]

    def default_model(self) -> str:
        """Return the default Claude Code model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "sonnet"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Claude Code model.

        Args:
            generic_name: Generic tier or direct model name.

        Returns:
            Claude Code model identifier.

        Raises:
            ValueError: For unsupported model names.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        raise ValueError(
            f"Unsupported model {generic_name!r} for claude-code. "
            f"Supported: {', '.join(self.supported_models())}"
        )

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Claude Code CLI command.

        Produces the exact same flags as the original spawner.py build_spawn_command().

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Command parts list.
        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        cmd = [
            binary,
            "--agent-id",
            request.agent_id,
            "--agent-name",
            request.name,
            "--team-name",
            request.team_name,
            "--agent-color",
            request.color,
            "--parent-session-id",
            request.lead_session_id,
            "--agent-type",
            request.agent_type,
            "--model",
            model,
        ]
        if request.plan_mode_required:
            cmd.append("--plan-mode-required")
        return cmd

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Claude Code environment variables.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Dict with CLAUDECODE and CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS.
        """
        return {
            "CLAUDECODE": "1",
            "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
        }
