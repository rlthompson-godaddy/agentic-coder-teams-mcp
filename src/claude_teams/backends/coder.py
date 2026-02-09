from claude_teams.backends.base import BaseBackend, SpawnRequest


class CoderBackend(BaseBackend):
    """Backend adapter for Just Every Code CLI (Codex fork)."""

    _name = "coder"
    _binary_name = "coder"

    _MODEL_MAP: dict[str, str] = {
        "fast": "claude-haiku-4.5",
        "balanced": "claude-sonnet-4.5",
        "powerful": "claude-opus-4.6",
    }

    def supported_models(self) -> list[str]:
        """Return supported Coder model names.

        Coder supports any model via ``-m``; these are the common ones.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "claude-haiku-4.5",
            "claude-sonnet-4.5",
            "claude-opus-4.6",
            "gpt-5.2-codex",
            "gpt-5.2",
            "o3",
        ]

    def default_model(self) -> str:
        """Return the default Coder model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "claude-sonnet-4.5"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Coder model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name (str): Generic tier or direct model name.

        Returns:
            str: Coder model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Coder CLI command for non-interactive execution.

        Uses the ``exec`` subcommand with ``--full-auto`` for low-friction
        sandboxed automatic execution.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts list.
        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        return [
            binary,
            "exec",
            "-m",
            model,
            "--full-auto",
            request.prompt,
        ]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Coder environment variables (none required).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Empty dict.
        """
        return {}
