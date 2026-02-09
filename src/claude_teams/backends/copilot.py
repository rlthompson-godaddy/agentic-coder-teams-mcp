from claude_teams.backends.base import BaseBackend, SpawnRequest


class CopilotBackend(BaseBackend):
    """Backend adapter for GitHub Copilot CLI."""

    _name = "copilot"
    _binary_name = "copilot"

    _MODEL_MAP: dict[str, str] = {
        "fast": "claude-haiku-4.5",
        "balanced": "claude-sonnet-4.5",
        "powerful": "claude-opus-4.6",
        "claude-sonnet-4.5": "claude-sonnet-4.5",
        "claude-haiku-4.5": "claude-haiku-4.5",
        "claude-opus-4.6": "claude-opus-4.6",
        "claude-opus-4.6-fast": "claude-opus-4.6-fast",
        "claude-opus-4.5": "claude-opus-4.5",
        "claude-sonnet-4": "claude-sonnet-4",
        "gpt-5.2-codex": "gpt-5.2-codex",
        "gpt-5.2": "gpt-5.2",
        "gpt-5.1-codex-max": "gpt-5.1-codex-max",
        "gpt-5.1-codex": "gpt-5.1-codex",
        "gpt-5.1": "gpt-5.1",
        "gpt-5": "gpt-5",
        "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-4.1": "gpt-4.1",
        "gemini-3-pro-preview": "gemini-3-pro-preview",
    }

    def supported_models(self) -> list[str]:
        """Return supported Copilot CLI model names.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "claude-sonnet-4.5",
            "claude-haiku-4.5",
            "claude-opus-4.6",
            "claude-opus-4.5",
            "claude-sonnet-4",
            "gpt-5.2-codex",
            "gpt-5.2",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-4.1",
            "gemini-3-pro-preview",
        ]

    def default_model(self) -> str:
        """Return the default Copilot CLI model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "claude-sonnet-4.5"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Copilot CLI model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name (str): Generic tier or direct model name.

        Returns:
            str: Copilot CLI model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Copilot CLI command for non-interactive prompt execution.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts list.
        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        return [
            binary,
            "-p",
            request.prompt,
            "--model",
            model,
            "--yolo",
            "--no-ask-user",
        ]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Copilot CLI environment variables (none required).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Empty dict.
        """
        return {}
