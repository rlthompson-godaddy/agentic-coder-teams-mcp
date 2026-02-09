from claude_teams.backends.base import BaseBackend, SpawnRequest


class AiderBackend(BaseBackend):
    """Backend adapter for Aider CLI."""

    _name = "aider"
    _binary_name = "aider"

    _MODEL_MAP: dict[str, str] = {
        "fast": "claude-3.5-haiku",
        "balanced": "claude-sonnet-4",
        "powerful": "claude-opus-4",
    }

    def supported_models(self) -> list[str]:
        """Return supported Aider model names.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "claude-3.5-haiku",
            "claude-sonnet-4",
            "claude-opus-4",
            "gpt-5.3-codex",
            "gemini-2.5-pro",
        ]

    def default_model(self) -> str:
        """Return the default Aider model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "claude-sonnet-4"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to an Aider model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name: Generic tier or direct model name.

        Returns:
            Aider model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Aider CLI command.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Command parts list.
        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        return [
            binary,
            "--model",
            model,
            "--message",
            request.prompt,
            "--yes-always",
        ]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Aider environment variables (none required).

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Empty dict.
        """
        return {}
