from claude_teams.backends.base import BaseBackend, SpawnRequest


class OpenCodeBackend(BaseBackend):
    """Backend adapter for OpenCode CLI."""

    _name = "opencode"
    _binary_name = "opencode"

    _MODEL_MAP: dict[str, str] = {
        "fast": "anthropic/claude-haiku-3.5",
        "balanced": "anthropic/claude-sonnet-4",
        "powerful": "anthropic/claude-opus-4",
    }

    def supported_models(self) -> list[str]:
        """Return supported OpenCode model names.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "anthropic/claude-sonnet-4",
            "anthropic/claude-opus-4",
            "openai/gpt-5.3-codex",
            "google/gemini-2.5-pro",
        ]

    def default_model(self) -> str:
        """Return the default OpenCode model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "anthropic/claude-sonnet-4"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to an OpenCode model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name: Generic tier or direct model name.

        Returns:
            OpenCode model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the OpenCode CLI command.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Command parts list.
        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        return [
            binary,
            "run",
            "--model",
            model,
            request.prompt,
        ]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return OpenCode environment variables (none required).

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Empty dict.
        """
        return {}
