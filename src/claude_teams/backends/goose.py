from claude_teams.backends.base import BaseBackend, SpawnRequest


class GooseBackend(BaseBackend):
    """Backend adapter for Goose CLI (Block)."""

    _name = "goose"
    _binary_name = "goose"

    _MODEL_MAP: dict[str, str] = {
        "fast": "claude-haiku-4.5",
        "balanced": "claude-sonnet-4.5",
        "powerful": "claude-opus-4.6",
    }

    # Goose uses provider:model pairs. Default to anthropic provider.
    _PROVIDER_MAP: dict[str, str] = {
        "fast": "anthropic",
        "balanced": "anthropic",
        "powerful": "anthropic",
    }

    def supported_models(self) -> list[str]:
        """Return supported Goose model names.

        Goose supports any provider/model combo; these are the common ones.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "claude-haiku-4.5",
            "claude-sonnet-4.5",
            "claude-opus-4.6",
            "gpt-5.2-codex",
            "gemini-2.5-pro",
        ]

    def default_model(self) -> str:
        """Return the default Goose model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "claude-sonnet-4.5"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Goose model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name (str): Generic tier or direct model name.

        Returns:
            str: Goose model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def _resolve_provider(self, generic_name: str) -> str | None:
        """Resolve the provider for a generic model tier.

        Args:
            generic_name (str): Generic tier or direct model name.

        Returns:
            str | None: Provider name, or None if not a generic tier.
        """
        return self._PROVIDER_MAP.get(generic_name)

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Goose CLI command for non-interactive execution.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts list.
        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        provider = self._resolve_provider(request.model)

        cmd = [
            binary,
            "run",
            "-t",
            request.prompt,
            "--model",
            model,
            "--no-session",
        ]
        if provider:
            cmd.extend(["--provider", provider])
        return cmd

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Goose environment variables (none required).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Empty dict.
        """
        return {}
