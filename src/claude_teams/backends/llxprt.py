from claude_teams.backends.base import BaseBackend, SpawnRequest


class LlxprtBackend(BaseBackend):
    """Backend adapter for LLxprt Code CLI."""

    _name = "llxprt"
    _binary_name = "llxprt"

    _MODEL_MAP: dict[str, str] = {
        "fast": "claude-haiku-4.5",
        "balanced": "claude-sonnet-4.5",
        "powerful": "claude-opus-4.6",
    }

    def supported_models(self) -> list[str]:
        """Return supported LLxprt model names.

        LLxprt supports any model via ``-m`` and ``--provider``; these
        are the common ones.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "claude-haiku-4.5",
            "claude-sonnet-4.5",
            "claude-opus-4.6",
        ]

    def default_model(self) -> str:
        """Return the default LLxprt model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "claude-sonnet-4.5"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to an LLxprt model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name (str): Generic tier or direct model name.

        Returns:
            str: LLxprt model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the LLxprt CLI command for non-interactive execution.

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
            "-m",
            model,
            "-y",
        ]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return LLxprt environment variables (none required).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Empty dict.
        """
        return {}
