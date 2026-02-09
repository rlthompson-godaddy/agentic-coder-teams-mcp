from claude_teams.backends.base import BaseBackend, SpawnRequest


class QwenBackend(BaseBackend):
    """Backend adapter for Qwen Code CLI."""

    _name = "qwen"
    _binary_name = "qwen"

    _MODEL_MAP: dict[str, str] = {
        "fast": "qwen-turbo",
        "balanced": "qwen-plus",
        "powerful": "qwen-max",
    }

    def supported_models(self) -> list[str]:
        """Return supported Qwen Code model names.

        Qwen Code supports any model via ``-m``; these are the common ones.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "claude-sonnet-4.5",
            "claude-opus-4.6",
            "gpt-5.2-codex",
            "gemini-2.5-pro",
        ]

    def default_model(self) -> str:
        """Return the default Qwen Code model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "qwen-plus"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Qwen Code model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name (str): Generic tier or direct model name.

        Returns:
            str: Qwen Code model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Qwen Code CLI command for non-interactive execution.

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
        """Return Qwen Code environment variables (none required).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Empty dict.
        """
        return {}
