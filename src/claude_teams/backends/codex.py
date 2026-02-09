from claude_teams.backends.base import BaseBackend, SpawnRequest


class CodexBackend(BaseBackend):
    """Backend adapter for OpenAI Codex CLI."""

    _name = "codex"
    _binary_name = "codex"

    _MODEL_MAP: dict[str, str] = {
        "fast": "gpt-5.1-codex-mini",
        "balanced": "gpt-5.3-codex",
        "powerful": "gpt-5.1-codex-max",
        "gpt-5.3-codex": "gpt-5.3-codex",
        "gpt-5.1-codex-max": "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
    }

    def supported_models(self) -> list[str]:
        """Return supported Codex model names.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return ["gpt-5.3-codex", "gpt-5.1-codex-max", "gpt-5.1-codex-mini"]

    def default_model(self) -> str:
        """Return the default Codex model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "gpt-5.3-codex"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Codex model.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name: Generic tier or direct model name.

        Returns:
            Codex model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Codex CLI command.

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Command parts list.
        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        cmd = [
            binary,
            "exec",
            "--model",
            model,
            "--full-auto",
            "-C",
            request.cwd,
        ]

        output_path = (request.extra or {}).get("output_last_message_path")
        if output_path:
            cmd.extend(["--output-last-message", output_path])

        cmd.append(request.prompt)
        return cmd

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Codex environment variables (none required).

        Args:
            request: Backend-agnostic spawn parameters.

        Returns:
            Empty dict.
        """
        return {}
