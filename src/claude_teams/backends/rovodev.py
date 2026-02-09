from claude_teams.backends.base import BaseBackend, SpawnRequest


class RovoDevBackend(BaseBackend):
    """Backend adapter for Atlassian Rovo Dev CLI.

    Rovo Dev is invoked via ``acli rovodev run``.  It does not expose a
    ``--model`` CLI flag; model selection is handled via the config file
    at ``~/.rovodev/config.yml``.
    """

    _name = "rovodev"
    _binary_name = "acli"

    _MODEL_MAP: dict[str, str] = {
        "fast": "gpt-5-mini-2025-08-07",
        "balanced": "gpt-5-2025-08-07",
        "powerful": "claude-opus-4-20250918",
    }

    def supported_models(self) -> list[str]:
        """Return known Rovo Dev model identifiers.

        Model selection is managed via ``~/.rovodev/config.yml``
        (``agent.modelId``) rather than CLI flags.  These are the
        well-known model IDs accepted by the config.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "gpt-5-2025-08-07",
            "gpt-5-mini-2025-08-07",
            "claude-opus-4-20250918",
            "claude-sonnet-4-20250514",
        ]

    def default_model(self) -> str:
        """Return the default Rovo Dev model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "gpt-5-2025-08-07"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic tier to a Rovo Dev model identifier.

        Rovo Dev does not accept a model via CLI flags, but the resolved
        name is used for informational / config-management purposes.
        Allows pass-through for unrecognized model names.

        Args:
            generic_name (str): Generic tier or direct model name.

        Returns:
            str: Rovo Dev model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Rovo Dev CLI command for non-interactive execution.

        The prompt is passed as a positional argument to
        ``acli rovodev run``.  ``--yolo`` auto-approves all file and
        bash operations.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts list.
        """
        binary = self.discover_binary()
        return [
            binary,
            "rovodev",
            "run",
            "--yolo",
            request.prompt,
        ]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Rovo Dev environment variables (none required).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Empty dict.
        """
        return {}
