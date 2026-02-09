from claude_teams.backends.base import BaseBackend, SpawnRequest


class VibeBackend(BaseBackend):
    """Backend adapter for Mistral Vibe CLI.

    Vibe does not expose a ``--model`` CLI flag; model selection is handled
    internally by the Vibe runtime.  The ``resolve_model`` method still maps
    generic tiers to Mistral model names for informational purposes, but
    ``build_command`` does not emit a model argument.
    """

    _name = "vibe"
    _binary_name = "vibe"

    _MODEL_MAP: dict[str, str] = {
        "fast": "devstral-small",
        "balanced": "devstral-2",
        "powerful": "devstral-2",
    }

    def supported_models(self) -> list[str]:
        """Return model aliases known to the Vibe CLI.

        Vibe manages model selection via ``~/.vibe/config.toml`` rather than
        CLI flags.  These are the well-known model aliases it ships with.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "devstral-2",
            "devstral-small",
        ]

    def default_model(self) -> str:
        """Return the default Vibe model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "devstral-2"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic or direct model name to a Mistral model identifier.

        Allows pass-through for unrecognized model names.

        Args:
            generic_name (str): Generic tier or direct model name.

        Returns:
            str: Mistral model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Vibe CLI command for programmatic execution.

        The ``-p`` flag activates programmatic mode which auto-approves all
        tool calls, outputs the response, and exits.  Vibe does not accept a
        ``--model`` flag so model selection is omitted.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts list.
        """
        binary = self.discover_binary()
        return [
            binary,
            "-p",
            request.prompt,
            "--output",
            "text",
        ]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Vibe environment variables (none required).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Empty dict.
        """
        return {}
