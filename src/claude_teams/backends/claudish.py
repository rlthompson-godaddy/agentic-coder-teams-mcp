from claude_teams.backends.base import BaseBackend, SpawnRequest


class ClaudishBackend(BaseBackend):
    """Backend adapter for Claudish (multi-provider Claude Code proxy).

    Claudish routes requests to any AI provider using ``provider@model``
    syntax (e.g. ``google@gemini-3-pro``, ``oai@gpt-5.2``).  In
    single-shot mode a ``--model`` flag is required and the prompt is
    passed as a positional argument.
    """

    _name = "claudish"
    _binary_name = "claudish"

    _MODEL_MAP: dict[str, str] = {
        "fast": "google@gemini-2.5-flash",
        "balanced": "oai@gpt-5.2",
        "powerful": "google@gemini-3-pro",
    }

    def supported_models(self) -> list[str]:
        """Return a curated set of Claudish provider@model identifiers.

        Claudish supports any OpenRouter model plus direct provider
        routing; these are common choices.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "google@gemini-2.5-flash",
            "google@gemini-3-pro",
            "oai@gpt-5.2",
            "oai@gpt-5.2-codex",
            "ollama@llama3.2",
        ]

    def default_model(self) -> str:
        """Return the default Claudish model.

        Returns:
            str: Default model identifier for this backend.
        """
        return "oai@gpt-5.2"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic tier or pass through a provider@model string.

        Args:
            generic_name (str): Generic tier or provider@model string.

        Returns:
            str: Claudish model identifier.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Claudish command for single-shot execution.

        Single-shot mode requires ``--model`` and passes the prompt as a
        positional argument.  ``-y`` auto-approves all permission prompts.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts list.
        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        return [
            binary,
            "--model",
            model,
            "-y",
            request.prompt,
        ]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Claudish environment variables (none required).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Empty dict.
        """
        return {}
