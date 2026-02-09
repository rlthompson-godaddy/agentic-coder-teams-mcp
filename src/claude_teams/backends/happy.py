from claude_teams.backends.base import BaseBackend, SpawnRequest


class HappyBackend(BaseBackend):
    """Backend adapter for Happy (Claude Code On the Go).

    Happy wraps Claude Code with mobile control capabilities.  It
    supports all standard Claude Code flags including ``--print``,
    ``--model``, and ``--yolo``.
    """

    _name = "happy"
    _binary_name = "happy"

    _MODEL_MAP: dict[str, str] = {
        "fast": "haiku",
        "balanced": "sonnet",
        "powerful": "opus",
    }

    def supported_models(self) -> list[str]:
        """Return supported Happy / Claude model aliases.

        Happy passes ``--model`` through to Claude Code, which accepts
        both aliases (``sonnet``, ``opus``) and full model names.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "haiku",
            "sonnet",
            "opus",
        ]

    def default_model(self) -> str:
        """Return the default Happy model alias.

        Returns:
            str: Default model identifier for this backend.
        """
        return "sonnet"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic tier to a Claude model alias.

        Allows pass-through for full model names or unknown aliases.

        Args:
            generic_name (str): Generic tier or model name/alias.

        Returns:
            str: Claude model alias or name.
        """
        if generic_name in self._MODEL_MAP:
            return self._MODEL_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Happy CLI command for non-interactive execution.

        ``--print`` activates non-interactive mode (print response and
        exit).  ``--yolo`` bypasses all permission prompts.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts list.
        """
        binary = self.discover_binary()
        model = self.resolve_model(request.model)
        return [
            binary,
            "--print",
            "--model",
            model,
            "--yolo",
            request.prompt,
        ]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Happy environment variables (none required).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Empty dict.
        """
        return {}
