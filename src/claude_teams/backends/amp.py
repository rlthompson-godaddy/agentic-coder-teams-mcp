from claude_teams.backends.base import BaseBackend, SpawnRequest


class AmpBackend(BaseBackend):
    """Backend adapter for Amp CLI (Sourcegraph).

    Amp uses an agent *mode* system (``-m free|rush|smart``) rather than
    direct model names.  Generic tiers are mapped to modes; direct model
    names are not supported via CLI flags.
    """

    _name = "amp"
    _binary_name = "amp-cli"

    _MODE_MAP: dict[str, str] = {
        "fast": "rush",
        "balanced": "smart",
        "powerful": "smart",
    }

    def supported_models(self) -> list[str]:
        """Return Amp agent modes presented as model choices.

        Amp does not accept model names directly; it uses agent modes that
        control model, system prompt, and tool selection together.

        Returns:
            list[str]: Curated list of supported model identifiers.
        """
        return [
            "rush",
            "smart",
            "free",
        ]

    def default_model(self) -> str:
        """Return the default Amp agent mode.

        Returns:
            str: Default model identifier for this backend.
        """
        return "smart"

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic tier to an Amp agent mode.

        Allows pass-through for unrecognized names (e.g. direct mode names).

        Args:
            generic_name (str): Generic tier or direct mode name.

        Returns:
            str: Amp agent mode identifier.
        """
        if generic_name in self._MODE_MAP:
            return self._MODE_MAP[generic_name]
        return generic_name

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the Amp CLI command for non-interactive execution.

        Uses ``-x`` (execute mode) which prints only the last assistant
        message and exits.  ``--dangerously-allow-all`` bypasses all
        confirmation prompts.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts list.
        """
        binary = self.discover_binary()
        mode = self.resolve_model(request.model)

        cmd = [
            binary,
            "-x",
            request.prompt,
            "--dangerously-allow-all",
        ]
        if mode in ("free", "rush", "smart"):
            cmd.extend(["-m", mode])
        return cmd

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return Amp environment variables (none required).

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Empty dict.
        """
        return {}
