import re
import shlex
import shutil
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast, runtime_checkable

from claude_code_tools.tmux_cli_controller import TmuxCLIController


class CaptureResult(TypedDict):
    """Result of executing a command in a tmux pane."""

    output: str
    exit_code: int


_SAFE_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class SpawnRequest:
    """Backend-agnostic spawn parameters."""

    agent_id: str
    name: str
    team_name: str
    prompt: str
    model: str
    agent_type: str
    color: str
    cwd: str
    lead_session_id: str
    plan_mode_required: bool = False
    extra: dict[str, str] | None = None


@dataclass(frozen=True)
class SpawnResult:
    """What a backend returns after spawning."""

    process_handle: str
    backend_type: str


@dataclass(frozen=True)
class HealthStatus:
    """Health check result."""

    alive: bool
    detail: str = ""


@runtime_checkable
class Backend(Protocol):
    """Protocol that all spawner backends must satisfy.

    Backends provide lifecycle management (spawn, health_check, kill,
    graceful_shutdown) and interactivity (capture, send, wait_idle,
    execute_in_pane) for agent processes running in tmux panes.
    """

    @property
    def name(self) -> str:
        """Unique backend identifier. E.g., 'claude-code', 'codex'.

        Returns:
            str: Backend name such as 'claude-code' or 'codex'.
        """
        ...

    @property
    def is_interactive(self) -> bool:
        """Whether this backend supports native team messaging.

        Interactive backends (e.g., claude-code) run as long-lived processes
        that can send messages to the team-lead inbox directly via the MCP
        messaging protocol.  Non-interactive (one-shot) backends run a
        command and exit; their output must be relayed by the server.

        Returns:
            bool: True if the backend handles its own messaging.
        """
        ...

    def retain_pane_after_exit(self, handle: str) -> None:
        """Keep the process pane alive after the command exits.

        Required for non-interactive backends so the server can capture
        pane output after the process completes.  Without this, tmux
        destroys the pane on exit and the output is lost.

        Args:
            handle (str): Backend-specific process handle (tmux pane ID).
        """
        ...

    @property
    def binary_name(self) -> str:
        """Name of the CLI binary. E.g., 'claude', 'codex', 'gemini'.

        Returns:
            str: Name of the CLI binary executable.
        """
        ...

    def is_available(self) -> bool:
        """Return True if the backend binary is found on PATH.

        Returns:
            bool: True if binary is available, False otherwise.
        """
        ...

    def discover_binary(self) -> str:
        """Return the full path to the backend binary.

        Raises:
            FileNotFoundError: If binary is not found on PATH.
        """
        ...

    def supported_models(self) -> list[str]:
        """Return a curated list of well-known model names for this backend.

        Note:
            This is a static, representative set — not an exhaustive list.
            The actual models available at runtime depend on authentication
            state, account tier, and which providers the user has configured.
            Multi-provider backends (e.g. Claudish, OpenCode) may support
            hundreds of models across all authenticated providers.

        Returns:
            list[str]: Curated list of supported model identifiers.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        ...

    def default_model(self) -> str:
        """Return the default model short-name for this backend.

        Returns:
            str: Default model identifier for this backend.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        ...

    def resolve_model(self, generic_name: str) -> str:
        """Map a generic model name to a backend-specific model ID.

        Args:
            generic_name (str): Generic tier ('fast', 'balanced', 'powerful')
                or backend-specific model name.

        Returns:
            str: Backend-specific model identifier.

        Raises:
            ValueError: For unsupported model names.
        """
        ...

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the full shell command (as list) to spawn the agent.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts suitable for tmux pane execution.
        """
        ...

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return extra environment variables needed for this backend.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Env vars to merge with the current environment.
        """
        ...

    def spawn(self, request: SpawnRequest) -> SpawnResult:
        """Spawn the agent process in a tmux pane.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            SpawnResult: Process handle and backend type.
        """
        ...

    def health_check(self, handle: str) -> HealthStatus:
        """Check if a spawned agent is still running.

        Args:
            handle (str): Backend-specific process handle (tmux pane ID).

        Returns:
            HealthStatus: Alive/dead status and detail string.
        """
        ...

    def kill(self, handle: str) -> None:
        """Force-kill a spawned agent by its handle.

        Args:
            handle (str): Backend-specific process handle.
        """
        ...

    def graceful_shutdown(self, handle: str, timeout_s: float = 10.0) -> bool:
        """Attempt graceful shutdown via interrupt signal.

        Args:
            handle (str): Backend-specific process handle.
            timeout_s (float): Maximum seconds to wait for shutdown.

        Returns:
            bool: True if agent stopped within timeout, False otherwise.
        """
        ...

    def capture(self, handle: str, lines: int | None = None) -> str:
        """Capture current output from a spawned agent's tmux pane.

        Args:
            handle (str): Tmux pane identifier.
            lines (int | None): Number of recent lines to capture.
                None captures the full visible buffer.

        Returns:
            str: Captured pane content.
        """
        ...

    def send(self, handle: str, text: str, *, enter: bool = True) -> None:
        """Send text input to a spawned agent's tmux pane.

        Useful for follow-up prompts, providing input, or interacting
        with a running agent.

        Args:
            handle (str): Tmux pane identifier.
            text (str): Text to send.
            enter (bool): Whether to press Enter after the text.
        """
        ...

    def wait_idle(
        self,
        handle: str,
        idle_time: float = 2.0,
        timeout: int | None = None,
    ) -> bool:
        """Wait until a spawned agent's pane output stabilizes.

        Detects idle by monitoring content hash consistency over time.

        Args:
            handle (str): Tmux pane identifier.
            idle_time (float): Seconds of stable output to consider idle.
            timeout (int | None): Maximum seconds to wait. None waits
                indefinitely.

        Returns:
            bool: True if pane became idle, False on timeout.
        """
        ...

    def execute_in_pane(
        self,
        handle: str,
        command: str,
        timeout: int = 30,
    ) -> CaptureResult:
        """Execute a shell command in a spawned agent's pane.

        Runs the command and returns both output and exit code. Best
        suited for shell commands in an idle pane, not for interactive
        agent sessions.

        Args:
            handle (str): Tmux pane identifier.
            command (str): Shell command to execute.
            timeout (int): Maximum seconds to wait for completion.

        Returns:
            CaptureResult: Dict with 'output' (str) and 'exit_code' (int).
                exit_code is -1 on timeout.
        """
        ...


class BaseBackend:
    """Convenience base class with shared tmux lifecycle management.

    Uses ``TmuxCLIController`` from ``claude-code-tools`` for all tmux
    interactions, providing reliable pane management, output capture,
    and agent interactivity.

    Backends may inherit from this or implement the ``Backend`` protocol
    directly.  Subclasses must set ``_name`` and ``_binary_name`` class
    attributes and implement ``build_command``, ``build_env``,
    ``supported_models``, ``default_model``, and ``resolve_model``.
    """

    _name: str
    _binary_name: str

    def __init__(self) -> None:
        self._controller: TmuxCLIController | None = None

    @property
    def controller(self) -> TmuxCLIController:
        """Lazy-initialized tmux controller instance.

        Returns:
            TmuxCLIController: Shared controller for tmux operations.
        """
        if self._controller is None:
            self._controller = TmuxCLIController()
        return self._controller

    @property
    def name(self) -> str:
        """Unique backend identifier.

        Returns:
            str: Backend name such as 'claude-code' or 'codex'.
        """
        return self._name

    @property
    def binary_name(self) -> str:
        """Name of the CLI binary.

        Returns:
            str: Name of the CLI binary executable.
        """
        return self._binary_name

    @property
    def is_interactive(self) -> bool:
        """Whether this backend supports native team messaging.

        Most backends are one-shot CLIs whose output must be relayed by
        the server.  Subclasses that handle their own messaging (e.g.,
        claude-code) should override this to return ``True``.

        Returns:
            bool: False by default (one-shot / non-interactive).
        """
        return False

    def retain_pane_after_exit(self, handle: str) -> None:
        """Set ``remain-on-exit`` on the tmux pane so output survives process exit.

        Without this, tmux destroys the pane when the command finishes and
        the server loses the ability to capture output for relay.

        Args:
            handle (str): Tmux pane identifier.
        """
        self.controller._run_tmux_command(
            ["set-option", "-p", "-t", handle, "remain-on-exit", "on"]
        )

    def is_available(self) -> bool:
        """Return True if the backend binary is found on PATH.

        Returns:
            bool: True if binary is available, False otherwise.
        """
        return shutil.which(self._binary_name) is not None

    def discover_binary(self) -> str:
        """Return the full path to the backend binary.

        Raises:
            FileNotFoundError: If binary is not found on PATH.
        """
        path = shutil.which(self._binary_name)
        if path is None:
            raise FileNotFoundError(
                f"Could not find '{self._binary_name}' on PATH. "
                f"Install {self._name} or add it to PATH."
            )
        return path

    # ------------------------------------------------------------------
    # Lifecycle: spawn / health / kill / shutdown
    # ------------------------------------------------------------------

    def spawn(self, request: SpawnRequest) -> SpawnResult:
        """Spawn the agent in a new tmux pane via ``TmuxCLIController``.

        Launches the agent command using ``launch_cli``, which creates a
        tmux pane and starts the command in a single operation.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            SpawnResult: Tmux pane identifier as process handle.

        Raises:
            ValueError: If an environment variable name is invalid.
            RuntimeError: If tmux pane creation fails.
        """
        cmd_parts = self.build_command(request)
        env_vars = self.build_env(request)

        for key in env_vars:
            if not _SAFE_ENV_KEY.match(key):
                raise ValueError(f"Invalid environment variable name: {key!r}")

        env_prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in env_vars.items()
        )
        cmd_str = " ".join(shlex.quote(part) for part in cmd_parts)
        if env_prefix:
            full_cmd = f"cd {shlex.quote(request.cwd)} && {env_prefix} {cmd_str}"
        else:
            full_cmd = f"cd {shlex.quote(request.cwd)} && {cmd_str}"

        pane_id = self.controller.launch_cli(full_cmd)
        if pane_id is None:
            raise RuntimeError(
                f"Failed to create tmux pane for agent {request.name!r}. "
                "Ensure tmux is running and tmux-cli is available."
            )
        return SpawnResult(process_handle=pane_id, backend_type=self._name)

    def health_check(self, handle: str) -> HealthStatus:
        """Check if a spawned agent's process is still running.

        A pane may still exist after the process exits when
        ``remain-on-exit`` is set (used for one-shot output capture).
        This method checks both pane existence *and* the ``pane_dead``
        tmux variable to report the true process state.

        Args:
            handle (str): Tmux pane identifier.

        Returns:
            HealthStatus: Alive if process is running, dead otherwise.
        """
        panes = self.controller.list_panes()
        pane_exists = any(
            handle in (pane.get("id", ""), pane.get("formatted_id", ""))
            for pane in panes
        )
        if not pane_exists:
            return HealthStatus(alive=False, detail="tmux pane not found")

        # Check if process exited but pane was retained (remain-on-exit).
        output, code = self.controller._run_tmux_command(
            ["display-message", "-t", handle, "-p", "#{pane_dead}"]
        )
        if code == 0 and output.strip() == "1":
            return HealthStatus(alive=False, detail="process exited (pane retained)")

        return HealthStatus(alive=True, detail="tmux pane check")

    def kill(self, handle: str) -> None:
        """Force-kill a spawned agent's tmux pane.

        Args:
            handle (str): Tmux pane identifier.
        """
        self.controller.kill_pane(pane_id=handle)

    def graceful_shutdown(self, handle: str, timeout_s: float = 10.0) -> bool:
        """Send interrupt (Ctrl+C) and wait for the pane to become idle.

        Args:
            handle (str): Tmux pane identifier.
            timeout_s (float): Maximum seconds to wait for shutdown.

        Returns:
            bool: True if pane became idle within timeout, False otherwise.
        """
        self.controller.send_interrupt(pane_id=handle)
        return self.controller.wait_for_idle(
            pane_id=handle,
            idle_time=1.0,
            timeout=int(timeout_s),
        )

    # ------------------------------------------------------------------
    # Interactivity: capture / send / wait / execute
    # ------------------------------------------------------------------

    def capture(self, handle: str, lines: int | None = None) -> str:
        """Capture current output from a spawned agent's tmux pane.

        Args:
            handle (str): Tmux pane identifier.
            lines (int | None): Number of recent lines to capture.
                None captures the full visible buffer.

        Returns:
            str: Captured pane content.
        """
        return self.controller.capture_pane(pane_id=handle, lines=lines)

    def send(self, handle: str, text: str, *, enter: bool = True) -> None:
        """Send text input to a spawned agent's tmux pane.

        Args:
            handle (str): Tmux pane identifier.
            text (str): Text to send.
            enter (bool): Whether to press Enter after the text.
        """
        self.controller.send_keys(text, pane_id=handle, enter=enter)

    def wait_idle(
        self,
        handle: str,
        idle_time: float = 2.0,
        timeout: int | None = None,
    ) -> bool:
        """Wait until a spawned agent's pane output stabilizes.

        Args:
            handle (str): Tmux pane identifier.
            idle_time (float): Seconds of stable output to consider idle.
            timeout (int | None): Maximum seconds to wait.

        Returns:
            bool: True if pane became idle, False on timeout.
        """
        return self.controller.wait_for_idle(
            pane_id=handle,
            idle_time=idle_time,
            timeout=timeout,
        )

    def execute_in_pane(
        self,
        handle: str,
        command: str,
        timeout: int = 30,
    ) -> CaptureResult:
        """Execute a shell command in a spawned agent's pane.

        Args:
            handle (str): Tmux pane identifier.
            command (str): Shell command to execute.
            timeout (int): Maximum seconds to wait for completion.

        Returns:
            CaptureResult: Dict with 'output' (str) and 'exit_code' (int).
        """
        return cast(
            CaptureResult,
            self.controller.execute(command, pane_id=handle, timeout=timeout),
        )

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    def build_command(self, request: SpawnRequest) -> list[str]:
        """Build the command to spawn the agent. Must be overridden.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            list[str]: Command parts suitable for tmux pane execution.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        """Return extra environment variables. Must be overridden.

        Args:
            request (SpawnRequest): Backend-agnostic spawn parameters.

        Returns:
            dict[str, str]: Env vars to merge with the current environment.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError

    def supported_models(self) -> list[str]:
        """Return supported model names. Must be overridden.

        Returns:
            list[str]: Curated list of supported model identifiers.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError

    def default_model(self) -> str:
        """Return the default model. Must be overridden.

        Returns:
            str: Default model identifier for this backend.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError

    def resolve_model(self, generic_name: str) -> str:
        """Resolve a generic model name. Must be overridden.

        Args:
            generic_name (str): Generic tier ('fast', 'balanced', 'powerful')
                or backend-specific model name.

        Returns:
            str: Backend-specific model identifier.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError
