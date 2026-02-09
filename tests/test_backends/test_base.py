from dataclasses import FrozenInstanceError, replace
from unittest.mock import MagicMock, patch

import pytest

from claude_teams.backends.base import (
    Backend,
    BaseBackend,
    HealthStatus,
    SpawnRequest,
    SpawnResult,
)


# ---------------------------------------------------------------------------
# Helper: concrete subclass of BaseBackend for testing
# ---------------------------------------------------------------------------


class _StubBackend(BaseBackend):
    """Minimal concrete backend for testing BaseBackend methods."""

    _name = "stub"
    _binary_name = "stub-cli"

    def build_command(self, request: SpawnRequest) -> list[str]:
        binary = self.discover_binary()
        return [binary, "--prompt", request.prompt]

    def build_env(self, request: SpawnRequest) -> dict[str, str]:
        return {"STUB_MODE": "1"}

    def supported_models(self) -> list[str]:
        return ["default"]

    def default_model(self) -> str:
        return "default"

    def resolve_model(self, generic_name: str) -> str:
        return generic_name


_DEFAULT_REQUEST = SpawnRequest(
    agent_id="worker@team",
    name="worker",
    team_name="team",
    prompt="do stuff",
    model="default",
    agent_type="general-purpose",
    color="blue",
    cwd="/tmp",
    lead_session_id="sess-1",
)


def _make_spawn_request(
    **overrides: str | bool | dict[str, str] | None,
) -> SpawnRequest:
    return replace(_DEFAULT_REQUEST, **overrides)


def _make_backend_with_mock_controller() -> tuple["_StubBackend", MagicMock]:
    """Create a _StubBackend with a mocked TmuxCLIController."""
    backend = _StubBackend()
    mock_ctrl = MagicMock()
    backend._controller = mock_ctrl
    return backend, mock_ctrl


# ---------------------------------------------------------------------------
# SpawnRequest dataclass
# ---------------------------------------------------------------------------


class TestSpawnRequest:
    def test_creates_with_all_fields(self):
        req = SpawnRequest(
            agent_id="a@t",
            name="a",
            team_name="t",
            prompt="do",
            model="sonnet",
            agent_type="general-purpose",
            color="blue",
            cwd="/tmp",
            lead_session_id="sess-1",
        )
        assert req.agent_id == "a@t"
        assert req.name == "a"
        assert req.team_name == "t"
        assert req.prompt == "do"
        assert req.model == "sonnet"
        assert req.agent_type == "general-purpose"
        assert req.color == "blue"
        assert req.cwd == "/tmp"
        assert req.lead_session_id == "sess-1"
        assert req.plan_mode_required is False
        assert req.extra is None

    def test_frozen_raises_on_mutation(self):
        req = _make_spawn_request()
        with pytest.raises(FrozenInstanceError):
            req.name = "other"  # type: ignore[invalid-assignment]

    def test_plan_mode_required_default_false(self):
        req = _make_spawn_request()
        assert req.plan_mode_required is False

    def test_extra_accepts_dict(self):
        req = _make_spawn_request(extra={"key": "val"})
        assert req.extra == {"key": "val"}


# ---------------------------------------------------------------------------
# SpawnResult dataclass
# ---------------------------------------------------------------------------


class TestSpawnResult:
    def test_creates_with_required_fields(self):
        result = SpawnResult(process_handle="%42", backend_type="claude-code")
        assert result.process_handle == "%42"
        assert result.backend_type == "claude-code"

    def test_frozen_raises_on_mutation(self):
        result = SpawnResult(process_handle="%1", backend_type="codex")
        with pytest.raises(FrozenInstanceError):
            result.process_handle = "%2"  # type: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# HealthStatus dataclass
# ---------------------------------------------------------------------------


class TestHealthStatus:
    def test_creates_with_alive_flag(self):
        health = HealthStatus(alive=True)
        assert health.alive is True
        assert health.detail == ""

    def test_detail_defaults_to_empty_string(self):
        health = HealthStatus(alive=False)
        assert health.detail == ""

    def test_detail_accepts_custom_string(self):
        health = HealthStatus(alive=True, detail="tmux pane check")
        assert health.detail == "tmux pane check"

    def test_frozen_raises_on_mutation(self):
        health = HealthStatus(alive=True)
        with pytest.raises(FrozenInstanceError):
            health.alive = False  # type: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Backend Protocol runtime check
# ---------------------------------------------------------------------------


class TestBackendProtocol:
    def test_stub_backend_satisfies_protocol(self):
        backend = _StubBackend()
        assert isinstance(backend, Backend)

    def test_plain_object_does_not_satisfy_protocol(self):
        assert not isinstance(object(), Backend)


# ---------------------------------------------------------------------------
# BaseBackend.controller property
# ---------------------------------------------------------------------------


class TestBaseBackendController:
    @patch("claude_teams.backends.base.TmuxCLIController")
    def test_creates_controller_lazily(self, mock_ctrl_cls: MagicMock):
        backend = _StubBackend()
        ctrl = backend.controller
        mock_ctrl_cls.assert_called_once()
        assert ctrl is mock_ctrl_cls.return_value

    @patch("claude_teams.backends.base.TmuxCLIController")
    def test_returns_same_controller_on_subsequent_calls(
        self, mock_ctrl_cls: MagicMock
    ):
        backend = _StubBackend()
        ctrl1 = backend.controller
        ctrl2 = backend.controller
        assert ctrl1 is ctrl2
        mock_ctrl_cls.assert_called_once()

    def test_accepts_injected_controller(self):
        backend = _StubBackend()
        mock_ctrl = MagicMock()
        backend._controller = mock_ctrl
        assert backend.controller is mock_ctrl


# ---------------------------------------------------------------------------
# BaseBackend.is_available
# ---------------------------------------------------------------------------


class TestBaseBackendIsAvailable:
    @patch("claude_teams.backends.base.shutil.which")
    def test_returns_true_when_binary_found(self, mock_which: MagicMock):
        mock_which.return_value = "/usr/bin/stub-cli"
        backend = _StubBackend()
        assert backend.is_available() is True
        mock_which.assert_called_once_with("stub-cli")

    @patch("claude_teams.backends.base.shutil.which")
    def test_returns_false_when_binary_not_found(self, mock_which: MagicMock):
        mock_which.return_value = None
        backend = _StubBackend()
        assert backend.is_available() is False


# ---------------------------------------------------------------------------
# BaseBackend.discover_binary
# ---------------------------------------------------------------------------


class TestBaseBackendDiscoverBinary:
    @patch("claude_teams.backends.base.shutil.which")
    def test_returns_full_path_when_found(self, mock_which: MagicMock):
        mock_which.return_value = "/usr/local/bin/stub-cli"
        backend = _StubBackend()
        assert backend.discover_binary() == "/usr/local/bin/stub-cli"

    @patch("claude_teams.backends.base.shutil.which")
    def test_raises_file_not_found_when_missing(self, mock_which: MagicMock):
        mock_which.return_value = None
        backend = _StubBackend()
        with pytest.raises(FileNotFoundError, match="stub-cli"):
            backend.discover_binary()


# ---------------------------------------------------------------------------
# BaseBackend.spawn
# ---------------------------------------------------------------------------


class TestBaseBackendSpawn:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/stub-cli")
    def test_returns_spawn_result_on_success(self, _mock_which: MagicMock):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.launch_cli.return_value = "remote:1.2"
        request = _make_spawn_request()

        result = backend.spawn(request)

        assert isinstance(result, SpawnResult)
        assert result.process_handle == "remote:1.2"
        assert result.backend_type == "stub"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/stub-cli")
    def test_calls_launch_cli_with_full_command(self, _mock_which: MagicMock):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.launch_cli.return_value = "remote:1.0"
        request = _make_spawn_request()

        backend.spawn(request)

        mock_ctrl.launch_cli.assert_called_once()
        full_cmd = mock_ctrl.launch_cli.call_args[0][0]
        assert "cd" in full_cmd
        assert "STUB_MODE=" in full_cmd
        assert "stub-cli" in full_cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/stub-cli")
    def test_raises_runtime_error_when_launch_fails(self, _mock_which: MagicMock):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.launch_cli.return_value = None
        request = _make_spawn_request()

        with pytest.raises(RuntimeError, match="Failed to create tmux pane"):
            backend.spawn(request)

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/stub-cli")
    def test_includes_env_prefix_in_command(self, _mock_which: MagicMock):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.launch_cli.return_value = "remote:1.0"
        request = _make_spawn_request()

        backend.spawn(request)

        full_cmd = mock_ctrl.launch_cli.call_args[0][0]
        assert "STUB_MODE=" in full_cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/stub-cli")
    def test_rejects_invalid_env_var_name(self, _mock_which: MagicMock):
        backend = _StubBackend()
        backend.build_env = lambda _req: {"INVALID-KEY": "val"}  # type: ignore[assignment]
        request = _make_spawn_request()

        with pytest.raises(ValueError, match="Invalid environment variable"):
            backend.spawn(request)


# ---------------------------------------------------------------------------
# BaseBackend.health_check
# ---------------------------------------------------------------------------


class TestBaseBackendHealthCheck:
    def test_returns_alive_when_pane_exists_by_id(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.list_panes.return_value = [
            {"id": "%1", "formatted_id": "remote:1.0"},
            {"id": "%42", "formatted_id": "remote:1.1"},
        ]

        status = backend.health_check("%42")

        assert status.alive is True
        assert status.detail == "tmux pane check"

    def test_returns_alive_when_pane_exists_by_formatted_id(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.list_panes.return_value = [
            {"id": "%1", "formatted_id": "remote:1.0"},
            {"id": "%42", "formatted_id": "remote:1.1"},
        ]

        status = backend.health_check("remote:1.1")

        assert status.alive is True

    def test_returns_dead_when_pane_missing(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.list_panes.return_value = [
            {"id": "%1", "formatted_id": "remote:1.0"},
        ]

        status = backend.health_check("%42")

        assert status.alive is False

    def test_returns_dead_when_no_panes(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.list_panes.return_value = []

        status = backend.health_check("%42")

        assert status.alive is False


# ---------------------------------------------------------------------------
# BaseBackend.kill
# ---------------------------------------------------------------------------


class TestBaseBackendKill:
    def test_calls_controller_kill_pane(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()

        backend.kill("%42")

        mock_ctrl.kill_pane.assert_called_once_with(pane_id="%42")

    def test_accepts_formatted_pane_id(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()

        backend.kill("remote:1.2")

        mock_ctrl.kill_pane.assert_called_once_with(pane_id="remote:1.2")


# ---------------------------------------------------------------------------
# BaseBackend.graceful_shutdown
# ---------------------------------------------------------------------------


class TestBaseBackendGracefulShutdown:
    def test_sends_interrupt_and_waits(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.wait_for_idle.return_value = True

        result = backend.graceful_shutdown("%42", timeout_s=5.0)

        assert result is True
        mock_ctrl.send_interrupt.assert_called_once_with(pane_id="%42")
        mock_ctrl.wait_for_idle.assert_called_once_with(
            pane_id="%42",
            idle_time=1.0,
            timeout=5,
        )

    def test_returns_false_when_timeout_exceeded(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.wait_for_idle.return_value = False

        result = backend.graceful_shutdown("%42", timeout_s=2.0)

        assert result is False
        mock_ctrl.send_interrupt.assert_called_once()


# ---------------------------------------------------------------------------
# BaseBackend.capture
# ---------------------------------------------------------------------------


class TestBaseBackendCapture:
    def test_captures_full_buffer(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.capture_pane.return_value = "line 1\nline 2\n"

        output = backend.capture("%42")

        assert output == "line 1\nline 2\n"
        mock_ctrl.capture_pane.assert_called_once_with(pane_id="%42", lines=None)

    def test_captures_limited_lines(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.capture_pane.return_value = "last line\n"

        output = backend.capture("%42", lines=1)

        assert output == "last line\n"
        mock_ctrl.capture_pane.assert_called_once_with(pane_id="%42", lines=1)


# ---------------------------------------------------------------------------
# BaseBackend.send
# ---------------------------------------------------------------------------


class TestBaseBackendSend:
    def test_sends_text_with_enter(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()

        backend.send("%42", "hello world")

        mock_ctrl.send_keys.assert_called_once_with(
            "hello world",
            pane_id="%42",
            enter=True,
        )

    def test_sends_text_without_enter(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()

        backend.send("%42", "partial", enter=False)

        mock_ctrl.send_keys.assert_called_once_with(
            "partial",
            pane_id="%42",
            enter=False,
        )


# ---------------------------------------------------------------------------
# BaseBackend.wait_idle
# ---------------------------------------------------------------------------


class TestBaseBackendWaitIdle:
    def test_waits_with_defaults(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.wait_for_idle.return_value = True

        result = backend.wait_idle("%42")

        assert result is True
        mock_ctrl.wait_for_idle.assert_called_once_with(
            pane_id="%42",
            idle_time=2.0,
            timeout=None,
        )

    def test_waits_with_custom_params(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.wait_for_idle.return_value = False

        result = backend.wait_idle("%42", idle_time=5.0, timeout=30)

        assert result is False
        mock_ctrl.wait_for_idle.assert_called_once_with(
            pane_id="%42",
            idle_time=5.0,
            timeout=30,
        )


# ---------------------------------------------------------------------------
# BaseBackend.execute_in_pane
# ---------------------------------------------------------------------------


class TestBaseBackendExecuteInPane:
    def test_executes_command_and_returns_result(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.execute.return_value = {"output": "ok", "exit_code": 0}

        result = backend.execute_in_pane("%42", "echo hello")

        assert result == {"output": "ok", "exit_code": 0}
        mock_ctrl.execute.assert_called_once_with(
            "echo hello",
            pane_id="%42",
            timeout=30,
        )

    def test_respects_custom_timeout(self):
        backend, mock_ctrl = _make_backend_with_mock_controller()
        mock_ctrl.execute.return_value = {"output": "", "exit_code": -1}

        result = backend.execute_in_pane("%42", "long cmd", timeout=120)

        assert result["exit_code"] == -1
        mock_ctrl.execute.assert_called_once_with(
            "long cmd",
            pane_id="%42",
            timeout=120,
        )
