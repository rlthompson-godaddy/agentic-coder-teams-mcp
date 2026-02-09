from unittest.mock import patch

from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.copilot import CopilotBackend


_DEFAULT_REQUEST = SpawnRequest(
    agent_id="worker@team",
    name="worker",
    team_name="team",
    prompt="do stuff",
    model="claude-sonnet-4.5",
    agent_type="general-purpose",
    color="blue",
    cwd="/tmp/work",
    lead_session_id="sess-1",
)


def _make_request(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
    return replace(_DEFAULT_REQUEST, **overrides)


class TestCopilotProperties:
    def test_name_is_copilot(self):
        backend = CopilotBackend()
        assert backend.name == "copilot"

    def test_binary_name_is_copilot(self):
        backend = CopilotBackend()
        assert backend.binary_name == "copilot"


class TestCopilotSupportedModels:
    def test_returns_expected_models(self):
        backend = CopilotBackend()
        models = backend.supported_models()
        assert "claude-sonnet-4.5" in models
        assert "claude-opus-4.6" in models
        assert "gpt-5.2-codex" in models
        assert "gemini-3-pro-preview" in models

    def test_returns_list(self):
        backend = CopilotBackend()
        assert isinstance(backend.supported_models(), list)
        assert len(backend.supported_models()) > 0


class TestCopilotDefaultModel:
    def test_returns_claude_sonnet_4_5(self):
        backend = CopilotBackend()
        assert backend.default_model() == "claude-sonnet-4.5"


class TestCopilotResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = CopilotBackend()
        assert backend.resolve_model("fast") == "claude-haiku-4.5"

    def test_resolves_balanced_to_sonnet(self):
        backend = CopilotBackend()
        assert backend.resolve_model("balanced") == "claude-sonnet-4.5"

    def test_resolves_powerful_to_opus(self):
        backend = CopilotBackend()
        assert backend.resolve_model("powerful") == "claude-opus-4.6"

    def test_resolves_direct_model_name(self):
        backend = CopilotBackend()
        assert backend.resolve_model("gpt-5.2-codex") == "gpt-5.2-codex"

    def test_passes_through_unknown_model_name(self):
        backend = CopilotBackend()
        assert backend.resolve_model("custom-model") == "custom-model"

    def test_passes_through_empty_string(self):
        backend = CopilotBackend()
        assert backend.resolve_model("") == ""


class TestCopilotBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/copilot")
    def test_produces_prompt_command(self, mock_which):
        backend = CopilotBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/copilot"
        assert "-p" in cmd
        assert "--model" in cmd
        assert "--yolo" in cmd
        assert "--no-ask-user" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/copilot")
    def test_includes_prompt_after_p_flag(self, mock_which):
        backend = CopilotBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        idx = cmd.index("-p")
        assert cmd[idx + 1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/copilot")
    def test_includes_model_flag(self, mock_which):
        backend = CopilotBackend()
        request = _make_request(model="gpt-5")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "gpt-5"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/copilot")
    def test_resolves_generic_model(self, mock_which):
        backend = CopilotBackend()
        request = _make_request(model="powerful")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4.6"


class TestCopilotBuildEnv:
    def test_returns_empty_dict(self):
        backend = CopilotBackend()
        request = _make_request()

        env = backend.build_env(request)

        assert env == {}


class TestCopilotAvailability:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/copilot")
    def test_available_when_binary_found(self, mock_which):
        backend = CopilotBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, mock_which):
        backend = CopilotBackend()
        assert backend.is_available() is False
