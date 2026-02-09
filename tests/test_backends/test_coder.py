from unittest.mock import patch

from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.coder import CoderBackend


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


class TestCoderProperties:
    def test_name_is_coder(self):
        backend = CoderBackend()
        assert backend.name == "coder"

    def test_binary_name_is_coder(self):
        backend = CoderBackend()
        assert backend.binary_name == "coder"


class TestCoderSupportedModels:
    def test_returns_expected_models(self):
        backend = CoderBackend()
        models = backend.supported_models()
        assert "claude-sonnet-4.5" in models
        assert "claude-opus-4.6" in models
        assert "gpt-5.2-codex" in models
        assert "o3" in models

    def test_returns_list(self):
        backend = CoderBackend()
        assert isinstance(backend.supported_models(), list)
        assert len(backend.supported_models()) > 0


class TestCoderDefaultModel:
    def test_returns_claude_sonnet(self):
        backend = CoderBackend()
        assert backend.default_model() == "claude-sonnet-4.5"


class TestCoderResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = CoderBackend()
        assert backend.resolve_model("fast") == "claude-haiku-4.5"

    def test_resolves_balanced_to_sonnet(self):
        backend = CoderBackend()
        assert backend.resolve_model("balanced") == "claude-sonnet-4.5"

    def test_resolves_powerful_to_opus(self):
        backend = CoderBackend()
        assert backend.resolve_model("powerful") == "claude-opus-4.6"

    def test_resolves_direct_model_name(self):
        backend = CoderBackend()
        assert backend.resolve_model("gpt-5.2-codex") == "gpt-5.2-codex"

    def test_passes_through_unknown_model_name(self):
        backend = CoderBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestCoderBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/coder")
    def test_produces_exec_command(self, _mock_which):
        backend = CoderBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/coder"
        assert "exec" in cmd
        assert "-m" in cmd
        assert "--full-auto" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/coder")
    def test_includes_prompt_as_last_arg(self, _mock_which):
        backend = CoderBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/coder")
    def test_resolves_generic_model(self, _mock_which):
        backend = CoderBackend()
        request = _make_request(model="powerful")

        cmd = backend.build_command(request)

        idx = cmd.index("-m")
        assert cmd[idx + 1] == "claude-opus-4.6"


class TestCoderBuildEnv:
    def test_returns_empty_dict(self):
        backend = CoderBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestCoderAvailability:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/coder")
    def test_available_when_binary_found(self, _mock_which):
        backend = CoderBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = CoderBackend()
        assert backend.is_available() is False
