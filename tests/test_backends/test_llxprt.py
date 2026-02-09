from unittest.mock import patch

from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.llxprt import LlxprtBackend


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


class TestLlxprtProperties:
    def test_name_is_llxprt(self):
        backend = LlxprtBackend()
        assert backend.name == "llxprt"

    def test_binary_name_is_llxprt(self):
        backend = LlxprtBackend()
        assert backend.binary_name == "llxprt"


class TestLlxprtSupportedModels:
    def test_returns_expected_models(self):
        backend = LlxprtBackend()
        models = backend.supported_models()
        assert "claude-haiku-4.5" in models
        assert "claude-sonnet-4.5" in models
        assert "claude-opus-4.6" in models


class TestLlxprtDefaultModel:
    def test_returns_claude_sonnet(self):
        backend = LlxprtBackend()
        assert backend.default_model() == "claude-sonnet-4.5"


class TestLlxprtResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = LlxprtBackend()
        assert backend.resolve_model("fast") == "claude-haiku-4.5"

    def test_resolves_balanced_to_sonnet(self):
        backend = LlxprtBackend()
        assert backend.resolve_model("balanced") == "claude-sonnet-4.5"

    def test_resolves_powerful_to_opus(self):
        backend = LlxprtBackend()
        assert backend.resolve_model("powerful") == "claude-opus-4.6"

    def test_resolves_direct_model_name(self):
        backend = LlxprtBackend()
        assert backend.resolve_model("claude-opus-4.6") == "claude-opus-4.6"

    def test_passes_through_unknown_model_name(self):
        backend = LlxprtBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestLlxprtBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/llxprt")
    def test_produces_prompt_command(self, _mock_which):
        backend = LlxprtBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/llxprt"
        assert "-p" in cmd
        assert "-m" in cmd
        assert "-y" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/llxprt")
    def test_includes_prompt_after_p_flag(self, _mock_which):
        backend = LlxprtBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        idx = cmd.index("-p")
        assert cmd[idx + 1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/llxprt")
    def test_resolves_generic_model(self, _mock_which):
        backend = LlxprtBackend()
        request = _make_request(model="powerful")

        cmd = backend.build_command(request)

        idx = cmd.index("-m")
        assert cmd[idx + 1] == "claude-opus-4.6"


class TestLlxprtBuildEnv:
    def test_returns_empty_dict(self):
        backend = LlxprtBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestLlxprtAvailability:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/llxprt")
    def test_available_when_binary_found(self, _mock_which):
        backend = LlxprtBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = LlxprtBackend()
        assert backend.is_available() is False
