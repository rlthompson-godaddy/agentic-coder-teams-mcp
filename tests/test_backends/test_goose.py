from unittest.mock import patch

from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.goose import GooseBackend


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


class TestGooseProperties:
    def test_name_is_goose(self):
        backend = GooseBackend()
        assert backend.name == "goose"

    def test_binary_name_is_goose(self):
        backend = GooseBackend()
        assert backend.binary_name == "goose"


class TestGooseSupportedModels:
    def test_returns_expected_models(self):
        backend = GooseBackend()
        models = backend.supported_models()
        assert "claude-sonnet-4.5" in models
        assert "claude-opus-4.6" in models
        assert "gpt-5.2-codex" in models
        assert "gemini-2.5-pro" in models


class TestGooseDefaultModel:
    def test_returns_claude_sonnet(self):
        backend = GooseBackend()
        assert backend.default_model() == "claude-sonnet-4.5"


class TestGooseResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = GooseBackend()
        assert backend.resolve_model("fast") == "claude-haiku-4.5"

    def test_resolves_balanced_to_sonnet(self):
        backend = GooseBackend()
        assert backend.resolve_model("balanced") == "claude-sonnet-4.5"

    def test_resolves_powerful_to_opus(self):
        backend = GooseBackend()
        assert backend.resolve_model("powerful") == "claude-opus-4.6"

    def test_resolves_direct_model_name(self):
        backend = GooseBackend()
        assert backend.resolve_model("gpt-5.2-codex") == "gpt-5.2-codex"

    def test_passes_through_unknown_model_name(self):
        backend = GooseBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestGooseBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/goose")
    def test_produces_run_command(self, _mock_which):
        backend = GooseBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/goose"
        assert "run" in cmd
        assert "-t" in cmd
        assert "--model" in cmd
        assert "--no-session" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/goose")
    def test_includes_prompt_after_t_flag(self, _mock_which):
        backend = GooseBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        idx = cmd.index("-t")
        assert cmd[idx + 1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/goose")
    def test_includes_provider_for_generic_tier(self, _mock_which):
        backend = GooseBackend()
        request = _make_request(model="powerful")

        cmd = backend.build_command(request)

        assert "--provider" in cmd
        idx = cmd.index("--provider")
        assert cmd[idx + 1] == "anthropic"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/goose")
    def test_no_provider_for_direct_model(self, _mock_which):
        backend = GooseBackend()
        request = _make_request(model="gpt-5.2-codex")

        cmd = backend.build_command(request)

        assert "--provider" not in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/goose")
    def test_resolves_generic_model(self, _mock_which):
        backend = GooseBackend()
        request = _make_request(model="fast")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-haiku-4.5"


class TestGooseBuildEnv:
    def test_returns_empty_dict(self):
        backend = GooseBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestGooseAvailability:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/goose")
    def test_available_when_binary_found(self, _mock_which):
        backend = GooseBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = GooseBackend()
        assert backend.is_available() is False
