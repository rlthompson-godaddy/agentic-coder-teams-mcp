from unittest.mock import patch

from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.claudish import ClaudishBackend


_DEFAULT_REQUEST = SpawnRequest(
    agent_id="worker@team",
    name="worker",
    team_name="team",
    prompt="do stuff",
    model="oai@gpt-5.2",
    agent_type="general-purpose",
    color="blue",
    cwd="/tmp/work",
    lead_session_id="sess-1",
)


def _make_request(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
    return replace(_DEFAULT_REQUEST, **overrides)


class TestClaudishProperties:
    def test_name_is_claudish(self):
        backend = ClaudishBackend()
        assert backend.name == "claudish"

    def test_binary_name_is_claudish(self):
        backend = ClaudishBackend()
        assert backend.binary_name == "claudish"


class TestClaudishSupportedModels:
    def test_returns_expected_models(self):
        backend = ClaudishBackend()
        models = backend.supported_models()
        assert "google@gemini-2.5-flash" in models
        assert "oai@gpt-5.2" in models
        assert "google@gemini-3-pro" in models

    def test_returns_list(self):
        backend = ClaudishBackend()
        assert isinstance(backend.supported_models(), list)
        assert len(backend.supported_models()) > 0


class TestClaudishDefaultModel:
    def test_returns_oai_gpt52(self):
        backend = ClaudishBackend()
        assert backend.default_model() == "oai@gpt-5.2"


class TestClaudishResolveModel:
    def test_resolves_fast_to_gemini_flash(self):
        backend = ClaudishBackend()
        assert backend.resolve_model("fast") == "google@gemini-2.5-flash"

    def test_resolves_balanced_to_gpt52(self):
        backend = ClaudishBackend()
        assert backend.resolve_model("balanced") == "oai@gpt-5.2"

    def test_resolves_powerful_to_gemini_pro(self):
        backend = ClaudishBackend()
        assert backend.resolve_model("powerful") == "google@gemini-3-pro"

    def test_resolves_direct_model_name(self):
        backend = ClaudishBackend()
        assert backend.resolve_model("ollama@llama3.2") == "ollama@llama3.2"

    def test_passes_through_unknown_model_name(self):
        backend = ClaudishBackend()
        assert backend.resolve_model("custom@model") == "custom@model"


class TestClaudishBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/claudish")
    def test_produces_single_shot_command(self, _mock_which):
        backend = ClaudishBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/claudish"
        assert "--model" in cmd
        assert "-y" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/claudish")
    def test_includes_prompt_as_last_arg(self, _mock_which):
        backend = ClaudishBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/claudish")
    def test_includes_model_with_provider_syntax(self, _mock_which):
        backend = ClaudishBackend()
        request = _make_request(model="google@gemini-3-pro")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "google@gemini-3-pro"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/claudish")
    def test_resolves_generic_model(self, _mock_which):
        backend = ClaudishBackend()
        request = _make_request(model="fast")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "google@gemini-2.5-flash"


class TestClaudishBuildEnv:
    def test_returns_empty_dict(self):
        backend = ClaudishBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestClaudishAvailability:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/claudish")
    def test_available_when_binary_found(self, _mock_which):
        backend = ClaudishBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = ClaudishBackend()
        assert backend.is_available() is False
