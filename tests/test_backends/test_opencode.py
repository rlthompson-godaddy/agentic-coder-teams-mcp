from unittest.mock import patch


from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.opencode import OpenCodeBackend


_DEFAULT_REQUEST = SpawnRequest(
    agent_id="worker@team",
    name="worker",
    team_name="team",
    prompt="do stuff",
    model="anthropic/claude-sonnet-4",
    agent_type="general-purpose",
    color="blue",
    cwd="/tmp/work",
    lead_session_id="sess-1",
)


def _make_request(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
    return replace(_DEFAULT_REQUEST, **overrides)


class TestOpenCodeProperties:
    def test_name_is_opencode(self):
        backend = OpenCodeBackend()
        assert backend.name == "opencode"

    def test_binary_name_is_opencode(self):
        backend = OpenCodeBackend()
        assert backend.binary_name == "opencode"


class TestOpenCodeSupportedModels:
    def test_returns_expected_models(self):
        backend = OpenCodeBackend()
        models = backend.supported_models()
        assert "anthropic/claude-sonnet-4" in models
        assert "anthropic/claude-opus-4" in models
        assert "openai/gpt-5.3-codex" in models
        assert "google/gemini-2.5-pro" in models


class TestOpenCodeDefaultModel:
    def test_returns_claude_sonnet_4(self):
        backend = OpenCodeBackend()
        assert backend.default_model() == "anthropic/claude-sonnet-4"


class TestOpenCodeResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = OpenCodeBackend()
        assert backend.resolve_model("fast") == "anthropic/claude-haiku-3.5"

    def test_resolves_balanced_to_sonnet(self):
        backend = OpenCodeBackend()
        assert backend.resolve_model("balanced") == "anthropic/claude-sonnet-4"

    def test_resolves_powerful_to_opus(self):
        backend = OpenCodeBackend()
        assert backend.resolve_model("powerful") == "anthropic/claude-opus-4"

    def test_passes_through_unknown_model_name(self):
        backend = OpenCodeBackend()
        assert backend.resolve_model("openai/gpt-5.3-codex") == "openai/gpt-5.3-codex"

    def test_passes_through_arbitrary_string(self):
        backend = OpenCodeBackend()
        assert backend.resolve_model("some-custom-model") == "some-custom-model"


class TestOpenCodeBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/opencode")
    def test_produces_run_command(self, mock_which):
        backend = OpenCodeBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/opencode"
        assert "run" in cmd
        assert "--model" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/opencode")
    def test_includes_prompt_as_last_arg(self, mock_which):
        backend = OpenCodeBackend()
        request = _make_request(prompt="implement feature")

        cmd = backend.build_command(request)

        assert cmd[-1] == "implement feature"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/opencode")
    def test_includes_model_value(self, mock_which):
        backend = OpenCodeBackend()
        request = _make_request(model="balanced")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "anthropic/claude-sonnet-4"


class TestOpenCodeBuildEnv:
    def test_returns_empty_dict(self):
        backend = OpenCodeBackend()
        request = _make_request()

        env = backend.build_env(request)

        assert env == {}
