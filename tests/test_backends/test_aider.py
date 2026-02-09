from unittest.mock import patch


from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.aider import AiderBackend


_DEFAULT_REQUEST = SpawnRequest(
    agent_id="worker@team",
    name="worker",
    team_name="team",
    prompt="do stuff",
    model="claude-sonnet-4",
    agent_type="general-purpose",
    color="blue",
    cwd="/tmp/work",
    lead_session_id="sess-1",
)


def _make_request(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
    return replace(_DEFAULT_REQUEST, **overrides)


class TestAiderProperties:
    def test_name_is_aider(self):
        backend = AiderBackend()
        assert backend.name == "aider"

    def test_binary_name_is_aider(self):
        backend = AiderBackend()
        assert backend.binary_name == "aider"


class TestAiderSupportedModels:
    def test_returns_expected_models(self):
        backend = AiderBackend()
        models = backend.supported_models()
        assert "claude-sonnet-4" in models
        assert "claude-opus-4" in models
        assert "gpt-5.3-codex" in models
        assert "gemini-2.5-pro" in models


class TestAiderDefaultModel:
    def test_returns_claude_sonnet_4(self):
        backend = AiderBackend()
        assert backend.default_model() == "claude-sonnet-4"


class TestAiderResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = AiderBackend()
        assert backend.resolve_model("fast") == "claude-3.5-haiku"

    def test_resolves_balanced_to_sonnet(self):
        backend = AiderBackend()
        assert backend.resolve_model("balanced") == "claude-sonnet-4"

    def test_resolves_powerful_to_opus(self):
        backend = AiderBackend()
        assert backend.resolve_model("powerful") == "claude-opus-4"

    def test_resolves_direct_model_name(self):
        backend = AiderBackend()
        assert backend.resolve_model("claude-sonnet-4") == "claude-sonnet-4"

    def test_passes_through_unknown_model_name(self):
        backend = AiderBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestAiderBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/aider")
    def test_produces_correct_flags(self, mock_which):
        backend = AiderBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/aider"
        assert "--model" in cmd
        assert "--message" in cmd
        assert "--yes-always" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/aider")
    def test_includes_prompt_in_message_flag(self, mock_which):
        backend = AiderBackend()
        request = _make_request(prompt="write tests")

        cmd = backend.build_command(request)

        idx = cmd.index("--message")
        assert cmd[idx + 1] == "write tests"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/aider")
    def test_includes_model_value(self, mock_which):
        backend = AiderBackend()
        request = _make_request(model="balanced")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-sonnet-4"


class TestAiderBuildEnv:
    def test_returns_empty_dict(self):
        backend = AiderBackend()
        request = _make_request()

        env = backend.build_env(request)

        assert env == {}
