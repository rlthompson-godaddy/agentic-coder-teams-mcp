from unittest.mock import patch

import pytest

from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.claude_code import ClaudeCodeBackend


_DEFAULT_REQUEST = SpawnRequest(
    agent_id="worker@team",
    name="worker",
    team_name="team",
    prompt="do stuff",
    model="sonnet",
    agent_type="general-purpose",
    color="blue",
    cwd="/tmp/work",
    lead_session_id="sess-1",
)


def _make_request(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
    return replace(_DEFAULT_REQUEST, **overrides)


class TestClaudeCodeProperties:
    def test_name_is_claude_code(self):
        backend = ClaudeCodeBackend()
        assert backend.name == "claude-code"

    def test_binary_name_is_claude(self):
        backend = ClaudeCodeBackend()
        assert backend.binary_name == "claude"


class TestClaudeCodeSupportedModels:
    def test_returns_expected_models(self):
        backend = ClaudeCodeBackend()
        models = backend.supported_models()
        assert "haiku" in models
        assert "sonnet" in models
        assert "opus" in models
        assert len(models) == 3


class TestClaudeCodeDefaultModel:
    def test_returns_sonnet(self):
        backend = ClaudeCodeBackend()
        assert backend.default_model() == "sonnet"


class TestClaudeCodeResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("fast") == "haiku"

    def test_resolves_balanced_to_sonnet(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("balanced") == "sonnet"

    def test_resolves_powerful_to_opus(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("powerful") == "opus"

    def test_resolves_direct_name_haiku(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("haiku") == "haiku"

    def test_resolves_direct_name_sonnet(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("sonnet") == "sonnet"

    def test_resolves_direct_name_opus(self):
        backend = ClaudeCodeBackend()
        assert backend.resolve_model("opus") == "opus"

    def test_raises_for_unsupported_model(self):
        backend = ClaudeCodeBackend()
        with pytest.raises(ValueError, match="Unsupported model"):
            backend.resolve_model("gpt-4")

    def test_raises_for_empty_string(self):
        backend = ClaudeCodeBackend()
        with pytest.raises(ValueError):
            backend.resolve_model("")


class TestClaudeCodeBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/claude")
    def test_produces_correct_flags(self, mock_which):
        backend = ClaudeCodeBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/claude"
        assert "--agent-id" in cmd
        assert "--agent-name" in cmd
        assert "--team-name" in cmd
        assert "--agent-color" in cmd
        assert "--parent-session-id" in cmd
        assert "--agent-type" in cmd
        assert "--model" in cmd
        # Values match request
        idx = cmd.index("--agent-id")
        assert cmd[idx + 1] == "worker@team"
        idx = cmd.index("--agent-name")
        assert cmd[idx + 1] == "worker"
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "sonnet"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/claude")
    def test_includes_plan_mode_required_when_set(self, mock_which):
        backend = ClaudeCodeBackend()
        request = _make_request(plan_mode_required=True)

        cmd = backend.build_command(request)

        assert "--plan-mode-required" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/claude")
    def test_excludes_plan_mode_required_when_false(self, mock_which):
        backend = ClaudeCodeBackend()
        request = _make_request(plan_mode_required=False)

        cmd = backend.build_command(request)

        assert "--plan-mode-required" not in cmd


class TestClaudeCodeBuildEnv:
    def test_returns_claude_env_vars(self):
        backend = ClaudeCodeBackend()
        request = _make_request()

        env = backend.build_env(request)

        assert env["CLAUDECODE"] == "1"
        assert env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"
        assert len(env) == 2
