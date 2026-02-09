from unittest.mock import patch


from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.codex import CodexBackend


_DEFAULT_REQUEST = SpawnRequest(
    agent_id="worker@team",
    name="worker",
    team_name="team",
    prompt="do stuff",
    model="gpt-5.3-codex",
    agent_type="general-purpose",
    color="blue",
    cwd="/tmp/work",
    lead_session_id="sess-1",
)


def _make_request(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
    return replace(_DEFAULT_REQUEST, **overrides)


class TestCodexProperties:
    def test_name_is_codex(self):
        backend = CodexBackend()
        assert backend.name == "codex"

    def test_binary_name_is_codex(self):
        backend = CodexBackend()
        assert backend.binary_name == "codex"


class TestCodexSupportedModels:
    def test_returns_expected_models(self):
        backend = CodexBackend()
        models = backend.supported_models()
        assert "gpt-5.3-codex" in models
        assert "gpt-5.1-codex-max" in models
        assert "gpt-5.1-codex-mini" in models


class TestCodexDefaultModel:
    def test_returns_gpt_5_3_codex(self):
        backend = CodexBackend()
        assert backend.default_model() == "gpt-5.3-codex"


class TestCodexResolveModel:
    def test_resolves_fast_to_mini(self):
        backend = CodexBackend()
        assert backend.resolve_model("fast") == "gpt-5.1-codex-mini"

    def test_resolves_balanced_to_codex(self):
        backend = CodexBackend()
        assert backend.resolve_model("balanced") == "gpt-5.3-codex"

    def test_resolves_powerful_to_max(self):
        backend = CodexBackend()
        assert backend.resolve_model("powerful") == "gpt-5.1-codex-max"

    def test_resolves_direct_model_name(self):
        backend = CodexBackend()
        assert backend.resolve_model("gpt-5.3-codex") == "gpt-5.3-codex"

    def test_passes_through_unknown_model_name(self):
        backend = CodexBackend()
        assert backend.resolve_model("custom-model") == "custom-model"

    def test_passes_through_empty_string(self):
        backend = CodexBackend()
        assert backend.resolve_model("") == ""


class TestCodexBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/codex")
    def test_produces_exec_command(self, mock_which):
        backend = CodexBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/codex"
        assert "exec" in cmd
        assert "--model" in cmd
        assert "--full-auto" in cmd
        assert "-C" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/codex")
    def test_includes_prompt_as_last_arg(self, mock_which):
        backend = CodexBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/codex")
    def test_includes_cwd_flag(self, mock_which):
        backend = CodexBackend()
        request = _make_request(cwd="/my/project")

        cmd = backend.build_command(request)

        idx = cmd.index("-C")
        assert cmd[idx + 1] == "/my/project"


class TestCodexBuildEnv:
    def test_returns_empty_dict(self):
        backend = CodexBackend()
        request = _make_request()

        env = backend.build_env(request)

        assert env == {}
