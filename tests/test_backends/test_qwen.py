from unittest.mock import patch

from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.qwen import QwenBackend


_DEFAULT_REQUEST = SpawnRequest(
    agent_id="worker@team",
    name="worker",
    team_name="team",
    prompt="do stuff",
    model="qwen-plus",
    agent_type="general-purpose",
    color="blue",
    cwd="/tmp/work",
    lead_session_id="sess-1",
)


def _make_request(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
    return replace(_DEFAULT_REQUEST, **overrides)


class TestQwenProperties:
    def test_name_is_qwen(self):
        backend = QwenBackend()
        assert backend.name == "qwen"

    def test_binary_name_is_qwen(self):
        backend = QwenBackend()
        assert backend.binary_name == "qwen"


class TestQwenSupportedModels:
    def test_returns_expected_models(self):
        backend = QwenBackend()
        models = backend.supported_models()
        assert "qwen-turbo" in models
        assert "qwen-plus" in models
        assert "qwen-max" in models
        assert "claude-sonnet-4.5" in models
        assert "gpt-5.2-codex" in models

    def test_returns_list(self):
        backend = QwenBackend()
        assert isinstance(backend.supported_models(), list)
        assert len(backend.supported_models()) > 0


class TestQwenDefaultModel:
    def test_returns_qwen_plus(self):
        backend = QwenBackend()
        assert backend.default_model() == "qwen-plus"


class TestQwenResolveModel:
    def test_resolves_fast_to_turbo(self):
        backend = QwenBackend()
        assert backend.resolve_model("fast") == "qwen-turbo"

    def test_resolves_balanced_to_plus(self):
        backend = QwenBackend()
        assert backend.resolve_model("balanced") == "qwen-plus"

    def test_resolves_powerful_to_max(self):
        backend = QwenBackend()
        assert backend.resolve_model("powerful") == "qwen-max"

    def test_resolves_direct_model_name(self):
        backend = QwenBackend()
        assert backend.resolve_model("claude-sonnet-4.5") == "claude-sonnet-4.5"

    def test_passes_through_unknown_model_name(self):
        backend = QwenBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestQwenBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/qwen")
    def test_produces_prompt_command(self, _mock_which):
        backend = QwenBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/qwen"
        assert "-p" in cmd
        assert "-m" in cmd
        assert "-y" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/qwen")
    def test_includes_prompt_after_p_flag(self, _mock_which):
        backend = QwenBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        idx = cmd.index("-p")
        assert cmd[idx + 1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/qwen")
    def test_includes_model_after_m_flag(self, _mock_which):
        backend = QwenBackend()
        request = _make_request(model="qwen-max")

        cmd = backend.build_command(request)

        idx = cmd.index("-m")
        assert cmd[idx + 1] == "qwen-max"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/qwen")
    def test_resolves_generic_model(self, _mock_which):
        backend = QwenBackend()
        request = _make_request(model="fast")

        cmd = backend.build_command(request)

        idx = cmd.index("-m")
        assert cmd[idx + 1] == "qwen-turbo"


class TestQwenBuildEnv:
    def test_returns_empty_dict(self):
        backend = QwenBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestQwenAvailability:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/qwen")
    def test_available_when_binary_found(self, _mock_which):
        backend = QwenBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = QwenBackend()
        assert backend.is_available() is False
