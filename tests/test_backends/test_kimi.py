from unittest.mock import patch

from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.kimi import KimiBackend


_DEFAULT_REQUEST = SpawnRequest(
    agent_id="worker@team",
    name="worker",
    team_name="team",
    prompt="do stuff",
    model="kimi-k2-thinking",
    agent_type="general-purpose",
    color="blue",
    cwd="/tmp/work",
    lead_session_id="sess-1",
)


def _make_request(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
    return replace(_DEFAULT_REQUEST, **overrides)


class TestKimiProperties:
    def test_name_is_kimi(self):
        backend = KimiBackend()
        assert backend.name == "kimi"

    def test_binary_name_is_kimi(self):
        backend = KimiBackend()
        assert backend.binary_name == "kimi"


class TestKimiSupportedModels:
    def test_returns_expected_models(self):
        backend = KimiBackend()
        models = backend.supported_models()
        assert "kimi-k2" in models
        assert "kimi-k2-thinking" in models
        assert "kimi-k2-thinking-turbo" in models

    def test_returns_list(self):
        backend = KimiBackend()
        assert isinstance(backend.supported_models(), list)
        assert len(backend.supported_models()) > 0


class TestKimiDefaultModel:
    def test_returns_kimi_k2_thinking(self):
        backend = KimiBackend()
        assert backend.default_model() == "kimi-k2-thinking"


class TestKimiResolveModel:
    def test_resolves_fast_to_k2(self):
        backend = KimiBackend()
        assert backend.resolve_model("fast") == "kimi-k2"

    def test_resolves_balanced_to_thinking(self):
        backend = KimiBackend()
        assert backend.resolve_model("balanced") == "kimi-k2-thinking"

    def test_resolves_powerful_to_thinking_turbo(self):
        backend = KimiBackend()
        assert backend.resolve_model("powerful") == "kimi-k2-thinking-turbo"

    def test_resolves_direct_model_name(self):
        backend = KimiBackend()
        assert backend.resolve_model("kimi-k2") == "kimi-k2"

    def test_passes_through_unknown_model_name(self):
        backend = KimiBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestKimiBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/kimi")
    def test_produces_print_command(self, _mock_which):
        backend = KimiBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/kimi"
        assert "--print" in cmd
        assert "-p" in cmd
        assert "-m" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/kimi")
    def test_includes_prompt_after_p_flag(self, _mock_which):
        backend = KimiBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        idx = cmd.index("-p")
        assert cmd[idx + 1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/kimi")
    def test_resolves_generic_model(self, _mock_which):
        backend = KimiBackend()
        request = _make_request(model="fast")

        cmd = backend.build_command(request)

        idx = cmd.index("-m")
        assert cmd[idx + 1] == "kimi-k2"


class TestKimiBuildEnv:
    def test_returns_empty_dict(self):
        backend = KimiBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestKimiAvailability:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/kimi")
    def test_available_when_binary_found(self, _mock_which):
        backend = KimiBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = KimiBackend()
        assert backend.is_available() is False
