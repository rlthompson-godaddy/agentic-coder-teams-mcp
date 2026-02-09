from unittest.mock import patch


from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.gemini import GeminiBackend


_DEFAULT_REQUEST = SpawnRequest(
    agent_id="worker@team",
    name="worker",
    team_name="team",
    prompt="do stuff",
    model="gemini-2.5-flash",
    agent_type="general-purpose",
    color="blue",
    cwd="/tmp/work",
    lead_session_id="sess-1",
)


def _make_request(**overrides: str | bool | dict[str, str] | None) -> SpawnRequest:
    return replace(_DEFAULT_REQUEST, **overrides)


class TestGeminiProperties:
    def test_name_is_gemini(self):
        backend = GeminiBackend()
        assert backend.name == "gemini"

    def test_binary_name_is_gemini(self):
        backend = GeminiBackend()
        assert backend.binary_name == "gemini"


class TestGeminiSupportedModels:
    def test_returns_expected_models(self):
        backend = GeminiBackend()
        models = backend.supported_models()
        assert "gemini-2.5-pro" in models
        assert "gemini-2.5-flash" in models
        assert "gemini-2.0-flash" in models


class TestGeminiDefaultModel:
    def test_returns_gemini_2_5_flash(self):
        backend = GeminiBackend()
        assert backend.default_model() == "gemini-2.5-flash"


class TestGeminiResolveModel:
    def test_resolves_fast_to_flash(self):
        backend = GeminiBackend()
        assert backend.resolve_model("fast") == "gemini-2.5-flash"

    def test_resolves_balanced_to_pro(self):
        backend = GeminiBackend()
        assert backend.resolve_model("balanced") == "gemini-2.5-pro"

    def test_resolves_powerful_to_pro(self):
        backend = GeminiBackend()
        assert backend.resolve_model("powerful") == "gemini-2.5-pro"

    def test_resolves_direct_model_name(self):
        backend = GeminiBackend()
        assert backend.resolve_model("gemini-2.0-flash") == "gemini-2.0-flash"

    def test_passes_through_unknown_model_name(self):
        backend = GeminiBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestGeminiBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/gemini")
    def test_produces_correct_flags(self, mock_which):
        backend = GeminiBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/gemini"
        assert "--prompt" in cmd
        assert "--model" in cmd
        assert "--yolo" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/gemini")
    def test_includes_prompt_value(self, mock_which):
        backend = GeminiBackend()
        request = _make_request(prompt="analyze this")

        cmd = backend.build_command(request)

        idx = cmd.index("--prompt")
        assert cmd[idx + 1] == "analyze this"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/gemini")
    def test_includes_model_value(self, mock_which):
        backend = GeminiBackend()
        request = _make_request(model="gemini-2.5-pro")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "gemini-2.5-pro"


class TestGeminiBuildEnv:
    def test_returns_empty_dict(self):
        backend = GeminiBackend()
        request = _make_request()

        env = backend.build_env(request)

        assert env == {}
