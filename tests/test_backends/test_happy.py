from unittest.mock import patch

from dataclasses import replace

from claude_teams.backends.base import SpawnRequest

from claude_teams.backends.happy import HappyBackend


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


class TestHappyProperties:
    def test_name_is_happy(self):
        backend = HappyBackend()
        assert backend.name == "happy"

    def test_binary_name_is_happy(self):
        backend = HappyBackend()
        assert backend.binary_name == "happy"


class TestHappySupportedModels:
    def test_returns_expected_models(self):
        backend = HappyBackend()
        models = backend.supported_models()
        assert "haiku" in models
        assert "sonnet" in models
        assert "opus" in models

    def test_returns_list(self):
        backend = HappyBackend()
        assert isinstance(backend.supported_models(), list)
        assert len(backend.supported_models()) > 0


class TestHappyDefaultModel:
    def test_returns_sonnet(self):
        backend = HappyBackend()
        assert backend.default_model() == "sonnet"


class TestHappyResolveModel:
    def test_resolves_fast_to_haiku(self):
        backend = HappyBackend()
        assert backend.resolve_model("fast") == "haiku"

    def test_resolves_balanced_to_sonnet(self):
        backend = HappyBackend()
        assert backend.resolve_model("balanced") == "sonnet"

    def test_resolves_powerful_to_opus(self):
        backend = HappyBackend()
        assert backend.resolve_model("powerful") == "opus"

    def test_resolves_direct_model_name(self):
        backend = HappyBackend()
        assert backend.resolve_model("opus") == "opus"

    def test_passes_through_unknown_model_name(self):
        backend = HappyBackend()
        assert backend.resolve_model("custom-model") == "custom-model"


class TestHappyBuildCommand:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/happy")
    def test_produces_print_command(self, _mock_which):
        backend = HappyBackend()
        request = _make_request()

        cmd = backend.build_command(request)

        assert cmd[0] == "/usr/bin/happy"
        assert "--print" in cmd
        assert "--model" in cmd
        assert "--yolo" in cmd

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/happy")
    def test_includes_prompt_as_last_arg(self, _mock_which):
        backend = HappyBackend()
        request = _make_request(prompt="fix the bug")

        cmd = backend.build_command(request)

        assert cmd[-1] == "fix the bug"

    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/happy")
    def test_resolves_generic_model(self, _mock_which):
        backend = HappyBackend()
        request = _make_request(model="fast")

        cmd = backend.build_command(request)

        idx = cmd.index("--model")
        assert cmd[idx + 1] == "haiku"


class TestHappyBuildEnv:
    def test_returns_empty_dict(self):
        backend = HappyBackend()
        request = _make_request()
        assert backend.build_env(request) == {}


class TestHappyAvailability:
    @patch("claude_teams.backends.base.shutil.which", return_value="/usr/bin/happy")
    def test_available_when_binary_found(self, _mock_which):
        backend = HappyBackend()
        assert backend.is_available() is True

    @patch("claude_teams.backends.base.shutil.which", return_value=None)
    def test_unavailable_when_binary_not_found(self, _mock_which):
        backend = HappyBackend()
        assert backend.is_available() is False
