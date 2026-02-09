from unittest.mock import MagicMock, patch

import pytest

from claude_teams.backends.base import Backend
from claude_teams.backends.registry import BackendRegistry


def _make_mock_backend(
    name: str = "mock", binary: str = "mock-cli", available: bool = True
) -> MagicMock:
    """Create a mock backend that satisfies the Backend protocol."""
    mock = MagicMock(spec=Backend)
    mock.name = name
    mock.binary_name = binary
    mock.is_available.return_value = available
    mock.supported_models.return_value = ["default"]
    mock.default_model.return_value = "default"
    return mock


class TestRegistryEmpty:
    def test_list_available_returns_empty_when_no_binaries_on_path(self):
        reg = BackendRegistry()
        with patch("claude_teams.backends.base.shutil.which", return_value=None):
            result = reg.list_available()
        assert result == []

    def test_default_backend_raises_when_none_available(self):
        reg = BackendRegistry()
        with patch("claude_teams.backends.base.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="No backends available"):
                reg.default_backend()


class TestRegistryDiscovery:
    def test_discovers_claude_code_when_binary_present(self):
        reg = BackendRegistry()

        def mock_which(name):
            return "/usr/bin/claude" if name == "claude" else None

        with patch("claude_teams.backends.base.shutil.which", side_effect=mock_which):
            available = reg.list_available()

        assert "claude-code" in available

    def test_discovers_multiple_backends_when_binaries_present(self):
        reg = BackendRegistry()

        def mock_which(name):
            mapping = {"claude": "/usr/bin/claude", "codex": "/usr/bin/codex"}
            return mapping.get(name)

        with patch("claude_teams.backends.base.shutil.which", side_effect=mock_which):
            available = reg.list_available()

        assert "claude-code" in available
        assert "codex" in available


class TestRegistryManualRegister:
    def test_register_and_get(self):
        reg = BackendRegistry()
        mock = _make_mock_backend("custom")
        reg.register("custom", mock)

        result = reg.get("custom")
        assert result is mock

    def test_get_raises_key_error_for_unknown_name(self):
        reg = BackendRegistry()
        # Force loaded state so it doesn't try auto-discovery
        reg._loaded = True
        with pytest.raises(KeyError, match="unknown"):
            reg.get("unknown")

    def test_register_overwrites_existing(self):
        reg = BackendRegistry()
        mock1 = _make_mock_backend("x")
        mock2 = _make_mock_backend("x")
        reg.register("x", mock1)
        reg.register("x", mock2)
        assert reg.get("x") is mock2


class TestRegistryDefaultBackend:
    def test_returns_claude_code_when_available(self):
        reg = BackendRegistry()
        reg._loaded = True
        mock_cc = _make_mock_backend("claude-code")
        mock_codex = _make_mock_backend("codex")
        reg._backends = {"claude-code": mock_cc, "codex": mock_codex}

        assert reg.default_backend() == "claude-code"

    def test_returns_first_available_when_claude_code_unavailable(self):
        reg = BackendRegistry()
        reg._loaded = True
        mock_codex = _make_mock_backend("codex")
        mock_gemini = _make_mock_backend("gemini")
        reg._backends = {"codex": mock_codex, "gemini": mock_gemini}

        result = reg.default_backend()
        # sorted: codex < gemini
        assert result == "codex"

    def test_raises_runtime_error_when_none_available(self):
        reg = BackendRegistry()
        reg._loaded = True
        reg._backends = {}

        with pytest.raises(RuntimeError, match="No backends available"):
            reg.default_backend()


class TestRegistryListAvailable:
    def test_returns_sorted_names(self):
        reg = BackendRegistry()
        reg._loaded = True
        reg._backends = {
            "codex": _make_mock_backend("codex"),
            "aider": _make_mock_backend("aider"),
            "claude-code": _make_mock_backend("claude-code"),
        }

        result = reg.list_available()
        assert result == ["aider", "claude-code", "codex"]


class TestRegistryIter:
    def test_yields_name_backend_tuples(self):
        reg = BackendRegistry()
        reg._loaded = True
        mock_a = _make_mock_backend("a")
        mock_b = _make_mock_backend("b")
        reg._backends = {"a": mock_a, "b": mock_b}

        items = list(reg)
        assert ("a", mock_a) in items
        assert ("b", mock_b) in items
        assert len(items) == 2


class TestRegistryLazyLoading:
    def test_ensure_loaded_called_only_once(self):
        reg = BackendRegistry()
        reg._loaded = True
        mock = _make_mock_backend("x")
        reg._backends = {"x": mock}

        # Multiple accesses should not re-load
        reg.list_available()
        reg.list_available()
        assert reg._loaded is True

    def test_ensure_loaded_sets_flag(self):
        reg = BackendRegistry()
        assert reg._loaded is False
        with patch("claude_teams.backends.base.shutil.which", return_value=None):
            reg._ensure_loaded()
        assert reg._loaded is True


class TestRegistryEntryPoints:
    def test_loads_entry_point_backend(self):
        reg = BackendRegistry()

        mock_ep = MagicMock()
        mock_ep.name = "custom-ep"
        mock_cls = MagicMock()
        mock_instance = _make_mock_backend("custom-ep")
        mock_cls.return_value = mock_instance
        mock_ep.load.return_value = mock_cls

        with patch("claude_teams.backends.base.shutil.which", return_value=None):
            with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
                available = reg.list_available()

        assert "custom-ep" in available

    def test_skips_entry_point_when_already_registered(self):
        reg = BackendRegistry()

        # Pre-register a backend with same name
        existing = _make_mock_backend("claude-code")
        reg._backends["claude-code"] = existing

        mock_ep = MagicMock()
        mock_ep.name = "claude-code"

        def mock_which(name):
            return "/usr/bin/claude" if name == "claude" else None

        with patch("claude_teams.backends.base.shutil.which", side_effect=mock_which):
            with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
                reg._ensure_loaded()

        # Entry point load() should not have been called
        mock_ep.load.assert_not_called()
