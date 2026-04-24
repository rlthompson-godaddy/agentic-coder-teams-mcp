"""Feature H / Slice H.2 — env-var precedence tests.

Locks the ``direct → CLAUDE_TEAMS_*`` precedence chain for every dimension
catalogued in ``CONFIG_PARITY_AUDIT.md``. Coverage is organized by resolver
so failures point directly at the layer that regressed:

- ``TestResolveSpawnCwd`` / ``TestResolveCapability`` / ``TestResolveDescription``
  / ``TestResolveBackendName`` cover the scalar resolvers.
- ``TestParseBoolEnv`` covers the shared bool-env parser.
- ``TestApplySpawnEnvDefaults`` covers the unified ``SpawnOptions`` resolver,
  including the "direct wins over env" ordering.
- ``TestEnvBeatsTemplate`` pins the "env wins over template" composition
  that emerges from ``model_fields_set`` bookkeeping — explicit proof the
  two layers stack correctly rather than relying on it transitively.
- ``TestEnvValidationBypassClosed`` proves env-sourced values hit the same
  pydantic schema constraints as direct args (max_length / pattern /
  min_length). Locks the contract: env is not a validation bypass route.
- ``TestMcpIntegration`` exercises the env layer through the MCP tool surface
  so a missed wire-up in a tool body fails here (not just in a unit test).
"""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from fastmcp import Client

from claude_teams import teams, templates
from claude_teams.backends import registry
from claude_teams.errors import (
    CwdNotAbsoluteError,
    InvalidEnvVarValueError,
    InvalidPermissionModeError,
)
from claude_teams.models import SpawnOptions
from claude_teams.orchestration import apply_template
from claude_teams.server_runtime import (
    _parse_bool_env,
    _resolve_backend_name,
    _resolve_capability,
    _resolve_description,
    _resolve_spawn_cwd,
    apply_spawn_env_defaults,
)
from claude_teams.templates import AgentTemplate
from tests._server_support import _data

# ---------------------------------------------------------------------------
# Scalar resolvers
# ---------------------------------------------------------------------------


class TestResolveSpawnCwd:
    """``_resolve_spawn_cwd`` — direct → env → ``Path.cwd()``."""

    def test_direct_value_takes_precedence_over_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", str(other))
        # Caller passed an explicit value — env is ignored even when set.
        assert _resolve_spawn_cwd(str(tmp_path)) == str(tmp_path)

    def test_env_used_when_direct_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", str(tmp_path))
        assert _resolve_spawn_cwd("") == str(tmp_path)

    def test_falls_back_to_process_cwd_when_neither_set(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("CLAUDE_TEAMS_DEFAULT_CWD", raising=False)
        # Whatever Path.cwd() returns is fine — the contract is "non-empty,
        # current working directory" and not a specific value.
        assert _resolve_spawn_cwd("") == str(Path.cwd())

    def test_env_value_still_validated(self, monkeypatch: pytest.MonkeyPatch):
        # Relative paths are rejected regardless of whether they came from
        # the direct arg or the env — env should not be a bypass route for
        # validation the direct arg also fails.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", "not/absolute")
        with pytest.raises(CwdNotAbsoluteError):
            _resolve_spawn_cwd("")


class TestResolveCapability:
    """``_resolve_capability`` — direct → ``CLAUDE_TEAMS_CAPABILITY`` → ``""``."""

    def test_direct_value_takes_precedence(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_CAPABILITY", "from-env")
        assert _resolve_capability("from-direct") == "from-direct"

    def test_env_used_when_direct_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_CAPABILITY", "from-env")
        assert _resolve_capability("") == "from-env"

    def test_empty_string_when_neither_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CLAUDE_TEAMS_CAPABILITY", raising=False)
        assert _resolve_capability("") == ""


class TestResolveDescription:
    """``_resolve_description`` — direct → ``CLAUDE_TEAMS_DEFAULT_DESCRIPTION``."""

    def test_direct_value_takes_precedence(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_DESCRIPTION", "env desc")
        assert _resolve_description("direct desc") == "direct desc"

    def test_env_used_when_direct_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_DESCRIPTION", "env desc")
        assert _resolve_description("") == "env desc"

    def test_empty_string_when_neither_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CLAUDE_TEAMS_DEFAULT_DESCRIPTION", raising=False)
        assert _resolve_description("") == ""


class TestResolveBackendName:
    """``_resolve_backend_name`` — direct → ``CLAUDE_TEAMS_DEFAULT_BACKEND``."""

    def test_direct_value_takes_precedence(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "codex")
        assert _resolve_backend_name("claude-code") == "claude-code"

    def test_env_used_when_direct_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "codex")
        assert _resolve_backend_name("") == "codex"

    def test_empty_string_when_neither_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CLAUDE_TEAMS_DEFAULT_BACKEND", raising=False)
        assert _resolve_backend_name("") == ""


# ---------------------------------------------------------------------------
# Bool-env parser
# ---------------------------------------------------------------------------


class TestParseBoolEnv:
    """Shared truthy/falsy coercion used by the bool-typed env vars."""

    @pytest.mark.parametrize(
        "raw", ["true", "True", "TRUE", " true ", "1", "yes", "YES", "on"]
    )
    def test_truthy_values(self, raw: str):
        assert _parse_bool_env("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", raw) is True

    @pytest.mark.parametrize(
        "raw", ["false", "False", "FALSE", " false ", "0", "no", "NO", "off"]
    )
    def test_falsy_values(self, raw: str):
        assert _parse_bool_env("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", raw) is False

    def test_unrecognized_raises_typed_error(self):
        with pytest.raises(InvalidEnvVarValueError) as exc_info:
            _parse_bool_env("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", "maybe")
        message = str(exc_info.value)
        # Error carries the env-var name and the offending value so an
        # operator can grep logs for the exact misconfiguration.
        assert "CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED" in message
        assert "'maybe'" in message


# ---------------------------------------------------------------------------
# Unified SpawnOptions resolver
# ---------------------------------------------------------------------------


class TestApplySpawnEnvDefaults:
    """``apply_spawn_env_defaults`` — per-field SpawnOptions env fallback."""

    def test_no_env_returns_same_object(self, monkeypatch: pytest.MonkeyPatch):
        # When no env overrides apply, the resolver short-circuits and
        # returns the same object — avoids a wasted ``model_copy``.
        for key in (
            "CLAUDE_TEAMS_DEFAULT_CWD",
            "CLAUDE_TEAMS_DEFAULT_MODEL",
            "CLAUDE_TEAMS_DEFAULT_BACKEND",
            "CLAUDE_TEAMS_DEFAULT_SUBAGENT_TYPE",
            "CLAUDE_TEAMS_DEFAULT_REASONING_EFFORT",
            "CLAUDE_TEAMS_DEFAULT_AGENT_PROFILE",
            "CLAUDE_TEAMS_DEFAULT_TEMPLATE",
            "CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED",
            "CLAUDE_TEAMS_PERMISSION_MODE",
            "CLAUDE_TEAMS_CAPABILITY",
        ):
            monkeypatch.delenv(key, raising=False)
        opts = SpawnOptions()
        assert apply_spawn_env_defaults(opts) is opts

    @pytest.mark.parametrize(
        ("env_var", "field", "value"),
        [
            ("CLAUDE_TEAMS_DEFAULT_MODEL", "model", "powerful"),
            ("CLAUDE_TEAMS_DEFAULT_BACKEND", "backend", "codex"),
            ("CLAUDE_TEAMS_DEFAULT_SUBAGENT_TYPE", "subagent_type", "code-reviewer"),
            ("CLAUDE_TEAMS_CAPABILITY", "capability", "cap-from-env"),
            ("CLAUDE_TEAMS_DEFAULT_REASONING_EFFORT", "reasoning_effort", "high"),
            ("CLAUDE_TEAMS_DEFAULT_AGENT_PROFILE", "agent_profile", "reviewer"),
            ("CLAUDE_TEAMS_DEFAULT_TEMPLATE", "template", "executor"),
        ],
    )
    def test_env_fills_unset_string_fields(
        self,
        env_var: str,
        field: str,
        value: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv(env_var, value)
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert getattr(resolved, field) == value
        # Env-filled fields enter ``model_fields_set`` so the downstream
        # template layer treats them as explicit.
        assert field in resolved.model_fields_set

    def test_cwd_env_fills_unset_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", str(tmp_path))
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert resolved.cwd == str(tmp_path)

    def test_direct_value_wins_over_env(self, monkeypatch: pytest.MonkeyPatch):
        # Explicit caller value must survive even when every env var is set
        # — "direct" is the top of the precedence chain.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "powerful")
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "codex")
        opts = SpawnOptions(model="fast", backend="claude-code")
        resolved = apply_spawn_env_defaults(opts)
        assert resolved.model == "fast"
        assert resolved.backend == "claude-code"

    def test_permission_mode_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_PERMISSION_MODE", "require_approval")
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert resolved.permission_mode == "require_approval"

    def test_invalid_permission_mode_env_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLAUDE_TEAMS_PERMISSION_MODE", "nonsense-mode")
        # Bad env value surfaces the same typed error as a bad direct arg —
        # operators see a consistent contract regardless of input surface.
        with pytest.raises(InvalidPermissionModeError):
            apply_spawn_env_defaults(SpawnOptions())

    @pytest.mark.parametrize(("raw", "expected"), [("true", True), ("false", False)])
    def test_plan_mode_required_env(
        self, raw: str, expected: bool, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", raw)
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert resolved.plan_mode_required is expected

    def test_invalid_plan_mode_required_env_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", "nope")
        with pytest.raises(InvalidEnvVarValueError):
            apply_spawn_env_defaults(SpawnOptions())

    def test_empty_env_value_does_not_override(self, monkeypatch: pytest.MonkeyPatch):
        # An env var set to "" is the same as absence — the env layer must
        # not clobber the pydantic default with an empty string. Prevents
        # a shell's ``export CLAUDE_TEAMS_DEFAULT_MODEL=`` from silently
        # stripping the "balanced" default.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "")
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert resolved.model == "balanced"

    def test_multiple_env_vars_compose(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Setting several env vars at once must fill every corresponding
        # unset field in one pass — not the first one encountered.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", str(tmp_path))
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "powerful")
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "codex")
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_PLAN_MODE_REQUIRED", "true")
        resolved = apply_spawn_env_defaults(SpawnOptions())
        assert resolved.cwd == str(tmp_path)
        assert resolved.model == "powerful"
        assert resolved.backend == "codex"
        assert resolved.plan_mode_required is True


# ---------------------------------------------------------------------------
# Env beats template composition
# ---------------------------------------------------------------------------


class TestEnvBeatsTemplate:
    """``apply_spawn_env_defaults`` + ``apply_template`` — env wins over template.

    The env layer fills unset fields and the resulting ``SpawnOptions``
    marks them as set in ``model_fields_set``. ``apply_template`` then
    inspects that set and skips any field it would otherwise fill from
    the template's ``default_*``. Without this composition, an operator
    pinning a default via env could still see a template's default win
    silently — the precedence chain the feature advertises would be a
    lie for template-selected spawns.
    """

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        yield
        templates._seed_builtin_templates()

    def test_env_filled_model_blocks_template_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        templates.register_template(
            AgentTemplate(
                name="env-vs-template-model",
                description="Fixture template for env-beats-template test.",
                default_model="template-model-default",
            )
        )
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "env-model")

        env_applied = apply_spawn_env_defaults(
            SpawnOptions(template="env-vs-template-model")
        )
        final_opts, _ = apply_template(env_applied, "do work")

        # Env value survives; template default does not override it.
        assert final_opts.model == "env-model"

    def test_env_filled_backend_blocks_template_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Separate test so a future regression in any one field shows
        # up independently — both assertions share the same failure mode
        # but different env vars and different template fields.
        templates.register_template(
            AgentTemplate(
                name="env-vs-template-backend",
                description="Fixture template for env-beats-template test.",
                default_backend="template-backend-default",
            )
        )
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "env-backend")

        env_applied = apply_spawn_env_defaults(
            SpawnOptions(template="env-vs-template-backend")
        )
        final_opts, _ = apply_template(env_applied, "do work")

        assert final_opts.backend == "env-backend"

    def test_template_still_fills_fields_env_did_not_touch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Negative control: env setting ``model`` must NOT block the
        # template from filling a different field (``subagent_type``).
        # Otherwise "env beats template" would collapse into "any env var
        # disables the entire template", which is not the contract.
        templates.register_template(
            AgentTemplate(
                name="env-partial-template",
                description="Fixture template for env-partial test.",
                default_model="template-model-default",
                default_subagent_type="template-subagent",
            )
        )
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "env-model")

        env_applied = apply_spawn_env_defaults(
            SpawnOptions(template="env-partial-template")
        )
        final_opts, _ = apply_template(env_applied, "do work")

        assert final_opts.model == "env-model"  # env won
        assert final_opts.subagent_type == "template-subagent"  # template won


# ---------------------------------------------------------------------------
# Schema validation on env-sourced values
# ---------------------------------------------------------------------------


class TestEnvValidationBypassClosed:
    """Env-sourced values flow through the same pydantic constraints as direct.

    Copilot review on PR #7 flagged that ``model_copy(update=...)`` inside
    ``apply_spawn_env_defaults`` and the scalar resolvers bypassed schema
    validation — so an oversize env value could land in a ``SpawnOptions``
    or a ``TeamConfig`` without tripping the ``min_length`` / ``max_length``
    / ``pattern`` gates the direct-arg path enforces. These tests lock the
    fix: env bypass is closed at each resolver's boundary.
    """

    def test_oversize_description_env_raises(self, monkeypatch: pytest.MonkeyPatch):
        # ``Description`` schema: max_length=4096.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_DESCRIPTION", "x" * 5000)
        with pytest.raises(InvalidEnvVarValueError) as exc_info:
            _resolve_description("")
        assert "CLAUDE_TEAMS_DEFAULT_DESCRIPTION" in str(exc_info.value)

    def test_oversize_capability_env_raises(self, monkeypatch: pytest.MonkeyPatch):
        # ``Capability`` schema: max_length=512.
        monkeypatch.setenv("CLAUDE_TEAMS_CAPABILITY", "y" * 1000)
        with pytest.raises(InvalidEnvVarValueError) as exc_info:
            _resolve_capability("")
        assert "CLAUDE_TEAMS_CAPABILITY" in str(exc_info.value)

    def test_oversize_backend_name_env_raises(self, monkeypatch: pytest.MonkeyPatch):
        # ``BackendName`` schema: max_length=64.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "b" * 200)
        with pytest.raises(InvalidEnvVarValueError) as exc_info:
            _resolve_backend_name("")
        assert "CLAUDE_TEAMS_DEFAULT_BACKEND" in str(exc_info.value)

    def test_oversize_cwd_env_raises(self, monkeypatch: pytest.MonkeyPatch):
        # ``Cwd`` schema: max_length=4096. Validation now fires before the
        # filesystem checks, so an oversize value fails with a typed error
        # naming the env var instead of a path-not-found downstream.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", "/" + "p" * 5000)
        with pytest.raises(InvalidEnvVarValueError) as exc_info:
            _resolve_spawn_cwd("")
        assert "CLAUDE_TEAMS_DEFAULT_CWD" in str(exc_info.value)

    def test_oversize_spawn_options_model_env_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # ``ModelName`` schema: max_length=128. Exercised through the
        # unified resolver so the ``model_validate``-on-merged-dict path
        # catches and re-raises with the env-var name attached.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "m" * 500)
        with pytest.raises(InvalidEnvVarValueError) as exc_info:
            apply_spawn_env_defaults(SpawnOptions())
        assert "CLAUDE_TEAMS_DEFAULT_MODEL" in str(exc_info.value)

    def test_spawn_options_template_env_pattern_violation_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # ``TemplateName`` schema: pattern ``^[A-Za-z0-9_-]+$``. A value
        # with path-traversal characters must be rejected at env-read
        # time, not at the downstream ``templates.get_template`` lookup.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_TEMPLATE", "../bad")
        with pytest.raises(InvalidEnvVarValueError) as exc_info:
            apply_spawn_env_defaults(SpawnOptions())
        assert "CLAUDE_TEAMS_DEFAULT_TEMPLATE" in str(exc_info.value)

    def test_direct_args_still_accepted_when_env_invalid_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Positive control: the new validation must only fire on
        # env-sourced values. A caller-explicit value stays unaffected
        # even when the same env var isn't set — otherwise the fix
        # would have widened the failure surface beyond env.
        monkeypatch.delenv("CLAUDE_TEAMS_DEFAULT_MODEL", raising=False)
        resolved = apply_spawn_env_defaults(SpawnOptions(model="fast"))
        assert resolved.model == "fast"


# ---------------------------------------------------------------------------
# Integration: env fallback through the MCP tool surface
# ---------------------------------------------------------------------------


class TestMcpIntegration:
    """End-to-end: env fallback reaches the backend call through MCP tools.

    Unit tests above prove the resolvers work in isolation; these tests
    prove the wrappers actually invoke them. A future refactor that removes
    ``apply_spawn_env_defaults`` from a dep factory, or a tool body that
    forgets to resolve its scalar, fails here rather than silently dropping
    the env fallback in production.
    """

    async def test_spawn_teammate_picks_up_env_model_default(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "powerful")
        mock = cast(MagicMock, registry._backends["claude-code"])

        await client.call_tool("team_create", {"team_name": "env-spawn-team"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "env-spawn-team",
                "name": "worker",
                "prompt": "do a thing",
            },
        )

        spawn_request = mock.spawn.call_args_list[0].args[0]
        # ``"powerful"`` resolves to ``"opus"`` via the mock backend's map,
        # so the env default actually flowed all the way to the backend.
        assert spawn_request.model == "opus"

    async def test_spawn_teammate_direct_arg_still_wins_over_env(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_MODEL", "powerful")
        mock = cast(MagicMock, registry._backends["claude-code"])

        await client.call_tool("team_create", {"team_name": "env-direct-win"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "env-direct-win",
                "name": "worker",
                "prompt": "do a thing",
                "options": {"model": "fast"},
            },
        )

        spawn_request = mock.spawn.call_args_list[0].args[0]
        assert spawn_request.model == "haiku"

    async def test_team_create_description_env_fallback(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(
            "CLAUDE_TEAMS_DEFAULT_DESCRIPTION", "Baseline description from env"
        )

        await client.call_tool("team_create", {"team_name": "env-desc-team"})

        cfg = await teams.read_config("env-desc-team")
        assert cfg.description == "Baseline description from env"

    async def test_team_create_direct_description_wins(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_DESCRIPTION", "from env")

        await client.call_tool(
            "team_create",
            {"team_name": "env-desc-direct", "description": "from direct arg"},
        )

        cfg = await teams.read_config("env-desc-direct")
        assert cfg.description == "from direct arg"

    async def test_list_agents_backend_env_fallback(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "claude-code")

        result = _data(await client.call_tool("list_agents", {"backend_name": ""}))

        assert result["backend"] == "claude-code"

    async def test_list_agents_cwd_env_fallback(
        self,
        client: Client,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_CWD", str(tmp_path))

        result = _data(
            await client.call_tool("list_agents", {"backend_name": "claude-code"})
        )

        assert result["cwd"] == str(tmp_path)

    async def test_list_agents_empty_backend_no_env_resolves_to_registry_default(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        # Copilot follow-up on PR #7: ``BackendName`` documents ``""`` as
        # "selects the default backend." ``list_agents`` must honor that
        # contract even after env fallback — otherwise ``backend_name=""``
        # with ``CLAUDE_TEAMS_DEFAULT_BACKEND`` unset would raise
        # ``BackendNotRegisteredError`` instead of querying the default.
        # The response must also report the concrete backend name, not
        # the caller's empty input, so clients can see which one ran.
        monkeypatch.delenv("CLAUDE_TEAMS_DEFAULT_BACKEND", raising=False)

        result = _data(await client.call_tool("list_agents", {"backend_name": ""}))

        # Mock registry seeds ``claude-code`` as the only backend, which
        # makes it the default. Concrete name, not the caller's "".
        assert result["backend"] == "claude-code"

    async def test_list_agents_env_override_beats_default_resolution(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        # Negative control for the default-resolution fix: env-sourced
        # backend must still win over the registry default. Prevents a
        # regression where the new "empty → default" path swallows the
        # env layer entirely.
        monkeypatch.setenv("CLAUDE_TEAMS_DEFAULT_BACKEND", "claude-code")

        result = _data(await client.call_tool("list_agents", {"backend_name": ""}))

        assert result["backend"] == "claude-code"

    async def test_list_agents_empty_backend_no_backends_available_errors(
        self, client: Client, monkeypatch: pytest.MonkeyPatch
    ):
        # Edge case: caller asks for the default backend but the registry
        # has none. Must surface ``ToolError`` (wrapping
        # ``NoBackendsAvailableError``) rather than a silent success with
        # a phantom payload. Locks the full handling of the spawn-path
        # contract ``orchestration._resolve_backend`` already enforces.
        monkeypatch.delenv("CLAUDE_TEAMS_DEFAULT_BACKEND", raising=False)
        registry._backends = {}

        result = await client.call_tool(
            "list_agents", {"backend_name": ""}, raise_on_error=False
        )

        assert result.is_error is True
