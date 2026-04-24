"""Context-free team orchestration core.

This module holds the spawn pipeline and preset expansion logic that
used to live inside the FastMCP tool handler. Extracting it decouples
the business logic from the MCP ``Context``, which lets both the MCP
tools and the Typer CLI drive team orchestration through the same
validated path.

Design notes
------------
- **Dependency injection via explicit parameters.** The caller passes
  the ``BackendRegistry`` in rather than reaching for a global. This
  matches how FastMCP lifespan already hands the registry to MCP
  handlers and keeps the module testable with a fake registry.
- **Optional progress callback.** The MCP surface emits an
  indeterminate-progress heartbeat via ``ctx.report_progress``. The
  core treats that as an optional async callback so the CLI can pass
  ``None`` (or a stdout-printing shim) without pulling in FastMCP.
- **No session state.** Tier unlock (``ctx.enable_components``) and
  ``has_teammates`` bookkeeping stay in the MCP wrapper — they are
  protocol concerns, not domain concerns.
- **Non-transactional preset expansion.** If a mid-fan-out spawn
  fails, the created team and any already-spawned members persist.
  Callers see a ``PresetMemberSpawnFailedError`` naming the failed
  member; the originating error (a ``ToolError`` subclass or a
  domain ``Exception``) is preserved on ``__cause__`` so callers can
  branch on subclass identity when they need to. A transactional
  rollback would itself have to handle ``team_delete`` failures, so
  the shipped contract keeps the failure surface simple and
  observable.
"""

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from fastmcp.exceptions import ToolError

from claude_teams import capabilities, messaging, presets, teams, templates
from claude_teams.async_utils import run_blocking
from claude_teams.backends import SpawnRequest
from claude_teams.backends.contracts import AgentProfile, Backend
from claude_teams.backends.registry import BackendRegistry
from claude_teams.errors import (
    AgentSelectUnsupportedToolError,
    BackendNotRegisteredError,
    BackendSpawnFailedError,
    InvalidReasoningEffortToolError,
    MemberAlreadyExistsError,
    NoBackendsAvailableError,
    PermissionBypassUnsupportedToolError,
    PresetMemberSpawnFailedError,
    ReasoningEffortUnsupportedToolError,
    ReservedAgentNameError,
    UnknownAgentProfileToolError,
    UnknownTemplateToolError,
    UnsupportedBackendModelError,
)
from claude_teams.models import (
    COLOR_PALETTE,
    InboxMessage,
    SpawnOptions,
    SpawnResult,
    TeamCreateResult,
    TeammateMember,
)

PermissionMode = Literal["default", "require_approval", "bypass"]


# Progress callback contract: ``(elapsed_seconds, message)``. Kept
# intentionally narrow so backends can be swapped for stdout, logging,
# or ``ctx.report_progress`` without leaking MCP types into the core.
ProgressCallback = Callable[[int, str], Awaitable[None]]

# Capability-mint hook contract: ``(lead_capability)``. Fires before
# member fan-out so the MCP wrapper can attach the session + unlock
# tools early, keeping ``team_delete`` / ``spawn_teammate`` reachable
# when a later member spawn raises mid-expansion.
CapabilityCallback = Callable[[str], Awaitable[None]]


class RelayOneShotResultCallback(Protocol):
    """Structural type for the one-shot relay callback.

    Mirrors the keyword-only signature of
    ``server_team_relay.relay_one_shot_result`` so the core can invoke
    the relay without reaching for an unconstrained
    ``Callable[..., ...]``. A Protocol lets ty verify both call sites
    (MCP wrapper + CLI factory) wire a compatible function.

    The ``Any`` slots on ``Coroutine`` are forced by the stdlib:
    ``asyncio.create_task`` accepts ``Coroutine[Any, Any, _T]`` because
    the event loop's yield/send types are open-ended. Narrowing those
    parameters here would make the shape unassignable to
    ``create_task`` at the call site.
    """

    def __call__(
        self,
        *,
        team_name: str,
        agent_name: str,
        backend_type: str,
        process_handle: str,
        result_file: Path | None,
        color: str,
    ) -> Coroutine[Any, Any, None]: ...


# Heartbeat cadence. Matches the prior MCP-tool value so perceived
# latency of spawn calls does not change after the refactor.
_HEARTBEAT_INTERVAL_S = 15


@dataclass(frozen=True)
class SpawnDependencies:
    """Bundle of injected helpers the core calls during a spawn.

    Packaging the callbacks into a single object keeps the
    ``spawn_teammate_core`` / ``expand_preset_core`` signatures narrow
    and gives both the MCP wrapper and the CLI one construction site
    for their shared contract. Fields map 1:1 onto helpers that already
    live in ``server_runtime`` / ``server_team_relay`` — injection
    avoids a cycle where ``orchestration`` would otherwise have to
    import those MCP-facing modules.

    Attributes:
        apply_env_defaults: Fills ``SpawnOptions`` fields the caller did
            not explicitly set from ``CLAUDE_TEAMS_DEFAULT_*`` env vars.
            Runs before :func:`apply_template` so the precedence chain is
            ``direct → env → template → pydantic default``. Injected so
            both the MCP spawn path and the CLI preset-expansion path
            apply the same defaults without the core reading env directly.
        resolve_permission_mode: Normalizes the caller-provided
            permission mode against the environment default.
        resolve_spawn_cwd: Validates a spawn cwd and returns the
            absolute path used for backend invocation.
        build_agent_auth_notice: Formats the capability advisory
            appended to the teammate's seed inbox.
        relay_one_shot_result: Coroutine factory that relays a
            one-shot backend's transcript back to the lead inbox.
        create_one_shot_result_path: Builds the per-agent transcript
            path used when a backend writes its result to disk.
        log_relay_task_exception: Done-callback attached to the relay
            task so non-fatal failures are logged rather than lost.
        log_retain_pane_failure: Callback invoked when
            ``Backend.retain_pane_after_exit`` raises. The spawn itself
            has already succeeded by the time this runs, so the failure
            is observability breadcrumb, not a reason to fail the call.
            Injected rather than hardcoded so the orchestration core
            stays free of logging-sink imports.

    """

    apply_env_defaults: Callable[[SpawnOptions], SpawnOptions]
    resolve_permission_mode: Callable[[PermissionMode | None], PermissionMode]
    resolve_spawn_cwd: Callable[[str], str]
    build_agent_auth_notice: Callable[[str, str], str]
    relay_one_shot_result: RelayOneShotResultCallback
    create_one_shot_result_path: Callable[[str, str], Path]
    log_relay_task_exception: Callable[[asyncio.Task[None]], None]
    log_retain_pane_failure: Callable[[BaseException], None]


@dataclass(frozen=True)
class PresetExpansionResult:
    """Aggregate result of expanding a preset into a running team.

    Attributes:
        team: Team creation result returned by ``teams.create_team``.
        lead_capability: Lead-scoped capability token minted during
            team initialization. The MCP wrapper attaches this to the
            session; the CLI prints it so the operator can export it.
        members: Per-member spawn results in preset declaration order.

    """

    team: TeamCreateResult
    lead_capability: str
    members: tuple[SpawnResult, ...]


def _resolve_backend(registry: BackendRegistry, backend_name: str | None) -> Backend:
    """Return the registry entry for ``backend_name`` or the default.

    Moved out of the MCP tool so both call sites raise the same
    ``ToolError`` shape when the registry is empty or the name is not
    registered.

    Raises:
        ToolError: Named backend is not registered, or the registry is
            empty and no default can be chosen.

    """
    if backend_name:
        try:
            return registry.get(backend_name)
        except BackendNotRegisteredError as exc:
            raise ToolError(str(exc)) from exc
    try:
        return registry.get(registry.default_backend())
    except (NoBackendsAvailableError, BackendNotRegisteredError) as exc:
        raise ToolError(str(exc)) from exc


def apply_template(opts: SpawnOptions, prompt: str) -> tuple[SpawnOptions, str]:
    r"""Resolve ``opts.template`` and layer its defaults onto the request.

    Precedence rules:

    - **Options** — template defaults fill in only where
      ``opts.model_fields_set`` indicates the caller did not explicitly
      pass the field.
    - **Prompt** — the template's ``role_prompt`` is prepended with a
      blank-line separator, forming ``{role_prompt}\n\n{user_prompt}``.
      Backend-required system wrappers applied later remain
      authoritative.

    Raises:
        UnknownTemplateToolError: Template name is not in the registry.

    """
    if opts.template is None:
        return opts, prompt

    try:
        template = templates.get_template(opts.template)
    except KeyError as exc:
        raise UnknownTemplateToolError(opts.template, templates.list_names()) from exc

    explicit = opts.model_fields_set
    updates: dict[str, str | bool] = {}
    if "backend" not in explicit and template.default_backend is not None:
        updates["backend"] = template.default_backend
    if "model" not in explicit and template.default_model is not None:
        updates["model"] = template.default_model
    if "subagent_type" not in explicit and template.default_subagent_type is not None:
        updates["subagent_type"] = template.default_subagent_type
    if (
        "reasoning_effort" not in explicit
        and template.default_reasoning_effort is not None
    ):
        updates["reasoning_effort"] = template.default_reasoning_effort
    if "agent_profile" not in explicit and template.default_agent_profile is not None:
        updates["agent_profile"] = template.default_agent_profile
    if (
        "permission_mode" not in explicit
        and template.default_permission_mode is not None
    ):
        updates["permission_mode"] = template.default_permission_mode
    if (
        "plan_mode_required" not in explicit
        and template.default_plan_mode_required is not None
    ):
        updates["plan_mode_required"] = template.default_plan_mode_required

    merged = opts.model_copy(update=updates) if updates else opts
    composed = f"{template.role_prompt}\n\n{prompt}" if template.role_prompt else prompt
    return merged, composed


async def _validate_reasoning_effort(backend_obj: Backend, effort: str | None) -> None:
    """Validate reasoning_effort against the backend's spec.

    Raises:
        ReasoningEffortUnsupportedToolError: Backend does not expose a spec.
        InvalidReasoningEffortToolError: Value is outside the spec's options.

    """
    if effort is None:
        return
    spec = await run_blocking(backend_obj.reasoning_effort_spec)
    if spec is None:
        raise ReasoningEffortUnsupportedToolError(backend_obj.name)
    if effort not in spec.options:
        raise InvalidReasoningEffortToolError(backend_obj.name, effort, spec.options)


async def _validate_agent_profile(
    backend_obj: Backend, profile: str | None, cwd: str
) -> AgentProfile | None:
    """Validate agent_profile against the backend's spec and discovery.

    Returns the matched profile so the caller can thread its path
    through ``SpawnRequest.extra`` without re-running ``discover_agents``
    at command-build time.

    Raises:
        AgentSelectUnsupportedToolError: Backend lacks an agent_select_spec.
        UnknownAgentProfileToolError: Name not among discovered profiles.

    """
    if profile is None:
        return None
    spec = await run_blocking(backend_obj.agent_select_spec)
    if spec is None:
        raise AgentSelectUnsupportedToolError(backend_obj.name)
    profiles = await run_blocking(backend_obj.discover_agents, cwd)
    for discovered in profiles:
        if discovered.name == profile:
            return discovered
    names = [p.name for p in profiles]
    raise UnknownAgentProfileToolError(backend_obj.name, profile, names)


async def _assign_color(team_name: str) -> str:
    """Return the next teammate color for the team."""
    config = await teams.read_config(team_name)
    teammate_count = sum(
        1 for member in config.members if isinstance(member, TeammateMember)
    )
    return COLOR_PALETTE[teammate_count % len(COLOR_PALETTE)]


async def spawn_teammate_core(
    *,
    registry: BackendRegistry,
    team_name: str,
    name: str,
    prompt: str,
    options: SpawnOptions,
    lead_session_id: str,
    deps: SpawnDependencies,
    progress: ProgressCallback | None = None,
) -> SpawnResult:
    """Validate, persist, and launch a new teammate.

    This is the orchestration core shared by the MCP
    ``spawn_teammate`` tool and the CLI ``preset launch`` command.
    It performs (in order):

    1. template resolution,
    2. backend / model / permission-mode resolution and validation,
    3. ``reasoning_effort`` and ``agent_profile`` validation,
    4. color assignment and member persistence,
    5. capability issue + inbox seed,
    6. ``backend.spawn()`` invocation under an optional heartbeat,
    7. process-handle writeback,
    8. one-shot relay scheduling for non-interactive backends.

    The helper callbacks bundled into ``deps`` are injected rather
    than imported directly to avoid a cycle between this module and
    ``server_runtime``/``server_team_relay``; both callers own those
    helpers already.

    Args:
        registry: Backend registry the caller has constructed or
            received from the lifespan.
        team_name: Target team name.
        name: Member name; rejected if equal to the reserved
            ``team-lead``.
        prompt: Caller-provided task prompt. May be augmented with a
            role-prompt header via ``options.template``.
        options: Spawn tuning knobs. Field-set semantics are preserved.
        lead_session_id: Session identifier recorded on the spawn
            request so backends can route lead-facing traffic.
        deps: Bundle of injected helpers; see ``SpawnDependencies``.
        progress: Optional heartbeat callback invoked every
            ``_HEARTBEAT_INTERVAL_S`` seconds during the spawn. Omit
            in contexts that do not surface progress (e.g., CLI).

    Returns:
        SpawnResult: populated with the new member's agent id, name,
            and team name.

    Raises:
        ToolError: Backend not registered, unsupported model, or the
            chosen member name already exists on the team (the domain
            ``MemberAlreadyExistsError`` is surfaced as ``ToolError``).
            Every named subclass below is also a ``ToolError``, so
            callers that only care about the MCP-safe shape can catch
            ``ToolError`` alone. ``BackendSpawnFailedError`` is also a
            ``ToolError`` but listed separately because its trigger
            (backend-side failure) differs from the validation errors.
        UnknownTemplateToolError: ``options.template`` names a
            template that is not in the template registry. Raised by
            ``apply_template`` before any backend work begins.
        PermissionBypassUnsupportedToolError: Resolved mode is
            ``bypass`` but the backend cannot honor it.
        ReasoningEffortUnsupportedToolError /
        InvalidReasoningEffortToolError: effort validation failures.
        AgentSelectUnsupportedToolError /
        UnknownAgentProfileToolError: profile validation failures.
        ReservedAgentNameError: Caller supplied ``team-lead``.
        BackendSpawnFailedError: Backend ``spawn()`` raised; the
            partially-persisted member and capability are rolled back
            before the error propagates.

    """
    opts = deps.apply_env_defaults(options)
    opts, prompt = apply_template(opts, prompt)

    backend_obj = _resolve_backend(registry, opts.backend)

    try:
        resolved_model = await run_blocking(backend_obj.resolve_model, opts.model)
    except UnsupportedBackendModelError as exc:
        raise ToolError(str(exc)) from exc

    resolved_permission_mode = deps.resolve_permission_mode(opts.permission_mode)
    if resolved_permission_mode == "bypass" and not await run_blocking(
        backend_obj.supports_permission_bypass
    ):
        raise PermissionBypassUnsupportedToolError(backend_obj.name)

    await _validate_reasoning_effort(backend_obj, opts.reasoning_effort)

    if name == "team-lead":
        raise ReservedAgentNameError()

    resolved_cwd = await run_blocking(deps.resolve_spawn_cwd, opts.cwd)
    resolved_profile = await _validate_agent_profile(
        backend_obj, opts.agent_profile, resolved_cwd
    )
    color = await _assign_color(team_name)

    member = TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type=opts.subagent_type,
        model=resolved_model,
        prompt=prompt,
        color=color,
        plan_mode_required=opts.plan_mode_required,
        joined_at=int(time.time() * 1000),
        tmux_pane_id="",
        cwd=resolved_cwd,
        backend_type=backend_obj.name,
        is_active=False,
        process_handle="",
    )

    try:
        await teams.add_member(team_name, member)
    except MemberAlreadyExistsError as exc:
        raise ToolError(str(exc)) from exc

    agent_capability = await capabilities.issue_agent_capability(team_name, name)
    await messaging.ensure_inbox(team_name, name)
    await messaging.append_message(
        team_name,
        name,
        InboxMessage(
            from_="team-lead",
            text=prompt + deps.build_agent_auth_notice(team_name, agent_capability),
            timestamp=messaging.now_iso(),
            read=False,
        ),
    )

    one_shot_result_path: Path | None = None
    extra: dict[str, str] = {"agent_capability": agent_capability}
    if backend_obj.name == "codex":
        one_shot_result_path = deps.create_one_shot_result_path(team_name, name)
        extra["output_last_message_path"] = str(one_shot_result_path)
    if resolved_profile is not None:
        extra["agent_profile_path"] = resolved_profile.path

    request = SpawnRequest(
        agent_id=member.agent_id,
        name=name,
        team_name=team_name,
        prompt=prompt,
        model=resolved_model,
        agent_type=opts.subagent_type,
        color=color,
        cwd=resolved_cwd,
        lead_session_id=lead_session_id,
        permission_mode=resolved_permission_mode,
        plan_mode_required=opts.plan_mode_required,
        reasoning_effort=opts.reasoning_effort,
        agent_profile=opts.agent_profile,
        extra=extra,
    )

    heartbeat_task: asyncio.Task[None] | None = None
    if progress is not None:

        async def _heartbeat() -> None:
            elapsed = 0
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
                elapsed += _HEARTBEAT_INTERVAL_S
                await progress(
                    elapsed,
                    f"Still spawning {backend_obj.name!r} for {name!r} "
                    f"({elapsed}s elapsed)...",
                )

        heartbeat_task = asyncio.create_task(_heartbeat())

    try:
        spawn_result = await run_blocking(backend_obj.spawn, request)
    except Exception as exc:
        await teams.remove_member(team_name, name)
        await capabilities.remove_agent_capability(team_name, name)
        raise BackendSpawnFailedError(exc) from exc
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    config = await teams.read_config(team_name)
    for config_member in config.members:
        if isinstance(config_member, TeammateMember) and config_member.name == name:
            config_member.process_handle = spawn_result.process_handle
            config_member.tmux_pane_id = spawn_result.process_handle
            break
    await teams.write_config(team_name, config)

    if not backend_obj.is_interactive:
        try:
            await run_blocking(
                backend_obj.retain_pane_after_exit, spawn_result.process_handle
            )
        except Exception as exc:
            deps.log_retain_pane_failure(exc)
        relay_task = asyncio.create_task(
            deps.relay_one_shot_result(
                team_name=team_name,
                agent_name=name,
                backend_type=backend_obj.name,
                process_handle=spawn_result.process_handle,
                result_file=one_shot_result_path,
                color=color,
            )
        )
        relay_task.add_done_callback(deps.log_relay_task_exception)

    return SpawnResult(
        agent_id=member.agent_id,
        name=member.name,
        team_name=team_name,
    )


async def expand_preset_core(
    *,
    registry: BackendRegistry,
    preset: presets.TeamPreset,
    team_name: str,
    session_id: str,
    description: str,
    deps: SpawnDependencies,
    progress: ProgressCallback | None = None,
    on_capability_minted: CapabilityCallback | None = None,
) -> PresetExpansionResult:
    """Expand a preset into a new team plus one teammate per member.

    Expansion is not transactional once fan-out begins. On a
    mid-fan-out failure the team and any already-spawned members
    persist so the caller can retry the remaining members or tear the
    team down. The caller receives a ``PresetMemberSpawnFailedError``
    naming the failing member; the originating error (a ``ToolError``
    subclass or a domain ``Exception``) is preserved on ``__cause__``.

    Setup failures (lead-capability initialization or the
    ``on_capability_minted`` callback) are different: the team has
    nothing useful on it yet, so we roll the team deletion inline to
    avoid leaving a dangling on-disk config.

    Args:
        registry: Backend registry (same shape as for
            ``spawn_teammate_core``).
        preset: Already-resolved ``TeamPreset``.
        team_name: Team name to create.
        session_id: Session identifier the team is scoped to.
        description: Team description. Callers should resolve the
            explicit-arg-vs-preset-default precedence before calling.
        deps: Bundle of injected helpers; see ``SpawnDependencies``.
        progress: Optional heartbeat callback forwarded to each
            per-member spawn.
        on_capability_minted: Optional async hook invoked with the
            lead capability token immediately after it is minted,
            before member fan-out begins. The MCP wrapper uses this to
            attach the session as lead and unlock team/teammate tools
            early so callers can still reach ``team_delete`` /
            ``spawn_teammate`` when a later member spawn fails
            mid-expansion. Raising from the callback triggers the
            same team-rollback as a capability-init failure.

    Returns:
        ``PresetExpansionResult`` with the team result, the lead
        capability, and per-member spawn results in preset order.

    Raises:
        TeamAlreadyExistsError: A team with ``team_name`` already
            exists. Surfaced directly (not wrapped) because no partial
            team state needs reporting — expansion never started.
        PresetMemberSpawnFailedError: A member spawn inside the fan-out
            raised. The wrapper carries the failing member name; the
            original error is preserved on ``__cause__``.

    """
    team_result = await teams.create_team(
        name=team_name,
        session_id=session_id,
        description=description,
    )
    try:
        lead_capability = await capabilities.initialize_team_capabilities(team_name)
        if on_capability_minted is not None:
            await on_capability_minted(lead_capability)
    except Exception:
        # Team was created but setup (capability init or the early-attach
        # callback) failed: the team is unusable yet persists on disk.
        # Roll the team deletion so ``preset launch`` stays idempotent
        # over retries.
        with contextlib.suppress(Exception):
            await teams.delete_team(team_name)
        raise

    member_results: list[SpawnResult] = []
    for member in preset.members:
        options = _preset_member_spawn_options(member)
        try:
            spawn_result = await spawn_teammate_core(
                registry=registry,
                team_name=team_name,
                name=member.name,
                prompt=member.prompt,
                options=options,
                lead_session_id=session_id,
                deps=deps,
                progress=progress,
            )
        except Exception as exc:
            # Observability: wrap so the failing member's name lands in
            # the outer message instead of only on ``__cause__``.
            raise PresetMemberSpawnFailedError(member.name, exc) from exc
        member_results.append(spawn_result)

    return PresetExpansionResult(
        team=team_result,
        lead_capability=lead_capability,
        members=tuple(member_results),
    )


def _preset_member_spawn_options(
    member: presets.PresetMemberSpec,
) -> SpawnOptions:
    """Translate a preset member entry into ``SpawnOptions``.

    Only fields the preset actually sets land in the resulting
    options' ``model_fields_set`` so ``apply_template`` sees the same
    "explicit vs defaulted" shape a direct MCP caller would produce.

    """
    updates: dict[str, str | bool] = {}
    if member.template is not None:
        updates["template"] = member.template
    if member.backend is not None:
        updates["backend"] = member.backend
    if member.model is not None:
        updates["model"] = member.model
    if member.subagent_type is not None:
        updates["subagent_type"] = member.subagent_type
    if member.reasoning_effort is not None:
        updates["reasoning_effort"] = member.reasoning_effort
    if member.agent_profile is not None:
        updates["agent_profile"] = member.agent_profile
    if member.cwd is not None:
        updates["cwd"] = member.cwd
    if member.plan_mode_required is not None:
        updates["plan_mode_required"] = member.plan_mode_required
    if member.permission_mode is not None:
        updates["permission_mode"] = member.permission_mode
    base = SpawnOptions()
    return base.model_copy(update=updates) if updates else base


__all__ = [
    "CapabilityCallback",
    "PermissionMode",
    "PresetExpansionResult",
    "ProgressCallback",
    "apply_template",
    "expand_preset_core",
    "spawn_teammate_core",
]
