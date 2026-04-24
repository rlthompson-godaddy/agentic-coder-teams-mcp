"""Bootstrap-tier MCP tools for team lifecycle management."""

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from claude_teams import capabilities, presets, teams, templates
from claude_teams.errors import (
    BackendNotRegisteredError,
    InvalidCapabilityError,
    NoBackendsAvailableError,
    SessionActiveTeamError,
    TeamAlreadyExistsError,
    TeamAlreadyExistsToolError,
    TeamNotFoundToolError,
    UnknownPresetToolError,
)
from claude_teams.models import (
    AgentListResult,
    AgentProfileInfo,
    BackendInfo,
    PresetInfo,
    PresetMemberInfo,
    PresetSpawnResult,
    TeamAttachResult,
    TeamCreateResult,
    TemplateInfo,
)
from claude_teams.orchestration import expand_preset_core
from claude_teams.server_runtime import (
    _ANN_ATTACH,
    _ANN_CREATE,
    _ANN_DELETE,
    _ANN_READ,
    _TAG_BOOTSTRAP,
    _TAG_TEAM,
    _TAG_TEAMMATE,
    _clear_session_principal,
    _get_lifespan,
    _require_lead,
    _resolve_backend_name,
    _resolve_capability,
    _resolve_description,
    _resolve_spawn_cwd,
    _set_session_principal,
)
from claude_teams.server_schema import (
    BackendName,
    Capability,
    Cwd,
    Description,
    PresetName,
    TeamName,
)
from claude_teams.server_team_spawn import _build_spawn_dependencies


async def team_create(
    team_name: TeamName,
    ctx: Context,
    description: Description = "",
) -> dict[str, object]:
    """Create a team and return the lead capability for this session."""
    active_team = await ctx.get_state("active_team")
    if active_team:
        raise SessionActiveTeamError(active_team)
    try:
        result = await teams.create_team(
            name=team_name,
            session_id=ctx.session_id,
            description=_resolve_description(description),
        )
    except TeamAlreadyExistsError as exc:
        raise TeamAlreadyExistsToolError(team_name) from exc
    lead_capability = await capabilities.initialize_team_capabilities(team_name)
    await _set_session_principal(
        ctx,
        team_name,
        "team-lead",
        "lead",
        lead_capability=lead_capability,
    )
    await ctx.enable_components(tags={_TAG_TEAM}, components={"tool", "prompt"})
    return TeamCreateResult(
        team_name=result.team_name,
        team_file_path=result.team_file_path,
        lead_agent_id=result.lead_agent_id,
        lead_capability=lead_capability,
    ).model_dump()


async def team_attach(
    team_name: TeamName,
    capability: Capability,
    ctx: Context,
) -> dict[str, object]:
    """Attach this MCP session to an existing team as lead or agent."""
    resolved_capability = _resolve_capability(capability)
    principal = await capabilities.resolve_principal(team_name, resolved_capability)
    if principal is None:
        raise InvalidCapabilityError()

    active_team = await ctx.get_state("active_team")
    if active_team and active_team != team_name:
        raise SessionActiveTeamError(active_team)

    await _set_session_principal(
        ctx,
        team_name,
        principal["name"],
        principal["role"],
        lead_capability=resolved_capability if principal["role"] == "lead" else None,
    )
    await ctx.enable_components(tags={_TAG_TEAM}, components={"tool", "prompt"})
    if await ctx.get_state("has_teammates"):
        await ctx.enable_components(tags={_TAG_TEAMMATE}, components={"tool"})

    return TeamAttachResult(
        team_name=team_name,
        principal_name=principal["name"],
        principal_role=principal["role"],
    ).model_dump()


async def team_delete(
    team_name: TeamName, ctx: Context, capability: Capability = ""
) -> dict[str, object]:
    """Delete a team and its files. Fails if teammates are still present."""
    await _require_lead(ctx, team_name, capability)
    try:
        result = await teams.delete_team(team_name)
    except (RuntimeError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e
    await _clear_session_principal(ctx)
    await ctx.disable_components(
        tags={_TAG_TEAM, _TAG_TEAMMATE}, components={"tool", "prompt"}
    )
    return result.model_dump()


async def read_config(
    team_name: TeamName, ctx: Context, capability: Capability = ""
) -> dict[str, object]:
    """Read the current team configuration including all members."""
    await _require_lead(ctx, team_name, capability)
    try:
        config = await teams.read_config(team_name)
    except FileNotFoundError:
        raise TeamNotFoundToolError(team_name) from None
    return config.model_dump(by_alias=True)


def list_backends(ctx: Context) -> list[dict[str, object]]:
    """List all available spawner backends with their supported models."""
    ls = _get_lifespan(ctx)
    reg = ls["registry"]
    result = []
    for name, backend_obj in reg:
        info = BackendInfo(
            name=name,
            binary=backend_obj.binary_name,
            available=backend_obj.is_available(),
            default_model=backend_obj.default_model(),
            supported_models=backend_obj.supported_models(),
        )
        result.append(info.model_dump(by_alias=True))
    return result


def list_agents(
    backend_name: BackendName,
    ctx: Context,
    cwd: Cwd = "",
) -> dict[str, object]:
    """List discoverable agent/persona profiles for a backend.

    Returns ``supported=False`` when the backend has no agent-selection
    mechanism; otherwise enumerates profiles visible from ``cwd`` (empty
    string resolves to the server's working directory). Both
    ``backend_name`` and ``cwd`` fall back to their ``CLAUDE_TEAMS_DEFAULT_*``
    env vars when the caller leaves them empty.

    An empty ``backend_name`` after env fallback selects the registry's
    default backend — mirroring the documented ``BackendName`` contract
    and the behaviour of :func:`claude_teams.orchestration._resolve_backend`
    on the spawn path. The concrete backend actually queried is reported
    back on ``AgentListResult.backend`` so clients can tell which one was
    resolved.
    """
    ls = _get_lifespan(ctx)
    reg = ls["registry"]
    resolved_backend = _resolve_backend_name(backend_name)
    try:
        if not resolved_backend:
            resolved_backend = reg.default_backend()
        backend_obj = reg.get(resolved_backend)
    except (NoBackendsAvailableError, BackendNotRegisteredError) as exc:
        raise ToolError(str(exc)) from exc

    resolved_cwd = str(_resolve_spawn_cwd(cwd))

    spec = backend_obj.agent_select_spec()
    if spec is None:
        return AgentListResult(
            backend=resolved_backend,
            supported=False,
            cwd=resolved_cwd,
            profiles=[],
        ).model_dump(by_alias=True)

    profiles = backend_obj.discover_agents(resolved_cwd)
    return AgentListResult(
        backend=resolved_backend,
        supported=True,
        cwd=resolved_cwd,
        profiles=[AgentProfileInfo(name=p.name, path=p.path) for p in profiles],
    ).model_dump(by_alias=True)


async def create_team_from_preset(
    preset_name: PresetName,
    ctx: Context,
    team_name: TeamName,
    description: Description = "",
) -> dict[str, object]:
    """Expand a registered preset into a team + teammates.

    Thin MCP wrapper over
    :func:`claude_teams.orchestration.expand_preset_core`. The wrapper
    owns session-state bookkeeping (``active_team``,
    ``has_teammates``), tier-unlock of the team / teammate tool
    components, and attaching the minted lead capability to the
    FastMCP session — the core owns preset expansion itself.

    The preset's ``description`` is a summary of the preset itself;
    ``team_description`` (or an explicit ``description`` argument) is
    what gets persisted on the created team. Explicit ``description``
    argument overrides the preset's ``team_description``.

    Expansion is not transactional. If a teammate spawn fails mid-fan-out
    the team and any already-spawned teammates persist; the caller sees
    a ``PresetMemberSpawnFailedError`` naming the failed member and can
    either retry the remaining members via ``spawn_teammate`` or tear
    the team down via ``team_delete``. A transactional rollback would
    itself have to handle ``team_delete`` failures, so the shipped
    contract keeps the failure surface simple and observable.
    """
    active_team = await ctx.get_state("active_team")
    if active_team:
        raise SessionActiveTeamError(active_team)

    try:
        preset = presets.get_preset(preset_name)
    except KeyError as exc:
        raise UnknownPresetToolError(preset_name, presets.list_names()) from exc

    effective_description = _resolve_description(description) or preset.team_description

    ls = _get_lifespan(ctx)
    reg = ls["registry"]

    async def _progress(elapsed: int, message: str) -> None:
        await ctx.report_progress(progress=elapsed, total=None, message=message)

    async def _attach_session(lead_capability: str) -> None:
        # Attach before member fan-out so a mid-expansion failure still
        # leaves ``team_delete`` and ``spawn_teammate`` reachable — the
        # non-transactional contract's retry/teardown promises assume
        # the session already holds the lead capability.
        await _set_session_principal(
            ctx, team_name, "team-lead", "lead", lead_capability=lead_capability
        )
        await ctx.enable_components(tags={_TAG_TEAM}, components={"tool", "prompt"})
        await ctx.set_state("has_teammates", True)
        await ctx.enable_components(tags={_TAG_TEAMMATE}, components={"tool"})

    try:
        expansion = await expand_preset_core(
            registry=reg,
            preset=preset,
            team_name=team_name,
            session_id=ctx.session_id,
            description=effective_description,
            deps=_build_spawn_dependencies(),
            progress=_progress,
            on_capability_minted=_attach_session,
        )
    except TeamAlreadyExistsError as exc:
        raise TeamAlreadyExistsToolError(team_name) from exc

    return PresetSpawnResult(
        team=expansion.team,
        members=list(expansion.members),
        preset=preset_name,
    ).model_dump(by_alias=True)


def _preset_to_info(preset: presets.TeamPreset) -> PresetInfo:
    """Project a ``TeamPreset`` into its MCP-visible ``PresetInfo``.

    Omits forward-compat metadata (``skill_roots``, ``mcp_servers``)
    until Feature G consumes it — adding fields later is additive and
    therefore safe for clients that pinned the current shape.
    """
    members = [
        PresetMemberInfo(
            name=m.name,
            prompt=m.prompt,
            template=m.template,
            backend=m.backend,
            model=m.model,
            subagent_type=m.subagent_type,
            reasoning_effort=m.reasoning_effort,
            agent_profile=m.agent_profile,
            cwd=m.cwd,
            plan_mode_required=m.plan_mode_required,
            permission_mode=m.permission_mode,
        )
        for m in preset.members
    ]
    return PresetInfo(
        name=preset.name,
        description=preset.description,
        team_description=preset.team_description,
        members=members,
    )


def list_presets(ctx: Context) -> list[dict[str, object]]:
    """List registered team presets.

    Presets are declarative team compositions: each expands, through
    ``create_team_from_preset``, into a ``team_create`` call followed by
    one ``spawn_teammate`` per listed member. Call this tool to discover
    available preset names before expanding one.
    """
    _ = ctx  # Context is required by FastMCP signature but unused here.
    return [
        _preset_to_info(p).model_dump(by_alias=True) for p in presets.list_presets()
    ]


def list_templates(ctx: Context) -> list[dict[str, object]]:
    """List registered agent templates.

    Templates are reusable role-context layers applied at
    ``spawn_teammate`` time: they prepend a role-prompt header and fill
    in default spawn options for fields the caller did not explicitly
    set. Call this tool first to discover which templates are available,
    then pass a name via ``options.template`` when spawning.
    """
    _ = ctx  # Context is required by FastMCP signature but unused here.
    return [
        TemplateInfo(
            name=t.name,
            description=t.description,
            role_prompt=t.role_prompt,
            default_backend=t.default_backend,
            default_model=t.default_model,
            default_subagent_type=t.default_subagent_type,
            default_reasoning_effort=t.default_reasoning_effort,
            default_agent_profile=t.default_agent_profile,
            default_permission_mode=t.default_permission_mode,
            default_plan_mode_required=t.default_plan_mode_required,
        ).model_dump(by_alias=True)
        for t in templates.list_templates()
    ]


def register_bootstrap_tools(mcp: FastMCP) -> None:
    """Register bootstrap-tier tools on the FastMCP app."""
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_CREATE)(team_create)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_ATTACH)(team_attach)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_DELETE)(team_delete)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_READ)(read_config)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_READ)(list_backends)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_READ)(list_agents)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_READ)(list_templates)
    mcp.tool(tags={_TAG_BOOTSTRAP}, annotations=_ANN_READ)(list_presets)
    mcp.tool(
        tags={_TAG_BOOTSTRAP},
        annotations=_ANN_CREATE,
        timeout=900.0,
    )(create_team_from_preset)
