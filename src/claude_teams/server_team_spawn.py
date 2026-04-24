"""Team-tier spawn and messaging MCP tools."""

from typing import Literal

from fastmcp import Context, FastMCP

from claude_teams import messaging, teams
from claude_teams.errors import (
    BroadcastSenderError,
    BroadcastSummaryEmptyToolError,
    BroadcastTooManyRecipientsError,
    MessageContentEmptyToolError,
    MessageRecipientEmptyToolError,
    MessageSummaryEmptyToolError,
    NotTeamMemberError,
    PlanRecipientEmptyToolError,
    ShutdownRecipientEmptyToolError,
    ShutdownResponseApprovalRequiredError,
    ShutdownSelfError,
    UnknownMessageTypeError,
)
from claude_teams.models import (
    MemberUnion,
    MessageRouting,
    SendMessageResult,
    ShutdownApproved,
    SpawnOptions,
    TeammateMember,
)
from claude_teams.orchestration import (
    SpawnDependencies,
    spawn_teammate_core,
)
from claude_teams.server_runtime import (
    _ANN_CREATE,
    _ANN_MUTATE,
    _TAG_TEAM,
    _TAG_TEAMMATE,
    _get_lifespan,
    _require_lead,
    _require_sender_or_lead,
    _resolve_permission_mode,
    _resolve_spawn_cwd,
    apply_spawn_env_defaults,
)
from claude_teams.server_schema import (
    AgentName,
    Capability,
    MessageContent,
    MessageSummary,
    Prompt,
    RequestId,
    SenderName,
    TeamName,
)
from claude_teams.server_team_relay import (
    build_agent_auth_notice,
    create_one_shot_result_path,
    log_relay_task_exception,
    log_retain_pane_failure,
    relay_one_shot_result,
)


def _find_member(config: teams.TeamConfig, member_name: str) -> MemberUnion | None:
    """Find a team member by name.

    Args:
        config (teams.TeamConfig): Team configuration.
        member_name (str): Member name to find.

    Returns:
        teams.MemberUnion | None: Matching member or None.

    """
    for member in config.members:
        if member.name == member_name:
            return member
    return None


def _build_spawn_dependencies() -> SpawnDependencies:
    """Bundle the MCP-owned helpers into ``SpawnDependencies``.

    The core needs injection rather than direct imports to keep it free
    of any FastMCP coupling; this factory lives in the MCP wrapper
    precisely so those imports — and their types — stay on this side.
    """
    return SpawnDependencies(
        apply_env_defaults=apply_spawn_env_defaults,
        resolve_permission_mode=_resolve_permission_mode,
        resolve_spawn_cwd=_resolve_spawn_cwd,
        build_agent_auth_notice=build_agent_auth_notice,
        relay_one_shot_result=relay_one_shot_result,
        create_one_shot_result_path=create_one_shot_result_path,
        log_relay_task_exception=log_relay_task_exception,
        log_retain_pane_failure=log_retain_pane_failure,
    )


async def spawn_teammate_tool(
    team_name: TeamName,
    name: AgentName,
    prompt: Prompt,
    ctx: Context,
    options: SpawnOptions | None = None,
) -> dict[str, object]:
    """Spawn a teammate via the selected backend.

    This is the thin MCP wrapper around
    :func:`claude_teams.orchestration.spawn_teammate_core`. The wrapper
    owns MCP-protocol concerns (auth, session state, progress relay);
    the core owns validation, persistence, and the backend call.

    Emits an indeterminate-progress heartbeat every 15 seconds while the
    backend spawn is in flight. One-shot backends (codex, qwen, gemini)
    can take several minutes to return; the heartbeat prevents the call
    from appearing silent. Quick spawns (interactive backends) finish
    before the first tick fires, so short-lived spawns emit no progress
    events at all — the request/response lifecycle is enough for those.

    Args:
        team_name: Team the new teammate joins.
        name: Member name for the new teammate (must be unique within the team
            and not equal to the reserved ``team-lead``).
        prompt: Initial prompt delivered to the teammate via its inbox.
        ctx: FastMCP request context (injected).
        options: Optional tuning knobs (backend, model, cwd,
            subagent_type, reasoning_effort, agent_profile,
            plan_mode_required, permission_mode, template, capability).
            Omit to accept all defaults.

    """
    opts = options or SpawnOptions()

    await _require_lead(ctx, team_name, opts.capability)
    ls = _get_lifespan(ctx)
    reg = ls["registry"]

    async def _progress(elapsed: int, message: str) -> None:
        await ctx.report_progress(progress=elapsed, total=None, message=message)

    spawn_result = await spawn_teammate_core(
        registry=reg,
        team_name=team_name,
        name=name,
        prompt=prompt,
        options=opts,
        lead_session_id=ctx.session_id,
        deps=_build_spawn_dependencies(),
        progress=_progress,
    )

    if not await ctx.get_state("has_teammates"):
        await ctx.set_state("has_teammates", True)
        await ctx.enable_components(tags={_TAG_TEAMMATE}, components={"tool"})

    return spawn_result.model_dump()


async def _handle_plain_message(
    team_name: str,
    ctx: Context,
    recipient: str,
    content: str,
    summary: str,
    sender: str,
    capability: str,
) -> dict[str, object]:
    """Handle ``message_type='message'`` — addressed plain message."""
    await _require_sender_or_lead(ctx, team_name, sender, capability)
    if not content:
        raise MessageContentEmptyToolError()
    if not summary:
        raise MessageSummaryEmptyToolError()
    if not recipient:
        raise MessageRecipientEmptyToolError()
    config = await teams.read_config(team_name)
    sender_member = _find_member(config, sender)
    if sender_member is None:
        raise NotTeamMemberError("Sender", sender, team_name)
    recipient_member = _find_member(config, recipient)
    if recipient_member is None:
        raise NotTeamMemberError("Recipient", recipient, team_name)
    target_color = (
        recipient_member.color if isinstance(recipient_member, TeammateMember) else None
    )
    sender_color = (
        sender_member.color if isinstance(sender_member, TeammateMember) else None
    )
    await messaging.send_plain_message(
        team_name,
        sender,
        recipient,
        content,
        summary=summary,
        color=sender_color,
    )
    routing: MessageRouting = {
        "sender": sender,
        "target": recipient,
        "targetColor": target_color,
        "summary": summary,
        "content": content,
    }
    return SendMessageResult(
        success=True,
        message=f"Message sent to {recipient}",
        routing=routing,
    ).model_dump(exclude_none=True)


_BROADCAST_RECIPIENT_LIMIT = 50


async def _handle_broadcast(
    team_name: str,
    ctx: Context,
    content: str,
    summary: str,
    sender: str,
    capability: str,
) -> dict[str, object]:
    """Handle ``message_type='broadcast'`` — lead-only fan-out to teammates.

    Recipient fan-out is capped at ``_BROADCAST_RECIPIENT_LIMIT``. Each
    delivery acquires a per-inbox file lock, so a broadcast to a large team
    could monopolise the messaging lane; the cap keeps a single call
    bounded. Teams that legitimately exceed the cap should send targeted
    messages instead.
    """
    await _require_lead(ctx, team_name, capability)
    if not summary:
        raise BroadcastSummaryEmptyToolError()
    if sender != "team-lead":
        raise BroadcastSenderError()
    config = await teams.read_config(team_name)
    teammates = [m for m in config.members if isinstance(m, TeammateMember)]
    if len(teammates) > _BROADCAST_RECIPIENT_LIMIT:
        raise BroadcastTooManyRecipientsError(
            len(teammates), _BROADCAST_RECIPIENT_LIMIT
        )
    for member in teammates:
        await messaging.send_plain_message(
            team_name,
            "team-lead",
            member.name,
            content,
            summary=summary,
            color=None,
        )
    return SendMessageResult(
        success=True,
        message=f"Broadcast sent to {len(teammates)} teammate(s)",
    ).model_dump(exclude_none=True)


async def _handle_shutdown_request(
    team_name: str,
    ctx: Context,
    recipient: str,
    content: str,
    capability: str,
) -> dict[str, object]:
    """Handle ``message_type='shutdown_request'`` — lead asks teammate to exit."""
    await _require_lead(ctx, team_name, capability)
    if not recipient:
        raise ShutdownRecipientEmptyToolError()
    if recipient == "team-lead":
        raise ShutdownSelfError()
    config = await teams.read_config(team_name)
    member_names = {member.name for member in config.members}
    if recipient not in member_names:
        raise NotTeamMemberError("Recipient", recipient, team_name)
    req_id = await messaging.send_shutdown_request(team_name, recipient, reason=content)
    return SendMessageResult(
        success=True,
        message=f"Shutdown request sent to {recipient}",
        request_id=req_id,
        target=recipient,
    ).model_dump(exclude_none=True)


async def _handle_shutdown_response(
    team_name: str,
    ctx: Context,
    request_id: str,
    approve: bool | None,
    content: str,
    sender: str,
    capability: str,
) -> dict[str, object]:
    """Handle ``message_type='shutdown_response'`` — teammate accepts or rejects.

    ``approve`` must be set to ``True`` or ``False`` explicitly; a missing
    value (``None``) is rejected so that a client omitting the flag cannot
    silently fall through to the rejection branch and leave the lead with a
    stale approval request. The sender must be a registered teammate — the
    approval path publishes backend/pane identifiers the lead uses to clean
    up the process, so an unknown sender would otherwise fabricate a
    ``ShutdownApproved`` payload with empty handles.
    """
    await _require_sender_or_lead(ctx, team_name, sender, capability)
    if approve is None:
        raise ShutdownResponseApprovalRequiredError()
    config = await teams.read_config(team_name)
    member = None
    for config_member in config.members:
        if isinstance(config_member, TeammateMember) and config_member.name == sender:
            member = config_member
            break
    if member is None:
        raise NotTeamMemberError("Sender", sender, team_name)
    if approve is True:
        payload = ShutdownApproved(
            request_id=request_id,
            from_=sender,
            timestamp=messaging.now_iso(),
            pane_id=member.tmux_pane_id,
            backend_type=member.backend_type,
            process_handle=member.process_handle or member.tmux_pane_id,
        )
        await messaging.send_structured_message(team_name, sender, "team-lead", payload)
        return SendMessageResult(
            success=True,
            message=f"Shutdown approved for request {request_id}",
        ).model_dump(exclude_none=True)
    await messaging.send_plain_message(
        team_name,
        sender,
        "team-lead",
        content or "Shutdown rejected",
        summary="shutdown_rejected",
    )
    return SendMessageResult(
        success=True,
        message=f"Shutdown rejected for request {request_id}",
    ).model_dump(exclude_none=True)


async def _handle_plan_approval_response(
    team_name: str,
    ctx: Context,
    recipient: str,
    content: str,
    approve: bool | None,
    sender: str,
    capability: str,
) -> dict[str, object]:
    """Handle ``message_type='plan_approval_response'`` — plan approval or rejection."""
    await _require_sender_or_lead(ctx, team_name, sender, capability)
    if not recipient:
        raise PlanRecipientEmptyToolError()
    config = await teams.read_config(team_name)
    member_names = {member.name for member in config.members}
    if recipient not in member_names:
        raise NotTeamMemberError("Recipient", recipient, team_name)
    if approve:
        await messaging.send_plain_message(
            team_name,
            sender,
            recipient,
            '{"type":"plan_approval","approved":true}',
            summary="plan_approved",
        )
    else:
        await messaging.send_plain_message(
            team_name,
            sender,
            recipient,
            content or "Plan rejected",
            summary="plan_rejected",
        )
    return SendMessageResult(
        success=True,
        message=f"Plan {'approved' if approve else 'rejected'} for {recipient}",
    ).model_dump(exclude_none=True)


async def send_message(
    team_name: TeamName,
    ctx: Context,
    message_type: Literal[
        "message",
        "broadcast",
        "shutdown_request",
        "shutdown_response",
        "plan_approval_response",
    ],
    recipient: str = "",
    content: MessageContent = "",
    summary: MessageSummary = "",
    request_id: RequestId = "",
    approve: bool | None = None,
    sender: SenderName = "team-lead",
    capability: Capability = "",
) -> dict[str, object]:
    """Send a team protocol message.

    Dispatches to a per-type handler. Each handler owns its own validation,
    lookups, and response shape — ``send_message`` itself is a thin router
    to keep the branch count low and make each protocol message type
    independently testable.

    Args:
        team_name: Team name.
        ctx: FastMCP request context.
        message_type: Message type.
        recipient: Target member when applicable.
        content: Message body or rejection reason.
        summary: Summary text for plain messages.
        request_id: Request identifier for response types.
        approve: Approval flag for response types.
        sender: Logical sender identity.
        capability: Optional capability override.

    Returns:
        dict: Send result payload.

    """
    if message_type == "message":
        return await _handle_plain_message(
            team_name, ctx, recipient, content, summary, sender, capability
        )
    if message_type == "broadcast":
        return await _handle_broadcast(
            team_name, ctx, content, summary, sender, capability
        )
    if message_type == "shutdown_request":
        return await _handle_shutdown_request(
            team_name, ctx, recipient, content, capability
        )
    if message_type == "shutdown_response":
        return await _handle_shutdown_response(
            team_name, ctx, request_id, approve, content, sender, capability
        )
    if message_type == "plan_approval_response":
        return await _handle_plan_approval_response(
            team_name, ctx, recipient, content, approve, sender, capability
        )
    raise UnknownMessageTypeError(message_type)


def register_team_spawn_tools(mcp: FastMCP) -> None:
    """Register team-tier spawn and messaging tools.

    ``spawn_teammate`` carries a generous 15-minute timeout because one-shot
    backends can take several minutes to return; other tools inherit the
    default FastMCP request timeout.

    Args:
        mcp (FastMCP): MCP server instance.

    """
    mcp.tool(
        name="spawn_teammate",
        tags={_TAG_TEAM},
        annotations=_ANN_CREATE,
        timeout=900.0,
    )(spawn_teammate_tool)
    mcp.tool(tags={_TAG_TEAM}, annotations=_ANN_MUTATE)(send_message)
