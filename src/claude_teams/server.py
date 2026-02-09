import asyncio
import time
import uuid
from pathlib import Path
from typing import Literal, TypedDict, cast

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan

from claude_teams import messaging, tasks, teams
from claude_teams.backends import (
    BackendRegistry,
    SpawnRequest,
    registry,
)
from claude_teams.models import (
    BackendInfo,
    InboxMessage,
    SendMessageResult,
    ShutdownApproved,
    SpawnResult,
    TeammateMember,
)
from claude_teams.spawner import assign_color
from claude_teams.teams import _VALID_NAME_RE


class _LifespanState(TypedDict):
    registry: BackendRegistry
    session_id: str
    active_team: str | None


@lifespan
async def app_lifespan(server):
    """Initialize and manage the MCP server lifespan.

    Args:
        server: The FastMCP server instance.

    Yields:
        _LifespanState: Dictionary containing registry, session_id, and active_team.
    """
    session_id = str(uuid.uuid4())
    yield {
        "registry": registry,
        "session_id": session_id,
        "active_team": None,
    }


mcp = FastMCP(
    name="claude-teams",
    instructions=(
        "MCP server for orchestrating Claude Code agent teams. "
        "Manages team creation, teammate spawning, messaging, and task tracking."
    ),
    lifespan=app_lifespan,
)


def _get_lifespan(ctx: Context) -> _LifespanState:
    """Extract and cast the lifespan state from the MCP context.

    Args:
        ctx (Context): FastMCP context containing lifespan state.

    Returns:
        _LifespanState: Typed lifespan state with registry, session_id, active_team.
    """
    return cast(_LifespanState, ctx.lifespan_context)


@mcp.tool
def team_create(
    team_name: str,
    ctx: Context,
    description: str = "",
) -> dict:
    """Create a new agent team. Sets up team config and task directories under ~/.claude/.
    One team per server session. Team names must be filesystem-safe
    (letters, numbers, hyphens, underscores).

    Args:
        team_name (str): Unique name for the new team.
        ctx (Context): FastMCP context containing lifespan state.
        description (str): Human-readable description of the team's purpose.

    Returns:
        dict: Result containing team name and success status.

    Raises:
        ToolError: If a team is already active in this session or team name is invalid.
    """
    ls = _get_lifespan(ctx)
    if ls.get("active_team"):
        raise ToolError(
            f"Session already has active team: {ls['active_team']}. One team per session."
        )
    result = teams.create_team(
        name=team_name, session_id=ls["session_id"], description=description
    )
    ls["active_team"] = team_name
    return result.model_dump()


@mcp.tool
def team_delete(team_name: str, ctx: Context) -> dict:
    """Delete a team and all its data. Fails if any teammates are still active.
    Removes both team config and task directories.

    Args:
        team_name (str): Name of the team to delete.
        ctx (Context): FastMCP context containing lifespan state.

    Returns:
        dict: Result containing success status.

    Raises:
        ToolError: If team not found or teammates are still active.
    """
    try:
        result = teams.delete_team(team_name)
    except (RuntimeError, FileNotFoundError) as e:
        raise ToolError(str(e))
    _get_lifespan(ctx)["active_team"] = None
    return result.model_dump()


@mcp.tool(name="spawn_teammate")
def spawn_teammate_tool(
    team_name: str,
    name: str,
    prompt: str,
    ctx: Context,
    model: str = "balanced",
    backend: str = "",
    subagent_type: str = "general-purpose",
    plan_mode_required: bool = False,
) -> dict:
    """Spawn a new teammate using any available backend. Backends: claude-code,
    codex, gemini, opencode, aider. Models: use generic tiers
    (fast/balanced/powerful) or backend-specific names. Leave backend empty to
    use the default (claude-code if available).

    Args:
        team_name (str): Name of the team to spawn the teammate in.
        name (str): Unique name for the new teammate.
        prompt (str): Initial prompt/instructions for the teammate.
        ctx (Context): FastMCP context containing lifespan state.
        model (str): Model tier or backend-specific model name.
        backend (str): Backend to use for spawning (empty for default).
        subagent_type (str): Type of agent (e.g., 'general-purpose').
        plan_mode_required (bool): Whether the agent requires plan mode.

    Returns:
        dict: SpawnResult with agent_id, name, and team_name.

    Raises:
        ToolError: If backend unavailable, model invalid, or name conflicts.
    """
    ls = _get_lifespan(ctx)
    reg = ls["registry"]

    # Resolve backend
    if backend:
        try:
            backend_obj = reg.get(backend)
        except KeyError as exc:
            raise ToolError(str(exc))
    else:
        try:
            default_name = reg.default_backend()
            backend_obj = reg.get(default_name)
        except (RuntimeError, KeyError) as exc:
            raise ToolError(str(exc))

    # Resolve model
    try:
        resolved_model = backend_obj.resolve_model(model)
    except ValueError as exc:
        raise ToolError(str(exc))

    # Validate name
    if not _VALID_NAME_RE.match(name):
        raise ToolError(
            f"Invalid agent name: {name!r}. Use only letters, numbers, hyphens, underscores."
        )
    if len(name) > 64:
        raise ToolError(f"Agent name too long ({len(name)} chars, max 64)")
    if name == "team-lead":
        raise ToolError("Agent name 'team-lead' is reserved")

    # Assign color
    color = assign_color(team_name)

    # Create member
    now_ms = int(time.time() * 1000)
    cwd = str(Path.cwd())
    member = TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type=subagent_type,
        model=resolved_model,
        prompt=prompt,
        color=color,
        plan_mode_required=plan_mode_required,
        joined_at=now_ms,
        tmux_pane_id="",
        cwd=cwd,
        backend_type=backend_obj.name,
        is_active=False,
        process_handle="",
    )

    # Register member in team config
    try:
        teams.add_member(team_name, member)
    except ValueError as exc:
        raise ToolError(str(exc))

    # Send initial prompt via inbox
    messaging.ensure_inbox(team_name, name)
    initial_msg = InboxMessage(
        from_="team-lead",
        text=prompt,
        timestamp=messaging.now_iso(),
        read=False,
    )
    messaging.append_message(team_name, name, initial_msg)

    # Spawn via backend
    request = SpawnRequest(
        agent_id=member.agent_id,
        name=name,
        team_name=team_name,
        prompt=prompt,
        model=resolved_model,
        agent_type=subagent_type,
        color=color,
        cwd=cwd,
        lead_session_id=ls["session_id"],
        plan_mode_required=plan_mode_required,
    )
    try:
        spawn_result = backend_obj.spawn(request)
    except Exception as exc:
        teams.remove_member(team_name, name)
        raise ToolError(f"Backend spawn failed: {exc}")

    # Update config with process handle
    config = teams.read_config(team_name)
    for member in config.members:
        if isinstance(member, TeammateMember) and member.name == name:
            member.process_handle = spawn_result.process_handle
            member.tmux_pane_id = spawn_result.process_handle
            break
    teams.write_config(team_name, config)

    return SpawnResult(
        agent_id=member.agent_id,
        name=member.name,
        team_name=team_name,
    ).model_dump()


@mcp.tool
def send_message(
    team_name: str,
    type: Literal[
        "message",
        "broadcast",
        "shutdown_request",
        "shutdown_response",
        "plan_approval_response",
    ],
    recipient: str = "",
    content: str = "",
    summary: str = "",
    request_id: str = "",
    approve: bool | None = None,
    sender: str = "team-lead",
) -> dict:
    """Send a message to a teammate or respond to a protocol request.
    Type 'message' sends a direct message (requires recipient, summary).
    Type 'broadcast' sends to all teammates (requires summary).
    Type 'shutdown_request' asks a teammate to shut down (requires recipient; content used as reason).
    Type 'shutdown_response' responds to a shutdown request (requires sender, request_id, approve).
    Type 'plan_approval_response' responds to a plan approval request (requires recipient, request_id, approve).

    Args:
        team_name (str): Name of the team.
        type (Literal): Message type (message, broadcast, shutdown_request, shutdown_response, plan_approval_response).
        recipient (str): Target agent name for direct messages.
        content (str): Message content/body.
        summary (str): Brief message summary.
        request_id (str): ID of the request being responded to.
        approve (bool | None): Whether to approve a request.
        sender (str): Name of the sending agent.

    Returns:
        dict: SendMessageResult with success status and routing details.

    Raises:
        ToolError: If required fields are missing or recipient not found.
    """

    if type == "message":
        if not content:
            raise ToolError("Message content must not be empty")
        if not summary:
            raise ToolError("Message summary must not be empty")
        if not recipient:
            raise ToolError("Message recipient must not be empty")
        config = teams.read_config(team_name)
        member_names = {member.name for member in config.members}
        if recipient not in member_names:
            raise ToolError(
                f"Recipient {recipient!r} is not a member of team {team_name!r}"
            )
        target_color = None
        for member in config.members:
            if member.name == recipient and isinstance(member, TeammateMember):
                target_color = member.color
                break
        messaging.send_plain_message(
            team_name,
            "team-lead",
            recipient,
            content,
            summary=summary,
            color=target_color,
        )
        return SendMessageResult(
            success=True,
            message=f"Message sent to {recipient}",
            routing={
                "sender": "team-lead",
                "target": recipient,
                "targetColor": target_color,
                "summary": summary,
                "content": content,
            },
        ).model_dump(exclude_none=True)

    elif type == "broadcast":
        if not summary:
            raise ToolError("Broadcast summary must not be empty")
        config = teams.read_config(team_name)
        count = 0
        for member in config.members:
            if isinstance(member, TeammateMember):
                messaging.send_plain_message(
                    team_name,
                    "team-lead",
                    member.name,
                    content,
                    summary=summary,
                    color=None,
                )
                count += 1
        return SendMessageResult(
            success=True,
            message=f"Broadcast sent to {count} teammate(s)",
        ).model_dump(exclude_none=True)

    elif type == "shutdown_request":
        if not recipient:
            raise ToolError("Shutdown request recipient must not be empty")
        if recipient == "team-lead":
            raise ToolError("Cannot send shutdown request to team-lead")
        config = teams.read_config(team_name)
        member_names = {member.name for member in config.members}
        if recipient not in member_names:
            raise ToolError(
                f"Recipient {recipient!r} is not a member of team {team_name!r}"
            )
        req_id = messaging.send_shutdown_request(team_name, recipient, reason=content)
        return SendMessageResult(
            success=True,
            message=f"Shutdown request sent to {recipient}",
            request_id=req_id,
            target=recipient,
        ).model_dump(exclude_none=True)

    elif type == "shutdown_response":
        if approve:
            config = teams.read_config(team_name)
            member = None
            for config_member in config.members:
                if (
                    isinstance(config_member, TeammateMember)
                    and config_member.name == sender
                ):
                    member = config_member
                    break
            pane_id = member.tmux_pane_id if member else ""
            process_handle = (
                (member.process_handle or member.tmux_pane_id) if member else ""
            )
            backend_type = member.backend_type if member else "tmux"
            payload = ShutdownApproved(
                request_id=request_id,
                from_=sender,
                timestamp=messaging.now_iso(),
                pane_id=pane_id,
                backend_type=backend_type,
                process_handle=process_handle,
            )
            messaging.send_structured_message(team_name, sender, "team-lead", payload)
            return SendMessageResult(
                success=True,
                message=f"Shutdown approved for request {request_id}",
            ).model_dump(exclude_none=True)
        else:
            messaging.send_plain_message(
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

    elif type == "plan_approval_response":
        if not recipient:
            raise ToolError("Plan approval recipient must not be empty")
        config = teams.read_config(team_name)
        member_names = {member.name for member in config.members}
        if recipient not in member_names:
            raise ToolError(
                f"Recipient {recipient!r} is not a member of team {team_name!r}"
            )
        if approve:
            messaging.send_plain_message(
                team_name,
                sender,
                recipient,
                '{"type":"plan_approval","approved":true}',
                summary="plan_approved",
            )
        else:
            messaging.send_plain_message(
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

    raise ToolError(f"Unknown message type: {type}")


@mcp.tool
def task_create(
    team_name: str,
    subject: str,
    description: str,
    active_form: str = "",
    metadata: dict | None = None,
) -> dict:
    """Create a new task for the team. Tasks are auto-assigned incrementing IDs.
    Optional metadata dict is stored alongside the task.

    Args:
        team_name (str): Name of the team.
        subject (str): Brief task title.
        description (str): Detailed task description.
        active_form (str): Present continuous form for in-progress display.
        metadata (dict | None): Optional metadata dictionary.

    Returns:
        dict: Created task with id, subject, status, and other fields.

    Raises:
        ToolError: If team not found or task creation fails.
    """
    try:
        task = tasks.create_task(team_name, subject, description, active_form, metadata)
    except ValueError as e:
        raise ToolError(str(e))
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool
def task_update(
    team_name: str,
    task_id: str,
    status: Literal["pending", "in_progress", "completed", "deleted"] | None = None,
    owner: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Update a task's fields. Setting owner auto-notifies the assignee via
    inbox. Setting status to 'deleted' removes the task file from disk.
    Metadata keys are merged into existing metadata (set a key to null to delete it).

    Args:
        team_name (str): Name of the team.
        task_id (str): ID of the task to update.
        status (Literal | None): New status (pending, in_progress, completed, deleted).
        owner (str | None): Name of the agent to assign the task to.
        subject (str | None): New task subject.
        description (str | None): New task description.
        active_form (str | None): New active form.
        add_blocks (list[str] | None): Task IDs this task blocks.
        add_blocked_by (list[str] | None): Task IDs that block this task.
        metadata (dict | None): Metadata to merge (set key to null to delete).

    Returns:
        dict: Updated task with all fields.

    Raises:
        ToolError: If task not found or update fails.
    """
    try:
        task = tasks.update_task(
            team_name,
            task_id,
            status=status,
            owner=owner,
            subject=subject,
            description=description,
            active_form=active_form,
            add_blocks=add_blocks,
            add_blocked_by=add_blocked_by,
            metadata=metadata,
        )
    except FileNotFoundError:
        raise ToolError(f"Task {task_id!r} not found in team {team_name!r}")
    except ValueError as e:
        raise ToolError(str(e))
    if owner is not None and task.owner is not None and task.status != "deleted":
        messaging.send_task_assignment(team_name, task, assigned_by="team-lead")
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool
def task_list(team_name: str) -> list[dict]:
    """List all tasks for a team with their current status and assignments.

    Args:
        team_name (str): Name of the team.

    Returns:
        list[dict]: List of tasks with id, subject, status, owner, etc.

    Raises:
        ToolError: If team not found or task listing fails.
    """
    try:
        result = tasks.list_tasks(team_name)
    except ValueError as e:
        raise ToolError(str(e))
    return [task.model_dump(by_alias=True, exclude_none=True) for task in result]


@mcp.tool
def task_get(team_name: str, task_id: str) -> dict:
    """Get full details of a specific task by ID.

    Args:
        team_name (str): Name of the team.
        task_id (str): ID of the task to retrieve.

    Returns:
        dict: Complete task details including all fields.

    Raises:
        ToolError: If task not found.
    """
    try:
        task = tasks.get_task(team_name, task_id)
    except FileNotFoundError:
        raise ToolError(f"Task {task_id!r} not found in team {team_name!r}")
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool
def read_inbox(
    team_name: str,
    agent_name: str,
    unread_only: bool = False,
    mark_as_read: bool = True,
) -> list[dict]:
    """Read messages from an agent's inbox. Returns all messages by default.
    Set unread_only=True to get only unprocessed messages.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent whose inbox to read.
        unread_only (bool): Whether to return only unread messages.
        mark_as_read (bool): Whether to mark returned messages as read.

    Returns:
        list[dict]: List of inbox messages with from, text, timestamp, read status.
    """
    msgs = messaging.read_inbox(
        team_name, agent_name, unread_only=unread_only, mark_as_read=mark_as_read
    )
    return [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs]


@mcp.tool
def read_config(team_name: str) -> dict:
    """Read the current team configuration including all members.

    Args:
        team_name (str): Name of the team.

    Returns:
        dict: Team config with name, description, lead_agent_id, members list.

    Raises:
        ToolError: If team not found.
    """
    try:
        config = teams.read_config(team_name)
    except FileNotFoundError:
        raise ToolError(f"Team {team_name!r} not found")
    return config.model_dump(by_alias=True)


@mcp.tool
def force_kill_teammate(team_name: str, agent_name: str) -> dict:
    """Forcibly kill a teammate. Uses the teammate's registered backend to
    perform the kill. Removes member from config and resets their tasks.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the teammate to kill.

    Returns:
        dict: Success status and message.

    Raises:
        ToolError: If teammate not found in team.
    """
    config = teams.read_config(team_name)
    member = None
    for config_member in config.members:
        if (
            isinstance(config_member, TeammateMember)
            and config_member.name == agent_name
        ):
            member = config_member
            break
    if member is None:
        raise ToolError(f"Teammate {agent_name!r} not found in team {team_name!r}")

    process_handle = member.process_handle or member.tmux_pane_id
    backend_type = member.backend_type

    # Fallback: legacy "tmux" backend_type maps to "claude-code"
    if backend_type == "tmux":
        backend_type = "claude-code"

    if process_handle:
        try:
            backend_obj = registry.get(backend_type)
            backend_obj.kill(process_handle)
        except KeyError:
            # Backend not available; skip kill (process may already be dead)
            pass

    teams.remove_member(team_name, agent_name)
    tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} has been stopped."}


@mcp.tool
async def poll_inbox(
    team_name: str,
    agent_name: str,
    timeout_ms: int = 30000,
) -> list[dict]:
    """Poll an agent's inbox for new unread messages, waiting up to timeout_ms.
    Returns unread messages and marks them as read. Convenience tool for MCP
    clients that cannot watch the filesystem.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent whose inbox to poll.
        timeout_ms (int): Maximum time to wait in milliseconds.

    Returns:
        list[dict]: List of unread messages (empty if timeout reached).
    """
    msgs = messaging.read_inbox(
        team_name, agent_name, unread_only=True, mark_as_read=True
    )
    if msgs:
        return [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs]
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        await asyncio.sleep(0.5)
        msgs = messaging.read_inbox(
            team_name, agent_name, unread_only=True, mark_as_read=True
        )
        if msgs:
            return [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs]
    return []


@mcp.tool
def process_shutdown_approved(team_name: str, agent_name: str) -> dict:
    """Process a teammate's shutdown by removing them from config and resetting
    their tasks. Call this after confirming shutdown_approved in the lead inbox.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the teammate to remove.

    Returns:
        dict: Success status and message.

    Raises:
        ToolError: If agent_name is 'team-lead'.
    """
    if agent_name == "team-lead":
        raise ToolError("Cannot process shutdown for team-lead")
    teams.remove_member(team_name, agent_name)
    tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} removed from team."}


@mcp.tool
def list_backends(ctx: Context) -> list[dict]:
    """List all available spawner backends with their supported models.
    Returns backend name, binary, availability, default model, and model options.

    Args:
        ctx (Context): FastMCP context containing lifespan state.

    Returns:
        list[dict]: List of backend info dicts with name, binary, models.
    """
    ls = _get_lifespan(ctx)
    reg = ls["registry"]
    result = []
    for name, backend_obj in reg:
        info = BackendInfo(
            name=name,
            binary=backend_obj.binary_name,
            available=True,
            default_model=backend_obj.default_model(),
            supported_models=backend_obj.supported_models(),
        )
        result.append(info.model_dump(by_alias=True))
    return result


@mcp.tool
def health_check(team_name: str, agent_name: str, ctx: Context) -> dict:
    """Check if a teammate's process is still running.
    Uses the teammate's registered backend for the health check.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the teammate to check.
        ctx (Context): FastMCP context containing lifespan state.

    Returns:
        dict: Health status with agent_name, alive, backend, detail.

    Raises:
        ToolError: If team or teammate not found, or backend unavailable.
    """
    try:
        config = teams.read_config(team_name)
    except FileNotFoundError:
        raise ToolError(f"Team {team_name!r} not found")

    member = None
    for config_member in config.members:
        if (
            isinstance(config_member, TeammateMember)
            and config_member.name == agent_name
        ):
            member = config_member
            break
    if member is None:
        raise ToolError(f"Teammate {agent_name!r} not found in team {team_name!r}")

    process_handle = member.process_handle or member.tmux_pane_id
    backend_type = member.backend_type

    # Fallback: legacy "tmux" backend_type maps to "claude-code"
    if backend_type == "tmux":
        backend_type = "claude-code"

    try:
        backend_obj = registry.get(backend_type)
    except KeyError as exc:
        raise ToolError(str(exc))

    status = backend_obj.health_check(process_handle)
    return {
        "agent_name": agent_name,
        "alive": status.alive,
        "backend": backend_type,
        "detail": status.detail,
    }


def main():
    """Entry point for the claude-teams MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
