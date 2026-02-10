import asyncio
import logging
import os
import re
import signal
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


_TAG_BOOTSTRAP = "bootstrap"
_TAG_TEAM = "team"
_TAG_TEAMMATE = "teammate"
_ONE_SHOT_RESULT_MAX_CHARS = 12000
_ONE_SHOT_TIMEOUT_S = 900
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")

logger = logging.getLogger(__name__)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and carriage returns from text.

    Used to clean tmux pane captures before relaying to team-lead.

    Args:
        text (str): Raw text potentially containing ANSI escape codes.

    Returns:
        str: Cleaned text with escape sequences removed.
    """
    return _ANSI_ESCAPE_RE.sub("", text)


class _LifespanState(TypedDict):
    registry: BackendRegistry
    session_id: str
    active_team: str | None
    has_teammates: bool


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
        "has_teammates": False,
    }


mcp = FastMCP(
    name="claude-teams",
    instructions=(
        "MCP server for orchestrating Claude Code agent teams. "
        "Manages team creation, teammate spawning, messaging, and task tracking."
    ),
    lifespan=app_lifespan,
)
mcp.enable(tags={_TAG_BOOTSTRAP}, only=True, components={"tool"})


def _get_lifespan(ctx: Context) -> _LifespanState:
    """Extract and cast the lifespan state from the MCP context.

    Args:
        ctx (Context): FastMCP context containing lifespan state.

    Returns:
        _LifespanState: Typed lifespan state with registry, session_id, active_team.
    """
    return cast(_LifespanState, ctx.lifespan_context)


@mcp.tool(tags={_TAG_BOOTSTRAP})
async def team_create(
    team_name: str,
    ctx: Context,
    description: str = "",
) -> dict:
    """Create a new agent team. Sets up team config and task directories under ~/.claude/.
    One team per server session. Team names must be filesystem-safe
    (letters, numbers, hyphens, underscores).
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
    await ctx.enable_components(tags={_TAG_TEAM}, components={"tool"})
    return result.model_dump()


@mcp.tool(tags={_TAG_BOOTSTRAP})
async def team_delete(team_name: str, ctx: Context) -> dict:
    """Delete a team and all its data. Fails if any teammates are still active.
    Removes both team config and task directories.
    """
    try:
        result = teams.delete_team(team_name)
    except (RuntimeError, FileNotFoundError) as e:
        raise ToolError(str(e))
    ls = _get_lifespan(ctx)
    ls["active_team"] = None
    ls["has_teammates"] = False
    await ctx.disable_components(tags={_TAG_TEAM, _TAG_TEAMMATE}, components={"tool"})
    return result.model_dump()


def _log_relay_task_exception(task: asyncio.Task) -> None:
    """Log unhandled exceptions from one-shot relay background tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("One-shot relay task failed: %s", exc, exc_info=exc)


@mcp.tool(name="spawn_teammate", tags={_TAG_TEAM})
async def spawn_teammate_tool(
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
    extra: dict[str, str] | None = None
    one_shot_result_path: Path | None = None

    # Codex supports file-based output capture via --output-last-message
    if backend_obj.name == "codex":
        one_shot_result_path = _create_one_shot_result_path(team_name, name)
        extra = {"output_last_message_path": str(one_shot_result_path)}

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
        extra=extra,
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

    if not ls["has_teammates"]:
        ls["has_teammates"] = True
        await ctx.enable_components(tags={_TAG_TEAMMATE}, components={"tool"})

    # Start result relay for all non-interactive (one-shot) backends.
    # Interactive backends (e.g. claude-code) handle their own messaging.
    if not backend_obj.is_interactive:
        # Keep the tmux pane alive after exit so we can capture its output.
        try:
            backend_obj.retain_pane_after_exit(spawn_result.process_handle)
        except Exception:
            logger.debug("Failed to set remain-on-exit for %s", name, exc_info=True)
        relay_task = asyncio.create_task(
            _relay_one_shot_result(
                team_name=team_name,
                agent_name=name,
                backend_type=backend_obj.name,
                process_handle=spawn_result.process_handle,
                result_file=one_shot_result_path,
                color=color,
            )
        )
        relay_task.add_done_callback(_log_relay_task_exception)

    return SpawnResult(
        agent_id=member.agent_id,
        name=member.name,
        team_name=team_name,
    ).model_dump()


@mcp.tool(tags={_TAG_TEAM})
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


@mcp.tool(tags={_TAG_TEAM})
def task_create(
    team_name: str,
    subject: str,
    description: str,
    active_form: str = "",
    metadata: dict | None = None,
) -> dict:
    """Create a new task for the team. Tasks are auto-assigned incrementing IDs.
    Optional metadata dict is stored alongside the task.
    """
    try:
        task = tasks.create_task(team_name, subject, description, active_form, metadata)
    except ValueError as e:
        raise ToolError(str(e))
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool(tags={_TAG_TEAM})
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


@mcp.tool(tags={_TAG_TEAM})
def task_list(team_name: str) -> list[dict]:
    """List all tasks for a team with their current status and assignments."""
    try:
        result = tasks.list_tasks(team_name)
    except ValueError as e:
        raise ToolError(str(e))
    return [task.model_dump(by_alias=True, exclude_none=True) for task in result]


@mcp.tool(tags={_TAG_TEAM})
def task_get(team_name: str, task_id: str) -> dict:
    """Get full details of a specific task by ID."""
    try:
        task = tasks.get_task(team_name, task_id)
    except FileNotFoundError:
        raise ToolError(f"Task {task_id!r} not found in team {team_name!r}")
    return task.model_dump(by_alias=True, exclude_none=True)


@mcp.tool(tags={_TAG_TEAM})
def read_inbox(
    team_name: str,
    agent_name: str,
    unread_only: bool = False,
    mark_as_read: bool = True,
) -> list[dict]:
    """Read messages from an agent's inbox. Returns all messages by default.
    Set unread_only=True to get only unprocessed messages.
    """
    msgs = messaging.read_inbox(
        team_name, agent_name, unread_only=unread_only, mark_as_read=mark_as_read
    )
    return [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs]


@mcp.tool(tags={_TAG_BOOTSTRAP})
def read_config(team_name: str) -> dict:
    """Read the current team configuration including all members."""
    try:
        config = teams.read_config(team_name)
    except FileNotFoundError:
        raise ToolError(f"Team {team_name!r} not found")
    return config.model_dump(by_alias=True)


def _resolve_teammate(
    team_name: str, agent_name: str
) -> tuple[TeammateMember, str | None, str]:
    """Look up a teammate and resolve process handle and backend type.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the teammate.

    Returns:
        tuple[TeammateMember, str | None, str]: The member, process handle,
            and normalised backend type.

    Raises:
        ToolError: If team or teammate is not found.
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

    return member, process_handle, backend_type


@mcp.tool(tags={_TAG_TEAMMATE})
def force_kill_teammate(team_name: str, agent_name: str) -> dict:
    """Forcibly kill a teammate. Uses the teammate's registered backend to
    perform the kill. Removes member from config and resets their tasks.
    """
    _member, process_handle, backend_type = _resolve_teammate(team_name, agent_name)

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


@mcp.tool(tags={_TAG_TEAMMATE})
async def poll_inbox(
    team_name: str,
    agent_name: str,
    timeout_ms: int = 30000,
) -> list[dict]:
    """Poll an agent's inbox for new unread messages, waiting up to timeout_ms.
    Returns unread messages and marks them as read. Convenience tool for MCP
    clients that cannot watch the filesystem.
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


@mcp.tool(tags={_TAG_TEAMMATE})
def process_shutdown_approved(team_name: str, agent_name: str) -> dict:
    """Process a teammate's shutdown by removing them from config and resetting
    their tasks. Call this after confirming shutdown_approved in the lead inbox.
    """
    if agent_name == "team-lead":
        raise ToolError("Cannot process shutdown for team-lead")
    teams.remove_member(team_name, agent_name)
    tasks.reset_owner_tasks(team_name, agent_name)
    return {"success": True, "message": f"{agent_name} removed from team."}


@mcp.tool(tags={_TAG_BOOTSTRAP})
def list_backends(ctx: Context) -> list[dict]:
    """List all available spawner backends with their supported models.
    Returns backend name, binary, availability, default model, and model options.
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


@mcp.tool(tags={_TAG_TEAMMATE})
def health_check(team_name: str, agent_name: str, ctx: Context) -> dict:
    """Check if a teammate's process is still running.
    Uses the teammate's registered backend for the health check.
    """
    _member, process_handle, backend_type = _resolve_teammate(team_name, agent_name)

    if not process_handle:
        raise ToolError(f"No process handle for teammate {agent_name!r}")

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


def _create_one_shot_result_path(team_name: str, agent_name: str) -> Path:
    runs_dir = messaging.TEAMS_DIR / team_name / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    return runs_dir / f"{agent_name}-{timestamp}.last-message.txt"


async def _relay_one_shot_result(
    team_name: str,
    agent_name: str,
    backend_type: str,
    process_handle: str,
    result_file: Path | None,
    color: str,
) -> None:
    """Wait for a one-shot backend to finish and relay its output to team-lead.

    Collects the result using two strategies in priority order:

    1. **File-based** (preferred) — backends like Codex write their final
       message to ``result_file`` via ``--output-last-message``.  If the file
       contains text it is used directly.
    2. **Tmux pane capture** (universal fallback) — after the process exits,
       the tmux pane buffer is captured and ANSI-stripped.  Works for every
       backend that runs in a tmux pane.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the teammate agent.
        backend_type (str): Backend identifier (e.g. ``"codex"``, ``"gemini"``).
        process_handle (str): Tmux pane identifier for health checks / capture.
        result_file (Path | None): Optional file path for file-based output.
            ``None`` for backends that do not support file-based output.
        color (str): Agent display color for inbox messages.
    """
    deadline = time.time() + _ONE_SHOT_TIMEOUT_S
    backend_obj = None
    text = ""

    try:
        backend_obj = registry.get(backend_type)
    except KeyError:
        logger.warning(
            "One-shot backend not available for result relay: %s", backend_type
        )

    # Phase 1: Poll until file appears, process dies, or timeout.
    while time.time() < deadline:
        if result_file is not None:
            try:
                if result_file.exists():
                    text = result_file.read_text().strip()
            except Exception:
                logger.exception("Failed reading one-shot result file: %s", result_file)
            if text:
                break

        if backend_obj is not None:
            status = backend_obj.health_check(process_handle)
            if not status.alive:
                break

        await asyncio.sleep(0.5)

    # Phase 2: Collect result — try file first, then pane capture.
    if not text and result_file is not None:
        try:
            if result_file.exists():
                text = result_file.read_text().strip()
        except Exception:
            logger.exception("Failed reading one-shot result file: %s", result_file)

    if not text and backend_obj is not None:
        try:
            captured = backend_obj.capture(process_handle)
            text = _strip_ansi(captured).strip()
        except Exception:
            logger.debug(
                "Failed to capture pane output for %s: %s",
                agent_name,
                process_handle,
                exc_info=True,
            )

    # Phase 3: Relay result or report failure.
    if not text and time.time() >= deadline:
        messaging.send_plain_message(
            team_name,
            agent_name,
            "team-lead",
            f"{agent_name} timed out before producing output.",
            summary="teammate_timeout",
            color=color,
        )
        return

    if not text:
        text = f"{agent_name} ({backend_type}) finished, but no output was captured."

    if len(text) > _ONE_SHOT_RESULT_MAX_CHARS:
        text = text[:_ONE_SHOT_RESULT_MAX_CHARS] + "\n\n[truncated]"

    messaging.send_plain_message(
        team_name,
        agent_name,
        "team-lead",
        text,
        summary="teammate_result",
        color=color,
    )

    if result_file is not None:
        try:
            result_file.unlink(missing_ok=True)
        except OSError:
            pass

    # Clean up the retained pane (remain-on-exit keeps it alive after exit).
    if backend_obj is not None:
        try:
            backend_obj.kill(process_handle)
        except Exception:
            pass


def main():
    """Entry point for the claude-teams MCP server."""
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    mcp.run()


if __name__ == "__main__":
    main()
