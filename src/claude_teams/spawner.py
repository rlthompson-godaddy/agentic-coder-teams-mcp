import shlex
import shutil
import time
from pathlib import Path

from claude_code_tools.tmux_cli_controller import TmuxCLIController

from claude_teams import messaging, teams
from claude_teams.backends.base import SpawnRequest
from claude_teams.backends.claude_code import ClaudeCodeBackend
from claude_teams.models import COLOR_PALETTE, InboxMessage, TeammateMember
from claude_teams.teams import _VALID_NAME_RE


def discover_claude_binary() -> str:
    """Find the claude binary on PATH.

    Returns:
        str: Full path to the claude binary.

    Raises:
        FileNotFoundError: If claude is not found on PATH.
    """
    path = shutil.which("claude")
    if path is None:
        raise FileNotFoundError(
            "Could not find 'claude' binary on PATH. "
            "Install Claude Code or ensure it is in your PATH."
        )
    return path


def assign_color(team_name: str, base_dir: Path | None = None) -> str:
    """Assign the next color from the palette to a new teammate.

    Args:
        team_name (str): Name of the team.
        base_dir (Path | None): Optional override for the base config directory.

    Returns:
        str: Color string from COLOR_PALETTE.
    """
    config = teams.read_config(team_name, base_dir)
    count = sum(1 for member in config.members if isinstance(member, TeammateMember))
    return COLOR_PALETTE[count % len(COLOR_PALETTE)]


def build_spawn_command(
    member: TeammateMember,
    claude_binary: str,
    lead_session_id: str,
) -> str:
    """Build the shell command string for spawning a Claude Code teammate.

    Delegates to the ClaudeCodeBackend for command construction while
    preserving the exact output format expected by callers.

    Args:
        member (TeammateMember): The teammate member to spawn.
        claude_binary (str): Path to the claude binary.
        lead_session_id (str): Session ID of the team lead.

    Returns:
        str: Shell command string suitable for tmux pane execution.
    """
    team_name = member.agent_id.split("@", 1)[1]

    request = SpawnRequest(
        agent_id=member.agent_id,
        name=member.name,
        team_name=team_name,
        prompt=member.prompt,
        model=member.model,
        agent_type=member.agent_type,
        color=member.color,
        cwd=member.cwd,
        lead_session_id=lead_session_id,
        plan_mode_required=member.plan_mode_required,
    )

    backend = ClaudeCodeBackend()
    env_vars = backend.build_env(request)

    model = backend.resolve_model(request.model)
    cmd_parts = [
        claude_binary,
        "--agent-id",
        request.agent_id,
        "--agent-name",
        request.name,
        "--team-name",
        request.team_name,
        "--agent-color",
        request.color,
        "--parent-session-id",
        request.lead_session_id,
        "--agent-type",
        request.agent_type,
        "--model",
        model,
    ]
    if request.plan_mode_required:
        cmd_parts.append("--plan-mode-required")

    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in env_vars.items()
    )
    cmd_str = " ".join(shlex.quote(part) for part in cmd_parts)
    if env_prefix:
        return f"cd {shlex.quote(member.cwd)} && {env_prefix} {cmd_str}"
    return f"cd {shlex.quote(member.cwd)} && {cmd_str}"


def spawn_teammate(
    team_name: str,
    name: str,
    prompt: str,
    claude_binary: str,
    lead_session_id: str,
    *,
    model: str = "sonnet",
    subagent_type: str = "general-purpose",
    cwd: str | None = None,
    plan_mode_required: bool = False,
    base_dir: Path | None = None,
) -> TeammateMember:
    """Spawn a new Claude Code teammate in a tmux pane.

    Registers the member, sends the initial prompt to the inbox, spawns
    via ``TmuxCLIController.launch_cli``, and updates the config with
    the pane identifier.

    Args:
        team_name (str): Name of the team.
        name (str): Name for the new teammate.
        prompt (str): Initial prompt/instructions for the teammate.
        claude_binary (str): Path to the claude binary.
        lead_session_id (str): Session ID of the team lead.
        model (str): Model to use (default: sonnet).
        subagent_type (str): Agent type string (default: general-purpose).
        cwd (str | None): Working directory (default: current directory).
        plan_mode_required (bool): Whether plan mode is required.
        base_dir (Path | None): Optional override for the base config directory.

    Returns:
        TeammateMember: The created member with process handle set.

    Raises:
        ValueError: If the name is invalid.
        RuntimeError: If tmux pane creation fails.
    """
    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            f"Invalid agent name: {name!r}. Use only letters, numbers, hyphens, underscores."
        )
    if len(name) > 64:
        raise ValueError(f"Agent name too long ({len(name)} chars, max 64)")
    if name == "team-lead":
        raise ValueError("Agent name 'team-lead' is reserved")

    color = assign_color(team_name, base_dir)
    now_ms = int(time.time() * 1000)

    member = TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type=subagent_type,
        model=model,
        prompt=prompt,
        color=color,
        plan_mode_required=plan_mode_required,
        joined_at=now_ms,
        tmux_pane_id="",
        cwd=cwd or str(Path.cwd()),
        backend_type="claude-code",
        is_active=False,
    )

    teams.add_member(team_name, member, base_dir)

    messaging.ensure_inbox(team_name, name, base_dir)
    initial_msg = InboxMessage(
        from_="team-lead",
        text=prompt,
        timestamp=messaging.now_iso(),
        read=False,
    )
    messaging.append_message(team_name, name, initial_msg, base_dir)

    cmd = build_spawn_command(member, claude_binary, lead_session_id)
    ctrl = TmuxCLIController()
    pane_id = ctrl.launch_cli(cmd)
    if pane_id is None:
        raise RuntimeError(
            f"Failed to create tmux pane for agent {name!r}. "
            "Ensure tmux is running and tmux-cli is available."
        )

    config = teams.read_config(team_name, base_dir)
    for config_member in config.members:
        if isinstance(config_member, TeammateMember) and config_member.name == name:
            config_member.tmux_pane_id = pane_id
            config_member.process_handle = pane_id
            break
    teams.write_config(team_name, config, base_dir)

    member.tmux_pane_id = pane_id
    member.process_handle = pane_id
    return member


def kill_tmux_pane(pane_id: str) -> None:
    """Kill a tmux pane by ID via ``TmuxCLIController``.

    Args:
        pane_id (str): The tmux pane identifier.
    """
    ctrl = TmuxCLIController()
    ctrl.kill_pane(pane_id=pane_id)
