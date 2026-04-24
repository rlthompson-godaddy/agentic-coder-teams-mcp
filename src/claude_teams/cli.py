"""Typer CLI for claude-teams.

Provides human-friendly commands that operate on the same file-based state
as the MCP server.  Both the CLI and MCP server can run concurrently thanks
to ``fcntl.flock()`` guards in the core modules.
"""

import asyncio
import json
import os
import signal
import uuid
from typing import Annotated, Literal

import typer
from fastmcp.exceptions import ToolError
from rich.console import Console
from rich.table import Table

from claude_teams import (
    capabilities,
    messaging,
    tasks,
    teams,
)
from claude_teams import (
    presets as presets_mod,
)
from claude_teams import (
    templates as templates_mod,
)
from claude_teams.async_utils import run_blocking
from claude_teams.backends.registry import registry
from claude_teams.errors import (
    BackendNotRegisteredError,
    InvalidNameError,
    NameTooLongError,
    TeamNotFoundValueError,
)
from claude_teams.models import TeammateMember
from claude_teams.orchestration import SpawnDependencies, expand_preset_core
from claude_teams.server import mcp
from claude_teams.server_runtime import (
    _resolve_capability,
    _resolve_description,
    _resolve_permission_mode,
    _resolve_spawn_cwd,
    apply_spawn_env_defaults,
)
from claude_teams.server_team_relay import (
    build_agent_auth_notice,
    create_one_shot_result_path,
    log_relay_task_exception,
    log_retain_pane_failure,
    relay_one_shot_result,
)

_INBOX_SUMMARY_TRUNC_LEN = 80

app = typer.Typer(
    name="claude-teams",
    help="CLI for orchestrating Claude Code agent teams.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

preset_app = typer.Typer(
    name="preset",
    help="Launch and inspect team presets.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(preset_app, name="preset")

console = Console()
err_console = Console(stderr=True)

JsonFlag = Annotated[
    bool,
    typer.Option("--json", "-j", help="Output as JSON instead of a table."),
]
CapabilityFlag = Annotated[
    str,
    typer.Option(
        "--capability",
        help="Lead or agent capability. Falls back to CLAUDE_TEAMS_CAPABILITY.",
    ),
]


def _run(awaitable):
    """Run an awaitable from the synchronous CLI surface.

    Args:
        awaitable: Awaitable value to execute.

    Returns:
        object: Awaitable result.

    """
    return asyncio.run(awaitable)


def _ensure_team_exists(team_name: str) -> None:
    try:
        exists = _run(teams.team_exists(team_name))
    except (InvalidNameError, NameTooLongError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not exists:
        err_console.print(f"[red]Team {team_name!r} not found.[/red]")
        raise typer.Exit(code=1)


def _require_cli_principal(team_name: str, capability: str) -> capabilities.Principal:
    _ensure_team_exists(team_name)
    principal = _run(
        capabilities.resolve_principal(team_name, _resolve_capability(capability))
    )
    if principal is None:
        err_console.print(
            "[red]This command requires a valid team capability. "
            "Pass --capability or set CLAUDE_TEAMS_CAPABILITY.[/red]"
        )
        raise typer.Exit(code=1)
    return principal


def _require_cli_lead(team_name: str, capability: str) -> capabilities.Principal:
    principal = _require_cli_principal(team_name, capability)
    if principal["role"] != "lead":
        err_console.print("[red]This command requires the team-lead capability.[/red]")
        raise typer.Exit(code=1)
    return principal


def _require_cli_self_or_lead(
    team_name: str, agent_name: str, capability: str
) -> capabilities.Principal:
    principal = _require_cli_principal(team_name, capability)
    if principal["role"] == "lead" or principal["name"] == agent_name:
        return principal
    err_console.print(
        f"[red]Authenticated principal {principal['name']!r} "
        f"cannot access inbox {agent_name!r}.[/red]"
    )
    raise typer.Exit(code=1)


@app.command()
def serve() -> None:
    """Start the MCP server."""
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    mcp.run()


@app.command()
def backends(output_json: JsonFlag = False) -> None:
    """List available spawner backends.

    Args:
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If no backends are available (exit code 1).

    """
    rows: list[dict[str, str | list[str]]] = []
    for name, backend in registry:
        rows.append(
            {
                "name": name,
                "binary": backend.binary_name,
                "default_model": backend.default_model(),
                "supported_models": backend.supported_models(),
            }
        )

    if output_json:
        console.print_json(json.dumps(rows))
        return

    if not rows:
        err_console.print("[yellow]No backends available.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Available Backends")
    table.add_column("Name", style="bold cyan")
    table.add_column("Binary")
    table.add_column("Default Model", style="green")
    table.add_column("Supported Models")
    for row in rows:
        table.add_row(
            str(row["name"]),
            str(row["binary"]),
            str(row["default_model"]),
            ", ".join(row["supported_models"]),
        )
    console.print(table)
    console.print(
        "\n[dim]Note: Supported models shown are a curated set. Actual"
        " availability depends on authentication state, account tier,"
        " and configured providers.[/dim]"
    )


@app.command()
def templates(output_json: JsonFlag = False) -> None:
    """List registered agent templates.

    Templates are reusable role-context layers applied at
    ``spawn_teammate`` time. Pass a template name via
    ``options.template`` to prepend the role prompt and fill in any
    spawn-option defaults the caller left unset.

    Args:
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If no templates are registered (exit code 1).

    """
    rows = [
        {
            "name": t.name,
            "description": t.description,
            "default_backend": t.default_backend or "",
            "default_model": t.default_model or "",
            "default_subagent_type": t.default_subagent_type or "",
        }
        for t in templates_mod.list_templates()
    ]

    if output_json:
        console.print_json(json.dumps(rows))
        return

    if not rows:
        err_console.print("[yellow]No templates registered.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Agent Templates")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Backend", style="green")
    table.add_column("Model")
    table.add_column("Subagent")
    for row in rows:
        table.add_row(
            row["name"],
            row["description"],
            row["default_backend"] or "[dim]-[/dim]",
            row["default_model"] or "[dim]-[/dim]",
            row["default_subagent_type"] or "[dim]-[/dim]",
        )
    console.print(table)


@app.command()
def presets(output_json: JsonFlag = False) -> None:
    """List registered team presets.

    Presets are declarative team compositions that expand into one
    ``team_create`` call followed by one ``spawn_teammate`` call per
    member. Use ``claude-teams preset launch`` to launch one.

    Args:
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If no presets are registered (exit code 1).

    """
    preset_list = presets_mod.list_presets()

    if output_json:
        rows = [
            {
                "name": p.name,
                "description": p.description,
                "team_description": p.team_description,
                "members": [
                    {
                        "name": m.name,
                        "template": m.template or "",
                        "backend": m.backend or "",
                    }
                    for m in p.members
                ],
            }
            for p in preset_list
        ]
        console.print_json(json.dumps(rows))
        return

    if not preset_list:
        err_console.print("[yellow]No presets registered.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Team Presets")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Members")
    for preset in preset_list:
        members_text = ", ".join(
            f"{m.name}" + (f" ({m.template})" if m.template else "")
            for m in preset.members
        )
        table.add_row(
            preset.name,
            preset.description,
            members_text or "[dim]-[/dim]",
        )
    console.print(table)
    console.print(
        "\n[dim]Note: Launch a preset with `claude-teams preset launch`.[/dim]"
    )


@app.command()
def config(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    capability: CapabilityFlag = "",
    output_json: JsonFlag = False,
) -> None:
    """Show the team configuration.

    Args:
        team_name (str): Name of the team.
        capability (str): Lead capability override or env fallback.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found (exit code 1).

    """
    _require_cli_lead(team_name, capability)
    cfg = _run(teams.read_config(team_name))

    if output_json:
        console.print_json(json.dumps(cfg.model_dump(by_alias=True)))
        return

    console.print(f"[bold]Team:[/bold] {cfg.name}")
    console.print(f"[bold]Description:[/bold] {cfg.description or '(none)'}")
    console.print(f"[bold]Lead:[/bold] {cfg.lead_agent_id}")
    console.print(f"[bold]Members:[/bold] {len(cfg.members)}")

    table = Table(title="Members")
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Model", style="green")
    table.add_column("Backend")
    table.add_column("Active")
    for member in cfg.members:
        if isinstance(member, TeammateMember):
            table.add_row(
                member.name,
                member.agent_type,
                member.model,
                member.backend_type,
                "[green]yes[/green]" if member.is_active else "[dim]no[/dim]",
            )
        else:
            table.add_row(member.name, member.agent_type, member.model, "-", "-")
    console.print(table)


@app.command()
def status(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    capability: CapabilityFlag = "",
    output_json: JsonFlag = False,
) -> None:
    """Show team tasks and member summary.

    Args:
        team_name (str): Name of the team.
        capability (str): Lead capability override or env fallback.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found or task listing fails (exit code 1).

    """
    _require_cli_lead(team_name, capability)
    cfg = _run(teams.read_config(team_name))

    try:
        task_list = _run(tasks.list_tasks(team_name))
    except (InvalidNameError, NameTooLongError, TeamNotFoundValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if output_json:
        payload = {
            "team": team_name,
            "member_count": len(cfg.members),
            "tasks": [
                task.model_dump(by_alias=True, exclude_none=True) for task in task_list
            ],
        }
        console.print_json(json.dumps(payload))
        return

    teammates = [member for member in cfg.members if isinstance(member, TeammateMember)]
    console.print(
        f"[bold]Team:[/bold] {team_name}  ({len(teammates)} teammate(s) + lead)"
    )

    if task_list:
        table = Table(title="Tasks")
        table.add_column("ID", style="dim")
        table.add_column("Status")
        table.add_column("Owner")
        table.add_column("Subject")
        for task in task_list:
            status_style = {
                "pending": "[yellow]pending[/yellow]",
                "in_progress": "[blue]in_progress[/blue]",
                "completed": "[green]completed[/green]",
            }.get(task.status, task.status)
            table.add_row(
                task.id, status_style, task.owner or "[dim]-[/dim]", task.subject
            )
        console.print(table)
    else:
        console.print("[dim]No tasks.[/dim]")


@app.command()
def inbox(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    agent_name: Annotated[str, typer.Argument(help="Agent name.")],
    capability: CapabilityFlag = "",
    unread_only: Annotated[
        bool, typer.Option("--unread", "-u", help="Show only unread messages.")
    ] = False,
    order: Annotated[
        str,
        typer.Option(
            "--order",
            help="Inbox ordering: oldest or newest.",
            case_sensitive=False,
        ),
    ] = "oldest",
    output_json: JsonFlag = False,
) -> None:
    """Read an agent's inbox messages.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent whose inbox to read.
        capability (str): Lead or matching agent capability override.
        unread_only (bool): Whether to show only unread messages.
        order (str): Inbox ordering, oldest or newest.
        output_json (bool): Whether to output as JSON instead of a table.

    """
    _require_cli_self_or_lead(team_name, agent_name, capability)
    normalized_order = order.lower()
    if normalized_order not in {"oldest", "newest"}:
        err_console.print("[red]order must be 'oldest' or 'newest'.[/red]")
        raise typer.Exit(code=1)
    inbox_order: Literal["oldest", "newest"] = (
        "newest" if normalized_order == "newest" else "oldest"
    )
    msgs = _run(
        messaging.read_inbox(
            team_name,
            agent_name,
            unread_only=unread_only,
            mark_as_read=False,
            order=inbox_order,
        )
    )

    if output_json:
        console.print_json(
            json.dumps(
                [msg.model_dump(by_alias=True, exclude_none=True) for msg in msgs]
            )
        )
        return

    if not msgs:
        console.print("[dim]Inbox empty.[/dim]")
        return

    table = Table(title=f"Inbox: {agent_name}")
    table.add_column("From", style="bold")
    table.add_column("Read")
    table.add_column("Time", style="dim")
    table.add_column("Summary / Text")
    for msg in msgs:
        read_mark = "[green]yes[/green]" if msg.read else "[red]no[/red]"
        display = msg.summary or (
            msg.text[:_INBOX_SUMMARY_TRUNC_LEN] + "..."
            if len(msg.text) > _INBOX_SUMMARY_TRUNC_LEN
            else msg.text
        )
        table.add_row(msg.from_, read_mark, msg.timestamp, display)
    console.print(table)


@app.command()
def health(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    agent_name: Annotated[str, typer.Argument(help="Agent name.")],
    capability: CapabilityFlag = "",
    output_json: JsonFlag = False,
) -> None:
    """Check if a teammate's process is alive.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent to check.
        capability (str): Lead capability override or env fallback.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found, teammate not found, or backend
            unavailable (exit code 1).

    """
    _require_cli_lead(team_name, capability)
    cfg = _run(teams.read_config(team_name))

    member = _find_teammate(cfg, agent_name)
    if member is None:
        err_console.print(f"[red]Teammate {agent_name!r} not found.[/red]")
        raise typer.Exit(code=1)

    process_handle = member.process_handle or member.tmux_pane_id
    backend_type = member.backend_type

    try:
        backend_obj = registry.get(backend_type)
    except BackendNotRegisteredError:
        err_console.print(f"[red]Backend {backend_type!r} not available.[/red]")
        raise typer.Exit(code=1) from None

    health_status = _run(run_blocking(backend_obj.health_check, process_handle))

    result = {
        "agent_name": agent_name,
        "alive": health_status.alive,
        "backend": backend_type,
        "detail": health_status.detail,
    }

    if output_json:
        console.print_json(json.dumps(result))
        return

    if health_status.alive:
        console.print(
            f"[green]{agent_name}[/green] is [bold green]alive[/bold green] "
            f"({backend_type})"
        )
    else:
        console.print(
            f"[red]{agent_name}[/red] is [bold red]dead[/bold red] ({backend_type})"
        )
    if health_status.detail:
        console.print(f"  [dim]{health_status.detail}[/dim]")


@app.command()
def kill(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    agent_name: Annotated[str, typer.Argument(help="Agent name to kill.")],
    capability: CapabilityFlag = "",
    output_json: JsonFlag = False,
) -> None:
    """Force-kill a teammate and remove from team.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent to kill.
        capability (str): Lead capability override or env fallback.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found or teammate not found (exit code 1).

    """
    _require_cli_lead(team_name, capability)
    cfg = _run(teams.read_config(team_name))

    member = _find_teammate(cfg, agent_name)
    if member is None:
        err_console.print(
            f"[red]Teammate {agent_name!r} not found in team {team_name!r}.[/red]"
        )
        raise typer.Exit(code=1)

    process_handle = member.process_handle or member.tmux_pane_id
    backend_type = member.backend_type

    backend_kill_skipped = False
    if process_handle:
        try:
            backend_obj = registry.get(backend_type)
            _run(run_blocking(backend_obj.kill, process_handle))
        except BackendNotRegisteredError:
            # Backend binary went missing between spawn and kill (e.g.
            # operator uninstalled the CLI). We cannot invoke
            # backend.kill, but we can still scrub the team config and
            # capability store. Surface the skipped kill so the operator
            # can manually clean up any lingering process.
            backend_kill_skipped = True
            err_console.print(
                f"[yellow]Warning: backend {backend_type!r} no longer "
                f"registered; skipped backend-level kill. Team-side "
                f"cleanup will continue.[/yellow]"
            )

    _run(teams.remove_member(team_name, agent_name))
    _run(capabilities.remove_agent_capability(team_name, agent_name))
    _run(tasks.reset_owner_tasks(team_name, agent_name))

    result: dict[str, str | bool] = {
        "success": True,
        "message": f"{agent_name} has been stopped.",
        "backend_kill_skipped": backend_kill_skipped,
    }

    if output_json:
        console.print_json(json.dumps(result))
        return

    console.print(
        f"[green]{agent_name} has been stopped and removed from {team_name}.[/green]"
    )


def _build_cli_spawn_dependencies() -> SpawnDependencies:
    """Assemble ``SpawnDependencies`` for the CLI orchestration surface.

    Intentionally mirrors the MCP wrapper's factory so both entrypoints
    drive ``orchestration`` with identical helpers — keeping CLI and MCP
    results behaviorally equivalent for the same inputs.
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


@preset_app.command("launch")
def preset_launch(
    preset_name: Annotated[str, typer.Argument(help="Registered preset name.")],
    team_name: Annotated[str, typer.Argument(help="Name for the new team.")],
    description: Annotated[
        str,
        typer.Option(
            "--description",
            "-d",
            help="Override the preset's default team description.",
        ),
    ] = "",
    output_json: JsonFlag = False,
) -> None:
    """Expand a preset into a new team and spawn its teammates.

    Prints the minted lead capability so the operator can export
    ``CLAUDE_TEAMS_CAPABILITY`` and then run lead-only CLI commands or
    attach an MCP session as the team lead.

    Args:
        preset_name (str): Registered preset name.
        team_name (str): Team name to create.
        description (str): Optional override for the preset's team
            description. Empty string falls back to the preset default.
        output_json (bool): Emit a JSON envelope instead of a rich report.

    Raises:
        typer.Exit: Preset unknown, team creation fails, or any teammate
            spawn fails (exit code 1). On mid-fan-out spawn failure the
            team and previously-spawned teammates persist — mirroring the
            MCP contract — so the operator can retry or tear down.

    """
    try:
        preset = presets_mod.get_preset(preset_name)
    except KeyError:
        available = ", ".join(sorted(presets_mod.list_names())) or "(none registered)"
        err_console.print(
            f"[red]Unknown preset {preset_name!r}. Available: {available}[/red]"
        )
        raise typer.Exit(code=1) from None

    effective_description = _resolve_description(description) or preset.team_description
    session_id = f"cli-{uuid.uuid4().hex}"

    try:
        expansion = _run(
            expand_preset_core(
                registry=registry,
                preset=preset,
                team_name=team_name,
                session_id=session_id,
                description=effective_description,
                deps=_build_cli_spawn_dependencies(),
                progress=None,
            )
        )
    except ToolError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        # Domain errors (e.g. TeamAlreadyExistsError, validation failures
        # raised below the MCP boundary) do not subclass ToolError. The
        # MCP wrapper converts them to typed ToolErrors; the CLI has no
        # such boundary, so catch broadly here and surface the message
        # verbatim. Exit code 1 mirrors the ToolError branch so shell
        # callers can keep their existing error handling.
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if output_json:
        payload = {
            "preset": preset_name,
            "team": expansion.team.model_dump(),
            "lead_capability": expansion.lead_capability,
            "members": [m.model_dump() for m in expansion.members],
        }
        console.print_json(json.dumps(payload))
        return

    console.print(
        f"[green]Launched preset {preset_name!r} -> team {team_name!r}[/green] "
        f"({len(expansion.members)} teammate(s))"
    )
    console.print(
        f"[bold]Lead capability:[/bold] [cyan]{expansion.lead_capability}[/cyan]"
    )
    console.print(
        "\n[dim]Export the token to drive the team as lead:\n"
        f"  export CLAUDE_TEAMS_CAPABILITY={expansion.lead_capability}[/dim]"
    )

    if expansion.members:
        table = Table(title="Spawned Members")
        table.add_column("Name", style="bold")
        table.add_column("Agent ID", style="dim")
        for member in expansion.members:
            table.add_row(member.name, member.agent_id)
        console.print(table)


def _find_teammate(cfg: teams.TeamConfig, agent_name: str) -> TeammateMember | None:
    """Find a TeammateMember by name in a TeamConfig.

    Args:
        cfg (teams.TeamConfig): Team configuration to search.
        agent_name (str): Name of the agent to find.

    Returns:
        TeammateMember | None: The teammate if found, None otherwise.

    """
    for member in cfg.members:
        if isinstance(member, TeammateMember) and member.name == agent_name:
            return member
    return None


if __name__ == "__main__":
    app()
