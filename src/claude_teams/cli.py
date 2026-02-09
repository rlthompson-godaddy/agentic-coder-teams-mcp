"""Typer CLI for claude-teams.

Provides human-friendly commands that operate on the same file-based state
as the MCP server.  Both the CLI and MCP server can run concurrently thanks
to ``fcntl.flock()`` guards in the core modules.
"""

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from claude_teams import messaging, tasks, teams
from claude_teams.backends.registry import registry
from claude_teams.models import TeammateMember
from claude_teams.server import mcp

app = typer.Typer(
    name="claude-teams",
    help="CLI for orchestrating Claude Code agent teams.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

JsonFlag = Annotated[
    bool,
    typer.Option("--json", "-j", help="Output as JSON instead of a table."),
]


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve() -> None:
    """Start the MCP server."""
    mcp.run()


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@app.command()
def config(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    output_json: JsonFlag = False,
) -> None:
    """Show the team configuration.

    Args:
        team_name (str): Name of the team.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found (exit code 1).
    """
    try:
        cfg = teams.read_config(team_name)
    except FileNotFoundError:
        err_console.print(f"[red]Team {team_name!r} not found.[/red]")
        raise typer.Exit(code=1)

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


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    output_json: JsonFlag = False,
) -> None:
    """Show team tasks and member summary.

    Args:
        team_name (str): Name of the team.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found or task listing fails (exit code 1).
    """
    try:
        cfg = teams.read_config(team_name)
    except FileNotFoundError:
        err_console.print(f"[red]Team {team_name!r} not found.[/red]")
        raise typer.Exit(code=1)

    try:
        task_list = tasks.list_tasks(team_name)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

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


# ---------------------------------------------------------------------------
# inbox
# ---------------------------------------------------------------------------


@app.command()
def inbox(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    agent_name: Annotated[str, typer.Argument(help="Agent name.")],
    unread_only: Annotated[
        bool, typer.Option("--unread", "-u", help="Show only unread messages.")
    ] = False,
    output_json: JsonFlag = False,
) -> None:
    """Read an agent's inbox messages.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent whose inbox to read.
        unread_only (bool): Whether to show only unread messages.
        output_json (bool): Whether to output as JSON instead of a table.
    """
    msgs = messaging.read_inbox(
        team_name,
        agent_name,
        unread_only=unread_only,
        mark_as_read=False,
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
            msg.text[:80] + "..." if len(msg.text) > 80 else msg.text
        )
        table.add_row(msg.from_, read_mark, msg.timestamp, display)
    console.print(table)


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@app.command()
def health(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    agent_name: Annotated[str, typer.Argument(help="Agent name.")],
    output_json: JsonFlag = False,
) -> None:
    """Check if a teammate's process is alive.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent to check.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found, teammate not found, or backend unavailable (exit code 1).
    """
    try:
        cfg = teams.read_config(team_name)
    except FileNotFoundError:
        err_console.print(f"[red]Team {team_name!r} not found.[/red]")
        raise typer.Exit(code=1)

    member = _find_teammate(cfg, agent_name)
    if member is None:
        err_console.print(f"[red]Teammate {agent_name!r} not found.[/red]")
        raise typer.Exit(code=1)

    process_handle = member.process_handle or member.tmux_pane_id
    backend_type = member.backend_type

    try:
        backend_obj = registry.get(backend_type)
    except KeyError:
        err_console.print(f"[red]Backend {backend_type!r} not available.[/red]")
        raise typer.Exit(code=1)

    health_status = backend_obj.health_check(process_handle)

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
            f"[green]{agent_name}[/green] is [bold green]alive[/bold green] ({backend_type})"
        )
    else:
        console.print(
            f"[red]{agent_name}[/red] is [bold red]dead[/bold red] ({backend_type})"
        )
    if health_status.detail:
        console.print(f"  [dim]{health_status.detail}[/dim]")


# ---------------------------------------------------------------------------
# kill
# ---------------------------------------------------------------------------


@app.command()
def kill(
    team_name: Annotated[str, typer.Argument(help="Team name.")],
    agent_name: Annotated[str, typer.Argument(help="Agent name to kill.")],
    output_json: JsonFlag = False,
) -> None:
    """Force-kill a teammate and remove from team.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent to kill.
        output_json (bool): Whether to output as JSON instead of a table.

    Raises:
        typer.Exit: If team not found or teammate not found (exit code 1).
    """
    try:
        cfg = teams.read_config(team_name)
    except FileNotFoundError:
        err_console.print(f"[red]Team {team_name!r} not found.[/red]")
        raise typer.Exit(code=1)

    member = _find_teammate(cfg, agent_name)
    if member is None:
        err_console.print(
            f"[red]Teammate {agent_name!r} not found in team {team_name!r}.[/red]"
        )
        raise typer.Exit(code=1)

    process_handle = member.process_handle or member.tmux_pane_id
    backend_type = member.backend_type

    if process_handle:
        try:
            backend_obj = registry.get(backend_type)
            backend_obj.kill(process_handle)
        except KeyError:
            pass  # backend unavailable; process may already be dead

    teams.remove_member(team_name, agent_name)
    tasks.reset_owner_tasks(team_name, agent_name)

    result = {"success": True, "message": f"{agent_name} has been stopped."}

    if output_json:
        console.print_json(json.dumps(result))
        return

    console.print(
        f"[green]{agent_name} has been stopped and removed from {team_name}.[/green]"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
