from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from claude_teams import teams, messaging
from claude_teams.models import COLOR_PALETTE, TeammateMember
from claude_teams.spawner import (
    assign_color,
    build_spawn_command,
    discover_claude_binary,
    kill_tmux_pane,
    spawn_teammate,
)


TEAM = "test-team"
SESSION_ID = "test-session-id"


@pytest.fixture
def team_dir(tmp_claude_dir: Path) -> Path:
    teams.create_team(TEAM, session_id=SESSION_ID, base_dir=tmp_claude_dir)
    return tmp_claude_dir


def _make_member(
    name: str,
    team: str = TEAM,
    color: str = "blue",
    model: str = "sonnet",
    agent_type: str = "general-purpose",
    cwd: str = "/tmp",
) -> TeammateMember:
    return TeammateMember(
        agent_id=f"{name}@{team}",
        name=name,
        agent_type=agent_type,
        model=model,
        prompt=f"You are {name}",
        color=color,
        joined_at=0,
        tmux_pane_id="",
        cwd=cwd,
    )


class TestDiscoverClaudeBinary:
    @patch("claude_teams.spawner.shutil.which")
    def test_found(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/local/bin/claude"
        assert discover_claude_binary() == "/usr/local/bin/claude"
        mock_which.assert_called_once_with("claude")

    @patch("claude_teams.spawner.shutil.which")
    def test_not_found(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        with pytest.raises(FileNotFoundError):
            discover_claude_binary()


class TestAssignColor:
    def test_first_teammate_is_blue(self, team_dir: Path) -> None:
        color = assign_color(TEAM, base_dir=team_dir)
        assert color == "blue"

    def test_cycles(self, team_dir: Path) -> None:
        for idx in range(len(COLOR_PALETTE)):
            member = _make_member(f"agent-{idx}", color=COLOR_PALETTE[idx])
            teams.add_member(TEAM, member, base_dir=team_dir)

        color = assign_color(TEAM, base_dir=team_dir)
        assert color == COLOR_PALETTE[0]


class TestBuildSpawnCommand:
    def test_format(self) -> None:
        member = _make_member("researcher")
        cmd = build_spawn_command(member, "/usr/local/bin/claude", "lead-sess-1")
        assert "CLAUDECODE=1" in cmd
        assert "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1" in cmd
        assert "/usr/local/bin/claude" in cmd
        assert "--agent-id" in cmd
        assert "--agent-name" in cmd
        assert "--team-name" in cmd
        assert "--agent-color" in cmd
        assert "--parent-session-id" in cmd
        assert "--agent-type" in cmd
        assert "--model" in cmd
        assert "cd /tmp" in cmd
        assert "--plan-mode-required" not in cmd

    def test_with_plan_mode(self) -> None:
        member = _make_member("researcher")
        member.plan_mode_required = True
        cmd = build_spawn_command(member, "/usr/local/bin/claude", "lead-sess-1")
        assert "--plan-mode-required" in cmd


class TestSpawnTeammateNameValidation:
    def test_should_reject_empty_name(self, team_dir: Path) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            spawn_teammate(
                TEAM, "", "prompt", "/bin/echo", SESSION_ID, base_dir=team_dir
            )

    def test_should_reject_name_with_special_chars(self, team_dir: Path) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            spawn_teammate(
                TEAM, "agent!@#", "prompt", "/bin/echo", SESSION_ID, base_dir=team_dir
            )

    def test_should_reject_name_exceeding_64_chars(self, team_dir: Path) -> None:
        with pytest.raises(ValueError, match="too long"):
            spawn_teammate(
                TEAM, "a" * 65, "prompt", "/bin/echo", SESSION_ID, base_dir=team_dir
            )

    def test_should_reject_reserved_name_team_lead(self, team_dir: Path) -> None:
        with pytest.raises(ValueError, match="reserved"):
            spawn_teammate(
                TEAM, "team-lead", "prompt", "/bin/echo", SESSION_ID, base_dir=team_dir
            )


class TestSpawnTeammate:
    @patch("claude_teams.spawner.TmuxCLIController")
    def test_registers_member_before_spawn(
        self, mock_ctrl_cls: MagicMock, team_dir: Path
    ) -> None:
        mock_ctrl_cls.return_value.launch_cli.return_value = "remote:1.0"
        spawn_teammate(
            TEAM,
            "researcher",
            "Do research",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
        )
        config = teams.read_config(TEAM, base_dir=team_dir)
        names = [member.name for member in config.members]
        assert "researcher" in names

    @patch("claude_teams.spawner.TmuxCLIController")
    def test_writes_prompt_to_inbox(
        self, mock_ctrl_cls: MagicMock, team_dir: Path
    ) -> None:
        mock_ctrl_cls.return_value.launch_cli.return_value = "remote:1.0"
        spawn_teammate(
            TEAM,
            "researcher",
            "Do research",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
        )
        msgs = messaging.read_inbox(TEAM, "researcher", base_dir=team_dir)
        assert len(msgs) == 1
        assert msgs[0].from_ == "team-lead"
        assert msgs[0].text == "Do research"

    @patch("claude_teams.spawner.TmuxCLIController")
    def test_updates_pane_id(self, mock_ctrl_cls: MagicMock, team_dir: Path) -> None:
        mock_ctrl_cls.return_value.launch_cli.return_value = "remote:1.0"
        member = spawn_teammate(
            TEAM,
            "researcher",
            "Do research",
            "/usr/local/bin/claude",
            SESSION_ID,
            base_dir=team_dir,
        )
        assert member.tmux_pane_id == "remote:1.0"
        config = teams.read_config(TEAM, base_dir=team_dir)
        found = [member for member in config.members if member.name == "researcher"]
        assert found[0].tmux_pane_id == "remote:1.0"


class TestKillTmuxPane:
    @patch("claude_teams.spawner.TmuxCLIController")
    def test_calls_controller_kill_pane(self, mock_ctrl_cls: MagicMock) -> None:
        kill_tmux_pane("%99")
        mock_ctrl_cls.return_value.kill_pane.assert_called_once_with(pane_id="%99")
