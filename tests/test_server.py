import json
import asyncio
import time
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from fastmcp import Client
from mcp.types import TextContent

from claude_teams import messaging, tasks, teams
from claude_teams.backends import registry as _registry
from claude_teams.backends.base import HealthStatus, SpawnResult as BackendSpawnResult
from claude_teams.models import TeammateMember
from claude_teams.server import mcp


def _make_teammate(name: str, team_name: str, pane_id: str = "%1") -> TeammateMember:
    return TeammateMember(
        agent_id=f"{name}@{team_name}",
        name=name,
        agent_type="teammate",
        model="claude-sonnet-4-20250514",
        prompt="Do stuff",
        color="blue",
        plan_mode_required=False,
        joined_at=int(time.time() * 1000),
        tmux_pane_id=pane_id,
        cwd="/tmp",
    )


def _make_mock_backend(name: str = "claude-code") -> MagicMock:
    """Create a mock backend that satisfies the Backend protocol."""
    mock = MagicMock()
    mock.name = name
    mock.binary_name = "claude"
    mock.is_available.return_value = True
    mock.discover_binary.return_value = "/usr/bin/echo"
    mock.supported_models.return_value = ["haiku", "sonnet", "opus"]
    mock.default_model.return_value = "sonnet"
    mock.resolve_model.side_effect = lambda m: {
        "fast": "haiku",
        "balanced": "sonnet",
        "powerful": "opus",
        "haiku": "haiku",
        "sonnet": "sonnet",
        "opus": "opus",
    }.get(m, m)
    mock.spawn.return_value = BackendSpawnResult(
        process_handle="%mock",
        backend_type=name,
    )
    mock.health_check.return_value = HealthStatus(alive=True, detail="mock check")
    mock.kill.return_value = None
    return mock


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(teams, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(teams, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(tasks, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(messaging, "TEAMS_DIR", tmp_path / "teams")

    # Register a mock claude-code backend in the registry
    mock_backend = _make_mock_backend("claude-code")
    _registry._loaded = True
    _registry._backends = {"claude-code": mock_backend}

    (tmp_path / "teams").mkdir()
    (tmp_path / "tasks").mkdir()
    async with Client(mcp) as c:
        yield c

    # Cleanup: reset registry state
    _registry._loaded = False
    _registry._backends = {}


@pytest.fixture
async def team_client(client: Client):
    """Client with a team created and a teammate spawned (all tiers unlocked)."""
    await client.call_tool("team_create", {"team_name": "test-team"})
    await client.call_tool(
        "spawn_teammate",
        {
            "team_name": "test-team",
            "name": "worker",
            "prompt": "help out",
        },
    )
    return client


def _text(result) -> str:
    """Extract text from the first content item of a tool result."""
    item = result.content[0]
    assert isinstance(item, TextContent)
    return item.text


def _data(result):
    """Extract raw Python data from a successful CallToolResult."""
    if result.content:
        return json.loads(_text(result))
    return result.data


# ---------------------------------------------------------------------------
# Progressive disclosure tests
# ---------------------------------------------------------------------------


class TestProgressiveDisclosure:
    async def test_only_bootstrap_tools_at_startup(self, client: Client):
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        # Bootstrap tools should be visible
        assert "team_create" in names
        assert "team_delete" in names
        assert "list_backends" in names
        assert "read_config" in names
        # Team-tier tools should NOT be visible
        assert "spawn_teammate" not in names
        assert "send_message" not in names
        assert "task_create" not in names
        # Teammate-tier tools should NOT be visible
        assert "force_kill_teammate" not in names
        assert "poll_inbox" not in names
        assert "health_check" not in names

    async def test_team_tools_visible_after_create(self, client: Client):
        await client.call_tool("team_create", {"team_name": "vis-test"})
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        # Team-tier tools should now be visible
        assert "spawn_teammate" in names
        assert "send_message" in names
        assert "task_create" in names
        assert "task_update" in names
        assert "task_list" in names
        assert "task_get" in names
        assert "read_inbox" in names
        # Teammate-tier tools should still NOT be visible
        assert "force_kill_teammate" not in names
        assert "poll_inbox" not in names

    async def test_teammate_tools_visible_after_spawn(self, client: Client):
        await client.call_tool("team_create", {"team_name": "vis-test2"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "vis-test2",
                "name": "coder",
                "prompt": "write code",
            },
        )
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        # All tiers should be visible
        assert "force_kill_teammate" in names
        assert "poll_inbox" in names
        assert "process_shutdown_approved" in names
        assert "health_check" in names

    async def test_tools_hidden_after_delete(self, client: Client):
        await client.call_tool("team_create", {"team_name": "vis-del"})
        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "vis-del",
                "name": "temp",
                "prompt": "temporary",
            },
        )
        # Remove member so delete succeeds
        teams.remove_member("vis-del", "temp")
        await client.call_tool("team_delete", {"team_name": "vis-del"})
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        # Only bootstrap should remain
        assert "team_create" in names
        assert "list_backends" in names
        assert "spawn_teammate" not in names
        assert "force_kill_teammate" not in names

    async def test_re_enable_cycle(self, client: Client):
        # Create -> delete -> re-create cycle
        await client.call_tool("team_create", {"team_name": "cycle1"})
        await client.call_tool("team_delete", {"team_name": "cycle1"})
        # After delete, team tools should be gone
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        assert "spawn_teammate" not in names
        # Re-create should bring them back
        await client.call_tool("team_create", {"team_name": "cycle2"})
        tool_list = await client.list_tools()
        names = {t.name for t in tool_list}
        assert "spawn_teammate" in names


class TestOneShotBackendRelay:
    async def test_should_relay_codex_result_to_team_lead(self, client: Client):
        await client.call_tool("team_create", {"team_name": "oneshot"})

        mock_codex = _make_mock_backend("codex")

        def _spawn_side_effect(request):
            output_path = Path(request.extra["output_last_message_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("codex teammate result")
            return BackendSpawnResult(process_handle="%codex", backend_type="codex")

        mock_codex.spawn.side_effect = _spawn_side_effect
        mock_codex.health_check.return_value = HealthStatus(
            alive=False,
            detail="one-shot complete",
        )
        _registry._backends["codex"] = mock_codex

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "oneshot",
                "name": "codex-worker",
                "backend": "codex",
                "model": "gpt-5.3-codex",
                "prompt": "reply with result",
            },
        )

        await asyncio.sleep(0.1)

        inbox = _data(
            await client.call_tool(
                "read_inbox",
                {
                    "team_name": "oneshot",
                    "agent_name": "team-lead",
                    "unread_only": True,
                },
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "codex-worker"
        assert inbox[0]["summary"] == "teammate_result"
        assert "codex teammate result" in inbox[0]["text"]

    async def test_should_relay_when_output_exists_even_if_pane_is_alive(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "oneshot-alive"})

        mock_codex = _make_mock_backend("codex")

        def _spawn_side_effect(request):
            output_path = Path(request.extra["output_last_message_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("codex pane-still-alive result")
            return BackendSpawnResult(
                process_handle="%codex-alive", backend_type="codex"
            )

        mock_codex.spawn.side_effect = _spawn_side_effect
        mock_codex.health_check.return_value = HealthStatus(
            alive=True,
            detail="tmux pane still open",
        )
        _registry._backends["codex"] = mock_codex

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "oneshot-alive",
                "name": "codex-worker",
                "backend": "codex",
                "model": "gpt-5.3-codex",
                "prompt": "reply with result",
            },
        )

        await asyncio.sleep(0.1)

        inbox = _data(
            await client.call_tool(
                "read_inbox",
                {
                    "team_name": "oneshot-alive",
                    "agent_name": "team-lead",
                    "unread_only": True,
                },
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "codex-worker"
        assert inbox[0]["summary"] == "teammate_result"
        assert "codex pane-still-alive result" in inbox[0]["text"]


# ---------------------------------------------------------------------------
# Existing tests — updated to use team_client where needed
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    async def test_should_reject_second_team_in_same_session(self, client: Client):
        await client.call_tool("team_create", {"team_name": "alpha"})
        result = await client.call_tool(
            "team_create", {"team_name": "beta"}, raise_on_error=False
        )
        assert result.is_error is True
        assert "alpha" in _text(result)

    async def test_should_reject_unknown_agent_in_force_kill(self, team_client: Client):
        result = await team_client.call_tool(
            "force_kill_teammate",
            {"team_name": "test-team", "agent_name": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_should_reject_invalid_message_type(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t_msg"})
        result = await client.call_tool(
            "send_message",
            {"team_name": "t_msg", "type": "bogus"},
            raise_on_error=False,
        )
        assert result.is_error is True


class TestDeletedTaskGuard:
    async def test_should_not_send_assignment_when_task_deleted(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t2"})
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "t2", "subject": "doomed", "description": "will delete"},
            )
        )
        await client.call_tool(
            "task_update",
            {
                "team_name": "t2",
                "task_id": created["id"],
                "status": "deleted",
                "owner": "worker",
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t2", "agent_name": "worker"}
            )
        )
        assert inbox == []

    async def test_should_send_assignment_when_owner_set_on_live_task(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "t2b"})
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "t2b", "subject": "live", "description": "stays"},
            )
        )
        await client.call_tool(
            "task_update",
            {"team_name": "t2b", "task_id": created["id"], "owner": "worker"},
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t2b", "agent_name": "worker"}
            )
        )
        assert len(inbox) == 1
        payload = json.loads(inbox[0]["text"])
        assert payload["type"] == "task_assignment"
        assert payload["taskId"] == created["id"]


class TestShutdownResponseSender:
    async def test_should_populate_correct_from_and_pane_id_on_approve(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "t3"})
        teams.add_member("t3", _make_teammate("worker", "t3", pane_id="%42"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t3",
                "type": "shutdown_response",
                "sender": "worker",
                "request_id": "req-1",
                "approve": True,
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t3", "agent_name": "team-lead"}
            )
        )
        assert len(inbox) == 1
        payload = json.loads(inbox[0]["text"])
        assert payload["type"] == "shutdown_approved"
        assert payload["from"] == "worker"
        assert payload["paneId"] == "%42"
        assert payload["requestId"] == "req-1"

    async def test_should_attribute_rejection_to_sender(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t3b"})
        teams.add_member("t3b", _make_teammate("rebel", "t3b"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t3b",
                "type": "shutdown_response",
                "sender": "rebel",
                "request_id": "req-2",
                "approve": False,
                "content": "still busy",
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t3b", "agent_name": "team-lead"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "rebel"
        assert inbox[0]["text"] == "still busy"


class TestPlanApprovalSender:
    async def test_should_use_sender_as_from_on_approve(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t_plan"})
        teams.add_member("t_plan", _make_teammate("dev", "t_plan"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t_plan",
                "type": "plan_approval_response",
                "sender": "team-lead",
                "recipient": "dev",
                "request_id": "plan-1",
                "approve": True,
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t_plan", "agent_name": "dev"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "team-lead"
        payload = json.loads(inbox[0]["text"])
        assert payload["type"] == "plan_approval"
        assert payload["approved"] is True

    async def test_should_use_sender_as_from_on_reject(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t_plan2"})
        teams.add_member("t_plan2", _make_teammate("dev2", "t_plan2"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t_plan2",
                "type": "plan_approval_response",
                "sender": "team-lead",
                "recipient": "dev2",
                "approve": False,
                "content": "needs error handling",
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t_plan2", "agent_name": "dev2"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["from"] == "team-lead"
        assert inbox[0]["text"] == "needs error handling"


class TestWiring:
    async def test_should_round_trip_task_create_and_list(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t4"})
        await client.call_tool(
            "task_create",
            {"team_name": "t4", "subject": "first", "description": "d1"},
        )
        await client.call_tool(
            "task_create",
            {"team_name": "t4", "subject": "second", "description": "d2"},
        )
        result = _data(await client.call_tool("task_list", {"team_name": "t4"}))
        assert len(result) == 2
        assert result[0]["subject"] == "first"
        assert result[1]["subject"] == "second"

    async def test_should_round_trip_send_message_and_read_inbox(self, client: Client):
        await client.call_tool("team_create", {"team_name": "t5"})
        teams.add_member("t5", _make_teammate("bob", "t5"))
        await client.call_tool(
            "send_message",
            {
                "team_name": "t5",
                "type": "message",
                "recipient": "bob",
                "content": "hello bob",
                "summary": "greeting",
            },
        )
        inbox = _data(
            await client.call_tool(
                "read_inbox", {"team_name": "t5", "agent_name": "bob"}
            )
        )
        assert len(inbox) == 1
        assert inbox[0]["text"] == "hello bob"
        assert inbox[0]["from"] == "team-lead"


class TestTeamDeleteClearsSession:
    async def test_should_allow_new_team_after_delete(self, client: Client):
        await client.call_tool("team_create", {"team_name": "first"})
        await client.call_tool("team_delete", {"team_name": "first"})
        result = await client.call_tool("team_create", {"team_name": "second"})
        data = _data(result)
        assert data["team_name"] == "second"


class TestSendMessageValidation:
    async def test_should_reject_empty_content(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv1"})
        teams.add_member("tv1", _make_teammate("bob", "tv1"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv1",
                "type": "message",
                "recipient": "bob",
                "content": "",
                "summary": "hi",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "content" in _text(result).lower()

    async def test_should_reject_empty_summary(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv2"})
        teams.add_member("tv2", _make_teammate("bob", "tv2"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv2",
                "type": "message",
                "recipient": "bob",
                "content": "hi",
                "summary": "",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "summary" in _text(result).lower()

    async def test_should_reject_empty_recipient(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv3"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv3",
                "type": "message",
                "recipient": "",
                "content": "hi",
                "summary": "hi",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "recipient" in _text(result).lower()

    async def test_should_reject_nonexistent_recipient(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv4"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv4",
                "type": "message",
                "recipient": "ghost",
                "content": "hi",
                "summary": "hi",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_should_pass_target_color(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv5"})
        teams.add_member("tv5", _make_teammate("bob", "tv5"))
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv5",
                "type": "message",
                "recipient": "bob",
                "content": "hey",
                "summary": "greet",
            },
        )
        data = _data(result)
        assert data["routing"]["targetColor"] == "blue"

    async def test_should_reject_broadcast_empty_summary(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv6"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tv6",
                "type": "broadcast",
                "content": "hello",
                "summary": "",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "summary" in _text(result).lower()

    async def test_should_reject_shutdown_request_to_team_lead(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv7"})
        result = await client.call_tool(
            "send_message",
            {"team_name": "tv7", "type": "shutdown_request", "recipient": "team-lead"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "team-lead" in _text(result)

    async def test_should_reject_shutdown_request_to_nonexistent(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tv8"})
        result = await client.call_tool(
            "send_message",
            {"team_name": "tv8", "type": "shutdown_request", "recipient": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)


class TestProcessShutdownGuard:
    async def test_should_reject_shutdown_of_team_lead(self, team_client: Client):
        result = await team_client.call_tool(
            "process_shutdown_approved",
            {"team_name": "test-team", "agent_name": "team-lead"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "team-lead" in _text(result)


class TestErrorWrapping:
    async def test_read_config_wraps_file_not_found(self, client: Client):
        result = await client.call_tool(
            "read_config",
            {"team_name": "nonexistent"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()

    async def test_task_get_wraps_file_not_found(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tew"})
        result = await client.call_tool(
            "task_get",
            {"team_name": "tew", "task_id": "999"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()

    async def test_task_update_wraps_file_not_found(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tew2"})
        result = await client.call_tool(
            "task_update",
            {"team_name": "tew2", "task_id": "999", "status": "completed"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()

    async def test_task_create_wraps_nonexistent_team(self, client: Client):
        # Create a team to unlock team-tier tools, then target a different team
        await client.call_tool("team_create", {"team_name": "real-team"})
        result = await client.call_tool(
            "task_create",
            {"team_name": "ghost-team", "subject": "x", "description": "y"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "does not exist" in _text(result).lower()

    async def test_task_update_wraps_validation_error(self, client: Client):
        await client.call_tool("team_create", {"team_name": "tew3"})
        created = _data(
            await client.call_tool(
                "task_create",
                {"team_name": "tew3", "subject": "S", "description": "d"},
            )
        )
        await client.call_tool(
            "task_update",
            {"team_name": "tew3", "task_id": created["id"], "status": "in_progress"},
        )
        result = await client.call_tool(
            "task_update",
            {"team_name": "tew3", "task_id": created["id"], "status": "pending"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "cannot transition" in _text(result).lower()

    async def test_task_list_wraps_nonexistent_team(self, client: Client):
        # Create a team to unlock team-tier tools, then target a different team
        await client.call_tool("team_create", {"team_name": "real-team2"})
        result = await client.call_tool(
            "task_list",
            {"team_name": "ghost-team"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "does not exist" in _text(result).lower()


class TestPollInbox:
    async def test_should_return_empty_on_timeout(self, team_client: Client):
        result = _data(
            await team_client.call_tool(
                "poll_inbox",
                {"team_name": "test-team", "agent_name": "nobody", "timeout_ms": 100},
            )
        )
        assert result == []

    async def test_should_return_messages_when_present(self, team_client: Client):
        await team_client.call_tool(
            "send_message",
            {
                "team_name": "test-team",
                "type": "message",
                "recipient": "worker",
                "content": "wake up",
                "summary": "nudge",
            },
        )
        result = _data(
            await team_client.call_tool(
                "poll_inbox",
                {"team_name": "test-team", "agent_name": "worker", "timeout_ms": 100},
            )
        )
        # worker already has the initial prompt message + new message
        assert any(msg["text"] == "wake up" for msg in result)

    async def test_should_return_existing_messages_with_zero_timeout(
        self, team_client: Client
    ):
        await team_client.call_tool(
            "send_message",
            {
                "team_name": "test-team",
                "type": "message",
                "recipient": "worker",
                "content": "instant",
                "summary": "fast",
            },
        )
        result = _data(
            await team_client.call_tool(
                "poll_inbox",
                {"team_name": "test-team", "agent_name": "worker", "timeout_ms": 0},
            )
        )
        assert any(msg["text"] == "instant" for msg in result)


class TestTeamDeleteErrorWrapping:
    async def test_should_reject_delete_with_active_members(self, client: Client):
        await client.call_tool("team_create", {"team_name": "td1"})
        teams.add_member("td1", _make_teammate("worker", "td1"))
        result = await client.call_tool(
            "team_delete",
            {"team_name": "td1"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "member" in _text(result).lower()

    async def test_should_reject_delete_nonexistent_team(self, client: Client):
        result = await client.call_tool(
            "team_delete",
            {"team_name": "ghost-team"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "Traceback" not in _text(result)


class TestPlanApprovalValidation:
    async def test_should_reject_plan_approval_to_nonexistent_recipient(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "tp1"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tp1",
                "type": "plan_approval_response",
                "recipient": "ghost",
                "approve": True,
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_should_reject_plan_approval_with_empty_recipient(
        self, client: Client
    ):
        await client.call_tool("team_create", {"team_name": "tp2"})
        result = await client.call_tool(
            "send_message",
            {
                "team_name": "tp2",
                "type": "plan_approval_response",
                "recipient": "",
                "approve": True,
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "recipient" in _text(result).lower()


# ---------------------------------------------------------------------------
# New backend-aware tools
# ---------------------------------------------------------------------------


class TestListBackends:
    async def test_returns_registered_backends(self, client: Client):
        result = _data(await client.call_tool("list_backends", {}))
        assert isinstance(result, list)
        assert len(result) >= 1
        backend_info = result[0]
        assert "name" in backend_info
        assert "binary" in backend_info
        assert "available" in backend_info
        assert "defaultModel" in backend_info
        assert "supportedModels" in backend_info

    async def test_returns_correct_backend_name(self, client: Client):
        result = _data(await client.call_tool("list_backends", {}))
        names = [backend["name"] for backend in result]
        assert "claude-code" in names

    async def test_returns_empty_when_no_backends(self, client: Client):
        # Clear all backends from registry
        _registry._backends = {}
        result = _data(await client.call_tool("list_backends", {}))
        assert result == []
        # Restore for subsequent tests
        _registry._backends = {"claude-code": _make_mock_backend("claude-code")}


class TestHealthCheck:
    async def test_returns_alive_for_running_teammate(self, team_client: Client):
        result = _data(
            await team_client.call_tool(
                "health_check", {"team_name": "test-team", "agent_name": "worker"}
            )
        )

        assert result["alive"] is True
        assert result["agent_name"] == "worker"
        assert "backend" in result
        assert "detail" in result

    async def test_returns_dead_when_backend_says_dead(self, team_client: Client):
        # Override mock to return dead
        mock_backend = cast(MagicMock, _registry._backends["claude-code"])
        mock_backend.health_check.return_value = HealthStatus(
            alive=False, detail="pane gone"
        )

        result = _data(
            await team_client.call_tool(
                "health_check", {"team_name": "test-team", "agent_name": "worker"}
            )
        )

        assert result["alive"] is False
        # Restore original behavior
        mock_backend.health_check.return_value = HealthStatus(
            alive=True, detail="mock check"
        )

    async def test_rejects_nonexistent_teammate(self, team_client: Client):
        result = await team_client.call_tool(
            "health_check",
            {"team_name": "test-team", "agent_name": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_rejects_nonexistent_team(self, team_client: Client):
        result = await team_client.call_tool(
            "health_check",
            {"team_name": "no-such-team", "agent_name": "worker"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "not found" in _text(result).lower()


class TestSpawnWithBackend:
    async def test_spawns_with_explicit_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb1"})
        result = _data(
            await client.call_tool(
                "spawn_teammate",
                {
                    "team_name": "sb1",
                    "name": "coder",
                    "prompt": "write code",
                    "backend": "claude-code",
                },
            )
        )
        assert result["name"] == "coder"
        assert result["team_name"] == "sb1"

    async def test_spawns_with_default_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb2"})
        result = _data(
            await client.call_tool(
                "spawn_teammate",
                {
                    "team_name": "sb2",
                    "name": "coder",
                    "prompt": "write code",
                },
            )
        )
        assert result["name"] == "coder"

    async def test_rejects_invalid_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb3"})
        result = await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb3",
                "name": "coder",
                "prompt": "write code",
                "backend": "nonexistent-backend",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "nonexistent-backend" in _text(result)

    async def test_resolves_generic_model_name(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb4"})

        mock_backend = cast(MagicMock, _registry._backends["claude-code"])
        mock_backend.resolve_model.reset_mock()

        await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb4",
                "name": "coder",
                "prompt": "write code",
                "model": "fast",
            },
        )
        mock_backend.resolve_model.assert_called_with("fast")

    async def test_rejects_invalid_model_for_backend(self, client: Client):
        await client.call_tool("team_create", {"team_name": "sb5"})

        mock_backend = cast(MagicMock, _registry._backends["claude-code"])
        original_side_effect = mock_backend.resolve_model.side_effect
        mock_backend.resolve_model.side_effect = ValueError("Unsupported model 'bogus'")

        result = await client.call_tool(
            "spawn_teammate",
            {
                "team_name": "sb5",
                "name": "coder",
                "prompt": "write code",
                "model": "bogus",
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "bogus" in _text(result)

        mock_backend.resolve_model.side_effect = original_side_effect


class TestForceKillWithBackend:
    async def test_kills_via_correct_backend(self, team_client: Client):
        mate = _make_teammate("victim", "test-team", pane_id="%77")
        mate.backend_type = "claude-code"
        mate.process_handle = "%77"
        teams.add_member("test-team", mate)

        mock_backend = cast(MagicMock, _registry._backends["claude-code"])
        mock_backend.kill.reset_mock()

        result = _data(
            await team_client.call_tool(
                "force_kill_teammate",
                {"team_name": "test-team", "agent_name": "victim"},
            )
        )

        assert result["success"] is True
        mock_backend.kill.assert_called_once_with("%77")

    async def test_legacy_tmux_backend_type_maps_to_claude_code(
        self, team_client: Client
    ):
        mate = _make_teammate("oldmate", "test-team", pane_id="%88")
        mate.backend_type = "tmux"
        mate.process_handle = "%88"
        teams.add_member("test-team", mate)

        mock_backend = cast(MagicMock, _registry._backends["claude-code"])
        mock_backend.kill.reset_mock()

        result = _data(
            await team_client.call_tool(
                "force_kill_teammate",
                {"team_name": "test-team", "agent_name": "oldmate"},
            )
        )

        assert result["success"] is True
        mock_backend.kill.assert_called_once_with("%88")

    async def test_rejects_nonexistent_teammate(self, team_client: Client):
        result = await team_client.call_tool(
            "force_kill_teammate",
            {"team_name": "test-team", "agent_name": "ghost"},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert "ghost" in _text(result)

    async def test_skips_kill_when_backend_unavailable(self, team_client: Client):
        mate = _make_teammate("orphan", "test-team", pane_id="%99")
        mate.backend_type = "nonexistent"
        mate.process_handle = "%99"
        teams.add_member("test-team", mate)

        # Should not raise even if backend is unavailable
        result = _data(
            await team_client.call_tool(
                "force_kill_teammate",
                {"team_name": "test-team", "agent_name": "orphan"},
            )
        )
        assert result["success"] is True
