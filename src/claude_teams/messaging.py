import fcntl
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from claude_teams.models import (
    InboxMessage,
    ShutdownRequest,
    TaskAssignment,
    TaskFile,
)

TEAMS_DIR = Path.home() / ".claude" / "teams"


def _teams_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "teams") if base_dir else TEAMS_DIR


@contextmanager
def file_lock(lock_path: Path):
    """Context manager providing exclusive file-based lock using fcntl.

    Args:
        lock_path (Path): Path to the lock file (created if missing).

    Yields:
        None: Control returns to caller while lock is held.
    """
    lock_path.touch(exist_ok=True)
    with open(lock_path) as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format with millisecond precision.

    Returns:
        str: ISO timestamp string (e.g., "2024-01-15T14:30:45.123Z").
    """
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def inbox_path(team_name: str, agent_name: str, base_dir: Path | None = None) -> Path:
    """Return the file path to an agent's inbox JSON file.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        Path: Full path to the agent's inbox file.
    """
    return _teams_dir(base_dir) / team_name / "inboxes" / f"{agent_name}.json"


def ensure_inbox(team_name: str, agent_name: str, base_dir: Path | None = None) -> Path:
    """Ensure an agent's inbox file exists, creating it if missing.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        Path: Full path to the agent's inbox file.
    """
    path = inbox_path(team_name, agent_name, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]")
    return path


def read_inbox(
    team_name: str,
    agent_name: str,
    unread_only: bool = False,
    mark_as_read: bool = True,
    base_dir: Path | None = None,
) -> list[InboxMessage]:
    """Read messages from an agent's inbox, optionally filtering and marking as read.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent.
        unread_only (bool): If True, return only unread messages.
        mark_as_read (bool): If True, mark returned messages as read.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        list[InboxMessage]: List of inbox messages matching the criteria.
    """
    path = inbox_path(team_name, agent_name, base_dir)
    if not path.exists():
        return []

    if mark_as_read:
        lock_path = path.parent / ".lock"
        with file_lock(lock_path):
            raw_list = json.loads(path.read_text())
            all_msgs = [InboxMessage.model_validate(entry) for entry in raw_list]

            if unread_only:
                result = [msg for msg in all_msgs if not msg.read]
            else:
                result = list(all_msgs)

            if result:
                for msg in all_msgs:
                    if msg in result:
                        msg.read = True
                serialized = [
                    msg.model_dump(by_alias=True, exclude_none=True) for msg in all_msgs
                ]
                path.write_text(json.dumps(serialized))

            return result
    else:
        raw_list = json.loads(path.read_text())
        all_msgs = [InboxMessage.model_validate(entry) for entry in raw_list]

        if unread_only:
            return [msg for msg in all_msgs if not msg.read]
        return list(all_msgs)


def append_message(
    team_name: str,
    agent_name: str,
    message: InboxMessage,
    base_dir: Path | None = None,
) -> None:
    """Append a message to an agent's inbox file with file locking.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent receiving the message.
        message (InboxMessage): Message object to append.
        base_dir (Path | None): Override for the base config directory.
    """
    path = ensure_inbox(team_name, agent_name, base_dir)
    lock_path = path.parent / ".lock"

    with file_lock(lock_path):
        raw_list = json.loads(path.read_text())
        raw_list.append(message.model_dump(by_alias=True, exclude_none=True))
        path.write_text(json.dumps(raw_list))


def send_plain_message(
    team_name: str,
    from_name: str,
    to_name: str,
    text: str,
    summary: str,
    color: str | None = None,
    base_dir: Path | None = None,
) -> None:
    """Send a plain text message from one agent to another.

    Args:
        team_name (str): Name of the team.
        from_name (str): Name of the sending agent.
        to_name (str): Name of the receiving agent.
        text (str): Message body.
        summary (str): Brief summary of the message.
        color (str | None): Optional color hint for UI display.
        base_dir (Path | None): Override for the base config directory.
    """
    msg = InboxMessage(
        from_=from_name,
        text=text,
        timestamp=now_iso(),
        read=False,
        summary=summary,
        color=color,
    )
    append_message(team_name, to_name, msg, base_dir)


def send_structured_message(
    team_name: str,
    from_name: str,
    to_name: str,
    payload: BaseModel,
    color: str | None = None,
    base_dir: Path | None = None,
) -> None:
    """Send a structured message containing a Pydantic model as JSON.

    Args:
        team_name (str): Name of the team.
        from_name (str): Name of the sending agent.
        to_name (str): Name of the receiving agent.
        payload (BaseModel): Pydantic model to serialize and send.
        color (str | None): Optional color hint for UI display.
        base_dir (Path | None): Override for the base config directory.
    """
    serialized = payload.model_dump_json(by_alias=True)
    msg = InboxMessage(
        from_=from_name,
        text=serialized,
        timestamp=now_iso(),
        read=False,
        color=color,
    )
    append_message(team_name, to_name, msg, base_dir)


def send_task_assignment(
    team_name: str,
    task: TaskFile,
    assigned_by: str,
    base_dir: Path | None = None,
) -> None:
    """Send a task assignment notification to the task's owner.

    Args:
        team_name (str): Name of the team.
        task (TaskFile): Task object being assigned.
        assigned_by (str): Name of the agent assigning the task.
        base_dir (Path | None): Override for the base config directory.

    Raises:
        ValueError: If the task has no owner assigned.
    """
    if task.owner is None:
        raise ValueError("Cannot send task assignment: task has no owner")
    payload = TaskAssignment(
        task_id=task.id,
        subject=task.subject,
        description=task.description,
        assigned_by=assigned_by,
        timestamp=now_iso(),
    )
    send_structured_message(
        team_name, assigned_by, task.owner, payload, base_dir=base_dir
    )


def send_shutdown_request(
    team_name: str,
    recipient: str,
    reason: str = "",
    base_dir: Path | None = None,
) -> str:
    """Send a shutdown request to an agent from the team lead.

    Args:
        team_name (str): Name of the team.
        recipient (str): Name of the agent to shut down.
        reason (str): Optional reason for the shutdown.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        str: Unique request ID for tracking the shutdown.
    """
    request_id = f"shutdown-{int(time.time() * 1000)}@{recipient}"
    payload = ShutdownRequest(
        request_id=request_id,
        from_="team-lead",
        reason=reason,
        timestamp=now_iso(),
    )
    send_structured_message(
        team_name, "team-lead", recipient, payload, base_dir=base_dir
    )
    return request_id
