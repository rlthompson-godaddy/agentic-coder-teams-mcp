import json
from collections import deque
from pathlib import Path
from typing import Literal

from claude_teams.filelock import file_lock
from claude_teams.models import TaskFile
from claude_teams.teams import team_exists


_TaskStatus = Literal["pending", "in_progress", "completed", "deleted"]

TASKS_DIR = Path.home() / ".claude" / "tasks"


def _tasks_dir(base_dir: Path | None = None) -> Path:
    return (base_dir / "tasks") if base_dir else TASKS_DIR


_STATUS_ORDER = {"pending": 0, "in_progress": 1, "completed": 2}


def _flush_pending_writes(pending_writes: dict[Path, TaskFile]) -> None:
    for task_file, task_obj in pending_writes.items():
        task_file.write_text(
            json.dumps(task_obj.model_dump(by_alias=True, exclude_none=True))
        )


def _would_create_cycle(
    team_dir: Path, from_id: str, to_id: str, pending_edges: dict[str, set[str]]
) -> bool:
    """True if making from_id blocked_by to_id creates a cycle.

    BFS from to_id through blocked_by chains (on-disk + pending);
    cycle if it reaches from_id.

    Args:
        team_dir (Path): Directory containing task JSON files.
        from_id (str): ID of the task that would be blocked.
        to_id (str): ID of the task that would block from_id.
        pending_edges (dict[str, set[str]]): In-memory edges not yet written.

    Returns:
        bool: True if adding the edge would create a circular dependency.
    """
    visited: set[str] = set()
    queue = deque([to_id])
    while queue:
        current = queue.popleft()
        if current == from_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        fpath = team_dir / f"{current}.json"
        if fpath.exists():
            task = TaskFile(**json.loads(fpath.read_text()))
            queue.extend(dep_id for dep_id in task.blocked_by if dep_id not in visited)
        queue.extend(
            dep_id
            for dep_id in pending_edges.get(current, set())
            if dep_id not in visited
        )
    return False


def next_task_id(team_name: str, base_dir: Path | None = None) -> str:
    """Find the next available integer task ID for a team.

    Args:
        team_name (str): Name of the team.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        str: Next available task ID as a string (e.g., "1", "42").
    """
    team_dir = _tasks_dir(base_dir) / team_name
    ids: list[int] = []
    for task_file in team_dir.glob("*.json"):
        try:
            ids.append(int(task_file.stem))
        except ValueError:
            continue
    return str(max(ids) + 1) if ids else "1"


def create_task(
    team_name: str,
    subject: str,
    description: str,
    active_form: str = "",
    metadata: dict | None = None,
    base_dir: Path | None = None,
) -> TaskFile:
    """Create a new task file on disk for the given team.

    Args:
        team_name (str): Name of the team that owns the task.
        subject (str): Brief title for the task.
        description (str): Detailed description of what needs to be done.
        active_form (str): Present-tense form displayed during progress.
        metadata (dict | None): Arbitrary key-value metadata.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TaskFile: The newly created task with its assigned ID.

    Raises:
        ValueError: If subject is empty or team does not exist.
    """
    if not subject or not subject.strip():
        raise ValueError("Task subject must not be empty")
    if not team_exists(team_name, base_dir):
        raise ValueError(f"Team {team_name!r} does not exist")
    team_dir = _tasks_dir(base_dir) / team_name
    team_dir.mkdir(parents=True, exist_ok=True)
    lock_path = team_dir / ".lock"

    with file_lock(lock_path):
        task_id = next_task_id(team_name, base_dir)
        task = TaskFile(
            id=task_id,
            subject=subject,
            description=description,
            active_form=active_form,
            status="pending",
            metadata=metadata,
        )
        fpath = team_dir / f"{task_id}.json"
        fpath.write_text(json.dumps(task.model_dump(by_alias=True, exclude_none=True)))

    return task


def get_task(team_name: str, task_id: str, base_dir: Path | None = None) -> TaskFile:
    """Read a single task by ID from disk.

    Args:
        team_name (str): Name of the team.
        task_id (str): ID of the task to retrieve.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TaskFile: The requested task object.

    Raises:
        FileNotFoundError: If the task file does not exist.
        json.JSONDecodeError: If the task file is malformed.
    """
    team_dir = _tasks_dir(base_dir) / team_name
    fpath = team_dir / f"{task_id}.json"
    raw = json.loads(fpath.read_text())
    return TaskFile(**raw)


def _link_dependency(
    task: TaskFile,
    task_id: str,
    dep_ids: list[str],
    forward_field: str,
    inverse_field: str,
    team_dir: Path,
    pending_writes: dict[Path, TaskFile],
) -> None:
    """Add dependency links between tasks in both directions.

    For each *dep_id* in *dep_ids*, appends it to ``task.{forward_field}``
    and ensures ``task_id`` is added to the other task's ``{inverse_field}``.

    Args:
        task (TaskFile): The task being updated.
        task_id (str): ID of the task being updated.
        dep_ids (list[str]): IDs of tasks to link.
        forward_field (str): Attribute on *task* (``"blocks"`` or ``"blocked_by"``).
        inverse_field (str): Attribute on the other task.
        team_dir (Path): Directory containing task JSON files.
        pending_writes (dict[Path, TaskFile]): Accumulator for batched writes.
    """
    forward_list: list[str] = getattr(task, forward_field)
    existing = set(forward_list)
    for dep_id in dep_ids:
        if dep_id not in existing:
            forward_list.append(dep_id)
            existing.add(dep_id)
        dep_path = team_dir / f"{dep_id}.json"
        if dep_path in pending_writes:
            other = pending_writes[dep_path]
        else:
            other = TaskFile(**json.loads(dep_path.read_text()))
        inverse_list: list[str] = getattr(other, inverse_field)
        if task_id not in inverse_list:
            inverse_list.append(task_id)
        pending_writes[dep_path] = other


def _remove_task_references(
    task_id: str,
    team_dir: Path,
    pending_writes: dict[Path, TaskFile],
    fields: tuple[str, ...] = ("blocked_by",),
) -> None:
    """Remove *task_id* from the specified fields across all sibling tasks.

    Iterates every task file in *team_dir* (skipping *task_id* itself) and
    removes *task_id* from each named list field.

    Args:
        task_id (str): ID to remove from other tasks' dependency lists.
        team_dir (Path): Directory containing task JSON files.
        pending_writes (dict[Path, TaskFile]): Accumulator for batched writes.
        fields (tuple[str, ...]): Attribute names to clean
            (e.g. ``("blocked_by",)`` or ``("blocked_by", "blocks")``).
    """
    for task_file in team_dir.glob("*.json"):
        try:
            int(task_file.stem)
        except ValueError:
            continue
        if task_file.stem == task_id:
            continue
        if task_file in pending_writes:
            other = pending_writes[task_file]
        else:
            other = TaskFile(**json.loads(task_file.read_text()))
        changed = False
        for field in fields:
            dep_list: list[str] = getattr(other, field)
            if task_id in dep_list:
                dep_list.remove(task_id)
                changed = True
        if changed:
            pending_writes[task_file] = other


def update_task(
    team_name: str,
    task_id: str,
    *,
    status: _TaskStatus | None = None,
    owner: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
    metadata: dict | None = None,
    base_dir: Path | None = None,
) -> TaskFile:
    """Update a task with validation and automatic dependency graph updates.

    Args:
        team_name (str): Name of the team.
        task_id (str): ID of the task to update.
        status (_TaskStatus | None): New status ("pending", "in_progress", "completed", "deleted").
        owner (str | None): New owner agent name.
        subject (str | None): New subject line.
        description (str | None): New description text.
        active_form (str | None): New active form text.
        add_blocks (list[str] | None): Task IDs that this task should block.
        add_blocked_by (list[str] | None): Task IDs that should block this task.
        metadata (dict | None): Metadata keys to merge (set key to None to delete).
        base_dir (Path | None): Override for the base config directory.

    Returns:
        TaskFile: The updated task object.

    Raises:
        ValueError: If status transition is invalid, circular dependency detected,
            or blocked tasks are incomplete.
        FileNotFoundError: If the task or referenced tasks do not exist.
    """
    team_dir = _tasks_dir(base_dir) / team_name
    lock_path = team_dir / ".lock"
    fpath = team_dir / f"{task_id}.json"

    with file_lock(lock_path):
        # --- Phase 1: Read ---
        task = TaskFile(**json.loads(fpath.read_text()))

        # --- Phase 2: Validate (no disk writes) ---
        pending_edges: dict[str, set[str]] = {}

        if add_blocks:
            for blocked_id in add_blocks:
                if blocked_id == task_id:
                    raise ValueError(f"Task {task_id} cannot block itself")
                if not (team_dir / f"{blocked_id}.json").exists():
                    raise ValueError(f"Referenced task {blocked_id!r} does not exist")
            for blocked_id in add_blocks:
                pending_edges.setdefault(blocked_id, set()).add(task_id)

        if add_blocked_by:
            for blocked_id in add_blocked_by:
                if blocked_id == task_id:
                    raise ValueError(f"Task {task_id} cannot be blocked by itself")
                if not (team_dir / f"{blocked_id}.json").exists():
                    raise ValueError(f"Referenced task {blocked_id!r} does not exist")
            for blocked_id in add_blocked_by:
                pending_edges.setdefault(task_id, set()).add(blocked_id)

        if add_blocks:
            for blocked_id in add_blocks:
                if _would_create_cycle(team_dir, blocked_id, task_id, pending_edges):
                    raise ValueError(
                        f"Adding block {task_id} -> {blocked_id} would create a circular dependency"
                    )

        if add_blocked_by:
            for blocked_id in add_blocked_by:
                if _would_create_cycle(team_dir, task_id, blocked_id, pending_edges):
                    raise ValueError(
                        f"Adding dependency {task_id} blocked_by {blocked_id} would create a circular dependency"
                    )

        if status is not None and status != "deleted":
            cur_order = _STATUS_ORDER[task.status]
            new_order = _STATUS_ORDER.get(status)
            if new_order is None:
                raise ValueError(f"Invalid status: {status!r}")
            if new_order < cur_order:
                raise ValueError(
                    f"Cannot transition from {task.status!r} to {status!r}"
                )
            effective_blocked_by = set(task.blocked_by)
            if add_blocked_by:
                effective_blocked_by.update(add_blocked_by)
            if status in ("in_progress", "completed") and effective_blocked_by:
                for blocker_id in effective_blocked_by:
                    blocker_path = team_dir / f"{blocker_id}.json"
                    if blocker_path.exists():
                        blocker = TaskFile(**json.loads(blocker_path.read_text()))
                        if blocker.status != "completed":
                            raise ValueError(
                                f"Cannot set status to {status!r}: "
                                f"blocked by task {blocker_id} (status: {blocker.status!r})"
                            )

        # --- Phase 3: Mutate (in-memory only) ---
        pending_writes: dict[Path, TaskFile] = {}

        if subject is not None:
            task.subject = subject
        if description is not None:
            task.description = description
        if active_form is not None:
            task.active_form = active_form
        if owner is not None:
            task.owner = owner

        if add_blocks:
            _link_dependency(
                task,
                task_id,
                add_blocks,
                "blocks",
                "blocked_by",
                team_dir,
                pending_writes,
            )

        if add_blocked_by:
            _link_dependency(
                task,
                task_id,
                add_blocked_by,
                "blocked_by",
                "blocks",
                team_dir,
                pending_writes,
            )

        if metadata is not None:
            current = task.metadata or {}
            for key, value in metadata.items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = value
            task.metadata = current if current else None

        if status is not None and status != "deleted":
            task.status = status
            if status == "completed":
                _remove_task_references(
                    task_id, team_dir, pending_writes, ("blocked_by",)
                )

        if status == "deleted":
            task.status = "deleted"
            _remove_task_references(
                task_id, team_dir, pending_writes, ("blocked_by", "blocks")
            )

        # --- Phase 4: Write ---
        if status == "deleted":
            _flush_pending_writes(pending_writes)
            fpath.unlink()
        else:
            fpath.write_text(
                json.dumps(task.model_dump(by_alias=True, exclude_none=True))
            )
            _flush_pending_writes(pending_writes)

    return task


def list_tasks(team_name: str, base_dir: Path | None = None) -> list[TaskFile]:
    """List all tasks for a team, sorted by task ID.

    Args:
        team_name (str): Name of the team.
        base_dir (Path | None): Override for the base config directory.

    Returns:
        list[TaskFile]: All task objects sorted by integer ID.

    Raises:
        ValueError: If the team does not exist.
    """
    if not team_exists(team_name, base_dir):
        raise ValueError(f"Team {team_name!r} does not exist")
    team_dir = _tasks_dir(base_dir) / team_name
    tasks: list[TaskFile] = []
    for task_file in team_dir.glob("*.json"):
        try:
            int(task_file.stem)
        except ValueError:
            continue
        tasks.append(TaskFile(**json.loads(task_file.read_text())))
    tasks.sort(key=lambda task: int(task.id))
    return tasks


def reset_owner_tasks(
    team_name: str, agent_name: str, base_dir: Path | None = None
) -> None:
    """Reset all non-completed tasks owned by an agent to pending with no owner.

    Args:
        team_name (str): Name of the team.
        agent_name (str): Name of the agent whose tasks should be reset.
        base_dir (Path | None): Override for the base config directory.
    """
    team_dir = _tasks_dir(base_dir) / team_name
    lock_path = team_dir / ".lock"

    with file_lock(lock_path):
        for task_file in team_dir.glob("*.json"):
            try:
                int(task_file.stem)
            except ValueError:
                continue
            task = TaskFile(**json.loads(task_file.read_text()))
            if task.owner == agent_name:
                if task.status != "completed":
                    task.status = "pending"
                task.owner = None
                task_file.write_text(
                    json.dumps(task.model_dump(by_alias=True, exclude_none=True))
                )
