import fcntl
from contextlib import contextmanager
from pathlib import Path


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
