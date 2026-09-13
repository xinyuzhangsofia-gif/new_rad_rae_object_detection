"""Atomic analysis state persistence, launcher locks, and runtime status."""

from datetime import datetime
import fcntl
import json
import os
from pathlib import Path


def now_text():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path, payload):
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def acquire_output_lock(output_dir, lock_name, conflict_message, timestamp=None):
    """Acquire the same nonblocking process-lifetime lock used by launchers."""
    lock_path = Path(output_dir) / str(lock_name)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(str(conflict_message).format(lock_path=lock_path)) from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    started_at = (
        now_text()
        if timestamp is None
        else timestamp() if callable(timestamp) else timestamp
    )
    lock_handle.write(
        f"pid={os.getpid()} started_at={started_at}\n"
    )
    lock_handle.flush()
    return lock_handle


def task_process_is_alive(task, require_report_path=False):
    """Confirm that a recorded PID still belongs to its evaluation command."""
    try:
        pid = int(task.get("pid"))
    except (TypeError, ValueError):
        return False
    command_path = Path(f"/proc/{pid}/cmdline")
    try:
        command = command_path.read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return False
    if (
        "evaluation.py" not in command
        or str(task.get("checkpoint_root", "")) not in command
    ):
        return False
    return (
        not require_report_path
        or str(task.get("report_path", "")) in command
    )


def initialize_runtime_task(discovered, previous):
    """Return the common pending runtime fields without deciding completion."""
    task = dict(discovered)
    task.update({
        "attempts": int(previous.get("attempts", 0)),
        "metrics": previous.get("metrics"),
        "status": "pending",
        "pid": None,
        "gpu": None,
        "started_at": previous.get("started_at"),
        "finished_at": previous.get("finished_at"),
        "returncode": previous.get("returncode"),
        "error": None,
    })
    return task


def state_counts(state):
    counts = {
        status: 0
        for status in ("pending", "running", "completed", "failed")
    }
    for task in state["tasks"].values():
        status = task.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts
