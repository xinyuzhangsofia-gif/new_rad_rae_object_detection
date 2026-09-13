"""Persistent queue identity, state, recovery, and atomic writes."""

import hashlib
import fcntl
import json
import os
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from training_utils.experiments.schema import QUEUE_TASK_IDENTITY_VERSION
from training_utils.experiments.tables import resolve_experiment_sheet_path


@contextmanager
def experiment_queue_lock(sheet_path):
    """Allow only one top-level scheduler to claim a table."""
    resolved_path = resolve_experiment_sheet_path(sheet_path)
    lock_path = resolved_path.with_name(f".{resolved_path.name}.queue.lock")
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RuntimeError(
                "Another train.py experiment queue is already using "
                f"{resolved_path}. Do not start a second queue manually."
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        yield lock_path
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _queue_task_slug(task):
    group_name = unicodedata.normalize("NFKD", str(task.experiment.name))
    group_name = "".join(
        character
        if (character.isalnum() or character in {"_", "-"})
        else "_"
        for character in group_name
    ).strip("_") or f"row{task.experiment.row_number}"
    return (
        f"{task.ordinal:03d}_{group_name}_seed{task.experiment.seed}_"
        f"{task.branch}"
    )


def _queue_task_state_key(task):
    experiment = task.experiment
    identity = {
        "identity_version": QUEUE_TASK_IDENTITY_VERSION,
        "group": experiment.name,
        "seed": int(experiment.seed),
        "branch": task.branch,
        "shared": list(experiment.shared_sequences),
        "source": list(experiment.source_sequences),
        "target": list(experiment.target_sequences),
        "test": list(experiment.test_sequences),
        "half": list(experiment.half_selection),
        "shared_parts": list(experiment.shared_parts),
        "source_parts": list(experiment.source_parts),
        "target_parts": list(experiment.target_parts),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"{_queue_task_slug(task)}_{digest}"


def _queue_state_path(sheet_path):
    resolved_path = resolve_experiment_sheet_path(sheet_path)
    return resolved_path.with_name(
        f".{resolved_path.name}.queue_state.json"
    )


def _load_queue_state(sheet_path):
    state_path = _queue_state_path(sheet_path)
    if not state_path.is_file():
        return state_path, {"version": 1, "tasks": {}}
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("tasks"),
        dict,
    ):
        raise ValueError(f"Invalid experiment queue state: {state_path}")
    payload.setdefault("version", 1)
    return state_path, payload


def _write_queue_state(state_path, state):
    state_path = Path(state_path)
    temporary_path = state_path.with_name(
        f".{state_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(state, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, state_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _update_queue_task_state(
        state_path,
        state,
        task,
        status,
        **fields,
    ):
    task_key = _queue_task_state_key(task)
    task_state = {
        "group": task.experiment.name,
        "seed": int(task.experiment.seed),
        "branch": task.branch,
        "status": str(status),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    task_state.update(fields)
    state["tasks"][task_key] = task_state
    _write_queue_state(state_path, state)
    return task_state


def _process_is_alive(process_id):
    try:
        os.kill(int(process_id), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _recover_parallel_queue_tasks(tasks, sheet_path):
    state_path, queue_state = _load_queue_state(sheet_path)
    pending_train = []
    pending_evaluation = []
    for task in tasks:
        task_state = queue_state["tasks"].get(
            _queue_task_state_key(task),
            {},
        )
        status = str(task_state.get("status", "")).strip().lower()
        process_id = task_state.get("pid")
        if status in {"training", "evaluating"} and _process_is_alive(
            process_id
        ):
            raise RuntimeError(
                f"Recorded {status} process pid={process_id} is still active "
                f"for {task.label}. Wait for it or stop it before restarting "
                "the experiment queue."
            )
        checkpoint_root = task_state.get("checkpoint_root")
        checkpoint_path = (
            None
            if checkpoint_root in (None, "")
            else Path(checkpoint_root).expanduser().resolve()
        )
        if checkpoint_path is not None and checkpoint_path.is_dir():
            pending_evaluation.append({
                "task": task,
                "checkpoint_root": checkpoint_path,
            })
            print(
                f"Experiment queue recovery: {task.label} already has "
                f"checkpoint={checkpoint_path}; skip retraining and queue "
                "evaluation.",
                flush=True,
            )
        else:
            pending_train.append(task)
    return state_path, queue_state, pending_train, pending_evaluation


