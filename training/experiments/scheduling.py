"""Experiment expansion, process guards, and deterministic GPU scheduling."""

import os
from pathlib import Path

from training.runtime import parse_gpu_ids
from training.experiments.schema import (
    ExperimentQueueTask,
    VALID_BRANCHES,
)
from training.post_training_evaluation import query_gpu_status


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def normalize_queue_branches(value):
    if isinstance(value, str):
        value = tuple(
            token.strip().lower()
            for token in value.split(",")
            if token.strip() != ""
        )
    else:
        value = tuple(str(token).strip().lower() for token in value)
    invalid = [branch for branch in value if branch not in VALID_BRANCHES]
    if invalid or not value:
        raise ValueError(
            "experiment_queue_branches must contain source and/or target, "
            f"got {value!r}"
        )
    return tuple(dict.fromkeys(value))


def _effective_train_signature(experiment, branch):
    if branch == "source":
        domain_parts = experiment.source_parts
    elif branch == "target":
        domain_parts = experiment.target_parts
    else:
        raise ValueError(f"Unsupported experiment branch: {branch!r}")
    return tuple(
        (int(sequence), str(position))
        for sequence, position in (
            tuple(experiment.shared_parts) + tuple(domain_parts)
        )
    )


def validate_experiment_queue_design(experiments):
    """Reject rows that could accidentally repeat an identical experiment."""
    identities = {}
    definitions = {}
    for experiment in experiments:
        identity = (experiment.name.strip().lower(), int(experiment.seed))
        previous_identity = identities.get(identity)
        if previous_identity is not None:
            raise ValueError(
                "Experiment table contains a duplicate group/seed identity: "
                f"{previous_identity.name!r} and {experiment.name!r}, "
                f"seed={experiment.seed}."
            )
        identities[identity] = experiment

        source_signature = _effective_train_signature(experiment, "source")
        target_signature = _effective_train_signature(experiment, "target")
        if source_signature == target_signature:
            raise ValueError(
                f"{experiment.name} seed={experiment.seed} would use the "
                "same effective training set for source and target: "
                f"{source_signature}. Correct source_seq/target_seq before "
                "starting the queue."
            )

        definition = (
            int(experiment.seed),
            tuple(experiment.shared_parts),
            tuple(experiment.source_parts),
            tuple(experiment.target_parts),
            tuple(experiment.test_parts),
        )
        previous_definition = definitions.get(definition)
        if previous_definition is not None:
            raise ValueError(
                "Experiment table repeats the same seed/split definition in "
                f"{previous_definition.name!r} and {experiment.name!r}."
            )
        definitions[definition] = experiment
    return True


def build_experiment_queue_tasks(experiments, branches, skip_completed):
    tasks = []
    ordinal = 0
    for experiment in experiments:
        for branch in branches:
            ordinal += 1
            if skip_completed and experiment.branch_complete(branch):
                print(
                    f"Experiment queue [{ordinal}/"
                    f"{len(experiments) * len(branches)}]: skip "
                    f"{experiment.name} {branch}; metrics already exist.",
                    flush=True,
                )
                continue
            tasks.append(ExperimentQueueTask(
                ordinal=ordinal,
                experiment=experiment,
                branch=branch,
            ))
    return tasks


def normalize_parallel_gpu_slots(value, worker_count, fallback_gpu_ids):
    worker_count = int(worker_count)
    if worker_count <= 0:
        raise ValueError("experiment_queue_train_workers must be positive.")
    if value in (None, "", ()):
        slots = tuple(str(fallback_gpu_ids) for _ in range(worker_count))
    elif isinstance(value, str):
        slots = (value,)
    else:
        slots = tuple(str(slot) for slot in value)
    if len(slots) < worker_count:
        raise ValueError(
            "experiment_queue_train_gpu_slots must provide at least one "
            f"GPU-id string per worker; workers={worker_count}, slots={slots}."
        )
    slots = slots[:worker_count]
    for slot in slots:
        gpu_ids = parse_gpu_ids(slot)
        if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
            raise ValueError(
                f"Invalid experiment queue training GPU slot: {slot!r}"
            )
    return slots


def validate_parallel_gpu_strategy(strategy, slots):
    strategy = str(strategy or "shared_dynamic").strip().lower()
    if strategy not in {"shared_dynamic", "isolated"}:
        raise ValueError(
            "experiment_queue_gpu_strategy must be 'shared_dynamic' or "
            f"'isolated', got {strategy!r}."
        )
    if strategy == "isolated":
        seen = set()
        for slot in slots:
            current = set(parse_gpu_ids(slot))
            overlap = seen.intersection(current)
            if overlap:
                raise ValueError(
                    "Isolated experiment GPU slots overlap on "
                    f"{sorted(overlap)}: {slots}."
                )
            seen.update(current)
    return strategy


def find_other_top_level_train_processes():
    """Return existing top-level train.py PIDs from this workspace."""
    process_root = Path("/proc")
    if not process_root.is_dir():
        return ()
    target_script = (PROJECT_ROOT / "train.py").resolve()
    matches = []
    for process_dir in process_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        process_id = int(process_dir.name)
        if process_id == os.getpid():
            continue
        try:
            command_tokens = [
                token.decode("utf-8", errors="replace")
                for token in (process_dir / "cmdline").read_bytes().split(b"\0")
                if token
            ]
            process_cwd = (process_dir / "cwd").resolve()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if len(command_tokens) < 2:
            continue
        script_index = 1
        while (
            script_index < len(command_tokens)
            and command_tokens[script_index].startswith("-")
        ):
            option = command_tokens[script_index]
            if option in {"-c", "-m"}:
                script_index = len(command_tokens)
                break
            script_index += 1
        if script_index >= len(command_tokens):
            continue
        candidate = Path(command_tokens[script_index])
        if candidate.name != "train.py":
            continue
        if not candidate.is_absolute():
            candidate = process_cwd / candidate
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate == target_script:
            matches.append(process_id)
    return tuple(sorted(matches))


def ensure_no_other_top_level_train_process():
    process_ids = find_other_top_level_train_processes()
    if process_ids:
        raise RuntimeError(
            "Another top-level MVRSS train.py is already running "
            f"(pid={','.join(str(pid) for pid in process_ids)}). Wait for it "
            "to finish before starting the table scheduler; its child workers "
            "will be launched automatically."
        )
    return True


def select_parallel_evaluation_gpu(
        candidate_gpu_ids,
        active_gpu_counts,
        min_free_memory_mb,
        reservation_memory_mb,
        gpu_status=None,
        max_active_per_gpu=None,
    ):
    """Choose an evaluation GPU while reserving room for active evaluations."""
    status_rows = query_gpu_status() if gpu_status is None else gpu_status
    status_by_id = {
        int(status["index"]): status
        for status in status_rows
    }
    missing = [
        int(gpu_id)
        for gpu_id in candidate_gpu_ids
        if int(gpu_id) not in status_by_id
    ]
    if missing:
        raise RuntimeError(
            f"nvidia-smi did not report evaluation GPU ids {missing}."
        )

    ranked = []
    for gpu_id in candidate_gpu_ids:
        gpu_id = int(gpu_id)
        status = status_by_id[gpu_id]
        active_count = int(active_gpu_counts.get(gpu_id, 0))
        if (
            max_active_per_gpu is not None
            and active_count >= int(max_active_per_gpu)
        ):
            continue
        effective_free = (
            int(status["free_memory_mb"])
            - active_count * int(reservation_memory_mb)
        )
        if effective_free < int(min_free_memory_mb):
            continue
        ranked.append((
            active_count,
            -effective_free,
            int(status.get("utilization_percent", 0)),
            gpu_id,
            status,
            effective_free,
        ))
    if not ranked:
        return None
    _, _, _, gpu_id, status, effective_free = min(ranked)
    selected = dict(status)
    selected["index"] = gpu_id
    selected["effective_free_memory_mb"] = effective_free
    return selected
