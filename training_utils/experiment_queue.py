"""Compatibility facade and high-level domain-shift queue orchestration.

Responsibility-specific implementations live in training_utils.experiments.
Historical imports from this module remain available.
"""

import copy
import os
import re
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from training_utils.experiments.execution import (
    _close_process_log,
    _launch_parallel_evaluation_task,
    _launch_parallel_training_task,
    _print_experiment_task_start,
    _read_training_job_result,
    _record_experiment_result,
    _refresh_completed_weather_summaries,
    _run_sequential_experiment_queue,
    _terminate_running_processes,
    _write_training_job_config,
    build_experiment_training_config,
)
from training_utils.experiments.scheduling import (
    _effective_train_signature,
    build_experiment_queue_tasks,
    ensure_no_other_top_level_train_process,
    find_other_top_level_train_processes,
    normalize_parallel_gpu_slots,
    normalize_queue_branches,
    select_parallel_evaluation_gpu as _select_parallel_evaluation_gpu,
    validate_experiment_queue_design,
    validate_parallel_gpu_strategy,
)
from training_utils.experiments.schema import (
    DomainShiftExperiment,
    ExperimentQueueTask,
    QUEUE_TASK_IDENTITY_VERSION,
    VALID_BRANCHES,
)
from training_utils.experiments.state import (
    _load_queue_state,
    _process_is_alive,
    _queue_state_path,
    _queue_task_slug,
    _queue_task_state_key,
    _recover_parallel_queue_tasks,
    _update_queue_task_state,
    _write_queue_state,
    experiment_queue_lock,
)
from training_utils.experiments.tables import (
    _branch_half_selection,
    _display_width,
    _ensure_header,
    _experiment_sort_key,
    _find_column,
    _float_cell,
    _format_txt_experiment_matrix,
    _half_selection_for_parts,
    _header_index,
    _merge_half_selections,
    _metric_is_present,
    _normalize_sequence_cell,
    _normalized_header,
    _optional_column,
    _pad_display,
    _parse_seed,
    _read_experiment_matrix,
    _read_txt_experiment_matrix,
    _report_matches_experiment,
    _same_sequences,
    _table_row_sort_key,
    _write_experiment_sheet,
    find_fresh_experiment_result,
    load_domain_shift_experiments,
    parse_sequence_cell,
    parse_sequence_cell_parts,
    resolve_experiment_sheet_path,
    resolve_experiment_sheet_paths,
    update_experiment_sheet_result,
    validate_experiment_sheet_is_full,
)
from training_utils.post_training_evaluation import (
    prepare_post_training_evaluation_launch,
    query_gpu_status,
    resolve_candidate_physical_gpu_ids,
)
from training_utils.runtime import parse_gpu_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def select_parallel_evaluation_gpu(
        candidate_gpu_ids,
        active_gpu_counts,
        min_free_memory_mb,
        reservation_memory_mb,
        gpu_status=None,
        max_active_per_gpu=None,
    ):
    """Compatibility wrapper preserving facade-level GPU-status patching."""
    if gpu_status is None:
        gpu_status = query_gpu_status()
    return _select_parallel_evaluation_gpu(
        candidate_gpu_ids=candidate_gpu_ids,
        active_gpu_counts=active_gpu_counts,
        min_free_memory_mb=min_free_memory_mb,
        reservation_memory_mb=reservation_memory_mb,
        gpu_status=gpu_status,
        max_active_per_gpu=max_active_per_gpu,
    )


def _run_parallel_experiment_queue(
        base_config,
        tasks,
        total_steps,
        sheet_path,
        results_base_dir,
        update_sheet_results,
    ):
    train_worker_count = int(
        base_config.get("experiment_queue_train_workers", 2)
    )
    evaluation_worker_count = int(
        base_config.get("experiment_queue_eval_workers", 2)
    )
    if train_worker_count <= 0 or evaluation_worker_count <= 0:
        raise ValueError(
            "experiment_queue_train_workers and "
            "experiment_queue_eval_workers must both be positive."
        )

    train_gpu_slots = normalize_parallel_gpu_slots(
        base_config.get("experiment_queue_train_gpu_slots"),
        worker_count=train_worker_count,
        fallback_gpu_ids=base_config.get("gpu_ids", "0"),
    )
    gpu_strategy = validate_parallel_gpu_strategy(
        base_config.get(
            "experiment_queue_gpu_strategy",
            "shared_dynamic",
        ),
        train_gpu_slots,
    )
    evaluation_gpu_pool_text = str(
        base_config.get(
            "experiment_queue_eval_gpu_pool",
            base_config.get("gpu_ids", "0"),
        )
    )
    evaluation_gpu_ids = tuple(resolve_candidate_physical_gpu_ids(
        evaluation_gpu_pool_text,
    ))
    min_free_memory_mb = int(
        base_config.get(
            "experiment_queue_eval_min_free_memory_mb",
            base_config.get("post_training_eval_min_free_memory_mb", 4096),
        )
    )
    reservation_memory_mb = int(
        base_config.get(
            "experiment_queue_eval_reservation_memory_mb",
            min_free_memory_mb,
        )
    )
    poll_seconds = max(
        0.1,
        float(base_config.get("experiment_queue_poll_seconds", 1.0)),
    )

    log_base_dir = Path(
        str(base_config.get("log_base_dir", "runs"))
    ).expanduser()
    if not log_base_dir.is_absolute():
        log_base_dir = PROJECT_ROOT / log_base_dir
    session_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_pid{os.getpid()}"
    )
    session_dir = log_base_dir / "experiment_queue" / session_name
    session_dir.mkdir(parents=True, exist_ok=False)

    print(
        "Parallel experiment queue: "
        f"train_workers={train_worker_count}, "
        f"eval_workers={evaluation_worker_count}, "
        f"strategy={gpu_strategy}, train_gpu_slots={train_gpu_slots}, "
        f"eval_gpu_pool={evaluation_gpu_ids}, logs={session_dir}",
        flush=True,
    )

    (
        queue_state_path,
        queue_state,
        pending_train,
        pending_evaluation,
    ) = _recover_parallel_queue_tasks(
        tasks=tasks,
        sheet_path=sheet_path,
    )
    available_train_slots = list(range(train_worker_count))
    active_train = {}
    active_evaluation = {}
    active_evaluation_gpu_counts = {
        int(gpu_id): 0
        for gpu_id in evaluation_gpu_ids
    }
    completed = []
    completed_report_paths = []
    last_memory_wait_message = 0.0

    try:
        while (
            pending_train
            or pending_evaluation
            or active_train
            or active_evaluation
        ):
            for process_id, state in list(active_train.items()):
                return_code = state["process"].poll()
                if return_code is None:
                    continue
                _close_process_log(state)
                active_train.pop(process_id)
                available_train_slots.append(state["slot_index"])
                available_train_slots.sort()
                if return_code != 0:
                    _update_queue_task_state(
                        queue_state_path,
                        queue_state,
                        state["task"],
                        "training_failed",
                        pid=state["process"].pid,
                        log_path=str(state["log_path"]),
                        return_code=int(return_code),
                    )
                    raise RuntimeError(
                        f"Training task {state['task'].label} failed with "
                        f"exit code {return_code}. See {state['log_path']}."
                    )
                checkpoint_root = _read_training_job_result(
                    state["result_path"]
                )
                _update_queue_task_state(
                    queue_state_path,
                    queue_state,
                    state["task"],
                    "trained",
                    checkpoint_root=str(checkpoint_root),
                    train_log_path=str(state["log_path"]),
                )
                pending_evaluation.append({
                    "task": state["task"],
                    "checkpoint_root": checkpoint_root,
                })
                print(
                    f"Experiment queue: training completed for "
                    f"{state['task'].label}; evaluation queued; "
                    f"checkpoint={checkpoint_root}",
                    flush=True,
                )

            for process_id, state in list(active_evaluation.items()):
                return_code = state["process"].poll()
                if return_code is None:
                    continue
                _close_process_log(state)
                active_evaluation.pop(process_id)
                gpu_id = state["physical_gpu_id"]
                active_evaluation_gpu_counts[gpu_id] -= 1
                if return_code != 0:
                    _update_queue_task_state(
                        queue_state_path,
                        queue_state,
                        state["task"],
                        "evaluation_failed",
                        pid=state["process"].pid,
                        checkpoint_root=str(state["checkpoint_root"]),
                        log_path=str(state["log_path"]),
                        return_code=int(return_code),
                    )
                    raise RuntimeError(
                        f"Evaluation task {state['task'].label} failed with "
                        f"exit code {return_code}. See {state['log_path']}."
                    )
                report_path, report = _record_experiment_result(
                    sheet_path=sheet_path,
                    results_base_dir=results_base_dir,
                    task=state["task"],
                    evaluation_started_at=state["started_at"],
                    update_sheet_results=update_sheet_results,
                )
                _update_queue_task_state(
                    queue_state_path,
                    queue_state,
                    state["task"],
                    "completed",
                    checkpoint_root=str(state["checkpoint_root"]),
                    evaluation_log_path=str(state["log_path"]),
                    report_path=str(report_path),
                    bev_ap=float(report["bev_ap"]),
                    threed_ap=float(report["threed_ap"]),
                )
                completed_report_paths.append(report_path)
                completed.append((
                    state["task"].ordinal,
                    state["task"].experiment.name,
                    state["task"].branch,
                ))
                print(
                    f"Experiment queue [{state['task'].ordinal}/"
                    f"{total_steps}]: completed {state['task'].label}.",
                    flush=True,
                )

            evaluation_waiting_for_memory = False
            while (
                pending_evaluation
                and len(active_evaluation) < evaluation_worker_count
            ):
                selected_gpu = select_parallel_evaluation_gpu(
                    candidate_gpu_ids=evaluation_gpu_ids,
                    active_gpu_counts=active_evaluation_gpu_counts,
                    min_free_memory_mb=min_free_memory_mb,
                    reservation_memory_mb=reservation_memory_mb,
                )
                if selected_gpu is None:
                    evaluation_waiting_for_memory = True
                    now = time.monotonic()
                    if now - last_memory_wait_message >= 30.0:
                        print(
                            "Experiment queue: evaluation is waiting for "
                            f"{min_free_memory_mb} MiB free GPU memory; "
                            "new training launches are temporarily paused.",
                            flush=True,
                        )
                        last_memory_wait_message = now
                    break
                pending_state = pending_evaluation.pop(0)
                evaluation_state = _launch_parallel_evaluation_task(
                    pending_state=pending_state,
                    physical_gpu_id=selected_gpu["index"],
                    session_dir=session_dir,
                )
                process_id = evaluation_state["process"].pid
                active_evaluation[process_id] = evaluation_state
                active_evaluation_gpu_counts[
                    evaluation_state["physical_gpu_id"]
                ] += 1
                _update_queue_task_state(
                    queue_state_path,
                    queue_state,
                    evaluation_state["task"],
                    "evaluating",
                    pid=evaluation_state["process"].pid,
                    checkpoint_root=str(
                        evaluation_state["checkpoint_root"]
                    ),
                    physical_gpu_id=int(
                        evaluation_state["physical_gpu_id"]
                    ),
                    log_path=str(evaluation_state["log_path"]),
                )
                print(
                    f"Experiment queue: evaluation started for "
                    f"{evaluation_state['task'].label} on physical "
                    f"cuda:{evaluation_state['physical_gpu_id']} "
                    f"({selected_gpu['free_memory_mb']} MiB reported free, "
                    f"log={evaluation_state['log_path']}).",
                    flush=True,
                )

            while (
                pending_train
                and available_train_slots
                and not evaluation_waiting_for_memory
            ):
                slot_index = available_train_slots.pop(0)
                task = pending_train.pop(0)
                gpu_slot = train_gpu_slots[slot_index]
                _print_experiment_task_start(
                    task,
                    total_steps,
                    gpu_slot=gpu_slot,
                )
                train_state = _launch_parallel_training_task(
                    base_config=base_config,
                    task=task,
                    gpu_slot=gpu_slot,
                    session_dir=session_dir,
                )
                train_state["slot_index"] = slot_index
                active_train[train_state["process"].pid] = train_state
                _update_queue_task_state(
                    queue_state_path,
                    queue_state,
                    task,
                    "training",
                    pid=train_state["process"].pid,
                    gpu_ids=str(gpu_slot),
                    log_path=str(train_state["log_path"]),
                )
                print(
                    f"Experiment queue: training subprocess pid="
                    f"{train_state['process'].pid}, "
                    f"log={train_state['log_path']}",
                    flush=True,
                )

            if (
                pending_evaluation
                and not active_train
                and not active_evaluation
                and evaluation_waiting_for_memory
            ):
                time.sleep(poll_seconds)
                continue
            if active_train or active_evaluation:
                time.sleep(poll_seconds)

    except BaseException:
        _terminate_running_processes(active_train, active_evaluation)
        raise
    finally:
        for state in active_train.values():
            _close_process_log(state)
        for state in active_evaluation.values():
            _close_process_log(state)

    _refresh_completed_weather_summaries(
        results_base_dir=results_base_dir,
        report_paths=completed_report_paths,
    )
    return [
        (name, branch)
        for _ordinal, name, branch in sorted(completed)
    ]


def _experiment_sheet_runtime_tag(sheet_path):
    stem = Path(sheet_path).stem
    if stem.endswith("_experiments"):
        stem = stem[:-len("_experiments")]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "weather"


def _run_seed_two_phase_experiment_queue(
        base_config,
        seed,
        table_batches,
        results_base_dir,
        update_sheet_results,
    ):
    """Train every task for one seed, then evaluate all trained checkpoints."""
    train_worker_count = int(
        base_config.get("experiment_queue_train_workers", 3)
    )
    evaluation_worker_count = int(
        base_config.get("experiment_queue_eval_workers", 18)
    )
    if train_worker_count <= 0 or evaluation_worker_count <= 0:
        raise ValueError(
            "experiment_queue_train_workers and "
            "experiment_queue_eval_workers must both be positive."
        )

    train_gpu_slots = normalize_parallel_gpu_slots(
        base_config.get("experiment_queue_train_gpu_slots"),
        worker_count=train_worker_count,
        fallback_gpu_ids=base_config.get("gpu_ids", "0"),
    )
    gpu_strategy = validate_parallel_gpu_strategy(
        base_config.get("experiment_queue_gpu_strategy", "isolated"),
        train_gpu_slots,
    )
    evaluation_gpu_ids = tuple(resolve_candidate_physical_gpu_ids(str(
        base_config.get(
            "experiment_queue_eval_gpu_pool",
            base_config.get("gpu_ids", "0"),
        )
    )))
    max_evaluations_per_gpu = int(
        base_config.get("experiment_queue_eval_max_per_gpu", 6)
    )
    if max_evaluations_per_gpu <= 0:
        raise ValueError(
            "experiment_queue_eval_max_per_gpu must be positive."
        )
    if evaluation_worker_count > (
        len(evaluation_gpu_ids) * max_evaluations_per_gpu
    ):
        raise ValueError(
            "experiment_queue_eval_workers exceeds the configured per-GPU "
            "capacity: "
            f"workers={evaluation_worker_count}, gpus={evaluation_gpu_ids}, "
            f"max_per_gpu={max_evaluations_per_gpu}."
        )
    evaluation_batch_size = int(
        base_config.get("experiment_queue_eval_batch_size", 8)
    )
    if evaluation_batch_size <= 0:
        raise ValueError(
            "experiment_queue_eval_batch_size must be positive."
        )
    min_free_memory_mb = int(
        base_config.get(
            "experiment_queue_eval_min_free_memory_mb",
            1500,
        )
    )
    reservation_memory_mb = int(
        base_config.get(
            "experiment_queue_eval_reservation_memory_mb",
            2500,
        )
    )
    poll_seconds = max(
        0.1,
        float(base_config.get("experiment_queue_poll_seconds", 1.0)),
    )

    log_base_dir = Path(
        str(base_config.get("log_base_dir", "runs"))
    ).expanduser()
    if not log_base_dir.is_absolute():
        log_base_dir = PROJECT_ROOT / log_base_dir
    session_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_seed{int(seed)}_pid{os.getpid()}"
    )
    session_dir = log_base_dir / "experiment_queue" / session_name
    session_dir.mkdir(parents=True, exist_ok=False)

    pending_train = []
    pending_evaluation = []
    total_seed_tasks = 0
    for table_index, sheet_path, total_steps, tasks in table_batches:
        table_config = copy.deepcopy(dict(base_config))
        table_config["experiment_sheet_path"] = str(sheet_path)
        (
            queue_state_path,
            queue_state,
            table_pending_train,
            table_pending_evaluation,
        ) = _recover_parallel_queue_tasks(
            tasks=tasks,
            sheet_path=sheet_path,
        )
        sheet_tag = _experiment_sheet_runtime_tag(sheet_path)

        def make_context(task):
            return {
                "task": task,
                "sheet_path": sheet_path,
                "table_index": int(table_index),
                "total_steps": int(total_steps),
                "table_config": table_config,
                "queue_state_path": queue_state_path,
                "queue_state": queue_state,
                "runtime_slug": (
                    f"{sheet_tag}_{_queue_task_slug(task)}"
                ),
                "runtime_label": f"{sheet_tag}/{task.label}",
            }

        contexts = {
            task: make_context(task)
            for task in tasks
        }
        pending_train.extend(
            contexts[task]
            for task in table_pending_train
        )
        for pending_state in table_pending_evaluation:
            pending_state = dict(pending_state)
            pending_state["runtime_context"] = contexts[
                pending_state["task"]
            ]
            pending_evaluation.append(pending_state)
        total_seed_tasks += len(tasks)

    print(
        f"Seed {int(seed)} two-phase queue: tasks={total_seed_tasks}, "
        f"pending_train={len(pending_train)}, "
        f"recovered_for_eval={len(pending_evaluation)}, "
        f"train_workers={train_worker_count}, "
        f"train_gpu_slots={train_gpu_slots}, strategy={gpu_strategy}; "
        f"eval_workers={evaluation_worker_count}, "
        f"eval_gpu_pool={evaluation_gpu_ids}, "
        f"eval_max_per_gpu={max_evaluations_per_gpu}, "
        f"eval_batch_size={evaluation_batch_size}, logs={session_dir}",
        flush=True,
    )

    available_train_slots = list(range(train_worker_count))
    active_train = {}
    active_evaluation = {}
    completed = []
    completed_report_paths = []

    try:
        print(
            f"Seed {int(seed)} phase 1/2: training started; evaluation is "
            "blocked until every training task for this seed finishes.",
            flush=True,
        )
        while pending_train or active_train:
            for process_id, state in list(active_train.items()):
                return_code = state["process"].poll()
                if return_code is None:
                    continue
                _close_process_log(state)
                active_train.pop(process_id)
                available_train_slots.append(state["slot_index"])
                available_train_slots.sort()
                context = state["runtime_context"]
                task = context["task"]
                if return_code != 0:
                    _update_queue_task_state(
                        context["queue_state_path"],
                        context["queue_state"],
                        task,
                        "training_failed",
                        pid=state["process"].pid,
                        log_path=str(state["log_path"]),
                        return_code=int(return_code),
                    )
                    raise RuntimeError(
                        f"Training task {context['runtime_label']} failed "
                        f"with exit code {return_code}. See "
                        f"{state['log_path']}."
                    )
                checkpoint_root = _read_training_job_result(
                    state["result_path"]
                )
                _update_queue_task_state(
                    context["queue_state_path"],
                    context["queue_state"],
                    task,
                    "trained",
                    checkpoint_root=str(checkpoint_root),
                    train_log_path=str(state["log_path"]),
                )
                pending_evaluation.append({
                    "task": task,
                    "checkpoint_root": checkpoint_root,
                    "runtime_context": context,
                })
                print(
                    f"Seed {int(seed)} training completed: "
                    f"{context['runtime_label']}; checkpoint="
                    f"{checkpoint_root}",
                    flush=True,
                )

            while pending_train and available_train_slots:
                slot_index = available_train_slots.pop(0)
                context = pending_train.pop(0)
                task = context["task"]
                gpu_slot = train_gpu_slots[slot_index]
                _print_experiment_task_start(
                    task,
                    context["total_steps"],
                    gpu_slot=gpu_slot,
                )
                train_state = _launch_parallel_training_task(
                    base_config=context["table_config"],
                    task=task,
                    gpu_slot=gpu_slot,
                    session_dir=session_dir,
                    task_slug=context["runtime_slug"],
                )
                train_state["slot_index"] = slot_index
                train_state["runtime_context"] = context
                active_train[train_state["process"].pid] = train_state
                _update_queue_task_state(
                    context["queue_state_path"],
                    context["queue_state"],
                    task,
                    "training",
                    pid=train_state["process"].pid,
                    gpu_ids=str(gpu_slot),
                    log_path=str(train_state["log_path"]),
                )
                print(
                    f"Seed {int(seed)} training started: "
                    f"{context['runtime_label']} on cuda:{gpu_slot}; "
                    f"pid={train_state['process'].pid}, "
                    f"log={train_state['log_path']}",
                    flush=True,
                )
            if active_train:
                time.sleep(poll_seconds)

        print(
            f"Seed {int(seed)} phase 1/2 completed. Phase 2/2: starting "
            f"{len(pending_evaluation)} evaluations.",
            flush=True,
        )
        active_evaluation_gpu_counts = {
            int(gpu_id): 0
            for gpu_id in evaluation_gpu_ids
        }
        last_memory_wait_message = 0.0
        while pending_evaluation or active_evaluation:
            for process_id, state in list(active_evaluation.items()):
                return_code = state["process"].poll()
                if return_code is None:
                    continue
                _close_process_log(state)
                active_evaluation.pop(process_id)
                gpu_id = state["physical_gpu_id"]
                active_evaluation_gpu_counts[gpu_id] -= 1
                context = state["runtime_context"]
                task = context["task"]
                if return_code != 0:
                    _update_queue_task_state(
                        context["queue_state_path"],
                        context["queue_state"],
                        task,
                        "evaluation_failed",
                        pid=state["process"].pid,
                        checkpoint_root=str(state["checkpoint_root"]),
                        log_path=str(state["log_path"]),
                        return_code=int(return_code),
                    )
                    raise RuntimeError(
                        f"Evaluation task {context['runtime_label']} failed "
                        f"with exit code {return_code}. See "
                        f"{state['log_path']}."
                    )
                report_path, report = _record_experiment_result(
                    sheet_path=context["sheet_path"],
                    results_base_dir=results_base_dir,
                    task=task,
                    evaluation_started_at=state["started_at"],
                    update_sheet_results=update_sheet_results,
                )
                _update_queue_task_state(
                    context["queue_state_path"],
                    context["queue_state"],
                    task,
                    "completed",
                    checkpoint_root=str(state["checkpoint_root"]),
                    evaluation_log_path=str(state["log_path"]),
                    report_path=str(report_path),
                    bev_ap=float(report["bev_ap"]),
                    threed_ap=float(report["threed_ap"]),
                )
                completed_report_paths.append(report_path)
                completed.append((
                    context["table_index"],
                    task.ordinal,
                    task.experiment.name,
                    task.branch,
                ))
                print(
                    f"Seed {int(seed)} evaluation completed: "
                    f"{context['runtime_label']} on cuda:{gpu_id}.",
                    flush=True,
                )

            evaluation_waiting_for_memory = False
            while (
                pending_evaluation
                and len(active_evaluation) < evaluation_worker_count
            ):
                selected_gpu = select_parallel_evaluation_gpu(
                    candidate_gpu_ids=evaluation_gpu_ids,
                    active_gpu_counts=active_evaluation_gpu_counts,
                    min_free_memory_mb=min_free_memory_mb,
                    reservation_memory_mb=reservation_memory_mb,
                    max_active_per_gpu=max_evaluations_per_gpu,
                )
                if selected_gpu is None:
                    evaluation_waiting_for_memory = True
                    now = time.monotonic()
                    if now - last_memory_wait_message >= 30.0:
                        print(
                            f"Seed {int(seed)} evaluation is waiting for "
                            f"GPU capacity ({min_free_memory_mb} MiB free, "
                            f"max {max_evaluations_per_gpu} per GPU).",
                            flush=True,
                        )
                        last_memory_wait_message = now
                    break
                pending_state = pending_evaluation.pop(0)
                context = pending_state["runtime_context"]
                evaluation_state = _launch_parallel_evaluation_task(
                    pending_state=pending_state,
                    physical_gpu_id=selected_gpu["index"],
                    session_dir=session_dir,
                    task_slug=context["runtime_slug"],
                    evaluation_batch_size=evaluation_batch_size,
                )
                evaluation_state["runtime_context"] = context
                active_evaluation[
                    evaluation_state["process"].pid
                ] = evaluation_state
                gpu_id = evaluation_state["physical_gpu_id"]
                active_evaluation_gpu_counts[gpu_id] += 1
                _update_queue_task_state(
                    context["queue_state_path"],
                    context["queue_state"],
                    context["task"],
                    "evaluating",
                    pid=evaluation_state["process"].pid,
                    checkpoint_root=str(
                        evaluation_state["checkpoint_root"]
                    ),
                    physical_gpu_id=int(gpu_id),
                    log_path=str(evaluation_state["log_path"]),
                )
                print(
                    f"Seed {int(seed)} evaluation started: "
                    f"{context['runtime_label']} on physical cuda:{gpu_id}; "
                    f"active_on_gpu={active_evaluation_gpu_counts[gpu_id]}/"
                    f"{max_evaluations_per_gpu}, "
                    f"batch_size={evaluation_batch_size}, "
                    f"log={evaluation_state['log_path']}",
                    flush=True,
                )

            if (
                pending_evaluation
                and not active_evaluation
                and evaluation_waiting_for_memory
            ):
                time.sleep(poll_seconds)
                continue
            if active_evaluation:
                time.sleep(poll_seconds)

    except BaseException:
        _terminate_running_processes(active_train, active_evaluation)
        raise
    finally:
        for state in active_train.values():
            _close_process_log(state)
        for state in active_evaluation.values():
            _close_process_log(state)

    _refresh_completed_weather_summaries(
        results_base_dir=results_base_dir,
        report_paths=completed_report_paths,
    )
    print(
        f"Seed {int(seed)} two-phase queue completed: "
        f"{len(completed)} evaluations.",
        flush=True,
    )
    return [
        (name, branch)
        for _table, _ordinal, name, branch in sorted(completed)
    ]


def run_domain_shift_experiment_queue(base_config, train_function):
    """Run one continuous queue across one or more ordered weather tables."""
    sheet_paths = resolve_experiment_sheet_paths(base_config)
    ensure_no_other_top_level_train_process()
    with ExitStack() as lock_stack:
        for sheet_path in sheet_paths:
            lock_stack.enter_context(experiment_queue_lock(sheet_path))

        branches = normalize_queue_branches(
            base_config.get("experiment_queue_branches", VALID_BRANCHES)
        )
        skip_completed = bool(
            base_config.get(
                "experiment_queue_skip_completed_branches",
                False,
            )
        )
        update_sheet_results = bool(
            base_config.get(
                "experiment_queue_update_sheet_results",
                True,
            )
        )
        results_base_dir = base_config.get(
            "experiment_results_base_dir",
            "evaluation_plots",
        )
        train_workers = int(
            base_config.get("experiment_queue_train_workers", 1)
        )
        evaluation_workers = int(
            base_config.get("experiment_queue_eval_workers", 1)
        )
        queue_order = str(
            base_config.get(
                "experiment_queue_order",
                "seed_then_weather",
            )
        ).strip().lower()
        if queue_order not in {"seed_then_weather", "weather_then_seed"}:
            raise ValueError(
                "experiment_queue_order must be 'seed_then_weather' or "
                f"'weather_then_seed', got {queue_order!r}."
            )
        execution_mode = str(
            base_config.get(
                "experiment_queue_execution_mode",
                "pipelined",
            )
        ).strip().lower()
        if execution_mode not in {"pipelined", "seed_two_phase"}:
            raise ValueError(
                "experiment_queue_execution_mode must be 'pipelined' or "
                f"'seed_two_phase', got {execution_mode!r}."
            )
        if (
            execution_mode == "seed_two_phase"
            and queue_order != "seed_then_weather"
        ):
            raise ValueError(
                "seed_two_phase execution requires "
                "experiment_queue_order='seed_then_weather'."
            )

        prepared_tables = []
        total_rows = 0
        total_pending = 0
        for sheet_path in sheet_paths:
            experiments = load_domain_shift_experiments(
                sheet_path,
                default_seed=base_config.get("seed", 42),
            )
            validate_experiment_queue_design(experiments)
            total_steps = len(experiments) * len(branches)
            tasks = build_experiment_queue_tasks(
                experiments=experiments,
                branches=branches,
                skip_completed=skip_completed,
            )
            prepared_tables.append((
                sheet_path,
                experiments,
                total_steps,
                tasks,
            ))
            total_rows += len(experiments)
            total_pending += len(tasks)

        print(
            "Multi-weather experiment queue: "
            f"tables={len(prepared_tables)}, rows={total_rows}, "
            f"branches={branches}, pending={total_pending}, "
            f"skip_completed={skip_completed}, order={queue_order}, "
            f"execution={execution_mode}",
            flush=True,
        )

        seed_queue_batches = []
        if queue_order == "seed_then_weather":
            available_seeds = {
                int(experiment.seed)
                for _path, experiments, _steps, _tasks in prepared_tables
                for experiment in experiments
            }
            configured_seed_order = base_config.get(
                "experiment_queue_seed_order",
                (42, 43, 44),
            )
            if isinstance(configured_seed_order, str):
                configured_seed_order = tuple(
                    part.strip()
                    for part in configured_seed_order.split(",")
                    if part.strip()
                )
            configured_seed_order = tuple(
                int(seed)
                for seed in configured_seed_order
            )
            if len(set(configured_seed_order)) != len(
                configured_seed_order
            ):
                raise ValueError(
                    "experiment_queue_seed_order contains duplicate seeds: "
                    f"{configured_seed_order}."
                )
            seeds = [
                seed
                for seed in configured_seed_order
                if seed in available_seeds
            ]
            seeds.extend(sorted(
                available_seeds.difference(configured_seed_order)
            ))
            queue_batches = []
            for seed_index, seed in enumerate(seeds, start=1):
                current_seed_batches = []
                for table_index, prepared_table in enumerate(
                    prepared_tables,
                    start=1,
                ):
                    sheet_path, experiments, total_steps, tasks = (
                        prepared_table
                    )
                    seed_experiments = tuple(
                        experiment
                        for experiment in experiments
                        if int(experiment.seed) == seed
                    )
                    if not seed_experiments:
                        continue
                    seed_tasks = tuple(
                        task
                        for task in tasks
                        if int(task.experiment.seed) == seed
                    )
                    current_seed_batches.append((
                        table_index,
                        sheet_path,
                        total_steps,
                        seed_tasks,
                    ))
                    queue_batches.append((
                        seed_index,
                        len(seeds),
                        seed,
                        table_index,
                        sheet_path,
                        seed_experiments,
                        total_steps,
                        seed_tasks,
                    ))
                seed_queue_batches.append((
                    seed_index,
                    len(seeds),
                    seed,
                    tuple(current_seed_batches),
                ))
        else:
            queue_batches = []
            for table_index, prepared_table in enumerate(
                prepared_tables,
                start=1,
            ):
                sheet_path, experiments, total_steps, tasks = prepared_table
                queue_batches.append((
                    None,
                    None,
                    None,
                    table_index,
                    sheet_path,
                    tuple(experiments),
                    total_steps,
                    tuple(tasks),
                ))

        launched = []
        if execution_mode == "seed_two_phase":
            if train_workers <= 1 and evaluation_workers <= 1:
                raise ValueError(
                    "seed_two_phase execution requires parallel workers."
                )
            for seed_index, seed_count, seed, table_batches in (
                seed_queue_batches
            ):
                print(
                    f"Seed batch [{seed_index}/{seed_count}]={seed}: "
                    f"weather_tables={len(table_batches)}, "
                    f"pending={sum(len(batch[3]) for batch in table_batches)}",
                    flush=True,
                )
                seed_launched = _run_seed_two_phase_experiment_queue(
                    base_config=base_config,
                    seed=seed,
                    table_batches=table_batches,
                    results_base_dir=results_base_dir,
                    update_sheet_results=update_sheet_results,
                )
                launched.extend(seed_launched)
                print(
                    f"Seed batch [{seed_index}/{seed_count}]={seed} "
                    f"completed ({len(seed_launched)} runs).",
                    flush=True,
                )

            if bool(
                base_config.get("experiment_queue_require_full_table", False)
            ):
                for sheet_path, _experiments, _total_steps, _tasks in (
                    prepared_tables
                ):
                    validate_experiment_sheet_is_full(sheet_path)
            print(
                f"Multi-weather experiment queue completed: {len(launched)} "
                f"training/evaluation runs across "
                f"{len(prepared_tables)} tables.",
                flush=True,
            )
            return launched

        for (
            seed_index,
            seed_count,
            seed,
            table_index,
            sheet_path,
            experiments,
            total_steps,
            tasks,
        ) in queue_batches:
            seed_label = ""
            if seed is not None:
                seed_label = f", seed [{seed_index}/{seed_count}]={seed}"
            print(
                f"Weather table [{table_index}/{len(prepared_tables)}]: "
                f"{sheet_path} ({len(experiments)} rows, "
                f"pending={len(tasks)}{seed_label})",
                flush=True,
            )
            table_config = copy.deepcopy(dict(base_config))
            table_config["experiment_sheet_path"] = str(sheet_path)
            if train_workers > 1 or evaluation_workers > 1:
                table_launched = _run_parallel_experiment_queue(
                    base_config=table_config,
                    tasks=tasks,
                    total_steps=total_steps,
                    sheet_path=sheet_path,
                    results_base_dir=results_base_dir,
                    update_sheet_results=update_sheet_results,
                )
            else:
                table_launched = _run_sequential_experiment_queue(
                    base_config=table_config,
                    train_function=train_function,
                    tasks=tasks,
                    total_steps=total_steps,
                    sheet_path=sheet_path,
                    results_base_dir=results_base_dir,
                    update_sheet_results=update_sheet_results,
                )
            launched.extend(table_launched)
            print(
                f"Weather table [{table_index}/{len(prepared_tables)}] "
                f"completed{seed_label}: {sheet_path.name} "
                f"({len(table_launched)} runs).",
                flush=True,
            )

        if bool(
            base_config.get("experiment_queue_require_full_table", False)
        ):
            for sheet_path, _experiments, _total_steps, _tasks in (
                prepared_tables
            ):
                validate_experiment_sheet_is_full(sheet_path)
        print(
            f"Multi-weather experiment queue completed: {len(launched)} "
            f"training/evaluation runs across {len(prepared_tables)} tables.",
            flush=True,
        )
        return launched
