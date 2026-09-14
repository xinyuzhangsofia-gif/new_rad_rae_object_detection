"""Training/evaluation child configuration, launch, and result handling."""

import copy
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

from training.experiments.schema import VALID_BRANCHES
from training.experiments.scheduling import _effective_train_signature
from training.experiments.state import _queue_task_slug
from training.experiments.tables import (
    _branch_half_selection,
    find_fresh_experiment_result,
    update_experiment_sheet_result,
)
from training.post_training_evaluation import (
    prepare_post_training_evaluation_launch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_training_job_config(path, config):
    path = Path(path)
    with path.open("wb") as output_file:
        pickle.dump(dict(config), output_file, protocol=pickle.HIGHEST_PROTOCOL)
        output_file.flush()
        os.fsync(output_file.fileno())


def _read_training_job_result(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Training worker result file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("ok", False)):
        raise RuntimeError(
            "Training worker reported failure: "
            f"{payload.get('error', 'unknown error')}\n"
            f"{payload.get('traceback', '')}"
        )
    checkpoint_root = payload.get("checkpoint_root")
    if checkpoint_root in (None, ""):
        raise RuntimeError(
            f"Training worker returned no checkpoint root: {path}"
        )
    return Path(checkpoint_root).expanduser().resolve()


def _close_process_log(state):
    log_file = state.get("log_file")
    if log_file is not None and not log_file.closed:
        log_file.close()


def _terminate_running_processes(*active_process_maps):
    states = [
        state
        for process_map in active_process_maps
        for state in process_map.values()
        if state["process"].poll() is None
    ]
    for state in states:
        state["process"].terminate()
    deadline = time.monotonic() + 10.0
    for state in states:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            state["process"].wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            state["process"].kill()
            state["process"].wait()
    for process_map in active_process_maps:
        for state in process_map.values():
            _close_process_log(state)


def build_experiment_training_config(base_config, experiment, branch):
    """Create one ordinary train.py configuration from one sheet row."""
    if branch not in VALID_BRANCHES:
        raise ValueError(f"Unsupported experiment branch: {branch!r}")
    config = copy.deepcopy(dict(base_config))
    source_control_enabled = config.get(
        "train_control_split_enabled",
        False,
    )
    branch_half_selection = _branch_half_selection(experiment, branch)
    config.update({
        "experiment_queue_enabled": False,
        "domain_shift_train_branch": branch,
        "shared_train_sequences": experiment.shared_sequences,
        "source_train_sequences": experiment.source_sequences,
        "target_train_sequences": experiment.target_sequences,
        "target_test_sequences": experiment.test_sequences,
        "train_sequence_half_selection": branch_half_selection,
        "controlled_sequence_parts": experiment.source_parts,
        "reference_sequence_parts": experiment.target_parts,
        "seed": experiment.seed,
        "post_training_eval_enabled": True,
        # Distribution control filters the source domain to match its paired
        # target references.  The target branch must always train on its
        # original, unfiltered data.
        "train_control_split_enabled": (
            source_control_enabled if branch == "source" else False
        ),
    })
    return config


def _print_experiment_task_start(task, total_steps, gpu_slot=None):
    experiment = task.experiment
    gpu_text = "" if gpu_slot is None else f", gpu_ids={gpu_slot}"
    print(
        f"Experiment queue [{task.ordinal}/{total_steps}]: start "
        f"{experiment.name} {task.branch} "
        f"(shared={experiment.shared_sequences}, "
        f"source={experiment.source_sequences}, "
        f"target={experiment.target_sequences}, "
        f"source_parts={experiment.source_parts}, "
        f"effective_train={_effective_train_signature(experiment, task.branch)}, "
        f"test={experiment.test_sequences}, "
        f"half={_branch_half_selection(experiment, task.branch)}, "
        f"seed={experiment.seed}{gpu_text})",
        flush=True,
    )


def _record_experiment_result(
        sheet_path,
        results_base_dir,
        task,
        evaluation_started_at,
        update_sheet_results,
    ):
    report_path, report = find_fresh_experiment_result(
        results_base_dir=results_base_dir,
        experiment=task.experiment,
        branch=task.branch,
        started_at=evaluation_started_at,
    )
    if update_sheet_results:
        updated_path = update_experiment_sheet_result(
            sheet_path=sheet_path,
            experiment=task.experiment,
            branch=task.branch,
            bev_ap=report["bev_ap"],
            threed_ap=report["threed_ap"],
        )
        print(
            f"Experiment queue result: {report_path} -> {updated_path}",
            flush=True,
        )
        from eval.reporting import refresh_total_result_summary

        refresh_total_result_summary(results_base_dir)
    return report_path, report


def _launch_parallel_training_task(
        base_config,
        task,
        gpu_slot,
        session_dir,
        task_slug=None,
    ):
    task_slug = task_slug or _queue_task_slug(task)
    config_path = session_dir / f"{task_slug}.config.pkl"
    result_path = session_dir / f"{task_slug}.train_result.json"
    log_path = session_dir / f"{task_slug}.train.log"
    child_config = build_experiment_training_config(
        base_config,
        task.experiment,
        task.branch,
    )
    child_config.update({
        "gpu_ids": str(gpu_slot),
        "post_training_eval_enabled": False,
        "experiment_queue_task_id": task_slug,
        "experiment_queue_group": task.experiment.name,
    })
    # A task that was interrupted after writing epoch checkpoints can resume
    # from the last complete checkpoint.  The mapping is intentionally
    # keyed by the queue runtime slug so it is unambiguous across weather
    # tables and experiment groups.
    resume_checkpoints = base_config.get(
        "experiment_queue_resume_checkpoints",
        {},
    )
    resume_checkpoint = resume_checkpoints.get(task_slug)
    if resume_checkpoint:
        child_config.update({
            "resume_checkpoint": str(resume_checkpoint),
            "start_epoch": None,
            "end_epoch": int(child_config.get("epochs", 30)),
            "load_optimizer": True,
            "initial_best_checkpoint": None,
            "resume_save_in_checkpoint_dir": True,
            "resume_tensorboard_log_dir": None,
        })
    _write_training_job_config(config_path, child_config)
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-m",
        "training.experiments.worker",
        "--config",
        str(config_path),
        "--result",
        str(result_path),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except BaseException:
        log_file.close()
        raise
    return {
        "process": process,
        "task": task,
        "gpu_slot": str(gpu_slot),
        "result_path": result_path,
        "log_path": log_path,
        "log_file": log_file,
        "started_at": time.time(),
    }


def _launch_parallel_evaluation_task(
        pending_state,
        physical_gpu_id,
        session_dir,
        task_slug=None,
        evaluation_batch_size=None,
    ):
    task = pending_state["task"]
    task_slug = task_slug or _queue_task_slug(task)
    log_path = session_dir / f"{task_slug}.evaluation.log"
    command, root_dir, environment = prepare_post_training_evaluation_launch(
        checkpoint_root=pending_state["checkpoint_root"],
        physical_gpu_id=physical_gpu_id,
        python_executable=sys.executable,
        project_dir=PROJECT_ROOT,
        batch_size=evaluation_batch_size,
    )
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command,
            cwd=str(root_dir),
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except BaseException:
        log_file.close()
        raise
    return {
        "process": process,
        "task": task,
        "checkpoint_root": pending_state["checkpoint_root"],
        "physical_gpu_id": int(physical_gpu_id),
        "log_path": log_path,
        "log_file": log_file,
        "started_at": time.time(),
    }


def _refresh_completed_weather_summaries(results_base_dir, report_paths):
    if not report_paths:
        return
    from eval.reporting import refresh_weather_domain_shift_summary

    base_path = Path(str(results_base_dir)).expanduser()
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    weather_names = set()
    for report_path in report_paths:
        try:
            relative = Path(report_path).resolve().relative_to(
                base_path.resolve()
            )
        except ValueError:
            continue
        if relative.parts:
            weather_names.add(relative.parts[0])
    for weather_name in sorted(weather_names):
        refresh_weather_domain_shift_summary(
            base_dir=base_path,
            weather_group=weather_name,
        )

