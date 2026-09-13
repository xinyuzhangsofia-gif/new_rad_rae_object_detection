#!/usr/bin/env python3
"""Re-evaluate rain/sleet experiments with distance-stratified official AP.

The launcher discovers the already-trained source/target checkpoint roots from
the experiment queue state files.  Each child evaluates epochs 5--24 once and
computes the full-range metric plus four distance bins from the same collected
predictions.  Results are written to ``experiments2`` without modifying the
original experiment tables or reports.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_analysis import discovery as analysis_discovery
from scripts.experiment_analysis import execution as analysis_execution
from scripts.experiment_analysis import results as analysis_results
from scripts.experiment_analysis import state as analysis_state


SOURCE_EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
SOURCE_REPORT_ROOT = PROJECT_ROOT / "evaluation_plots"
WEATHERS = ("rain", "sleet")
DISTANCE_BINS = (
    (0.0, 30.0, "0_30m"),
    (30.0, 60.0, "30_60m"),
    (60.0, 90.0, "60_90m"),
    (90.0, 120.0, "90_120m"),
)
DISTANCE_BIN_TEXT = "0-30,30-60,60-90,90-120"
METADATA_HEADERS = analysis_discovery.METADATA_HEADERS
GEOMETRIES = ("BEV", "3D")
BRANCHES = ("source", "target")
STATE_VERSION = 1


def utc_now_text():
    return analysis_state.now_text()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate completed rain/sleet domain-shift checkpoints with "
            "official AP split into 0-30, 30-60, 60-90, and 90-120 m."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "experiments2"),
        help="Root for expanded tables, reports, logs, TensorBoard, and state.",
    )
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--max-workers", type=int, default=9)
    parser.add_argument("--max-per-gpu", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate discovery and create blank tables without launching.",
    )
    args = parser.parse_args(argv)
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.output_dir == SOURCE_EXPERIMENT_DIR.resolve():
        raise ValueError(
            f"Refusing to overwrite the source experiment directory: {args.output_dir}"
        )
    args.gpus = tuple(
        int(token.strip())
        for token in str(args.gpus).split(",")
        if token.strip() != ""
    )
    if not args.gpus:
        raise ValueError("At least one physical GPU id is required.")
    if any(gpu < 0 for gpu in args.gpus):
        raise ValueError(f"GPU ids must be non-negative, got {args.gpus!r}.")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError(f"GPU ids must be unique, got {args.gpus!r}.")
    if args.max_workers <= 0 or args.max_per_gpu <= 0:
        raise ValueError("max-workers and max-per-gpu must be positive.")
    if args.max_workers > len(args.gpus) * args.max_per_gpu:
        raise ValueError(
            "max-workers exceeds GPU slot capacity: "
            f"{args.max_workers} > {len(args.gpus)}*{args.max_per_gpu}."
        )
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.poll_seconds <= 0.0:
        raise ValueError("poll-seconds must be positive.")
    return args


def read_experiment_rows(weather):
    """Return source table metadata rows in their existing display order."""
    return analysis_discovery.read_experiment_rows(
        SOURCE_EXPERIMENT_DIR,
        weather,
        METADATA_HEADERS,
    )


def row_identity(row):
    return analysis_discovery.row_identity(row)


def queue_state_path(weather):
    return analysis_discovery.queue_state_path(
        SOURCE_EXPERIMENT_DIR,
        weather,
    )


def load_canonical_checkpoint_records(weather, experiment_rows):
    """Select one completed record for every valid group/seed/branch."""
    return analysis_discovery.load_completed_checkpoint_records(
        state_path=queue_state_path(weather),
        experiment_rows=experiment_rows,
        branches=BRANCHES,
        weather=weather,
    )


def derive_output_report_path(original_report_path, reports_root):
    return analysis_discovery.derive_output_report_path(
        original_report_path,
        SOURCE_REPORT_ROOT,
        reports_root,
    )


def discover_tasks(all_rows, output_dir):
    return analysis_discovery.discover_paired_branch_tasks(
        all_rows=all_rows,
        output_dir=output_dir,
        weathers=WEATHERS,
        branches=BRANCHES,
        load_records=load_canonical_checkpoint_records,
        source_report_root=SOURCE_REPORT_ROOT,
        expected_count=54,
    )


def atomic_write_text(path, text):
    return analysis_state.atomic_write_text(path, text)


def atomic_write_json(path, payload):
    return analysis_state.atomic_write_json(path, payload)


def acquire_output_lock(output_dir):
    """Hold an exclusive lock for the lifetime of one launcher process."""
    return analysis_state.acquire_output_lock(
        output_dir=output_dir,
        lock_name=".distance_evaluation.lock",
        conflict_message=(
            "Another distance-evaluation launcher holds {lock_path}."
        ),
        timestamp=utc_now_text,
    )


def validate_physical_gpus(gpus):
    """Fail before launch if nvidia-smi cannot see every configured GPU."""
    return analysis_execution.validate_physical_gpus(
        gpus,
        run=subprocess.run,
    )


def task_process_is_alive(task):
    """Return True only when a recorded PID is still this task's evaluator."""
    return analysis_state.task_process_is_alive(task)


def parse_average_report(report_path):
    report_path = Path(report_path)
    if not report_path.is_file():
        return None
    text = report_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"average_AP_epoch_5_to_24 \(epochs_used=(\d+)\): ([^\n]+)",
        text,
    )
    if match is None or int(match.group(1)) != 20:
        return None
    values = {
        key: float(value)
        for key, value in re.findall(
            r"([A-Za-z0-9@._]+)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            match.group(2),
        )
    }
    required = {"bev@0.3", "3d@0.3"}
    for _, _, tag in DISTANCE_BINS:
        required.add(f"bev@0.3_range_{tag}")
        required.add(f"3d@0.3_range_{tag}")
    if not required.issubset(values):
        return None
    metrics = {
        "all": {
            "BEV": values["bev@0.3"],
            "3D": values["3d@0.3"],
        }
    }
    for _, _, tag in DISTANCE_BINS:
        metrics[tag] = {
            "BEV": values[f"bev@0.3_range_{tag}"],
            "3D": values[f"3d@0.3_range_{tag}"],
        }
    return metrics


def report_matches_task(report_path, task):
    """Reject a same-named report that belongs to another experiment row."""
    return analysis_results.report_matches_task_metadata(
        report_path,
        task,
        parse_average_report,
    )


def find_completed_report(task, reports_root):
    return analysis_results.find_completed_report(
        task,
        reports_root,
        report_matches_task,
    )


def initialize_state(state_path, discovered_tasks, reports_root):
    previous_tasks = {}
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        previous_tasks = previous.get("tasks", {})

    tasks = {}
    for task_id, discovered in discovered_tasks.items():
        previous = previous_tasks.get(task_id, {})
        if previous.get("status") == "running" and task_process_is_alive(previous):
            raise RuntimeError(
                f"Recorded evaluator is still alive for {task_id} "
                f"(pid={previous.get('pid')}). Stop it or let it finish before restart."
            )
        task = analysis_state.initialize_runtime_task(discovered, previous)
        if previous.get("status") == "completed":
            report = find_completed_report(task, reports_root)
            metrics = None if report is None else parse_average_report(report)
            if metrics is not None:
                task["status"] = "completed"
                task["metrics"] = metrics
                task["report_path"] = str(report)
        tasks[task_id] = task

    state = {
        "version": STATE_VERSION,
        "updated_at": utc_now_text(),
        "settings": {
            "epochs": [5, 24],
            "distance_m": [[lower, upper] for lower, upper, _ in DISTANCE_BINS],
            "distance_definition": "sqrt(x^2+y^2+z^2)",
            "distance_intervals": "left-closed, right-open",
        },
        "tasks": tasks,
    }
    atomic_write_json(state_path, state)
    return state


def metric_headers():
    headers = list(METADATA_HEADERS)
    for block in ("all",) + tuple(tag for _, _, tag in DISTANCE_BINS):
        suffix = "" if block == "all" else f"_{block}"
        headers.extend(
            (
                f"BEV_src{suffix}",
                f"3D_src{suffix}",
                f"BEV_tgt{suffix}",
                f"3D_tgt{suffix}",
                f"TD_BEV{suffix}",
                f"TD_3D{suffix}",
            )
        )
    return headers


def task_lookup(state):
    return {
        (
            task["weather"],
            task["group"],
            int(task["seed"]),
            task["branch"],
        ): task
        for task in state["tasks"].values()
    }


def format_metric(value):
    return "" if value is None else f"{float(value):.4f}"


def build_table_matrix(weather, experiment_rows, state):
    headers = metric_headers()
    lookup = task_lookup(state)
    output_rows = []
    td_values = {block: [] for block in ("all",) + tuple(
        tag for _, _, tag in DISTANCE_BINS
    )}

    for metadata in experiment_rows:
        row = [str(metadata[header]) for header in METADATA_HEADERS]
        identity = row_identity(metadata)
        if identity is None:
            row.extend([""] * (len(headers) - len(row)))
            output_rows.append(row)
            continue
        group, seed = identity
        source = lookup[(weather, group, seed, "source")]
        target = lookup[(weather, group, seed, "target")]
        source_metrics = (
            source.get("metrics") if source.get("status") == "completed" else None
        )
        target_metrics = (
            target.get("metrics") if target.get("status") == "completed" else None
        )
        for block in td_values:
            source_block = None if source_metrics is None else source_metrics.get(block)
            target_block = None if target_metrics is None else target_metrics.get(block)
            source_bev = None if source_block is None else source_block.get("BEV")
            source_3d = None if source_block is None else source_block.get("3D")
            target_bev = None if target_block is None else target_block.get("BEV")
            target_3d = None if target_block is None else target_block.get("3D")
            td_bev = (
                None
                if source_bev is None or target_bev is None
                else float(target_bev) - float(source_bev)
            )
            td_3d = (
                None
                if source_3d is None or target_3d is None
                else float(target_3d) - float(source_3d)
            )
            row.extend(
                format_metric(value)
                for value in (
                    source_bev,
                    source_3d,
                    target_bev,
                    target_3d,
                    td_bev,
                    td_3d,
                )
            )
            if td_bev is not None and td_3d is not None:
                td_values[block].append((td_bev, td_3d))
        output_rows.append(row)

    complete_pairs = len(td_values["all"])
    average_row = [f"average({complete_pairs})"] + ["-"] * 5
    for block in td_values:
        values = td_values[block]
        if values:
            average_bev = sum(value[0] for value in values) / len(values)
            average_3d = sum(value[1] for value in values) / len(values)
        else:
            average_bev = average_3d = None
        average_row.extend(
            ("", "", "", "", format_metric(average_bev), format_metric(average_3d))
        )
    output_rows.append(average_row)
    return [headers] + output_rows


def format_table(matrix):
    widths = [
        max(len(str(row[index])) for row in matrix)
        for index in range(len(matrix[0]))
    ]

    def format_row(row, header=False):
        cells = []
        for index, value in enumerate(row):
            value = str(value)
            left_aligned = header or index < len(METADATA_HEADERS)
            cells.append(
                f"{value:<{widths[index]}}"
                if left_aligned
                else f"{value:>{widths[index]}}"
            )
        return "  ".join(cells).rstrip()

    header = format_row(matrix[0], header=True)
    lines = [header, "-" * len(header)]
    lines.extend(format_row(row) for row in matrix[1:])
    return "\n".join(lines) + "\n"


def write_tables(all_rows, state, output_dir):
    for weather in WEATHERS:
        matrix = build_table_matrix(weather, all_rows[weather], state)
        atomic_write_text(
            output_dir / f"{weather}_experiments.txt",
            format_table(matrix),
        )


def write_readme(output_dir):
    text = (
        "Rain and sleet checkpoint re-evaluation with distance-stratified "
        "official K-Radar AP@IoU 0.3.\n\n"
        "Distance is the radar-center norm sqrt(x^2+y^2+z^2). GT and "
        "detections are filtered independently. Bins are [0,30), [30,60), "
        "[60,90), and [90,120) metres. Every value is the mean over epochs "
        "5-24 inclusive. TD = target-trained AP - source-trained AP.\n"
    )
    atomic_write_text(output_dir / "README.txt", text)


def build_evaluation_command(task, args):
    return analysis_execution.build_evaluation_command(
        checkpoint_root=task["checkpoint_root"],
        batch_size=args.batch_size,
        table_output_base_dir=args.output_dir / "evaluation_reports",
        tensorboard_log_dir=args.output_dir / "tensorboard",
        project_root=PROJECT_ROOT,
        analysis_arguments=(
            "--distance-range-eval-enabled", "true",
            "--distance-range-bins", DISTANCE_BIN_TEXT,
        ),
        python_executable=sys.executable,
    )


def state_counts(state):
    return analysis_state.state_counts(state)


def save_progress(state_path, state, all_rows, output_dir):
    state["updated_at"] = utc_now_text()
    atomic_write_json(state_path, state)
    write_tables(all_rows, state, output_dir)


def choose_gpu(gpu_use, gpus, max_per_gpu):
    return analysis_execution.choose_gpu(gpu_use, gpus, max_per_gpu)


def run_queue(args, state_path, state, all_rows):
    reports_root = args.output_dir / "evaluation_reports"

    def resolve_result(task):
        report = find_completed_report(task, reports_root)
        metrics = None if report is None else parse_average_report(report)
        return metrics, report

    return analysis_execution.run_evaluation_jobs(
        args=args,
        state=state,
        build_command=lambda task: build_evaluation_command(task, args),
        resolve_result=resolve_result,
        save_progress=lambda: save_progress(
            state_path, state, all_rows, args.output_dir
        ),
        project_root=PROJECT_ROOT,
        now=utc_now_text,
        announce_stop=True,
        skip_stop_when_empty=True,
        popen=subprocess.Popen,
        environment=os.environ,
        signal_module=signal,
        sleep=time.sleep,
    )


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_reports").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    lock_handle = acquire_output_lock(args.output_dir)
    try:
        all_rows = {weather: read_experiment_rows(weather) for weather in WEATHERS}
        discovered_tasks = discover_tasks(all_rows, args.output_dir)
        state_path = args.output_dir / "distance_evaluation_state.json"
        state = initialize_state(
            state_path,
            discovered_tasks,
            args.output_dir / "evaluation_reports",
        )
        write_readme(args.output_dir)
        write_tables(all_rows, state, args.output_dir)

        counts = state_counts(state)
        print(
            f"Validated {len(state['tasks'])} tasks; state={state_path}; "
            f"status={counts}",
            flush=True,
        )
        if args.dry_run:
            first_pending = next(
                (
                    task
                    for task in state["tasks"].values()
                    if task["status"] == "pending"
                ),
                None,
            )
            if first_pending is not None:
                print(
                    "Example command:",
                    " ".join(build_evaluation_command(first_pending, args)),
                    flush=True,
                )
            print("Dry run complete; no evaluation processes were launched.", flush=True)
            return 0
        validate_physical_gpus(args.gpus)
        return run_queue(args, state_path, state, all_rows)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
