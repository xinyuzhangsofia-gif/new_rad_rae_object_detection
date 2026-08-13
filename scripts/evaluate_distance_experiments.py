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
from datetime import datetime
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
METADATA_HEADERS = (
    "group",
    "seed",
    "shared_seq",
    "source_seq",
    "target_seq",
    "test_seq",
)
GEOMETRIES = ("BEV", "3D")
BRANCHES = ("source", "target")
STATE_VERSION = 1


def utc_now_text():
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
    table_path = SOURCE_EXPERIMENT_DIR / f"{weather}_experiments.txt"
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing experiment table: {table_path}")
    rows = []
    for line in table_path.read_text(encoding="utf-8").splitlines()[2:]:
        cells = line.split()
        if not cells or cells[0].lower().startswith("average"):
            continue
        if len(cells) < len(METADATA_HEADERS):
            raise ValueError(f"Malformed row in {table_path}: {line!r}")
        metadata = dict(zip(METADATA_HEADERS, cells[: len(METADATA_HEADERS)]))
        rows.append(metadata)
    if not rows:
        raise ValueError(f"No experiment rows found in {table_path}")
    return rows


def row_identity(row):
    seed_text = str(row["seed"]).strip()
    if not seed_text.isdigit():
        return None
    required = ("shared_seq", "source_seq", "target_seq", "test_seq")
    if any(str(row[key]).strip() in {"", "-"} for key in required):
        return None
    return str(row["group"]), int(seed_text)


def queue_state_path(weather):
    return SOURCE_EXPERIMENT_DIR / f".{weather}_experiments.txt.queue_state.json"


def load_canonical_checkpoint_records(weather, experiment_rows):
    """Select one completed record for every valid group/seed/branch."""
    state_path = queue_state_path(weather)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    expected_pairs = {
        identity
        for identity in (row_identity(row) for row in experiment_rows)
        if identity is not None
    }
    candidates = {}
    for record in payload.get("tasks", {}).values():
        if str(record.get("status", "")).lower() != "completed":
            continue
        branch = str(record.get("branch", "")).lower()
        if branch not in BRANCHES:
            continue
        try:
            identity = (str(record["group"]), int(record["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        if identity not in expected_pairs:
            continue
        checkpoint_root = Path(str(record.get("checkpoint_root", ""))).expanduser()
        if not checkpoint_root.is_dir():
            continue
        key = (*identity, branch)
        previous = candidates.get(key)
        if previous is None or str(record.get("updated_at", "")) > str(
            previous.get("updated_at", "")
        ):
            candidates[key] = dict(record)

    missing = [
        (group, seed, branch)
        for group, seed in sorted(expected_pairs, key=lambda item: (item[1], item[0]))
        for branch in BRANCHES
        if (group, seed, branch) not in candidates
    ]
    if missing:
        raise RuntimeError(
            f"{weather}: missing completed checkpoint records: {missing!r}"
        )
    return candidates


def derive_output_report_path(original_report_path, reports_root):
    original = Path(str(original_report_path)).expanduser().resolve()
    try:
        relative = original.relative_to(SOURCE_REPORT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Report path is outside {SOURCE_REPORT_ROOT}: {original}"
        ) from exc
    return reports_root / relative


def discover_tasks(all_rows, output_dir):
    reports_root = output_dir / "evaluation_reports"
    logs_root = output_dir / "logs"
    tasks = {}
    for weather in WEATHERS:
        records = load_canonical_checkpoint_records(weather, all_rows[weather])
        for row in all_rows[weather]:
            identity = row_identity(row)
            if identity is None:
                continue
            group, seed = identity
            for branch in BRANCHES:
                record = records[(group, seed, branch)]
                task_id = f"{weather}_{group}_seed{seed}_{branch}"
                report_path = derive_output_report_path(
                    record["report_path"], reports_root
                )
                tasks[task_id] = {
                    "task_id": task_id,
                    "weather": weather,
                    "group": group,
                    "seed": seed,
                    "branch": branch,
                    "checkpoint_root": str(
                        Path(record["checkpoint_root"]).expanduser().resolve()
                    ),
                    "source_report_path": str(
                        Path(record["report_path"]).expanduser().resolve()
                    ),
                    "report_path": str(report_path),
                    "log_path": str(logs_root / f"{task_id}.log"),
                }
    expected = 2 * sum(
        row_identity(row) is not None
        for rows in all_rows.values()
        for row in rows
    )
    if len(tasks) != expected or expected != 54:
        raise RuntimeError(
            f"Expected 54 canonical branch tasks, discovered {len(tasks)} "
            f"from {expected} expected entries."
        )
    return tasks


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


def acquire_output_lock(output_dir):
    """Hold an exclusive lock for the lifetime of one launcher process."""
    lock_path = Path(output_dir) / ".distance_evaluation.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            f"Another distance-evaluation launcher holds {lock_path}."
        ) from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} started_at={utc_now_text()}\n")
    lock_handle.flush()
    return lock_handle


def validate_physical_gpus(gpus):
    """Fail before launch if nvidia-smi cannot see every configured GPU."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    available = {
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip() != ""
    }
    missing = sorted(set(gpus) - available)
    if missing:
        raise RuntimeError(
            f"Configured physical GPU ids are unavailable: {missing}; "
            f"nvidia-smi reported {sorted(available)}."
        )


def task_process_is_alive(task):
    """Return True only when a recorded PID is still this task's evaluator."""
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
    return (
        "evaluation.py" in command
        and str(task.get("checkpoint_root", "")) in command
    )


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
    report_path = Path(report_path)
    if parse_average_report(report_path) is None:
        return False
    metadata = {}
    for line in report_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if line.strip() == "":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    try:
        report_checkpoint_root = Path(metadata["checkpoint_root"]).expanduser().resolve()
    except (KeyError, OSError):
        return False
    return (
        report_checkpoint_root == Path(task["checkpoint_root"]).resolve()
        and metadata.get("domain_shift_train_branch") == task["branch"]
        and metadata.get("weather_group") == task["weather"]
        and metadata.get("seed") == str(task["seed"])
    )


def find_completed_report(task, reports_root):
    expected = Path(task["report_path"])
    if report_matches_task(expected, task):
        return expected
    filename = f"seed{task['seed']}_{task['branch']}_result.txt"
    weather_root = reports_root / task["weather"]
    candidates = sorted(
        weather_root.rglob(filename) if weather_root.is_dir() else (),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if report_matches_task(candidate, task):
            return candidate
    return None


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
        task = dict(discovered)
        task["attempts"] = int(previous.get("attempts", 0))
        task["metrics"] = previous.get("metrics")
        task["status"] = "pending"
        task["pid"] = None
        task["gpu"] = None
        task["started_at"] = previous.get("started_at")
        task["finished_at"] = previous.get("finished_at")
        task["returncode"] = previous.get("returncode")
        task["error"] = None
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
    return [
        sys.executable,
        str(PROJECT_ROOT / "evaluation.py"),
        "--checkpoint-root",
        task["checkpoint_root"],
        "--start-epoch",
        "5",
        "--end-epoch",
        "24",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        "0",
        "--cuda",
        "cuda:0",
        "--gpu-ids",
        "0",
        "--official-eval-version",
        "revised",
        "--official-eval-iou-backend",
        "cuda",
        "--official-eval-iou-mode",
        "all",
        "--official-detection-metrics-enabled",
        "true",
        "--custom-iou-range-eval-enabled",
        "false",
        "--nuscenes-style-eval-enabled",
        "false",
        "--loss-eval-enabled",
        "false",
        "--group-checkpoint-plot-best-only",
        "false",
        "--distance-range-eval-enabled",
        "true",
        "--distance-range-bins",
        DISTANCE_BIN_TEXT,
        "--max-detections",
        "64",
        "--heatmap-nms-kernel",
        "3",
        "--heatmap-score-mode",
        "peak_times_local_mean",
        "--yolox-nms-iou",
        "0.65",
        "--ap-score-thresh",
        "0.01",
        "--score-thresh",
        "0.3",
        "--eval-ignore-suppress-enabled",
        "false",
        "--table-txt-enabled",
        "true",
        "--table-output-base-dir",
        str(args.output_dir / "evaluation_reports"),
        "--evaluation-tensorboard-log-dir",
        str(args.output_dir / "tensorboard"),
        "--domain-comparison-enabled",
        "false",
        "--plot-output",
        "none",
    ]


def state_counts(state):
    counts = {status: 0 for status in ("pending", "running", "completed", "failed")}
    for task in state["tasks"].values():
        status = task.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def save_progress(state_path, state, all_rows, output_dir):
    state["updated_at"] = utc_now_text()
    atomic_write_json(state_path, state)
    write_tables(all_rows, state, output_dir)


def choose_gpu(gpu_use, gpus, max_per_gpu):
    available = [gpu for gpu in gpus if gpu_use[gpu] < max_per_gpu]
    if not available:
        return None
    return min(available, key=lambda gpu: (gpu_use[gpu], gpus.index(gpu)))


def run_queue(args, state_path, state, all_rows):
    gpu_use = {gpu: 0 for gpu in args.gpus}
    running = {}
    pending_ids = [
        task_id
        for task_id, task in state["tasks"].items()
        if task["status"] == "pending"
    ]
    reports_root = args.output_dir / "evaluation_reports"

    def stop_children(reason):
        if not running:
            return
        print(f"Stopping {len(running)} child evaluator(s): {reason}", flush=True)
        for process, _, _ in running.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 15.0
        for process, _, _ in running.values():
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for task_id, (process, log_file, _) in running.items():
            log_file.write(f"[{utc_now_text()}] launcher stopped child: {reason}\n")
            log_file.close()
            task = state["tasks"][task_id]
            task.update(
                {
                    "status": "pending",
                    "pid": None,
                    "gpu": None,
                    "returncode": process.returncode,
                    "finished_at": utc_now_text(),
                    "error": f"launcher stopped child: {reason}",
                }
            )
        save_progress(state_path, state, all_rows, args.output_dir)

    previous_handlers = {}

    def interrupt_handler(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signal_number] = signal.signal(
            signal_number, interrupt_handler
        )

    try:
        while pending_ids or running:
            while pending_ids and len(running) < args.max_workers:
                gpu = choose_gpu(gpu_use, args.gpus, args.max_per_gpu)
                if gpu is None:
                    break
                task_id = pending_ids.pop(0)
                task = state["tasks"][task_id]
                log_path = Path(task["log_path"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = log_path.open("a", encoding="utf-8", buffering=1)
                command = build_evaluation_command(task, args)
                log_file.write(
                    f"\n[{utc_now_text()}] launch physical cuda:{gpu}\n"
                    f"command: {' '.join(command)}\n"
                )
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                gpu_use[gpu] += 1
                task.update(
                    {
                        "status": "running",
                        "attempts": int(task.get("attempts", 0)) + 1,
                        "pid": process.pid,
                        "gpu": gpu,
                        "started_at": utc_now_text(),
                        "finished_at": None,
                        "returncode": None,
                        "error": None,
                    }
                )
                running[task_id] = (process, log_file, gpu)
                print(
                    f"Started {task_id} pid={process.pid} physical_cuda={gpu} "
                    f"({len(running)}/{args.max_workers} active)",
                    flush=True,
                )
                save_progress(state_path, state, all_rows, args.output_dir)

            finished_ids = []
            for task_id, (process, log_file, gpu) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                finished_ids.append(task_id)
                log_file.write(f"[{utc_now_text()}] returncode={returncode}\n")
                log_file.close()
                gpu_use[gpu] -= 1
                task = state["tasks"][task_id]
                task["pid"] = None
                task["gpu"] = None
                task["finished_at"] = utc_now_text()
                task["returncode"] = int(returncode)
                report = find_completed_report(task, reports_root)
                metrics = None if report is None else parse_average_report(report)
                if returncode == 0 and metrics is not None:
                    task["status"] = "completed"
                    task["metrics"] = metrics
                    task["report_path"] = str(report)
                    print(f"Completed {task_id}: {report}", flush=True)
                else:
                    task["status"] = "failed"
                    task["metrics"] = None
                    task["error"] = (
                        f"evaluation returncode={returncode}; "
                        f"valid report={'yes' if metrics is not None else 'no'}"
                    )
                    print(
                        f"FAILED {task_id}: {task['error']} (log={task['log_path']})",
                        flush=True,
                    )
                save_progress(state_path, state, all_rows, args.output_dir)
            for task_id in finished_ids:
                del running[task_id]

            if not pending_ids and not running:
                break
            time.sleep(args.poll_seconds)
    except BaseException as exc:
        stop_children(str(exc))
        raise
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)

    counts = state_counts(state)
    print(f"Queue finished: {counts}", flush=True)
    return 0 if counts.get("failed", 0) == 0 else 1


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
