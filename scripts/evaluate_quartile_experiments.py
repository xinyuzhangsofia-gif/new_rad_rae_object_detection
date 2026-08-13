#!/usr/bin/env python3
"""Re-evaluate rain/sleet experiments by GT-box distance quartile.

The already-trained canonical source/target checkpoints are discovered from
the original experiment queue state.  Results, reports, logs, TensorBoard
events, and restart state are isolated under ``experiments3`` by default.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import fcntl
import json
import math
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

from scripts import evaluate_distance_experiments as distance_launcher


SOURCE_EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
PROTECTED_DISTANCE_OUTPUT_DIR = PROJECT_ROOT / "experiments2"
SOURCE_REPORT_ROOT = PROJECT_ROOT / "evaluation_plots"
WEATHERS = ("rain", "sleet")
QUARTILES = ("q1", "q2", "q3", "q4")
METADATA_HEADERS = distance_launcher.METADATA_HEADERS
BRANCHES = distance_launcher.BRANCHES
STATE_VERSION = 1


def utc_now_text():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate completed rain/sleet domain-shift checkpoints with "
            "official AP split by the GT-box distance quartiles."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "experiments3"),
        help="Root for quartile tables, reports, logs, TensorBoard, and state.",
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
    protected_output_dirs = {
        SOURCE_EXPERIMENT_DIR.resolve(),
        PROTECTED_DISTANCE_OUTPUT_DIR.resolve(),
    }
    if args.output_dir in protected_output_dirs:
        raise ValueError(
            "Refusing to overwrite a protected experiment directory: "
            f"{args.output_dir}"
        )
    args.gpus = tuple(
        int(token.strip())
        for token in str(args.gpus).split(",")
        if token.strip()
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
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive.")
    return args


def read_experiment_rows(weather):
    return distance_launcher.read_experiment_rows(weather)


def row_identity(row):
    return distance_launcher.row_identity(row)


def discover_tasks(all_rows, output_dir):
    """Discover the same 54 canonical branches, redirected to experiments3."""
    return distance_launcher.discover_tasks(all_rows, output_dir)


def atomic_write_text(path, text):
    return distance_launcher.atomic_write_text(path, text)


def atomic_write_json(path, payload):
    return distance_launcher.atomic_write_json(path, payload)


def acquire_output_lock(output_dir):
    """Use a quartile-specific lock, independent from the distance launcher."""
    lock_path = Path(output_dir) / ".quartile_evaluation.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            f"Another quartile-evaluation launcher holds {lock_path}."
        ) from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} started_at={utc_now_text()}\n")
    lock_handle.flush()
    return lock_handle


def validate_physical_gpus(gpus):
    return distance_launcher.validate_physical_gpus(gpus)


def task_process_is_alive(task):
    return distance_launcher.task_process_is_alive(task)


def _parse_number(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in {"inf", "+inf", "infinity", "+infinity"}:
            return math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_value(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _normalize_quartile_entry(entry, fallback_tag=None):
    if not isinstance(entry, dict):
        return None
    lowered = {str(key).strip().lower(): value for key, value in entry.items()}
    tag = _first_value(lowered, ("quartile", "tag", "name", "id", "bin"))
    tag = fallback_tag if tag is None else str(tag).strip().lower()
    match = re.search(r"q[1-4]", str(tag).lower())
    if match is None:
        return None
    tag = match.group(0)
    lower = _parse_number(
        _first_value(
            lowered,
            ("lower_m", "lower", "min_m", "minimum_m", "range_min_m", "lo"),
        )
    )
    upper = _parse_number(
        _first_value(
            lowered,
            ("upper_m", "upper", "max_m", "maximum_m", "range_max_m", "hi"),
        )
    )
    count = _parse_number(
        _first_value(
            lowered,
            (
                "bbox_count",
                "n_bbox",
                "gt_bbox_count",
                "gt_count",
                "count",
                "num_boxes",
            ),
        )
    )
    if lower is None or upper is None or count is None:
        bounds = _first_value(lowered, ("bounds_m", "bounds", "range_m", "range"))
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            lower = lower if lower is not None else _parse_number(bounds[0])
            upper = upper if upper is not None else _parse_number(bounds[1])
    if lower is None or upper is None or count is None:
        return None
    rounded_count = int(round(count))
    if not math.isclose(count, rounded_count, abs_tol=1e-6) or rounded_count < 0:
        return None
    return {
        "tag": tag,
        "lower_m": lower,
        "upper_m": upper,
        "N_bbox": rounded_count,
    }


def _metadata_candidates(text):
    """Yield JSON/Python-like payloads from quartile metadata report lines."""
    pattern = re.compile(r"(?im)^\s*distance_quartile_bins\s*:\s*(.+?)\s*$")
    for match in pattern.finditer(text):
        payload = match.group(1).strip()
        for loader in (json.loads, ast.literal_eval):
            try:
                yield loader(payload)
                break
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue


def parse_quartile_metadata(text):
    """Parse one complete q1--q4 bounds/count definition from a report."""
    for candidate in _metadata_candidates(text):
        if isinstance(candidate, dict):
            entries = []
            for tag in QUARTILES:
                value = candidate.get(tag, candidate.get(tag.upper()))
                if value is not None:
                    entries.append(_normalize_quartile_entry(value, tag))
            if not entries and isinstance(candidate.get("bins"), list):
                entries = [
                    _normalize_quartile_entry(value)
                    for value in candidate["bins"]
                ]
        elif isinstance(candidate, list):
            entries = [_normalize_quartile_entry(value) for value in candidate]
        else:
            entries = []
        by_tag = {
            entry["tag"]: entry for entry in entries if entry is not None
        }
        if set(by_tag) == set(QUARTILES):
            return {tag: by_tag[tag] for tag in QUARTILES}

    # Fallback for flat metadata lines or per-epoch text rows.
    by_tag = {}
    for tag in QUARTILES:
        fields = {}
        aliases = {
            "lower_m": ("lower_m", "lower", "min_m"),
            "upper_m": ("upper_m", "upper", "max_m"),
            "N_bbox": ("bbox_count", "n_bbox", "gt_count", "count"),
        }
        for output_key, names in aliases.items():
            for name in names:
                match = re.search(
                    rf"(?im)\b(?:distance_quartile_)?{tag}_{name}\s*[:=]\s*"
                    rf"([+-]?(?:\d+(?:\.\d*)?|\.\d+|inf(?:inity)?))\b",
                    text,
                )
                if match:
                    fields[output_key] = _parse_number(match.group(1))
                    break
        if len(fields) == 3:
            count = int(round(fields["N_bbox"]))
            if math.isclose(fields["N_bbox"], count, abs_tol=1e-6):
                by_tag[tag] = {
                    "tag": tag,
                    "lower_m": fields["lower_m"],
                    "upper_m": fields["upper_m"],
                    "N_bbox": count,
                }
    return by_tag if set(by_tag) == set(QUARTILES) else None


def parse_average_report(report_path):
    report_path = Path(report_path)
    if not report_path.is_file():
        return None
    text = report_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"average_AP_epoch_5_to_24 \(epochs_used=(\d+)\): ([^\n]+)", text
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
    for tag in QUARTILES:
        required.update(
            (f"bev@0.3_quartile_{tag}", f"3d@0.3_quartile_{tag}")
        )
    metadata = parse_quartile_metadata(text)
    if not required.issubset(values) or metadata is None:
        return None
    metrics = {"all": {"BEV": values["bev@0.3"], "3D": values["3d@0.3"]}}
    for tag in QUARTILES:
        metrics[tag] = {
            "BEV": values[f"bev@0.3_quartile_{tag}"],
            "3D": values[f"3d@0.3_quartile_{tag}"],
            "lower_m": metadata[tag]["lower_m"],
            "upper_m": metadata[tag]["upper_m"],
            "N_bbox": metadata[tag]["N_bbox"],
        }
    return metrics


def report_matches_task(report_path, task):
    report_path = Path(report_path)
    if parse_average_report(report_path) is None:
        return False
    metadata = {}
    for line in report_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.strip():
            break
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    try:
        root = Path(metadata["checkpoint_root"]).expanduser().resolve()
    except (KeyError, OSError):
        return False
    return (
        root == Path(task["checkpoint_root"]).resolve()
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
    return next(
        (candidate for candidate in candidates if report_matches_task(candidate, task)),
        None,
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
                f"(pid={previous.get('pid')})."
            )
        task = dict(discovered)
        task.update(
            {
                "attempts": int(previous.get("attempts", 0)),
                "metrics": previous.get("metrics"),
                "status": "pending",
                "pid": None,
                "gpu": None,
                "started_at": previous.get("started_at"),
                "finished_at": previous.get("finished_at"),
                "returncode": previous.get("returncode"),
                "error": None,
            }
        )
        # Recover any valid finished report, even if the previous launcher was
        # killed after the evaluator wrote its report but before state changed
        # from ``running`` to ``completed``.
        report = find_completed_report(task, reports_root)
        metrics = None if report is None else parse_average_report(report)
        if metrics is not None:
            task.update(
                status="completed", metrics=metrics, report_path=str(report)
            )
        tasks[task_id] = task
    state = {
        "version": STATE_VERSION,
        "updated_at": utc_now_text(),
        "settings": {
            "epochs": [5, 24],
            "quartile_basis": "ground_truth_bbox_radar_center_distance_rank",
            "distance_definition": "sqrt(x^2+y^2+z^2)",
            "quartiles": list(QUARTILES),
        },
        "tasks": tasks,
    }
    atomic_write_json(state_path, state)
    return state


def metric_headers():
    headers = list(METADATA_HEADERS)
    headers.extend(("BEV_src", "3D_src", "BEV_tgt", "3D_tgt", "TD_BEV", "TD_3D"))
    for tag in QUARTILES:
        headers.extend(
            (
                f"range_m_{tag}",
                f"N_bbox_{tag}",
                f"BEV_src_{tag}",
                f"3D_src_{tag}",
                f"BEV_tgt_{tag}",
                f"3D_tgt_{tag}",
                f"TD_BEV_{tag}",
                f"TD_3D_{tag}",
            )
        )
    return headers


def task_lookup(state):
    return {
        (task["weather"], task["group"], int(task["seed"]), task["branch"]): task
        for task in state["tasks"].values()
    }


def format_metric(value):
    return "" if value is None else f"{float(value):.4f}"


def format_bound(value):
    return "inf" if math.isinf(float(value)) else f"{float(value):.4f}"


def _validate_pair_metadata(source, target, tag):
    if source is None and target is None:
        return None
    selected = source if source is not None else target
    if source is not None and target is not None:
        same_count = int(source["N_bbox"]) == int(target["N_bbox"])
        same_bounds = all(
            math.isclose(
                float(source[key]), float(target[key]), rel_tol=0.0, abs_tol=1e-6
            )
            for key in ("lower_m", "upper_m")
        )
        if not same_count or not same_bounds:
            raise ValueError(
                f"Source/target quartile metadata mismatch for {tag}: "
                f"source={source}, target={target}"
            )
    return selected


def build_table_matrix(weather, experiment_rows, state):
    headers = metric_headers()
    lookup = task_lookup(state)
    output_rows = []
    td_values = {block: [] for block in ("all",) + QUARTILES}
    for metadata in experiment_rows:
        row = [str(metadata[header]) for header in METADATA_HEADERS]
        identity = row_identity(metadata)
        if identity is None:
            row.extend([""] * (len(headers) - len(row)))
            output_rows.append(row)
            continue
        group, seed = identity
        source_task = lookup[(weather, group, seed, "source")]
        target_task = lookup[(weather, group, seed, "target")]
        source = source_task.get("metrics") if source_task.get("status") == "completed" else None
        target = target_task.get("metrics") if target_task.get("status") == "completed" else None

        source_all = None if source is None else source.get("all")
        target_all = None if target is None else target.get("all")
        source_bev = None if source_all is None else source_all.get("BEV")
        source_3d = None if source_all is None else source_all.get("3D")
        target_bev = None if target_all is None else target_all.get("BEV")
        target_3d = None if target_all is None else target_all.get("3D")
        td_bev = None if source_bev is None or target_bev is None else target_bev - source_bev
        td_3d = None if source_3d is None or target_3d is None else target_3d - source_3d
        row.extend(format_metric(value) for value in (source_bev, source_3d, target_bev, target_3d, td_bev, td_3d))
        if td_bev is not None and td_3d is not None:
            td_values["all"].append((td_bev, td_3d))

        for tag in QUARTILES:
            source_block = None if source is None else source.get(tag)
            target_block = None if target is None else target.get(tag)
            metadata_block = _validate_pair_metadata(source_block, target_block, tag)
            range_text = (
                ""
                if metadata_block is None
                else f"[{format_bound(metadata_block['lower_m'])},{format_bound(metadata_block['upper_m'])})"
            )
            bbox_count = "" if metadata_block is None else str(metadata_block["N_bbox"])
            source_bev = None if source_block is None else source_block.get("BEV")
            source_3d = None if source_block is None else source_block.get("3D")
            target_bev = None if target_block is None else target_block.get("BEV")
            target_3d = None if target_block is None else target_block.get("3D")
            td_bev = None if source_bev is None or target_bev is None else target_bev - source_bev
            td_3d = None if source_3d is None or target_3d is None else target_3d - source_3d
            row.extend((range_text, bbox_count))
            row.extend(format_metric(value) for value in (source_bev, source_3d, target_bev, target_3d, td_bev, td_3d))
            if td_bev is not None and td_3d is not None:
                td_values[tag].append((td_bev, td_3d))
        output_rows.append(row)

    average_row = [f"average({len(td_values['all'])})"] + ["-"] * 5
    values = td_values["all"]
    average_row.extend(
        ("", "", "", "", format_metric(sum(x for x, _ in values) / len(values)) if values else "", format_metric(sum(y for _, y in values) / len(values)) if values else "")
    )
    for tag in QUARTILES:
        values = td_values[tag]
        average_row.extend(
            ("", "", "", "", "", "", format_metric(sum(x for x, _ in values) / len(values)) if values else "", format_metric(sum(y for _, y in values) / len(values)) if values else "")
        )
    output_rows.append(average_row)
    return [headers] + output_rows


def format_table(matrix):
    return distance_launcher.format_table(matrix)


def write_tables(all_rows, state, output_dir):
    for weather in WEATHERS:
        matrix = build_table_matrix(weather, all_rows[weather], state)
        atomic_write_text(output_dir / f"{weather}_experiments.txt", format_table(matrix))


def write_readme(output_dir):
    atomic_write_text(
        output_dir / "README.txt",
        "Rain and sleet checkpoint re-evaluation with official K-Radar AP@IoU 0.3 "
        "split by GT-box distance rank quartiles.\n\n"
        "Distance is sqrt(x^2+y^2+z^2) at each GT box's radar-frame center. "
        "Quartile boundaries and N_bbox are derived only from ground-truth boxes; "
        "predictions are filtered using those fixed bounds. Every AP is the mean "
        "over epochs 5-24 inclusive. TD = target-trained AP - source-trained AP.\n",
    )


def build_evaluation_command(task, args):
    return [
        sys.executable,
        str(PROJECT_ROOT / "evaluation.py"),
        "--checkpoint-root", task["checkpoint_root"],
        "--start-epoch", "5", "--end-epoch", "24",
        "--batch-size", str(args.batch_size), "--num-workers", "0",
        "--cuda", "cuda:0", "--gpu-ids", "0",
        "--official-eval-version", "revised",
        "--official-eval-iou-backend", "cuda",
        "--official-eval-iou-mode", "all",
        "--official-detection-metrics-enabled", "true",
        "--custom-iou-range-eval-enabled", "false",
        "--nuscenes-style-eval-enabled", "false",
        "--loss-eval-enabled", "false",
        "--group-checkpoint-plot-best-only", "false",
        "--distance-range-eval-enabled", "false",
        "--distance-quartile-eval-enabled", "true",
        "--max-detections", "64", "--heatmap-nms-kernel", "3",
        "--heatmap-score-mode", "peak_times_local_mean",
        "--yolox-nms-iou", "0.65", "--ap-score-thresh", "0.01",
        "--score-thresh", "0.3", "--eval-ignore-suppress-enabled", "false",
        "--table-txt-enabled", "true",
        "--table-output-base-dir", str(args.output_dir / "evaluation_reports"),
        "--evaluation-tensorboard-log-dir", str(args.output_dir / "tensorboard"),
        "--domain-comparison-enabled", "false", "--plot-output", "none",
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
    return distance_launcher.choose_gpu(gpu_use, gpus, max_per_gpu)


def run_queue(args, state_path, state, all_rows):
    gpu_use = {gpu: 0 for gpu in args.gpus}
    running = {}
    pending_ids = [key for key, task in state["tasks"].items() if task["status"] == "pending"]
    reports_root = args.output_dir / "evaluation_reports"

    def stop_children(reason):
        for process, _, _ in running.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 15.0
        for process, _, _ in running.values():
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for task_id, (process, log_file, _) in running.items():
            log_file.write(f"[{utc_now_text()}] launcher stopped child: {reason}\n")
            log_file.close()
            state["tasks"][task_id].update(status="pending", pid=None, gpu=None, returncode=process.returncode, finished_at=utc_now_text(), error=f"launcher stopped child: {reason}")
        save_progress(state_path, state, all_rows, args.output_dir)

    old_handlers = {}
    def interrupt_handler(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    for number in (signal.SIGINT, signal.SIGTERM):
        old_handlers[number] = signal.signal(number, interrupt_handler)
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
                log_file.write(f"\n[{utc_now_text()}] launch physical cuda:{gpu}\ncommand: {' '.join(command)}\n")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=environment, stdout=log_file, stderr=subprocess.STDOUT)
                gpu_use[gpu] += 1
                task.update(status="running", attempts=int(task.get("attempts", 0)) + 1, pid=process.pid, gpu=gpu, started_at=utc_now_text(), finished_at=None, returncode=None, error=None)
                running[task_id] = (process, log_file, gpu)
                print(f"Started {task_id} pid={process.pid} physical_cuda={gpu} ({len(running)}/{args.max_workers} active)", flush=True)
                save_progress(state_path, state, all_rows, args.output_dir)

            for task_id, (process, log_file, gpu) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_file.write(f"[{utc_now_text()}] returncode={returncode}\n")
                log_file.close()
                gpu_use[gpu] -= 1
                task = state["tasks"][task_id]
                task.update(pid=None, gpu=None, finished_at=utc_now_text(), returncode=int(returncode))
                report = find_completed_report(task, reports_root)
                metrics = None if report is None else parse_average_report(report)
                if returncode == 0 and metrics is not None:
                    task.update(status="completed", metrics=metrics, report_path=str(report), error=None)
                    print(f"Completed {task_id}: {report}", flush=True)
                else:
                    task.update(status="failed", metrics=None, error=f"evaluation returncode={returncode}; valid report={'yes' if metrics is not None else 'no'}")
                    print(f"FAILED {task_id}: {task['error']} (log={task['log_path']})", flush=True)
                del running[task_id]
                save_progress(state_path, state, all_rows, args.output_dir)
            if pending_ids or running:
                time.sleep(args.poll_seconds)
    except BaseException as exc:
        stop_children(str(exc))
        raise
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
    counts = state_counts(state)
    print(f"Queue finished: {counts}", flush=True)
    return 0 if counts.get("failed", 0) == 0 else 1


def main(argv=None):
    args = parse_args(argv)
    for child in ("logs", "evaluation_reports", "tensorboard"):
        (args.output_dir / child).mkdir(parents=True, exist_ok=True)
    lock_handle = acquire_output_lock(args.output_dir)
    try:
        all_rows = {weather: read_experiment_rows(weather) for weather in WEATHERS}
        discovered = discover_tasks(all_rows, args.output_dir)
        state_path = args.output_dir / "quartile_evaluation_state.json"
        state = initialize_state(state_path, discovered, args.output_dir / "evaluation_reports")
        write_readme(args.output_dir)
        write_tables(all_rows, state, args.output_dir)
        counts = state_counts(state)
        print(f"Validated {len(state['tasks'])} tasks; state={state_path}; status={counts}", flush=True)
        if args.dry_run:
            first = next((task for task in state["tasks"].values() if task["status"] == "pending"), None)
            if first is not None:
                print("Example command:", " ".join(build_evaluation_command(first, args)), flush=True)
            print("Dry run complete; no evaluation processes were launched.", flush=True)
            return 0
        validate_physical_gpus(args.gpus)
        return run_queue(args, state_path, state, all_rows)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
