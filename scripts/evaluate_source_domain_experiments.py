#!/usr/bin/env python3
"""Evaluate source checkpoints on controlled normal-domain test sets.

This launcher is intentionally isolated from the training queue and from the
running ``experiments3`` quartile evaluation.  It pairs each canonical source
checkpoint's existing adverse-weather report with one new evaluation of the
same checkpoint on a controlled normal test.  The reported source-domain
difference is::

    SD = AP(normal controlled test) - AP(adverse-weather target test)

Positive SD therefore means performance was lost under adverse weather.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import fcntl
import hashlib
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
from scripts import evaluate_quartile_experiments as quartile_launcher


SOURCE_EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
DEFAULT_EXPERIMENTS3_DIR = PROJECT_ROOT / "experiments3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments4"
WEATHERS = ("heavy_snow", "light_snow", "overcast", "rain", "sleet")
QUARTILES = ("q1", "q2", "q3", "q4")
METADATA_HEADERS = distance_launcher.METADATA_HEADERS
EXPECTED_EXPERIMENTS3_TASKS = 54
EXPECTED_SD_TASKS = 80
EXPECTED_SD_TASKS_BY_WEATHER = {
    "heavy_snow": 11,
    "light_snow": 24,
    "overcast": 18,
    "rain": 15,
    "sleet": 12,
}
WEATHER_REFERENCE_WEATHERS = ("heavy_snow", "light_snow", "overcast")
EXPECTED_NEW_WEATHER_REFERENCE_TASKS = 53
STATE_VERSION = 2


def now_text():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all Heavy Snow, Light Snow, Overcast, Rain, and Sleet "
            "source-trained checkpoints on controlled normal tests and compute "
            "SD = normal AP - weather AP."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root for SD tables, reports, logs, TensorBoard, controls, and state.",
    )
    parser.add_argument(
        "--experiments3-dir",
        default=str(DEFAULT_EXPERIMENTS3_DIR),
        help="Completed target-weather quartile experiment root.",
    )
    parser.add_argument(
        "--control-spec-index",
        default=None,
        help="Control index JSON; default: OUTPUT/control_specs/index.json.",
    )
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--max-workers", type=int, default=9)
    parser.add_argument("--max-per-gpu", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--wait-for-quartile-completion",
        action="store_true",
        help=(
            "Before doing any GPU work, poll experiments3 until exactly "
            "54/54 tasks are completed. A failed experiments3 task aborts "
            "the wait; retry it with the experiments3 launcher first."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Discover all 80 source-domain tasks plus the 53 missing weather "
            "reference tasks without launching evaluation children."
        ),
    )
    args = parser.parse_args(argv)
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.experiments3_dir = Path(args.experiments3_dir).expanduser().resolve()
    args.control_spec_index = Path(
        args.control_spec_index
        or args.output_dir / "control_specs" / "index.json"
    ).expanduser().resolve()

    protected = {
        SOURCE_EXPERIMENT_DIR.resolve(),
        (PROJECT_ROOT / "experiments2").resolve(),
        args.experiments3_dir,
    }
    if args.output_dir in protected:
        raise ValueError(
            f"Refusing to overwrite a protected experiment directory: {args.output_dir}"
        )

    try:
        args.gpus = tuple(
            int(token.strip())
            for token in str(args.gpus).split(",")
            if token.strip()
        )
    except ValueError as exc:
        raise ValueError(f"Invalid GPU list: {args.gpus!r}") from exc
    if not args.gpus or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("At least one non-negative physical GPU id is required.")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError(f"GPU ids must be unique, got {args.gpus!r}.")
    if args.max_workers <= 0 or args.max_per_gpu <= 0:
        raise ValueError("max-workers and max-per-gpu must be positive.")
    capacity = len(args.gpus) * args.max_per_gpu
    if args.max_workers > capacity:
        raise ValueError(
            f"max-workers exceeds GPU capacity: {args.max_workers} > {capacity}."
        )
    if args.batch_size <= 0 or args.poll_seconds <= 0.0:
        raise ValueError("batch-size and poll-seconds must be positive.")
    return args


def atomic_write_text(path, text):
    distance_launcher.atomic_write_text(path, text)


def atomic_write_json(path, payload):
    distance_launcher.atomic_write_json(path, payload)


def acquire_output_lock(output_dir):
    lock_path = Path(output_dir) / ".sd_evaluation.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"Another SD launcher holds {lock_path}.") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} started_at={now_text()}\n")
    lock_handle.flush()
    return lock_handle


def read_experiment_rows(weather):
    return distance_launcher.read_experiment_rows(weather)


def row_identity(row):
    return distance_launcher.row_identity(row)


def load_completed_source_checkpoint_records(weather, experiment_rows):
    """Resolve one completed source branch without requiring target completion."""
    state_path = distance_launcher.queue_state_path(weather)
    payload = _load_json(state_path)
    expected = {
        identity for identity in map(row_identity, experiment_rows)
        if identity is not None
    }
    candidates = {}
    for record in payload.get("tasks", {}).values():
        if str(record.get("branch", "")).lower() != "source":
            continue
        if str(record.get("status", "")).lower() != "completed":
            continue
        try:
            identity = (str(record["group"]), int(record["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        if identity not in expected:
            continue
        checkpoint_root = Path(str(record.get("checkpoint_root", ""))).expanduser()
        if not checkpoint_root.is_dir():
            continue
        previous = candidates.get(identity)
        if previous is None or str(record.get("updated_at", "")) > str(
            previous.get("updated_at", "")
        ):
            candidates[identity] = dict(record)
    missing = sorted(expected - set(candidates))
    if missing:
        raise RuntimeError(
            f"{weather}: missing completed source checkpoint records: {missing!r}"
        )
    return candidates


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_input_path(value, index_path):
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()
    index_candidate = (Path(index_path).parent / path).resolve()
    if project_candidate.exists() or not index_candidate.exists():
        return project_candidate
    return index_candidate


def _first(mapping, *keys):
    if not isinstance(mapping, dict):
        return None
    lowered = {str(key).strip().lower(): value for key, value in mapping.items()}
    for key in keys:
        value = lowered.get(str(key).lower())
        if value is not None:
            return value
    return None


def _nested_first(mappings, *keys):
    for mapping in mappings:
        value = _first(mapping, *keys)
        if value is not None:
            return value
    return None


def _integer(value):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    rounded = int(round(number))
    return rounded if math.isclose(number, rounded, abs_tol=1e-6) else None


def _sequence_tuple(value):
    if value in (None, "", (), []):
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        if parsed is not None and parsed != value:
            return _sequence_tuple(parsed)
        return tuple(int(token) for token in re.findall(r"\d+", value))
    if isinstance(value, (list, tuple, set)):
        return tuple(int(item) for item in value)
    return (int(value),)


def _metadata_header(report_path):
    metadata = {}
    for line in Path(report_path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.strip():
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _paths_equal(value, expected):
    if value in (None, "") or expected in (None, ""):
        return value in (None, "") and expected in (None, "")
    value_path = Path(str(value)).expanduser()
    if not value_path.is_absolute():
        value_path = PROJECT_ROOT / value_path
    return value_path.resolve() == Path(expected).expanduser().resolve()


def _quartile_bounds(metrics):
    if not isinstance(metrics, dict):
        return None
    result = {}
    for tag in QUARTILES:
        block = metrics.get(tag)
        if not isinstance(block, dict):
            return None
        try:
            result[tag] = {
                "tag": tag,
                "lower_m": float(block["lower_m"]),
                "upper_m": float(block["upper_m"]),
                "N_bbox": int(block["N_bbox"]),
            }
        except (KeyError, TypeError, ValueError):
            return None
    return result


def _bounds_match(left, right, tolerance=1e-8):
    if left is None or right is None:
        return False
    for tag in QUARTILES:
        if tag not in left or tag not in right:
            return False
        for key in ("lower_m", "upper_m"):
            a = float(left[tag][key])
            b = float(right[tag][key])
            if math.isinf(a) or math.isinf(b):
                if not (math.isinf(a) and math.isinf(b)):
                    return False
            elif not math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance):
                return False
    return True


def _normalize_external_bins(value):
    if value in (None, "", [], {}):
        return None
    payload = value
    if isinstance(payload, dict):
        payload = _first(payload, "distance_quartile_bins", "quartile_bins", "bins")
    if isinstance(payload, dict):
        payload = [payload.get(tag) or payload.get(tag.upper()) for tag in QUARTILES]
    if not isinstance(payload, (list, tuple)) or len(payload) != 4:
        return None
    result = {}
    for tag, item in zip(QUARTILES, payload):
        if not isinstance(item, dict):
            return None
        lower = _first(item, "lower_m", "lower", "min_m")
        upper = _first(item, "upper_m", "upper", "max_m")
        count = _first(item, "bbox_count", "n_bbox", "num_gt", "count")
        try:
            upper_value = (
                math.inf
                if str(upper).strip().lower() in {"inf", "+inf", "infinity"}
                else float(upper)
            )
            result[tag] = {
                "tag": tag,
                "lower_m": float(lower),
                "upper_m": upper_value,
                "N_bbox": None if count is None else int(count),
            }
        except (TypeError, ValueError):
            return None
    return result


def _normalize_quartile_counts(value):
    if value is None:
        return {}
    if isinstance(value, (list, tuple)):
        return {
            tag: int(count)
            for tag, count in zip(QUARTILES, value)
            if _integer(count) is not None
        }
    if not isinstance(value, dict):
        return {}
    result = {}
    for tag in QUARTILES:
        item = value.get(tag, value.get(tag.upper()))
        if isinstance(item, dict):
            item = _first(item, "count", "n_bbox", "bbox_count", "num_gt")
        count = _integer(item)
        if count is not None:
            result[tag] = count
    return result


def load_control_specs(index_path):
    """Load target controls while tolerating direct or nested stats fields."""
    index_path = Path(index_path)
    if not index_path.is_file():
        return {}, [f"missing control index: {index_path}"]
    try:
        payload = _load_json(index_path)
    except (OSError, ValueError) as exc:
        return {}, [f"invalid control index {index_path}: {exc}"]
    raw_controls = payload.get("controls", payload)
    if isinstance(raw_controls, list):
        entries = [(None, entry) for entry in raw_controls]
    elif isinstance(raw_controls, dict):
        entries = list(raw_controls.items())
    else:
        return {}, [f"control index has no controls mapping/list: {index_path}"]

    controls = {}
    errors = []
    for raw_key, raw_spec in entries:
        if not isinstance(raw_spec, dict):
            errors.append(f"control {raw_key!r} is not an object")
            continue
        target_sequences = _sequence_tuple(
            _first(
                raw_spec,
                "target_sequences",
                "target_sequence",
                "target_seq",
                "test_sequences",
                "test_sequence",
                "test_seq",
            )
        )
        if not target_sequences and raw_key is not None:
            matches = re.findall(r"\d+", str(raw_key))
            target_sequences = tuple(int(value) for value in matches)
        if not target_sequences:
            errors.append(f"control {raw_key!r} has no target sequence")
            continue

        stats_path = _resolve_input_path(
            _first(raw_spec, "stats_path", "statistics_path"), index_path
        )
        stats = {}
        if stats_path is not None and stats_path.is_file():
            try:
                stats = _load_json(stats_path)
            except (OSError, ValueError) as exc:
                errors.append(f"invalid stats for target {target_sequences}: {exc}")
        source_stats = stats.get("source", {}) if isinstance(stats, dict) else {}
        target_stats = stats.get("target", {}) if isinstance(stats, dict) else {}
        validation_stats = stats.get("validation", {}) if isinstance(stats, dict) else {}
        weather_group = str(
            _nested_first((raw_spec, stats, target_stats), "weather_group") or ""
        ).strip().lower()
        if not weather_group:
            errors.append(f"control {raw_key!r} has no weather_group")
            continue
        control_key = (weather_group, tuple(target_sequences))
        if control_key in controls:
            errors.append(f"duplicate control for {control_key!r}")
            continue
        sources = (raw_spec, source_stats, stats)
        targets = (raw_spec, target_stats, stats)

        source_sequences = _sequence_tuple(
            _nested_first(
                sources,
                "source_sequence",
                "source_sequences",
                "normal_sequence",
                "normal_sequences",
                "eval_val_sequences",
            )
        )
        manifest_path = _resolve_input_path(
            _nested_first(
                sources,
                "manifest_path",
                "frame_manifest_path",
                "eval_frame_manifest_path",
                "test_manifest_path",
            ),
            index_path,
        )
        override_path = _resolve_input_path(
            _nested_first(
                sources,
                "object_override_path",
                "override_path",
                "gt_object_ignore_override_path",
                "eval_gt_object_ignore_override_path",
            ),
            index_path,
        )
        selected_frames_value = _nested_first(
            sources,
            "source_frames_selected",
            "selected_frame_count",
            "eligible_frame_count",
            "num_frames",
        )
        if selected_frames_value is None:
            selected_names = _first(source_stats, "selected_frame_names", "frame_names")
            selected_frames_value = len(selected_names) if isinstance(selected_names, list) else None
        source_frames = _integer(selected_frames_value)
        target_count = _integer(
            _nested_first(
                targets,
                "target_eligible_bbox_count",
                "eligible_bbox_count",
                "bbox_count",
                "n_bbox",
            )
        )
        source_count = _integer(
            _nested_first(
                sources,
                "source_kept_bbox_count",
                "eligible_bbox_after_control",
                "kept_bbox_count",
                "n_bbox_after",
            )
        )
        masked_source_count = _integer(
            _nested_first(
                sources,
                "source_masked_bbox_count",
                "masked_eligible_bbox_count",
                "masked_bbox_count",
            )
        )
        expected_neutral_count = _integer(
            _nested_first(
                (raw_spec, source_stats, stats),
                "expected_override_object_count",
            )
        )
        masked_count = (
            expected_neutral_count
            if expected_neutral_count is not None
            else masked_source_count
        )
        deficit = _integer(
            _nested_first(
                sources,
                "source_bbox_deficit",
                "deficit_bbox_count",
                "bbox_deficit",
            )
        )
        target_bins = _normalize_external_bins(
            _nested_first(
                targets,
                "target_quartile_bins",
                "quartile_bins",
                "distance_quartile_bins",
            )
        )
        target_q_counts = _normalize_quartile_counts(
            _nested_first(targets, "quartile_counts", "target_quartile_counts")
        )
        source_q_counts = _normalize_quartile_counts(
            _nested_first(
                sources,
                "quartile_counts_after",
                "source_quartile_counts",
                "quartile_counts",
            )
        )
        supplied_status = str(
            _nested_first((raw_spec, validation_stats), "status", "control_status")
            or ""
        ).strip()
        normal_test_id = str(
            _first(raw_spec, "normal_test_id", "control_id", "id") or ""
        ).strip()
        if not normal_test_id:
            source_tag = "_".join(str(value) for value in source_sequences) or "unknown"
            target_tag = "_".join(str(value) for value in target_sequences)
            normal_test_id = f"normal_seq{source_tag}_target{target_tag}"

        file_errors = []
        if not source_sequences:
            file_errors.append("missing source sequence")
        if manifest_path is None or not manifest_path.is_file():
            file_errors.append(f"missing manifest: {manifest_path}")
        if override_path is None or not override_path.is_file():
            file_errors.append(f"missing override: {override_path}")
        if source_frames is None or source_frames <= 0:
            file_errors.append("missing/invalid source frame count")
        if source_count is None or source_count < 0:
            file_errors.append("missing/invalid controlled source bbox count")
        if target_count is None or target_count < 0:
            file_errors.append("missing/invalid target bbox count")
        if masked_count is None or masked_count < 0:
            file_errors.append("missing/invalid expected neutral GT count")
        if (
            expected_neutral_count is not None
            and masked_source_count is not None
            and expected_neutral_count != masked_source_count
        ):
            file_errors.append(
                "source masked bbox count disagrees with expected override count: "
                f"{masked_source_count} != {expected_neutral_count}"
            )

        if deficit is None and target_count is not None and source_count is not None:
            deficit = max(0, target_count - source_count)
        exact = bool(
            not file_errors
            and deficit == 0
            and source_count == target_count
        )
        supplied_lower = supplied_status.lower()
        explicitly_invalid = any(
            token in supplied_lower for token in ("invalid", "failed", "error")
        )
        if explicitly_invalid:
            file_errors.append(f"control status is {supplied_status!r}")
        if file_errors:
            status = "invalid: " + "; ".join(file_errors)
        elif exact:
            status = "exact"
        elif deficit is not None and deficit > 0:
            status = f"valid_deficit({deficit})"
        else:
            status = supplied_status or "valid"

        controls[control_key] = {
            "weather_group": weather_group,
            "target_sequences": tuple(target_sequences),
            "target_sequence": (
                target_sequences[0] if len(target_sequences) == 1 else None
            ),
            "source_sequences": source_sequences,
            "normal_test_id": normal_test_id,
            "manifest_path": None if manifest_path is None else str(manifest_path),
            "override_path": None if override_path is None else str(override_path),
            "stats_path": None if stats_path is None else str(stats_path),
            "status": status,
            "valid": not file_errors,
            "exact": exact,
            "N_frame_normal": source_frames,
            "N_bbox_normal": source_count,
            "N_bbox_weather": target_count,
            "N_bbox_masked": masked_count,
            "bbox_deficit": deficit,
            "target_quartile_bins": target_bins,
            "target_quartile_counts": target_q_counts,
            "source_quartile_counts": source_q_counts,
            "manifest_sha256": _file_sha256(manifest_path) if manifest_path else None,
            "override_sha256": _file_sha256(override_path) if override_path else None,
            "stats_sha256": _file_sha256(stats_path) if stats_path else None,
            "errors": file_errors,
        }
    return controls, errors


def _epoch_checkpoint_identity(checkpoint_root):
    root = Path(checkpoint_root)
    if not root.is_dir():
        return None, [f"missing checkpoint root: {root}"]
    selected = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = re.search(r"(?i)epoch[_-]?0*(\d+)", path.name)
        if match:
            epoch = int(match.group(1))
            if 5 <= epoch <= 24:
                selected.setdefault(epoch, []).append(path)
    errors = []
    for epoch in range(5, 25):
        matches = selected.get(epoch, [])
        if len(matches) != 1:
            errors.append(
                f"checkpoint epoch {epoch} has {len(matches)} matching files under {root}"
            )
    if errors:
        return None, errors
    rows = []
    for epoch in range(5, 25):
        path = selected[epoch][0]
        stat = path.stat()
        rows.append((epoch, path.name, stat.st_size, stat.st_mtime_ns))
    return _stable_hash({"root": str(root.resolve()), "epochs": rows}), []


def _load_experiments3_state(experiments3_dir):
    state_path = Path(experiments3_dir) / "quartile_evaluation_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing experiments3 state: {state_path}")
    payload = _load_json(state_path)
    tasks = payload.get("tasks", {})
    if not isinstance(tasks, dict) or len(tasks) != EXPECTED_EXPERIMENTS3_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_EXPERIMENTS3_TASKS} experiments3 tasks, "
            f"found {len(tasks) if isinstance(tasks, dict) else 'invalid state'}."
        )
    return state_path, payload


def experiments3_completion_errors(experiments3_state):
    errors = []
    tasks = experiments3_state.get("tasks", {})
    completed = sum(
        str(task.get("status", "")).lower() == "completed"
        for task in tasks.values()
    )
    if len(tasks) != EXPECTED_EXPERIMENTS3_TASKS or completed != EXPECTED_EXPERIMENTS3_TASKS:
        errors.append(
            f"experiments3 is incomplete: {completed}/{EXPECTED_EXPERIMENTS3_TASKS} completed"
        )
    for task_id, task in tasks.items():
        if str(task.get("status", "")).lower() != "completed":
            continue
        report_path = task.get("report_path")
        if report_path is None or not quartile_launcher.report_matches_task(
            report_path, task
        ):
            errors.append(f"invalid completed experiments3 report: {task_id}")
    return errors


def experiments3_progress(experiments3_state):
    counts = {}
    tasks = experiments3_state.get("tasks", {})
    for task in tasks.values():
        status = str(task.get("status", "unknown")).lower()
        counts[status] = counts.get(status, 0) + 1
    return counts


def wait_for_experiments3(experiments3_dir, poll_seconds):
    """Wait without launching children until quartile evaluation is complete."""
    last_summary = None
    while True:
        _, state = _load_experiments3_state(experiments3_dir)
        counts = experiments3_progress(state)
        failed = counts.get("failed", 0)
        if failed:
            failed_ids = [
                task_id
                for task_id, task in state.get("tasks", {}).items()
                if str(task.get("status", "")).lower() == "failed"
            ]
            raise RuntimeError(
                "experiments3 has failed task(s); retry them with the quartile "
                f"launcher before SD evaluation: {failed_ids!r}"
            )
        completed = counts.get("completed", 0)
        if completed == EXPECTED_EXPERIMENTS3_TASKS:
            errors = experiments3_completion_errors(state)
            if errors:
                raise RuntimeError(
                    "experiments3 reports failed final validation: "
                    + "; ".join(errors[:10])
                )
            print(
                f"experiments3 gate ready: {completed}/{EXPECTED_EXPERIMENTS3_TASKS} completed",
                flush=True,
            )
            return state
        summary = ", ".join(
            f"{key}={value}" for key, value in sorted(counts.items())
        )
        if summary != last_summary:
            print(
                f"Waiting for experiments3 ({summary}); no SD GPU work started.",
                flush=True,
            )
            last_summary = summary
        time.sleep(poll_seconds)


def _test_set_tag(sequences):
    values = _sequence_tuple(sequences)
    if not values:
        raise ValueError(f"Empty test sequence set: {sequences!r}")
    return "_".join(str(value) for value in values)


def discover_weather_reference_tasks(all_rows, output_dir, controls):
    """Discover source-checkpoint target-weather quartile prerequisites."""
    tasks = {}
    for weather in WEATHER_REFERENCE_WEATHERS:
        records = load_completed_source_checkpoint_records(
            weather, all_rows[weather]
        )
        for row in all_rows[weather]:
            identity = row_identity(row)
            if identity is None:
                continue
            group, seed = identity
            target_sequences = _sequence_tuple(row["test_seq"])
            control = controls.get((weather, target_sequences))
            if control is None or not control.get("valid"):
                raise RuntimeError(
                    f"Missing valid control for {(weather, target_sequences)!r}."
                )
            record = records[(group, seed)]
            checkpoint_root = Path(record["checkpoint_root"]).resolve()
            checkpoint_identity, checkpoint_errors = _epoch_checkpoint_identity(
                checkpoint_root
            )
            if checkpoint_errors:
                raise RuntimeError(
                    f"Invalid checkpoints for {weather} {group} seed{seed}: "
                    + "; ".join(checkpoint_errors)
                )
            task_id = f"{weather}_{group}_seed{seed}_source_weather"
            report_path = (
                Path(output_dir)
                / "weather_reference_reports"
                / weather
                / f"test_set_{_test_set_tag(target_sequences)}"
                / f"{group}_seed{seed}_source_weather_result.txt"
            ).resolve()
            task = {
                "task_id": task_id,
                "stage": "weather_reference",
                "weather": weather,
                "group": group,
                "seed": seed,
                "branch": "source",
                "metadata": {
                    header: str(row[header]) for header in METADATA_HEADERS
                },
                "target_sequences": target_sequences,
                "checkpoint_root": str(checkpoint_root),
                "checkpoint_identity": checkpoint_identity,
                "fixed_quartile_bins": control["target_quartile_bins"],
                "expected_weather_bbox_count": control["N_bbox_weather"],
                "report_path": str(report_path),
                "log_path": str(
                    Path(output_dir) / "logs" / f"{task_id}.log"
                ),
            }
            task["input_signature"] = _stable_hash({
                "version": STATE_VERSION,
                "checkpoint_identity": checkpoint_identity,
                "target_sequences": target_sequences,
                "derived_quartile_basis": control["target_quartile_bins"],
                "epochs": [5, 24],
                "official_eval": ["revised", "cuda", "all", True, 0.3],
            })
            tasks[task_id] = task
    if len(tasks) != EXPECTED_NEW_WEATHER_REFERENCE_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_NEW_WEATHER_REFERENCE_TASKS} new weather "
            f"reference tasks, discovered {len(tasks)}."
        )
    return tasks


def build_weather_reference_command(task, args):
    return [
        sys.executable,
        str(PROJECT_ROOT / "evaluation.py"),
        "--checkpoint-root", task["checkpoint_root"],
        "--start-epoch", "5", "--end-epoch", "24",
        "--batch-size", str(args.batch_size), "--num-workers", "0",
        "--cuda", "cuda:0", "--gpu-ids", "0",
        "--eval-val-sequences", ",".join(
            str(value) for value in task["target_sequences"]
        ),
        "--eval-report-path", task["report_path"],
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
        "--yolox-nms-iou", "0.65",
        "--ap-score-thresh", "0.01", "--score-thresh", "0.3",
        "--eval-ignore-suppress-enabled", "false",
        "--table-txt-enabled", "true",
        "--table-output-base-dir", str(
            args.output_dir / "weather_reference_reports"
        ),
        "--evaluation-tensorboard-log-dir", str(
            args.output_dir / "tensorboard_weather_reference"
        ),
        "--domain-comparison-enabled", "false", "--plot-output", "none",
    ]


def parse_completed_weather_reference(task):
    report_path = Path(task["report_path"])
    if not quartile_launcher.report_matches_task(report_path, task):
        return None
    metadata = _metadata_header(report_path)
    reported_sequences = _sequence_tuple(
        metadata.get("eval_val_sequences", metadata.get("val_sequences"))
    )
    if reported_sequences != tuple(task["target_sequences"]):
        return None
    if not _paths_equal(metadata.get("eval_report_path"), report_path):
        return None
    mode = metadata.get("distance_quartile_bins_mode", "").strip().lower()
    if mode not in {"derived", "rank", "automatic"}:
        return None
    metrics = quartile_launcher.parse_average_report(report_path)
    if metrics is None:
        return None
    if not _bounds_match(
        _quartile_bounds(metrics), task["fixed_quartile_bins"]
    ):
        return None
    actual_count = sum(int(metrics[tag]["N_bbox"]) for tag in QUARTILES)
    if actual_count != int(task["expected_weather_bbox_count"]):
        return None
    try:
        report_mtime = report_path.stat().st_mtime_ns
        checkpoint_mtime = max(
            path.stat().st_mtime_ns
            for path in Path(task["checkpoint_root"]).iterdir()
            if path.is_file()
            and re.search(r"(?i)epoch[_-]?0*(\d+)", path.name)
            and 5 <= int(re.search(r"(?i)epoch[_-]?0*(\d+)", path.name).group(1)) <= 24
        )
    except (OSError, ValueError):
        return None
    if report_mtime < checkpoint_mtime:
        return None
    return metrics


def initialize_weather_reference_state(state_path, discovered):
    previous = {}
    if Path(state_path).is_file():
        previous = _load_json(state_path).get("tasks", {})
    tasks = {}
    for task_id, found in discovered.items():
        old = previous.get(task_id, {})
        if old.get("status") == "running" and task_process_is_alive(old):
            raise RuntimeError(
                f"Recorded weather evaluator remains alive for {task_id} "
                f"(pid={old.get('pid')})."
            )
        task = dict(found)
        task.update(
            status="pending",
            attempts=int(old.get("attempts", 0)),
            metrics=None,
            pid=None,
            gpu=None,
            started_at=old.get("started_at"),
            finished_at=old.get("finished_at"),
            returncode=old.get("returncode"),
            error=None,
        )
        if old.get("input_signature") in (None, task["input_signature"]):
            metrics = parse_completed_weather_reference(task)
            if metrics is not None:
                task.update(status="completed", metrics=metrics)
        tasks[task_id] = task
    state = {
        "version": STATE_VERSION,
        "updated_at": now_text(),
        "settings": {
            "epochs": [5, 24],
            "stage": "source checkpoints on adverse-weather target tests",
            "official_metric": "K-Radar revised AP@IoU 0.3",
            "quartile_basis": "target-weather GT distance rank",
        },
        "tasks": tasks,
    }
    atomic_write_json(state_path, state)
    return state


def run_weather_reference_queue(args, state_path, state):
    """Run the three missing weather groups before normal-domain controls."""
    gpu_use = {gpu: 0 for gpu in args.gpus}
    running = {}
    pending = [
        task_id for task_id, task in state["tasks"].items()
        if task.get("status") == "pending"
    ]

    def save():
        state["updated_at"] = now_text()
        atomic_write_json(state_path, state)

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
            log_file.write(f"[{now_text()}] launcher stopped child: {reason}\n")
            log_file.close()
            state["tasks"][task_id].update(
                status="pending", pid=None, gpu=None,
                finished_at=now_text(), returncode=process.returncode,
                error=f"launcher stopped child: {reason}",
            )
        save()

    old_handlers = {}

    def interrupt_handler(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for number in (signal.SIGINT, signal.SIGTERM):
        old_handlers[number] = signal.signal(number, interrupt_handler)
    try:
        while pending or running:
            while pending and len(running) < args.max_workers:
                gpu = distance_launcher.choose_gpu(
                    gpu_use, args.gpus, args.max_per_gpu
                )
                if gpu is None:
                    break
                task_id = pending.pop(0)
                task = state["tasks"][task_id]
                command = build_weather_reference_command(task, args)
                log_path = Path(task["log_path"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = log_path.open("a", encoding="utf-8", buffering=1)
                log_file.write(
                    f"\n[{now_text()}] launch physical cuda:{gpu}\n"
                    f"command: {' '.join(command)}\n"
                )
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    command, cwd=str(PROJECT_ROOT), env=environment,
                    stdout=log_file, stderr=subprocess.STDOUT,
                )
                gpu_use[gpu] += 1
                task.update(
                    status="running", attempts=int(task.get("attempts", 0)) + 1,
                    pid=process.pid, gpu=gpu, started_at=now_text(),
                    finished_at=None, returncode=None, error=None,
                )
                running[task_id] = (process, log_file, gpu)
                print(
                    f"Started weather reference {task_id} pid={process.pid} "
                    f"physical_cuda={gpu} ({len(running)}/{args.max_workers} active)",
                    flush=True,
                )
                save()
            for task_id, (process, log_file, gpu) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_file.write(f"[{now_text()}] returncode={returncode}\n")
                log_file.close()
                gpu_use[gpu] -= 1
                task = state["tasks"][task_id]
                task.update(
                    pid=None, gpu=None, finished_at=now_text(),
                    returncode=int(returncode),
                )
                metrics = parse_completed_weather_reference(task)
                if returncode == 0 and metrics is not None:
                    task.update(status="completed", metrics=metrics, error=None)
                    print(
                        f"Completed weather reference {task_id}: "
                        f"{task['report_path']}", flush=True,
                    )
                else:
                    task.update(
                        status="failed", metrics=None,
                        error=(
                            f"evaluation returncode={returncode}; valid exact "
                            f"report={'yes' if metrics else 'no'}"
                        ),
                    )
                    print(
                        f"FAILED weather reference {task_id}: {task['error']} "
                        f"(log={task['log_path']})", flush=True,
                    )
                del running[task_id]
                save()
            if pending or running:
                time.sleep(args.poll_seconds)
    except BaseException as exc:
        stop_children(str(exc))
        raise
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
    counts = state_counts(state)
    print(f"Weather-reference queue finished: {counts}", flush=True)
    return counts.get("failed", 0) == 0


def _weather_task_index(experiments3_state):
    result = {}
    for task_id, task in experiments3_state.get("tasks", {}).items():
        try:
            key = (
                str(task["weather"]),
                str(task["group"]),
                int(task["seed"]),
                str(task["branch"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if key in result:
            raise RuntimeError(f"Duplicate experiments3 task identity: {key!r}")
        result[key] = dict(task, task_id=task_id)
    return result


def build_all_weather_reference_state(experiments3_state, new_reference_state):
    """Combine reusable Rain/Sleet reports with newly evaluated weather reports."""
    combined = {}
    for task_id, task in experiments3_state.get("tasks", {}).items():
        if (
            task.get("weather") in {"rain", "sleet"}
            and task.get("branch") == "source"
        ):
            combined[task_id] = dict(task, task_id=task_id)
    for task_id, task in new_reference_state.get("tasks", {}).items():
        combined[task_id] = dict(task, task_id=task_id)
    expected = EXPECTED_SD_TASKS
    if len(combined) != expected:
        raise RuntimeError(
            f"Expected {expected} all-weather source reference reports, "
            f"found {len(combined)}."
        )
    bad = [
        task_id for task_id, task in combined.items()
        if str(task.get("status", "")).lower() != "completed"
        or not quartile_launcher.report_matches_task(task.get("report_path", ""), task)
    ]
    if bad:
        raise RuntimeError(
            f"All-weather weather-reference reports are incomplete/invalid: {bad[:10]!r}"
        )
    return {"tasks": combined}


def discover_tasks(all_rows, output_dir, experiments3_state, controls):
    """Build all 80 source tasks, retaining readiness errors for dry-run."""
    weather_index = _weather_task_index(experiments3_state)
    tasks = {}
    for weather in WEATHERS:
        for row in all_rows[weather]:
            identity = row_identity(row)
            if identity is None:
                continue
            group, seed = identity
            key = (weather, group, seed, "source")
            if key not in weather_index:
                raise RuntimeError(f"Missing experiments3 source task: {key!r}")
            weather_task = weather_index[key]
            checkpoint_root = Path(
                str(weather_task.get("checkpoint_root", ""))
            ).expanduser().resolve()
            target_sequences = _sequence_tuple(row["test_seq"])
            target_tag = _test_set_tag(target_sequences)
            control = controls.get((weather, target_sequences))
            readiness = []
            if str(weather_task.get("status", "")).lower() != "completed":
                readiness.append(
                    f"experiments3 source task is {weather_task.get('status', 'unknown')}"
                )

            checkpoint_identity, checkpoint_errors = _epoch_checkpoint_identity(
                checkpoint_root
            )
            readiness.extend(checkpoint_errors)
            weather_report_path = weather_task.get("report_path")
            weather_metrics = None
            fixed_bins = None
            if weather_report_path is None or not quartile_launcher.report_matches_task(
                weather_report_path, weather_task
            ):
                readiness.append("completed target-weather source report is unavailable/invalid")
            else:
                weather_report_path = str(Path(weather_report_path).resolve())
                weather_metrics = quartile_launcher.parse_average_report(
                    weather_report_path
                )
                fixed_bins = _quartile_bounds(weather_metrics)

            if control is None:
                readiness.append(f"missing control spec for target {target_tag}")
                control = {
                    "weather_group": weather,
                    "target_sequences": target_sequences,
                    "source_sequences": (),
                    "normal_test_id": f"missing_target{target_tag}",
                    "manifest_path": None,
                    "override_path": None,
                    "status": "missing_control_spec",
                    "valid": False,
                    "exact": False,
                    "N_frame_normal": None,
                    "N_bbox_normal": None,
                    "N_bbox_weather": None,
                    "source_quartile_counts": {},
                    "target_quartile_counts": {},
                    "target_quartile_bins": None,
                    "manifest_sha256": None,
                    "override_sha256": None,
                    "stats_sha256": None,
                }
            elif not control["valid"]:
                readiness.extend(control.get("errors", ["invalid control spec"]))

            control_bins = control.get("target_quartile_bins")
            if fixed_bins is None and control_bins is not None:
                fixed_bins = control_bins
            if fixed_bins is not None and control_bins is not None:
                if not _bounds_match(fixed_bins, control_bins):
                    readiness.append(
                        f"control/reference quartile bounds mismatch for target {target_tag}"
                    )

            weather_bbox_count = (
                None
                if fixed_bins is None
                else sum(int(fixed_bins[tag]["N_bbox"]) for tag in QUARTILES)
            )
            control_weather_count = control.get("N_bbox_weather")
            if (
                weather_bbox_count is not None
                and control_weather_count is not None
                and int(control_weather_count) != weather_bbox_count
            ):
                readiness.append(
                    "control target bbox count does not match experiments3 report: "
                    f"{control_weather_count} != {weather_bbox_count}"
                )
            if weather_bbox_count is not None:
                control = dict(control)
                control["N_bbox_weather"] = weather_bbox_count

            report_path = (
                Path(output_dir)
                / "evaluation_reports"
                / weather
                / f"test_set_{target_tag}"
                / f"{group}_seed{seed}_source_normal_result.txt"
            ).resolve()
            task_id = f"{weather}_{group}_seed{seed}_source_normal"
            signature_payload = {
                "version": STATE_VERSION,
                "checkpoint_identity": checkpoint_identity,
                "weather_report_sha256": _file_sha256(weather_report_path)
                if weather_report_path
                else None,
                "manifest_sha256": control.get("manifest_sha256"),
                "override_sha256": control.get("override_sha256"),
                "stats_sha256": control.get("stats_sha256"),
                "source_sequences": control.get("source_sequences"),
                "fixed_bins": fixed_bins,
                "epochs": [5, 24],
                # Match the existing experiments3 target-weather evaluation
                # exactly.  AP@0.3 is the quantity used by the SD table, but
                # keeping ``all`` and the detection summaries enabled avoids
                # a hidden configuration difference between the two domains.
                "official_eval": ["revised", "cuda", "all", True, 0.3],
            }
            tasks[task_id] = {
                "task_id": task_id,
                "weather": weather,
                "group": group,
                "seed": seed,
                "branch": "source",
                "metadata": {header: str(row[header]) for header in METADATA_HEADERS},
                "target_sequences": target_sequences,
                "target_sequence": (
                    target_sequences[0] if len(target_sequences) == 1 else None
                ),
                "checkpoint_root": str(checkpoint_root),
                "checkpoint_identity": checkpoint_identity,
                "weather_experiments3_task_id": weather_task["task_id"],
                "weather_report_path": weather_report_path,
                "weather_report_sha256": _file_sha256(weather_report_path)
                if weather_report_path
                else None,
                "weather_metrics": weather_metrics,
                "fixed_quartile_bins": fixed_bins,
                "control": control,
                "report_path": str(report_path),
                "log_path": str(Path(output_dir) / "logs" / f"{task_id}.log"),
                "input_signature": _stable_hash(signature_payload),
                "ready": not readiness,
                "readiness_errors": readiness,
            }

    if len(tasks) != EXPECTED_SD_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_SD_TASKS} source-domain tasks, discovered {len(tasks)}."
        )
    counts = {
        weather: sum(task["weather"] == weather for task in tasks.values())
        for weather in WEATHERS
    }
    if counts != EXPECTED_SD_TASKS_BY_WEATHER:
        raise RuntimeError(f"Unexpected source-domain weather counts: {counts!r}")
    return tasks


def _fixed_bins_cli_text(task):
    bins = task.get("fixed_quartile_bins")
    if bins is None:
        raise ValueError(f"{task['task_id']} has no fixed quartile bins")
    payload = [
        {
            "tag": tag,
            "lower_m": bins[tag]["lower_m"],
            "upper_m": (
                "inf" if math.isinf(float(bins[tag]["upper_m"])) else bins[tag]["upper_m"]
            ),
        }
        for tag in QUARTILES
    ]
    return json.dumps(payload, separators=(",", ":"))


def build_evaluation_command(task, args):
    control = task["control"]
    if not task.get("ready"):
        raise ValueError(
            f"Task {task['task_id']} is not ready: {task['readiness_errors']}"
        )
    return [
        sys.executable,
        str(PROJECT_ROOT / "evaluation.py"),
        "--checkpoint-root", task["checkpoint_root"],
        "--start-epoch", "5", "--end-epoch", "24",
        "--batch-size", str(args.batch_size), "--num-workers", "0",
        "--cuda", "cuda:0", "--gpu-ids", "0",
        "--eval-val-sequences", ",".join(
            str(value) for value in control["source_sequences"]
        ),
        "--eval-frame-manifest-path", control["manifest_path"],
        "--eval-gt-object-ignore-override-path", control["override_path"],
        "--eval-report-path", task["report_path"],
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
        "--distance-quartile-bins", _fixed_bins_cli_text(task),
        "--max-detections", "64", "--heatmap-nms-kernel", "3",
        "--heatmap-score-mode", "peak_times_local_mean",
        "--yolox-nms-iou", "0.65",
        "--ap-score-thresh", "0.01", "--score-thresh", "0.3",
        "--eval-ignore-suppress-enabled", "false",
        "--table-txt-enabled", "true",
        "--table-output-base-dir", str(args.output_dir / "evaluation_reports"),
        "--evaluation-tensorboard-log-dir", str(args.output_dir / "tensorboard"),
        "--domain-comparison-enabled", "false", "--plot-output", "none",
    ]


def _report_is_newer_than_inputs(report_path, task):
    try:
        report_mtime = Path(report_path).stat().st_mtime_ns
    except OSError:
        return False
    inputs = [
        task.get("weather_report_path"),
        task.get("control", {}).get("manifest_path"),
        task.get("control", {}).get("override_path"),
        task.get("control", {}).get("stats_path"),
    ]
    for input_path in inputs:
        if input_path in (None, ""):
            continue
        try:
            if Path(input_path).stat().st_mtime_ns > report_mtime:
                return False
        except OSError:
            return False
    checkpoint_root = Path(task.get("checkpoint_root", ""))
    try:
        for path in checkpoint_root.iterdir():
            if not path.is_file():
                continue
            match = re.search(r"(?i)epoch[_-]?0*(\d+)", path.name)
            if match and 5 <= int(match.group(1)) <= 24:
                if path.stat().st_mtime_ns > report_mtime:
                    return False
    except OSError:
        return False
    return True


def parse_completed_normal_report(task):
    report_path = Path(task["report_path"])
    metrics = quartile_launcher.parse_average_report(report_path)
    if metrics is None or not _report_is_newer_than_inputs(report_path, task):
        return None
    metadata = _metadata_header(report_path)
    if not _paths_equal(metadata.get("checkpoint_root"), task["checkpoint_root"]):
        return None
    if metadata.get("domain_shift_train_branch") != "source":
        return None
    if metadata.get("weather_group") != task["weather"]:
        return None
    if metadata.get("seed") != str(task["seed"]):
        return None
    control = task["control"]
    expected_sequences = tuple(control["source_sequences"])
    reported_sequences = _sequence_tuple(
        metadata.get("eval_val_sequences", metadata.get("val_sequences"))
    )
    if reported_sequences != expected_sequences:
        return None
    if not _paths_equal(
        metadata.get("eval_frame_manifest_path"), control["manifest_path"]
    ):
        return None
    if not _paths_equal(
        metadata.get("eval_gt_object_ignore_override_path"), control["override_path"]
    ):
        return None
    if not _paths_equal(metadata.get("eval_report_path"), report_path):
        return None
    expected_neutral_count = task["control"].get("N_bbox_masked")
    reported_neutral_count = _integer(metadata.get("official_neutral_gt_count"))
    if (
        expected_neutral_count is None
        or reported_neutral_count is None
        or reported_neutral_count != int(expected_neutral_count)
    ):
        return None
    mode = metadata.get("distance_quartile_bins_mode", "").strip().lower()
    if mode not in {"fixed", "provided", "external"}:
        return None
    if not _bounds_match(_quartile_bounds(metrics), task["fixed_quartile_bins"]):
        return None
    expected_total = control.get("N_bbox_normal")
    actual_total = sum(int(metrics[tag]["N_bbox"]) for tag in QUARTILES)
    if expected_total is not None and actual_total != int(expected_total):
        return None
    expected_q = control.get("source_quartile_counts", {})
    if expected_q and any(
        tag in expected_q and int(metrics[tag]["N_bbox"]) != int(expected_q[tag])
        for tag in QUARTILES
    ):
        return None
    return metrics


def task_process_is_alive(task):
    try:
        pid = int(task.get("pid"))
    except (TypeError, ValueError):
        return False
    path = Path(f"/proc/{pid}/cmdline")
    try:
        command = path.read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return False
    return (
        "evaluation.py" in command
        and str(task.get("checkpoint_root", "")) in command
        and str(task.get("report_path", "")) in command
    )


def initialize_state(state_path, discovered):
    previous_tasks = {}
    if Path(state_path).is_file():
        previous_tasks = _load_json(state_path).get("tasks", {})
    tasks = {}
    for task_id, found in discovered.items():
        previous = previous_tasks.get(task_id, {})
        if previous.get("status") == "running" and task_process_is_alive(previous):
            raise RuntimeError(
                f"Recorded evaluator remains alive for {task_id} "
                f"(pid={previous.get('pid')})."
            )
        task = dict(found)
        task.update(
            status="pending",
            attempts=int(previous.get("attempts", 0)),
            metrics=None,
            pid=None,
            gpu=None,
            started_at=previous.get("started_at"),
            finished_at=previous.get("finished_at"),
            returncode=previous.get("returncode"),
            error=None,
        )
        previous_signature = previous.get("input_signature")
        if previous_signature in (None, task["input_signature"]):
            metrics = parse_completed_normal_report(task)
            if metrics is not None:
                task.update(status="completed", metrics=metrics)
        tasks[task_id] = task
    state = {
        "version": STATE_VERSION,
        "updated_at": now_text(),
        "settings": {
            "epochs": [5, 24],
            "formula": "SD = AP_normal - AP_weather",
            "official_metric": "K-Radar revised AP@IoU 0.3",
            "quartile_basis": "fixed target-weather GT distance quartiles",
        },
        "tasks": tasks,
    }
    atomic_write_json(state_path, state)
    return state


def metric_headers():
    headers = list(METADATA_HEADERS)
    headers.extend(
        (
            "normal_test_id",
            "control_status",
            "N_frame_normal",
            "N_bbox_normal",
            "N_bbox_weather",
            "BEV_normal",
            "3D_normal",
            "BEV_weather",
            "3D_weather",
            "SD_BEV",
            "SD_3D",
        )
    )
    for tag in QUARTILES:
        headers.extend(
            (
                f"range_m_{tag}",
                f"N_bbox_normal_{tag}",
                f"N_bbox_weather_{tag}",
                f"BEV_normal_{tag}",
                f"3D_normal_{tag}",
                f"BEV_weather_{tag}",
                f"3D_weather_{tag}",
            )
        )
    return headers


def _format_metric(value):
    return "" if value is None else f"{float(value):.4f}"


def _format_bound(value):
    return "inf" if math.isinf(float(value)) else f"{float(value):.4f}"


def _task_index(state):
    return {
        (task["weather"], task["group"], int(task["seed"])): task
        for task in state["tasks"].values()
    }


def _average_row(headers, label, values_by_block):
    row = {header: "" for header in headers}
    row["group"] = label
    for header in METADATA_HEADERS[1:]:
        row[header] = "-"
    for block, values in values_by_block.items():
        if not values:
            continue
        suffix = "" if block == "all" else f"_{block}"
        for index, metric_name in enumerate(
            ("BEV_normal", "3D_normal", "BEV_weather", "3D_weather")
        ):
            row[f"{metric_name}{suffix}"] = _format_metric(
                sum(value[index] for value in values) / len(values)
            )
        if block == "all":
            row["SD_BEV"] = _format_metric(
                sum(value[0] - value[2] for value in values) / len(values)
            )
            row["SD_3D"] = _format_metric(
                sum(value[1] - value[3] for value in values) / len(values)
            )
    return [row[header] for header in headers]


def build_table_matrix(weather, experiment_rows, state):
    headers = metric_headers()
    index = _task_index(state)
    output = []
    all_values = {block: [] for block in ("all",) + QUARTILES}
    exact_values = {block: [] for block in ("all",) + QUARTILES}

    for metadata in experiment_rows:
        values = {header: "" for header in headers}
        for header in METADATA_HEADERS:
            values[header] = str(metadata[header])
        identity = row_identity(metadata)
        if identity is None:
            output.append([values[header] for header in headers])
            continue
        group, seed = identity
        task = index[(weather, group, seed)]
        control = task["control"]
        values.update(
            normal_test_id=control.get("normal_test_id", ""),
            control_status=control.get("status", ""),
            N_frame_normal="" if control.get("N_frame_normal") is None else str(control["N_frame_normal"]),
            N_bbox_normal="" if control.get("N_bbox_normal") is None else str(control["N_bbox_normal"]),
            N_bbox_weather="" if control.get("N_bbox_weather") is None else str(control["N_bbox_weather"]),
        )
        normal = task.get("metrics") if task.get("status") == "completed" else None
        weather_metrics = task.get("weather_metrics")
        normal_all = None if normal is None else normal.get("all")
        weather_all = None if weather_metrics is None else weather_metrics.get("all")
        normal_bev = None if normal_all is None else normal_all.get("BEV")
        normal_3d = None if normal_all is None else normal_all.get("3D")
        weather_bev = None if weather_all is None else weather_all.get("BEV")
        weather_3d = None if weather_all is None else weather_all.get("3D")
        sd_bev = None if normal_bev is None or weather_bev is None else normal_bev - weather_bev
        sd_3d = None if normal_3d is None or weather_3d is None else normal_3d - weather_3d
        for key, value in (
            ("BEV_normal", normal_bev), ("3D_normal", normal_3d),
            ("BEV_weather", weather_bev), ("3D_weather", weather_3d),
            ("SD_BEV", sd_bev), ("SD_3D", sd_3d),
        ):
            values[key] = _format_metric(value)
        if sd_bev is not None and sd_3d is not None and control.get("valid"):
            overall_aps = (normal_bev, normal_3d, weather_bev, weather_3d)
            all_values["all"].append(overall_aps)
            if control.get("exact"):
                exact_values["all"].append(overall_aps)

        bins = task.get("fixed_quartile_bins") or {}
        expected_normal_q = control.get("source_quartile_counts", {})
        for tag in QUARTILES:
            fixed = bins.get(tag)
            normal_block = None if normal is None else normal.get(tag)
            weather_block = None if weather_metrics is None else weather_metrics.get(tag)
            if fixed is not None:
                values[f"range_m_{tag}"] = (
                    f"[{_format_bound(fixed['lower_m'])},{_format_bound(fixed['upper_m'])})"
                )
            normal_count = (
                normal_block.get("N_bbox")
                if normal_block is not None
                else expected_normal_q.get(tag)
            )
            weather_count = None if weather_block is None else weather_block.get("N_bbox")
            values[f"N_bbox_normal_{tag}"] = "" if normal_count is None else str(normal_count)
            values[f"N_bbox_weather_{tag}"] = "" if weather_count is None else str(weather_count)
            normal_bev = None if normal_block is None else normal_block.get("BEV")
            normal_3d = None if normal_block is None else normal_block.get("3D")
            weather_bev = None if weather_block is None else weather_block.get("BEV")
            weather_3d = None if weather_block is None else weather_block.get("3D")
            for key, value in (
                (f"BEV_normal_{tag}", normal_bev), (f"3D_normal_{tag}", normal_3d),
                (f"BEV_weather_{tag}", weather_bev), (f"3D_weather_{tag}", weather_3d),
            ):
                values[key] = _format_metric(value)
            if (
                None not in (normal_bev, normal_3d, weather_bev, weather_3d)
                and control.get("valid")
            ):
                quartile_aps = (normal_bev, normal_3d, weather_bev, weather_3d)
                all_values[tag].append(quartile_aps)
                if control.get("exact"):
                    exact_values[tag].append(quartile_aps)
        output.append([values[header] for header in headers])

    output.append(
        _average_row(
            headers,
            f"average_all_valid({len(all_values['all'])})",
            all_values,
        )
    )
    output.append(
        _average_row(
            headers,
            f"average_exact({len(exact_values['all'])})",
            exact_values,
        )
    )
    return [headers] + output


def format_table(matrix):
    widths = [
        max(len(str(row[index])) for row in matrix)
        for index in range(len(matrix[0]))
    ]
    text_columns = set(METADATA_HEADERS) | {"normal_test_id", "control_status"}

    def one_row(row, header=False):
        cells = []
        for index, value in enumerate(row):
            value = str(value)
            left = header or matrix[0][index] in text_columns
            cells.append(
                f"{value:<{widths[index]}}" if left else f"{value:>{widths[index]}}"
            )
        return "  ".join(cells).rstrip()

    header = one_row(matrix[0], header=True)
    return "\n".join(
        [header, "-" * len(header)] + [one_row(row) for row in matrix[1:]]
    ) + "\n"


def write_tables(all_rows, state, output_dir):
    for weather in WEATHERS:
        matrix = build_table_matrix(weather, all_rows[weather], state)
        atomic_write_text(
            Path(output_dir) / f"{weather}_sd_experiments.txt",
            format_table(matrix),
        )
    write_all_weather_summary(state, output_dir)


def write_all_weather_summary(state, output_dir):
    blocks = ("all",)
    headers = ["weather", "completed", "valid", "exact"]
    for block in blocks:
        suffix = "" if block == "all" else f"_{block}"
        headers.extend((
            f"SD_BEV_all_valid{suffix}",
            f"SD_3D_all_valid{suffix}",
            f"SD_BEV_exact{suffix}",
            f"SD_3D_exact{suffix}",
        ))

    def aggregate(tasks, exact_only, block):
        values = []
        for task in tasks:
            control = task.get("control", {})
            if not control.get("valid") or (exact_only and not control.get("exact")):
                continue
            if task.get("status") != "completed":
                continue
            normal = (task.get("metrics") or {}).get(block)
            weather = (task.get("weather_metrics") or {}).get(block)
            if normal is None or weather is None:
                continue
            values.append((
                float(normal["BEV"]) - float(weather["BEV"]),
                float(normal["3D"]) - float(weather["3D"]),
            ))
        if not values:
            return None, None
        return (
            sum(value[0] for value in values) / len(values),
            sum(value[1] for value in values) / len(values),
        )

    groups = [
        (weather, [
            task for task in state["tasks"].values()
            if task["weather"] == weather
        ])
        for weather in WEATHERS
    ]
    groups.append(("ALL", list(state["tasks"].values())))
    rows = []
    for label, tasks in groups:
        row = {
            "weather": label,
            "completed": str(sum(task.get("status") == "completed" for task in tasks)),
            "valid": str(sum(task.get("control", {}).get("valid", False) for task in tasks)),
            "exact": str(sum(task.get("control", {}).get("exact", False) for task in tasks)),
        }
        for block in blocks:
            suffix = "" if block == "all" else f"_{block}"
            valid_bev, valid_3d = aggregate(tasks, False, block)
            exact_bev, exact_3d = aggregate(tasks, True, block)
            row[f"SD_BEV_all_valid{suffix}"] = _format_metric(valid_bev)
            row[f"SD_3D_all_valid{suffix}"] = _format_metric(valid_3d)
            row[f"SD_BEV_exact{suffix}"] = _format_metric(exact_bev)
            row[f"SD_3D_exact{suffix}"] = _format_metric(exact_3d)
        rows.append([row.get(header, "") for header in headers])
    atomic_write_text(
        Path(output_dir) / "all_weather_sd_summary.txt",
        format_table([headers] + rows),
    )


def write_readme(output_dir):
    atomic_write_text(
        Path(output_dir) / "README.txt",
        (
            "Source-domain performance difference for Heavy Snow, Light Snow, "
            "Overcast, Rain, and Sleet.\n\n"
            "Every AP is official revised K-Radar AP@IoU 0.3 averaged over "
            "epochs 5-24 of the same source-trained checkpoint. BEV_normal/"
            "3D_normal use the controlled normal test; BEV_weather/3D_weather "
            "reuse/recompute that checkpoint's adverse-weather target-test "
            "result with identical settings. SD = AP_normal - AP_weather, so positive SD denotes "
            "adverse-weather degradation. SD is reported only for the overall "
            "AP. q1-q4 retain their range, count, and domain AP values but do "
            "not include separate SD columns; they use the target-weather GT "
            "distance boundaries for both domains.\n\n"
            "average_all_valid includes every completed valid control, including "
            "reported source-GT deficits. average_exact includes only controls "
            "whose normal and weather eligible Sedan totals match exactly. Both "
            "rows retain mean BEV/3D AP for the overall result and q1-q4; only "
            "the overall result includes mean SD.\n"
        ),
    )


def state_counts(state):
    counts = {key: 0 for key in ("pending", "running", "completed", "failed")}
    for task in state["tasks"].values():
        status = task.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def save_progress(state_path, state, all_rows, output_dir):
    state["updated_at"] = now_text()
    atomic_write_json(state_path, state)
    write_tables(all_rows, state, output_dir)


def run_queue(args, state_path, state, all_rows):
    gpu_use = {gpu: 0 for gpu in args.gpus}
    running = {}
    pending = [
        task_id
        for task_id, task in state["tasks"].items()
        if task.get("status") == "pending"
    ]

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
            log_file.write(f"[{now_text()}] launcher stopped child: {reason}\n")
            log_file.close()
            state["tasks"][task_id].update(
                status="pending",
                pid=None,
                gpu=None,
                finished_at=now_text(),
                returncode=process.returncode,
                error=f"launcher stopped child: {reason}",
            )
        save_progress(state_path, state, all_rows, args.output_dir)

    old_handlers = {}

    def interrupt_handler(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signal_number] = signal.signal(signal_number, interrupt_handler)
    try:
        while pending or running:
            while pending and len(running) < args.max_workers:
                gpu = distance_launcher.choose_gpu(
                    gpu_use, args.gpus, args.max_per_gpu
                )
                if gpu is None:
                    break
                task_id = pending.pop(0)
                task = state["tasks"][task_id]
                command = build_evaluation_command(task, args)
                log_path = Path(task["log_path"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = log_path.open("a", encoding="utf-8", buffering=1)
                log_file.write(
                    f"\n[{now_text()}] launch physical cuda:{gpu}\n"
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
                    status="running",
                    attempts=int(task.get("attempts", 0)) + 1,
                    pid=process.pid,
                    gpu=gpu,
                    started_at=now_text(),
                    finished_at=None,
                    returncode=None,
                    error=None,
                )
                running[task_id] = (process, log_file, gpu)
                print(
                    f"Started {task_id} pid={process.pid} physical_cuda={gpu} "
                    f"({len(running)}/{args.max_workers} active)",
                    flush=True,
                )
                save_progress(state_path, state, all_rows, args.output_dir)

            for task_id, (process, log_file, gpu) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_file.write(f"[{now_text()}] returncode={returncode}\n")
                log_file.close()
                gpu_use[gpu] -= 1
                task = state["tasks"][task_id]
                task.update(
                    pid=None,
                    gpu=None,
                    finished_at=now_text(),
                    returncode=int(returncode),
                )
                metrics = parse_completed_normal_report(task)
                if returncode == 0 and metrics is not None:
                    task.update(status="completed", metrics=metrics, error=None)
                    print(f"Completed {task_id}: {task['report_path']}", flush=True)
                else:
                    task.update(
                        status="failed",
                        metrics=None,
                        error=(
                            f"evaluation returncode={returncode}; "
                            f"valid exact report={'yes' if metrics else 'no'}"
                        ),
                    )
                    print(
                        f"FAILED {task_id}: {task['error']} (log={task['log_path']})",
                        flush=True,
                    )
                del running[task_id]
                save_progress(state_path, state, all_rows, args.output_dir)
            if pending or running:
                time.sleep(args.poll_seconds)
    except BaseException as exc:
        stop_children(str(exc))
        raise
    finally:
        for signal_number, handler in old_handlers.items():
            signal.signal(signal_number, handler)
    counts = state_counts(state)
    print(f"SD queue finished: {counts}", flush=True)
    return 0 if counts.get("failed", 0) == 0 else 1


def main(argv=None):
    args = parse_args(argv)
    for child in (
        "logs", "evaluation_reports", "weather_reference_reports",
        "tensorboard", "tensorboard_weather_reference", "control_specs",
    ):
        (args.output_dir / child).mkdir(parents=True, exist_ok=True)
    lock_handle = acquire_output_lock(args.output_dir)
    try:
        all_rows = {
            weather: read_experiment_rows(weather) for weather in WEATHERS
        }
        _, experiments3_state = _load_experiments3_state(args.experiments3_dir)
        controls, control_index_errors = load_control_specs(args.control_spec_index)
        if control_index_errors:
            raise RuntimeError(
                "Invalid all-weather control index: "
                + "; ".join(control_index_errors[:10])
            )
        if len(controls) != 17:
            raise RuntimeError(
                f"Expected 17 distinct target-test controls, found {len(controls)}."
            )

        reference_discovered = discover_weather_reference_tasks(
            all_rows, args.output_dir, controls
        )
        reference_state_path = (
            args.output_dir / "weather_reference_evaluation_state.json"
        )
        reference_state = initialize_weather_reference_state(
            reference_state_path, reference_discovered
        )
        completion_errors = experiments3_completion_errors(experiments3_state)

        if args.dry_run:
            combined_tasks = {
                task_id: dict(task, task_id=task_id)
                for task_id, task in experiments3_state.get("tasks", {}).items()
                if task.get("weather") in {"rain", "sleet"}
                and task.get("branch") == "source"
            }
            combined_tasks.update({
                task_id: dict(task, task_id=task_id)
                for task_id, task in reference_state["tasks"].items()
            })
            discovered = discover_tasks(
                all_rows, args.output_dir, {"tasks": combined_tasks}, controls
            )
            state_path = args.output_dir / "sd_evaluation_state.json"
            state = initialize_state(state_path, discovered)
            write_readme(args.output_dir)
            write_tables(all_rows, state, args.output_dir)
            ready = sum(
                task.get("ready", False) for task in state["tasks"].values()
            )
            counts = state_counts(state)
            reference_counts = state_counts(reference_state)
            print(
                f"Validated {len(reference_state['tasks'])} missing-weather "
                f"reference tasks {reference_counts} and {len(state['tasks'])} "
                f"all-weather SD tasks (ready={ready}, "
                f"not_ready={len(state['tasks']) - ready}); status={counts}",
                flush=True,
            )
            not_ready = [
                task for task in state["tasks"].values() if not task.get("ready")
            ]
            for task in not_ready[:10]:
                print(
                    f"Not ready {task['task_id']}: "
                    + "; ".join(task["readiness_errors"]),
                    flush=True,
                )
            if len(not_ready) > 10:
                print(f"... {len(not_ready) - 10} additional not-ready tasks", flush=True)
            first_ready = next(
                (task for task in state["tasks"].values() if task.get("ready")),
                None,
            )
            if first_ready is not None:
                print(
                    "Example command:",
                    " ".join(build_evaluation_command(first_ready, args)),
                    flush=True,
                )
            print(
                "Dry run complete; no evaluation processes were launched. "
                f"experiments3 gate: {'ready' if not completion_errors else completion_errors[0]}",
                flush=True,
            )
            return 0

        if args.wait_for_quartile_completion:
            experiments3_state = wait_for_experiments3(
                args.experiments3_dir,
                args.poll_seconds,
            )
            completion_errors = experiments3_completion_errors(experiments3_state)
        if completion_errors:
            raise RuntimeError(
                "Refusing to launch before experiments3 is fully complete: "
                + "; ".join(completion_errors[:10])
            )

        distance_launcher.validate_physical_gpus(args.gpus)
        if not run_weather_reference_queue(
            args, reference_state_path, reference_state
        ):
            raise RuntimeError(
                "Weather-reference stage failed; see experiments4/logs and "
                "resume this launcher after fixing the failed evaluator."
            )
        combined_reference_state = build_all_weather_reference_state(
            experiments3_state, reference_state
        )
        discovered = discover_tasks(
            all_rows,
            args.output_dir,
            combined_reference_state,
            controls,
        )
        state_path = args.output_dir / "sd_evaluation_state.json"
        state = initialize_state(state_path, discovered)
        write_readme(args.output_dir)
        write_tables(all_rows, state, args.output_dir)

        ready = sum(task.get("ready", False) for task in state["tasks"].values())
        counts = state_counts(state)
        print(
            f"Validated {len(state['tasks'])} all-weather SD tasks "
            f"(ready={ready}, not_ready={len(state['tasks']) - ready}); "
            f"status={counts}; state={state_path}",
            flush=True,
        )
        not_ready = {
            task_id: task["readiness_errors"]
            for task_id, task in state["tasks"].items()
            if not task.get("ready")
        }
        if not_ready:
            preview = "; ".join(
                f"{task_id}: {', '.join(errors)}"
                for task_id, errors in list(not_ready.items())[:5]
            )
            raise RuntimeError(
                f"Refusing to launch {len(not_ready)} not-ready SD tasks: {preview}"
            )
        return run_queue(args, state_path, state, all_rows)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
