"""Ordered source/target domain-shift experiments loaded from CSV/TXT tables."""

import copy
import csv
import fcntl
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from training_utils.post_training_evaluation import (
    prepare_post_training_evaluation_launch,
    query_gpu_status,
    resolve_candidate_physical_gpu_ids,
)
from training_utils.runtime import parse_gpu_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_BRANCHES = ("source", "target")


@dataclass(frozen=True)
class DomainShiftExperiment:
    row_number: int
    name: str
    seed: int
    shared_sequences: tuple
    source_sequences: tuple
    target_sequences: tuple
    test_sequences: tuple
    half_selection: tuple
    source_complete: bool
    target_complete: bool

    def branch_complete(self, branch):
        if branch == "source":
            return self.source_complete
        if branch == "target":
            return self.target_complete
        raise ValueError(f"Unsupported experiment branch: {branch!r}")


@dataclass(frozen=True)
class ExperimentQueueTask:
    ordinal: int
    experiment: DomainShiftExperiment
    branch: str

    @property
    def label(self):
        return f"{self.experiment.name}/{self.branch}"


def resolve_experiment_sheet_path(path):
    sheet_path = Path(str(path)).expanduser()
    if not sheet_path.is_absolute():
        sheet_path = PROJECT_ROOT / sheet_path
    return sheet_path.resolve()


def _normalized_header(value):
    return " ".join(
        str(value or "").replace("\ufeff", "").strip().lower().split()
    )


def _find_column(headers, *accepted_names):
    normalized = {
        _normalized_header(header): header
        for header in headers
    }
    for accepted_name in accepted_names:
        matched = normalized.get(_normalized_header(accepted_name))
        if matched is not None:
            return matched
    raise ValueError(
        "Experiment sheet is missing a required column; expected one of "
        f"{accepted_names}, found {list(headers)}"
    )


def _optional_column(headers, *accepted_names):
    try:
        return _find_column(headers, *accepted_names)
    except ValueError:
        return None


def _parse_seed(value, fallback):
    text = str(value or "").strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    match = re.search(r"seed\s*(\d+)", text, flags=re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    return int(fallback)


def _normalize_sequence_cell(value):
    text = str(value or "").strip().lower()
    replacements = {
        "(first half)": "_first",
        "(last half)": "_last",
        "(first)": "_first",
        "(last)": "_last",
        " first half": "_first",
        " last half": "_last",
    }
    for old_text, new_text in replacements.items():
        text = text.replace(old_text, new_text)
    return text


def parse_sequence_cell(value, column_name):
    """Parse ``9``, ``9_first`` or ``9(first half)`` sequence cells."""
    text = _normalize_sequence_cell(value)
    if text in {"", "-", "none", "nan"}:
        raise ValueError(f"{column_name} cannot be empty")

    sequences = []
    half_selection = {}
    for token in (part.strip() for part in text.split(",")):
        if token == "":
            continue
        match = re.fullmatch(r"(\d+)(?:_(first|last))?", token)
        if match is None:
            raise ValueError(
                f"Invalid {column_name} value {value!r}; expected values like "
                "9, 9_first, 9(last half), or a comma-separated list."
            )
        sequence = int(match.group(1))
        half = match.group(2)
        if sequence not in sequences:
            sequences.append(sequence)
        if half is not None:
            existing = half_selection.get(sequence)
            if existing is not None and existing != half:
                raise ValueError(
                    f"Sequence {sequence} has conflicting half selections "
                    f"in {column_name}: {existing!r} and {half!r}"
                )
            half_selection[sequence] = half

    if not sequences:
        raise ValueError(f"{column_name} cannot be empty")
    return tuple(sequences), half_selection


def _merge_half_selections(*selections):
    merged = {}
    for selection in selections:
        for sequence, half in selection.items():
            existing = merged.get(sequence)
            if existing is not None and existing != half:
                raise ValueError(
                    f"Sequence {sequence} has conflicting half selections "
                    f"{existing!r} and {half!r}"
                )
            merged[sequence] = half
    return tuple(sorted(merged.items()))


def _experiment_sort_key(experiment):
    return (
        int(experiment.seed),
        tuple(int(sequence) for sequence in experiment.test_sequences),
        int(experiment.row_number),
    )


def _metric_is_present(row, column):
    if column is None:
        return False
    value = str(row.get(column, "") or "").strip().lower()
    return value not in {"", "-", "none", "nan"}


def _read_txt_experiment_matrix(sheet_path):
    lines = sheet_path.read_text(encoding="utf-8-sig").splitlines()
    header_index = None
    headers = None
    required_headers = {
        "shared_seq",
        "source_seq",
        "target_seq",
        "test_seq",
    }
    for index, line in enumerate(lines):
        candidate = line.split()
        normalized = {
            _normalized_header(value)
            for value in candidate
        }
        if required_headers.issubset(normalized):
            header_index = index
            headers = candidate
            break
    if header_index is None or headers is None:
        raise ValueError(
            "TXT experiment table is missing a whitespace-separated header "
            f"containing {sorted(required_headers)}: {sheet_path}"
        )

    rows = [headers]
    for line_number, line in enumerate(
        lines[header_index + 1:],
        start=header_index + 2,
    ):
        stripped = line.strip()
        if stripped == "" or set(stripped) <= {"-", " "}:
            continue
        values = stripped.split()
        if len(values) != len(headers):
            raise ValueError(
                f"TXT experiment row {line_number} has {len(values)} values, "
                f"but the header has {len(headers)}: {line}"
            )
        rows.append(values)
    return rows


def _read_experiment_matrix(sheet_path):
    suffix = sheet_path.suffix.lower()
    if suffix == ".csv":
        with sheet_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as input_file:
            return list(csv.reader(input_file))
    if suffix == ".txt":
        return _read_txt_experiment_matrix(sheet_path)
    raise ValueError(
        "The experiment queue expects an Excel-compatible CSV or an aligned "
        f"TXT table, got: {sheet_path}"
    )


def load_domain_shift_experiments(sheet_path, default_seed=42):
    """Load valid experiment rows in their original CSV/TXT order."""
    resolved_path = resolve_experiment_sheet_path(sheet_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Experiment sheet not found: {resolved_path}")
    matrix = _read_experiment_matrix(resolved_path)
    if not matrix:
        raise ValueError(f"Experiment sheet has no header: {resolved_path}")
    headers = tuple(matrix[0])
    name_column = _optional_column(
        headers,
        "group",
        "experiment",
        "experiment_group",
    ) or headers[0]
    seed_column = _optional_column(headers, "seed")
    shared_column = _find_column(
        headers,
        "shared train sequence",
        "shared_seq",
    )
    source_column = _find_column(
        headers,
        "source train set",
        "source_seq",
    )
    target_column = _find_column(
        headers,
        "target train set",
        "target_seq",
    )
    test_column = _find_column(
        headers,
        "target test set",
        "test_seq",
    )
    source_bev_column = _optional_column(
        headers,
        "bev_AP(source)",
        "BEV_src",
    )
    source_3d_column = _optional_column(
        headers,
        "3d_AP(source)",
        "3D_src",
    )
    target_bev_column = _optional_column(
        headers,
        "bev_AP(target)",
        "BEV_tgt",
    )
    target_3d_column = _optional_column(
        headers,
        "3d_AP(target)",
        "3D_tgt",
    )
    header_seed = _parse_seed(name_column, default_seed)

    experiments = []
    for row_number, values in enumerate(matrix[1:], start=2):
        row = dict(zip(headers, values))
        name = str(row.get(name_column, "") or "").strip()
        required_values = (
            row.get(shared_column),
            row.get(source_column),
            row.get(target_column),
            row.get(test_column),
        )
        missing_markers = {"", "-", "none", "nan"}
        required_missing = [
            str(value or "").strip().lower() in missing_markers
            for value in required_values
        ]
        if all(required_missing):
            continue
        if name.lower().startswith("average"):
            continue
        if any(required_missing):
            raise ValueError(
                f"Row {row_number} ({name or 'unnamed group'}): shared_seq, "
                "source_seq, target_seq, and test_seq must either all be "
                "filled or all be '-' for an unused template row."
            )

        shared, shared_half = parse_sequence_cell(
            row.get(shared_column),
            "shared train sequence",
        )
        source, source_half = parse_sequence_cell(
            row.get(source_column),
            "source train set",
        )
        target, target_half = parse_sequence_cell(
            row.get(target_column),
            "target train set",
        )
        test, test_half = parse_sequence_cell(
            row.get(test_column),
            "target test set",
        )
        if test_half:
            raise ValueError(
                f"Row {row_number}: target test sequences cannot use "
                "first/last-half selection."
            )

        experiments.append(DomainShiftExperiment(
            row_number=row_number,
            name=name or f"row_{row_number}",
            seed=_parse_seed(
                row.get(seed_column) if seed_column is not None else name,
                header_seed,
            ),
            shared_sequences=shared,
            source_sequences=source,
            target_sequences=target,
            test_sequences=test,
            half_selection=_merge_half_selections(
                shared_half,
                source_half,
                target_half,
            ),
            source_complete=(
                _metric_is_present(row, source_bev_column)
                and _metric_is_present(row, source_3d_column)
            ),
            target_complete=(
                _metric_is_present(row, target_bev_column)
                and _metric_is_present(row, target_3d_column)
            ),
        ))

    if not experiments:
        raise ValueError(
            f"Experiment sheet contains no valid experiment rows: {resolved_path}"
        )
    return sorted(experiments, key=_experiment_sort_key)


def _same_sequences(left, right):
    return tuple(sorted(int(value) for value in left)) == tuple(
        sorted(int(value) for value in right)
    )


def _report_matches_experiment(report, experiment, branch):
    return (
        report.get("branch") == branch
        and int(report.get("seed", -1)) == int(experiment.seed)
        and _same_sequences(
            report.get("shared_train_sequences", ()),
            experiment.shared_sequences,
        )
        and _same_sequences(
            report.get("source_train_sequences", ()),
            experiment.source_sequences,
        )
        and _same_sequences(
            report.get("target_train_sequences", ()),
            experiment.target_sequences,
        )
        and _same_sequences(
            report.get("target_test_sequences", ()),
            experiment.test_sequences,
        )
        and tuple(sorted(report.get("shared_half_selection", ())))
        == tuple(sorted(experiment.half_selection))
    )


def find_fresh_experiment_result(
        results_base_dir,
        experiment,
        branch,
        started_at,
    ):
    """Find the result report produced by the just-completed evaluation."""
    from eval.reporting import _read_domain_shift_result_report

    base_path = Path(str(results_base_dir)).expanduser()
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    candidates = []
    for report_path in base_path.rglob(
        f"seed{experiment.seed}_{branch}_result.txt"
    ):
        if report_path.stat().st_mtime < float(started_at) - 2.0:
            continue
        report = _read_domain_shift_result_report(report_path)
        if report is None:
            continue
        if _report_matches_experiment(report, experiment, branch):
            candidates.append((report_path.stat().st_mtime, report_path, report))
    if not candidates:
        raise RuntimeError(
            "Training finished, but no fresh matching evaluation result was "
            f"found for {experiment.name} {branch} under {base_path}."
        )
    _, report_path, report = max(candidates, key=lambda item: item[0])
    return report_path, report


def _header_index(headers, *accepted_names):
    normalized_names = {
        _normalized_header(name)
        for name in accepted_names
    }
    for index, header in enumerate(headers):
        if _normalized_header(header) in normalized_names:
            return index
    return None


def _ensure_header(headers, rows, name, *aliases):
    index = _header_index(headers, name, *aliases)
    if index is not None:
        return index
    headers.append(name)
    for row in rows:
        row.append("")
    return len(headers) - 1


def _float_cell(value):
    text = str(value or "").strip()
    if text in {"", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _table_row_sort_key(row, seed_index, test_index, original_index):
    seed_text = str(row[seed_index] or "").strip()
    seed = (
        int(seed_text)
        if re.fullmatch(r"\d+", seed_text)
        else 10**12
    )
    test_text = str(row[test_index] or "").strip().lower()
    if test_text in {"", "-", "none", "nan"}:
        return seed, 1, (), int(original_index)
    try:
        test_sequences, _ = parse_sequence_cell(test_text, "test_seq")
    except ValueError:
        return seed, 1, (), int(original_index)
    return (
        seed,
        0,
        tuple(int(sequence) for sequence in test_sequences),
        int(original_index),
    )


def _display_width(value):
    return sum(
        2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
        for character in str(value)
    )


def _pad_display(value, width, align_left):
    text = str(value)
    padding = " " * max(0, int(width) - _display_width(text))
    return text + padding if align_left else padding + text


def _format_txt_experiment_matrix(rows):
    if not rows:
        return ""
    width = len(rows[0])
    normalized_rows = []
    for row in rows:
        normalized = list(row[:width])
        if len(normalized) < width:
            normalized.extend([""] * (width - len(normalized)))
        normalized_rows.append([
            "-" if str(value).strip() == "" else str(value).strip()
            for value in normalized
        ])

    headers = normalized_rows[0]
    column_widths = [
        max(_display_width(row[index]) for row in normalized_rows)
        for index in range(width)
    ]
    test_column_index = _header_index(
        headers,
        "target test set",
        "test_seq",
    )
    left_aligned_through = 5 if test_column_index is None else test_column_index

    lines = []
    for row_index, row in enumerate(normalized_rows):
        lines.append("  ".join(
            _pad_display(
                value,
                column_widths[index],
                align_left=(
                    row_index == 0
                    or index <= left_aligned_through
                ),
            )
            for index, value in enumerate(row)
        ).rstrip())
        if row_index == 0:
            lines.append("-" * _display_width(lines[0]))
    return "\n".join(lines) + "\n"


def _write_experiment_sheet(sheet_path, rows):
    temporary_path = sheet_path.with_suffix(sheet_path.suffix + ".tmp")
    if sheet_path.suffix.lower() == ".txt":
        temporary_path.write_text(
            _format_txt_experiment_matrix(rows),
            encoding="utf-8",
        )
    else:
        with temporary_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as output_file:
            csv.writer(output_file).writerows(rows)
    temporary_path.replace(sheet_path)


def update_experiment_sheet_result(
        sheet_path,
        experiment,
        branch,
        bev_ap,
        threed_ap,
    ):
    """Write one branch result, TD values, and average TD back to the CSV."""
    resolved_path = resolve_experiment_sheet_path(sheet_path)
    rows = _read_experiment_matrix(resolved_path)
    if not rows:
        raise ValueError(f"Experiment sheet is empty: {resolved_path}")

    headers = rows[0]
    data_rows = rows[1:]
    width = len(headers)
    for row in data_rows:
        if len(row) < width:
            row.extend([""] * (width - len(row)))

    group_index = _header_index(headers, "group", "experiment")
    if group_index is None:
        group_index = 0
    seed_index = _ensure_header(headers, data_rows, "seed")
    bev_source_index = _ensure_header(
        headers,
        data_rows,
        "BEV_src",
        "bev_AP(source)",
    )
    threed_source_index = _ensure_header(
        headers,
        data_rows,
        "3D_src",
        "3d_AP(source)",
    )
    bev_target_index = _ensure_header(
        headers,
        data_rows,
        "BEV_tgt",
        "bev_AP(target)",
    )
    threed_target_index = _ensure_header(
        headers,
        data_rows,
        "3D_tgt",
        "3d_AP(target)",
    )
    td_bev_index = _ensure_header(headers, data_rows, "TD_BEV")
    td_3d_index = _ensure_header(headers, data_rows, "TD_3D")
    test_index = _header_index(
        headers,
        "target test set",
        "test_seq",
    )
    if test_index is None:
        raise ValueError(
            f"Experiment sheet has no test sequence column: {resolved_path}"
        )

    matched_row = None
    for row in data_rows:
        if (
            str(row[group_index]).strip() == experiment.name
            and _parse_seed(row[seed_index], experiment.seed) == experiment.seed
        ):
            matched_row = row
            break
    if matched_row is None:
        raise RuntimeError(
            f"Cannot find experiment row {experiment.name!r} in {resolved_path}."
        )

    if branch == "source":
        matched_row[bev_source_index] = f"{float(bev_ap):.4f}"
        matched_row[threed_source_index] = f"{float(threed_ap):.4f}"
    elif branch == "target":
        matched_row[bev_target_index] = f"{float(bev_ap):.4f}"
        matched_row[threed_target_index] = f"{float(threed_ap):.4f}"
    else:
        raise ValueError(f"Unsupported experiment branch: {branch!r}")

    complete_td_values = []
    average_row = None
    for row in data_rows:
        group_text = str(row[group_index]).strip().lower()
        if group_text.startswith("average"):
            average_row = row
            continue
        source_bev = _float_cell(row[bev_source_index])
        source_3d = _float_cell(row[threed_source_index])
        target_bev = _float_cell(row[bev_target_index])
        target_3d = _float_cell(row[threed_target_index])
        if None not in (source_bev, source_3d, target_bev, target_3d):
            td_bev = target_bev - source_bev
            td_3d = target_3d - source_3d
            row[td_bev_index] = f"{td_bev:.4f}"
            row[td_3d_index] = f"{td_3d:.4f}"
            complete_td_values.append((td_bev, td_3d))
        else:
            row[td_bev_index] = ""
            row[td_3d_index] = ""

    if average_row is None:
        average_row = [""] * len(headers)
        data_rows.append(average_row)
    average_row[group_index] = f"average({len(complete_td_values)})"
    for index in range(len(headers)):
        if index not in {group_index, td_bev_index, td_3d_index}:
            average_row[index] = ""
    if complete_td_values:
        average_row[td_bev_index] = (
            f"{sum(value[0] for value in complete_td_values) / len(complete_td_values):.4f}"
        )
        average_row[td_3d_index] = (
            f"{sum(value[1] for value in complete_td_values) / len(complete_td_values):.4f}"
        )
    else:
        average_row[td_bev_index] = ""
        average_row[td_3d_index] = ""

    ordered_rows = [
        (index, row)
        for index, row in enumerate(data_rows)
        if row is not average_row
    ]
    ordered_rows.sort(
        key=lambda item: _table_row_sort_key(
            item[1],
            seed_index=seed_index,
            test_index=test_index,
            original_index=item[0],
        )
    )
    data_rows = [row for _, row in ordered_rows] + [average_row]

    updated_matrix = [headers] + data_rows
    _write_experiment_sheet(resolved_path, updated_matrix)
    if resolved_path.suffix.lower() == ".txt":
        workbook_path = resolved_path.with_suffix(".xlsx")
        if workbook_path.is_file():
            from scripts.sync_experiment_xlsx_to_txt import (
                sync_result_matrix_to_workbook,
            )

            sync_result_matrix_to_workbook(
                workbook_path,
                updated_matrix,
            )
    return resolved_path


def validate_experiment_sheet_is_full(sheet_path):
    experiments = load_domain_shift_experiments(sheet_path)
    incomplete = []
    for experiment in experiments:
        if not experiment.source_complete:
            incomplete.append(f"{experiment.name}:source")
        if not experiment.target_complete:
            incomplete.append(f"{experiment.name}:target")
    if incomplete:
        raise RuntimeError(
            "Experiment queue ended with incomplete AP cells: "
            + ", ".join(incomplete)
        )
    return True


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
        domain_sequences = experiment.source_sequences
    elif branch == "target":
        domain_sequences = experiment.target_sequences
    else:
        raise ValueError(f"Unsupported experiment branch: {branch!r}")
    half_selection = dict(experiment.half_selection)
    sequences = {
        int(sequence)
        for sequence in (
            tuple(experiment.shared_sequences) + tuple(domain_sequences)
        )
    }
    return tuple(
        (
            sequence,
            half_selection.get(sequence, "full"),
        )
        for sequence in sorted(sequences)
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
            tuple(sorted(experiment.shared_sequences)),
            tuple(sorted(experiment.source_sequences)),
            tuple(sorted(experiment.target_sequences)),
            tuple(sorted(experiment.test_sequences)),
            tuple(sorted(experiment.half_selection)),
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
        "group": experiment.name,
        "seed": int(experiment.seed),
        "branch": task.branch,
        "shared": list(experiment.shared_sequences),
        "source": list(experiment.source_sequences),
        "target": list(experiment.target_sequences),
        "test": list(experiment.test_sequences),
        "half": list(experiment.half_selection),
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
    config.update({
        "experiment_queue_enabled": False,
        "domain_shift_train_branch": branch,
        "shared_train_sequences": experiment.shared_sequences,
        "source_train_sequences": experiment.source_sequences,
        "target_train_sequences": experiment.target_sequences,
        "target_test_sequences": experiment.test_sequences,
        "train_sequence_half_selection": dict(experiment.half_selection),
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
        f"effective_train={_effective_train_signature(experiment, task.branch)}, "
        f"test={experiment.test_sequences}, "
        f"half={dict(experiment.half_selection)}, "
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
    return report_path, report


def _run_sequential_experiment_queue(
        base_config,
        train_function,
        tasks,
        total_steps,
        sheet_path,
        results_base_dir,
        update_sheet_results,
    ):
    launched = []
    for task in tasks:
        _print_experiment_task_start(task, total_steps)
        child_config = build_experiment_training_config(
            base_config,
            task.experiment,
            task.branch,
        )
        started_at = time.time()
        train_function(child_config)
        if update_sheet_results:
            _record_experiment_result(
                sheet_path=sheet_path,
                results_base_dir=results_base_dir,
                task=task,
                evaluation_started_at=started_at,
                update_sheet_results=True,
            )
        launched.append((task.experiment.name, task.branch))
        print(
            f"Experiment queue [{task.ordinal}/{total_steps}]: completed "
            f"{task.experiment.name} {task.branch}.",
            flush=True,
        )
    return launched


def _launch_parallel_training_task(
        base_config,
        task,
        gpu_slot,
        session_dir,
    ):
    task_slug = _queue_task_slug(task)
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
    _write_training_job_config(config_path, child_config)
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-m",
        "training_utils.experiment_worker",
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
    ):
    task = pending_state["task"]
    task_slug = _queue_task_slug(task)
    log_path = session_dir / f"{task_slug}.evaluation.log"
    command, root_dir, environment = prepare_post_training_evaluation_launch(
        checkpoint_root=pending_state["checkpoint_root"],
        physical_gpu_id=physical_gpu_id,
        python_executable=sys.executable,
        project_dir=PROJECT_ROOT,
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


def run_domain_shift_experiment_queue(base_config, train_function):
    """Run a locked sequential or asynchronous table-driven experiment queue."""
    sheet_path = base_config.get("experiment_sheet_path")
    if sheet_path in (None, ""):
        raise ValueError(
            "Set experiment_sheet_path when experiment_queue_enabled=True."
        )

    ensure_no_other_top_level_train_process()
    with experiment_queue_lock(sheet_path):
        experiments = load_domain_shift_experiments(
            sheet_path,
            default_seed=base_config.get("seed", 42),
        )
        validate_experiment_queue_design(experiments)
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
        total_steps = len(experiments) * len(branches)
        tasks = build_experiment_queue_tasks(
            experiments=experiments,
            branches=branches,
            skip_completed=skip_completed,
        )
        train_workers = int(
            base_config.get("experiment_queue_train_workers", 1)
        )
        evaluation_workers = int(
            base_config.get("experiment_queue_eval_workers", 1)
        )

        print(
            "Experiment queue: "
            f"{resolve_experiment_sheet_path(sheet_path)} "
            f"({len(experiments)} rows, branches={branches}, "
            f"pending={len(tasks)}, skip_completed={skip_completed})",
            flush=True,
        )

        if train_workers > 1 or evaluation_workers > 1:
            launched = _run_parallel_experiment_queue(
                base_config=base_config,
                tasks=tasks,
                total_steps=total_steps,
                sheet_path=sheet_path,
                results_base_dir=results_base_dir,
                update_sheet_results=update_sheet_results,
            )
        else:
            launched = _run_sequential_experiment_queue(
                base_config=base_config,
                train_function=train_function,
                tasks=tasks,
                total_steps=total_steps,
                sheet_path=sheet_path,
                results_base_dir=results_base_dir,
                update_sheet_results=update_sheet_results,
            )

        if bool(
            base_config.get("experiment_queue_require_full_table", False)
        ):
            validate_experiment_sheet_is_full(sheet_path)
        print(
            f"Experiment queue completed: {len(launched)} "
            "training/evaluation runs.",
            flush=True,
        )
        return launched
