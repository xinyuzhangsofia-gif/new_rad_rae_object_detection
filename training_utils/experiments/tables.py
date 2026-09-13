"""Experiment-table discovery, parsing, validation, and result updates."""

import csv
import re
import time
import unicodedata
from pathlib import Path

from training_utils.experiments.schema import (
    DomainShiftExperiment,
    VALID_BRANCHES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_experiment_sheet_path(path):
    sheet_path = Path(str(path)).expanduser()
    if not sheet_path.is_absolute():
        sheet_path = PROJECT_ROOT / sheet_path
    return sheet_path.resolve()


def resolve_experiment_sheet_paths(base_config):
    """Resolve an ordered multi-weather table list with single-table fallback."""
    configured_paths = base_config.get("experiment_sheet_paths")
    if configured_paths in (None, "", ()):
        configured_paths = base_config.get("experiment_sheet_path")
    if configured_paths in (None, "", ()):
        raise ValueError(
            "Set experiment_sheet_paths (or legacy experiment_sheet_path) "
            "when experiment_queue_enabled=True."
        )
    if isinstance(configured_paths, (str, Path)):
        configured_paths = (configured_paths,)
    else:
        configured_paths = tuple(configured_paths)
    if not configured_paths:
        raise ValueError("experiment_sheet_paths must not be empty.")

    resolved_paths = tuple(
        resolve_experiment_sheet_path(path)
        for path in configured_paths
    )
    duplicate_paths = sorted({
        str(path)
        for path in resolved_paths
        if resolved_paths.count(path) > 1
    })
    if duplicate_paths:
        raise ValueError(
            "experiment_sheet_paths contains duplicate tables: "
            f"{duplicate_paths}"
        )
    return resolved_paths


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


def parse_sequence_cell_parts(value, column_name):
    """Parse sequence tokens while preserving independent first/last parts."""
    text = _normalize_sequence_cell(value)
    if text in {"", "-", "none", "nan"}:
        raise ValueError(f"{column_name} cannot be empty")

    sequences = []
    parts = []
    positions_by_sequence = {}
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
        half = match.group(2) or "full"
        if sequence not in sequences:
            sequences.append(sequence)
        part = (sequence, half)
        if part in parts:
            raise ValueError(
                f"Sequence part {sequence}_{half} is duplicated in "
                f"{column_name}: {value!r}"
            )
        parts.append(part)
        positions_by_sequence.setdefault(sequence, set()).add(half)

    if not sequences:
        raise ValueError(f"{column_name} cannot be empty")

    half_selection = {}
    for sequence, positions in positions_by_sequence.items():
        if "full" in positions and len(positions) > 1:
            raise ValueError(
                f"Sequence {sequence} cannot combine a full selection with "
                f"first/last selections in {column_name}: {value!r}"
            )
        if positions == {"first"}:
            half_selection[sequence] = "first"
        elif positions == {"last"}:
            half_selection[sequence] = "last"
        elif positions not in ({"full"}, {"first", "last"}):
            raise ValueError(
                f"Unsupported selections for sequence {sequence} in "
                f"{column_name}: {sorted(positions)}"
            )

    return tuple(sequences), half_selection, tuple(parts)


def parse_sequence_cell(value, column_name):
    """Parse a sequence cell and return its unique IDs and effective halves."""
    sequences, half_selection, _parts = parse_sequence_cell_parts(
        value,
        column_name,
    )
    return sequences, half_selection


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


def _half_selection_for_parts(*part_groups):
    selections = []
    for parts in part_groups:
        positions_by_sequence = {}
        for sequence, position in parts:
            positions_by_sequence.setdefault(int(sequence), set()).add(
                str(position)
            )
        selection = {}
        for sequence, positions in positions_by_sequence.items():
            if positions == {"first"}:
                selection[sequence] = "first"
            elif positions == {"last"}:
                selection[sequence] = "last"
        selections.append(selection)
    return dict(_merge_half_selections(*selections))


def _branch_half_selection(experiment, branch):
    active_parts = (
        experiment.source_parts
        if branch == "source"
        else experiment.target_parts
    )
    return _half_selection_for_parts(
        experiment.shared_parts,
        active_parts,
    )


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

        shared, shared_half, shared_parts = parse_sequence_cell_parts(
            row.get(shared_column),
            "shared train sequence",
        )
        source, source_half, source_parts = parse_sequence_cell_parts(
            row.get(source_column),
            "source train set",
        )
        target, target_half, target_parts = parse_sequence_cell_parts(
            row.get(target_column),
            "target train set",
        )
        test, test_half, test_parts = parse_sequence_cell_parts(
            row.get(test_column),
            "target test set",
        )
        if any(position != "full" for _sequence, position in test_parts):
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
            shared_parts=shared_parts,
            source_parts=source_parts,
            target_parts=target_parts,
            test_parts=test_parts,
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
        == tuple(sorted(_branch_half_selection(experiment, branch).items()))
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


