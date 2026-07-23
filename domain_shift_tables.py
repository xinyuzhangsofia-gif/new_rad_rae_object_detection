from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEQUENCE_INFO_PATH = PROJECT_ROOT / "sequence_information.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"

BEV_METRIC_KEY = "official_bev_mAP_0.3"
THREED_METRIC_KEY = "official_3d_mAP_0.3"
OVERALL_SCORE_FORMULA = (
    "(official_bev_mAP_0.3 + official_3d_mAP_0.3) / 2"
)

TABLE_SPECS = (
    ("best_bev", "table1_best_bev.txt"),
    ("best_3d", "table2_best_3d.txt"),
    ("best_overall", "table3_best_overall.txt"),
)
TABLE_CORNER_HEADER = "Source -> Target (APBEV/AP3D)"
LEGACY_TABLE_CORNER_HEADER = "Target domain"

WEATHER_ORDER = (
    "normal",
    "overcast",
    "rain",
    "fog",
    "sleet",
    "lightsnow",
    "heavysnow",
)
WEATHER_DISPLAY_NAMES = {
    "normal": "Normal",
    "overcast": "Overcast",
    "rain": "Rain",
    "fog": "Fog",
    "sleet": "Sleet",
    "lightsnow": "Light Snow",
    "heavysnow": "Heavy Snow",
}


class DomainShiftTableError(RuntimeError):
    pass


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_sequence_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, bool):
        raise ValueError(f"Boolean value is not a sequence ID: {value!r}")
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, (list, tuple, set)):
        sequence_ids: list[int] = []
        for item in value:
            sequence_ids.extend(normalize_sequence_ids(item))
        return tuple(sorted(set(sequence_ids)))

    text = str(value).strip().lower()
    if text in {"", "none", "null", "unknown", "train_unknown", "val_unknown"}:
        return ()
    text = re.sub(r"\b(?:train|val|test)[_-]?sequences?\b", "", text)
    text = re.sub(r"\b(?:train|val|test)[_-]?seq\b", "", text)
    text = re.sub(r"\bseq", "", text)

    sequence_ids = []
    for match in re.finditer(r"(?<!\d)(\d+)\s*-\s*(\d+)(?!\d)|(?<!\d)(\d+)(?!\d)", text):
        if match.group(3) is not None:
            sequence_ids.append(int(match.group(3)))
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        low, high = sorted((start, end))
        sequence_ids.extend(range(low, high + 1))
    return tuple(sorted(set(sequence_ids)))


def load_sequence_information(
    path: str | os.PathLike[str] = DEFAULT_SEQUENCE_INFO_PATH,
) -> dict[int, dict[str, Any]]:
    sequence_path = Path(path)
    if not sequence_path.is_file():
        raise FileNotFoundError(f"Sequence information file not found: {sequence_path}")

    required_columns = {
        "sequence_id",
        "environment",
        "time",
        "weather",
        "frames",
        "empty_frames",
        "empty_rate_percent",
        "object_count",
        "top_classes",
    }
    information: dict[int, dict[str, Any]] = {}
    with sequence_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise DomainShiftTableError(
                f"Sequence information is missing columns: {sorted(missing_columns)}"
            )
        for row in reader:
            sequence_id = int(row["sequence_id"])
            if sequence_id in information:
                raise DomainShiftTableError(
                    f"Duplicate sequence ID {sequence_id} in {sequence_path}"
                )
            information[sequence_id] = {
                "sequence_id": sequence_id,
                "environment": row["environment"].strip().lower(),
                "time": row["time"].strip().lower(),
                "weather": row["weather"].strip().lower(),
                "frames": int(row["frames"]),
                "empty_frames": int(row["empty_frames"]),
                "empty_rate_percent": int(row["empty_rate_percent"]),
                "object_count": int(row["object_count"]),
                "top_classes": row["top_classes"].strip(),
            }
    return information


def _ordered_unique(values: Iterable[str], preferred_order: Iterable[str]) -> list[str]:
    unique_values = set(values)
    ordered = [value for value in preferred_order if value in unique_values]
    ordered.extend(sorted(unique_values.difference(ordered)))
    return ordered


def weather_display_name(weather: str) -> str:
    return WEATHER_DISPLAY_NAMES.get(weather, weather.replace("_", " ").title())


def _domain_registry_key(
    role: str,
    sequence_ids: tuple[int, ...],
    controlled: bool,
) -> str:
    sequence_text = ",".join(str(sequence_id) for sequence_id in sequence_ids)
    return f"{role}|{sequence_text}|controlled={int(controlled)}"


def _domain_label(entry: Mapping[str, Any]) -> str:
    name = str(entry["name"])
    if entry.get("controlled", False):
        name = f"{name} Controlled"
    sequence_text = ", ".join(str(value) for value in entry["sequence_ids"])
    details = [f"Sequences: {sequence_text}"]
    weather_conditions = list(entry.get("weather_conditions", ()))
    if len(weather_conditions) > 1:
        weather_text = " + ".join(
            weather_display_name(weather) for weather in weather_conditions
        )
        details.append(f"Weather: {weather_text}")
    return f"{name} ({'; '.join(details)})"


def resolve_domain_descriptor(
    registry: dict[str, Any],
    sequence_information: Mapping[int, Mapping[str, Any]],
    role: str,
    sequences: Any,
    controlled: bool = False,
) -> dict[str, Any]:
    if role not in {"source", "target"}:
        raise ValueError(f"Unsupported domain role: {role!r}")
    sequence_ids = normalize_sequence_ids(sequences)
    if not sequence_ids:
        raise DomainShiftTableError(f"{role.title()} domain has no sequence IDs.")

    missing_ids = [
        sequence_id
        for sequence_id in sequence_ids
        if sequence_id not in sequence_information
    ]
    if missing_ids:
        raise DomainShiftTableError(
            f"Sequence information is missing IDs used by the {role} domain: {missing_ids}"
        )

    rows = [sequence_information[sequence_id] for sequence_id in sequence_ids]
    weather_conditions = _ordered_unique(
        (str(row["weather"]) for row in rows),
        WEATHER_ORDER,
    )
    non_normal_weather = [
        weather for weather in weather_conditions if weather != "normal"
    ]
    main_weather = non_normal_weather or weather_conditions
    base_name = " + ".join(
        weather_display_name(weather) for weather in main_weather
    )
    registry_key = _domain_registry_key(role, sequence_ids, bool(controlled))
    entries = registry.setdefault("domains", {})
    entry = entries.get(registry_key)

    if entry is None:
        existing_ordinals = [
            int(candidate.get("ordinal", 0))
            for candidate in entries.values()
            if candidate.get("role") == role
            and candidate.get("base_name") == base_name
        ]
        ordinal = max(existing_ordinals, default=0) + 1
        entry = {
            "key": registry_key,
            "role": role,
            "sequence_ids": list(sequence_ids),
            "weather_conditions": weather_conditions,
            "environment_conditions": _ordered_unique(
                (str(row["environment"]) for row in rows),
                (),
            ),
            "time_conditions": _ordered_unique(
                (str(row["time"]) for row in rows),
                ("day", "night"),
            ),
            "main_weather_conditions": main_weather,
            "base_name": base_name,
            "ordinal": ordinal,
            "name": f"{base_name} {ordinal}",
            "controlled": bool(controlled),
            "created_at": utc_now_text(),
        }
        entries[registry_key] = entry

    descriptor = dict(entry)
    descriptor["label"] = _domain_label(entry)
    descriptor["sequence_information"] = [dict(row) for row in rows]
    return descriptor


def _strip_source_from_model_variant(model_variant_name: str) -> str:
    variant = str(model_variant_name or "model_unknown").strip()
    variant = re.split(r"_(?:ig_)?train_", variant, maxsplit=1)[0]
    return variant.removesuffix("_controlled")


def _format_float_for_name(value: float) -> str:
    return f"{float(value):.8g}".replace("+", "")


def build_model_configuration(
    model_type: str,
    model_variant_name: str | None = None,
    checkpoint_config: Mapping[str, Any] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    checkpoint_config = dict(checkpoint_config or {})
    model_overrides = dict(model_overrides or {})
    base_name = _strip_source_from_model_variant(model_variant_name or model_type)

    decoder_channels = model_overrides.get("decoder_hidden_channels")
    feature_channels = model_overrides.get("feature_channels")
    if base_name == str(model_type) and decoder_channels is not None:
        base_name += f"_{int(decoder_channels)}"
        if feature_channels is not None:
            base_name += f"_{int(feature_channels)}"

    learning_rate = checkpoint_config.get(
        "lr",
        checkpoint_config.get("learning_rate"),
    )
    batch_size = checkpoint_config.get("batch_size")
    seed = checkpoint_config.get("seed")
    heatmap_radius = checkpoint_config.get("heatmap_radius")
    gwd_loss_weight = checkpoint_config.get(
        "centerpoint_gwd_loss_weight",
        checkpoint_config.get("centerpoint_giou_loss_weight"),
    )
    quality_loss_weight = checkpoint_config.get("quality_loss_weight")
    quality_loss_active = str(model_type) == "model6"

    learning_rate_name = (
        "unknown"
        if learning_rate is None
        else _format_float_for_name(float(learning_rate))
    )
    batch_size_name = "unknown" if batch_size is None else str(int(batch_size))
    seed_name = "unknown" if seed is None else str(int(seed))
    if None in (heatmap_radius, gwd_loss_weight):
        loss_name = "unknown"
    else:
        loss_parts = [
            f"hm{int(heatmap_radius)}",
            f"gwd{_format_float_for_name(float(gwd_loss_weight))}",
        ]
        if quality_loss_active:
            quality_name = (
                "unknown"
                if quality_loss_weight is None
                else _format_float_for_name(float(quality_loss_weight))
            )
            loss_parts.append(f"q{quality_name}")
        loss_name = "_".join(loss_parts)
    config_name = (
        f"{base_name}_lr{learning_rate_name}_bs{batch_size_name}_"
        f"seed{seed_name}_loss_{loss_name}"
    )

    configuration = {
        "name": config_name,
        "model_type": str(model_type),
        "model_variant_base": base_name,
        "decoder_hidden_channels": (
            None if decoder_channels is None else int(decoder_channels)
        ),
        "feature_channels": (
            None if feature_channels is None else int(feature_channels)
        ),
        "learning_rate": (
            None if learning_rate is None else float(learning_rate)
        ),
        "batch_size": batch_size,
        "max_detections": checkpoint_config.get(
            "max_detections",
            checkpoint_config.get("num_boxes"),
        ),
        "heatmap_radius": heatmap_radius,
        "centerpoint_gwd_loss_weight": gwd_loss_weight,
        "quality_loss_active": quality_loss_active,
        "quality_loss_weight": (
            quality_loss_weight if quality_loss_active else None
        ),
        "train_scope": checkpoint_config.get("train_scope"),
        "train_ratio": checkpoint_config.get("train_ratio"),
        "seed": seed,
    }
    return config_name, configuration


def _finite_metric(result: Mapping[str, Any], metric_key: str) -> float | None:
    value = result.get(metric_key)
    if value is None:
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def select_comparison_epochs(
    results: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidates = []
    for result in results:
        if result.get("epoch") is None:
            continue
        bev_ap = _finite_metric(result, BEV_METRIC_KEY)
        threed_ap = _finite_metric(result, THREED_METRIC_KEY)
        if bev_ap is None or threed_ap is None:
            continue
        candidates.append({
            "epoch": int(result["epoch"]),
            "bev_ap_0.3": bev_ap,
            "3d_ap_0.3": threed_ap,
            "overall_score": (bev_ap + threed_ap) / 2.0,
            "checkpoint_path": str(result.get("checkpoint_path", "")),
        })
    if not candidates:
        raise DomainShiftTableError(
            f"No completed epoch contains both {BEV_METRIC_KEY} and {THREED_METRIC_KEY}."
        )

    def select(metric_key: str) -> dict[str, Any]:
        return max(
            candidates,
            key=lambda candidate: (
                float(candidate[metric_key]),
                -int(candidate["epoch"]),
            ),
        )

    return {
        "best_bev": dict(select("bev_ap_0.3")),
        "best_3d": dict(select("3d_ap_0.3")),
        "best_overall": dict(select("overall_score")),
    }


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("._")
    return sanitized or "unknown"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(data, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
    )


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    with path.open("r", encoding="utf-8") as input_file:
        loaded = json.load(input_file)
    if not isinstance(loaded, dict):
        raise DomainShiftTableError(f"Expected a JSON object in {path}")
    return loaded


def _is_text_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"[-+| ]+", line)) and "-" in line


def _normalize_comparison_table_rows(
    rows: list[list[str]],
    path: Path,
) -> tuple[list[str], list[list[str]]]:
    if not rows:
        return [TABLE_CORNER_HEADER], []
    header = list(rows[0])
    if header and header[0] == LEGACY_TABLE_CORNER_HEADER:
        return _convert_legacy_weather_comparison_table(header, rows[1:])
    if not header or header[0] != TABLE_CORNER_HEADER:
        raise DomainShiftTableError(
            f"Invalid comparison table header in {path}: {header!r}"
        )
    normalized_rows = []
    for row in rows[1:]:
        normalized_rows.append(list(row) + [""] * (len(header) - len(row)))
    if any(label.startswith("-> ") for label in header[1:]):
        return _transpose_source_rows_to_weather_table(header, normalized_rows)
    return header, normalized_rows


def _read_comparison_table(
    path: Path,
) -> tuple[list[str], list[list[str]], list[str]]:
    if not path.is_file():
        return [TABLE_CORNER_HEADER], [], []
    raw_rows = []
    context_lines = []
    table_started = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not table_started and TABLE_CORNER_HEADER in stripped:
            table_started = True
        elif not table_started:
            # Preserve indentation for multiline context lists.
            context_lines.append(line.rstrip())
            continue
        if _is_text_table_separator(stripped):
            continue
        if "|" in line:
            raw_rows.append([cell.strip() for cell in line.split("|")])
        elif not raw_rows:
            raw_rows.append([stripped])
    header, rows = _normalize_comparison_table_rows(raw_rows, path)
    return header, rows, context_lines


def _paper_target_label(label: str) -> str:
    return f"-> {_short_domain_label(label)}"


def _paper_source_label(label: str) -> str:
    return f"{_short_domain_label(label)} ->"


def _short_domain_label(label: str) -> str:
    normalized = str(label).strip()
    if normalized.startswith("-> "):
        normalized = normalized[3:].strip()
    if normalized.endswith(" ->"):
        normalized = normalized[:-3].strip()
    return re.sub(r"\s+\(Sequences:.*\)$", "", normalized)


def _merge_context_lines(
    existing_lines: Iterable[str],
    incoming_lines: Iterable[str],
) -> list[str]:
    merged = []
    for line in [*existing_lines, *incoming_lines]:
        normalized = str(line)
        if normalized.strip() and normalized not in merged:
            merged.append(normalized)
    return merged


def _remove_context_block(lines: Iterable[str], key: str) -> list[str]:
    result = []
    skipping_items = False
    for line in lines:
        normalized = str(line)
        stripped = normalized.strip()
        if stripped == key or stripped.startswith(f"{key} "):
            skipping_items = True
            continue
        if skipping_items and normalized.lstrip().startswith("- "):
            continue
        skipping_items = False
        result.append(normalized)
    return result


def _format_table_cell(selection: Mapping[str, Any]) -> str:
    return (
        f"{float(selection['bev_ap_0.3']):.2f}/"
        f"{float(selection['3d_ap_0.3']):.2f}"
    )


def _format_legacy_table_cell(value: str) -> str:
    match = re.search(
        r"BEV@0\.3=([-+]?\d+(?:\.\d+)?)\s*;\s*"
        r"3D@0\.3=([-+]?\d+(?:\.\d+)?)",
        value,
    )
    if match is None:
        return value
    return f"{float(match.group(1)):.2f}/{float(match.group(2)):.2f}"


def _convert_legacy_weather_comparison_table(
    legacy_header: list[str],
    legacy_rows: list[list[str]],
) -> tuple[list[str], list[list[str]]]:
    normalized_legacy_rows = [
        list(row) + [""] * (len(legacy_header) - len(row))
        for row in legacy_rows
        if row
    ]
    header = [TABLE_CORNER_HEADER] + [
        _paper_source_label(source_label)
        for source_label in legacy_header[1:]
    ]
    rows = [
        [
            _paper_target_label(target_row[0]),
            *[
                _format_legacy_table_cell(value)
                for value in target_row[1:len(legacy_header)]
            ],
        ]
        for target_row in normalized_legacy_rows
    ]
    return header, rows


def _transpose_source_rows_to_weather_table(
    source_rows_header: list[str],
    source_rows: list[list[str]],
) -> tuple[list[str], list[list[str]]]:
    header = [TABLE_CORNER_HEADER] + [row[0] for row in source_rows]
    rows = []
    for target_index, target_label in enumerate(
        source_rows_header[1:],
        start=1,
    ):
        rows.append([
            target_label,
            *[
                _format_legacy_table_cell(source_row[target_index])
                for source_row in source_rows
            ],
        ])
    return header, rows


def _write_comparison_table(
    path: Path,
    header: list[str],
    rows: list[list[str]],
    context_lines: Iterable[str] = (),
) -> None:
    all_rows = [header, *rows]
    column_count = max((len(row) for row in all_rows), default=1)
    normalized_rows = [
        [str(cell) for cell in row] + [""] * (column_count - len(row))
        for row in all_rows
    ]
    widths = [
        max(len(row[column_index]) for row in normalized_rows)
        for column_index in range(column_count)
    ]

    def render_row(row: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        ).rstrip()

    normalized_context = list(context_lines)
    for context_key in ("source_domains:", "target_domains:"):
        normalized_context = _remove_context_block(
            normalized_context,
            context_key,
        )
    if len(header) > 1:
        normalized_context.append("source_domains:")
        normalized_context.extend(f"  - {label}" for label in header[1:])
    if rows:
        normalized_context.append("target_domains:")
        normalized_context.extend(f"  - {row[0]}" for row in rows)
    normalized_context = [
        line for line in normalized_context if str(line).strip()
    ]
    output_lines = normalized_context
    if output_lines:
        output_lines.append("")
    output_lines.append(render_row(normalized_rows[0]))
    output_lines.append("-+-".join("-" * width for width in widths))
    output_lines.extend(render_row(row) for row in normalized_rows[1:])
    _atomic_write_text(path, "\n".join(output_lines) + "\n")


def remove_target_from_comparison_table(
    path: str | os.PathLike[str],
    target_label: str,
    context_lines: Iterable[str] = (),
) -> int:
    table_path = Path(path)
    if not table_path.is_file():
        _write_comparison_table(
            table_path,
            [TABLE_CORNER_HEADER],
            [],
            context_lines,
        )
        return 0
    header, rows, existing_context_lines = _read_comparison_table(table_path)
    target_row_label = _paper_target_label(target_label)
    filtered_rows = [
        row for row in rows if not row or row[0] != target_row_label
    ]
    removed_count = len(rows) - len(filtered_rows)
    if removed_count:
        _write_comparison_table(
            table_path,
            header,
            filtered_rows,
            _merge_context_lines(existing_context_lines, context_lines),
        )
    return removed_count


def add_comparison_table_context(
    path: str | os.PathLike[str],
    context_lines: Iterable[str],
) -> None:
    table_path = Path(path)
    header, rows, existing_context_lines = _read_comparison_table(table_path)
    short_header = [
        header[0],
        *[_paper_source_label(label) for label in header[1:]],
    ]
    short_rows = [
        [_paper_target_label(row[0]), *row[1:]]
        for row in rows
    ]
    _write_comparison_table(
        table_path,
        short_header,
        short_rows,
        _merge_context_lines(existing_context_lines, context_lines),
    )


def convert_legacy_csv_table(
    csv_path: str | os.PathLike[str],
    text_path: str | os.PathLike[str],
) -> None:
    source_path = Path(csv_path)
    target_path = Path(text_path)
    with source_path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.reader(input_file))
    header, normalized_rows = _normalize_comparison_table_rows(rows, source_path)
    _write_comparison_table(target_path, header, normalized_rows)


def _update_comparison_table(
    path: Path,
    source_label: str,
    target_label: str,
    cell_value: str | None,
    context_lines: Iterable[str] = (),
) -> None:
    header, rows, existing_context_lines = _read_comparison_table(path)
    source_column_label = _paper_source_label(source_label)
    if source_column_label not in header:
        header.append(source_column_label)
        for row in rows:
            row.append("")
    source_index = header.index(source_column_label)

    target_row_label = _paper_target_label(target_label)
    target_row = next(
        (row for row in rows if row and row[0] == target_row_label),
        None,
    )
    if target_row is None:
        target_row = [target_row_label] + [""] * (len(header) - 1)
        rows.append(target_row)
    if len(target_row) < len(header):
        target_row.extend([""] * (len(header) - len(target_row)))
    if cell_value is not None:
        target_row[source_index] = cell_value

    _write_comparison_table(
        path,
        header,
        rows,
        _merge_context_lines(existing_context_lines, context_lines),
    )


def _record_file_name(
    source_sequences: tuple[int, ...],
    target_sequences: tuple[int, ...],
    controlled: bool,
) -> str:
    source_text = "-".join(str(value) for value in source_sequences)
    target_text = "-".join(str(value) for value in target_sequences)
    controlled_suffix = "_controlled" if controlled else ""
    return f"source_seq{source_text}{controlled_suffix}__target_seq{target_text}.json"


def _comparison_table_context(
    metadata: Mapping[str, Any],
    evaluation_group: str,
    criterion: str,
    source_label: str,
    target_label: str,
) -> list[str]:
    selection_metrics = {
        "best_bev": "official_bev_mAP_0.3",
        "best_3d": "official_3d_mAP_0.3",
        "best_overall": OVERALL_SCORE_FORMULA,
    }
    return [
        f"model_type: {metadata.get('model_type', 'unknown')}",
        f"model_variant: {metadata.get('model_type', 'unknown')}",
        f"model_configuration: {metadata.get('model_configuration_name', 'unknown')}",
        f"evaluation_group: {evaluation_group}",
        f"include_bus_as_target: {evaluation_group == 'before'}",
        f"bus_ignored_during_evaluation: {evaluation_group == 'after'}",
        f"train_sequences: {metadata.get('train_sequences', 'unknown')}",
        f"val_sequences: {metadata.get('val_sequences', 'unknown')}",
        f"train_control_split_enabled: {bool(metadata.get('train_control_split_enabled', False))}",
        f"eval_scope: {metadata.get('eval_scope', 'unknown')}",
        f"ap_score_thresh: {metadata.get('ap_score_thresh', 'unknown')}",
        f"score_thresh: {metadata.get('score_thresh', 'unknown')}",
        f"source_evaluation_txt: {metadata.get('source_txt_path', 'unknown')}",
        f"current_source_domain: {source_label}",
        f"current_target_domain: {target_label}",
        "source_domain_details:",
        f"  - {source_label}",
        "target_domain_details:",
        f"  - {target_label}",
        f"table_selection: {criterion}",
        f"selection_metric: {selection_metrics[criterion]}",
        "selection_rule: one best epoch per source-target pair",
        "metric_pair: BEV@0.3/3D@0.3",
        "normal_only_target_rows: excluded",
    ]


def update_domain_shift_tables(
    results: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
    sequence_info_path: str | os.PathLike[str] = DEFAULT_SEQUENCE_INFO_PATH,
) -> dict[str, Any]:
    result_list = [dict(result) for result in results]
    selections = select_comparison_epochs(result_list)
    sequence_information = load_sequence_information(sequence_info_path)

    source_sequences = normalize_sequence_ids(metadata.get("train_sequences"))
    target_sequences = normalize_sequence_ids(metadata.get("val_sequences"))
    controlled = bool(metadata.get("train_control_split_enabled", False))
    include_bus_as_target = bool(metadata.get("include_bus_as_target", True))
    evaluation_group = "before" if include_bus_as_target else "after"

    model_configuration_name = str(
        metadata.get("model_configuration_name")
        or _strip_source_from_model_variant(
            str(metadata.get("model_type", "model_unknown"))
        )
    )
    model_configuration_name = _sanitize_path_component(model_configuration_name)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".domain_shift_tables.lock"

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        registry_path = output_root / "domain_registry.json"
        registry = _load_json(
            registry_path,
            {
                "version": 1,
                "sequence_information_source": str(Path(sequence_info_path)),
                "domains": {},
            },
        )
        source_domain = resolve_domain_descriptor(
            registry,
            sequence_information,
            "source",
            source_sequences,
            controlled=controlled,
        )
        target_domain = resolve_domain_descriptor(
            registry,
            sequence_information,
            "target",
            target_sequences,
            controlled=False,
        )
        target_recorded_in_table = target_domain["weather_conditions"] != [
            "normal"
        ]

        configuration_dir = output_root / model_configuration_name
        configuration_path = configuration_dir / "configuration.json"
        existing_configuration_data = _load_json(configuration_path, {})
        observed_settings = list(
            existing_configuration_data.get("observed_parameter_settings") or ()
        )
        current_settings = dict(metadata.get("model_configuration") or {})
        current_settings_key = json.dumps(
            current_settings,
            sort_keys=True,
            ensure_ascii=True,
        )
        observed_settings_keys = {
            json.dumps(settings, sort_keys=True, ensure_ascii=True)
            for settings in observed_settings
        }
        if current_settings_key not in observed_settings_keys:
            observed_settings.append(current_settings)

        configuration_data = {
            "model_configuration_name": model_configuration_name,
            "identity": {
                key: current_settings.get(key)
                for key in (
                    "model_type",
                    "model_variant_base",
                    "decoder_hidden_channels",
                    "feature_channels",
                    "learning_rate",
                    "batch_size",
                    "seed",
                    "heatmap_radius",
                    "centerpoint_gwd_loss_weight",
                    "quality_loss_active",
                    "quality_loss_weight",
                )
            },
            "observed_parameter_settings": observed_settings,
            "metric_unit": "percentage_points",
            "bev_selection_metric": BEV_METRIC_KEY,
            "3d_selection_metric": THREED_METRIC_KEY,
            "overall_score_formula": OVERALL_SCORE_FORMULA,
            "updated_at": utc_now_text(),
        }
        _write_json(configuration_path, configuration_data)

        table_paths = {}
        for group_name in ("before", "after"):
            candidate_group_dir = configuration_dir / group_name
            candidate_group_dir.mkdir(parents=True, exist_ok=True)
            for criterion, file_name in TABLE_SPECS:
                table_path = candidate_group_dir / file_name
                context_lines = _comparison_table_context(
                    metadata,
                    group_name,
                    criterion,
                    source_domain["label"],
                    target_domain["label"],
                )
                if group_name == evaluation_group:
                    table_paths[criterion] = str(table_path)
                if not target_recorded_in_table:
                    remove_target_from_comparison_table(
                        table_path,
                        target_domain["label"],
                        context_lines,
                    )
                    continue
                cell_value = None
                if group_name == evaluation_group:
                    cell_value = _format_table_cell(selections[criterion])
                _update_comparison_table(
                    table_path,
                    source_label=source_domain["label"],
                    target_label=target_domain["label"],
                    cell_value=cell_value,
                    context_lines=context_lines,
                )

        group_dir = configuration_dir / evaluation_group

        record = {
            "status": "complete",
            "updated_at": utc_now_text(),
            "model_configuration_name": model_configuration_name,
            "evaluation_group": evaluation_group,
            "bus_ignored_during_evaluation": not include_bus_as_target,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "selection": {
                "bev_metric": BEV_METRIC_KEY,
                "3d_metric": THREED_METRIC_KEY,
                "overall_score_formula": OVERALL_SCORE_FORMULA,
                **selections,
            },
            "epochs": [
                {
                    "epoch": int(result["epoch"]),
                    "checkpoint_path": str(result.get("checkpoint_path", "")),
                    "bev_ap_0.3": _finite_metric(result, BEV_METRIC_KEY),
                    "3d_ap_0.3": _finite_metric(result, THREED_METRIC_KEY),
                }
                for result in result_list
                if result.get("epoch") is not None
            ],
            "evaluation_metadata": dict(metadata),
        }
        record_path = group_dir / "records" / _record_file_name(
            source_sequences,
            target_sequences,
            controlled,
        )
        _write_json(record_path, record)
        registry["updated_at"] = utc_now_text()
        _write_json(registry_path, registry)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return {
        "model_configuration_name": model_configuration_name,
        "evaluation_group": evaluation_group,
        "source_domain": source_domain["label"],
        "target_domain": target_domain["label"],
        "target_recorded_in_table": target_recorded_in_table,
        "selections": selections,
        "table_paths": table_paths,
        "record_path": str(record_path),
    }


def _parse_bool_text(value: Any, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise DomainShiftTableError(f"Invalid boolean value in evaluation TXT: {value!r}")


def parse_evaluation_table_txt(
    path: str | os.PathLike[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table_path = Path(path)
    lines = table_path.read_text(encoding="utf-8").splitlines()
    raw_metadata: dict[str, str] = {}
    table_header_index = None
    table_headers: list[str] = []

    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("epoch") and "bev@0.3" in stripped and "3d@0.3" in stripped:
            table_header_index = line_index
            table_headers = re.split(r"\s+", stripped)
            break
        if ":" in stripped and not stripped.startswith("best_epoch_"):
            key, value = stripped.split(":", 1)
            raw_metadata[key.strip()] = value.strip()

    if table_header_index is None:
        raise DomainShiftTableError(
            f"No completed epoch metric table found in {table_path}"
        )
    required_metric_headers = {"epoch", "bev@0.3", "3d@0.3"}
    if not required_metric_headers.issubset(table_headers):
        raise DomainShiftTableError(
            f"Missing required metric columns in {table_path}: {table_headers}"
        )

    results = []
    for line in lines[table_header_index + 1:]:
        stripped = line.strip()
        if not stripped or set(stripped) == {"-"}:
            continue
        values = re.split(r"\s+", stripped)
        if not values[0].isdigit() or len(values) < len(table_headers):
            continue
        row = dict(zip(table_headers, values))
        try:
            results.append({
                "epoch": int(row["epoch"]),
                BEV_METRIC_KEY: float(row["bev@0.3"]),
                THREED_METRIC_KEY: float(row["3d@0.3"]),
                "checkpoint_path": "",
            })
        except ValueError as error:
            raise DomainShiftTableError(
                f"Invalid numeric epoch row in {table_path}: {line!r}"
            ) from error
    if not results:
        raise DomainShiftTableError(f"No valid epoch rows found in {table_path}")

    expected_best_epochs = {}
    for line in lines[:table_header_index]:
        match = re.match(
            r"best_epoch_(bev|3d)@0\.3:\s*epoch\s+(\d+)",
            line.strip(),
        )
        if match:
            expected_best_epochs[match.group(1)] = int(match.group(2))
    selections = select_comparison_epochs(results)
    computed_best_epochs = {
        "bev": int(selections["best_bev"]["epoch"]),
        "3d": int(selections["best_3d"]["epoch"]),
    }
    for metric_name, expected_epoch in expected_best_epochs.items():
        if computed_best_epochs[metric_name] != expected_epoch:
            raise DomainShiftTableError(
                f"Best-{metric_name} epoch validation failed for {table_path}: "
                f"file says {expected_epoch}, parsed rows give "
                f"{computed_best_epochs[metric_name]}."
            )

    model_variant_name = raw_metadata.get(
        "model_variant",
        raw_metadata.get("model_type", table_path.parent.name),
    )
    base_model_match = re.match(r"(model\d+)", str(model_variant_name))
    base_model_type = (
        raw_metadata.get("model_type")
        if raw_metadata.get("model_variant")
        else None
    )
    if not base_model_type or not re.fullmatch(r"model\d+", base_model_type):
        base_model_type = (
            base_model_match.group(1) if base_model_match else "model_unknown"
        )

    include_bus_as_target = _parse_bool_text(
        raw_metadata.get("include_bus_as_target"),
        default="_ig_" not in str(model_variant_name),
    )
    controlled = _parse_bool_text(
        raw_metadata.get("train_control_split_enabled"),
        default=str(model_variant_name).endswith("_controlled"),
    )
    metadata = {
        "source_txt_path": str(table_path),
        "model_type": str(model_variant_name),
        "base_model_type": str(base_model_type),
        "train_sequences": normalize_sequence_ids(
            raw_metadata.get("train_sequences")
        ),
        "val_sequences": normalize_sequence_ids(raw_metadata.get("val_sequences")),
        "include_bus_as_target": include_bus_as_target,
        "train_control_split_enabled": controlled,
        "eval_scope": raw_metadata.get("eval_scope"),
        "checkpoint_root": raw_metadata.get("checkpoint_root"),
        "ap_score_thresh": raw_metadata.get("ap_score_thresh"),
        "score_thresh": raw_metadata.get("score_thresh"),
    }
    return results, metadata


def _load_results_json(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = _load_json(path, {})
    results = data.get("results")
    metadata = data.get("metadata")
    if not isinstance(results, list) or not isinstance(metadata, dict):
        raise DomainShiftTableError(
            "Input JSON must contain a results list and a metadata object."
        )
    return results, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update source-to-target domain-shift comparison tables."
    )
    parser.add_argument(
        "results_json",
        type=Path,
        help="Structured JSON containing {'results': [...], 'metadata': {...}}.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sequence-info",
        type=Path,
        default=DEFAULT_SEQUENCE_INFO_PATH,
    )
    args = parser.parse_args()
    results, metadata = _load_results_json(args.results_json)
    summary = update_domain_shift_tables(
        results,
        metadata,
        output_dir=args.output_dir,
        sequence_info_path=args.sequence_info,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
