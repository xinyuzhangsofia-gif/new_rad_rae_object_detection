"""Evaluation result serialization and human-readable formatting."""

from datetime import datetime
from numbers import Real
from pathlib import Path
import re

import numpy as np
import torch
import yaml

from eval.custom_iou_range import format_custom_iou_suffix
from eval.report_paths import checkpoint_group_name, resolve_plot_checkpoint_root
from eval.result_selection import (
    result_main_metric_key,
    result_main_metric_value,
    select_best_main_metric_result,
    select_best_result_by_metric,
)


def read_report_metadata(report_path):
    """Read the leading ``key: value`` metadata block from a TXT report."""
    metadata = {}
    for line in Path(report_path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.strip():
            break
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def load_evaluation_yaml(yaml_path):
    """Load an evaluation YAML without normalizing historical fields."""
    return yaml.safe_load(
        Path(yaml_path).read_text(encoding="utf-8")
    ) or {}


def metric_text(value):
    if value is None:
        return "-"
    numeric_value = float(value)
    if np.isnan(numeric_value):
        return "-"
    return f"{numeric_value:.4f}"


def official_ap_value(value):
    if value is None:
        return None
    numeric_value = float(value)
    if np.isnan(numeric_value):
        return np.nan
    return numeric_value / 100.0


def official_ap_text(value):
    return metric_text(official_ap_value(value))


def yaml_safe_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): yaml_safe_value(child_value)
            for key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [yaml_safe_value(item) for item in value]
    return str(value)


def available_official_iou_suffixes(result):
    suffixes = []
    for suffix in ("0.3", "0.5"):
        bev_key = f"official_bev_mAP_{suffix}"
        d3_key = f"official_3d_mAP_{suffix}"
        if bev_key in result or d3_key in result:
            suffixes.append(suffix)
    return suffixes


def collect_method_summary(result):
    official_summary = {
        "main_metric_key": result.get("official_main_metric_key"),
        "main_metric_value": official_ap_value(result.get("official_main_metric_value")),
    }
    for suffix in available_official_iou_suffixes(result):
        official_summary[f"bev_mAP_{suffix}"] = official_ap_value(
            result.get(f"official_bev_mAP_{suffix}")
        )
        official_summary[f"3d_mAP_{suffix}"] = official_ap_value(
            result.get(f"official_3d_mAP_{suffix}")
        )

    summary = {
        "evaluation": {
            "coordinate_mode": result.get("effective_eval_coordinate_mode"),
            "primary_geometry": result.get("evaluation_primary_geometry"),
            "main_metric_key": result_main_metric_key(result),
            "main_metric_value": official_ap_value(
                result_main_metric_value(result)
            ),
        },
        "official": official_summary,
    }
    if "coco_bev_mAP" in result:
        summary["coco_style"] = {
            "bev_mAP": result.get("coco_bev_mAP"),
            "bev_AP_0.50": result.get("coco_bev_AP_0.50"),
            "bev_AP_0.75": result.get("coco_bev_AP_0.75"),
            "3d_mAP": result.get("coco_3d_mAP"),
            "3d_AP_0.50": result.get("coco_3d_AP_0.50"),
            "3d_AP_0.75": result.get("coco_3d_AP_0.75"),
        }
    if "custom_iou_bev_mAP" in result:
        summary["custom_iou_range"] = {
            "iou_thresholds": result.get("custom_iou_thresholds"),
            "bev_mAP": result.get("custom_iou_bev_mAP"),
            "3d_mAP": result.get("custom_iou_3d_mAP"),
            "precision": result.get("custom_iou_precision"),
            "recall": result.get("custom_iou_recall"),
            "f1": result.get("custom_iou_f1"),
        }
    if "nuscenes_mAP" in result:
        summary["nuscenes_style"] = {
            "mAP": result.get("nuscenes_mAP"),
            "AP_0.5m": result.get("nuscenes_AP_0.5m"),
            "AP_1.0m": result.get("nuscenes_AP_1.0m"),
            "AP_2.0m": result.get("nuscenes_AP_2.0m"),
            "AP_4.0m": result.get("nuscenes_AP_4.0m"),
            "mATE": result.get("nuscenes_mATE"),
            "mASE": result.get("nuscenes_mASE"),
            "mAOE": result.get("nuscenes_mAOE"),
        }
    return summary


def build_yaml_export(results, plot_metadata=None):
    if len(results) == 0:
        raise ValueError("Cannot export YAML because evaluation results are empty.")

    resolved_plot_metadata = dict(plot_metadata or {})
    checkpoint_root = resolve_plot_checkpoint_root(
        results,
        plot_metadata=resolved_plot_metadata,
    )
    resolved_plot_metadata.setdefault("checkpoint_root", checkpoint_root)
    if checkpoint_root != "-":
        resolved_plot_metadata.setdefault(
            "checkpoint_group",
            checkpoint_group_name(checkpoint_root),
        )

    best_result = select_best_main_metric_result(results)

    checkpoint_entries = []
    for result in results:
        checkpoint_entries.append({
            "epoch": int(result["epoch"]),
            "checkpoint_path": str(result.get("checkpoint_path", "")),
            "method_summary": collect_method_summary(result),
            "all_metrics": yaml_safe_value(result),
        })

    export_data = {
        "generated_at": datetime.now().isoformat(),
        "plot_metadata": yaml_safe_value(resolved_plot_metadata),
        "best_result": {
            "epoch": int(best_result["epoch"]),
            "checkpoint_path": str(best_result.get("checkpoint_path", "")),
            "method_summary": collect_method_summary(best_result),
        },
        "checkpoints": checkpoint_entries,
    }
    return export_data


def save_evaluation_yaml(results, yaml_output_path, plot_metadata=None):
    export_data = yaml_safe_value(
        build_yaml_export(results, plot_metadata=plot_metadata)
    )
    with open(yaml_output_path, "w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(
            export_data,
            yaml_file,
            sort_keys=False,
            allow_unicode=True,
        )


def print_checkpoint_metrics(epoch, metrics):
    main_key = result_main_metric_key(metrics)
    parts = [
        f"epoch={epoch}",
        f"{main_key}={official_ap_text(result_main_metric_value(metrics))}",
        f"ap_score_thr={metric_text(metrics.get('ap_score_thresh'))}",
    ]
    for suffix in available_official_iou_suffixes(metrics):
        parts.append(
            f"bev@{suffix}={official_ap_text(metrics.get(f'official_bev_mAP_{suffix}'))}"
        )
    for suffix in available_official_iou_suffixes(metrics):
        parts.append(
            f"3d@{suffix}={official_ap_text(metrics.get(f'official_3d_mAP_{suffix}'))}"
        )
    for metric_key, metric_label in distance_quartile_metric_specs([metrics]):
        parts.append(
            f"{metric_label}={official_ap_text(metrics.get(metric_key))}"
        )
    if "official_detection_precision" in metrics:
        parts.extend([
            f"score_thr={metric_text(metrics.get('official_detection_score_threshold'))}",
            f"p={metric_text(metrics.get('official_detection_precision'))}",
            f"r={metric_text(metrics.get('official_detection_recall'))}",
            f"f1={metric_text(metrics.get('official_detection_f1'))}",
            f"tp={metrics.get('official_detection_tp', 0)}",
            f"fp={metrics.get('official_detection_fp', 0)}",
            f"fn={metrics.get('official_detection_fn', 0)}",
        ])
    if "eval_ignore_suppressed_predictions" in metrics:
        parts.append(
            f"ign_sup={int(metrics.get('eval_ignore_suppressed_predictions', 0))}"
        )
    parts.append(
        f"frames={metrics.get('evaluation_num_eval_frames', metrics.get('official_num_eval_frames', 0))}"
    )
    print(" ".join(parts))
    if "coco_bev_mAP" in metrics:
        print(
            f"  coco-style "
            f"bev_mAP={metric_text(metrics.get('coco_bev_mAP'))} "
            f"bev@0.50={metric_text(metrics.get('coco_bev_AP_0.50'))} "
            f"bev@0.75={metric_text(metrics.get('coco_bev_AP_0.75'))} "
            f"3d_mAP={metric_text(metrics.get('coco_3d_mAP'))} "
            f"3d@0.50={metric_text(metrics.get('coco_3d_AP_0.50'))} "
            f"3d@0.75={metric_text(metrics.get('coco_3d_AP_0.75'))}"
        )
    if "custom_iou_bev_mAP" in metrics:
        custom_iou_text = ", ".join(
            format_custom_iou_suffix(value)
            for value in metrics.get("custom_iou_thresholds", [])
        )
        print(
            f"  custom-iou-range "
            f"iou=[{custom_iou_text}] "
            f"ap_score_thr={metric_text(metrics.get('ap_score_thresh'))} "
            f"bev_mAP={metric_text(metrics.get('custom_iou_bev_mAP'))} "
            f"3d_mAP={metric_text(metrics.get('custom_iou_3d_mAP'))} "
            f"p={metric_text(metrics.get('custom_iou_precision'))} "
            f"r={metric_text(metrics.get('custom_iou_recall'))} "
            f"f1={metric_text(metrics.get('custom_iou_f1'))}"
        )
    if "nuscenes_mAP" in metrics:
        print(
            f"  nuscenes-style "
            f"mAP={metric_text(metrics.get('nuscenes_mAP'))} "
            f"AP@0.5m={metric_text(metrics.get('nuscenes_AP_0.5m'))} "
            f"AP@1.0m={metric_text(metrics.get('nuscenes_AP_1.0m'))} "
            f"AP@2.0m={metric_text(metrics.get('nuscenes_AP_2.0m'))} "
            f"AP@4.0m={metric_text(metrics.get('nuscenes_AP_4.0m'))} "
            f"mATE={metric_text(metrics.get('nuscenes_mATE'))} "
            f"mASE={metric_text(metrics.get('nuscenes_mASE'))} "
            f"mAOE={metric_text(metrics.get('nuscenes_mAOE'))}"
        )


def _plain_text_table(headers, rows):
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in string_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(row_values):
        return "| " + " | ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(row_values)
        ) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [separator, format_row(headers), separator]
    for row in string_rows:
        lines.append(format_row(row))
    lines.append(separator)
    return "\n".join(lines)


def _aligned_text_table(
        headers,
        rows,
        left_aligned_columns=0,
        column_gap="  ",
    ):
    """Format a borderless fixed-width table like the epoch TXT tables."""
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in string_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values):
        cells = []
        for index, value in enumerate(values):
            value = str(value)
            if index < int(left_aligned_columns):
                cells.append(value.ljust(widths[index]))
            else:
                cells.append(value.rjust(widths[index]))
        return str(column_gap).join(cells).rstrip()

    header_line = format_row(headers)
    lines = [header_line, "-" * len(header_line)]
    lines.extend(format_row(row) for row in string_rows)
    return "\n".join(lines)


def format_custom_iou_range_text(thresholds):
    values = [float(value) for value in thresholds]
    if len(values) == 0:
        return "iou=[]"
    if len(values) == 1:
        return f"iou={format_custom_iou_suffix(values[0])}"

    step = values[1] - values[0]
    is_uniform = all(
        abs((values[index] - values[index - 1]) - step) <= 1e-6
        for index in range(1, len(values))
    )
    if is_uniform and step > 0.0:
        return (
            f"iou={format_custom_iou_suffix(values[0])}"
            f" -> {format_custom_iou_suffix(values[-1])}"
            f" (step={format_custom_iou_suffix(step)})"
        )

    threshold_text = ", ".join(
        format_custom_iou_suffix(value)
        for value in values
    )
    return f"iou=[{threshold_text}]"


def distance_quartile_metric_specs(rows):
    """Return ordered quartile AP@0.3 keys and parse-safe report labels."""
    tags = []
    for row in rows:
        for quartile in row.get("distance_quartile_bins", []):
            if not isinstance(quartile, dict):
                continue
            tag = quartile.get("tag")
            if tag not in (None, "") and str(tag) not in tags:
                tags.append(str(tag))
        for metric_key in row:
            match = re.fullmatch(
                r"official_(?:bev|3d)_mAP_0\.3_quartile_(q\d+)",
                str(metric_key),
            )
            if match is not None and match.group(1) not in tags:
                tags.append(match.group(1))

    def tag_sort_key(tag):
        match = re.fullmatch(r"q(\d+)", str(tag).lower())
        if match is None:
            return (10**9, str(tag))
        return (int(match.group(1)), str(tag))

    specs = []
    for tag in sorted(tags, key=tag_sort_key):
        for geometry in ("bev", "3d"):
            metric_key = f"official_{geometry}_mAP_0.3_quartile_{tag}"
            if any(metric_key in row for row in rows):
                specs.append(
                    (metric_key, f"{geometry}@0.3_quartile_{tag}")
                )
    return tuple(specs)


def format_eval_table(rows, metric_group="all"):
    if len(rows) == 0:
        return ""
    if metric_group not in {"all", "core", "custom", "nuscenes"}:
        raise ValueError(
            "metric_group must be one of: all, core, custom, nuscenes"
        )
    if (
        metric_group == "custom"
        and not any("custom_iou_bev_mAP" in row for row in rows)
    ):
        return ""
    if (
        metric_group == "nuscenes"
        and not any("nuscenes_mAP" in row for row in rows)
    ):
        return ""

    columns = [("epoch", "epoch", 5, "int")]
    if (
        metric_group in {"all", "core"}
        and any("official_bev_mAP_0.3" in row for row in rows)
    ):
        columns.extend([
            ("bev@0.3", "official_bev_mAP_0.3", 9, "float"),
            ("bev@0.5", "official_bev_mAP_0.5", 9, "float"),
            ("3d@0.3", "official_3d_mAP_0.3", 9, "float"),
            ("3d@0.5", "official_3d_mAP_0.5", 9, "float"),
        ])
    if metric_group in {"all", "core"}:
        for metric_key, metric_label in distance_quartile_metric_specs(rows):
            columns.append(
                (metric_label, metric_key, max(12, len(metric_label)), "float")
            )
    if (
        metric_group in {"all", "core"}
        and any("official_detection_precision" in row for row in rows)
    ):
        columns.extend([
            ("p", "official_detection_precision", 8, "float"),
            ("r", "official_detection_recall", 8, "float"),
            ("f1", "official_detection_f1", 8, "float"),
        ])
    if (
        metric_group in {"all", "core"}
        and any("eval_ignore_suppressed_predictions" in row for row in rows)
    ):
        columns.append(("ign_sup", "eval_ignore_suppressed_predictions", 8, "int"))
    if (
        metric_group in {"all", "core"}
        and any("official_neutral_gt_count" in row for row in rows)
    ):
        columns.append(("neutral_gt", "official_neutral_gt_count", 10, "int"))
    if (
        metric_group in {"all", "custom"}
        and any("custom_iou_bev_mAP" in row for row in rows)
    ):
        columns.extend(
            [
                ("c_bev_mAP", "custom_iou_bev_mAP", 11, "float"),
                ("c_3d_mAP", "custom_iou_3d_mAP", 10, "float"),
                ("c_p", "custom_iou_precision", 8, "float"),
                ("c_r", "custom_iou_recall", 8, "float"),
                ("c_f1", "custom_iou_f1", 8, "float"),
            ]
        )
    if (
        metric_group in {"all", "nuscenes"}
        and any("nuscenes_mAP" in row for row in rows)
    ):
        columns.extend(
            [
                ("n_mAP", "nuscenes_mAP", 8, "float"),
                ("n@0.5m", "nuscenes_AP_0.5m", 8, "float"),
                ("n@1m", "nuscenes_AP_1.0m", 8, "float"),
                ("n@2m", "nuscenes_AP_2.0m", 8, "float"),
                ("n@4m", "nuscenes_AP_4.0m", 8, "float"),
                ("mATE", "nuscenes_mATE", 8, "float"),
                ("mASE", "nuscenes_mASE", 8, "float"),
                ("mAOE", "nuscenes_mAOE", 8, "float"),
            ]
        )

    header = " ".join(f"{label:>{width}}" for label, _, width, _ in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        row_text = []
        for _, key, width, kind in columns:
            if kind == "int":
                row_text.append(f"{int(row.get(key, 0)):>{width}d}")
            else:
                row_text.append(f"{float(row.get(key, 0.0)):>{width}.4f}")
        lines.append(" ".join(row_text))
    return "\n".join(lines)


def format_best_epoch_summary(rows):
    summary_specs = (
        ("official_bev_mAP_0.3", "bev@0.3"),
        ("official_3d_mAP_0.3", "3d@0.3"),
    )
    lines = []
    thresholds = next(
        (
            row.get("custom_iou_thresholds")
            for row in rows
            if isinstance(row.get("custom_iou_thresholds"), (list, tuple))
            and len(row.get("custom_iou_thresholds")) > 0
        ),
        None,
    )
    if thresholds is not None:
        lines.append(f"custom_iou_range: {format_custom_iou_range_text(thresholds)}")
        lines.append("custom_iou_range_metrics: c_bev_mAP=BEV mAP, c_3d_mAP=3D mAP")
        summary_specs = summary_specs + (
            ("custom_iou_bev_mAP", "custom_bev_mAP"),
            ("custom_iou_3d_mAP", "custom_3d_mAP"),
        )
    for metric_key, metric_label in summary_specs:
        best_row = select_best_result_by_metric(rows, metric_key)
        if best_row is None:
            continue
        lines.append(
            f"best_epoch_{metric_label}: "
            f"epoch {int(best_row['epoch'])} "
            f"({metric_label}={float(best_row.get(metric_key, 0.0)):.4f})"
        )
    return lines


def format_epoch_range_average_ap_summary(
        rows,
        start_epoch=5,
        end_epoch=25,
    ):
    """Format mean AP values over ``[start_epoch, end_epoch)``."""
    if int(start_epoch) >= int(end_epoch):
        raise ValueError(
            f"start_epoch must be < end_epoch, got {start_epoch}>={end_epoch}"
        )

    selected_rows = [
        row
        for row in rows
        if int(start_epoch) <= int(row.get("epoch", -1)) < int(end_epoch)
    ]
    if len(selected_rows) == 0:
        return []

    metric_specs = (
        ("official_bev_mAP_0.3", "bev@0.3"),
        ("official_bev_mAP_0.5", "bev@0.5"),
        ("official_3d_mAP_0.3", "3d@0.3"),
        ("official_3d_mAP_0.5", "3d@0.5"),
        ("custom_iou_bev_mAP", "c_bev_mAP"),
        ("custom_iou_3d_mAP", "c_3d_mAP"),
        ("nuscenes_mAP", "n_mAP"),
        ("nuscenes_AP_0.5m", "n@0.5m"),
        ("nuscenes_AP_1.0m", "n@1m"),
        ("nuscenes_AP_2.0m", "n@2m"),
        ("nuscenes_AP_4.0m", "n@4m"),
    ) + distance_quartile_metric_specs(selected_rows)
    average_parts = []
    for metric_key, metric_label in metric_specs:
        values = [
            float(row[metric_key])
            for row in selected_rows
            if metric_key in row
            and isinstance(row[metric_key], Real)
            and np.isfinite(float(row[metric_key]))
        ]
        if len(values) == 0:
            continue
        average_parts.append(
            f"{metric_label}={float(np.mean(values)):.4f}"
        )

    if len(average_parts) == 0:
        return []
    return [
        (
            f"average_AP_epoch_{int(start_epoch)}_to_{int(end_epoch) - 1} "
            f"(epochs_used={len(selected_rows)}): "
            + " | ".join(average_parts)
        )
    ]


def save_eval_table_txt(
        rows,
        output_path,
        metadata=None,
        selected_full_rows=None,
    ):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []
    if metadata:
        for key, value in metadata.items():
            sections.append(f"{key}: {value}")
        sections.append("")

    summary_lines = format_best_epoch_summary(rows)
    if summary_lines:
        sections.extend(summary_lines)

    average_ap_lines = format_epoch_range_average_ap_summary(rows)
    if average_ap_lines:
        sections.extend(average_ap_lines)

    if summary_lines or average_ap_lines:
        sections.append("")

    if selected_full_rows:
        sections.append("all_epoch_selection_metrics:")
        selection_table_text = format_eval_table(rows)
        if selection_table_text != "":
            sections.append(selection_table_text)
        sections.append("")
        sections.append("selected_checkpoint_full_metrics:")
        full_table_text = format_eval_table(selected_full_rows)
        if full_table_text != "":
            sections.append(full_table_text)
    else:
        table_text = format_eval_table(rows)
        if table_text != "":
            sections.append(table_text)

    output_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return output_path
