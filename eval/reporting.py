"""Evaluation formatting, plots, tables, YAML, and TensorBoard output."""

import ast
import json
from numbers import Real
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from configs.coordinates import BOX_COORDINATE_POLAR, validate_box_coordinate_mode
from data.coordinates import SCOPE_FULL
from data.dataloader import normalize_sequence_list
from data.geometry import object_center_in_scope, prepare_cartesian_objects
from eval.custom_iou_range import format_custom_iou_suffix
from training_utils.configuration import (
    normalize_train_sequence_half_ratio,
    normalize_train_sequence_half_selection,
)
from training_utils.checkpoints import format_sequence_run_name

SPLIT_BBOX_COUNT_CLASS_NAMES = ("Sedan", "Bus or Truck")


__all__ = [
    'metric_text',
    'official_ap_value',
    'official_ap_text',
    'result_main_metric_key',
    'result_main_metric_value',
    'attach_evaluation_main_metric',
    'sequence_name_for_filename',
    'build_plot_metadata',
    'plot_output_requested',
    'plot_output_is_auto',
    'plot_checkpoint_name',
    'checkpoint_group_name',
    'checkpoint_epoch_number',
    'append_filename_tag',
    'resolve_plot_output_path',
    'metric_label',
    'metric_class_names_for_result',
    'merged_metric_class_names',
    'class_display_name_map_for_results',
    'display_class_name',
    'pick_plot_metric_keys',
    'available_plot_iou_suffixes',
    'main_plot_iou_suffix',
    'detection_table_columns',
    'result_row_label',
    'style_metric_table',
    'build_official_plot_section',
    'build_coco_plot_section',
    'build_custom_iou_plot_section',
    'build_nuscenes_plot_section',
    'resolve_plot_checkpoint_root',
    'plot_checkpoint_summary_lines',
    'save_evaluation_plot',
    'yaml_safe_value',
    'available_official_iou_suffixes',
    'collect_method_summary',
    'build_yaml_export',
    'resolve_yaml_output_path',
    'save_evaluation_yaml',
    'print_checkpoint_metrics',
    '_plain_text_table',
    'sanitize_filename',
    'weather_prefixed_model_variant_name',
    'format_sequence_tag',
    'resolve_output_base_dir',
    'evaluation_output_dir',
    'refresh_weather_domain_shift_summary',
    'refresh_total_result_summary',
    'create_evaluation_tensorboard_writer',
    'write_evaluation_tensorboard_result',
    'next_available_output_path',
    'format_custom_iou_range_text',
    'distance_range_metric_specs',
    'distance_quartile_metric_specs',
    'format_eval_table',
    'format_best_epoch_summary',
    'format_epoch_range_average_ap_summary',
    'default_eval_table_txt_path',
    'save_eval_table_txt',
    'init_split_bbox_count_summary',
    'iter_subset_global_indices',
    'compute_subset_bbox_count_summary',
    'build_split_statistics_metadata',
    'select_best_result_by_metric',
    'selection_iou_mode_for_group_plot',
    'group_checkpoint_plot_best_only_active'
]


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


def result_main_metric_key(result):
    return result.get(
        "evaluation_main_metric_key",
        result.get("official_main_metric_key", "official_bev_mAP_0.3"),
    )


def result_main_metric_value(result):
    value = result.get("evaluation_main_metric_value")
    if value is None:
        value = result.get("official_main_metric_value", 0.0)
    return float(value)


def attach_evaluation_main_metric(metrics, primary_geometry):
    primary_geometry = validate_box_coordinate_mode(primary_geometry)
    if primary_geometry == BOX_COORDINATE_POLAR:
        primary_key = (
            "polar_bev_mAP_0.3"
            if "polar_bev_mAP_0.3" in metrics
            else "polar_bev_mAP"
        )
    else:
        primary_key = metrics.get(
            "official_main_metric_key",
            "official_bev_mAP_0.3",
        )
    primary_value = float(metrics.get(primary_key, 0.0))
    metrics["evaluation_main_metric_key"] = primary_key
    metrics["evaluation_main_metric_value"] = primary_value
    metrics["mAP"] = primary_value
    return metrics


def sequence_name_for_filename(sequences, empty_name):
    sequences = normalize_sequence_list(sequences, name=empty_name)
    if sequences is None or len(sequences) == 0:
        return empty_name
    return format_sequence_run_name(sequences)


def build_plot_metadata(args, model_variant_name, source_metadata):
    source_train_sequences = source_metadata.get("train_sequences")
    source_val_sequences = source_metadata.get("val_sequences")
    return {
        "model_type": str(model_variant_name or "model_unknown"),
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoint_group": checkpoint_group_name(args.checkpoint_root),
        "base_model_type": str(source_metadata.get("base_model_type", "model_unknown")),
        "weather_group": source_metadata.get("weather_group"),
        "seed": source_metadata.get("seed", args.seed),
        "model_configuration_name": source_metadata.get("model_configuration_name"),
        "model_configuration": source_metadata.get("model_configuration", {}),
        "split_mode": str(args.split_mode),
        "train_sequences": sequence_name_for_filename(source_train_sequences, "train_unknown"),
        "checkpoint_train_sequences": sequence_name_for_filename(source_train_sequences, "train_unknown"),
        "train_sequence_half_selection": source_metadata.get(
            "train_sequence_half_selection",
            {},
        ),
        "train_sequence_half_ratio": source_metadata.get(
            "train_sequence_half_ratio"
        ),
        "checkpoint_val_sequences": sequence_name_for_filename(source_val_sequences, "val_unknown"),
        "val_sequences": sequence_name_for_filename(args.val_sequences, "val_unknown"),
        "eval_val_sequences": getattr(args, "eval_val_sequences", None),
        "eval_frame_manifest_path": getattr(
            args, "eval_frame_manifest_path", None
        ),
        "eval_gt_object_ignore_override_path": getattr(
            args, "eval_gt_object_ignore_override_path", None
        ),
        "eval_report_path": getattr(args, "eval_report_path", None),
        "eval_scope": str(args.eval_scope),
        "eval_coordinate_mode": str(args.eval_coordinate_mode),
        "effective_eval_coordinate_mode": str(args.effective_eval_coordinate_mode),
        "evaluation_primary_geometry": str(args.evaluation_primary_geometry),
        "box_coordinate_mode": str(args.box_coordinate_mode),
        "include_bus_as_target": bool(args.include_bus_as_target),
        "checkpoint_include_bus_as_target": bool(
            source_metadata.get("include_bus_as_target", args.include_bus_as_target)
        ),
        "gt_object_ignore_override_path": source_metadata.get(
            "gt_object_ignore_override_path",
            args.gt_object_ignore_override_path,
        ),
        "train_control_split_enabled": bool(
            source_metadata.get("train_control_split_enabled", False)
        ),
        "start_epoch": (
            None if args.start_epoch is None else int(args.start_epoch)
        ),
        "end_epoch": None if args.end_epoch is None else int(args.end_epoch),
        "heatmap_score_mode": str(args.heatmap_score_mode),
        "ap_score_thresh": float(args.ap_score_thresh),
        "score_thresh": float(args.score_thresh),
        "official_iou_mode": str(args.official_eval_iou_mode),
        "official_ap03_only": bool(args.official_ap03_only),
        "official_detection_metrics_enabled": bool(args.official_detection_metrics_enabled),
        "group_checkpoint_plot_best_only": bool(args.group_checkpoint_plot_best_only),
        "custom_iou_range_eval_enabled": bool(args.custom_iou_range_eval_enabled),
        "custom_iou_thresholds": [float(value) for value in args.custom_iou_thresholds],
        "distance_range_eval_enabled": bool(args.distance_range_eval_enabled),
        "distance_range_bins": [
            [float(lower_m), float(upper_m)]
            for lower_m, upper_m in args.distance_range_bins
        ],
        "distance_quartile_eval_enabled": bool(
            getattr(args, "distance_quartile_eval_enabled", False)
        ),
        "distance_quartile_bins_requested": getattr(
            args, "distance_quartile_bins", None
        ),
        "coco_style_eval_enabled": bool(args.coco_style_eval_enabled),
        "nuscenes_style_eval_enabled": bool(args.nuscenes_style_eval_enabled),
        "official_eval_enabled": bool(args.official_eval_enabled),
        "official_geometry_source": str(args.official_geometry_source),
        "polar_eval_enabled": bool(args.polar_eval_enabled),
        "polar_geometry_source": str(args.polar_geometry_source),
        "polar_iou_thresholds": [float(value) for value in args.polar_iou_thresholds],
    }


def plot_output_requested(plot_output):
    if plot_output is None:
        return False
    if isinstance(plot_output, str):
        normalized = plot_output.strip()
        if normalized == "":
            return False
        if normalized.lower() in {"no", "none", "false", "0", "null"}:
            return False
        return True
    return bool(plot_output)


def plot_output_is_auto(plot_output):
    if not isinstance(plot_output, str):
        return bool(plot_output)
    normalized = plot_output.strip().lower()
    return normalized in {"yes", "true", "1", "auto"}


def plot_checkpoint_name(checkpoint_path):
    return os.path.splitext(os.path.basename(checkpoint_path))[0]


def checkpoint_group_name(checkpoint_root):
    """Return a stable directory name for one training checkpoint root."""
    if checkpoint_root in (None, ""):
        return "checkpoint_unknown"

    root_text = str(checkpoint_root).rstrip("/\\")
    name = Path(root_text).name
    if name in ("", ".", ".."):
        name = "checkpoint_unknown"
    if name.endswith((".pth", ".pt", ".ckpt")):
        name = Path(name).stem
    return sanitize_filename(name) or "checkpoint_unknown"


def checkpoint_epoch_number(checkpoint_path):
    checkpoint_name = plot_checkpoint_name(checkpoint_path)
    match = re.search(r"_epoch_(\d+)_", checkpoint_name)
    if match is None:
        return None
    return int(match.group(1))


def append_filename_tag(output_path, tag):
    if tag in (None, ""):
        return output_path
    base, suffix = os.path.splitext(output_path)
    return f"{base}__{tag}{suffix}"


def resolve_plot_output_path(
        args,
        checkpoint_paths,
        model_type,
        checkpoint_path=None,
        selection_tag=None,
        weather_group=None,
        train_sequences=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        seed=None,
    ):
    plot_output = args.plot_output
    if not plot_output_requested(plot_output):
        return None

    val_tag = format_sequence_tag(args.val_sequences, "val_seq")
    if checkpoint_path is None:
        checkpoint_root = args.checkpoint_root
        if os.path.isfile(checkpoint_root):
            checkpoint_path = checkpoint_root
        else:
            _, checkpoint_path = checkpoint_paths[0]
    epoch_number = checkpoint_epoch_number(checkpoint_path)

    def auto_plot_output_path():
        output_dir = evaluation_output_dir(
            base_dir="evaluation_plots",
            weather_group=weather_group,
            val_sequences=args.val_sequences,
            train_sequences=train_sequences,
            train_sequence_half_selection=train_sequence_half_selection,
            train_sequence_half_ratio=train_sequence_half_ratio,
            seed=seed,
            model_type=model_type,
        )
        stem_parts = [
            sanitize_filename(val_tag),
        ]
        if selection_tag:
            stem_parts.append(sanitize_filename(selection_tag))
        if epoch_number is not None:
            stem_parts.append(f"e{int(epoch_number):03d}")
        stem = "__".join(part for part in stem_parts if part not in {"", None})
        return str(
            next_available_output_path(
                output_dir=output_dir,
                stem=stem,
                suffix=".png",
            )
        )

    if isinstance(plot_output, str):
        normalized = plot_output.strip()
        if plot_output_is_auto(normalized):
            return auto_plot_output_path()
        return append_filename_tag(normalized, selection_tag)

    if bool(plot_output):
        return auto_plot_output_path()

    return None


def metric_label(metric_key):
    if metric_key.startswith("official_bev_mAP_"):
        return f"BEV mAP@{metric_key.rsplit('_', 1)[-1]}"
    if metric_key.startswith("official_3d_mAP_"):
        return f"3D mAP@{metric_key.rsplit('_', 1)[-1]}"
    return metric_key


def metric_class_names_for_result(result):
    class_names = result.get("metric_class_names")
    if isinstance(class_names, (list, tuple)) and len(class_names) > 0:
        return [str(class_name) for class_name in class_names]

    for key in (
        "official_per_class",
        "official_detection_per_class",
        "custom_iou_per_class",
        "custom_iou_detection_per_class",
        "coco_per_class",
        "nuscenes_per_class",
        "polar_per_class",
    ):
        values = result.get(key)
        if isinstance(values, dict) and len(values) > 0:
            return [str(class_name) for class_name in values.keys()]
    return []


def merged_metric_class_names(results):
    merged = []
    seen = set()
    for result in results:
        for class_name in metric_class_names_for_result(result):
            if class_name in seen:
                continue
            merged.append(class_name)
            seen.add(class_name)
    return merged


def class_display_name_map_for_results(results):
    merged = {}
    for result in results:
        values = result.get("class_display_name_map", {})
        if isinstance(values, dict):
            for class_name, display_name in values.items():
                merged[str(class_name)] = str(display_name)
    return merged


def display_class_name(class_name, display_name_map=None):
    if display_name_map is not None and class_name in display_name_map:
        return display_name_map[class_name]
    return {
        "sed": "Sedan",
        "bus": "Bus",
    }.get(class_name, class_name)


def pick_plot_metric_keys(results):
    preferred_order = [
        "official_bev_mAP_0.3",
        "official_bev_mAP_0.5",
        "official_bev_mAP_0.7",
        "official_3d_mAP_0.3",
        "official_3d_mAP_0.5",
        "official_3d_mAP_0.7",
    ]
    metric_keys = []
    for metric_key in preferred_order:
        if any(result.get(metric_key) is not None for result in results):
            metric_keys.append(metric_key)
    return metric_keys


def available_plot_iou_suffixes(results):
    preferred_suffixes = ["0.3", "0.5"]
    available_metric_keys = set(pick_plot_metric_keys(results))
    suffixes = []
    for suffix in preferred_suffixes:
        bev_key = f"official_bev_mAP_{suffix}"
        d3_key = f"official_3d_mAP_{suffix}"
        if bev_key in available_metric_keys or d3_key in available_metric_keys:
            suffixes.append(suffix)
    return suffixes


def main_plot_iou_suffix(results):
    for result in results:
        main_key = result.get("official_main_metric_key")
        if isinstance(main_key, str) and main_key.startswith("official_bev_mAP_"):
            return main_key.rsplit("_", 1)[-1]
    return "0.3"


def detection_table_columns():
    return [
        ("Precision", "official_detection_precision", "metric"),
        ("Recall", "official_detection_recall", "metric"),
        ("F1", "official_detection_f1", "metric"),
        ("TP", "official_detection_tp", "int"),
        ("FP", "official_detection_fp", "int"),
        ("FN", "official_detection_fn", "int"),
    ]


def result_row_label(result, object_name, multi_epoch):
    if multi_epoch:
        return f"Epoch {result['epoch']} / {object_name}"
    return object_name


def style_metric_table(table, row_meta, best_epoch, font_size=9.5):
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor("#9ca3af")
        if row_idx == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold", color="#111827")
            continue

        meta = row_meta[row_idx - 1]
        is_best_overall = (
            int(meta["epoch"]) == int(best_epoch)
            and meta.get("row_kind") == "overall"
        )
        if meta.get("row_kind") == "overall":
            cell.set_facecolor("#dbeafe" if is_best_overall else "#eef2ff")
        else:
            stripe_index = int(meta.get("stripe_index", row_idx - 1))
            cell.set_facecolor("#f8fafc" if stripe_index % 2 == 0 else "#ffffff")

        if col_idx == 0 and is_best_overall:
            cell.set_text_props(weight="bold", color="#1d4ed8")


def build_official_plot_section(results, iou_suffixes, multi_epoch):
    class_names = merged_metric_class_names(results)
    display_name_map = class_display_name_map_for_results(results)
    columns = ["Object"]
    for iou_suffix in iou_suffixes:
        columns.extend([f"BEV AP@{iou_suffix}", f"3D AP@{iou_suffix}"])
    include_polar = any("polar_bev_mAP_0.3" in result for result in results)
    if include_polar:
        columns.extend(["Polar AP@0.3", "Polar AP@0.5"])
    include_detection_metrics = any(
        isinstance(result.get("official_detection_precision"), (int, float))
        for result in results
    )
    if include_detection_metrics:
        columns.extend(label for label, _, _ in detection_table_columns())

    rows = []
    row_meta = []
    stripe_index = 0
    for result in results:
        row = [result_row_label(result, "Overall", multi_epoch)]
        for iou_suffix in iou_suffixes:
            row.extend([
                official_ap_text(result.get(f"official_bev_mAP_{iou_suffix}")),
                official_ap_text(result.get(f"official_3d_mAP_{iou_suffix}")),
            ])
        if include_polar:
            row.extend([
                official_ap_text(result.get("polar_bev_mAP_0.3")),
                official_ap_text(result.get("polar_bev_mAP_0.5")),
            ])
        if include_detection_metrics:
            for _, result_key, value_type in detection_table_columns():
                value = result.get(result_key)
                if value_type == "metric":
                    row.append(metric_text(value))
                else:
                    row.append("-" if value is None else str(int(value)))
        rows.append(row)
        row_meta.append({
            "epoch": int(result["epoch"]),
            "row_kind": "overall",
            "stripe_index": stripe_index,
        })
        stripe_index += 1

        per_class = result.get("official_detection_per_class", {})
        for class_name in class_names:
            class_stats = per_class.get(class_name, {})
            row = [result_row_label(result, display_class_name(class_name, display_name_map), multi_epoch)]
            for iou_suffix in iou_suffixes:
                row.extend([
                    official_ap_text(result.get(f"official_{class_name}_bev_AP_{iou_suffix}")),
                    official_ap_text(result.get(f"official_{class_name}_3d_AP_{iou_suffix}")),
                ])
            if include_polar:
                row.extend([
                    official_ap_text(result.get(f"polar_{class_name}_bev_AP_0.3")),
                    official_ap_text(result.get(f"polar_{class_name}_bev_AP_0.5")),
                ])
            if include_detection_metrics:
                for label, _, value_type in detection_table_columns():
                    class_value = class_stats.get(label.lower())
                    if value_type == "metric":
                        row.append(metric_text(class_value))
                    else:
                        row.append("-" if class_value is None else str(int(class_value)))
            rows.append(row)
            row_meta.append({
                "epoch": int(result["epoch"]),
                "row_kind": class_name,
                "stripe_index": stripe_index,
            })
            stripe_index += 1

    return {
        "title": (
            "Official K-Radar"
            if len(iou_suffixes) > 0
            else "Polar BEV"
        ),
        "columns": columns,
        "rows": rows,
        "row_meta": row_meta,
    }


def build_coco_plot_section(results, multi_epoch):
    if not any("coco_bev_mAP" in result for result in results):
        return None
    class_names = merged_metric_class_names(results)
    display_name_map = class_display_name_map_for_results(results)

    columns = [
        "Object",
        "BEV mAP",
        "BEV AP@0.50",
        "BEV AP@0.75",
        "3D mAP",
        "3D AP@0.50",
        "3D AP@0.75",
    ]

    rows = []
    row_meta = []
    stripe_index = 0
    include_per_class = not multi_epoch
    for result in results:
        rows.append([
            result_row_label(result, "Overall", multi_epoch),
            metric_text(result.get("coco_bev_mAP")),
            metric_text(result.get("coco_bev_AP_0.50")),
            metric_text(result.get("coco_bev_AP_0.75")),
            metric_text(result.get("coco_3d_mAP")),
            metric_text(result.get("coco_3d_AP_0.50")),
            metric_text(result.get("coco_3d_AP_0.75")),
        ])
        row_meta.append({
            "epoch": int(result["epoch"]),
            "row_kind": "overall",
            "stripe_index": stripe_index,
        })
        stripe_index += 1

        if not include_per_class:
            continue

        for class_name in class_names:
            rows.append([
                result_row_label(result, display_class_name(class_name, display_name_map), multi_epoch),
                metric_text(result.get(f"coco_{class_name}_bev_mAP")),
                metric_text(result.get(f"coco_{class_name}_bev_AP_0.50")),
                metric_text(result.get(f"coco_{class_name}_bev_AP_0.75")),
                metric_text(result.get(f"coco_{class_name}_3d_mAP")),
                metric_text(result.get(f"coco_{class_name}_3d_AP_0.50")),
                metric_text(result.get(f"coco_{class_name}_3d_AP_0.75")),
            ])
            row_meta.append({
                "epoch": int(result["epoch"]),
                "row_kind": class_name,
                "stripe_index": stripe_index,
            })
            stripe_index += 1

    return {
        "title": "COCO-Style",
        "columns": columns,
        "rows": rows,
        "row_meta": row_meta,
    }


def build_custom_iou_plot_section(results, multi_epoch):
    if not any("custom_iou_bev_mAP" in result for result in results):
        return None
    class_names = merged_metric_class_names(results)
    display_name_map = class_display_name_map_for_results(results)

    columns = ["Object", "BEV mAP", "3D mAP", "Precision", "Recall", "F1"]

    rows = []
    row_meta = []
    stripe_index = 0
    include_per_class = not multi_epoch
    for result in results:
        row = [
            result_row_label(result, "Overall", multi_epoch),
            metric_text(result.get("custom_iou_bev_mAP")),
            metric_text(result.get("custom_iou_3d_mAP")),
            metric_text(result.get("custom_iou_precision")),
            metric_text(result.get("custom_iou_recall")),
            metric_text(result.get("custom_iou_f1")),
        ]
        rows.append(row)
        row_meta.append({
            "epoch": int(result["epoch"]),
            "row_kind": "overall",
            "stripe_index": stripe_index,
        })
        stripe_index += 1

        if not include_per_class:
            continue

        for class_name in class_names:
            row = [
                result_row_label(result, display_class_name(class_name, display_name_map), multi_epoch),
                metric_text(result.get(f"custom_iou_{class_name}_bev_mAP")),
                metric_text(result.get(f"custom_iou_{class_name}_3d_mAP")),
                metric_text(result.get(f"custom_iou_{class_name}_precision")),
                metric_text(result.get(f"custom_iou_{class_name}_recall")),
                metric_text(result.get(f"custom_iou_{class_name}_f1")),
            ]
            rows.append(row)
            row_meta.append({
                "epoch": int(result["epoch"]),
                "row_kind": class_name,
                "stripe_index": stripe_index,
            })
            stripe_index += 1

    return {
        "title": "Custom IoU Range",
        "columns": columns,
        "rows": rows,
        "row_meta": row_meta,
    }


def build_nuscenes_plot_section(results, multi_epoch):
    if not any("nuscenes_mAP" in result for result in results):
        return None
    class_names = merged_metric_class_names(results)
    display_name_map = class_display_name_map_for_results(results)

    columns = [
        "Object",
        "mAP",
        "AP@0.5m",
        "AP@1.0m",
        "AP@2.0m",
        "AP@4.0m",
        "ATE",
        "ASE",
        "AOE",
    ]

    rows = []
    row_meta = []
    stripe_index = 0
    include_per_class = not multi_epoch
    for result in results:
        rows.append([
            result_row_label(result, "Overall", multi_epoch),
            metric_text(result.get("nuscenes_mAP")),
            metric_text(result.get("nuscenes_AP_0.5m")),
            metric_text(result.get("nuscenes_AP_1.0m")),
            metric_text(result.get("nuscenes_AP_2.0m")),
            metric_text(result.get("nuscenes_AP_4.0m")),
            metric_text(result.get("nuscenes_mATE")),
            metric_text(result.get("nuscenes_mASE")),
            metric_text(result.get("nuscenes_mAOE")),
        ])
        row_meta.append({
            "epoch": int(result["epoch"]),
            "row_kind": "overall",
            "stripe_index": stripe_index,
        })
        stripe_index += 1

        if not include_per_class:
            continue

        for class_name in class_names:
            rows.append([
                result_row_label(result, display_class_name(class_name, display_name_map), multi_epoch),
                metric_text(result.get(f"nuscenes_{class_name}_mAP")),
                metric_text(result.get(f"nuscenes_{class_name}_AP_0.5m")),
                metric_text(result.get(f"nuscenes_{class_name}_AP_1.0m")),
                metric_text(result.get(f"nuscenes_{class_name}_AP_2.0m")),
                metric_text(result.get(f"nuscenes_{class_name}_AP_4.0m")),
                metric_text(result.get(f"nuscenes_{class_name}_ATE")),
                metric_text(result.get(f"nuscenes_{class_name}_ASE")),
                metric_text(result.get(f"nuscenes_{class_name}_AOE")),
            ])
            row_meta.append({
                "epoch": int(result["epoch"]),
                "row_kind": class_name,
                "stripe_index": stripe_index,
            })
            stripe_index += 1

    return {
        "title": "nuScenes-Style",
        "columns": columns,
        "rows": rows,
        "row_meta": row_meta,
    }


def resolve_plot_checkpoint_root(results, plot_metadata=None):
    """Resolve the training run root shown inside an evaluation plot."""
    if isinstance(plot_metadata, dict):
        for metadata_key in (
            "checkpoint_root",
            "group_checkpoint_plot_source_checkpoint_root",
        ):
            checkpoint_root = plot_metadata.get(metadata_key)
            if checkpoint_root not in (None, ""):
                return str(checkpoint_root)

    checkpoint_parents = {
        str(Path(str(result["checkpoint_path"])).parent)
        for result in results
        if result.get("checkpoint_path") not in (None, "")
    }
    if len(checkpoint_parents) == 1:
        return next(iter(checkpoint_parents))
    if len(checkpoint_parents) > 1:
        return " | ".join(sorted(checkpoint_parents))
    return "-"


def plot_checkpoint_summary_lines(results, plot_metadata=None):
    """Build compact checkpoint identity lines for the plot header."""
    checkpoint_root = resolve_plot_checkpoint_root(
        results,
        plot_metadata=plot_metadata,
    )
    summary_lines = [f"Checkpoint root: {checkpoint_root}"]

    checkpoint_paths = sorted({
        str(result["checkpoint_path"])
        for result in results
        if result.get("checkpoint_path") not in (None, "")
    })
    if len(checkpoint_paths) == 1:
        summary_lines.append(
            f"Checkpoint file: {Path(checkpoint_paths[0]).name}"
        )
    elif len(checkpoint_paths) > 1:
        epochs = sorted({
            int(result["epoch"])
            for result in results
            if result.get("epoch") is not None
        })
        epoch_text = "-"
        if len(epochs) == 1:
            epoch_text = str(epochs[0])
        elif len(epochs) > 1:
            epoch_text = f"{epochs[0]}-{epochs[-1]}"
        summary_lines.append(
            f"Checkpoint files: {len(checkpoint_paths)} | Epochs: {epoch_text}"
        )
    else:
        summary_lines.append("Checkpoint file: -")
    return summary_lines


def save_evaluation_plot(results, plot_output_path, plot_metadata=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iou_suffixes = available_plot_iou_suffixes(results)

    best_result = max(
        results,
        key=result_main_metric_value,
    )
    best_metric_key = result_main_metric_key(best_result)
    multi_epoch = len(results) > 1

    sections = [
        build_official_plot_section(results, iou_suffixes, multi_epoch),
    ]
    custom_iou_section = build_custom_iou_plot_section(results, multi_epoch)
    if custom_iou_section is not None:
        sections.append(custom_iou_section)
    coco_section = build_coco_plot_section(results, multi_epoch)
    if coco_section is not None:
        sections.append(coco_section)
    nuscenes_section = build_nuscenes_plot_section(results, multi_epoch)
    if nuscenes_section is not None:
        sections.append(nuscenes_section)

    summary_lines = [
        (
            f"Best epoch: {best_result['epoch']} | "
            f"{best_metric_key} = "
            f"{official_ap_text(result_main_metric_value(best_result))}"
        )
    ]
    summary_lines.extend(
        plot_checkpoint_summary_lines(
            results,
            plot_metadata=plot_metadata,
        )
    )
    if plot_metadata is not None:
        shown_ap_iou_text = ", ".join(iou_suffixes)
        summary_lines.append(
            (
                f"Model: {plot_metadata.get('model_type', '-')} | "
                f"Split: {plot_metadata.get('split_mode', '-')} | "
                f"Scope: {plot_metadata.get('eval_scope', '-')}"
            )
        )
        summary_lines.append(
            f"Weather group: {plot_metadata.get('weather_group') or 'not recorded'}"
        )
        summary_lines.append(
            (
                f"Train: {plot_metadata.get('train_sequences', '-')} | "
                f"Val: {plot_metadata.get('val_sequences', '-')}"
            )
        )
        summary_lines.append(
            (
                f"Frames: {best_result.get('evaluation_num_eval_frames', best_result.get('official_num_eval_frames', 0))} | "
                f"Backend: {best_result.get('official_iou_backend_used', '-')} | "
                f"Shown AP IoU: {shown_ap_iou_text or 'polar'}"
            )
        )
        ap_score_thresh = plot_metadata.get("ap_score_thresh")
        if ap_score_thresh is not None:
            summary_lines.append(f"AP score threshold: {ap_score_thresh:.2f}")
        if plot_metadata.get("official_detection_metrics_enabled", False):
            score_thresh = plot_metadata.get(
                "score_thresh",
                plot_metadata.get("detection_score_thresh"),
            )
            if score_thresh is not None:
                summary_lines.append(
                    (
                        f"Det score threshold: {score_thresh:.2f} | "
                        f"Det IoU: {best_result.get('official_detection_iou_threshold', 0.0):.2f}"
                    )
                )
    if "custom_iou_bev_mAP" in best_result:
        custom_iou_text = ", ".join(
            format_custom_iou_suffix(value)
            for value in best_result.get("custom_iou_thresholds", [])
        )
        summary_lines.append(
            (
                f"Custom IoU range: {custom_iou_text} | "
                f"BEV mAP={metric_text(best_result.get('custom_iou_bev_mAP'))} | "
                f"3D mAP={metric_text(best_result.get('custom_iou_3d_mAP'))}"
            )
        )
        summary_lines.append(
            (
                "Custom IoU detection: "
                f"P={metric_text(best_result.get('custom_iou_precision'))} | "
                f"R={metric_text(best_result.get('custom_iou_recall'))} | "
                f"F1={metric_text(best_result.get('custom_iou_f1'))}"
            )
        )
    if "nuscenes_mAP" in best_result:
        summary_lines.append(
            (
                f"nuScenes-style: mAP={metric_text(best_result.get('nuscenes_mAP'))} | "
                f"AP@0.5m={metric_text(best_result.get('nuscenes_AP_0.5m'))} | "
                f"AP@1.0m={metric_text(best_result.get('nuscenes_AP_1.0m'))} | "
                f"AP@2.0m={metric_text(best_result.get('nuscenes_AP_2.0m'))} | "
                f"AP@4.0m={metric_text(best_result.get('nuscenes_AP_4.0m'))}"
            )
        )
        summary_lines.append(
            (
                "nuScenes-style errors: "
                f"mATE={metric_text(best_result.get('nuscenes_mATE'))} | "
                f"mASE={metric_text(best_result.get('nuscenes_mASE'))} | "
                f"mAOE={metric_text(best_result.get('nuscenes_mAOE'))}"
            )
        )

    max_columns = max(len(section["columns"]) for section in sections)
    fig_width = max(13.0, 1.15 * max_columns)
    summary_height = 0.62 + 0.22 * len(summary_lines)
    section_heights = [
        0.42 + 0.36 * (len(section["rows"]) + 1)
        for section in sections
    ]
    fig_height = max(
        7.0,
        summary_height + sum(section_heights) + 0.22 * (len(sections) - 1) + 0.35,
    )
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        nrows=len(sections) + 1,
        ncols=1,
        height_ratios=[summary_height, *section_heights],
        left=0.035,
        right=0.965,
        top=0.98,
        bottom=0.025,
        hspace=0.22,
    )

    summary_ax = fig.add_subplot(grid[0])
    summary_ax.axis("off")
    summary_ax.text(
        0.5,
        0.98,
        "Evaluation Results",
        transform=summary_ax.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )
    summary_ax.text(
        0.5,
        0.78,
        "\n".join(summary_lines),
        transform=summary_ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.0,
        color="#374151",
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#f8fafc",
            "edgecolor": "#d1d5db",
            "linewidth": 0.8,
        },
    )

    for section_index, (section, section_height) in enumerate(
        zip(sections, section_heights),
        start=1,
    ):
        section_ax = fig.add_subplot(grid[section_index])
        section_ax.axis("off")
        section_ax.text(
            0.0,
            0.98,
            section["title"],
            transform=section_ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#111827",
        )
        title_fraction = min(0.24, 0.34 / section_height)
        table_top = 1.0 - title_fraction
        table = section_ax.table(
            cellText=section["rows"],
            colLabels=section["columns"],
            bbox=[0.0, 0.0, 1.0, table_top],
            cellLoc="center",
            colLoc="center",
        )
        style_metric_table(
            table,
            row_meta=section["row_meta"],
            best_epoch=int(best_result["epoch"]),
            font_size=9.5,
        )

    fig.savefig(plot_output_path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


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
    if "polar_bev_mAP" in result:
        summary["polar_bev"] = {
            "iou_thresholds": result.get("polar_iou_thresholds"),
            "mAP": result.get("polar_bev_mAP"),
            "AP_0.3": result.get("polar_bev_mAP_0.3"),
            "AP_0.5": result.get("polar_bev_mAP_0.5"),
            "num_gt": result.get("polar_bev_num_gt"),
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

    best_result = max(
        results,
        key=result_main_metric_value,
    )

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


def resolve_yaml_output_path(plot_output_path):
    base, _ = os.path.splitext(plot_output_path)
    return f"{base}.yml"


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
    for metric_key, metric_label in distance_range_metric_specs([metrics]):
        parts.append(
            f"{metric_label}={official_ap_text(metrics.get(metric_key))}"
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
    if "polar_bev_mAP" in metrics:
        threshold_text = ", ".join(
            f"{float(value):.2f}" for value in metrics.get("polar_iou_thresholds", [])
        )
        print(
            f"  polar-bev-ap "
            f"iou=[{threshold_text}] "
            f"mAP={official_ap_text(metrics.get('polar_bev_mAP'))} "
            f"bev@0.3={official_ap_text(metrics.get('polar_bev_mAP_0.3'))} "
            f"bev@0.5={official_ap_text(metrics.get('polar_bev_mAP_0.5'))} "
            f"gt={int(metrics.get('polar_bev_num_gt', 0))}"
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


def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9.,_-]+", "_", str(text)).strip("_")


def weather_prefixed_model_variant_name(weather_group, model_variant_name):
    """Prefix evaluation output identities with checkpoint weather metadata.

    Checkpoints created before ``weather_group`` was introduced return the
    unmodified model variant, so their existing evaluation outputs remain
    backward-compatible.
    """
    if weather_group in (None, ""):
        return str(model_variant_name or "model_unknown")
    weather_tag = sanitize_filename(str(weather_group).strip().lower())
    if weather_tag == "":
        return str(model_variant_name or "model_unknown")
    return f"{weather_tag}_{str(model_variant_name or 'model_unknown')}"


def format_sequence_tag(sequences, prefix):
    if sequences in (None, "", ()):
        return f"{prefix}_unknown"
    values = [str(int(sequence)) for sequence in sequences]
    return f"{prefix}_{'_'.join(values)}"


def _compact_sequence_values(
        sequences,
        half_selection=None,
        half_ratio=None,
    ):
    normalized_sequences = normalize_sequence_list(
        sequences,
        name="domain_shift_output_sequences",
    )
    if normalized_sequences in (None, ()):
        return "unknown"
    normalized_half_selection = normalize_train_sequence_half_selection(
        half_selection
    )
    normalized_half_ratio = (
        normalize_train_sequence_half_ratio(
            0.5 if half_ratio is None else half_ratio
        )
        if normalized_half_selection
        else None
    )
    ratio_suffix = ""
    if (
        normalized_half_ratio is not None
        and abs(normalized_half_ratio - 0.5) > 1e-12
    ):
        ratio_suffix = f"{normalized_half_ratio * 100:g}pct"

    parts = []
    for sequence in sorted(normalized_sequences):
        sequence = int(sequence)
        part = str(sequence)
        if sequence in normalized_half_selection:
            part += (
                f"_{normalized_half_selection[sequence]}{ratio_suffix}"
            )
        parts.append(part)
    return "_".join(parts)


def resolve_output_base_dir(base_dir):
    output_dir = Path(base_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    return output_dir


def evaluation_output_dir(
        base_dir,
        weather_group=None,
        val_sequences=None,
        train_sequences=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        seed=None,
        model_type=None,
        domain_shift_train_branch=None,
        shared_train_sequences=None,
        source_train_sequences=None,
        target_train_sequences=None,
        target_test_sequences=None,
    ):
    """Build the shared TXT/PNG/YAML directory for one evaluation run."""
    weather_text = (
        "weather_unknown"
        if weather_group in (None, "")
        else str(weather_group).strip().lower()
    )
    weather_tag = sanitize_filename(weather_text) or "weather_unknown"
    if (
        domain_shift_train_branch in {"source", "target"}
        and shared_train_sequences not in (None, ())
        and source_train_sequences not in (None, ())
        and target_train_sequences not in (None, ())
        and target_test_sequences not in (None, ())
    ):
        test_tag = sanitize_filename(
            format_sequence_tag(target_test_sequences, "test_set")
        )
        shared_tag = _compact_sequence_values(
            shared_train_sequences,
            half_selection=train_sequence_half_selection,
            half_ratio=train_sequence_half_ratio,
        )
        source_tag = _compact_sequence_values(source_train_sequences)
        target_tag = _compact_sequence_values(target_train_sequences)
        pair_tag = sanitize_filename(
            f"shared{shared_tag}_s{source_tag}_t{target_tag}"
        )
        return (
            resolve_output_base_dir(base_dir)
            / weather_tag
            / test_tag
            / pair_tag
        )

    val_tag = sanitize_filename(format_sequence_tag(val_sequences, "val_seq"))
    model_tag = sanitize_filename(model_type or "model_unknown")
    normalized_train_sequences = normalize_sequence_list(
        train_sequences,
        name="evaluation_output_train_sequences",
    )
    if normalized_train_sequences is None or len(normalized_train_sequences) == 0:
        sequence_tag = "seq_unknown"
    else:
        half_selection = normalize_train_sequence_half_selection(
            train_sequence_half_selection
        )
        half_ratio = (
            normalize_train_sequence_half_ratio(
                0.5
                if train_sequence_half_ratio is None
                else train_sequence_half_ratio
            )
            if half_selection
            else None
        )
        ratio_suffix = ""
        if half_ratio is not None and abs(half_ratio - 0.5) > 1e-12:
            ratio_suffix = f"{half_ratio * 100:g}pct"
        sequence_parts = []
        for sequence in normalized_train_sequences:
            sequence = int(sequence)
            sequence_part = str(sequence)
            if sequence in half_selection:
                sequence_part += f"_{half_selection[sequence]}{ratio_suffix}"
            sequence_parts.append(sequence_part)
        sequence_tag = "seq" + "_".join(sequence_parts)
    seed_tag = "seed_unknown" if seed is None else f"seed{int(seed)}"
    train_seed_tag = sanitize_filename(
        f"train_{model_tag}_{sequence_tag}_{seed_tag}"
    )
    return (
        resolve_output_base_dir(base_dir)
        / weather_tag
        / val_tag
        / train_seed_tag
    )


def create_evaluation_tensorboard_writer(args, model_variant_name, source_metadata):
    base_dir = resolve_output_base_dir(args.evaluation_tensorboard_log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_tag = sanitize_filename(model_variant_name or "model_unknown")
    val_tag = sanitize_filename(format_sequence_tag(args.val_sequences, "val_seq"))
    log_dir = base_dir / "evaluation" / f"{timestamp}__{model_tag}__{val_tag}"
    writer = SummaryWriter(log_dir=str(log_dir))
    config = {
        "model_variant": model_variant_name,
        "weather_group": source_metadata.get("weather_group"),
        "checkpoint_root": str(args.checkpoint_root),
        "train_sequences": source_metadata.get("train_sequences"),
        "checkpoint_val_sequences": source_metadata.get("val_sequences"),
        "val_sequences": args.val_sequences,
        "include_bus_as_target": bool(args.include_bus_as_target),
        "eval_scope": args.eval_scope,
        "eval_coordinate_mode": args.eval_coordinate_mode,
        "effective_eval_coordinate_mode": args.effective_eval_coordinate_mode,
        "box_coordinate_mode": args.box_coordinate_mode,
        "evaluation_primary_geometry": args.evaluation_primary_geometry,
        "official_eval_version": args.official_eval_version,
        "official_eval_iou_mode": args.official_eval_iou_mode,
        "official_eval_enabled": bool(args.official_eval_enabled),
        "official_geometry_source": args.official_geometry_source,
        "ap_score_thresh": float(args.ap_score_thresh),
        "score_thresh": float(args.score_thresh),
        "custom_iou_range_eval_enabled": bool(args.custom_iou_range_eval_enabled),
        "custom_iou_thresholds": [float(value) for value in args.custom_iou_thresholds],
        "distance_range_eval_enabled": bool(args.distance_range_eval_enabled),
        "distance_range_bins": [
            [float(lower_m), float(upper_m)]
            for lower_m, upper_m in args.distance_range_bins
        ],
        "distance_quartile_eval_enabled": bool(
            getattr(args, "distance_quartile_eval_enabled", False)
        ),
        "group_checkpoint_plot_best_only": bool(args.group_checkpoint_plot_best_only),
        "polar_eval_enabled": bool(args.polar_eval_enabled),
        "polar_geometry_source": args.polar_geometry_source,
        "polar_iou_thresholds": [float(value) for value in args.polar_iou_thresholds],
    }
    writer.add_text(
        "run/config",
        json.dumps(config, indent=2, default=str),
        global_step=0,
    )
    writer.flush()
    return writer, log_dir


def write_evaluation_tensorboard_result(writer, result, namespace="evaluation"):
    if writer is None:
        return

    metric_prefixes = (
        "evaluation_",
        "official_",
        "custom_iou_",
        "coco_",
        "nuscenes_",
        "polar_",
        "distance_range_",
        "distance_quartile_",
        "val_",
    )
    epoch = int(result["epoch"])
    for key, value in result.items():
        if not isinstance(value, Real) or isinstance(value, bool):
            continue
        if not key.startswith(metric_prefixes):
            continue
        writer.add_scalar(f"{namespace}/metrics/{key}", float(value), epoch)
    writer.flush()


def next_available_output_path(output_dir, stem, suffix):
    candidate = output_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = output_dir / f"{stem}__{index:02d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


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


def distance_range_metric_specs(rows):
    """Return ordered distance-bin metric keys and parse-safe labels."""
    tags = []
    for row in rows:
        for distance_range in row.get("distance_range_bins", []):
            if not isinstance(distance_range, dict):
                continue
            tag = distance_range.get("tag")
            if tag not in (None, "") and str(tag) not in tags:
                tags.append(str(tag))
        for metric_key in row:
            match = re.fullmatch(
                r"official_(?:bev|3d)_mAP_0\.3_range_(.+)",
                str(metric_key),
            )
            if match is not None and match.group(1) not in tags:
                tags.append(match.group(1))

    def tag_sort_key(tag):
        match = re.fullmatch(
            r"(\d+(?:p\d+)?)_(\d+(?:p\d+)?)m",
            str(tag),
        )
        if match is None:
            return (float("inf"), float("inf"), str(tag))
        return (
            float(match.group(1).replace("p", ".")),
            float(match.group(2).replace("p", ".")),
            str(tag),
        )

    metric_specs = []
    for tag in sorted(tags, key=tag_sort_key):
        for geometry in ("bev", "3d"):
            metric_key = f"official_{geometry}_mAP_0.3_range_{tag}"
            if any(metric_key in row for row in rows):
                metric_specs.append(
                    (metric_key, f"{geometry}@0.3_range_{tag}")
                )
    return tuple(metric_specs)


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
    if metric_group in {"all", "core"} and any("val_loss" in row for row in rows):
        columns.extend(
            [
                ("val_loss", "val_loss", 10, "float"),
                ("val_box", "val_box_loss", 10, "float"),
                ("val_cls", "val_cls_loss", 10, "float"),
            ]
        )
        if any("val_heatmap_loss" in row for row in rows):
            columns.append(("val_hm", "val_heatmap_loss", 10, "float"))
        if any("val_quality_loss" in row for row in rows):
            columns.append(("val_q", "val_quality_loss", 10, "float"))
        if any("val_obj_loss" in row for row in rows):
            columns.append(("val_obj", "val_obj_loss", 10, "float"))
        if any("val_l1_loss" in row for row in rows):
            columns.append(("val_l1", "val_l1_loss", 10, "float"))
        if any("val_gwd_loss" in row for row in rows):
            columns.append(("val_gwd", "val_gwd_loss", 10, "float"))
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
        for metric_key, metric_label in distance_range_metric_specs(rows):
            columns.append(
                (metric_label, metric_key, max(12, len(metric_label)), "float")
            )
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
        and any("polar_bev_mAP_0.3" in row for row in rows)
    ):
        columns.extend(
            [
                ("pbev@0.3", "polar_bev_mAP_0.3", 10, "float"),
                ("pbev@0.5", "polar_bev_mAP_0.5", 10, "float"),
            ]
        )
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
    if any("polar_bev_mAP" in row for row in rows):
        summary_specs = summary_specs + (
            ("polar_bev_mAP_0.3", "pbev@0.3"),
            ("polar_bev_mAP_0.5", "pbev@0.5"),
            ("polar_bev_mAP", "polar_bev_mAP"),
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
        ("polar_bev_mAP_0.3", "pbev@0.3"),
        ("polar_bev_mAP_0.5", "pbev@0.5"),
        ("custom_iou_bev_mAP", "c_bev_mAP"),
        ("custom_iou_3d_mAP", "c_3d_mAP"),
        ("nuscenes_mAP", "n_mAP"),
        ("nuscenes_AP_0.5m", "n@0.5m"),
        ("nuscenes_AP_1.0m", "n@1m"),
        ("nuscenes_AP_2.0m", "n@2m"),
        ("nuscenes_AP_4.0m", "n@4m"),
    ) + distance_range_metric_specs(
        selected_rows
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


def default_eval_table_txt_path(
        model_variant_name,
        val_sequences=None,
        checkpoint_root=None,
        base_dir="evaluation_plots",
        weather_group=None,
        train_sequences=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        seed=None,
        base_model_type=None,
        domain_shift_train_branch=None,
        shared_train_sequences=None,
        source_train_sequences=None,
        target_train_sequences=None,
        target_test_sequences=None,
    ):
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = evaluation_output_dir(
        base_dir=base_dir,
        weather_group=weather_group,
        val_sequences=val_sequences,
        train_sequences=train_sequences,
        train_sequence_half_selection=train_sequence_half_selection,
        train_sequence_half_ratio=train_sequence_half_ratio,
        seed=seed,
        model_type=base_model_type,
        domain_shift_train_branch=domain_shift_train_branch,
        shared_train_sequences=shared_train_sequences,
        source_train_sequences=source_train_sequences,
        target_train_sequences=target_train_sequences,
        target_test_sequences=target_test_sequences,
    )
    if domain_shift_train_branch in {"source", "target"}:
        seed_tag = (
            "seed_unknown"
            if seed is None
            else f"seed{int(seed)}"
        )
        return output_dir / (
            f"{seed_tag}_{domain_shift_train_branch}_result.txt"
        )

    filename_parts = [timestamp]
    if val_sequences is not None:
        filename_parts.append(format_sequence_tag(val_sequences, "val_seq"))
    return next_available_output_path(
        output_dir=output_dir,
        stem="__".join(filename_parts),
        suffix=".txt",
    )


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


def _parse_report_sequence_value(value):
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return ()
    if isinstance(parsed, int):
        return (int(parsed),)
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)
    return ()


def _parse_report_half_selection(value):
    pairs = re.findall(
        r"(\d+)\s*:\s*['\"]?(first|last)['\"]?",
        str(value),
    )
    return tuple(
        sorted((int(sequence), position) for sequence, position in pairs)
    )


def _read_domain_shift_result_report(report_path):
    text = Path(report_path).read_text(encoding="utf-8")
    metadata = {}
    for line in text.splitlines():
        if line.strip() == "":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    average_match = re.search(
        r"average_AP_epoch_5_to_24 \(epochs_used=(\d+)\): ([^\n]+)",
        text,
    )
    if average_match is None or int(average_match.group(1)) != 20:
        return None
    average_values = {
        key.strip(): float(value)
        for key, value in re.findall(
            r"([A-Za-z0-9@._]+)=(-?\d+(?:\.\d+)?)",
            average_match.group(2),
        )
    }
    if "bev@0.3" not in average_values or "3d@0.3" not in average_values:
        return None

    branch = metadata.get("domain_shift_train_branch")
    if branch not in {"source", "target"}:
        return None
    return {
        "branch": branch,
        "seed": int(metadata.get("seed", 0)),
        "shared_train_sequences": _parse_report_sequence_value(
            metadata.get("shared_train_sequences", "")
        ),
        "source_train_sequences": _parse_report_sequence_value(
            metadata.get("source_train_sequences", "")
        ),
        "target_train_sequences": _parse_report_sequence_value(
            metadata.get("target_train_sequences", "")
        ),
        "target_test_sequences": _parse_report_sequence_value(
            metadata.get("target_test_sequences", "")
        ),
        "shared_half_selection": _parse_report_half_selection(
            metadata.get("train_sequence_half_selection", "")
        ),
        "bev_ap": average_values["bev@0.3"],
        "threed_ap": average_values["3d@0.3"],
    }


def _summary_sequence_text(sequences, half_selection=()):
    half_selection = dict(half_selection)
    return ",".join(
        (
            f"{int(sequence)}_{half_selection[int(sequence)]}"
            if int(sequence) in half_selection
            else str(int(sequence))
        )
        for sequence in sorted(sequences)
    ) or "-"


def refresh_weather_domain_shift_summary(
        base_dir,
        weather_group,
    ):
    """Rebuild one weather's source-vs-target average-AP summary."""
    weather_tag = sanitize_filename(
        str(weather_group or "weather_unknown").strip().lower()
    ) or "weather_unknown"
    weather_dir = resolve_output_base_dir(base_dir) / weather_tag
    reports = {}
    for report_path in sorted(
        weather_dir.glob(
            "test_set_*/*/seed*_*_result.txt"
        )
    ):
        report = _read_domain_shift_result_report(report_path)
        if report is None:
            continue
        key = (
            report["seed"],
            report["shared_train_sequences"],
            report["shared_half_selection"],
            report["source_train_sequences"],
            report["target_train_sequences"],
            report["target_test_sequences"],
        )
        reports.setdefault(key, {})[report["branch"]] = report

    if len(reports) == 0:
        return None

    rows = []
    td_bev_values = []
    td_3d_values = []
    for key in sorted(reports):
        seed, shared, shared_half, source, target, target_test = key
        source_result = reports[key].get("source")
        target_result = reports[key].get("target")
        source_bev = None if source_result is None else source_result["bev_ap"]
        source_3d = None if source_result is None else source_result["threed_ap"]
        target_bev = None if target_result is None else target_result["bev_ap"]
        target_3d = None if target_result is None else target_result["threed_ap"]
        td_value = (
            None
            if source_bev is None or target_bev is None
            else target_bev - source_bev
        )
        td_3d_value = (
            None
            if source_3d is None or target_3d is None
            else target_3d - source_3d
        )
        if td_value is not None and td_3d_value is not None:
            td_bev_values.append(td_value)
            td_3d_values.append(td_3d_value)
        value_text = lambda value: "-" if value is None else f"{value:.4f}"
        rows.append([
            f"seed{seed}",
            _summary_sequence_text(shared, shared_half),
            _summary_sequence_text(source),
            _summary_sequence_text(target),
            _summary_sequence_text(target_test),
            value_text(source_bev),
            value_text(source_3d),
            value_text(target_bev),
            value_text(target_3d),
            value_text(td_value),
            value_text(td_3d_value),
        ])

    paired_count = len(td_bev_values)
    average_td_bev = (
        sum(td_bev_values) / paired_count
        if paired_count > 0
        else None
    )
    average_td_3d = (
        sum(td_3d_values) / paired_count
        if paired_count > 0
        else None
    )
    value_text = lambda value: "-" if value is None else f"{value:.4f}"
    rows.append([
        f"average({paired_count})",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        value_text(average_td_bev),
        value_text(average_td_3d),
    ])

    headers = (
        "seed",
        "shared_seq",
        "source_seq",
        "target_seq",
        "test_seq",
        "BEV_src",
        "3D_src",
        "BEV_tgt",
        "3D_tgt",
        "TD_BEV",
        "TD_3D",
    )
    summary_path = weather_dir / "domain_shift_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "Average AP uses epochs 5-24 inclusive (20 epochs).\n"
        "src=source-trained model; tgt=target-trained model; "
        "TD_BEV=BEV_tgt-BEV_src; TD_3D=3D_tgt-3D_src; "
        "average(n) uses n complete source/target pairs.\n\n"
        + _aligned_text_table(
            headers,
            rows,
            left_aligned_columns=5,
            column_gap=" ",
        )
        + "\n",
        encoding="utf-8",
    )
    refresh_total_result_summary(base_dir)
    return summary_path


def refresh_total_result_summary(base_dir):
    """Rebuild the cross-weather target-AP and domain-shift summary."""
    def mean_and_sample_std_text(values):
        values = [float(value) for value in values]
        mean_value = sum(values) / len(values)
        if len(values) > 1:
            variance = sum(
                (value - mean_value) ** 2
                for value in values
            ) / (len(values) - 1)
            std_value = variance ** 0.5
        else:
            std_value = 0.0
        return f"{mean_value:.4f} ± {std_value:.4f}"

    def mean_text(values):
        values = [float(value) for value in values]
        return f"{sum(values) / len(values):.4f}"

    output_base_dir = resolve_output_base_dir(base_dir)
    rows = []
    for weather_dir in sorted(
        path
        for path in output_base_dir.iterdir()
        if path.is_dir()
    ):
        reports = {}
        for report_path in sorted(
            weather_dir.glob("test_set_*/*/seed*_*_result.txt")
        ):
            report = _read_domain_shift_result_report(report_path)
            if report is None:
                continue
            key = (
                report["seed"],
                report["shared_train_sequences"],
                report["shared_half_selection"],
                report["source_train_sequences"],
                report["target_train_sequences"],
                report["target_test_sequences"],
            )
            reports.setdefault(key, {})[report["branch"]] = report

        complete_pairs = [
            pair
            for pair in reports.values()
            if pair.get("source") is not None
            and pair.get("target") is not None
        ]
        if not complete_pairs:
            continue

        source_bev_values = [
            pair["source"]["bev_ap"]
            for pair in complete_pairs
        ]
        source_3d_values = [
            pair["source"]["threed_ap"]
            for pair in complete_pairs
        ]
        target_bev_values = [
            pair["target"]["bev_ap"]
            for pair in complete_pairs
        ]
        target_3d_values = [
            pair["target"]["threed_ap"]
            for pair in complete_pairs
        ]
        td_bev_values = [
            pair["target"]["bev_ap"] - pair["source"]["bev_ap"]
            for pair in complete_pairs
        ]
        td_3d_values = [
            pair["target"]["threed_ap"]
            - pair["source"]["threed_ap"]
            for pair in complete_pairs
        ]
        rows.append([
            weather_dir.name,
            mean_text(source_bev_values),
            mean_text(source_3d_values),
            mean_text(target_bev_values),
            mean_text(target_3d_values),
            mean_and_sample_std_text(td_bev_values),
            mean_and_sample_std_text(td_3d_values),
        ])

    if not rows:
        return None

    total_result_path = output_base_dir / "total_result.txt"
    total_result_path.write_text(
        "Source and target AP use epochs 5-24 inclusive.\n"
        "Only complete source/target pairs are included; "
        "AP values are means; TD=target-source and is shown as "
        "mean ± sample std.\n\n"
        + _aligned_text_table(
            (
                "weather",
                "BEV_src",
                "3D_src",
                "BEV_tgt",
                "3D_tgt",
                "TD_BEV",
                "TD_3D",
            ),
            rows,
            left_aligned_columns=1,
            column_gap="  ",
        )
        + "\n",
        encoding="utf-8",
    )
    return total_result_path


def init_split_bbox_count_summary():
    return {
        "frames": 0,
        "bbox_total": 0,
        "bbox_by_class": {
            class_name: 0
            for class_name in SPLIT_BBOX_COUNT_CLASS_NAMES
        },
    }


def iter_subset_global_indices(dataset_subset):
    if hasattr(dataset_subset, "indices") and hasattr(dataset_subset, "dataset"):
        return [int(index) for index in dataset_subset.indices], dataset_subset.dataset
    return list(range(len(dataset_subset))), dataset_subset


def compute_subset_bbox_count_summary(dataset_subset):
    subset_indices, base_dataset = iter_subset_global_indices(dataset_subset)
    if not hasattr(base_dataset, "_resolve_index") or not hasattr(
        base_dataset,
        "sequence_datasets",
    ):
        raise TypeError(
            "Expected KRadarMultiSequenceGTDetectionDataset or its Subset for bbox stats."
        )

    summary = init_split_bbox_count_summary()
    summary["frames"] = len(subset_indices)

    for global_index in subset_indices:
        dataset_idx, sample_idx = base_dataset._resolve_index(int(global_index))
        sequence_dataset = base_dataset.sequence_datasets[dataset_idx]
        all_objects = sequence_dataset.gt_by_file_idx.get(int(sample_idx), [])
        if sequence_dataset.scope_mode != SCOPE_FULL:
            radar_data = sequence_dataset.radar_dataset[int(sample_idx)]
            all_objects = prepare_cartesian_objects(
                all_objects, radar_data["full_rae_shape"]
            )
        for obj in all_objects:
            class_name = str(obj.get("cls", ""))
            if class_name not in summary["bbox_by_class"]:
                continue
            if not object_center_in_scope(obj, sequence_dataset.scope_mode):
                continue
            summary["bbox_by_class"][class_name] += 1
            summary["bbox_total"] += 1

    return summary


def build_split_statistics_metadata(train_dataset, test_dataset):
    train_summary = (
        init_split_bbox_count_summary()
        if train_dataset is None
        else compute_subset_bbox_count_summary(train_dataset)
    )
    test_summary = compute_subset_bbox_count_summary(test_dataset)

    total_frames = train_summary["frames"] + test_summary["frames"]
    total_bboxes = train_summary["bbox_total"] + test_summary["bbox_total"]

    return {
        "train_frames": train_summary["frames"],
        "test_frames": test_summary["frames"],
        "train_ratio_frames": (
            float(train_summary["frames"] / total_frames)
            if total_frames > 0
            else 0.0
        ),
        "train_bbox_total": train_summary["bbox_total"],
        "train_bbox_sedan": train_summary["bbox_by_class"]["Sedan"],
        "train_bbox_bus": train_summary["bbox_by_class"]["Bus or Truck"],
        "test_bbox_total": test_summary["bbox_total"],
        "test_bbox_sedan": test_summary["bbox_by_class"]["Sedan"],
        "test_bbox_bus": test_summary["bbox_by_class"]["Bus or Truck"],
        "train_ratio_bboxes": (
            float(train_summary["bbox_total"] / total_bboxes)
            if total_bboxes > 0
            else 0.0
        ),
    }


def select_best_result_by_metric(results, metric_key):
    best_result = None
    best_value = float("-inf")
    best_epoch = None

    for result in results:
        if metric_key not in result or "epoch" not in result:
            continue
        value = float(result.get(metric_key, 0.0))
        epoch = int(result.get("epoch", 0))
        if (
            best_result is None
            or value > best_value
            or (value == best_value and epoch < best_epoch)
        ):
            best_result = result
            best_value = value
            best_epoch = epoch
    return best_result


def selection_iou_mode_for_group_plot(args):
    if args.official_eval_iou_mode in {"easy", "all"}:
        return args.official_eval_iou_mode
    return "easy"


def group_checkpoint_plot_best_only_active(args, checkpoint_paths):
    return (
        bool(args.group_checkpoint_plot_best_only)
        and plot_output_requested(args.plot_output)
        and os.path.isdir(args.checkpoint_root)
        and len(checkpoint_paths) > 1
    )
