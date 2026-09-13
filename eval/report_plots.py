"""Evaluation figure construction and rendering."""

from eval.report_paths import (
    checkpoint_group_name,
    plot_checkpoint_summary_lines,
    sequence_name_for_filename,
)
from eval.result_selection import result_main_metric_key, result_main_metric_value
from eval.result_serialization import metric_text, official_ap_text

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


