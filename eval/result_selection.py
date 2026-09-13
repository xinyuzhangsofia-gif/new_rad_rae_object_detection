"""Deterministic evaluation result and checkpoint selection."""

import os

from configs.coordinates import BOX_COORDINATE_POLAR, validate_box_coordinate_mode
from eval.report_paths import plot_output_requested

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


def select_best_main_metric_result(results):
    """Select the first highest main-metric result, matching ``max`` ties."""
    return max(results, key=result_main_metric_value)


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

