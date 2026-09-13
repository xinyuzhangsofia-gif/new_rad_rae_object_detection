"""TensorBoard output for standalone evaluation runs."""

from datetime import datetime
import json
from numbers import Real

from torch.utils.tensorboard import SummaryWriter

from eval.report_paths import (
    format_sequence_tag,
    resolve_output_base_dir,
    sanitize_filename,
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


