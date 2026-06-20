import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from training_utils.checkpoints import format_timestamp_model_sequence_run_name


def print_training_history(history):
    if len(history) == 0:
        return

    print("\nTraining history")
    print(
        f"{'epoch':>5} "
        f"{'train_loss':>11} "
        f"{'train_box':>10} "
        f"{'train_cls':>10} "
        f"{'train_hm':>10} "
        f"{'val_loss':>9} "
        f"{'val_box':>9} "
        f"{'val_cls':>9} "
        f"{'val_hm':>9} "
        f"{'2D@0.3':>8} "
        f"{'val_precision':>14} "
        f"{'val_recall':>11} "
        f"{'val_iou':>9} "
        f"{'val_f1':>8} "
        f"{'IoU_thr':>8} "
        f"{'TP':>6} "
        f"{'FP':>6} "
        f"{'FN':>6}"
    )
    print("-" * 158)

    for row in history:
        print(
            f"{row['epoch']:5d} "
            f"{row['train_loss']:11.4f} "
            f"{row['train_box_loss']:10.4f} "
            f"{row['train_cls_loss']:10.4f} "
            f"{row.get('train_heatmap_loss', 0.0):10.4f} "
            f"{row['val_loss']:9.4f} "
            f"{row['val_box_loss']:9.4f} "
            f"{row['val_cls_loss']:9.4f} "
            f"{row.get('val_heatmap_loss', 0.0):9.4f} "
            f"{row.get('val_2d_mAP_0.3', row['val_mAP']):8.4f} "
            f"{row['val_precision']:14.4f} "
            f"{row['val_recall']:11.4f} "
            f"{row['val_iou']:9.4f} "
            f"{row['val_f1']:8.4f} "
            f"{row['iou']:8.4f} "
            f"{row['tp']:6d} "
            f"{row['fp']:6d} "
            f"{row['fn']:6d}"
        )


def create_tensorboard_writer(base_dir, experiment_name, sequence, model_type=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = format_timestamp_model_sequence_run_name(sequence, model_type, timestamp)
    log_dir = os.path.join(base_dir, experiment_name, run_name)

    suffix = 1
    unique_log_dir = log_dir
    while os.path.exists(unique_log_dir):
        unique_log_dir = f"{log_dir}_{suffix}"
        suffix += 1

    os.makedirs(unique_log_dir, exist_ok=False)
    return SummaryWriter(log_dir=unique_log_dir)


def write_tensorboard_run_config(
        writer,
        cfg,
        num_epochs,
        batch_size,
        train_size,
        val_size,
        learning_rate,
        max_detections,
        num_classes,
        class_names,
        eval_iou_thresh,
        model_type=None,
        train_scope=None,
        split_mode=None,
        train_sequences=None,
        val_sequences=None,
        best_metric_key=None,
        official_eval_enabled=False,
        official_eval_version="revised",
        official_eval_iou_backend="auto",
        official_eval_iou_mode="easy",
        official_eval_include_empty_gt_frames=False,
    ):
    config_text = "\n".join([
        f"sequence: {cfg.sequence}",
        f"sequences: {getattr(cfg, 'sequences', None)}",
        f"split_mode: {split_mode}",
        f"train_sequences: {train_sequences}",
        f"val_sequences: {val_sequences}",
        f"model_type: {model_type}",
        f"train_scope: {train_scope}",
        f"num_epochs: {num_epochs}",
        f"batch_size: {batch_size}",
        f"train_size: {train_size}",
        f"val_size: {val_size}",
        f"learning_rate: {learning_rate}",
        f"max_detections: {max_detections}",
        f"num_classes: {num_classes}",
        f"class_names: {class_names}",
        f"eval_iou_thresh: {eval_iou_thresh}",
        f"best_metric_key: {best_metric_key}",
        f"official_eval_enabled: {official_eval_enabled}",
        f"official_eval_version: {official_eval_version}",
        f"official_eval_iou_backend: {official_eval_iou_backend}",
        f"official_eval_iou_mode: {official_eval_iou_mode}",
        f"official_eval_include_empty_gt_frames: {official_eval_include_empty_gt_frames}",
    ])
    writer.add_text("run/config", config_text, 0)
    writer.flush()


def write_tensorboard_metrics(writer, epoch, train_metrics, val_metrics, f1, learning_rate):
    writer.add_scalar("training_metrics/train_loss", train_metrics["train_loss"], epoch)
    writer.add_scalar("training_metrics/train_box_loss", train_metrics["train_box_loss"], epoch)
    writer.add_scalar("training_metrics/train_cls_loss", train_metrics["train_cls_loss"], epoch)
    writer.add_scalar(
        "training_metrics/train_heatmap_loss",
        train_metrics.get("train_heatmap_loss", 0.0),
        epoch
    )
    writer.add_scalar(
        "training_metrics/train_quality_loss",
        train_metrics.get("train_quality_loss", 0.0),
        epoch
    )
    if "train_obj_loss" in train_metrics:
        writer.add_scalar("training_metrics/train_obj_loss", train_metrics["train_obj_loss"], epoch)
    if "train_l1_loss" in train_metrics:
        writer.add_scalar("training_metrics/train_l1_loss", train_metrics["train_l1_loss"], epoch)

    writer.add_scalar("validation_metrics/val_loss", val_metrics["val_loss"], epoch)
    writer.add_scalar("validation_metrics/val_box_loss", val_metrics["val_box_loss"], epoch)
    writer.add_scalar("validation_metrics/val_cls_loss", val_metrics["val_cls_loss"], epoch)
    writer.add_scalar(
        "validation_metrics/val_quality_loss",
        val_metrics.get("val_quality_loss", 0.0),
        epoch
    )
    if "val_obj_loss" in val_metrics:
        writer.add_scalar("validation_metrics/val_obj_loss", val_metrics["val_obj_loss"], epoch)
    if "val_l1_loss" in val_metrics:
        writer.add_scalar("validation_metrics/val_l1_loss", val_metrics["val_l1_loss"], epoch)
    writer.add_scalar("validation_metrics/val_mAP", val_metrics["mAP"], epoch)
    writer.add_scalar("validation_metrics/val_2d_mAP_0.3", val_metrics.get("2d_mAP_0.3", val_metrics["mAP"]), epoch)
    writer.add_scalar("validation_metrics/val_2d_mAP_0.5", val_metrics.get("2d_mAP_0.5", 0.0), epoch)
    writer.add_scalar("validation_metrics/val_3d_mAP_0.3", val_metrics.get("3d_mAP_0.3", 0.0), epoch)
    writer.add_scalar("validation_metrics/val_3d_mAP_0.5", val_metrics.get("3d_mAP_0.5", 0.0), epoch)
    writer.add_scalar("validation_metrics/val_precision", val_metrics["precision"], epoch)
    writer.add_scalar("validation_metrics/val_recall", val_metrics["recall"], epoch)
    writer.add_scalar("validation_metrics/val_iou", val_metrics["val_iou"], epoch)
    writer.add_scalar("validation_metrics/val_f1", f1, epoch)
    writer.add_scalar("validation_metrics/TP", val_metrics["tp"], epoch)
    writer.add_scalar("validation_metrics/FP", val_metrics["fp"], epoch)
    writer.add_scalar("validation_metrics/FN", val_metrics["fn"], epoch)
    writer.add_scalar(
        "validation_metrics/selection_metric_value",
        val_metrics.get("selection_metric_value", val_metrics["mAP"]),
        epoch,
    )

    for key, value in sorted(val_metrics.items()):
        if key.startswith("official_") and isinstance(value, (int, float)):
            writer.add_scalar(f"validation_metrics/{key}", value, epoch)

    writer.add_scalar("parameters/learning_rate", learning_rate, epoch)
    writer.add_scalar("parameters/eval_iou_thresh", val_metrics["iou_thresh"], epoch)
    writer.flush()
