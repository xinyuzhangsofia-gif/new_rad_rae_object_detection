import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from training_utils.checkpoints import format_timestamp_model_sequence_run_name


def print_training_history(history):
    if len(history) == 0:
        return

    has_heatmap = any(("train_heatmap_loss" in row) or ("val_heatmap_loss" in row) for row in history)

    print("\nTraining history")
    header = (
        f"{'epoch':>5} "
        f"{'train_loss':>11} "
        f"{'train_box':>10} "
        f"{'train_cls':>10} "
    )
    if has_heatmap:
        header += f"{'train_hm':>10} "
    header += (
        f"{'val_loss':>9} "
        f"{'val_box':>9} "
        f"{'val_cls':>9} "
    )
    if has_heatmap:
        header += f"{'val_hm':>9} "
    header += (
        f"{'BEV@0.3':>9} "
        f"{'BEV@0.5':>9} "
        f"{'BEV@0.7':>9} "
        f"{'3D@0.3':>8} "
        f"{'Sel':>9}"
    )
    print(header)
    print("-" * len(header))

    for row in history:
        line = (
            f"{row['epoch']:5d} "
            f"{row['train_loss']:11.4f} "
            f"{row['train_box_loss']:10.4f} "
            f"{row['train_cls_loss']:10.4f} "
        )
        if has_heatmap:
            line += f"{row.get('train_heatmap_loss', 0.0):10.4f} "
        line += (
            f"{row['val_loss']:9.4f} "
            f"{row['val_box_loss']:9.4f} "
            f"{row['val_cls_loss']:9.4f} "
        )
        if has_heatmap:
            line += f"{row.get('val_heatmap_loss', 0.0):9.4f} "
        line += (
            f"{row.get('val_bev_mAP_0.3', row['val_mAP']):9.4f} "
            f"{row.get('val_bev_mAP_0.5', 0.0):9.4f} "
            f"{row.get('val_bev_mAP_0.7', 0.0):9.4f} "
            f"{row.get('val_3d_mAP_0.3', 0.0):8.4f} "
            f"{row.get('selection_metric_value', row['val_mAP']):9.4f}"
        )
        print(line)


def print_epoch_evaluation_summary(epoch, val_metrics, f1):
    del f1

    selection_key = val_metrics.get("selection_metric_key", "mAP")
    selection_value = float(val_metrics.get("selection_metric_value", val_metrics["mAP"]))
    print(
        f"Epoch {epoch}: "
        f"{selection_key}={selection_value:.4f}"
    )
    print(
        "  official revised:",
        f"bev@0.3={val_metrics.get('official_bev_mAP_0.3', 0.0):.4f}",
        f"bev@0.5={val_metrics.get('official_bev_mAP_0.5', 0.0):.4f}",
        f"bev@0.7={val_metrics.get('official_bev_mAP_0.7', 0.0):.4f}",
        f"3d@0.3={val_metrics.get('official_3d_mAP_0.3', 0.0):.4f}",
        f"3d@0.5={val_metrics.get('official_3d_mAP_0.5', 0.0):.4f}",
        f"3d@0.7={val_metrics.get('official_3d_mAP_0.7', 0.0):.4f}",
        f"p={val_metrics.get('official_detection_precision', 0.0):.4f}",
        f"r={val_metrics.get('official_detection_recall', 0.0):.4f}",
        f"f1={val_metrics.get('official_detection_f1', 0.0):.4f}",
        f"tp={int(val_metrics.get('official_detection_tp', 0))}",
        f"fp={int(val_metrics.get('official_detection_fp', 0))}",
        f"fn={int(val_metrics.get('official_detection_fn', 0))}",
    )
    if "coco_bev_mAP" in val_metrics:
        print(
            "  coco-style:",
            f"bev_mAP={val_metrics.get('coco_bev_mAP', 0.0):.4f}",
            f"bev@0.50={val_metrics.get('coco_bev_AP_0.50', 0.0):.4f}",
            f"bev@0.75={val_metrics.get('coco_bev_AP_0.75', 0.0):.4f}",
            f"3d_mAP={val_metrics.get('coco_3d_mAP', 0.0):.4f}",
            f"3d@0.50={val_metrics.get('coco_3d_AP_0.50', 0.0):.4f}",
            f"3d@0.75={val_metrics.get('coco_3d_AP_0.75', 0.0):.4f}",
        )
    if "nuscenes_mAP" in val_metrics:
        print(
            "  nuscenes-style:",
            f"mAP={val_metrics.get('nuscenes_mAP', 0.0):.4f}",
            f"AP@0.5m={val_metrics.get('nuscenes_AP_0.5m', 0.0):.4f}",
            f"AP@1.0m={val_metrics.get('nuscenes_AP_1.0m', 0.0):.4f}",
            f"AP@2.0m={val_metrics.get('nuscenes_AP_2.0m', 0.0):.4f}",
            f"AP@4.0m={val_metrics.get('nuscenes_AP_4.0m', 0.0):.4f}",
            f"mATE={val_metrics.get('nuscenes_mATE', 0.0):.4f}",
            f"mASE={val_metrics.get('nuscenes_mASE', 0.0):.4f}",
            f"mAOE={val_metrics.get('nuscenes_mAOE', 0.0):.4f}",
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
        coco_style_eval_enabled=False,
        nuscenes_style_eval_enabled=False,
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
        f"best_metric_key: {best_metric_key}",
        f"official_eval_enabled: {official_eval_enabled}",
        f"official_eval_version: {official_eval_version}",
        f"official_eval_iou_backend: {official_eval_iou_backend}",
        f"official_eval_iou_mode: {official_eval_iou_mode}",
        f"coco_style_eval_enabled: {coco_style_eval_enabled}",
        f"nuscenes_style_eval_enabled: {nuscenes_style_eval_enabled}",
    ])
    writer.add_text("run/config", config_text, 0)
    writer.flush()


def write_tensorboard_metrics(writer, epoch, train_metrics, val_metrics, f1, learning_rate):
    writer.add_scalar("training_metrics/train_loss", train_metrics["train_loss"], epoch)
    writer.add_scalar("training_metrics/train_box_loss", train_metrics["train_box_loss"], epoch)
    writer.add_scalar("training_metrics/train_cls_loss", train_metrics["train_cls_loss"], epoch)
    if "train_heatmap_loss" in train_metrics:
        writer.add_scalar(
            "training_metrics/train_heatmap_loss",
            train_metrics["train_heatmap_loss"],
            epoch
        )
    if "train_quality_loss" in train_metrics:
        writer.add_scalar(
            "training_metrics/train_quality_loss",
            train_metrics["train_quality_loss"],
            epoch
        )
    if "train_obj_loss" in train_metrics:
        writer.add_scalar("training_metrics/train_obj_loss", train_metrics["train_obj_loss"], epoch)
    if "train_l1_loss" in train_metrics:
        writer.add_scalar("training_metrics/train_l1_loss", train_metrics["train_l1_loss"], epoch)
    if "train_gwd_loss" in train_metrics:
        writer.add_scalar("training_metrics/train_gwd_loss", train_metrics["train_gwd_loss"], epoch)

    writer.add_scalar("validation_metrics/val_loss", val_metrics["val_loss"], epoch)
    writer.add_scalar("validation_metrics/val_box_loss", val_metrics["val_box_loss"], epoch)
    writer.add_scalar("validation_metrics/val_cls_loss", val_metrics["val_cls_loss"], epoch)
    if "val_heatmap_loss" in val_metrics:
        writer.add_scalar(
            "validation_metrics/val_heatmap_loss",
            val_metrics["val_heatmap_loss"],
            epoch
        )
    if "val_quality_loss" in val_metrics:
        writer.add_scalar(
            "validation_metrics/val_quality_loss",
            val_metrics["val_quality_loss"],
            epoch
        )
    if "val_obj_loss" in val_metrics:
        writer.add_scalar("validation_metrics/val_obj_loss", val_metrics["val_obj_loss"], epoch)
    if "val_l1_loss" in val_metrics:
        writer.add_scalar("validation_metrics/val_l1_loss", val_metrics["val_l1_loss"], epoch)
    if "val_gwd_loss" in val_metrics:
        writer.add_scalar("validation_metrics/val_gwd_loss", val_metrics["val_gwd_loss"], epoch)
    writer.add_scalar("validation_metrics/val_mAP", val_metrics["mAP"], epoch)
    writer.add_scalar(
        "validation_metrics/val_bev_mAP_0.3",
        val_metrics.get("official_bev_mAP_0.3", val_metrics["mAP"]),
        epoch,
    )
    writer.add_scalar(
        "validation_metrics/val_3d_mAP_0.3",
        val_metrics.get("official_3d_mAP_0.3", 0.0),
        epoch,
    )
    writer.add_scalar(
        "validation_metrics/selection_metric_value",
        val_metrics.get("selection_metric_value", val_metrics["mAP"]),
        epoch,
    )

    for key, value in sorted(val_metrics.items()):
        if (
            (key.startswith("official_") or key.startswith("coco_") or key.startswith("nuscenes_"))
            and isinstance(value, (int, float))
        ):
            writer.add_scalar(f"validation_metrics/{key}", value, epoch)

    writer.add_scalar("parameters/learning_rate", learning_rate, epoch)
    writer.flush()
