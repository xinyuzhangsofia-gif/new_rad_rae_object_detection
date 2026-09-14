import os

from torch.utils.tensorboard import SummaryWriter


def print_epoch_evaluation_summary(epoch, val_metrics, f1):
    del f1

    if "selection_metric_key" in val_metrics:
        selection_key = val_metrics["selection_metric_key"]
        selection_value = float(val_metrics["selection_metric_value"])
        print(f"Epoch {epoch}: {selection_key}={selection_value:.4f}")
    else:
        print(
            f"Epoch {epoch}: val_loss={val_metrics['val_loss']:.4f} "
            "(monitoring only; best selection disabled)"
        )
    if any(key.startswith("official_") for key in val_metrics):
        print(
            "  official revised:",
            f"bev@0.3={val_metrics.get('official_bev_mAP_0.3', 0.0):.4f}",
            f"3d@0.3={val_metrics.get('official_3d_mAP_0.3', 0.0):.4f}",
            f"p={val_metrics.get('official_detection_precision', 0.0):.4f}",
            f"r={val_metrics.get('official_detection_recall', 0.0):.4f}",
            f"f1={val_metrics.get('official_detection_f1', 0.0):.4f}",
            f"tp={int(val_metrics.get('official_detection_tp', 0))}",
            f"fp={int(val_metrics.get('official_detection_fp', 0))}",
            f"fn={int(val_metrics.get('official_detection_fn', 0))}",
        )
    if any(key.startswith("polar_") for key in val_metrics):
        print(
            "  polar:",
            f"bev@0.3={val_metrics.get('polar_bev_mAP_0.3', 0.0):.4f}",
            f"bev@0.5={val_metrics.get('polar_bev_mAP_0.5', 0.0):.4f}",
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


def create_tensorboard_writer(
        base_dir,
        run_relative_path=None,
        existing_log_dir=None,
    ):
    if existing_log_dir not in (None, ""):
        log_dir = os.path.abspath(os.path.expanduser(str(existing_log_dir)))
        if not os.path.isdir(log_dir):
            raise FileNotFoundError(
                f"TensorBoard resume directory not found: {log_dir}"
        )
        return SummaryWriter(log_dir=log_dir)

    if run_relative_path in (None, ""):
        raise ValueError(
            "run_relative_path is required when creating a TensorBoard run"
        )
    log_dir = os.path.join(base_dir, os.fspath(run_relative_path))

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
        configured_model_type=None,
        cartesian_training_workflow=None,
        loss_mode=None,
        train_scope=None,
        split_mode=None,
        train_sequences=None,
        val_sequences=None,
        domain_shift_train_branch=None,
        shared_train_sequences=None,
        source_train_sequences=None,
        target_train_sequences=None,
        target_test_sequences=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        training_eval_enabled=True,
        training_eval_train_set_enabled=False,
        training_eval_best_metric_key=None,
        training_eval_official_enabled=False,
        training_eval_official_version="revised",
        training_eval_iou_backend="auto",
        training_eval_iou_mode="easy",
        training_eval_detection_metrics_enabled=False,
        training_eval_ap_score_thresh=0.01,
        training_eval_score_thresh=0.3,
        training_eval_polar_enabled=False,
        training_eval_polar_iou_thresholds=None,
        training_eval_coco_style_enabled=False,
        training_eval_nuscenes_style_enabled=False,
        gt_object_ignore_override_path=None,
        train_control_split_enabled=False,
        train_control_split_dir=None,
        centerpoint_gwd_loss_weight=None,
        quality_loss_weight=None,
        quality_loss_active=None,
        model7_decoder_hidden_channels=None,
        box_coordinate_mode=None,
        cartesian_gt_root=None,
        weather_group=None,
        weather_group_source=None,
        sequence_information_path=None,
        test_sequence_weather=None,
    ):
    config_text = "\n".join([
        f"sequence: {cfg.sequence}",
        f"sequences: {getattr(cfg, 'sequences', None)}",
        f"split_mode: {split_mode}",
        f"domain_shift_train_branch: {domain_shift_train_branch}",
        f"shared_train_sequences: {shared_train_sequences}",
        f"source_train_sequences: {source_train_sequences}",
        f"target_train_sequences: {target_train_sequences}",
        f"target_test_sequences: {target_test_sequences}",
        f"train_sequences: {train_sequences}",
        f"val_sequences: {val_sequences}",
        f"train_sequence_half_selection: {train_sequence_half_selection}",
        f"train_sequence_half_ratio: {train_sequence_half_ratio}",
        f"model_type: {model_type}",
        f"configured_model_type: {configured_model_type}",
        f"cartesian_training_workflow: {cartesian_training_workflow}",
        f"loss_mode: {loss_mode}",
        f"train_scope: {train_scope}",
        f"box_coordinate_mode: {box_coordinate_mode}",
        f"weather_group: {weather_group}",
        f"weather_group_source: {weather_group_source}",
        f"sequence_information_path: {sequence_information_path}",
        f"test_sequence_weather: {test_sequence_weather}",
        f"cartesian_gt_root: {cartesian_gt_root}",
        f"training_eval_enabled: {training_eval_enabled}",
        f"training_eval_train_set_enabled: {training_eval_train_set_enabled}",
        f"num_epochs: {num_epochs}",
        f"batch_size: {batch_size}",
        f"train_size: {train_size}",
        f"val_size: {val_size}",
        f"learning_rate: {learning_rate}",
        f"max_detections: {max_detections}",
        f"num_classes: {num_classes}",
        f"class_names: {class_names}",
        f"training_eval_best_metric_key: {training_eval_best_metric_key}",
        f"training_eval_official_enabled: {training_eval_official_enabled}",
        f"training_eval_official_version: {training_eval_official_version}",
        f"training_eval_iou_backend: {training_eval_iou_backend}",
        f"training_eval_iou_mode: {training_eval_iou_mode}",
        f"training_eval_detection_metrics_enabled: {training_eval_detection_metrics_enabled}",
        f"training_eval_ap_score_thresh: {training_eval_ap_score_thresh}",
        f"training_eval_score_thresh: {training_eval_score_thresh}",
        f"training_eval_polar_enabled: {training_eval_polar_enabled}",
        f"training_eval_polar_iou_thresholds: {training_eval_polar_iou_thresholds}",
        f"training_eval_coco_style_enabled: {training_eval_coco_style_enabled}",
        f"training_eval_nuscenes_style_enabled: {training_eval_nuscenes_style_enabled}",
        f"gt_object_ignore_override_path: {gt_object_ignore_override_path}",
        f"train_control_split_enabled: {train_control_split_enabled}",
        f"train_control_split_dir: {train_control_split_dir}",
        f"centerpoint_gwd_loss_weight: {centerpoint_gwd_loss_weight}",
        f"quality_loss_weight: {quality_loss_weight}",
        f"quality_loss_active: {quality_loss_active}",
        f"model7_decoder_hidden_channels: {model7_decoder_hidden_channels}",
    ])
    writer.add_text("run/config", config_text, 0)
    writer.flush()


def write_tensorboard_metrics(writer, epoch, train_metrics, val_metrics, f1, learning_rate):
    writer.add_scalar("training_metrics/train_loss", train_metrics["train_loss"], epoch)
    writer.add_scalar("training_metrics/train_box_loss", train_metrics["train_box_loss"], epoch)
    if "train_heatmap_loss" in train_metrics:
        writer.add_scalar(
            "training_metrics/train_heatmap_loss",
            train_metrics["train_heatmap_loss"],
            epoch
        )
    else:
        writer.add_scalar("training_metrics/train_cls_loss", train_metrics["train_cls_loss"], epoch)
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
    if "val_heatmap_loss" in val_metrics:
        writer.add_scalar(
            "validation_metrics/val_heatmap_loss",
            val_metrics["val_heatmap_loss"],
            epoch
        )
    else:
        writer.add_scalar("validation_metrics/val_cls_loss", val_metrics["val_cls_loss"], epoch)
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
    for key in (
        "official_bev_mAP_0.3",
        "official_3d_mAP_0.3",
        "official_detection_tp",
        "official_detection_fp",
        "official_detection_fn",
        "official_detection_precision",
        "official_detection_recall",
        "official_detection_f1",
    ):
        value = val_metrics.get(key)
        if isinstance(value, (int, float)):
            writer.add_scalar(f"validation_metrics/{key}", value, epoch)
    for key, value in val_metrics.items():
        if key.startswith("polar_") and isinstance(value, (int, float)):
            writer.add_scalar(f"validation_metrics/{key}", value, epoch)

    writer.add_scalar("parameters/learning_rate", learning_rate, epoch)
    writer.flush()
