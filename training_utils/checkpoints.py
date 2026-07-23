import os
import shutil
import copy
from datetime import datetime

import torch

from training_utils.torch_load import load_torch_checkpoint


MODEL_RUN_NAME_PREFIXES = {
    "model1": "con2d_heatmap_model1",
    "model2": "bifpn_heatmap_model2",
    "model3": "fpn_nodeform_heatmap_model3",
    "model4": "deform_heatmap_model4",
    "model5": "fpn_heatmap_model5",
    "model6": "fpn_quality_heatmap_model6",
    "model7": "swin_heatmap_model7",
    "model8": "cfe_heatmap_model8",
    "model9": "cfe_bifpn_heatmap_model9",
    "model10": "fpn_split_heatmap_model10",
}

EXPERIMENT_NAME = "object_detection"


def _create_unique_checkpoint_dir(checkpoint_dir):
    suffix = 1
    unique_checkpoint_dir = checkpoint_dir
    while os.path.exists(unique_checkpoint_dir):
        unique_checkpoint_dir = f"{checkpoint_dir}_{suffix}"
        suffix += 1

    os.makedirs(unique_checkpoint_dir, exist_ok=False)
    return unique_checkpoint_dir


def format_sequence_run_name(sequences):
    if isinstance(sequences, int):
        return f"seq{sequences}"

    sequences = tuple(sequences)
    if len(sequences) == 1:
        return f"seq{sequences[0]}"

    ranges = []
    start = sequences[0]
    previous = sequences[0]
    for sequence in sequences[1:]:
        if sequence == previous + 1:
            previous = sequence
            continue

        ranges.append((start, previous))
        start = sequence
        previous = sequence
    ranges.append((start, previous))

    range_texts = [
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges
    ]
    return 'seq' + '_'.join(range_texts)


def get_model_run_name_prefix(model_type):
    if model_type is None or model_type == "":
        return None

    model_text = str(model_type)
    if model_text.startswith("model"):
        model_number = model_text[len("model"):]
        if model_number.isdigit():
            return f"model_{model_number}"

    return model_text


def format_model_sequence_run_name(sequences, model_type=None):
    sequence_name = format_sequence_run_name(sequences)
    model_prefix = get_model_run_name_prefix(model_type)
    if model_prefix is None:
        return sequence_name

    return f"{model_prefix}__{sequence_name}"


def format_timestamp_model_sequence_run_name(sequences, model_type=None, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    sequence_name = format_sequence_run_name(sequences)
    model_prefix = get_model_run_name_prefix(model_type)
    if model_prefix is None:
        return f"{timestamp}__{sequence_name}"

    return f"{timestamp}__{model_prefix}__{sequence_name}"


def _configured_sequences(cfg):
    sequences = getattr(cfg, "sequences", None)
    if sequences is None:
        sequences = (cfg.sequence,)
    if isinstance(sequences, int):
        sequences = (sequences,)
    return tuple(sequences)


def _payload_model_type(payload):
    if isinstance(payload, dict):
        config = payload.get("config", {})
        return config.get("run_model_type", config.get("model_type"))
    return None


def _payload_sequences(payload):
    if not isinstance(payload, dict):
        return None

    config = payload.get("config", {})
    sequences = config.get("sequences")
    if sequences is not None:
        return tuple(sequences)

    sequence = config.get("sequence")
    if sequence is not None:
        return (sequence,)

    return None


def format_checkpoint_filename(
        name_prefix,
        epoch,
        saved_at,
        metric_key,
        metric_value,
        model_type,
        sequences,
    ):
    model_name = get_model_run_name_prefix(model_type) or "model_unknown"
    sequence_name = format_sequence_run_name(sequences)
    date_text = str(saved_at).replace("-", "").replace("/", "")
    if len(date_text) >= 8 and date_text[:8].isdigit():
        date_text = date_text[4:8]

    filename_parts = [date_text, model_name]
    if name_prefix not in (None, "", "candidate"):
        filename_parts.append(str(name_prefix))
    filename_parts.extend([f"epoch_{epoch:03d}", sequence_name])
    return "_".join(filename_parts) + ".pth"


def create_checkpoint_run_dir(base_dir, experiment_name, sequence, model_type=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = format_timestamp_model_sequence_run_name(sequence, model_type, timestamp)
    checkpoint_dir = os.path.join(base_dir, experiment_name, run_name)
    return _create_unique_checkpoint_dir(checkpoint_dir)


def create_checkpoint_run_dirs(base_dir, experiment_name, sequences, model_type=None):
    sequences = tuple(sequences)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = format_timestamp_model_sequence_run_name(sequences, model_type, timestamp)
    checkpoint_dir = os.path.join(base_dir, experiment_name, run_name)
    return {sequences: _create_unique_checkpoint_dir(checkpoint_dir)}


def metric_for_filename(value):
    return f"{value:.4f}".replace(".", "p")


def metric_key_for_filename(metric_key):
    key_text = str(metric_key or "metric")
    key_text = key_text.replace(".", "p")
    return "".join(
        character if (character.isalnum() or character in {"_", "-"}) else "_"
        for character in key_text
    )


def build_checkpoint_payload(
        model,
        optimizer,
        scheduler,
        args,
        cfg,
        epoch,
        train_metrics,
        val_metrics,
        f1,
        learning_rate,
        saved_at,
        is_best,
        clone_for_memory=False
    ):
    model_for_state_dict = model.module if isinstance(model, torch.nn.DataParallel) else model
    payload = {
        "epoch": epoch,
        "saved_at": saved_at,
        "model_state_dict": model_for_state_dict.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "f1": f1,
        "learning_rate": learning_rate,
        "is_best": is_best,
        "config": {
            "sequence": cfg.sequence,
            "sequences": getattr(cfg, "sequences", None),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_detections": args.max_detections,
            "heatmap_radius": getattr(args, "heatmap_radius", None),
            "centerpoint_gwd_loss_weight": getattr(
                args,
                "centerpoint_gwd_loss_weight",
                None,
            ),
            "quality_loss_weight": getattr(args, "quality_loss_weight", None),
            "num_classes": args.num_classes,
            "model_type": getattr(args, "model_type", None),
            "configured_model_type": getattr(
                args,
                "configured_model_type",
                getattr(args, "model_type", None),
            ),
            "cartesian_training_workflow": getattr(
                args,
                "cartesian_training_workflow",
                None,
            ),
            "loss_mode": getattr(args, "loss_mode", "auto"),
            "run_model_type": getattr(args, "run_model_type", getattr(args, "model_type", None)),
            "class_names": getattr(args, "class_names", None),
            "class_to_idx": getattr(args, "class_to_idx", None),
            "include_bus_as_target": getattr(args, "include_bus_as_target", None),
            "ignore_class_names": getattr(args, "ignore_class_names", None),
            "ignore_mask_margin": getattr(args, "ignore_mask_margin", None),
            "ignore_mask_expand_ratio": getattr(args, "ignore_mask_expand_ratio", None),
            "gt_object_ignore_override_path": getattr(args, "gt_object_ignore_override_path", None),
            "train_control_split_enabled": getattr(args, "train_control_split_enabled", False),
            "train_control_split_dir": getattr(args, "train_control_split_dir", None),
            "init_from_checkpoint": getattr(args, "init_from_checkpoint", None),
            "train_ratio": args.train_ratio,
            "train_scope": getattr(args, "train_scope", "full"),
            "box_coordinate_mode": getattr(
                args,
                "box_coordinate_mode",
                "polar",
            ),
            "cartesian_gt_root": getattr(
                args,
                "cartesian_gt_root",
                None,
            ),
            "training_eval_enabled": getattr(args, "training_eval_enabled", True),
            "best_metric_key": (
                getattr(args, "best_metric_key", "auto")
                if getattr(args, "training_eval_enabled", True)
                else None
            ),
            "official_eval_enabled": getattr(args, "official_eval_enabled", False),
            "official_eval_version": getattr(args, "official_eval_version", "revised"),
            "official_eval_iou_backend": getattr(args, "official_eval_iou_backend", "auto"),
            "official_eval_iou_mode": getattr(args, "official_eval_iou_mode", "easy"),
            "polar_eval_enabled": getattr(
                args,
                "polar_eval_enabled",
                False,
            ),
            "polar_iou_thresholds": getattr(
                args,
                "polar_iou_thresholds",
                None,
            ),
            "coco_style_eval_enabled": getattr(args, "coco_style_eval_enabled", False),
            "nuscenes_style_eval_enabled": getattr(args, "nuscenes_style_eval_enabled", False),
            "split_mode": getattr(args, "split_mode", None),
            "split_dir": getattr(args, "split_dir", None),
            "train_sequences": getattr(args, "train_sequences", None),
            "val_sequences": getattr(args, "val_sequences", None),
            "sequence_tail_val_ratio": getattr(args, "sequence_tail_val_ratio", None),
            "sequence_tail_boundary_drop_frames": getattr(
                args,
                "sequence_tail_boundary_drop_frames",
                None,
            ),
            "controled_sequences": getattr(args, "controled_sequences", None),
            "reference_sequences": getattr(args, "reference_sequences", None),
            "controlled_split_base_dir": getattr(args, "controlled_split_base_dir", None),
            "control_window_position": getattr(args, "control_window_position", None),
            "control_ridx_bins": getattr(args, "control_ridx_bins", None),
            "control_num_trials": getattr(args, "control_num_trials", None),
            "seed": args.seed,
            "limit_samples": args.limit_samples,
        },
    }

    if "selection_metric_key" in val_metrics:
        payload["selection_metric_key"] = val_metrics["selection_metric_key"]
        payload["selection_metric_value"] = val_metrics[
            "selection_metric_value"
        ]
    if "mAP" in val_metrics:
        payload["mAP"] = val_metrics["mAP"]

    if clone_for_memory:
        payload = clone_checkpoint_payload_for_memory(payload)

    return payload


def clone_checkpoint_payload_for_memory(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {
            key: clone_checkpoint_payload_for_memory(child_value)
            for key, child_value in value.items()
        }
    if isinstance(value, list):
        return [clone_checkpoint_payload_for_memory(child_value) for child_value in value]
    if isinstance(value, tuple):
        return tuple(clone_checkpoint_payload_for_memory(child_value) for child_value in value)
    return copy.deepcopy(value)


def get_model_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def save_epoch_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        scheduler,
        args,
        cfg,
        epoch,
        train_metrics,
        val_metrics,
        f1,
        learning_rate,
        is_best
    ):
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_key = val_metrics.get("selection_metric_key", "mAP")
    metric_value = val_metrics.get(
        "selection_metric_value",
        val_metrics.get("mAP", 0.0),
    )
    filename = format_checkpoint_filename(
        name_prefix=None,
        epoch=epoch,
        saved_at=saved_at,
        metric_key=metric_key,
        metric_value=metric_value,
        model_type=getattr(args, "run_model_type", getattr(args, "model_type", None)),
        sequences=_configured_sequences(cfg),
    )
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        cfg=cfg,
        epoch=epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        f1=f1,
        learning_rate=learning_rate,
        saved_at=saved_at,
        is_best=is_best
    )

    torch.save(payload, checkpoint_path)

    return checkpoint_path


def save_named_checkpoint_copy(
        checkpoint_dir,
        source_checkpoint_path,
        best_epoch,
        best_map,
        name_prefix,
        model_type=None,
        sequences=None,
        metric_key=None,
    ):
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type is None or sequences is None:
        source_checkpoint = load_torch_checkpoint(source_checkpoint_path, map_location="cpu")
        if model_type is None:
            model_type = _payload_model_type(source_checkpoint)
        if sequences is None:
            sequences = _payload_sequences(source_checkpoint)
        if metric_key is None and isinstance(source_checkpoint, dict):
            metric_key = source_checkpoint.get("selection_metric_key")
    if sequences is None:
        sequences = ("unknown",)
    if metric_key is None:
        metric_key = "mAP"

    best_filename = format_checkpoint_filename(
        name_prefix=name_prefix,
        epoch=best_epoch,
        saved_at=saved_at,
        metric_key=metric_key,
        metric_value=best_map,
        model_type=model_type,
        sequences=sequences,
    )
    best_checkpoint_path = os.path.join(checkpoint_dir, best_filename)
    shutil.copy2(source_checkpoint_path, best_checkpoint_path)
    return best_checkpoint_path


def save_named_checkpoint_payload(
        checkpoint_dir,
        payload,
        best_epoch,
        best_map,
        name_prefix,
        model_type=None,
        sequences=None,
        metric_key=None,
    ):
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type is None:
        model_type = _payload_model_type(payload)
    if sequences is None:
        sequences = _payload_sequences(payload)
    if metric_key is None and isinstance(payload, dict):
        metric_key = payload.get("selection_metric_key")
    if sequences is None:
        sequences = ("unknown",)
    if metric_key is None:
        metric_key = "mAP"

    best_filename = format_checkpoint_filename(
        name_prefix=name_prefix,
        epoch=best_epoch,
        saved_at=saved_at,
        metric_key=metric_key,
        metric_value=best_map,
        model_type=model_type,
        sequences=sequences,
    )
    best_checkpoint_path = os.path.join(checkpoint_dir, best_filename)
    torch.save(payload, best_checkpoint_path)
    return best_checkpoint_path


def remove_named_checkpoints(checkpoint_dir, name_prefix):
    if not os.path.isdir(checkpoint_dir):
        return

    for filename in os.listdir(checkpoint_dir):
        is_legacy_match = filename.startswith(f"{name_prefix}_epoch_")
        is_new_match = f"_{name_prefix}_epoch_" in filename
        if (is_legacy_match or is_new_match) and filename.endswith(".pth"):
            os.remove(os.path.join(checkpoint_dir, filename))


def save_replacing_named_checkpoint_copy(
        checkpoint_dir,
        source_checkpoint_path,
        best_epoch,
        best_map,
        name_prefix,
        model_type=None,
        sequences=None,
        metric_key=None,
    ):
    remove_named_checkpoints(checkpoint_dir, name_prefix)
    return save_named_checkpoint_copy(
        checkpoint_dir=checkpoint_dir,
        source_checkpoint_path=source_checkpoint_path,
        best_epoch=best_epoch,
        best_map=best_map,
        name_prefix=name_prefix,
        model_type=model_type,
        sequences=sequences,
        metric_key=metric_key,
    )


def save_replacing_named_checkpoint_payload(
        checkpoint_dir,
        payload,
        best_epoch,
        best_map,
        name_prefix,
        model_type=None,
        sequences=None,
        metric_key=None,
    ):
    remove_named_checkpoints(checkpoint_dir, name_prefix)
    return save_named_checkpoint_payload(
        checkpoint_dir=checkpoint_dir,
        payload=payload,
        best_epoch=best_epoch,
        best_map=best_map,
        name_prefix=name_prefix,
        model_type=model_type,
        sequences=sequences,
        metric_key=metric_key,
    )


def save_best_checkpoint_copy(
        checkpoint_dir,
        source_checkpoint_path,
        best_epoch,
        best_map,
        model_type=None,
        sequences=None,
        metric_key=None,
    ):
    return save_named_checkpoint_copy(
        checkpoint_dir=checkpoint_dir,
        source_checkpoint_path=source_checkpoint_path,
        best_epoch=best_epoch,
        best_map=best_map,
        name_prefix="best",
        model_type=model_type,
        sequences=sequences,
        metric_key=metric_key,
    )
