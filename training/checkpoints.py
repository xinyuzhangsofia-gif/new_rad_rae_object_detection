import os
import shutil
import copy
from dataclasses import dataclass
from datetime import datetime

import torch

from training.configuration import (
    normalize_train_sequence_half_ratio,
    normalize_train_sequence_half_selection,
)
from training.torch_load import load_torch_checkpoint


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


def checkpoint_run_relative_path(checkpoint_dir, checkpoint_base_dir):
    """Return the semantic run path below the configured checkpoint root."""
    checkpoint_root = os.path.abspath(os.path.expanduser(str(checkpoint_base_dir)))
    run_dir = os.path.abspath(os.path.expanduser(str(checkpoint_dir)))
    if os.path.commonpath((checkpoint_root, run_dir)) != checkpoint_root:
        raise ValueError(
            f"Checkpoint directory {run_dir!r} is outside root {checkpoint_root!r}"
        )
    return os.path.relpath(run_dir, checkpoint_root)


def _create_unique_checkpoint_dir(checkpoint_dir):
    suffix = 1
    unique_checkpoint_dir = checkpoint_dir
    while os.path.exists(unique_checkpoint_dir):
        unique_checkpoint_dir = f"{checkpoint_dir}_{suffix}"
        suffix += 1

    os.makedirs(unique_checkpoint_dir, exist_ok=False)
    return unique_checkpoint_dir


def format_sequence_run_name(
        sequences,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
    ):
    if isinstance(sequences, int):
        sequences = (sequences,)

    sequences = tuple(sequences)
    half_selection = normalize_train_sequence_half_selection(
        train_sequence_half_selection
    )
    if half_selection:
        half_ratio = normalize_train_sequence_half_ratio(
            0.5
            if train_sequence_half_ratio is None
            else train_sequence_half_ratio
        )
        ratio_suffix = (
            ""
            if abs(half_ratio - 0.5) <= 1e-12
            else f"{half_ratio * 100:g}pct"
        )
        values = [
            (
                f"{int(sequence)}_{half_selection[int(sequence)]}{ratio_suffix}"
                if int(sequence) in half_selection
                else str(int(sequence))
            )
            for sequence in sequences
        ]
        return "seq" + "_".join(values)
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


def format_timestamp_model_sequence_run_name(
        sequences,
        model_type=None,
        timestamp=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
    ):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    sequence_name = format_sequence_run_name(
        sequences,
        train_sequence_half_selection=train_sequence_half_selection,
        train_sequence_half_ratio=train_sequence_half_ratio,
    )
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


def _checkpoint_sequences(args, cfg):
    train_sequences = getattr(args, "train_sequences", None)
    val_sequences = getattr(args, "val_sequences", None)
    if train_sequences not in (None, "", ()):
        if isinstance(train_sequences, int):
            train_sequences = (train_sequences,)
        if isinstance(val_sequences, int):
            val_sequences = (val_sequences,)
        combined = list(train_sequences)
        for sequence in val_sequences or ():
            if sequence not in combined:
                combined.append(sequence)
        return tuple(combined)
    return _configured_sequences(cfg)


def _payload_model_type(payload):
    if isinstance(payload, dict):
        config = payload.get("config", {})
        return config.get("run_model_type", config.get("model_type"))
    return None


def _payload_sequences(payload):
    if not isinstance(payload, dict):
        return None

    config = payload.get("config", {})
    train_sequences = config.get("train_sequences")
    val_sequences = config.get("val_sequences")
    if train_sequences not in (None, "", ()):
        if isinstance(train_sequences, int):
            train_sequences = (train_sequences,)
        if isinstance(val_sequences, int):
            val_sequences = (val_sequences,)
        combined = list(train_sequences)
        for sequence in val_sequences or ():
            if sequence not in combined:
                combined.append(sequence)
        return tuple(combined)

    sequences = config.get("sequences")
    if sequences is not None:
        return tuple(sequences)

    sequence = config.get("sequence")
    if sequence is not None:
        return (sequence,)

    return None


def _payload_half_selection(payload):
    if not isinstance(payload, dict):
        return {}, None
    config = payload.get("config", {})
    return (
        config.get("train_sequence_half_selection"),
        config.get("train_sequence_half_ratio"),
    )


def _payload_checkpoint_filename_style(payload):
    if not isinstance(payload, dict):
        return "legacy"
    config = payload.get("config", {})
    return str(config.get("checkpoint_filename_style", "legacy")).strip().lower()


def format_checkpoint_filename(
        name_prefix,
        epoch,
        saved_at,
        metric_key,
        metric_value,
        model_type,
        sequences,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        compact=False,
    ):
    date_text = str(saved_at).replace("-", "").replace("/", "")
    if len(date_text) >= 8 and date_text[:8].isdigit():
        date_text = date_text[4:8]

    if compact:
        filename_parts = [date_text]
        if name_prefix not in (None, "", "candidate"):
            filename_parts.append(str(name_prefix))
        filename_parts.append(f"epoch_{epoch:03d}")
        return "_".join(filename_parts) + ".pth"

    model_name = get_model_run_name_prefix(model_type) or "model_unknown"
    sequence_name = format_sequence_run_name(
        sequences,
        train_sequence_half_selection=train_sequence_half_selection,
        train_sequence_half_ratio=train_sequence_half_ratio,
    )
    filename_parts = [date_text, model_name]
    if name_prefix not in (None, "", "candidate"):
        filename_parts.append(str(name_prefix))
    filename_parts.extend([f"epoch_{epoch:03d}", sequence_name])
    return "_".join(filename_parts) + ".pth"


def create_checkpoint_run_dir(
        base_dir,
        experiment_name,
        sequence,
        model_type=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
    ):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = format_timestamp_model_sequence_run_name(
        sequence,
        model_type,
        timestamp,
        train_sequence_half_selection=train_sequence_half_selection,
        train_sequence_half_ratio=train_sequence_half_ratio,
    )
    checkpoint_dir = os.path.join(base_dir, experiment_name, run_name)
    return _create_unique_checkpoint_dir(checkpoint_dir)


def create_checkpoint_run_dirs(
        base_dir,
        experiment_name,
        sequences,
        model_type=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        domain_shift_experiment_enabled=False,
        domain_shift_train_branch=None,
        weather_group=None,
        train_sequences=None,
        test_sequences=None,
    ):
    sequences = tuple(sequences)
    branch = (
        None
        if domain_shift_train_branch in (None, "")
        else str(domain_shift_train_branch).strip().lower()
    )
    use_weather_train_test_layout = (
        bool(domain_shift_experiment_enabled)
        or branch in {"source", "target"}
    )
    if use_weather_train_test_layout:
        if weather_group in (None, ""):
            raise ValueError(
                "Weather-related Domain Shift checkpoints require weather_group"
            )
        if train_sequences in (None, "", ()):
            raise ValueError(
                "Weather-related Domain Shift checkpoints require train_sequences"
            )
        if test_sequences in (None, "", ()):
            raise ValueError(
                "Weather-related Domain Shift checkpoints require test_sequences"
            )

        weather_name = "".join(
            character
            if (character.isalnum() or character in {"_", "-"})
            else "_"
            for character in str(weather_group).strip().lower()
        ).strip("_")
        if weather_name == "":
            weather_name = "weather_unknown"
        train_name = format_sequence_run_name(
            train_sequences,
            train_sequence_half_selection=train_sequence_half_selection,
            train_sequence_half_ratio=train_sequence_half_ratio,
        )
        test_name = format_sequence_run_name(test_sequences)
        date_text = datetime.now().strftime("%m%d")
        run_name = f"{date_text}_train_{train_name}_test_{test_name}"
        checkpoint_dir = os.path.join(base_dir, weather_name, run_name)
        return {sequences: _create_unique_checkpoint_dir(checkpoint_dir)}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = format_timestamp_model_sequence_run_name(
        sequences,
        model_type,
        timestamp,
        train_sequence_half_selection=train_sequence_half_selection,
        train_sequence_half_ratio=train_sequence_half_ratio,
    )
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
        "weather_group": getattr(args, "weather_group", None),
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
            "checkpoint_filename_style": getattr(
                args,
                "checkpoint_filename_style",
                "legacy",
            ),
            "resume_checkpoint": getattr(args, "resume_checkpoint", None),
            "resume_save_in_checkpoint_dir": getattr(
                args,
                "resume_save_in_checkpoint_dir",
                False,
            ),
            "resume_tensorboard_log_dir": getattr(
                args,
                "resume_tensorboard_log_dir",
                None,
            ),
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
            "model7_decoder_hidden_channels": getattr(
                args,
                "model7_decoder_hidden_channels",
                None,
            ),
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
            "train_scope": getattr(args, "train_scope", "full"),
            "box_coordinate_mode": getattr(
                args,
                "box_coordinate_mode",
                "polar",
            ),
            "weather_group": getattr(args, "weather_group", None),
            "weather_group_source": getattr(
                args,
                "weather_group_source",
                None,
            ),
            "sequence_information_path": getattr(
                args,
                "sequence_information_path",
                None,
            ),
            "test_sequence_weather": getattr(
                args,
                "test_sequence_weather",
                None,
            ),
            "cartesian_gt_root": getattr(
                args,
                "cartesian_gt_root",
                None,
            ),
            "training_eval_enabled": getattr(args, "training_eval_enabled", True),
            "training_eval_train_set_enabled": getattr(
                args,
                "training_eval_train_set_enabled",
                False,
            ),
            "training_eval_best_metric_key": (
                getattr(args, "training_eval_best_metric_key", "auto")
                if getattr(args, "training_eval_enabled", True)
                else None
            ),
            "training_eval_official_enabled": getattr(
                args,
                "training_eval_official_enabled",
                False,
            ),
            "training_eval_official_version": getattr(
                args,
                "training_eval_official_version",
                "revised",
            ),
            "training_eval_iou_backend": getattr(
                args,
                "training_eval_iou_backend",
                "auto",
            ),
            "training_eval_iou_mode": getattr(
                args,
                "training_eval_iou_mode",
                "easy",
            ),
            "training_eval_detection_metrics_enabled": getattr(
                args,
                "training_eval_detection_metrics_enabled",
                False,
            ),
            "training_eval_ap_score_thresh": getattr(
                args,
                "training_eval_ap_score_thresh",
                0.01,
            ),
            "training_eval_score_thresh": getattr(
                args,
                "training_eval_score_thresh",
                0.3,
            ),
            "training_eval_coco_style_enabled": getattr(
                args,
                "training_eval_coco_style_enabled",
                False,
            ),
            "training_eval_nuscenes_style_enabled": getattr(
                args,
                "training_eval_nuscenes_style_enabled",
                False,
            ),
            "split_mode": getattr(args, "split_mode", None),
            "split_dir": getattr(args, "split_dir", None),
            "domain_shift_experiment_enabled": getattr(
                args,
                "domain_shift_experiment_enabled",
                False,
            ),
            "domain_shift_train_branch": getattr(
                args,
                "domain_shift_train_branch",
                None,
            ),
            "shared_train_sequences": getattr(
                args,
                "shared_train_sequences",
                None,
            ),
            "source_train_sequences": getattr(
                args,
                "source_train_sequences",
                None,
            ),
            "target_train_sequences": getattr(
                args,
                "target_train_sequences",
                None,
            ),
            "target_test_sequences": getattr(
                args,
                "target_test_sequences",
                None,
            ),
            "train_sequences": getattr(args, "train_sequences", None),
            "val_sequences": getattr(args, "val_sequences", None),
            "train_sequence_half_selection": getattr(
                args,
                "train_sequence_half_selection",
                None,
            ),
            "train_sequence_half_ratio": getattr(
                args,
                "train_sequence_half_ratio",
                None,
            ),
            "controled_sequences": getattr(args, "controled_sequences", None),
            "reference_sequences": getattr(args, "reference_sequences", None),
            "controlled_split_base_dir": getattr(args, "controlled_split_base_dir", None),
            "control_window_position": getattr(args, "control_window_position", None),
            "control_class_names": getattr(args, "control_class_names", None),
            "control_range_m_bins": getattr(args, "control_range_m_bins", None),
            # Keep the legacy field readable for older checkpoints/configs.
            "control_ridx_bins": getattr(args, "control_ridx_bins", None),
            "control_num_trials": getattr(args, "control_num_trials", None),
            "control_total_bbox_tolerance_ratio": getattr(
                args,
                "control_total_bbox_tolerance_ratio",
                None,
            ),
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
        sequences=_checkpoint_sequences(args, cfg),
        train_sequence_half_selection=getattr(
            args,
            "train_sequence_half_selection",
            None,
        ),
        train_sequence_half_ratio=getattr(
            args,
            "train_sequence_half_ratio",
            None,
        ),
        compact=(
            str(
                getattr(args, "checkpoint_filename_style", "legacy")
            ).strip().lower()
            == "compact"
        ),
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
    source_checkpoint = load_torch_checkpoint(source_checkpoint_path, map_location="cpu")
    if model_type is None:
        model_type = _payload_model_type(source_checkpoint)
    if sequences is None:
        sequences = _payload_sequences(source_checkpoint)
    if metric_key is None and isinstance(source_checkpoint, dict):
        metric_key = source_checkpoint.get("selection_metric_key")
    half_selection, half_ratio = _payload_half_selection(source_checkpoint)
    compact = (
        _payload_checkpoint_filename_style(source_checkpoint) == "compact"
    )
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
        train_sequence_half_selection=half_selection,
        train_sequence_half_ratio=half_ratio,
        compact=compact,
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
    half_selection, half_ratio = _payload_half_selection(payload)
    compact = _payload_checkpoint_filename_style(payload) == "compact"
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
        train_sequence_half_selection=half_selection,
        train_sequence_half_ratio=half_ratio,
        compact=compact,
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

def default_best_metric_key(
        training_eval_enabled=True,
        official_eval_enabled=False,
        official_eval_iou_mode="easy",
    ):
    if not training_eval_enabled:
        return None
    if official_eval_enabled:
        iou_suffix = {
            "easy": "0.3",
            "mod": "0.5",
            "hard": "0.7",
            "all": "0.3",
        }.get(official_eval_iou_mode, "0.3")
        return f"official_bev_mAP_{iou_suffix}"
    return "mAP"


def resolve_best_metric_key(
        requested_key,
        val_metrics,
        training_eval_enabled=True,
        official_eval_enabled=False,
        official_eval_iou_mode="easy",
    ):
    if not training_eval_enabled:
        return None

    metric_key = requested_key
    if metric_key is None or metric_key == "" or metric_key == "auto":
        metric_key = default_best_metric_key(
            training_eval_enabled=training_eval_enabled,
            official_eval_enabled=official_eval_enabled,
            official_eval_iou_mode=official_eval_iou_mode,
        )

    if metric_key in val_metrics:
        return metric_key

    if "val_loss" in val_metrics:
        return "val_loss"

    if "mAP" in val_metrics:
        return "mAP"

    available_keys = sorted(val_metrics.keys())
    raise KeyError(f"Metric {metric_key!r} not found in val_metrics. Available keys: {available_keys}")


def selection_metric_value(val_metrics):
    if (
        "selection_metric_key" not in val_metrics
        or "selection_metric_value" not in val_metrics
    ):
        raise ValueError("Best-checkpoint selection metric is not available.")
    metric_key = val_metrics["selection_metric_key"]
    metric_value = val_metrics["selection_metric_value"]
    return metric_key, float(metric_value)


def metric_prefers_lower(metric_key):
    return str(metric_key).endswith("_loss") or str(metric_key) == "val_loss"


def build_epoch_eval_metrics(
        train_metrics,
        eval_metrics,
        val_loss_metrics,
        training_eval_enabled=True,
        best_metric_key="auto",
        official_eval_enabled=False,
        official_eval_iou_mode="easy",
    ):
    del train_metrics

    val_metrics = val_loss_metrics.copy()
    if training_eval_enabled:
        val_metrics.setdefault("mAP", 0.0)
        if eval_metrics is not None:
            val_metrics.update(eval_metrics["val_eval_metrics"])
    f1 = float(val_metrics.get("official_detection_f1", 0.0))

    if training_eval_enabled:
        resolved_metric_key = resolve_best_metric_key(
            requested_key=best_metric_key,
            val_metrics=val_metrics,
            training_eval_enabled=True,
            official_eval_enabled=official_eval_enabled,
            official_eval_iou_mode=official_eval_iou_mode,
        )
        val_metrics["selection_metric_key"] = resolved_metric_key
        val_metrics["selection_metric_value"] = float(
            val_metrics[resolved_metric_key]
        )

    return val_metrics, f1


@dataclass
class BestCheckpointState:
    map_score: float = -1.0
    metric_key: str = "mAP"
    epoch: int = -1
    train_metrics: object = None
    metrics: object = None
    f1: float = 0.0
    checkpoint_path: object = None
    checkpoint_payload: object = None
    global_best_path: object = None

    def reset(self):
        self.map_score = -1.0
        self.metric_key = "mAP"
        self.epoch = -1
        self.train_metrics = None
        self.metrics = None
        self.f1 = 0.0
        self.checkpoint_path = None
        self.checkpoint_payload = None
        self.global_best_path = None

    def update(
            self,
            epoch,
            train_metrics,
            val_metrics,
            f1,
            checkpoint_path=None,
            checkpoint_payload=None,
            global_best_path=None
        ):
        metric_key, metric_value = selection_metric_value(val_metrics)
        self.map_score = metric_value
        self.metric_key = metric_key
        self.epoch = epoch
        self.train_metrics = train_metrics.copy()
        self.metrics = val_metrics.copy()
        self.f1 = f1
        self.checkpoint_path = checkpoint_path
        self.checkpoint_payload = checkpoint_payload
        self.global_best_path = global_best_path

    def is_better(self, val_metrics):
        metric_key, metric_value = selection_metric_value(val_metrics)
        if self.epoch < 0:
            return True
        if metric_prefers_lower(metric_key):
            return metric_value < self.map_score
        return metric_value > self.map_score


def save_epoch_and_update_best_checkpoint(
        best_state,
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
        total_epochs,
        checkpoint_epoch_step,
        best_selection_enabled=True,
    ):
    is_best = (
        best_selection_enabled
        and best_state.is_better(val_metrics)
    )
    should_save_checkpoint = (
        epoch % checkpoint_epoch_step == 0
        or epoch == total_epochs
    )

    if not should_save_checkpoint and not is_best:
        return None

    if should_save_checkpoint:
        checkpoint_path = save_epoch_checkpoint(
            checkpoint_dir=checkpoint_dir,
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
            is_best=is_best
        )
        checkpoint_payload = None
    else:
        checkpoint_path = None
        checkpoint_payload = build_checkpoint_payload(
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
            saved_at="memory",
            is_best=is_best,
            clone_for_memory=True
        )

    if is_best:
        best_metric_key, best_metric_value = selection_metric_value(val_metrics)
        if checkpoint_path is not None:
            global_best_path = save_replacing_named_checkpoint_copy(
                checkpoint_dir=checkpoint_dir,
                source_checkpoint_path=checkpoint_path,
                best_epoch=epoch,
                best_map=best_metric_value,
                name_prefix="global_best",
                model_type=getattr(args, "model_type", None),
                sequences=getattr(cfg, "sequences", None) or (cfg.sequence,),
                metric_key=best_metric_key,
            )
        else:
            global_best_path = save_replacing_named_checkpoint_payload(
                checkpoint_dir=checkpoint_dir,
                payload=checkpoint_payload,
                best_epoch=epoch,
                best_map=best_metric_value,
                name_prefix="global_best",
                model_type=getattr(args, "model_type", None),
                sequences=getattr(cfg, "sequences", None) or (cfg.sequence,),
                metric_key=best_metric_key,
            )

        best_state.update(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=checkpoint_payload,
            global_best_path=global_best_path
        )

    return checkpoint_path


def save_window_best_checkpoint_if_ready(
        window_best_state,
        checkpoint_dirs,
        checkpoint_key,
        checkpoint_path,
        epoch,
        total_epochs,
        train_metrics,
        val_metrics,
        f1,
        window_size
    ):
    if checkpoint_path is None:
        return None, None

    if window_best_state.is_better(val_metrics):
        window_best_state.update(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            checkpoint_path=checkpoint_path
        )

    should_save_window_best = (
        window_best_state.checkpoint_path is not None
        and (epoch % window_size == 0 or epoch == total_epochs)
    )
    if not should_save_window_best:
        return None, None

    best_checkpoint_paths = {}
    for sequence, sequence_checkpoint_dir in checkpoint_dirs.items():
        best_checkpoint_paths[sequence] = save_best_checkpoint_copy(
            checkpoint_dir=sequence_checkpoint_dir,
            source_checkpoint_path=window_best_state.checkpoint_path,
            best_epoch=window_best_state.epoch,
            best_map=window_best_state.map_score,
            sequences=sequence,
            metric_key=window_best_state.metric_key,
        )
    best_checkpoint_path = best_checkpoint_paths[checkpoint_key]
    window_best_state.reset()

    return best_checkpoint_path, best_checkpoint_paths


def save_global_best_checkpoint(best_state, checkpoint_dirs, checkpoint_key):
    if best_state.global_best_path is not None:
        return best_state.global_best_path, {checkpoint_key: best_state.global_best_path}

    if best_state.checkpoint_path is None and best_state.checkpoint_payload is None:
        return None, None

    global_best_checkpoint_paths = {}
    for sequence, sequence_checkpoint_dir in checkpoint_dirs.items():
        if best_state.checkpoint_path is not None:
            global_best_checkpoint_paths[sequence] = save_named_checkpoint_copy(
                checkpoint_dir=sequence_checkpoint_dir,
                source_checkpoint_path=best_state.checkpoint_path,
                best_epoch=best_state.epoch,
                best_map=best_state.map_score,
                name_prefix="global_best",
                sequences=sequence,
                metric_key=best_state.metric_key,
            )
        else:
            global_best_checkpoint_paths[sequence] = save_named_checkpoint_payload(
                checkpoint_dir=sequence_checkpoint_dir,
                payload=best_state.checkpoint_payload,
                best_epoch=best_state.epoch,
                best_map=best_state.map_score,
                name_prefix="global_best",
                sequences=sequence,
                metric_key=best_state.metric_key,
            )
    global_best_checkpoint_path = global_best_checkpoint_paths[checkpoint_key]

    return global_best_checkpoint_path, global_best_checkpoint_paths
