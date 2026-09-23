import os
import re
import shutil
import copy
from dataclasses import dataclass
from datetime import datetime

import torch

from training.configuration import (
    normalize_train_sequence_half_ratio,
    normalize_train_sequence_half_selection,
)


EXPERIMENT_NAME = "object_detection"

CURRENT_CHECKPOINT_REQUIRED_FIELDS = (
    "model_state_dict",
    "config",
    "epoch",
)
CURRENT_CHECKPOINT_REQUIRED_CONFIG_FIELDS = (
    "model_type",
    "box_coordinate_mode",
    "loss_mode",
    "num_classes",
    "include_bus_as_target",
    "class_names",
    "class_to_idx",
    "model7_decoder_hidden_channels",
    "max_detections",
    "train_scope",
    "cartesian_gt_root",
    "split_mode",
    "split_dir",
    "train_sequences",
    "val_sequences",
    "train_sequence_half_selection",
    "train_sequence_half_ratio",
    "seed",
    "domain_shift_experiment_enabled",
    "domain_shift_train_branch",
    "shared_train_sequences",
    "source_train_sequences",
    "target_train_sequences",
    "target_test_sequences",
    "train_control_split_enabled",
    "train_control_split_dir",
    "controlled_split_base_dir",
    "control_window_position",
    "control_class_names",
    "control_range_m_bins",
    "control_num_trials",
    "control_total_bbox_tolerance_ratio",
)


def validate_current_checkpoint(checkpoint):
    """Validate and return the canonical config stored by the current saver."""
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must be a dictionary produced by the current repository."
        )

    for field in CURRENT_CHECKPOINT_REQUIRED_FIELDS:
        if field not in checkpoint or checkpoint[field] is None:
            raise ValueError(
                f"Checkpoint is missing required current metadata: {field}"
            )

    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing required current metadata: config")
    for field in CURRENT_CHECKPOINT_REQUIRED_CONFIG_FIELDS:
        if field not in config:
            raise ValueError(
                "Checkpoint is missing required current metadata: "
                f"config.{field}"
            )

    for field in (
        "model_type",
        "box_coordinate_mode",
        "loss_mode",
        "num_classes",
        "include_bus_as_target",
        "class_names",
        "class_to_idx",
    ):
        if config[field] is None or config[field] == "":
            raise ValueError(
                "Checkpoint is missing required current metadata: "
                f"config.{field}"
            )

    num_classes = int(config["num_classes"])
    try:
        class_ids = {int(class_id) for class_id in dict(config["class_names"])}
        mapped_ids = {
            int(class_id)
            for class_id in dict(config["class_to_idx"]).values()
        }
    except (TypeError, ValueError):
        raise ValueError(
            "Checkpoint has invalid current metadata: config.class_names/class_to_idx"
        ) from None
    expected_class_ids = set(range(num_classes))
    if class_ids != expected_class_ids or mapped_ids != expected_class_ids:
        raise ValueError(
            "Checkpoint has invalid current metadata: "
            "config.class_names/class_to_idx do not match config.num_classes"
        )

    if (
        bool(config["domain_shift_experiment_enabled"])
        or config["domain_shift_train_branch"] not in (None, "")
    ):
        if config["domain_shift_train_branch"] not in {"source", "target"}:
            raise ValueError(
                "Checkpoint has invalid current metadata: "
                "config.domain_shift_train_branch"
            )
        for field in (
            "domain_shift_train_branch",
            "shared_train_sequences",
            "source_train_sequences",
            "target_train_sequences",
            "target_test_sequences",
        ):
            if config[field] in (None, "", ()):
                raise ValueError(
                    "Checkpoint is missing required current metadata: "
                    f"config.{field}"
                )

    return config


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


def format_checkpoint_filename(
        name_prefix,
        epoch,
        saved_at,
    ):
    if name_prefix not in (None, "best", "global_best"):
        raise ValueError(
            "name_prefix must be None, 'best', or 'global_best' for the "
            "current compact checkpoint layout"
        )
    date_text = str(saved_at).replace("-", "").replace("/", "")
    if len(date_text) >= 8 and date_text[:8].isdigit():
        date_text = date_text[4:8]

    filename_parts = [date_text]
    if name_prefix is not None:
        filename_parts.append(str(name_prefix))
    filename_parts.append(f"epoch_{epoch:03d}")
    return "_".join(filename_parts) + ".pth"


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


def build_checkpoint_payload(
        model,
        optimizer,
        scheduler,
        args,
        default_sequence,
        dataset_sequences,
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
            "sequence": default_sequence,
            "sequences": dataset_sequences,
            "epochs": args.epochs,
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
            "heatmap_radius": args.heatmap_radius,
            "centerpoint_gwd_loss_weight": args.centerpoint_gwd_loss_weight,
            "quality_loss_weight": args.quality_loss_weight,
            "num_classes": args.num_classes,
            "model_type": args.model_type,
            "configured_model_type": args.configured_model_type,
            "cartesian_training_workflow": args.cartesian_training_workflow,
            "loss_mode": args.loss_mode,
            "model7_decoder_hidden_channels": args.model7_decoder_hidden_channels,
            "run_model_type": args.run_model_type,
            "class_names": args.class_names,
            "class_to_idx": args.class_to_idx,
            "include_bus_as_target": args.include_bus_as_target,
            "ignore_class_names": args.ignore_class_names,
            "ignore_mask_margin": args.ignore_mask_margin,
            "ignore_mask_expand_ratio": args.ignore_mask_expand_ratio,
            "gt_object_ignore_override_path": args.gt_object_ignore_override_path,
            "train_control_split_enabled": args.train_control_split_enabled,
            "train_control_split_dir": args.train_control_split_dir,
            "train_scope": args.train_scope,
            "box_coordinate_mode": args.box_coordinate_mode,
            "weather_group": getattr(args, "weather_group", None),
            "weather_group_source": getattr(
                args,
                "weather_group_source",
                None,
            ),
            "sequence_information_path": args.sequence_information_path,
            "test_sequence_weather": getattr(
                args,
                "test_sequence_weather",
                None,
            ),
            "cartesian_gt_root": args.cartesian_gt_root,
            "training_eval_enabled": args.training_eval_enabled,
            "training_eval_train_set_enabled": (
                args.training_eval_train_set_enabled
            ),
            "training_eval_best_metric_key": (
                args.training_eval_best_metric_key
                if args.training_eval_enabled
                else None
            ),
            "training_eval_official_enabled": args.training_eval_official_enabled,
            "training_eval_official_version": args.training_eval_official_version,
            "training_eval_iou_backend": args.training_eval_iou_backend,
            "training_eval_iou_mode": args.training_eval_iou_mode,
            "training_eval_detection_metrics_enabled": (
                args.training_eval_detection_metrics_enabled
            ),
            "training_eval_ap_score_thresh": args.training_eval_ap_score_thresh,
            "training_eval_score_thresh": args.training_eval_score_thresh,
            "training_eval_coco_style_enabled": (
                args.training_eval_coco_style_enabled
            ),
            "training_eval_nuscenes_style_enabled": (
                args.training_eval_nuscenes_style_enabled
            ),
            "split_mode": args.split_mode,
            "split_dir": args.split_dir,
            "domain_shift_experiment_enabled": args.domain_shift_experiment_enabled,
            "domain_shift_train_branch": args.domain_shift_train_branch,
            "shared_train_sequences": args.shared_train_sequences,
            "source_train_sequences": args.source_train_sequences,
            "target_train_sequences": args.target_train_sequences,
            "target_test_sequences": args.target_test_sequences,
            "train_sequences": args.train_sequences,
            "val_sequences": args.val_sequences,
            "train_sequence_half_selection": args.train_sequence_half_selection,
            "train_sequence_half_ratio": args.train_sequence_half_ratio,
            "controlled_split_base_dir": args.controlled_split_base_dir,
            "control_window_position": args.control_window_position,
            "control_class_names": args.control_class_names,
            "control_range_m_bins": args.control_range_m_bins,
            "control_num_trials": args.control_num_trials,
            "control_total_bbox_tolerance_ratio": (
                args.control_total_bbox_tolerance_ratio
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

    validate_current_checkpoint(payload)
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


def save_epoch_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        scheduler,
        args,
        default_sequence,
        dataset_sequences,
        epoch,
        train_metrics,
        val_metrics,
        f1,
        learning_rate,
        is_best
    ):
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = format_checkpoint_filename(
        name_prefix=None,
        epoch=epoch,
        saved_at=saved_at,
    )
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        default_sequence=default_sequence,
        dataset_sequences=dataset_sequences,
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
        name_prefix,
    ):
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_filename = format_checkpoint_filename(
        name_prefix=name_prefix,
        epoch=best_epoch,
        saved_at=saved_at,
    )
    best_checkpoint_path = os.path.join(checkpoint_dir, best_filename)
    shutil.copy2(source_checkpoint_path, best_checkpoint_path)
    return best_checkpoint_path


def save_named_checkpoint_payload(
        checkpoint_dir,
        payload,
        best_epoch,
        name_prefix,
    ):
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_filename = format_checkpoint_filename(
        name_prefix=name_prefix,
        epoch=best_epoch,
        saved_at=saved_at,
    )
    best_checkpoint_path = os.path.join(checkpoint_dir, best_filename)
    torch.save(payload, best_checkpoint_path)
    return best_checkpoint_path


def remove_named_checkpoints(checkpoint_dir, name_prefix):
    if not os.path.isdir(checkpoint_dir):
        return

    pattern = rf"^\d{{4}}_{re.escape(str(name_prefix))}_epoch_\d+\.pth$"
    for filename in os.listdir(checkpoint_dir):
        if re.fullmatch(pattern, filename):
            os.remove(os.path.join(checkpoint_dir, filename))


def save_replacing_named_checkpoint_copy(
        checkpoint_dir,
        source_checkpoint_path,
        best_epoch,
        name_prefix,
    ):
    remove_named_checkpoints(checkpoint_dir, name_prefix)
    return save_named_checkpoint_copy(
        checkpoint_dir=checkpoint_dir,
        source_checkpoint_path=source_checkpoint_path,
        best_epoch=best_epoch,
        name_prefix=name_prefix,
    )


def save_replacing_named_checkpoint_payload(
        checkpoint_dir,
        payload,
        best_epoch,
        name_prefix,
    ):
    remove_named_checkpoints(checkpoint_dir, name_prefix)
    return save_named_checkpoint_payload(
        checkpoint_dir=checkpoint_dir,
        payload=payload,
        best_epoch=best_epoch,
        name_prefix=name_prefix,
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
        eval_metrics,
        val_loss_metrics,
        training_eval_enabled=True,
        best_metric_key="auto",
        official_eval_enabled=False,
        official_eval_iou_mode="easy",
    ):
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
    metric_key: str | None = None
    metric_value: float = -1.0
    epoch: int = -1
    checkpoint_path: object = None
    checkpoint_payload: object = None
    global_best_path: object = None

    def update(
            self,
            epoch,
            val_metrics,
            checkpoint_path=None,
            checkpoint_payload=None,
            global_best_path=None
        ):
        metric_key, metric_value = selection_metric_value(val_metrics)
        self.metric_key = metric_key
        self.metric_value = metric_value
        self.epoch = epoch
        self.checkpoint_path = checkpoint_path
        self.checkpoint_payload = checkpoint_payload
        self.global_best_path = global_best_path

    def is_better(self, val_metrics):
        metric_key, metric_value = selection_metric_value(val_metrics)
        if self.epoch < 0:
            return True
        if metric_key != self.metric_key:
            raise ValueError(
                "Best-checkpoint selection metric changed from "
                f"{self.metric_key!r} to {metric_key!r}."
            )
        if metric_prefers_lower(metric_key):
            return metric_value < self.metric_value
        return metric_value > self.metric_value


def save_epoch_and_update_best_checkpoint(
        best_state,
        checkpoint_dir,
        model,
        optimizer,
        scheduler,
        args,
        default_sequence,
        dataset_sequences,
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
            default_sequence=default_sequence,
            dataset_sequences=dataset_sequences,
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
            default_sequence=default_sequence,
            dataset_sequences=dataset_sequences,
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
        if checkpoint_path is not None:
            global_best_path = save_replacing_named_checkpoint_copy(
                checkpoint_dir=checkpoint_dir,
                source_checkpoint_path=checkpoint_path,
                best_epoch=epoch,
                name_prefix="global_best",
            )
        else:
            global_best_path = save_replacing_named_checkpoint_payload(
                checkpoint_dir=checkpoint_dir,
                payload=checkpoint_payload,
                best_epoch=epoch,
                name_prefix="global_best",
            )

        best_state.update(
            epoch=epoch,
            val_metrics=val_metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=checkpoint_payload,
            global_best_path=global_best_path
        )

    return checkpoint_path


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
                name_prefix="global_best",
            )
        else:
            global_best_checkpoint_paths[sequence] = save_named_checkpoint_payload(
                checkpoint_dir=sequence_checkpoint_dir,
                payload=best_state.checkpoint_payload,
                best_epoch=best_state.epoch,
                name_prefix="global_best",
            )
    global_best_checkpoint_path = global_best_checkpoint_paths[checkpoint_key]

    return global_best_checkpoint_path, global_best_checkpoint_paths
