"""Resume-specific restoration and policies for the shared training workflow."""

import os
from pathlib import Path
import sys

import torch

from configs.coordinates import require_cartesian_data
from configs.training import RESUME_CONFIG
from training_utils.checkpoints import save_replacing_named_checkpoint_copy
from training_utils.runner import (
    build_training_args,
    create_training_checkpoint_directories,
    run_training,
    validate_training_args,
)
from training_utils.torch_load import load_torch_checkpoint


def validate_resume_args(args):
    """Apply shared validation plus resume-only epoch requirements."""
    validate_training_args(args)
    return _validate_resume_options(args)


def _validate_resume_options(args):
    if args.resume_checkpoint == "":
        raise ValueError(
            "Set RESUME_CONFIG['resume_checkpoint'] in "
            "configs/resume.py before running train_resume.py"
        )
    if args.end_epoch <= 0:
        raise ValueError("RESUME_CONFIG['end_epoch'] must be greater than 0")
    return args


def build_resume_args(resume_config=None):
    config = RESUME_CONFIG if resume_config is None else resume_config
    return _validate_resume_options(build_training_args(config))


def load_resume_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    device,
    load_optimizer=True,
    expected_model_type=None,
    expected_num_classes=None,
    expected_include_bus_as_target=None,
    expected_box_coordinate_mode=None,
    checkpoint_loader=load_torch_checkpoint,
):
    """Restore an interrupted run while preserving legacy compatibility checks."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = checkpoint_loader(checkpoint_path, map_location=device)
    checkpoint_config = (
        checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    )
    checkpoint_model_type = checkpoint_config.get("model_type")
    checkpoint_num_classes = checkpoint_config.get("num_classes")
    checkpoint_include_bus_as_target = checkpoint_config.get(
        "include_bus_as_target"
    )
    checkpoint_box_coordinate_mode = checkpoint_config.get(
        "box_coordinate_mode"
    )
    if checkpoint_box_coordinate_mode is not None:
        require_cartesian_data(checkpoint_box_coordinate_mode)

    if (
        expected_model_type is not None
        and checkpoint_model_type is not None
        and checkpoint_model_type != expected_model_type
    ):
        raise ValueError(
            f"Checkpoint model_type={checkpoint_model_type!r}, but "
            f"RESUME_CONFIG model_type={expected_model_type!r}"
        )
    if (
        expected_num_classes is not None
        and checkpoint_num_classes is not None
        and int(checkpoint_num_classes) != int(expected_num_classes)
    ):
        raise ValueError(
            f"Checkpoint num_classes={checkpoint_num_classes}, but current "
            f"config expects num_classes={expected_num_classes}. Use "
            "init_from_checkpoint for 2-class -> 1-class initialization "
            "instead of train_resume.py."
        )
    if (
        expected_include_bus_as_target is not None
        and checkpoint_include_bus_as_target is not None
        and bool(checkpoint_include_bus_as_target)
        != bool(expected_include_bus_as_target)
    ):
        raise ValueError(
            "Checkpoint include_bus_as_target does not match RESUME_CONFIG "
            f"({checkpoint_include_bus_as_target} vs "
            f"{expected_include_bus_as_target})."
        )
    if (
        expected_box_coordinate_mode is not None
        and checkpoint_box_coordinate_mode is not None
        and str(checkpoint_box_coordinate_mode)
        != str(expected_box_coordinate_mode)
    ):
        raise ValueError(
            "Checkpoint box_coordinate_mode does not match RESUME_CONFIG "
            f"({checkpoint_box_coordinate_mode!r} vs "
            f"{expected_box_coordinate_mode!r})."
        )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model_for_state_dict = (
        model.module if isinstance(model, torch.nn.DataParallel) else model
    )
    model_for_state_dict.load_state_dict(state_dict)

    optimizer_loaded = False
    if (
        load_optimizer
        and isinstance(checkpoint, dict)
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        optimizer_loaded = True

    scheduler_loaded = False
    if (
        scheduler is not None
        and isinstance(checkpoint, dict)
        and checkpoint.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scheduler_loaded = True

    checkpoint_epoch = None
    if isinstance(checkpoint, dict) and checkpoint.get("epoch") is not None:
        checkpoint_epoch = int(checkpoint["epoch"])

    return (
        checkpoint_epoch,
        checkpoint_model_type,
        optimizer_loaded,
        scheduler_loaded,
        checkpoint_config,
    )


def resolve_resume_start_epoch(configured_start_epoch, checkpoint_epoch):
    """Resolve the first one-based resumed epoch."""
    if configured_start_epoch is not None:
        return int(configured_start_epoch)
    if checkpoint_epoch is None:
        raise ValueError(
            "Set RESUME_CONFIG['start_epoch']; checkpoint does not store an epoch."
        )
    return int(checkpoint_epoch) + 1


def restore_resume_training_state(*, args, model, optimizer, scheduler, device):
    """Restore state after optimizer construction and DataParallel wrapping."""
    (
        checkpoint_epoch,
        _checkpoint_model_type,
        optimizer_loaded,
        scheduler_loaded,
        _checkpoint_config,
    ) = load_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=args.resume_checkpoint,
        device=device,
        load_optimizer=args.load_optimizer,
        expected_model_type=args.model_type,
        expected_num_classes=args.num_classes,
        expected_include_bus_as_target=args.include_bus_as_target,
        expected_box_coordinate_mode=args.box_coordinate_mode,
    )
    args.start_epoch = resolve_resume_start_epoch(
        args.start_epoch,
        checkpoint_epoch,
    )
    if args.start_epoch > args.end_epoch:
        raise ValueError(
            f"start_epoch {args.start_epoch} is greater than "
            f"end_epoch {args.end_epoch}"
        )
    print(
        f"Loaded checkpoint epoch={checkpoint_epoch}, "
        f"optimizer_loaded={optimizer_loaded}, scheduler_loaded={scheduler_loaded}"
    )
    print(f"Resume training epochs: {args.start_epoch}-{args.end_epoch}")
    return args.start_epoch


def resolve_resume_checkpoint_directories(args, configured_sequences):
    """Reuse the interrupted run directory or create the standard layout."""
    if not getattr(args, "resume_save_in_checkpoint_dir", False):
        return create_training_checkpoint_directories(args, configured_sequences)

    checkpoint_dir = str(
        Path(args.resume_checkpoint).expanduser().resolve().parent
    )
    checkpoint_key = configured_sequences
    checkpoint_dirs = {checkpoint_key: checkpoint_dir}
    existing_epoch_files = []
    for epoch_number in range(args.start_epoch, args.end_epoch + 1):
        existing_epoch_files.extend(
            Path(checkpoint_dir).glob(f"*epoch_{epoch_number:03d}*.pth")
        )
    if existing_epoch_files:
        existing_text = ", ".join(
            str(path) for path in sorted(existing_epoch_files)
        )
        raise FileExistsError(
            "Refusing to overwrite existing resumed epoch checkpoints: "
            f"{existing_text}"
        )
    return checkpoint_dirs, checkpoint_key, checkpoint_dir


def initialize_best_state(best_state, initial_best_checkpoint, checkpoint_dir):
    """Restore best-checkpoint metadata and copy its stable named artifact."""
    if initial_best_checkpoint in (None, ""):
        return None
    if not os.path.exists(initial_best_checkpoint):
        raise FileNotFoundError(
            f"Initial best checkpoint not found: {initial_best_checkpoint}"
        )

    checkpoint = load_torch_checkpoint(initial_best_checkpoint, map_location="cpu")
    best_epoch = int(checkpoint.get("epoch", 0))
    best_metric_key = checkpoint.get(
        "selection_metric_key",
        checkpoint.get("val_metrics", {}).get("selection_metric_key", "mAP"),
    )
    best_map = float(
        checkpoint.get(
            "selection_metric_value",
            checkpoint.get("val_metrics", {}).get(
                "selection_metric_value",
                checkpoint.get(
                    "mAP",
                    checkpoint.get("val_metrics", {}).get("mAP", -1.0),
                ),
            ),
        )
    )
    copied_path = save_replacing_named_checkpoint_copy(
        checkpoint_dir=checkpoint_dir,
        source_checkpoint_path=initial_best_checkpoint,
        best_epoch=best_epoch,
        best_map=best_map,
        name_prefix="global_best",
    )
    best_state.map_score = best_map
    best_state.metric_key = best_metric_key
    best_state.epoch = best_epoch
    best_state.global_best_path = copied_path
    return copied_path


def main(resume_config=None):
    if resume_config is None and len(sys.argv) > 1:
        raise ValueError(
            "train_resume.py reads the merged RESUME_CONFIG exported from "
            "configs/training.py. Edit overrides in configs/resume.py, then "
            "run: python train_resume.py"
        )

    args = build_resume_args(resume_config=resume_config)
    args.epochs = args.end_epoch
    print(f"Resume checkpoint: {args.resume_checkpoint}")
    if getattr(args, "training_eval_enabled", True):
        print(f"Initial best checkpoint: {args.initial_best_checkpoint}")
    elif args.initial_best_checkpoint:
        print("Initial best checkpoint: ignored because best selection is disabled")

    def initialize_resume_best_state(best_state, checkpoint_dir):
        initial_best_path = initialize_best_state(
            best_state=best_state,
            initial_best_checkpoint=args.initial_best_checkpoint,
            checkpoint_dir=checkpoint_dir,
        )
        if initial_best_path is not None:
            print(f"Copied initial global best to: {initial_best_path}")

    return run_training(
        args,
        restore_training_state=restore_resume_training_state,
        resolve_checkpoint_directories=resolve_resume_checkpoint_directories,
        existing_tensorboard_log_dir=getattr(
            args, "resume_tensorboard_log_dir", None
        ),
        initialize_best_state_callback=initialize_resume_best_state,
        # Legacy resume omitted this keyword and therefore used evaluator default.
        include_detection_metrics_setting=False,
        print_checkpoint_directory=True,
        print_saved_checkpoints=True,
        print_global_best=True,
    )


__all__ = [
    "build_resume_args",
    "initialize_best_state",
    "load_resume_checkpoint",
    "main",
    "resolve_resume_checkpoint_directories",
    "resolve_resume_start_epoch",
    "restore_resume_training_state",
    "validate_resume_args",
]
