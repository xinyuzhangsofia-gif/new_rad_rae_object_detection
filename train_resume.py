import os
import sys
from types import SimpleNamespace

import torch

from cfg_model import SCOPE_CHOICES
from dataloader import (
    build_train_val_dataloaders,
    get_dataset_sequences_for_split,
    prepare_model_inputs,
)
from evaluation import (
    evaluate_train_val_iou,
    resolve_official_eval_class_name_map,
)
from models import MODEL_TYPES, build_model
from train import (
    train_one_epoch,
    validate_loss,
)
from train_cfg import RESUME_CONFIG
from train_mode_utils import (
    apply_task_configuration,
    apply_model15_lr_defaults,
    build_model15_lr_scheduler,
    prepare_controlled_train_data,
    resolve_loss_mode,
    resolve_run_model_type,
    use_official_model15_lr_mode,
)
from training_utils.checkpoints import (
    create_checkpoint_run_dirs,
    EXPERIMENT_NAME,
    save_replacing_named_checkpoint_copy,
)
from training_utils.logging_utils import (
    create_tensorboard_writer,
    print_epoch_evaluation_summary,
    write_tensorboard_metrics,
    write_tensorboard_run_config,
)
from training_utils.other_helping_functions import (
    append_training_history,
    BestCheckpointState,
    build_epoch_eval_metrics,
    save_epoch_and_update_best_checkpoint,
    save_global_best_checkpoint,
    set_seed,
)
from training_utils.runtime import select_device_and_gpus
from training_utils.torch_load import load_torch_checkpoint
from zxy_config import DataConfig


def build_resume_args():
    args = SimpleNamespace(**RESUME_CONFIG)
    validate_resume_args(args)
    if args.resume_checkpoint == "":
        raise ValueError("Set RESUME_CONFIG['resume_checkpoint'] in train_cfg.py before running train_resume.py")
    if args.end_epoch <= 0:
        raise ValueError("RESUME_CONFIG['end_epoch'] must be greater than 0")
    return args


def validate_resume_args(args):
    if args.train_scope not in SCOPE_CHOICES:
        raise ValueError(f"train_scope must be one of {SCOPE_CHOICES}, got {args.train_scope!r}")
    if args.split_mode not in ("random", "file", "sequence"):
        raise ValueError(
            f"split_mode must be 'random', 'file', or 'sequence', got {args.split_mode!r}"
        )
    if args.model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown or unsupported model_type: {args.model_type}")
    return args


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
    ):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_model_type = checkpoint_config.get("model_type")
    checkpoint_num_classes = checkpoint_config.get("num_classes")
    checkpoint_include_bus_as_target = checkpoint_config.get("include_bus_as_target")

    if (
        expected_model_type is not None
        and checkpoint_model_type is not None
        and checkpoint_model_type != expected_model_type
    ):
        raise ValueError(
            f"Checkpoint model_type={checkpoint_model_type!r}, but RESUME_CONFIG model_type={expected_model_type!r}"
        )
    if (
        expected_num_classes is not None
        and checkpoint_num_classes is not None
        and int(checkpoint_num_classes) != int(expected_num_classes)
    ):
        raise ValueError(
            f"Checkpoint num_classes={checkpoint_num_classes}, but current config expects num_classes={expected_num_classes}. "
            "Use init_from_checkpoint for 2-class -> 1-class initialization instead of train_resume.py."
        )
    if (
        expected_include_bus_as_target is not None
        and checkpoint_include_bus_as_target is not None
        and bool(checkpoint_include_bus_as_target) != bool(expected_include_bus_as_target)
    ):
        raise ValueError(
            "Checkpoint include_bus_as_target does not match RESUME_CONFIG "
            f"({checkpoint_include_bus_as_target} vs {expected_include_bus_as_target})."
        )

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model_for_state_dict = model.module if isinstance(model, torch.nn.DataParallel) else model
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
    if isinstance(checkpoint, dict):
        if checkpoint.get("epoch") is not None:
            checkpoint_epoch = int(checkpoint["epoch"])

    return checkpoint_epoch, checkpoint_model_type, optimizer_loaded, scheduler_loaded, checkpoint_config


def initialize_best_state(best_state, initial_best_checkpoint, checkpoint_dir):
    if initial_best_checkpoint is None or initial_best_checkpoint == "":
        return None
    if not os.path.exists(initial_best_checkpoint):
        raise FileNotFoundError(f"Initial best checkpoint not found: {initial_best_checkpoint}")

    checkpoint = load_torch_checkpoint(initial_best_checkpoint, map_location="cpu")
    best_epoch = int(checkpoint.get("epoch", 0))
    best_metric_key = checkpoint.get(
        "selection_metric_key",
        checkpoint.get("val_metrics", {}).get("selection_metric_key", "mAP"),
    )
    best_map = float(
        checkpoint.get(
            "selection_metric_value",
            checkpoint.get(
                "val_metrics",
                {},
            ).get(
                "selection_metric_value",
                checkpoint.get("mAP", checkpoint.get("val_metrics", {}).get("mAP", -1.0)),
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


def main():
    if len(sys.argv) > 1:
        raise ValueError("train_resume.py reads settings from train_cfg.py. Edit RESUME_CONFIG, then run: python train_resume.py")

    args = build_resume_args()
    if args.checkpoint_epoch_step <= 0:
        raise ValueError("checkpoint_epoch_step must be greater than 0")
    if (
        getattr(args, "training_eval_enabled", True)
        and getattr(args, "official_eval_enabled", False)
        and getattr(args, "official_eval_iou_backend", "auto") == "auto"
    ):
        args.official_eval_iou_backend = "cpu"

    set_seed(args.seed)
    cfg = DataConfig()
    configured_sequences = get_dataset_sequences_for_split(
        cfg=cfg,
        split_mode=args.split_mode,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
    )
    args.epochs = args.end_epoch
    args = apply_task_configuration(args)
    args = prepare_controlled_train_data(args)
    args = apply_model15_lr_defaults(args)
    args.official_class_name_map, _ = resolve_official_eval_class_name_map(
        args.class_names
    )
    args.run_model_type = resolve_run_model_type(
        model_type=args.model_type,
        include_bus_as_target=args.include_bus_as_target,
    )

    print(f"Training classes: {args.class_names}")
    print(f"Bus target enabled: {args.include_bus_as_target}")
    print(f"Ignore-mask classes: {args.ignore_class_names}")
    print(
        f"Ignore-mask region: GT box * {args.ignore_mask_expand_ratio} + margin {args.ignore_mask_margin}"
    )
    if args.train_control_split_enabled:
        print(f"Train control split: {args.train_control_split_dir}")
    if args.gt_object_ignore_override_path is not None:
        print(f"GT object ignore override: {args.gt_object_ignore_override_path}")
    print(f"Resume checkpoint: {args.resume_checkpoint}")
    print(f"Initial best checkpoint: {args.initial_best_checkpoint}")
    if use_official_model15_lr_mode(args.model_type):
        print("Model15 LR mode: official RADE-Net CosineAnnealingLR")

    device, gpu_ids = select_device_and_gpus(args.gpu_ids)
    train_dataset, val_dataset, train_loader, val_loader = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        class_to_idx=args.class_to_idx,
        ignore_class_names=args.ignore_class_names,
        gt_object_ignore_override_path=args.gt_object_ignore_override_path,
        split_mode=args.split_mode,
        split_dir=getattr(args, "split_dir", "split"),
        scope_mode=args.train_scope,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
        train_control_split_enabled=args.train_control_split_enabled,
        train_control_split_dir=args.train_control_split_dir,
    )
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty.")

    model = build_model(
        model_type=args.model_type,
        device=device,
        num_classes=args.num_classes,
    )
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_model15_lr_scheduler(
        args=args,
        optimizer=optimizer,
        num_train_samples=len(train_dataset),
    )
    checkpoint_epoch, checkpoint_model_type, optimizer_loaded, scheduler_loaded, checkpoint_config = load_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=args.resume_checkpoint,
        device=device,
        load_optimizer=args.load_optimizer,
        expected_model_type=args.model_type,
        expected_num_classes=args.num_classes,
        expected_include_bus_as_target=args.include_bus_as_target,
    )

    if args.start_epoch is None:
        if checkpoint_epoch is None:
            raise ValueError("Set RESUME_CONFIG['start_epoch']; checkpoint does not store an epoch.")
        args.start_epoch = checkpoint_epoch + 1
    if args.start_epoch > args.end_epoch:
        raise ValueError(f"start_epoch {args.start_epoch} is greater than end_epoch {args.end_epoch}")

    print(
        f"Loaded checkpoint epoch={checkpoint_epoch}, "
        f"optimizer_loaded={optimizer_loaded}, scheduler_loaded={scheduler_loaded}"
    )
    print(f"Resume training epochs: {args.start_epoch}-{args.end_epoch}")

    checkpoint_dirs = create_checkpoint_run_dirs(
        base_dir=args.checkpoint_base_dir,
        experiment_name=EXPERIMENT_NAME,
        sequences=configured_sequences,
        model_type=args.run_model_type,
    )
    checkpoint_key = next(iter(checkpoint_dirs))
    checkpoint_dir = checkpoint_dirs[checkpoint_key]
    print(f"Saving checkpoints to: {checkpoint_dir}")

    writer = create_tensorboard_writer(
        base_dir=args.log_base_dir,
        experiment_name=EXPERIMENT_NAME,
        sequence=configured_sequences,
        model_type=args.run_model_type,
    )
    write_tensorboard_run_config(
        writer=writer,
        cfg=cfg,
        num_epochs=args.end_epoch,
        batch_size=args.batch_size,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        learning_rate=args.lr,
        max_detections=args.max_detections,
        num_classes=args.num_classes,
        class_names=args.class_names,
        model_type=args.run_model_type,
        train_scope=args.train_scope,
        split_mode=args.split_mode,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
        training_eval_enabled=getattr(args, "training_eval_enabled", True),
        best_metric_key=getattr(args, "best_metric_key", "auto"),
        official_eval_enabled=getattr(args, "official_eval_enabled", False),
        official_eval_version=getattr(args, "official_eval_version", "revised"),
        official_eval_iou_backend=getattr(args, "official_eval_iou_backend", "auto"),
        official_eval_iou_mode=getattr(args, "official_eval_iou_mode", "easy"),
        coco_style_eval_enabled=getattr(args, "coco_style_eval_enabled", False),
        nuscenes_style_eval_enabled=getattr(args, "nuscenes_style_eval_enabled", False),
        gt_object_ignore_override_path=getattr(args, "gt_object_ignore_override_path", None),
        train_control_split_enabled=getattr(args, "train_control_split_enabled", False),
        train_control_split_dir=getattr(args, "train_control_split_dir", None),
    )

    history = []
    best_state = BestCheckpointState()
    initial_best_path = initialize_best_state(
        best_state=best_state,
        initial_best_checkpoint=args.initial_best_checkpoint,
        checkpoint_dir=checkpoint_dir,
    )
    if initial_best_path is not None:
        print(f"Copied initial global best to: {initial_best_path}")

    loss_mode = resolve_loss_mode(args.model_type)
    for epoch_number in range(args.start_epoch, args.end_epoch + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch_number - 1,
            num_epochs=args.end_epoch,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            heatmap_radius=args.heatmap_radius,
            centerpoint_giou_loss_weight=args.centerpoint_giou_loss_weight,
            quality_loss_weight=args.quality_loss_weight,
            ignore_mask_margin=args.ignore_mask_margin,
            ignore_mask_expand_ratio=args.ignore_mask_expand_ratio,
            loss_mode=loss_mode,
            num_classes=args.num_classes,
        )
        val_loss_metrics = validate_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            heatmap_radius=args.heatmap_radius,
            centerpoint_giou_loss_weight=args.centerpoint_giou_loss_weight,
            quality_loss_weight=args.quality_loss_weight,
            ignore_mask_margin=args.ignore_mask_margin,
            ignore_mask_expand_ratio=args.ignore_mask_expand_ratio,
            loss_mode=loss_mode,
            num_classes=args.num_classes,
        )
        eval_metrics = None
        if getattr(args, "training_eval_enabled", True):
            eval_metrics = evaluate_train_val_iou(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                device=device,
                num_classes=args.num_classes,
                official_class_name_map=args.official_class_name_map,
                prepare_model_inputs=prepare_model_inputs,
                max_detections=args.max_detections,
                scope_mode=args.train_scope,
                evaluate_train=args.eval_train,
                official_eval_enabled=getattr(args, "official_eval_enabled", False),
                official_eval_version=getattr(args, "official_eval_version", "revised"),
                official_eval_iou_backend=getattr(args, "official_eval_iou_backend", "auto"),
                official_eval_iou_mode=getattr(args, "official_eval_iou_mode", "easy"),
                coco_style_eval_enabled=getattr(args, "coco_style_eval_enabled", False),
                nuscenes_style_eval_enabled=getattr(args, "nuscenes_style_eval_enabled", False),
                ap_score_thresh=getattr(args, "ap_score_thresh", 0.01),
                detection_score_thresh=getattr(args, "score_thresh", 0.3),
            )
        val_metrics, f1 = build_epoch_eval_metrics(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            val_loss_metrics=val_loss_metrics,
            training_eval_enabled=getattr(args, "training_eval_enabled", True),
            best_metric_key=getattr(args, "best_metric_key", "auto"),
            official_eval_enabled=getattr(args, "official_eval_enabled", False),
            official_eval_iou_mode=getattr(args, "official_eval_iou_mode", "easy"),
        )
        print_epoch_evaluation_summary(epoch=epoch_number, val_metrics=val_metrics, f1=f1)

        learning_rate = optimizer.param_groups[0]["lr"]
        write_tensorboard_metrics(
            writer=writer,
            epoch=epoch_number,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            learning_rate=learning_rate,
        )

        checkpoint_path = save_epoch_and_update_best_checkpoint(
            best_state=best_state,
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            cfg=cfg,
            epoch=epoch_number,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            learning_rate=learning_rate,
            total_epochs=args.end_epoch,
            checkpoint_epoch_step=args.checkpoint_epoch_step,
        )
        if checkpoint_path is not None:
            print(f"Saved checkpoint: {checkpoint_path}")

        append_training_history(
            history=history,
            epoch=epoch_number,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
        )

    writer.close()
    global_best_path, _ = save_global_best_checkpoint(
        best_state=best_state,
        checkpoint_dirs=checkpoint_dirs,
        checkpoint_key=checkpoint_key,
    )
    if global_best_path is not None:
        print(f"Current global best checkpoint: {global_best_path}")


if __name__ == "__main__":
    main()
