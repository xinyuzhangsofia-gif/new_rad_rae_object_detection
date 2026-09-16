"""Shared training workflow and normal-training orchestration.

The root ``train.py`` module remains a compatibility facade. Resume training
injects checkpoint restoration and run-directory policies into ``run_training``
instead of maintaining a second epoch pipeline.
"""

import sys
from types import SimpleNamespace

import torch

from configs.data import DataConfig
from configs.training import TRAIN_CONFIG
from data.coordinates import SCOPE_CHOICES
from data.dataloader import (
    build_train_val_dataloaders,
    get_dataset_sequences_for_split,
    prepare_model_inputs,
)
from eval.evaluation_config import resolve_official_eval_class_name_map
from eval.metrics_runner import evaluate_train_val_iou
from models import MODEL_TYPES, build_model
from training.checkpoints import (
    BestCheckpointState,
    build_epoch_eval_metrics,
    checkpoint_run_relative_path,
    create_checkpoint_run_dirs,
    EXPERIMENT_NAME,
    save_epoch_and_update_best_checkpoint,
    save_global_best_checkpoint,
)
from training.configuration import (
    apply_domain_shift_training_configuration,
    apply_model15_lr_defaults,
    apply_task_configuration,
    apply_test_sequence_weather_configuration,
    apply_training_coordinate_mode,
    build_model15_lr_scheduler,
    model_uses_separate_quality_loss,
    prepare_controlled_train_data,
    resolve_loss_mode,
    resolve_run_model_type,
    use_official_model15_lr_mode,
    validate_training_split_mode,
)
from training.experiments.queue import run_domain_shift_experiment_queue
from training.logging_utils import (
    create_tensorboard_writer,
    print_epoch_evaluation_summary,
    write_tensorboard_metrics,
    write_tensorboard_run_config,
)
from training.post_training_evaluation import run_post_training_evaluation
from training.runtime import select_device_and_gpus, set_seed
from training.loop import train_one_epoch, validate_loss


def validate_training_args(args):
    """Validate settings shared by normal and resume training."""
    if args.train_scope not in SCOPE_CHOICES:
        raise ValueError(
            f"train_scope must be one of {SCOPE_CHOICES}, "
            f"got {args.train_scope!r}"
        )
    validate_training_split_mode(args.split_mode)
    if args.model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown or unsupported model_type: {args.model_type}")
    return args


def build_training_args(config):
    """Resolve and validate the common flat training configuration."""
    args = SimpleNamespace(**config)
    args = apply_domain_shift_training_configuration(args)
    args = apply_test_sequence_weather_configuration(args)
    return validate_training_args(args)


def build_train_args(train_config=None):
    config = TRAIN_CONFIG if train_config is None else train_config
    return build_training_args(config)


def prepare_training_configuration(args):
    """Apply common pre-runtime configuration checks."""
    args = apply_training_coordinate_mode(args)
    if args.checkpoint_epoch_step <= 0:
        raise ValueError("checkpoint_epoch_step must be greater than 0")
    if (
        getattr(args, "training_eval_enabled", True)
        and getattr(args, "training_eval_official_enabled", False)
        and getattr(args, "training_eval_iou_backend", "auto") == "auto"
    ):
        args.training_eval_iou_backend = "cpu"

    return args


def prepare_training_task_configuration(args):
    """Derive class, controlled-split, scheduler, and model run settings."""
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
    return args


def print_training_configuration(args, loss_mode):
    """Print the common effective configuration for either workflow."""
    print(f"Training classes: {args.class_names}")
    print(
        "Test weather group: "
        f"{getattr(args, 'weather_group', 'unspecified')} "
        f"(auto from {getattr(args, 'sequence_information_path', 'unknown')})"
    )
    if getattr(args, "domain_shift_experiment_enabled", False):
        print(
            "Domain-shift training: "
            f"branch={args.domain_shift_train_branch}, "
            f"shared={args.shared_train_sequences}, "
            f"source={args.source_train_sequences}, "
            f"target={args.target_train_sequences}, "
            f"target_test={args.target_test_sequences}"
        )
        print(
            "Domain-shift effective split: "
            f"train={args.train_sequences}, val={args.val_sequences}"
        )
    print(f"Box coordinate mode: {args.box_coordinate_mode}")
    if args.model_type == "model7":
        print(
            "Model7 decoder hidden channels: "
            f"{args.model7_decoder_hidden_channels}"
        )
    if args.cartesian_training_workflow == "radenet_official_in_model7":
        print(
            "Model7 Cartesian detector: original RADE-Net head implemented "
            "directly inside model7"
        )
    elif args.cartesian_training_workflow == "centerpoint_cartesian_in_model7":
        print(
            "Model7 Cartesian detector: CenterPoint decoder with metric "
            "Cartesian box regression"
        )
    print(
        "Training evaluator: "
        + (
            "official Cartesian rotated BEV"
            if args.box_coordinate_mode == "cartesian"
            else "Polar R-A axis-aligned BEV"
        )
    )
    print(f"Bus target enabled: {args.include_bus_as_target}")
    print(f"Ignore object_label=-1: {args.ignore_object_label_minus_one}")
    print(f"Ignore out-of-scope GT: {args.ignore_out_of_scope_gt}")
    print(f"Ignore-mask classes: {args.ignore_class_names}")
    print(
        f"Ignore-mask region: GT box * {args.ignore_mask_expand_ratio} "
        f"+ margin {args.ignore_mask_margin}"
    )
    print(f"Loss mode: {loss_mode} (configured: {args.loss_mode})")
    if loss_mode == "centerpoint":
        print(
            "CenterPoint box loss: offset + height + size + yaw + "
            f"{args.centerpoint_gwd_loss_weight:g} * GWD"
        )
        if model_uses_separate_quality_loss(args.model_type):
            print(
                "Separate quality loss: enabled, "
                f"weight={args.quality_loss_weight:g}"
            )
        else:
            print("Separate quality loss: inactive for this model")
    if not getattr(args, "training_eval_enabled", True):
        print("Best checkpoint selection: disabled (epoch checkpoints only)")
    if args.train_control_split_enabled:
        print(f"Train control split: {args.train_control_split_dir}")
    if getattr(args, "train_sequence_half_selection", {}):
        print(
            "Train sequence half selection: "
            f"{args.train_sequence_half_selection}, "
            f"ratio={args.train_sequence_half_ratio:g}"
        )
    if args.gt_object_ignore_override_path is not None:
        print(f"GT object ignore override: {args.gt_object_ignore_override_path}")
    if use_official_model15_lr_mode(args.model_type):
        print("Model15 LR mode: official RADE-Net CosineAnnealingLR")


def build_training_data(args, cfg):
    """Build the datasets and loaders used by both training workflows."""
    result = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        train_sequence_half_selection=getattr(
            args, "train_sequence_half_selection", None
        ),
        train_sequence_half_ratio=getattr(
            args, "train_sequence_half_ratio", 0.5
        ),
        class_to_idx=args.class_to_idx,
        ignore_class_names=args.ignore_class_names,
        gt_object_ignore_override_path=args.gt_object_ignore_override_path,
        split_mode=args.split_mode,
        split_dir=getattr(
            args,
            "split_dir",
            "data/manifests/kradar",
        ),
        scope_mode=args.train_scope,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
        train_control_split_enabled=args.train_control_split_enabled,
        train_control_split_dir=args.train_control_split_dir,
        box_coordinate_mode=args.box_coordinate_mode,
        cartesian_gt_root=args.cartesian_gt_root,
        ignore_object_label_minus_one=args.ignore_object_label_minus_one,
        ignore_out_of_scope_gt=args.ignore_out_of_scope_gt,
    )
    if len(result[1]) == 0:
        raise ValueError("Validation split is empty.")
    return result


def build_training_components(
    args,
    device,
    gpu_ids,
    train_dataset,
    loss_mode,
):
    """Build model, DataParallel wrapper, optimizer, and scheduler in order."""
    model = build_model(
        model_type=args.model_type,
        device=device,
        num_classes=args.num_classes,
        decoder_hidden_channels=(
            args.model7_decoder_hidden_channels
            if args.model_type == "model7"
            else None
        ),
        box_coordinate_mode=args.box_coordinate_mode,
        loss_mode=loss_mode,
    )
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=gpu_ids,
            output_device=gpu_ids[0],
        )
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_model15_lr_scheduler(
        args=args,
        optimizer=optimizer,
        num_train_samples=len(train_dataset),
    )
    return model, optimizer, scheduler


def create_training_checkpoint_directories(args, configured_sequences):
    """Create the standard checkpoint directory layout."""
    checkpoint_dirs = create_checkpoint_run_dirs(
        base_dir=args.checkpoint_base_dir,
        experiment_name=EXPERIMENT_NAME,
        sequences=configured_sequences,
        model_type=args.run_model_type,
        train_sequence_half_selection=getattr(
            args, "train_sequence_half_selection", None
        ),
        train_sequence_half_ratio=getattr(
            args, "train_sequence_half_ratio", None
        ),
        domain_shift_experiment_enabled=getattr(
            args,
            "domain_shift_experiment_enabled",
            False,
        ),
        domain_shift_train_branch=getattr(
            args,
            "domain_shift_train_branch",
            None,
        ),
        weather_group=getattr(args, "weather_group", None),
        train_sequences=args.train_sequences,
        test_sequences=args.val_sequences,
    )
    checkpoint_key = next(iter(checkpoint_dirs))
    return checkpoint_dirs, checkpoint_key, checkpoint_dirs[checkpoint_key]


def write_training_run_config(
    writer,
    cfg,
    args,
    train_dataset,
    val_dataset,
    loss_mode,
):
    """Write the shared TensorBoard run metadata without changing tag names."""
    write_tensorboard_run_config(
        writer=writer,
        cfg=cfg,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        learning_rate=args.lr,
        max_detections=args.max_detections,
        num_classes=args.num_classes,
        class_names=args.class_names,
        model_type=args.run_model_type,
        configured_model_type=args.configured_model_type,
        cartesian_training_workflow=args.cartesian_training_workflow,
        loss_mode=loss_mode,
        train_scope=args.train_scope,
        split_mode=args.split_mode,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
        domain_shift_train_branch=getattr(
            args, "domain_shift_train_branch", None
        ),
        shared_train_sequences=getattr(args, "shared_train_sequences", None),
        source_train_sequences=getattr(args, "source_train_sequences", None),
        target_train_sequences=getattr(args, "target_train_sequences", None),
        target_test_sequences=getattr(args, "target_test_sequences", None),
        train_sequence_half_selection=getattr(
            args, "train_sequence_half_selection", None
        ),
        train_sequence_half_ratio=getattr(
            args, "train_sequence_half_ratio", None
        ),
        training_eval_enabled=getattr(args, "training_eval_enabled", True),
        training_eval_train_set_enabled=getattr(
            args,
            "training_eval_train_set_enabled",
            False,
        ),
        training_eval_best_metric_key=(
            getattr(args, "training_eval_best_metric_key", "auto")
            if getattr(args, "training_eval_enabled", True)
            else None
        ),
        training_eval_official_enabled=getattr(
            args, "training_eval_official_enabled", False
        ),
        training_eval_official_version=getattr(
            args, "training_eval_official_version", "revised"
        ),
        training_eval_iou_backend=getattr(
            args, "training_eval_iou_backend", "auto"
        ),
        training_eval_iou_mode=getattr(
            args, "training_eval_iou_mode", "easy"
        ),
        training_eval_detection_metrics_enabled=getattr(
            args, "training_eval_detection_metrics_enabled", False
        ),
        training_eval_ap_score_thresh=getattr(
            args, "training_eval_ap_score_thresh", 0.01
        ),
        training_eval_score_thresh=getattr(
            args, "training_eval_score_thresh", 0.3
        ),
        training_eval_coco_style_enabled=getattr(
            args, "training_eval_coco_style_enabled", False
        ),
        training_eval_nuscenes_style_enabled=getattr(
            args, "training_eval_nuscenes_style_enabled", False
        ),
        gt_object_ignore_override_path=getattr(
            args, "gt_object_ignore_override_path", None
        ),
        train_control_split_enabled=getattr(
            args, "train_control_split_enabled", False
        ),
        train_control_split_dir=getattr(args, "train_control_split_dir", None),
        centerpoint_gwd_loss_weight=args.centerpoint_gwd_loss_weight,
        quality_loss_weight=args.quality_loss_weight,
        quality_loss_active=model_uses_separate_quality_loss(args.model_type),
        model7_decoder_hidden_channels=args.model7_decoder_hidden_channels,
        box_coordinate_mode=args.box_coordinate_mode,
        cartesian_gt_root=args.cartesian_gt_root,
        weather_group=getattr(args, "weather_group", None),
        weather_group_source=getattr(args, "weather_group_source", None),
        sequence_information_path=getattr(
            args, "sequence_information_path", None
        ),
        test_sequence_weather=getattr(args, "test_sequence_weather", None),
    )


def _training_evaluation_kwargs(args, include_detection_metrics_setting):
    kwargs = {
        "num_classes": args.num_classes,
        "official_class_name_map": args.official_class_name_map,
        "prepare_model_inputs": prepare_model_inputs,
        "max_detections": args.max_detections,
        "scope_mode": args.train_scope,
        "evaluate_train": getattr(
            args,
            "training_eval_train_set_enabled",
            False,
        ),
        "official_eval_enabled": getattr(
            args, "training_eval_official_enabled", False
        ),
        "official_eval_version": getattr(
            args, "training_eval_official_version", "revised"
        ),
        "official_eval_iou_backend": getattr(
            args, "training_eval_iou_backend", "auto"
        ),
        "official_eval_iou_mode": getattr(
            args, "training_eval_iou_mode", "easy"
        ),
        "coco_style_eval_enabled": getattr(
            args, "training_eval_coco_style_enabled", False
        ),
        "nuscenes_style_eval_enabled": getattr(
            args, "training_eval_nuscenes_style_enabled", False
        ),
        "ap_score_thresh": getattr(
            args, "training_eval_ap_score_thresh", 0.01
        ),
        "score_thresh": getattr(
            args, "training_eval_score_thresh", 0.3
        ),
        "box_coordinate_mode": args.box_coordinate_mode,
    }
    if include_detection_metrics_setting:
        kwargs["official_detection_metrics_enabled"] = getattr(
            args,
            "training_eval_detection_metrics_enabled",
            True,
        )
    return kwargs


def run_training_epochs(
    *,
    args,
    cfg,
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    writer,
    best_state,
    checkpoint_dir,
    start_epoch,
    end_epoch,
    loss_mode,
    include_detection_metrics_setting=True,
    print_saved_checkpoints=False,
):
    """Run the common ordered train/validate/evaluate/save epoch sequence."""
    for epoch_number in range(start_epoch, end_epoch + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch_number - 1,
            num_epochs=end_epoch,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            heatmap_radius=args.heatmap_radius,
            centerpoint_gwd_loss_weight=args.centerpoint_gwd_loss_weight,
            quality_loss_weight=args.quality_loss_weight,
            ignore_mask_margin=args.ignore_mask_margin,
            ignore_mask_expand_ratio=args.ignore_mask_expand_ratio,
            loss_mode=loss_mode,
            num_classes=args.num_classes,
            box_coordinate_mode=args.box_coordinate_mode,
        )
        val_loss_metrics = validate_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            heatmap_radius=args.heatmap_radius,
            centerpoint_gwd_loss_weight=args.centerpoint_gwd_loss_weight,
            quality_loss_weight=args.quality_loss_weight,
            ignore_mask_margin=args.ignore_mask_margin,
            ignore_mask_expand_ratio=args.ignore_mask_expand_ratio,
            loss_mode=loss_mode,
            num_classes=args.num_classes,
            box_coordinate_mode=args.box_coordinate_mode,
        )

        eval_metrics = None
        if getattr(args, "training_eval_enabled", True):
            evaluation_kwargs = _training_evaluation_kwargs(
                args,
                include_detection_metrics_setting,
            )
            eval_metrics = evaluate_train_val_iou(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                device=device,
                **evaluation_kwargs,
            )

        val_metrics, f1 = build_epoch_eval_metrics(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            val_loss_metrics=val_loss_metrics,
            training_eval_enabled=getattr(args, "training_eval_enabled", True),
            best_metric_key=getattr(
                args, "training_eval_best_metric_key", "auto"
            ),
            official_eval_enabled=getattr(
                args, "training_eval_official_enabled", False
            ),
            official_eval_iou_mode=getattr(
                args, "training_eval_iou_mode", "easy"
            ),
        )
        print_epoch_evaluation_summary(
            epoch=epoch_number,
            val_metrics=val_metrics,
            f1=f1,
        )

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
            total_epochs=end_epoch,
            checkpoint_epoch_step=args.checkpoint_epoch_step,
            best_selection_enabled=getattr(args, "training_eval_enabled", True),
        )
        if print_saved_checkpoints and checkpoint_path is not None:
            print(f"Saved checkpoint: {checkpoint_path}")


def run_training(
    args,
    *,
    restore_training_state=None,
    resolve_checkpoint_directories=None,
    existing_tensorboard_log_dir=None,
    initialize_best_state_callback=None,
    include_detection_metrics_setting=True,
    print_checkpoint_directory=False,
    print_saved_checkpoints=False,
    print_global_best=False,
):
    """Run shared preparation, epoch execution, and finalization."""
    args = prepare_training_configuration(args)
    set_seed(args.seed)
    cfg = DataConfig()
    configured_sequences = get_dataset_sequences_for_split(
        cfg=cfg,
        split_mode=args.split_mode,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
    )
    args = prepare_training_task_configuration(args)
    loss_mode = resolve_loss_mode(
        args.model_type,
        box_coordinate_mode=args.box_coordinate_mode,
        loss_mode=args.loss_mode,
    )
    print_training_configuration(args, loss_mode)

    device, gpu_ids = select_device_and_gpus(args.gpu_ids)
    train_dataset, val_dataset, train_loader, val_loader = build_training_data(
        args,
        cfg,
    )
    model, optimizer, scheduler = build_training_components(
        args=args,
        device=device,
        gpu_ids=gpu_ids,
        train_dataset=train_dataset,
        loss_mode=loss_mode,
    )

    start_epoch = 1
    if restore_training_state is not None:
        start_epoch = restore_training_state(
            args=args,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
    end_epoch = args.epochs

    directory_resolver = (
        create_training_checkpoint_directories
        if resolve_checkpoint_directories is None
        else resolve_checkpoint_directories
    )
    checkpoint_dirs, checkpoint_key, checkpoint_dir = directory_resolver(
        args,
        configured_sequences,
    )
    if print_checkpoint_directory:
        print(f"Saving checkpoints to: {checkpoint_dir}")

    tensorboard_run_relative_path = None
    if existing_tensorboard_log_dir in (None, ""):
        tensorboard_run_relative_path = checkpoint_run_relative_path(
            checkpoint_dir,
            args.checkpoint_base_dir,
        )
    writer = create_tensorboard_writer(
        base_dir=args.log_base_dir,
        run_relative_path=tensorboard_run_relative_path,
        existing_log_dir=existing_tensorboard_log_dir,
    )
    write_training_run_config(
        writer=writer,
        cfg=cfg,
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_mode=loss_mode,
    )

    best_state = BestCheckpointState()
    if (
        getattr(args, "training_eval_enabled", True)
        and initialize_best_state_callback is not None
    ):
        initialize_best_state_callback(best_state, checkpoint_dir)

    run_training_epochs(
        args=args,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        writer=writer,
        best_state=best_state,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        loss_mode=loss_mode,
        include_detection_metrics_setting=include_detection_metrics_setting,
        print_saved_checkpoints=print_saved_checkpoints,
    )

    writer.close()
    if getattr(args, "training_eval_enabled", True):
        global_best_path, _ = save_global_best_checkpoint(
            best_state=best_state,
            checkpoint_dirs=checkpoint_dirs,
            checkpoint_key=checkpoint_key,
        )
        if print_global_best and global_best_path is not None:
            print(f"Current global best checkpoint: {global_best_path}")
    del model, optimizer, scheduler, train_loader, val_loader
    run_post_training_evaluation(
        checkpoint_root=checkpoint_dir,
        gpu_ids_text=args.gpu_ids,
        enabled=getattr(args, "post_training_eval_enabled", False),
        min_free_memory_mb=getattr(
            args,
            "post_training_eval_min_free_memory_mb",
            4096,
        ),
    )
    return checkpoint_dir


def main(train_config=None, _experiment_queue_child=False):
    if train_config is None and len(sys.argv) > 1:
        raise ValueError(
            "Training reads settings from configs/training.py; "
            "edit that configuration file instead of passing command-line arguments."
        )

    config = TRAIN_CONFIG if train_config is None else train_config
    if (
        not _experiment_queue_child
        and bool(config.get("experiment_queue_enabled", False))
    ):
        return run_domain_shift_experiment_queue(base_config=config)

    args = build_train_args(train_config=train_config)
    return run_training(args)


if __name__ == "__main__":
    main()
