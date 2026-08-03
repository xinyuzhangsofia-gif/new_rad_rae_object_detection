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
from training_utils.checkpoints import (
    create_checkpoint_run_dirs,
    EXPERIMENT_NAME,
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
from training_utils.post_training_evaluation import (
    run_post_training_evaluation,
)
from training_utils.experiment_queue import (
    run_domain_shift_experiment_queue,
)
from training_utils.training_loop import (
    train_one_epoch,
    validate_loss,
)
from train_cfg import TRAIN_CONFIG
from train_mode_utils import (
    apply_domain_shift_training_configuration,
    apply_test_sequence_weather_configuration,
    apply_training_coordinate_mode,
    apply_task_configuration,
    apply_model15_lr_defaults,
    build_model15_lr_scheduler,
    initialize_model_from_checkpoint,
    prepare_controlled_train_data,
    resolve_loss_mode,
    resolve_run_model_type,
    model_uses_separate_quality_loss,
    use_official_model15_lr_mode,
)
from zxy_config import DataConfig


def build_train_args(train_config=None):
    config = TRAIN_CONFIG if train_config is None else train_config
    args = SimpleNamespace(**config)
    args = apply_domain_shift_training_configuration(args)
    args = apply_test_sequence_weather_configuration(args)
    if args.train_scope not in SCOPE_CHOICES:
        raise ValueError(f"train_scope must be one of {SCOPE_CHOICES}, got {args.train_scope!r}")
    if args.split_mode not in ("random", "file", "sequence", "sequence_tail"):
        raise ValueError(
            "split_mode must be 'random', 'file', 'sequence', or "
            f"'sequence_tail', got {args.split_mode!r}"
        )
    if args.model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown or unsupported model_type: {args.model_type}")
    return args


def main(train_config=None, _experiment_queue_child=False):
    if train_config is None and len(sys.argv) > 1:
        raise ValueError(
            "Training reads settings from train_cfg.py or train_cfg_v2.py; "
            "edit the selected config file instead of passing command-line arguments."
        )

    config = TRAIN_CONFIG if train_config is None else train_config
    if (
        not _experiment_queue_child
        and bool(config.get("experiment_queue_enabled", False))
    ):
        return run_domain_shift_experiment_queue(
            base_config=config,
            train_function=lambda child_config: main(
                child_config,
                _experiment_queue_child=True,
            ),
        )

    args = build_train_args(train_config=train_config)
    args = apply_training_coordinate_mode(args)
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
    print(
        "Ignore object_label=-1: "
        f"{args.ignore_object_label_minus_one}"
    )
    print(
        "Ignore out-of-scope GT: "
        f"{args.ignore_out_of_scope_gt}"
    )
    print(f"Ignore-mask classes: {args.ignore_class_names}")
    print(
        f"Ignore-mask region: GT box * {args.ignore_mask_expand_ratio} + margin {args.ignore_mask_margin}"
    )
    loss_mode = resolve_loss_mode(
        args.model_type,
        box_coordinate_mode=args.box_coordinate_mode,
        loss_mode=args.loss_mode,
    )
    print(f"Loss mode: {loss_mode} (configured: {args.loss_mode})")
    if loss_mode == "centerpoint":
        print(
            "CenterPoint box loss: offset + height + size + yaw + "
            f"{args.centerpoint_gwd_loss_weight:g} * GWD"
        )
        if model_uses_separate_quality_loss(args.model_type):
            print(f"Separate quality loss: enabled, weight={args.quality_loss_weight:g}")
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
    if args.split_mode == "sequence_tail":
        print(
            "Chronological internal validation: "
            f"last {args.sequence_tail_val_ratio:.1%} per train sequence, "
            f"drop {args.sequence_tail_boundary_drop_frames} boundary frame(s)"
        )
    if args.gt_object_ignore_override_path is not None:
        print(f"GT object ignore override: {args.gt_object_ignore_override_path}")
    if use_official_model15_lr_mode(args.model_type):
        print("Model15 LR mode: official RADE-Net CosineAnnealingLR")

    device, gpu_ids = select_device_and_gpus(args.gpu_ids)

    (train_dataset, val_dataset, train_loader, val_loader) = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        train_sequence_half_selection=getattr(
            args,
            "train_sequence_half_selection",
            None,
        ),
        train_sequence_half_ratio=getattr(
            args,
            "train_sequence_half_ratio",
            0.5,
        ),
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
        sequence_tail_val_ratio=getattr(args, "sequence_tail_val_ratio", 0.1),
        sequence_tail_boundary_drop_frames=getattr(
            args,
            "sequence_tail_boundary_drop_frames",
            0,
        ),
        box_coordinate_mode=args.box_coordinate_mode,
        cartesian_gt_root=args.cartesian_gt_root,
        polar_gt_root=args.polar_gt_root,
        ignore_object_label_minus_one=args.ignore_object_label_minus_one,
        ignore_out_of_scope_gt=args.ignore_out_of_scope_gt,
    )
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty.")

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
    initialize_model_from_checkpoint(
        model=model,
        checkpoint_path=getattr(args, "init_from_checkpoint", ""),
        map_location=device,
        include_bus_as_target=args.include_bus_as_target,
    )

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=gpu_ids,
            output_device=gpu_ids[0]
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
    history = []
    best_state = BestCheckpointState()

    checkpoint_dirs = create_checkpoint_run_dirs(
        base_dir=args.checkpoint_base_dir,
        experiment_name=EXPERIMENT_NAME,
        sequences=configured_sequences,
        model_type=args.run_model_type,
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
        checkpoint_layout=getattr(args, "checkpoint_layout", "legacy"),
        weather_group=getattr(args, "weather_group", None),
        train_sequences=args.train_sequences,
        test_sequences=args.val_sequences,
    )
    checkpoint_key = next(iter(checkpoint_dirs))
    checkpoint_dir = checkpoint_dirs[checkpoint_key]

    writer = create_tensorboard_writer(
        base_dir=args.log_base_dir,
        experiment_name=EXPERIMENT_NAME,
        sequence=configured_sequences,
        model_type=args.run_model_type,
    )
    
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
            args,
            "domain_shift_train_branch",
            None,
        ),
        shared_train_sequences=getattr(args, "shared_train_sequences", None),
        source_train_sequences=getattr(args, "source_train_sequences", None),
        target_train_sequences=getattr(args, "target_train_sequences", None),
        target_test_sequences=getattr(args, "target_test_sequences", None),
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
        training_eval_enabled=getattr(args, "training_eval_enabled", True),
        best_metric_key=(
            getattr(args, "best_metric_key", "auto")
            if getattr(args, "training_eval_enabled", True)
            else None
        ),
        official_eval_enabled=getattr(args, "official_eval_enabled", False),
        official_eval_version=getattr(args, "official_eval_version", "revised"),
        official_eval_iou_backend=getattr(args, "official_eval_iou_backend", "auto"),
        official_eval_iou_mode=getattr(args, "official_eval_iou_mode", "easy"),
        polar_eval_enabled=getattr(args, "polar_eval_enabled", False),
        polar_iou_thresholds=getattr(args, "polar_iou_thresholds", None),
        coco_style_eval_enabled=getattr(args, "coco_style_eval_enabled", False),
        nuscenes_style_eval_enabled=getattr(args, "nuscenes_style_eval_enabled", False),
        gt_object_ignore_override_path=getattr(args, "gt_object_ignore_override_path", None),
        train_control_split_enabled=getattr(args, "train_control_split_enabled", False),
        train_control_split_dir=getattr(args, "train_control_split_dir", None),
        sequence_tail_val_ratio=getattr(args, "sequence_tail_val_ratio", None),
        sequence_tail_boundary_drop_frames=getattr(
            args,
            "sequence_tail_boundary_drop_frames",
            None,
        ),
        centerpoint_gwd_loss_weight=args.centerpoint_gwd_loss_weight,
        quality_loss_weight=args.quality_loss_weight,
        quality_loss_active=model_uses_separate_quality_loss(args.model_type),
        model7_decoder_hidden_channels=args.model7_decoder_hidden_channels,
        box_coordinate_mode=args.box_coordinate_mode,
        cartesian_gt_root=args.cartesian_gt_root,
        weather_group=getattr(args, "weather_group", None),
        weather_group_source=getattr(args, "weather_group_source", None),
        sequence_information_path=getattr(
            args,
            "sequence_information_path",
            None,
        ),
        test_sequence_weather=getattr(args, "test_sequence_weather", None),
    )

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            num_epochs=args.epochs,
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
                polar_eval_enabled=getattr(args, "polar_eval_enabled", False),
                polar_iou_thresholds=getattr(
                    args,
                    "polar_iou_thresholds",
                    (0.3, 0.5),
                ),
                box_coordinate_mode=args.box_coordinate_mode,
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
        print_epoch_evaluation_summary(epoch=epoch + 1, val_metrics=val_metrics, f1=f1)

        learning_rate = optimizer.param_groups[0]["lr"]
        write_tensorboard_metrics(
            writer=writer, epoch=epoch + 1, train_metrics=train_metrics,
            val_metrics=val_metrics, f1=f1, learning_rate=learning_rate
        )

        save_epoch_and_update_best_checkpoint(
            best_state=best_state, checkpoint_dir=checkpoint_dir, model=model,
            optimizer=optimizer, scheduler=scheduler, args=args, cfg=cfg, epoch=epoch + 1,
            train_metrics=train_metrics, val_metrics=val_metrics, f1=f1,
            learning_rate=learning_rate,
            total_epochs=args.epochs,
            checkpoint_epoch_step=args.checkpoint_epoch_step,
            best_selection_enabled=getattr(
                args,
                "training_eval_enabled",
                True,
            ),
        )
        
        append_training_history(
            history=history, epoch=epoch + 1, train_metrics=train_metrics,
            val_metrics=val_metrics, f1=f1
        )
        
    writer.close()
    if getattr(args, "training_eval_enabled", True):
        save_global_best_checkpoint(
            best_state=best_state,
            checkpoint_dirs=checkpoint_dirs,
            checkpoint_key=checkpoint_key,
        )
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

if __name__ == "__main__":
    main()
