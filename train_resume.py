from types import SimpleNamespace
import os
import sys

import torch

from dataloader import (
    build_train_val_dataloaders,
    get_dataset_sequences_for_split,
    prepare_model_inputs,
)
from dataset import CLASS_NAMES, CLASS_TO_IDX
from evaluation import evaluate_train_val_iou
from train import (
    build_model,
    NUM_CLASSES,
    select_device_and_gpus,
    train_one_epoch,
    validate_loss,
)
from train_cfg import RESUME_CONFIG
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
    if args.train_scope not in ("full", "narrow"):
        raise ValueError(f"train_scope must be 'full' or 'narrow', got {args.train_scope!r}")
    if args.split_mode not in ("random", "file", "sequence"):
        raise ValueError(
            f"split_mode must be 'random', 'file', or 'sequence', got {args.split_mode!r}"
        )
    if args.model_type not in {
            "model1", "model2", "model3", "model4", "model5", "model6",
            "model7", "model8", "model9", "model10", "model11", "model12",
            "model13", "model14", "model15",
    }:
        raise ValueError(f"Unknown or unsupported model_type: {args.model_type}")
    return args


def load_resume_checkpoint(model, optimizer, checkpoint_path, device, load_optimizer=True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
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

    checkpoint_epoch = None
    checkpoint_model_type = None
    if isinstance(checkpoint, dict):
        if checkpoint.get("epoch") is not None:
            checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_model_type = checkpoint.get("config", {}).get("model_type")

    return checkpoint_epoch, checkpoint_model_type, optimizer_loaded


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

    set_seed(args.seed)
    cfg = DataConfig()
    configured_sequences = get_dataset_sequences_for_split(
        cfg=cfg,
        split_mode=args.split_mode,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
    )
    args.epochs = args.end_epoch
    args.num_classes = NUM_CLASSES
    args.class_names = CLASS_NAMES.copy()
    args.class_to_idx = CLASS_TO_IDX.copy()

    print(f"Training classes: {CLASS_NAMES}")
    print(f"Resume checkpoint: {args.resume_checkpoint}")
    print(f"Initial best checkpoint: {args.initial_best_checkpoint}")

    device, gpu_ids = select_device_and_gpus(args.gpu_ids)
    train_dataset, val_dataset, train_loader, val_loader = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        class_to_idx=CLASS_TO_IDX,
        split_mode=args.split_mode,
        split_dir=getattr(args, "split_dir", "split"),
        scope_mode=args.train_scope,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
    )
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty.")

    model = build_model(model_type=args.model_type, device=device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_epoch, checkpoint_model_type, optimizer_loaded = load_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=args.resume_checkpoint,
        device=device,
        load_optimizer=args.load_optimizer,
    )
    if checkpoint_model_type is not None and checkpoint_model_type != args.model_type:
        raise ValueError(
            f"Checkpoint model_type={checkpoint_model_type!r}, but RESUME_CONFIG model_type={args.model_type!r}"
        )

    if args.start_epoch is None:
        if checkpoint_epoch is None:
            raise ValueError("Set RESUME_CONFIG['start_epoch']; checkpoint does not store an epoch.")
        args.start_epoch = checkpoint_epoch + 1
    if args.start_epoch > args.end_epoch:
        raise ValueError(f"start_epoch {args.start_epoch} is greater than end_epoch {args.end_epoch}")

    print(f"Loaded checkpoint epoch={checkpoint_epoch}, optimizer_loaded={optimizer_loaded}")
    print(f"Resume training epochs: {args.start_epoch}-{args.end_epoch}")

    checkpoint_dirs = create_checkpoint_run_dirs(
        base_dir=args.checkpoint_base_dir,
        experiment_name=EXPERIMENT_NAME,
        sequences=configured_sequences,
        model_type=args.model_type,
    )
    checkpoint_key = next(iter(checkpoint_dirs))
    checkpoint_dir = checkpoint_dirs[checkpoint_key]
    print(f"Saving checkpoints to: {checkpoint_dir}")

    writer = create_tensorboard_writer(
        base_dir=args.log_base_dir,
        experiment_name=EXPERIMENT_NAME,
        sequence=configured_sequences,
        model_type=args.model_type,
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
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        model_type=args.model_type,
        train_scope=args.train_scope,
        split_mode=args.split_mode,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
        best_metric_key=getattr(args, "best_metric_key", "auto"),
        official_eval_enabled=getattr(args, "official_eval_enabled", False),
        official_eval_version=getattr(args, "official_eval_version", "revised"),
        official_eval_iou_backend=getattr(args, "official_eval_iou_backend", "auto"),
        official_eval_iou_mode=getattr(args, "official_eval_iou_mode", "easy"),
        coco_style_eval_enabled=getattr(args, "coco_style_eval_enabled", False),
        nuscenes_style_eval_enabled=getattr(args, "nuscenes_style_eval_enabled", False),
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

    if args.model_type in {"model12", "model14"}:
        loss_mode = "yolox"
    elif args.model_type == "model15":
        loss_mode = "radenet"
    else:
        loss_mode = "centerpoint"
    for epoch_number in range(args.start_epoch, args.end_epoch + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch_number - 1,
            num_epochs=args.end_epoch,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            heatmap_radius=args.heatmap_radius,
            centerpoint_giou_loss_weight=args.centerpoint_giou_loss_weight,
            quality_loss_weight=args.quality_loss_weight,
            loss_mode=loss_mode,
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
            loss_mode=loss_mode,
        )
        eval_metrics = evaluate_train_val_iou(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            num_classes=NUM_CLASSES,
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
        )
        val_metrics, f1 = build_epoch_eval_metrics(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            val_loss_metrics=val_loss_metrics,
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
            print(f"Saved candidate checkpoint: {checkpoint_path}")

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
