from types import SimpleNamespace
import sys
import torch
from cfg_model import SCOPE_CHOICES
from dataloader import (
    build_train_val_dataloaders,
    get_dataset_sequences_for_split,
    prepare_model_inputs,
)
from dataset import CLASS_NAMES, CLASS_TO_IDX
from evaluation import evaluate_train_val_iou
from models import *
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
from training_utils.runtime import NUM_CLASSES, select_device_and_gpus
from training_utils.training_loop import (
    train_one_epoch,
    validate_loss,
)
from train_cfg import TRAIN_CONFIG
from zxy_config import DataConfig


def build_train_args():
    args = SimpleNamespace(**TRAIN_CONFIG)
    if args.train_scope not in SCOPE_CHOICES:
        raise ValueError(f"train_scope must be one of {SCOPE_CHOICES}, got {args.train_scope!r}")
    if args.split_mode not in ("random", "file", "sequence"):
        raise ValueError(
            f"split_mode must be 'random', 'file', or 'sequence', got {args.split_mode!r}"
        )
    if args.model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown or unsupported model_type: {args.model_type}")
    return args


def main():
    if len(sys.argv) > 1:
        raise ValueError("train.py reads settings from train_cfg.py. Edit train_cfg.py, then run: python train.py")

    args = build_train_args()
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
    args.num_classes = NUM_CLASSES
    args.class_names = CLASS_NAMES.copy()
    args.class_to_idx = CLASS_TO_IDX.copy()

    print(f"Training classes: {CLASS_NAMES}")

    device, gpu_ids = select_device_and_gpus(args.gpu_ids)

    (train_dataset, val_dataset, train_loader, val_loader) = build_train_val_dataloaders(
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
        model = torch.nn.DataParallel(
            model,
            device_ids=gpu_ids,
            output_device=gpu_ids[0]
        )
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_state = BestCheckpointState()

    checkpoint_dirs = create_checkpoint_run_dirs(
        base_dir=args.checkpoint_base_dir,
        experiment_name=EXPERIMENT_NAME,
        sequences=configured_sequences,
        model_type=args.model_type
    )
    checkpoint_key = next(iter(checkpoint_dirs))
    checkpoint_dir = checkpoint_dirs[checkpoint_key]

    writer = create_tensorboard_writer(
        base_dir=args.log_base_dir,
        experiment_name=EXPERIMENT_NAME,
        sequence=configured_sequences,
        model_type=args.model_type
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

    for epoch in range(args.epochs):
        if args.model_type in {"model12", "model14"}:
            loss_mode = "yolox"
        elif args.model_type == "model15":
            loss_mode = "radenet"
        else:
            loss_mode = "centerpoint"
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=args.epochs,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            heatmap_radius=args.heatmap_radius,
            centerpoint_giou_loss_weight=args.centerpoint_giou_loss_weight,
            quality_loss_weight=args.quality_loss_weight,
            loss_mode=loss_mode,
            num_classes=NUM_CLASSES,
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
            num_classes=NUM_CLASSES,
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
        print_epoch_evaluation_summary(epoch=epoch + 1, val_metrics=val_metrics, f1=f1)

        learning_rate = optimizer.param_groups[0]["lr"]
        write_tensorboard_metrics(
            writer=writer, epoch=epoch + 1, train_metrics=train_metrics,
            val_metrics=val_metrics, f1=f1, learning_rate=learning_rate
        )

        save_epoch_and_update_best_checkpoint(
            best_state=best_state, checkpoint_dir=checkpoint_dir, model=model,
            optimizer=optimizer, args=args, cfg=cfg, epoch=epoch + 1,
            train_metrics=train_metrics, val_metrics=val_metrics, f1=f1,
            learning_rate=learning_rate,
            total_epochs=args.epochs,
            checkpoint_epoch_step=args.checkpoint_epoch_step
        )
        
        append_training_history(
            history=history, epoch=epoch + 1, train_metrics=train_metrics,
            val_metrics=val_metrics, f1=f1
        )
        
    writer.close()
    save_global_best_checkpoint(best_state=best_state, checkpoint_dirs=checkpoint_dirs, checkpoint_key=checkpoint_key)

if __name__ == "__main__":
    main()
