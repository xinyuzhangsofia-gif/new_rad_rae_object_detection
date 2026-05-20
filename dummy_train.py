import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dummy_dataloader import (
    build_train_val_dataloaders,
    get_config_sequences,
    prepare_model_inputs,
)
from dummy_evaluation import *
from dummy_module import MVRSS3DModel
from utils_dummy.checkpoints import (
    create_checkpoint_run_dir,
    create_checkpoint_run_dirs,
    save_best_checkpoint_copy,
    save_epoch_checkpoint,
    save_named_checkpoint_copy,
)
from utils_dummy.logging_utils import (
    create_tensorboard_writer,
    print_training_history,
    write_tensorboard_metrics,
    write_tensorboard_run_config,
)
from zxy_config import DataConfig

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def greedy_iou_match(pred_boxes, gt_boxes, iou_thresh=0.0):
    """
    pred_boxes: [num_queries, 7]
    gt_boxes:   [num_gt, 7]

    return:
        matched_pred_indices: LongTensor [num_matched]
        matched_gt_indices:   LongTensor [num_matched]
    """

    device = pred_boxes.device

    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )

    pred_ra_boxes = boxes_rae_to_ra_xyxy(pred_boxes)
    gt_ra_boxes = boxes_rae_to_ra_xyxy(gt_boxes)

    ious = box_iou_2d(pred_ra_boxes, gt_ra_boxes)
    flat_ious = ious.reshape(-1)
    order = flat_ious.argsort(descending=True)

    matched_pred = []
    matched_gt = []

    used_pred = set()
    used_gt = set()

    num_gt = gt_boxes.shape[0]

    for flat_idx in order:
        flat_idx = flat_idx.item()

        pred_idx = flat_idx // num_gt
        gt_idx = flat_idx % num_gt

        iou_value = ious[pred_idx, gt_idx].item()

        if iou_value < iou_thresh:
            break

        if pred_idx in used_pred:
            continue

        if gt_idx in used_gt:
            continue

        matched_pred.append(pred_idx)
        matched_gt.append(gt_idx)

        used_pred.add(pred_idx)
        used_gt.add(gt_idx)

        if len(used_gt) == num_gt:
            break

    matched_pred_indices = torch.tensor(
        matched_pred,
        dtype=torch.long,
        device=device
    )

    matched_gt_indices = torch.tensor(
        matched_gt,
        dtype=torch.long,
        device=device
    )

    return matched_pred_indices, matched_gt_indices


@torch.no_grad()
def greedy_cost_match(
        pred_boxes,
        gt_boxes,
        cost_bbox=1.0,
        cost_iou=1.0
    ):
    device = pred_boxes.device

    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )

    bbox_cost = torch.cdist(
        pred_boxes[:, :6],
        gt_boxes[:, :6],
        p=1
    )

    pred_ra_boxes = boxes_rae_to_ra_xyxy(pred_boxes)
    gt_ra_boxes = boxes_rae_to_ra_xyxy(gt_boxes)

    ious = box_iou_2d(pred_ra_boxes, gt_ra_boxes)
    iou_cost = 1.0 - ious

    total_cost = cost_bbox * bbox_cost + cost_iou * iou_cost

    flat_cost = total_cost.reshape(-1)
    order = flat_cost.argsort(descending=False)

    matched_pred = []
    matched_gt = []

    used_pred = set()
    used_gt = set()

    num_gt = gt_boxes.shape[0]

    for flat_idx in order:
        flat_idx = flat_idx.item()

        pred_idx = flat_idx // num_gt
        gt_idx = flat_idx % num_gt

        if pred_idx in used_pred:
            continue

        if gt_idx in used_gt:
            continue

        matched_pred.append(pred_idx)
        matched_gt.append(gt_idx)

        used_pred.add(pred_idx)
        used_gt.add(gt_idx)

        if len(used_gt) == num_gt:
            break

    matched_pred_indices = torch.tensor(
        matched_pred,
        dtype=torch.long,
        device=device
    )

    matched_gt_indices = torch.tensor(
        matched_gt,
        dtype=torch.long,
        device=device
    )

    return matched_pred_indices, matched_gt_indices

@torch.no_grad()
def hungarian_cost_match(
        pred_boxes,
        gt_boxes,
        cost_bbox=1.0,
        cost_iou=1.0
    ):
    device = pred_boxes.device

    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )

    bbox_cost = torch.cdist(
        pred_boxes[:, :6],
        gt_boxes[:, :6],
        p=1
    )

    pred_ra_boxes = boxes_rae_to_ra_xyxy(pred_boxes)
    gt_ra_boxes = boxes_rae_to_ra_xyxy(gt_boxes)

    ious = box_iou_2d(pred_ra_boxes, gt_ra_boxes)
    iou_cost = 1.0 - ious

    total_cost = cost_bbox * bbox_cost + cost_iou * iou_cost

    cost_matrix = total_cost.detach().cpu().numpy()

    from scipy.optimize import linear_sum_assignment
    matched_pred, matched_gt = linear_sum_assignment(cost_matrix)

    matched_pred_indices = torch.as_tensor(
        matched_pred,
        dtype=torch.long,
        device=device
    )

    matched_gt_indices = torch.as_tensor(
        matched_gt,
        dtype=torch.long,
        device=device
    )

    return matched_pred_indices, matched_gt_indices

def detection_loss(
        outputs,
        gt_boxes_list,
        gt_labels_list,
        num_classes,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        background_weight=0.1
    ):
    if isinstance(outputs, dict):
        pred_boxes = outputs["box_pred"].sigmoid()
        pred_logits = outputs["cls_pred"]
    else:
        box_dim = 7
        expected_output_dim = box_dim + num_classes + 1

        if outputs.shape[-1] != expected_output_dim:
            raise ValueError(
                f"Expected output dim {expected_output_dim}, got {outputs.shape[-1]}"
            )

        pred_boxes = outputs[:, :, :box_dim].sigmoid()
        pred_logits = outputs[:, :, box_dim:]

    device = pred_boxes.device
    batch_size = pred_boxes.shape[0]
    num_queries = pred_boxes.shape[1]

    target_classes = torch.full(
        (batch_size, num_queries),
        fill_value=num_classes,
        dtype=torch.long,
        device=device
    )

    matched_pred_boxes_all = []
    matched_gt_boxes_all = []

    for b in range(batch_size):
        gt_boxes = gt_boxes_list[b].to(device)
        gt_labels = gt_labels_list[b].to(device)

        if gt_boxes.shape[0] == 0:
            continue

        pred_boxes_b = pred_boxes[b]

        # matched_pred_indices, matched_gt_indices = greedy_cost_match(
        #     pred_boxes=pred_boxes_b,
        #     gt_boxes=gt_boxes,
        #     cost_bbox=1.0,
        #     cost_iou=1.0
        # )
        matched_pred_indices, matched_gt_indices = hungarian_cost_match(
            pred_boxes=pred_boxes_b,
            gt_boxes=gt_boxes,
            cost_bbox=1.0,
            cost_iou=1.0
        )

        if matched_pred_indices.numel() == 0:
            continue

        
        target_classes[b, matched_pred_indices] = gt_labels[matched_gt_indices]

        
        matched_pred_boxes_all.append(
            pred_boxes_b[matched_pred_indices]
        )
        matched_gt_boxes_all.append(
            gt_boxes[matched_gt_indices]
        )

    class_weights = torch.ones(num_classes + 1, device=device)
    class_weights[-1] = background_weight

    cls_loss = F.cross_entropy(
        pred_logits.reshape(-1, num_classes + 1),
        target_classes.reshape(-1),
        weight=class_weights
    )

    if len(matched_pred_boxes_all) > 0:
        matched_pred_boxes_all = torch.cat(matched_pred_boxes_all, dim=0)
        matched_gt_boxes_all = torch.cat(matched_gt_boxes_all, dim=0)

        box_loss = F.smooth_l1_loss(
            matched_pred_boxes_all,
            matched_gt_boxes_all
        )
    else:
        box_loss = torch.tensor(0.0, device=device)

    total_loss = (
        box_loss_weight * box_loss
        + cls_loss_weight * cls_loss
    )

    loss_dict = {
        "total_loss": total_loss.item(),
        "box_loss": box_loss.item(),
        "cls_loss": cls_loss.item()
    }

    return total_loss, loss_dict

def train_one_epoch(
        model,
        dataloader,
        optimizer,
        device,
        num_classes,
        epoch=None,
        num_epochs=None,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        background_weight=0.1,
        writer=None,
        global_step=0,
        log_interval=1
    ):
    model.train()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    num_batches = 0

    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch + 1}/{num_epochs}"
    else:
        desc = "Training"

    pbar = tqdm(dataloader, desc=desc, ncols=120)

    for batch in pbar:
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        loss, loss_dict = detection_loss(
            outputs=outputs,
            gt_boxes_list=batch["gt_boxes"],
            gt_labels_list=batch["gt_labels"],
            num_classes=num_classes,
            box_loss_weight=box_loss_weight,
            cls_loss_weight=cls_loss_weight,
            background_weight=background_weight
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        num_batches += 1
        global_step += 1

        avg_loss = total_loss_sum / num_batches
        avg_box = box_loss_sum / num_batches
        avg_cls = cls_loss_sum / num_batches

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "box": f"{avg_box:.4f}",
            "cls": f"{avg_cls:.4f}",
        })

    avg_total_loss = total_loss_sum / max(num_batches, 1)
    avg_box_loss = box_loss_sum / max(num_batches, 1)
    avg_cls_loss = cls_loss_sum / max(num_batches, 1)

    return {
        "train_loss": avg_total_loss,
        "train_box_loss": avg_box_loss,
        "train_cls_loss": avg_cls_loss,
        "global_step": global_step,
    }


@torch.no_grad()
def validate_loss(
        model,
        dataloader,
        device,
        num_classes,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        background_weight=0.1
    ):
    model.eval()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation loss", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        _, loss_dict = detection_loss(
            outputs=outputs,
            gt_boxes_list=batch["gt_boxes"],
            gt_labels_list=batch["gt_labels"],
            num_classes=num_classes,
            box_loss_weight=box_loss_weight,
            cls_loss_weight=cls_loss_weight,
            background_weight=background_weight
        )

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        num_batches += 1

    return {
        "val_loss": total_loss_sum / max(num_batches, 1),
        "val_box_loss": box_loss_sum / max(num_batches, 1),
        "val_cls_loss": cls_loss_sum / max(num_batches, 1),
    }


def default_val_loss_metrics():
    return {
        "val_loss": 0.0,
        "val_box_loss": 0.0,
        "val_cls_loss": 0.0,
    }

def main():

    parser = argparse.ArgumentParser(description="Train the dummy MVRSS detection module.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-boxes", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--background-weight", type=float, default=0.6)
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--checkpoint-base-dir", default="checkpoints")
    parser.add_argument("--log-base-dir", default="runs")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = DataConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        full_dataset,
        train_dataset,
        val_dataset,
        train_loader,
        val_loader
    ) = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples
    )

    model = MVRSS3DModel(
        d_in=64,
        e_in=37,
        num_boxes=args.num_boxes,
        box_dim=7,
        num_classes=args.num_classes,
        feature_channels=64,
        fusion_hidden_channels=64,
        decoder_hidden_channels=128,
        pooled_size=(8, 8)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_map = -1.0
    best_epoch = -1
    best_train_metrics = None
    best_metrics = None
    best_f1 = 0.0
    best_checkpoint_path = None
    best_checkpoint_paths = None
    best_epoch_checkpoint_path = None
    global_best_checkpoint_path = None
    global_best_checkpoint_paths = None
    window_best_map = -1.0
    window_best_epoch = -1
    window_best_train_metrics = None
    window_best_metrics = None
    window_best_f1 = 0.0
    window_best_checkpoint_path = None
    global_step = 0

    configured_sequences = get_config_sequences(cfg)
    primary_sequence = (
        cfg.sequence
        if cfg.sequence in configured_sequences
        else configured_sequences[0]
    )
    is_multi_sequence = len(configured_sequences) > 1
    best_window_size = 100 if is_multi_sequence else 10
    if len(configured_sequences) == 1:
        checkpoint_key = primary_sequence
        checkpoint_dir = create_checkpoint_run_dir(
            base_dir=args.checkpoint_base_dir,
            experiment_name="mvrss_detection",
            sequence=primary_sequence
        )
        checkpoint_dirs = {checkpoint_key: checkpoint_dir}
    else:
        checkpoint_dirs = create_checkpoint_run_dirs(
            base_dir=args.checkpoint_base_dir,
            experiment_name="mvrss_detection",
            sequences=configured_sequences
        )
        checkpoint_key = next(iter(checkpoint_dirs))
        checkpoint_dir = checkpoint_dirs[checkpoint_key]
    writer, log_dir = create_tensorboard_writer(
        base_dir=args.log_base_dir,
        experiment_name="mvrss_detection",
        sequence=cfg.sequence
    )
    write_tensorboard_run_config(
        writer=writer,
        cfg=cfg,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        learning_rate=args.lr,
        num_boxes=args.num_boxes,
        background_weight=args.background_weight,
        eval_iou_thresh=args.eval_iou_thresh
    )
    try:
        for epoch in range(args.epochs):
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                num_classes=args.num_classes,
                epoch=epoch,
                num_epochs=args.epochs,
                box_loss_weight=1.0,
                cls_loss_weight=1.0,
                background_weight=args.background_weight,
                writer=writer,
                global_step=global_step,
                log_interval=1
            )
            global_step = train_metrics["global_step"]
            if len(val_dataset) > 0:
                val_loss_metrics = validate_loss(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    num_classes=args.num_classes,
                    box_loss_weight=1.0,
                    cls_loss_weight=1.0,
                    background_weight=args.background_weight
                )
            else:
                val_loss_metrics = default_val_loss_metrics()

            if args.no_eval:
                train_metrics["train_iou"] = 0.0
                val_metrics = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "mAP": 0.0,
                    "ap_per_class": {
                        class_id: 0.0
                        for class_id in range(args.num_classes)
                    },
                    "val_iou": 0.0,
                    "iou_thresh": args.eval_iou_thresh,
                }
                val_metrics.update(val_loss_metrics)
                f1 = 0.0
            elif len(val_dataset) == 0:
                train_eval_metrics = evaluate_precision_recall(
                    model=model,
                    dataloader=train_loader,
                    device=device,
                    num_classes=args.num_classes,
                    prepare_model_inputs=prepare_model_inputs,
                    score_thresh=args.score_thresh,
                    iou_thresh=args.eval_iou_thresh,
                    max_detections=min(args.num_boxes, 20)
                )
                train_metrics["train_iou"] = train_eval_metrics["mean_iou"]
                val_metrics = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "mAP": 0.0,
                    "ap_per_class": {
                        class_id: 0.0
                        for class_id in range(args.num_classes)
                    },
                    "val_iou": 0.0,
                    "iou_thresh": args.eval_iou_thresh,
                }
                val_metrics.update(val_loss_metrics)
                f1 = 0.0
            else:
                eval_metrics = evaluate_train_val_iou(
                    model=model,
                    train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    device=device,
                    num_classes=args.num_classes,
                    prepare_model_inputs=prepare_model_inputs,
                    score_thresh=args.score_thresh,
                    iou_thresh=args.eval_iou_thresh,
                    max_detections=min(args.num_boxes, 20)
                )
                train_metrics["train_iou"] = eval_metrics["train_eval_iou"]
                val_metrics = eval_metrics["val_eval_metrics"]
                val_metrics["val_iou"] = eval_metrics["val_eval_iou"]
                val_metrics.update(val_loss_metrics)

                precision = val_metrics["precision"]
                recall = val_metrics["recall"]
                f1 = 2 * precision * recall / (precision + recall + 1e-6)

            learning_rate = optimizer.param_groups[0]["lr"]
            write_tensorboard_metrics(
                writer=writer,
                epoch=epoch + 1,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                f1=f1,
                learning_rate=learning_rate
            )

            is_best = val_metrics["mAP"] > best_map
            if is_best:
                best_map = val_metrics["mAP"]
                best_epoch = epoch + 1
                best_train_metrics = train_metrics.copy()
                best_metrics = val_metrics.copy()
                best_f1 = f1

            is_window_best = val_metrics["mAP"] > window_best_map
            checkpoint_path = None
            if is_best or is_window_best:
                checkpoint_path = save_epoch_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    cfg=cfg,
                    epoch=epoch + 1,
                    global_step=global_step,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    f1=f1,
                    learning_rate=learning_rate,
                    is_best=is_best
                )

            if is_best:
                best_epoch_checkpoint_path = checkpoint_path

            if is_window_best:
                window_best_map = val_metrics["mAP"]
                window_best_epoch = epoch + 1
                window_best_train_metrics = train_metrics.copy()
                window_best_metrics = val_metrics.copy()
                window_best_f1 = f1
                window_best_checkpoint_path = checkpoint_path

            should_save_window_best = (
                window_best_checkpoint_path is not None
                and ((epoch + 1) % best_window_size == 0 or (epoch + 1) == args.epochs)
            )
            if should_save_window_best:
                best_checkpoint_paths = {}
                for sequence, sequence_checkpoint_dir in checkpoint_dirs.items():
                    best_checkpoint_paths[sequence] = save_best_checkpoint_copy(
                        checkpoint_dir=sequence_checkpoint_dir,
                        source_checkpoint_path=window_best_checkpoint_path,
                        best_epoch=window_best_epoch,
                        best_map=window_best_map
                    )
                best_checkpoint_path = best_checkpoint_paths[checkpoint_key]
                print(
                    f"Saved {best_window_size}-epoch best model: "
                    f"epoch={window_best_epoch}, "
                    f"train_loss={window_best_train_metrics['train_loss']:.4f}, "
                    f"train_iou={window_best_train_metrics['train_iou']:.4f}, "
                    f"val_iou={window_best_metrics['val_iou']:.4f}, "
                    f"F1={window_best_f1:.4f}, "
                    f"mAP={window_best_map:.4f}, "
                    f"IoU={window_best_metrics['iou_thresh']:.4f}, "
                    f"P={window_best_metrics['precision']:.4f}, "
                    f"R={window_best_metrics['recall']:.4f}, "
                    f"TP={window_best_metrics['tp']}, "
                    f"FP={window_best_metrics['fp']}, "
                    f"FN={window_best_metrics['fn']}, "
                    f"best_paths={best_checkpoint_paths}"
                )
                window_best_map = -1.0
                window_best_epoch = -1
                window_best_train_metrics = None
                window_best_metrics = None
                window_best_f1 = 0.0
                window_best_checkpoint_path = None

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_metrics["train_loss"],
                "train_box_loss": train_metrics["train_box_loss"],
                "train_cls_loss": train_metrics["train_cls_loss"],
                "train_iou": train_metrics["train_iou"],
                "val_loss": val_metrics["val_loss"],
                "val_box_loss": val_metrics["val_box_loss"],
                "val_cls_loss": val_metrics["val_cls_loss"],
                "val_mAP": val_metrics["mAP"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_iou": val_metrics["val_iou"],
                "val_f1": f1,
                "iou": val_metrics["iou_thresh"],
                "tp": val_metrics["tp"],
                "fp": val_metrics["fp"],
                "fn": val_metrics["fn"],
            })
    finally:
        writer.close()

    print_training_history(history)

    if is_multi_sequence and best_epoch_checkpoint_path is not None:
        global_best_checkpoint_paths = {}
        for sequence, sequence_checkpoint_dir in checkpoint_dirs.items():
            global_best_checkpoint_paths[sequence] = save_named_checkpoint_copy(
                checkpoint_dir=sequence_checkpoint_dir,
                source_checkpoint_path=best_epoch_checkpoint_path,
                best_epoch=best_epoch,
                best_map=best_map,
                name_prefix="global_best"
            )
        global_best_checkpoint_path = global_best_checkpoint_paths[checkpoint_key]
        print(
            f"Saved global best model: "
            f"epoch={best_epoch}, "
            f"mAP={best_map:.4f}, "
            f"best_paths={global_best_checkpoint_paths}"
        )

    print("\nBest model summary")
    print("global_best_epoch:", best_epoch)
    print("global_best_mAP:", best_map)
    print("global_best_epoch_checkpoint_path:", best_epoch_checkpoint_path)
    print("global_best_checkpoint_path:", global_best_checkpoint_path)
    print("global_best_checkpoint_paths:", global_best_checkpoint_paths)
    print(f"last_saved_{best_window_size}_epoch_best_checkpoint_path:", best_checkpoint_path)
    print(f"last_saved_{best_window_size}_epoch_best_checkpoint_paths:", best_checkpoint_paths)
    print("checkpoint_dir:", checkpoint_dir)
    print("checkpoint_dirs:", checkpoint_dirs)
    print("global_best_metrics:", best_metrics)





if __name__ == "__main__":
    main()
