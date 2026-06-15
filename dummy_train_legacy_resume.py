import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dummy_dataloader import build_train_val_dataloaders, get_config_sequences, prepare_model_inputs
from dummy_dataset import CLASS_NAMES, CLASS_TO_IDX
from dummy_evaluation import (
    box_iou_2d,
    boxes_3d_to_ra_xyxy,
    compute_k_radar_map,
)
from dummy_module import MVRSS3DModelDeform
from utils_dummy.checkpoints import (
    create_checkpoint_run_dirs,
    save_replacing_named_checkpoint_copy,
)
from utils_dummy.logging_utils import (
    create_tensorboard_writer,
    write_tensorboard_metrics,
    write_tensorboard_run_config,
)
from utils_dummy.other_helping_dunctions import (
    BestCheckpointState,
    append_training_history,
    build_epoch_eval_metrics,
    save_epoch_and_update_best_checkpoint,
    save_global_best_checkpoint,
    set_seed,
)
from zxy_config import DataConfig


NUM_CLASSES = 2
BOX_DIM = 7


def parse_gpu_ids(gpu_ids_text):
    return [
        int(gpu_id.strip())
        for gpu_id in gpu_ids_text.split(",")
        if gpu_id.strip() != ""
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume legacy query-style MVRSS training from old box_pred/cls_pred checkpoints."
    )
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--initial-best-checkpoint", default=None)
    parser.add_argument("--start-epoch", type=int, default=101)
    parser.add_argument("--end-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=180)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-boxes", type=int, default=64)
    parser.add_argument("--background-weight", type=float, default=0.5)
    parser.add_argument("--box-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--iou-loss-weight", type=float, default=1.0)
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-mode", default="file", choices=["random", "file"])
    parser.add_argument("--split-dir", default="split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--checkpoint-epoch-step", type=int, default=10)
    parser.add_argument("--checkpoint-base-dir", default="checkpoints")
    parser.add_argument("--log-base-dir", default="runs")
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument("--model-type", default="model4")
    parser.add_argument("--pooled-size", type=int, default=16)
    parser.add_argument("--no-load-optimizer", action="store_true")
    return parser.parse_args()


def select_device_and_gpus(gpu_ids_text):
    gpu_ids = parse_gpu_ids(gpu_ids_text)
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        available_gpu_count = torch.cuda.device_count()
        unavailable_gpu_ids = [
            gpu_id for gpu_id in gpu_ids
            if gpu_id < 0 or gpu_id >= available_gpu_count
        ]
        if len(unavailable_gpu_ids) > 0:
            raise ValueError(
                f"Requested GPU ids {unavailable_gpu_ids}, "
                f"but only {available_gpu_count} CUDA device(s) are available."
            )
        return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids

    return torch.device("cpu"), []


def build_legacy_model(args, device):
    return MVRSS3DModelDeform(
        d_in=64,
        e_in=37,
        num_boxes=args.num_boxes,
        box_dim=BOX_DIM,
        num_classes=NUM_CLASSES,
        pooled_size=(args.pooled_size, args.pooled_size),
    ).to(device)


def load_resume_checkpoint(model, optimizer, checkpoint_path, device, load_optimizer=True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_for_state_dict = model.module if isinstance(model, torch.nn.DataParallel) else model
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
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
    if isinstance(checkpoint, dict) and checkpoint.get("epoch") is not None:
        checkpoint_epoch = int(checkpoint["epoch"])

    return checkpoint_epoch, optimizer_loaded


def greedy_match_queries(pred_boxes, gt_boxes):
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return []

    l1_cost = torch.cdist(pred_boxes, gt_boxes, p=1) / pred_boxes.shape[-1]
    iou_cost = 1.0 - box_iou_2d(
        boxes_3d_to_ra_xyxy(pred_boxes),
        boxes_3d_to_ra_xyxy(gt_boxes),
    )
    cost = l1_cost + iou_cost

    matches = []
    used_queries = set()
    used_targets = set()
    max_matches = min(pred_boxes.shape[0], gt_boxes.shape[0])
    for _ in range(max_matches):
        best_query = -1
        best_target = -1
        best_cost = None
        for query_idx in range(cost.shape[0]):
            if query_idx in used_queries:
                continue
            for target_idx in range(cost.shape[1]):
                if target_idx in used_targets:
                    continue
                cost_value = cost[query_idx, target_idx]
                if best_cost is None or cost_value < best_cost:
                    best_cost = cost_value
                    best_query = query_idx
                    best_target = target_idx

        if best_query < 0:
            break

        used_queries.add(best_query)
        used_targets.add(best_target)
        matches.append((best_query, best_target))

    return matches


def legacy_detection_loss(
        outputs,
        gt_boxes_list,
        gt_labels_list,
        num_classes,
        background_weight=0.5,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        iou_loss_weight=1.0,
    ):
    pred_boxes = outputs["box_pred"].sigmoid().clamp(min=1e-4, max=1.0 - 1e-4)
    cls_logits = outputs["cls_pred"]
    batch_size, num_queries, _ = pred_boxes.shape
    device = pred_boxes.device

    cls_targets = torch.full(
        (batch_size, num_queries),
        fill_value=num_classes,
        dtype=torch.long,
        device=device,
    )
    box_losses = []
    iou_losses = []
    iou_values = []

    for batch_idx in range(batch_size):
        gt_boxes = gt_boxes_list[batch_idx].to(device)
        gt_labels = gt_labels_list[batch_idx].to(device)
        valid_gt = gt_labels < num_classes
        gt_boxes = gt_boxes[valid_gt].clamp(min=1e-4, max=1.0 - 1e-4)
        gt_labels = gt_labels[valid_gt]
        if gt_boxes.numel() == 0:
            continue

        matches = greedy_match_queries(pred_boxes[batch_idx].detach(), gt_boxes)
        for query_idx, target_idx in matches:
            cls_targets[batch_idx, query_idx] = gt_labels[target_idx].long()
            pred_box = pred_boxes[batch_idx, query_idx]
            gt_box = gt_boxes[target_idx]
            box_losses.append(F.l1_loss(pred_box, gt_box, reduction="mean"))
            iou = box_iou_2d(
                boxes_3d_to_ra_xyxy(pred_box.unsqueeze(0)),
                boxes_3d_to_ra_xyxy(gt_box.unsqueeze(0)),
            ).squeeze()
            iou_values.append(iou.detach())
            iou_losses.append(1.0 - iou)

    class_weights = cls_logits.new_ones(num_classes + 1)
    class_weights[num_classes] = background_weight
    cls_loss = F.cross_entropy(
        cls_logits.reshape(-1, num_classes + 1),
        cls_targets.reshape(-1),
        weight=class_weights,
    )

    if len(box_losses) == 0:
        box_loss = pred_boxes.new_tensor(0.0)
        iou_loss = pred_boxes.new_tensor(0.0)
        mean_iou = 0.0
    else:
        box_loss = torch.stack(box_losses).mean()
        iou_loss = torch.stack(iou_losses).mean()
        mean_iou = float(torch.stack(iou_values).mean().item())

    total_loss = (
        box_loss_weight * box_loss
        + cls_loss_weight * cls_loss
        + iou_loss_weight * iou_loss
    )

    return total_loss, {
        "total_loss": float(total_loss.item()),
        "box_loss": float(box_loss.item()),
        "cls_loss": float(cls_loss.item()),
        "iou_loss": float(iou_loss.item()),
        "mean_iou": mean_iou,
    }


def legacy_outputs_to_detections(outputs, num_classes, max_detections):
    boxes = outputs["box_pred"].sigmoid().clamp(min=1e-4, max=1.0 - 1e-4)
    class_scores = outputs["cls_pred"].softmax(dim=-1)[..., :num_classes]
    scores, labels = class_scores.max(dim=-1)
    topk_count = min(max_detections, scores.shape[1])
    top_scores, top_indices = scores.topk(topk_count, dim=1)
    top_labels = labels.gather(dim=1, index=top_indices)
    gather_index = top_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1])
    top_boxes = boxes.gather(dim=1, index=gather_index)
    return top_boxes, top_scores, top_labels


@torch.no_grad()
def evaluate_legacy_precision_recall(
        model,
        dataloader,
        device,
        num_classes,
        score_thresh,
        iou_thresh,
        max_detections,
    ):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    total_iou_count = 0
    predictions_by_class = {class_id: [] for class_id in range(num_classes)}
    gt_by_class = {class_id: {} for class_id in range(num_classes)}
    sequence_counter = 0

    for batch in tqdm(dataloader, desc="Evaluation", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)
        pred_boxes, pred_scores, pred_labels = legacy_outputs_to_detections(
            outputs=outputs,
            num_classes=num_classes,
            max_detections=max_detections,
        )

        for batch_idx in range(pred_boxes.shape[0]):
            if "sequence_id" in batch:
                sequence_id = batch["sequence_id"][batch_idx]
            elif "file_idx" in batch:
                sequence_id = batch["file_idx"][batch_idx]
            else:
                sequence_id = sequence_counter
                sequence_counter += 1

            gt_boxes_all = batch["gt_boxes"][batch_idx].to(device)
            gt_labels_all = batch["gt_labels"][batch_idx].to(device)
            valid_gt = gt_labels_all < num_classes
            gt_boxes = gt_boxes_all[valid_gt]
            gt_labels = gt_labels_all[valid_gt]

            for class_id in range(num_classes):
                gt_by_class[class_id][sequence_id] = {
                    "boxes": gt_boxes[gt_labels == class_id].detach().cpu()
                }

            boxes_b = pred_boxes[batch_idx]
            scores_b = pred_scores[batch_idx]
            labels_b = pred_labels[batch_idx]

            for pred_box, pred_label, pred_score in zip(boxes_b, labels_b, scores_b):
                predictions_by_class[int(pred_label.item())].append({
                    "sequence_id": sequence_id,
                    "score": float(pred_score.item()),
                    "box": pred_box.detach().cpu(),
                })

            keep = scores_b > score_thresh
            point_boxes = boxes_b[keep]
            point_labels = labels_b[keep]
            point_scores = scores_b[keep]

            if point_boxes.shape[0] == 0:
                total_fn += gt_boxes.shape[0]
                continue
            if gt_boxes.shape[0] == 0:
                total_fp += point_boxes.shape[0]
                continue

            ious = box_iou_2d(
                boxes_3d_to_ra_xyxy(point_boxes),
                boxes_3d_to_ra_xyxy(gt_boxes),
            )
            matched_gt = set()
            order = point_scores.argsort(descending=True)
            for pred_idx_tensor in order:
                pred_idx = int(pred_idx_tensor.item())
                best_iou = -1.0
                best_gt_idx = -1
                for gt_idx in range(gt_boxes.shape[0]):
                    if gt_idx in matched_gt:
                        continue
                    if int(point_labels[pred_idx].item()) != int(gt_labels[gt_idx].item()):
                        continue
                    iou_value = float(ious[pred_idx, gt_idx].item())
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0 and best_iou >= iou_thresh:
                    total_tp += 1
                    total_iou += best_iou
                    total_iou_count += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

            total_fn += gt_boxes.shape[0] - len(matched_gt)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    mean_iou = total_iou / max(total_iou_count, 1)
    mean_ap, ap_per_class, _ = compute_k_radar_map(
        predictions_by_class=predictions_by_class,
        gt_by_class=gt_by_class,
        num_classes=num_classes,
        iou_thresh=iou_thresh,
    )

    return {
        "precision": precision,
        "recall": recall,
        "mAP": mean_ap,
        "ap_per_class": ap_per_class,
        "mean_iou": mean_iou,
        "iou_thresh": iou_thresh,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def evaluate_legacy_train_val_iou(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_classes,
        score_thresh,
        iou_thresh,
        max_detections,
    ):
    train_eval_metrics = evaluate_legacy_precision_recall(
        model=model,
        dataloader=train_dataloader,
        device=device,
        num_classes=num_classes,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        max_detections=max_detections,
    )
    val_eval_metrics = evaluate_legacy_precision_recall(
        model=model,
        dataloader=val_dataloader,
        device=device,
        num_classes=num_classes,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        max_detections=max_detections,
    )
    return {
        "train_eval_iou": train_eval_metrics["mean_iou"],
        "val_eval_iou": val_eval_metrics["mean_iou"],
        "train_eval_metrics": train_eval_metrics,
        "val_eval_metrics": val_eval_metrics,
    }


def run_loss_epoch(
        model,
        dataloader,
        device,
        args,
        optimizer=None,
        epoch=None,
        num_epochs=None,
    ):
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    iou_sum = 0.0
    num_batches = 0
    desc = "Training" if is_train else "Validation loss"
    if is_train and epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch + 1}/{num_epochs}"

    for batch in tqdm(dataloader, desc=desc, ncols=120, leave=not is_train):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)
        loss, loss_dict = legacy_detection_loss(
            outputs=outputs,
            gt_boxes_list=batch["gt_boxes"],
            gt_labels_list=batch["gt_labels"],
            num_classes=NUM_CLASSES,
            background_weight=args.background_weight,
            box_loss_weight=args.box_loss_weight,
            cls_loss_weight=args.cls_loss_weight,
            iou_loss_weight=args.iou_loss_weight,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        iou_sum += loss_dict["mean_iou"]
        num_batches += 1

    prefix = "train" if is_train else "val"
    return {
        f"{prefix}_loss": loss_sum / max(num_batches, 1),
        f"{prefix}_box_loss": box_loss_sum / max(num_batches, 1),
        f"{prefix}_cls_loss": cls_loss_sum / max(num_batches, 1),
        f"{prefix}_heatmap_loss": 0.0,
        f"{prefix}_quality_loss": 0.0,
        f"{prefix}_iou": iou_sum / max(num_batches, 1),
    }


def initialize_best_state(best_state, initial_best_checkpoint, checkpoint_dir):
    if initial_best_checkpoint is None or initial_best_checkpoint == "":
        return None
    if not os.path.exists(initial_best_checkpoint):
        raise FileNotFoundError(f"Initial best checkpoint not found: {initial_best_checkpoint}")

    checkpoint = torch.load(initial_best_checkpoint, map_location="cpu")
    best_epoch = int(checkpoint.get("epoch", 0))
    best_map = float(checkpoint.get("mAP", checkpoint.get("val_metrics", {}).get("mAP", -1.0)))
    copied_path = save_replacing_named_checkpoint_copy(
        checkpoint_dir=checkpoint_dir,
        source_checkpoint_path=initial_best_checkpoint,
        best_epoch=best_epoch,
        best_map=best_map,
        name_prefix="global_best",
    )
    best_state.map_score = best_map
    best_state.epoch = best_epoch
    best_state.global_best_path = copied_path
    return copied_path


def main():
    args = parse_args()
    if args.checkpoint_epoch_step <= 0:
        raise ValueError("--checkpoint-epoch-step must be greater than 0")

    set_seed(args.seed)
    cfg = DataConfig()
    configured_sequences = get_config_sequences(cfg)
    args.epochs = args.end_epoch
    args.num_classes = NUM_CLASSES
    args.class_names = CLASS_NAMES.copy()
    args.class_to_idx = CLASS_TO_IDX.copy()
    args.match_iou_thresh = None

    print(f"Training classes: {CLASS_NAMES}")
    print(f"Resume checkpoint: {args.resume_checkpoint}")
    print(f"Initial best checkpoint: {args.initial_best_checkpoint}")
    print(f"Resume training epochs: {args.start_epoch}-{args.end_epoch}")
    print("Legacy query-style model: dummy_module.MVRSS3DModelDeform")

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
        split_dir=args.split_dir,
    )
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty.")

    model = build_legacy_model(args=args, device=device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_epoch, optimizer_loaded = load_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=args.resume_checkpoint,
        device=device,
        load_optimizer=not args.no_load_optimizer,
    )
    print(f"Loaded checkpoint epoch={checkpoint_epoch}, optimizer_loaded={optimizer_loaded}")

    checkpoint_dirs = create_checkpoint_run_dirs(
        base_dir=args.checkpoint_base_dir,
        experiment_name="mvrss_detection_resume",
        sequences=configured_sequences,
        model_type=args.model_type,
    )
    checkpoint_key = next(iter(checkpoint_dirs))
    checkpoint_dir = checkpoint_dirs[checkpoint_key]
    print(f"Saving checkpoints to: {checkpoint_dir}")

    writer = create_tensorboard_writer(
        base_dir=args.log_base_dir,
        experiment_name="mvrss_detection_resume",
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
        num_boxes=args.num_boxes,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        eval_iou_thresh=args.eval_iou_thresh,
        model_type=f"{args.model_type}_legacy_query",
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

    for epoch_number in range(args.start_epoch, args.end_epoch + 1):
        train_metrics = run_loss_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            args=args,
            optimizer=optimizer,
            epoch=epoch_number - 1,
            num_epochs=args.end_epoch,
        )

        with torch.no_grad():
            val_loss_metrics = run_loss_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                args=args,
            )

        eval_metrics = evaluate_legacy_train_val_iou(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            num_classes=NUM_CLASSES,
            score_thresh=args.score_thresh,
            iou_thresh=args.eval_iou_thresh,
            max_detections=args.num_boxes,
        )
        val_metrics, f1 = build_epoch_eval_metrics(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            val_loss_metrics=val_loss_metrics,
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
