import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dummy_dataloader import (build_train_val_dataloaders, get_config_sequences, prepare_model_inputs)
from dummy_dataset import CLASS_NAMES, CLASS_TO_IDX
from dummy_evaluation import boxes_3d_to_ra_xyxy, evaluate_train_val_iou
from model_bifpn_heatmap import RADRAEBiFPNCenterPointModel
from model_deform_heatmap import RADRAEStageDeformCenterPointModel
from model_fpn_heatmap import RADRAEFPNDeformCenterPointModel
from model_con2d_heatmap import RADRAEStageCenterPointModel 
from utils_dummy.checkpoints import *
from utils_dummy.logging_utils import *
from utils_dummy.other_helping_dunctions import *
from zxy_config import DataConfig


NUM_CLASSES = 2


def parse_gpu_ids(gpu_ids_text):
    return [
        int(gpu_id.strip())
        for gpu_id in gpu_ids_text.split(",")
        if gpu_id.strip() != ""
    ]


def pairwise_box_giou_2d(boxes1, boxes2):
    """
    Computes Generalized IoU (GIoU) between two sets of boxes.
    Boxes should be in [x_min, y_min, x_max, y_max] format.
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    # Standard IoU Intersections
    left_top = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    iou = inter / union

    # Enclosing Box
    enclose_left_top = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclose_right_bottom = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclose_wh = (enclose_right_bottom - enclose_left_top).clamp(min=0)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1] + 1e-6

    # GIoU Calculation
    giou = iou - (enclose_area - union) / enclose_area
    return giou


def gaussian2d(radius, sigma=None, device="cpu"):
    diameter = 2 * radius + 1
    if sigma is None:
        sigma = diameter / 6

    x = torch.arange(0, diameter, device=device).float()
    y = torch.arange(0, diameter, device=device).float()
    y, x = torch.meshgrid(y, x, indexing="ij")

    center = radius
    gaussian = torch.exp(
        -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)
    )
    return gaussian


def draw_gaussian(heatmap, center_y, center_x, radius):
    R, A = heatmap.shape
    device = heatmap.device
    gaussian = gaussian2d(radius, device=device)

    left = min(center_x, radius)
    right = min(A - center_x - 1, radius)
    top = min(center_y, radius)
    bottom = min(R - center_y - 1, radius)

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return

    masked_heatmap = heatmap[
        center_y - top: center_y + bottom + 1,
        center_x - left: center_x + right + 1
    ]
    masked_gaussian = gaussian[
        radius - top: radius + bottom + 1,
        radius - left: radius + right + 1
    ]

    torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def heatmap_focal_loss(logits, targets, alpha=2.0, beta=4.0):
    pred = logits.sigmoid().clamp(min=1e-4, max=1.0 - 1e-4)
    pos_inds = targets.eq(1.0).float()
    neg_inds = targets.lt(1.0).float()
    neg_weights = torch.pow(1.0 - targets, beta)

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_inds
    neg_loss = (
        torch.log(1.0 - pred)
        * torch.pow(pred, alpha)
        * neg_weights
        * neg_inds
    )

    num_pos = pos_inds.sum()
    loss = -(pos_loss.sum() + neg_loss.sum())
    return loss / torch.clamp(num_pos, min=1.0)


def normalized_boxes_to_centerpoint_targets(box, H, W):
    r_norm = box[0].clamp(0.0, 1.0)
    a_norm = box[1].clamp(0.0, 1.0)
    e_norm = box[2].clamp(0.0, 1.0)
    size_norm = box[3:6].clamp(min=1e-4, max=1.0)
    yaw_norm = box[6].clamp(0.0, 1.0)

    y_float = r_norm * H
    x_float = a_norm * W
    center_y = int(torch.floor(y_float).clamp(0, H - 1).item())
    center_x = int(torch.floor(x_float).clamp(0, W - 1).item())

    offset_y = (y_float - center_y).clamp(0.0, 1.0)
    offset_x = (x_float - center_x).clamp(0.0, 1.0)

    yaw_rad = (yaw_norm * 2.0 * torch.pi) - torch.pi
    yaw_sin_cos = torch.stack([torch.sin(yaw_rad), torch.cos(yaw_rad)])

    return {
        "center_y": center_y,
        "center_x": center_x,
        "center_offset": torch.stack([offset_y, offset_x]),
        "center_height": e_norm.unsqueeze(0),
        "size": size_norm,
        "yaw": yaw_sin_cos,
    }


def build_centerpoint_targets(
        gt_boxes,
        gt_labels,
        cls_logits,
        num_classes,
        radius=3
    ):
    B, _, H, W = cls_logits.shape
    device = cls_logits.device

    heatmap_targets = torch.zeros((B, num_classes, H, W), device=device)
    reg_targets = {
        "center_offset": torch.zeros((B, 2, H, W), device=device),
        "center_height": torch.zeros((B, 1, H, W), device=device),
        "size": torch.zeros((B, 3, H, W), device=device),
        "yaw": torch.zeros((B, 2, H, W), device=device),
        "box": torch.zeros((B, 7, H, W), device=device),
    }
    reg_mask = torch.zeros((B, 1, H, W), device=device)

    for b in range(B):
        boxes_b = gt_boxes[b].to(device)
        labels_b = gt_labels[b].to(device)

        if boxes_b.numel() == 0:
            continue

        for box, cls_id in zip(boxes_b, labels_b):
            cls_id = int(cls_id.item())
            if cls_id < 0 or cls_id >= num_classes:
                continue

            target = normalized_boxes_to_centerpoint_targets(
                box=box,
                H=H,
                W=W
            )
            center_y = target["center_y"]
            center_x = target["center_x"]

            draw_gaussian(
                heatmap=heatmap_targets[b, cls_id],
                center_y=center_y,
                center_x=center_x,
                radius=radius
            )

            reg_targets["center_offset"][b, :, center_y, center_x] = target["center_offset"]
            reg_targets["center_height"][b, :, center_y, center_x] = target["center_height"]
            reg_targets["size"][b, :, center_y, center_x] = target["size"]
            reg_targets["yaw"][b, :, center_y, center_x] = target["yaw"]
            reg_targets["box"][b, :, center_y, center_x] = box.clamp(0.0, 1.0)
            reg_mask[b, :, center_y, center_x] = 1.0

    return heatmap_targets, reg_targets, reg_mask


def masked_l1_loss(pred, target, mask):
    mask = mask.expand_as(pred)
    denom = torch.clamp(mask.sum(), min=1.0)
    return F.l1_loss(pred * mask, target * mask, reduction="sum") / denom


def dense_centerpoint_outputs_to_boxes(outputs):
    cls_logits = outputs["cls_logits"]
    B, _, H, W = cls_logits.shape
    device = cls_logits.device
    dtype = cls_logits.dtype

    y_grid = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)
    x_grid = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)

    center_offset = outputs["center_offset"].sigmoid()
    center_height = outputs["center_height"].sigmoid()
    size = outputs["size"].sigmoid()
    yaw = outputs["yaw"]

    r_center = (y_grid + center_offset[:, 0]) / max(H, 1)
    a_center = (x_grid + center_offset[:, 1]) / max(W, 1)
    e_center = center_height[:, 0]
    yaw_angle = torch.atan2(yaw[:, 0], yaw[:, 1])
    yaw_norm = (yaw_angle + torch.pi) / (2.0 * torch.pi)

    return torch.stack(
        [
            r_center,
            a_center,
            e_center,
            size[:, 0],
            size[:, 1],
            size[:, 2],
            yaw_norm,
        ],
        dim=1
    ).clamp(min=1e-4, max=1.0 - 1e-4)


def centerpoint_giou_loss(outputs, target_boxes, mask):
    positive_mask = mask.squeeze(1).bool()
    if positive_mask.sum() == 0:
        return outputs["cls_logits"].new_tensor(0.0)

    pred_box_map = dense_centerpoint_outputs_to_boxes(outputs)
    pred_boxes = pred_box_map.permute(0, 2, 3, 1)[positive_mask]
    gt_boxes = target_boxes.permute(0, 2, 3, 1)[positive_mask]

    pred_ra_boxes = boxes_3d_to_ra_xyxy(pred_boxes)
    gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)
    gious = pairwise_box_giou_2d(pred_ra_boxes, gt_ra_boxes).diag()

    return (1.0 - gious).mean()


def centerpoint_detection_loss(
        outputs,
        gt_boxes_list,
        gt_labels_list,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        giou_loss_weight=2.0,
        heatmap_radius=3,
        num_classes=NUM_CLASSES
    ):
    cls_logits = outputs["cls_logits"]
    heatmap_targets, reg_targets, reg_mask = build_centerpoint_targets(
        gt_boxes=gt_boxes_list,
        gt_labels=gt_labels_list,
        cls_logits=cls_logits,
        num_classes=num_classes,
        radius=heatmap_radius
    )

    cls_loss = heatmap_focal_loss(
        logits=cls_logits,
        targets=heatmap_targets
    )

    pred_center_offset = outputs["center_offset"].sigmoid()
    pred_center_height = outputs["center_height"].sigmoid()
    pred_size = outputs["size"].sigmoid()
    pred_yaw = F.normalize(outputs["yaw"], dim=1)

    offset_loss = masked_l1_loss(
        pred=pred_center_offset,
        target=reg_targets["center_offset"],
        mask=reg_mask
    )
    height_loss = masked_l1_loss(
        pred=pred_center_height,
        target=reg_targets["center_height"],
        mask=reg_mask
    )
    size_loss = masked_l1_loss(
        pred=pred_size,
        target=reg_targets["size"],
        mask=reg_mask
    )
    yaw_loss = masked_l1_loss(
        pred=pred_yaw,
        target=reg_targets["yaw"],
        mask=reg_mask
    )
    giou_loss = centerpoint_giou_loss(
        outputs=outputs,
        target_boxes=reg_targets["box"],
        mask=reg_mask
    )

    box_loss = (
        offset_loss
        + height_loss
        + size_loss
        + yaw_loss
        + (giou_loss_weight * giou_loss)
    )
    total_loss = (box_loss_weight * box_loss) + (cls_loss_weight * cls_loss)

    loss_dict = {
        "total_loss": total_loss.item(),
        "box_loss": box_loss.item(),
        "cls_loss": cls_loss.item(),
        "heatmap_loss": cls_loss.item(),
        "offset_loss": offset_loss.item(),
        "height_loss": height_loss.item(),
        "size_loss": size_loss.item(),
        "yaw_loss": yaw_loss.item(),
        "giou_loss": giou_loss.item(),
        "num_center_targets": int(reg_mask.sum().item()),
    }

    return total_loss, loss_dict


def train_one_epoch(
        model,
        dataloader,
        optimizer,
        device,
        epoch=None,
        num_epochs=None,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        heatmap_radius=3,
        centerpoint_giou_loss_weight=2.0
    ):
    model.train()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    heatmap_loss_sum = 0.0
    num_batches = 0

    desc = f"Epoch {epoch + 1}/{num_epochs}" if epoch is not None else "Training"
    pbar = tqdm(dataloader, desc=desc, ncols=120)

    for batch in pbar:
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        loss, loss_dict = centerpoint_detection_loss(
            outputs=outputs,
            gt_boxes_list=batch["gt_boxes"],
            gt_labels_list=batch["gt_labels"],
            box_loss_weight=box_loss_weight,
            cls_loss_weight=cls_loss_weight,
            giou_loss_weight=centerpoint_giou_loss_weight,
            heatmap_radius=heatmap_radius,
            num_classes=NUM_CLASSES
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        heatmap_loss_sum += loss_dict["heatmap_loss"]
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{(total_loss_sum / num_batches):.4f}",
            "box": f"{(box_loss_sum / num_batches):.4f}",
            "cls": f"{(cls_loss_sum / num_batches):.4f}",
            "hm": f"{(heatmap_loss_sum / num_batches):.4f}",
        })

    return {
        "train_loss": total_loss_sum / max(num_batches, 1),
        "train_box_loss": box_loss_sum / max(num_batches, 1),
        "train_cls_loss": cls_loss_sum / max(num_batches, 1),
        "train_heatmap_loss": heatmap_loss_sum / max(num_batches, 1),
    }


@torch.no_grad()
def validate_loss(
        model,
        dataloader,
        device,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        heatmap_radius=3,
        centerpoint_giou_loss_weight=2.0
    ):
    model.eval()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    heatmap_loss_sum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation loss", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        _, loss_dict = centerpoint_detection_loss(
            outputs=outputs,
            gt_boxes_list=batch["gt_boxes"],
            gt_labels_list=batch["gt_labels"],
            box_loss_weight=box_loss_weight,
            cls_loss_weight=cls_loss_weight,
            giou_loss_weight=centerpoint_giou_loss_weight,
            heatmap_radius=heatmap_radius,
            num_classes=NUM_CLASSES
        )

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        heatmap_loss_sum += loss_dict["heatmap_loss"]
        num_batches += 1

    return {
        "val_loss": total_loss_sum / max(num_batches, 1),
        "val_box_loss": box_loss_sum / max(num_batches, 1),
        "val_cls_loss": cls_loss_sum / max(num_batches, 1),
        "val_heatmap_loss": heatmap_loss_sum / max(num_batches, 1),
    }


def argparse_args():
    parser = argparse.ArgumentParser(description="Train the dummy MVRSS detection module.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=50)#50
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-boxes", type=int, default=64)
    parser.add_argument("--heatmap-radius", type=int, default=3)
    parser.add_argument("--centerpoint-giou-loss-weight", type=float, default=2.0)
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
    parser.add_argument("--model-type", type=str, default="model5", choices=["model1", "model2", "model4", "model5"])
    args = parser.parse_args()
    return args


def main():
    args = argparse_args()
    if args.checkpoint_epoch_step <= 0:
        raise ValueError(f"--checkpoint-epoch-step must be greater than 0")

    set_seed(args.seed)
    cfg = DataConfig()
    configured_sequences = get_config_sequences(cfg)
    args.num_classes = NUM_CLASSES
    args.class_names = CLASS_NAMES.copy()
    args.class_to_idx = CLASS_TO_IDX.copy()

    print(f"Training classes: {CLASS_NAMES}")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
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
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        gpu_ids = []
        device = torch.device("cpu")

    (train_dataset, val_dataset, train_loader, val_loader) = build_train_val_dataloaders(
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

    if args.model_type == "model1":
        model = RADRAEStageCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=NUM_CLASSES,
            decoder_hidden_channels=128,
            num_boxes=args.num_boxes,
        ).to(device)
    elif args.model_type == "model2":
        model = RADRAEBiFPNCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=NUM_CLASSES,
            decoder_hidden_channels=128,
            num_boxes=args.num_boxes,
        ).to(device)
    elif args.model_type == "model4":
        model = RADRAEStageDeformCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=NUM_CLASSES,
            decoder_hidden_channels=128,
            num_boxes=args.num_boxes
        ).to(device)
    elif args.model_type == "model5":
        model = RADRAEFPNDeformCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=NUM_CLASSES,
            decoder_hidden_channels=128,
            num_boxes=args.num_boxes
        ).to(device)
    else:
        raise ValueError(f"Unknown or unsupported model_type: {args.model_type}")

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
        experiment_name="mvrss_detection",
        sequences=configured_sequences
    )
    checkpoint_key = next(iter(checkpoint_dirs))
    checkpoint_dir = checkpoint_dirs[checkpoint_key]

    writer = create_tensorboard_writer(
        base_dir=args.log_base_dir,
        experiment_name="mvrss_detection",
        sequence=configured_sequences
    )
    
    write_tensorboard_run_config(
        writer=writer, cfg=cfg, num_epochs=args.epochs, batch_size=args.batch_size,
        train_size=len(train_dataset), val_size=len(val_dataset), learning_rate=args.lr,
        num_boxes=args.num_boxes, num_classes=NUM_CLASSES, class_names=CLASS_NAMES,
        eval_iou_thresh=args.eval_iou_thresh
    )

    for epoch in range(args.epochs):
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
            centerpoint_giou_loss_weight=args.centerpoint_giou_loss_weight
        )
        
        val_loss_metrics = validate_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            box_loss_weight=1.0,
            cls_loss_weight=1.0,
            heatmap_radius=args.heatmap_radius,
            centerpoint_giou_loss_weight=args.centerpoint_giou_loss_weight
        )


        eval_metrics = evaluate_train_val_iou(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            num_classes=NUM_CLASSES,
            prepare_model_inputs=prepare_model_inputs,
            score_thresh=args.score_thresh,
            iou_thresh=args.eval_iou_thresh,
            max_detections=args.num_boxes
        )
        
        val_metrics, f1 = build_epoch_eval_metrics(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            val_loss_metrics=val_loss_metrics
        )

        learning_rate = optimizer.param_groups[0]["lr"]
        write_tensorboard_metrics(
            writer=writer, epoch=epoch + 1, train_metrics=train_metrics,
            val_metrics=val_metrics, f1=f1, learning_rate=learning_rate
        )

        checkpoint_path = save_epoch_and_update_best_checkpoint(
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
