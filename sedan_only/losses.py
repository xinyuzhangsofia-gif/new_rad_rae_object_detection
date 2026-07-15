import math

import torch
import torch.nn.functional as F

from cfg_model import get_rae_scope_start_and_shape
from training_utils.losses import (
    build_centerpoint_targets,
    build_radenet_gaussian_heatmap,
    centerpoint_gwd_loss,
    centerpoint_quality_loss,
    gaussian_wasserstein_distance_batch,
    masked_l1_loss,
)
from training_utils.radenet_utils import (
    raw_local_rae_boxes_to_metric_boxes,
    regression_cell_to_metric_box,
)


SEDAN_CLASS_ID = 0
BUS_CLASS_ID = 1


def _filter_sedan_targets(gt_boxes_list, gt_labels_list):
    sedan_boxes_list = []
    sedan_labels_list = []
    for boxes, labels in zip(gt_boxes_list, gt_labels_list):
        keep = labels == SEDAN_CLASS_ID
        sedan_boxes = boxes[keep]
        sedan_boxes_list.append(sedan_boxes)
        sedan_labels_list.append(
            torch.zeros(
                (sedan_boxes.shape[0],),
                dtype=torch.long,
                device=labels.device,
            )
        )
    return sedan_boxes_list, sedan_labels_list


def _build_object_ignore_mask(
        gt_boxes_list,
        gt_labels_list,
        height,
        width,
        device,
        ignore_margin=1.0,
        ignore_expand_ratio=1.0,
        gt_ignore_boxes_list=None,
    ):
    mask = torch.zeros((len(gt_boxes_list), 1, height, width), device=device)

    for batch_idx, (boxes, labels) in enumerate(zip(gt_boxes_list, gt_labels_list)):
        if gt_ignore_boxes_list is None:
            keep = labels == BUS_CLASS_ID
            if not keep.any():
                continue
            ignore_boxes = boxes[keep].to(device)
        else:
            ignore_boxes = gt_ignore_boxes_list[batch_idx].to(device)

        if ignore_boxes.numel() == 0:
            continue

        for box in ignore_boxes:
            center_y = float(box[0].clamp(0.0, 1.0).item()) * float(height)
            center_x = float(box[1].clamp(0.0, 1.0).item()) * float(width)
            half_h = (
                float(box[3].abs().clamp(min=1e-4).item())
                * float(height)
                * 0.5
                * float(ignore_expand_ratio)
            ) + float(ignore_margin)
            half_w = (
                float(box[4].abs().clamp(min=1e-4).item())
                * float(width)
                * 0.5
                * float(ignore_expand_ratio)
            ) + float(ignore_margin)

            y0 = max(0, int(math.floor(center_y - half_h)))
            y1 = min(height - 1, int(math.ceil(center_y + half_h)))
            x0 = max(0, int(math.floor(center_x - half_w)))
            x1 = min(width - 1, int(math.ceil(center_x + half_w)))
            if y0 > y1 or x0 > x1:
                continue
            mask[batch_idx, 0, y0:y1 + 1, x0:x1 + 1] = 1.0

    return mask


def _filter_sedan_raw_targets(gt_boxes_raw_list, gt_labels_list):
    sedan_boxes_raw_list = []
    sedan_labels_list = []
    for boxes, labels in zip(gt_boxes_raw_list, gt_labels_list):
        keep = labels == SEDAN_CLASS_ID
        sedan_boxes = boxes[keep]
        sedan_boxes_raw_list.append(sedan_boxes)
        sedan_labels_list.append(
            torch.zeros(
                (sedan_boxes.shape[0],),
                dtype=torch.long,
                device=labels.device,
            )
        )
    return sedan_boxes_raw_list, sedan_labels_list


def _build_object_ignore_mask_raw(
        gt_boxes_raw_list,
        gt_labels_list,
        scope_modes,
        full_rae_shapes,
        height,
        width,
        device,
        ignore_margin=1.0,
        ignore_expand_ratio=1.0,
        gt_ignore_boxes_raw_list=None,
    ):
    mask = torch.zeros((len(gt_boxes_raw_list), 1, height, width), device=device)

    for batch_idx, (boxes, labels) in enumerate(zip(gt_boxes_raw_list, gt_labels_list)):
        if gt_ignore_boxes_raw_list is None:
            keep = labels == BUS_CLASS_ID
            if not keep.any():
                continue
            ignore_boxes = boxes[keep].to(device)
        else:
            ignore_boxes = gt_ignore_boxes_raw_list[batch_idx].to(device)

        if ignore_boxes.numel() == 0:
            continue

        _, scope_shape = get_rae_scope_start_and_shape(
            scope_modes[batch_idx],
            full_rae_shapes[batch_idx],
        )
        scope_h = int(scope_shape[0])
        scope_w = int(scope_shape[1])
        scale_y = 0.0 if height <= 1 or scope_h <= 1 else float(height - 1) / float(scope_h - 1)
        scale_x = 0.0 if width <= 1 or scope_w <= 1 else float(width - 1) / float(scope_w - 1)

        for box in ignore_boxes:
            center_y = float(box[0].item()) * scale_y
            center_x = float(box[1].item()) * scale_x
            half_h = (
                float(box[3].abs().clamp(min=1e-4).item())
                * scale_y
                * 0.5
                * float(ignore_expand_ratio)
            ) + float(ignore_margin)
            half_w = (
                float(box[4].abs().clamp(min=1e-4).item())
                * scale_x
                * 0.5
                * float(ignore_expand_ratio)
            ) + float(ignore_margin)

            y0 = max(0, int(math.floor(center_y - half_h)))
            y1 = min(height - 1, int(math.ceil(center_y + half_h)))
            x0 = max(0, int(math.floor(center_x - half_w)))
            x1 = min(width - 1, int(math.ceil(center_x + half_w)))
            if y0 > y1 or x0 > x1:
                continue
            mask[batch_idx, 0, y0:y1 + 1, x0:x1 + 1] = 1.0

    return mask


def masked_heatmap_focal_loss(logits, targets, valid_mask, alpha=2.0, beta=4.0):
    pred = logits.sigmoid().clamp(min=1e-4, max=1.0 - 1e-4)
    valid_mask = valid_mask.to(dtype=targets.dtype)

    pos_inds = targets.eq(1.0).float() * valid_mask
    neg_inds = targets.lt(1.0).float() * valid_mask
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


def masked_radenet_continuous_focal_loss(heatmap, gaussian_map, valid_mask, alpha=2.0, gamma=4.0):
    eps = 1e-6
    valid_mask = valid_mask.to(dtype=gaussian_map.dtype)
    pos_weights = gaussian_map.eq(1.0).to(dtype=gaussian_map.dtype) * valid_mask
    neg_weights = torch.pow(1.0 - gaussian_map, gamma) * valid_mask

    heatmap = torch.clamp(heatmap, min=eps, max=1.0 - eps)
    heatmap_complement = torch.clamp(1.0 - heatmap, min=eps, max=1.0 - eps)

    pos_loss = -torch.log(heatmap) * torch.pow(heatmap_complement, alpha) * pos_weights
    neg_loss = -torch.log(heatmap_complement) * torch.pow(heatmap, alpha) * neg_weights

    num_pos = pos_weights.sum()
    if num_pos > 0:
        return (pos_loss.sum() + neg_loss.sum()) / (num_pos + eps)

    valid_count = valid_mask.sum()
    if valid_count > 0:
        return neg_loss.sum() / (valid_count + eps)
    return torch.mean(neg_loss)


def _normalize_radenet_loss_term(loss_term):
    return loss_term / torch.clamp(loss_term.detach(), min=1e-6)


def _gather_regression_at_centers(regression_map, center_indices):
    if center_indices.numel() == 0:
        return regression_map.new_zeros((0, regression_map.shape[0]))
    y_idx = center_indices[:, 0].long()
    x_idx = center_indices[:, 1].long()
    return regression_map[:, y_idx, x_idx].transpose(0, 1)


def sedan_only_centerpoint_detection_loss(
        outputs,
        gt_boxes_list,
        gt_labels_list,
        gt_ignore_boxes_list=None,
        scope_modes=None,
        full_rae_shapes=None,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        giou_loss_weight=2.0,
        quality_loss_weight=0.25,
        heatmap_radius=3,
        object_ignore_margin=1.0,
        object_ignore_expand_ratio=1.0,
    ):
    if "qfl_cls_logits" in outputs:
        raise ValueError(
            "sedan_only_centerpoint_detection_loss currently supports CenterPoint-style cls_logits heads only."
        )

    cls_logits = outputs["cls_logits"][:, :1]
    sedan_boxes_list, sedan_labels_list = _filter_sedan_targets(
        gt_boxes_list=gt_boxes_list,
        gt_labels_list=gt_labels_list,
    )

    heatmap_targets, reg_targets, reg_mask = build_centerpoint_targets(
        gt_boxes=sedan_boxes_list,
        gt_labels=sedan_labels_list,
        cls_logits=cls_logits,
        num_classes=1,
        radius=heatmap_radius,
        reg_reference=outputs["center_offset"],
    )

    object_ignore_mask = _build_object_ignore_mask(
        gt_boxes_list=gt_boxes_list,
        gt_labels_list=gt_labels_list,
        height=cls_logits.shape[-2],
        width=cls_logits.shape[-1],
        device=cls_logits.device,
        ignore_margin=object_ignore_margin,
        ignore_expand_ratio=object_ignore_expand_ratio,
        gt_ignore_boxes_list=gt_ignore_boxes_list,
    )
    cls_valid_mask = 1.0 - object_ignore_mask
    cls_valid_mask = torch.maximum(cls_valid_mask, heatmap_targets.gt(0.0).float())

    cls_loss = masked_heatmap_focal_loss(
        logits=cls_logits,
        targets=heatmap_targets,
        valid_mask=cls_valid_mask,
    )

    pred_center_offset = outputs["center_offset"].sigmoid()
    pred_center_height = outputs["center_height"].sigmoid()
    pred_size = outputs["size"].sigmoid()
    pred_yaw = F.normalize(outputs["yaw"], dim=1)

    offset_loss = masked_l1_loss(
        pred=pred_center_offset,
        target=reg_targets["center_offset"],
        mask=reg_mask,
    )
    height_loss = masked_l1_loss(
        pred=pred_center_height,
        target=reg_targets["center_height"],
        mask=reg_mask,
    )
    size_loss = masked_l1_loss(
        pred=pred_size,
        target=reg_targets["size"],
        mask=reg_mask,
    )
    yaw_loss = masked_l1_loss(
        pred=pred_yaw,
        target=reg_targets["yaw"],
        mask=reg_mask,
    )
    # Old axis-aligned GIoU path, kept for quick rollback:
    # giou_loss = centerpoint_giou_loss(
    #     outputs=outputs,
    #     target_boxes=reg_targets["box"],
    #     mask=reg_mask,
    # )
    giou_loss = centerpoint_gwd_loss(
        outputs=outputs,
        target_boxes=reg_targets["box"],
        mask=reg_mask,
        scope_modes=scope_modes,
        full_rae_shapes=full_rae_shapes,
    )
    quality_loss = centerpoint_quality_loss(
        outputs=outputs,
        target_boxes=reg_targets["box"],
        mask=reg_mask,
    )

    box_loss = (
        offset_loss
        + height_loss
        + size_loss
        + yaw_loss
        + (giou_loss_weight * giou_loss)
    )
    total_loss = (
        (box_loss_weight * box_loss)
        + (cls_loss_weight * cls_loss)
        + (quality_loss_weight * quality_loss)
    )

    return total_loss, {
        "total_loss": total_loss.item(),
        "box_loss": box_loss.item(),
        "cls_loss": cls_loss.item(),
        "heatmap_loss": cls_loss.item(),
        "quality_loss": quality_loss.item(),
        "offset_loss": offset_loss.item(),
        "height_loss": height_loss.item(),
        "size_loss": size_loss.item(),
        "yaw_loss": yaw_loss.item(),
        "giou_loss": giou_loss.item(),
        "gwd_loss": giou_loss.item(),
        "num_center_targets": int(reg_mask.sum().item()),
        "ignore_pixels": int(object_ignore_mask.sum().item()),
    }


def sedan_only_radenet_detection_loss(
        outputs,
        gt_boxes_raw_list,
        gt_labels_list,
        scope_modes,
        full_rae_shapes,
        gt_ignore_boxes_raw_list=None,
        num_classes=1,
        gaussian_sigma=3.0,
        object_ignore_margin=1.0,
        object_ignore_expand_ratio=1.0,
    ):
    if "heatmap" not in outputs or "regression" not in outputs:
        raise KeyError("Sedan-only RADE-Net loss requires 'heatmap' and 'regression' outputs.")

    heatmap = outputs["heatmap"][:, :num_classes]
    regression = outputs["regression"]
    batch_size, _, height, width = heatmap.shape
    device = heatmap.device

    sedan_boxes_raw_list, sedan_labels_list = _filter_sedan_raw_targets(
        gt_boxes_raw_list=gt_boxes_raw_list,
        gt_labels_list=gt_labels_list,
    )
    gaussian_targets = build_radenet_gaussian_heatmap(
        gt_boxes_raw_list=sedan_boxes_raw_list,
        gt_labels_list=sedan_labels_list,
        scope_modes=scope_modes,
        full_rae_shapes=full_rae_shapes,
        batch_size=batch_size,
        num_classes=num_classes,
        height=height,
        width=width,
        device=device,
        sigma=gaussian_sigma,
    )

    object_ignore_mask = _build_object_ignore_mask_raw(
        gt_boxes_raw_list=gt_boxes_raw_list,
        gt_labels_list=gt_labels_list,
        scope_modes=scope_modes,
        full_rae_shapes=full_rae_shapes,
        height=height,
        width=width,
        device=device,
        ignore_margin=object_ignore_margin,
        ignore_expand_ratio=object_ignore_expand_ratio,
        gt_ignore_boxes_raw_list=gt_ignore_boxes_raw_list,
    )
    cls_valid_mask = 1.0 - object_ignore_mask
    cls_valid_mask = torch.maximum(cls_valid_mask, gaussian_targets.gt(0.0).float())
    heatmap_loss = masked_radenet_continuous_focal_loss(
        heatmap=heatmap,
        gaussian_map=gaussian_targets,
        valid_mask=cls_valid_mask,
    )

    gwd_losses = []
    smooth_l1_losses = []
    for batch_idx in range(batch_size):
        raw_boxes = sedan_boxes_raw_list[batch_idx].to(device)
        if raw_boxes.numel() == 0:
            zero = regression.new_tensor(0.0)
            gwd_losses.append(zero)
            smooth_l1_losses.append(zero)
            continue

        gt_metric_boxes = raw_local_rae_boxes_to_metric_boxes(
            raw_boxes=raw_boxes,
            scope_mode=scope_modes[batch_idx],
            full_rae_shape=full_rae_shapes[batch_idx],
        )

        _, scope_shape = get_rae_scope_start_and_shape(scope_modes[batch_idx], full_rae_shapes[batch_idx])
        scope_h = int(scope_shape[0])
        scope_w = int(scope_shape[1])
        scale_y = 0.0 if height <= 1 or scope_h <= 1 else float(height - 1) / float(scope_h - 1)
        scale_x = 0.0 if width <= 1 or scope_w <= 1 else float(width - 1) / float(scope_w - 1)

        center_y = torch.round(raw_boxes[:, 0] * scale_y).clamp(min=0.0, max=float(height - 1))
        center_x = torch.round(raw_boxes[:, 1] * scale_x).clamp(min=0.0, max=float(width - 1))
        center_indices = torch.stack([center_y, center_x], dim=-1)

        pred_reg = _gather_regression_at_centers(regression[batch_idx], center_indices)
        pred_metric_boxes = regression_cell_to_metric_box(
            pred_reg=pred_reg,
            y_idx=center_y,
            x_idx=center_x,
            feature_shape=(height, width),
            scope_mode=scope_modes[batch_idx],
            full_rae_shape=full_rae_shapes[batch_idx],
        )

        pred_gwd_boxes = torch.cat(
            [
                pred_metric_boxes[:, :6],
                F.normalize(pred_reg[:, 6:8], dim=-1),
            ],
            dim=-1,
        )
        _, gwd_loss = gaussian_wasserstein_distance_batch(pred_gwd_boxes, gt_metric_boxes)
        gwd_losses.append(gwd_loss.mean())
        smooth_l1_losses.append(F.smooth_l1_loss(pred_gwd_boxes, gt_metric_boxes, reduction="mean"))

    gwd_loss = torch.mean(torch.stack(gwd_losses)) if gwd_losses else regression.new_tensor(0.0)
    smooth_l1_loss = (
        torch.mean(torch.stack(smooth_l1_losses))
        if smooth_l1_losses else regression.new_tensor(0.0)
    )

    backprop_loss = (
        2.0 * _normalize_radenet_loss_term(heatmap_loss)
        + _normalize_radenet_loss_term(gwd_loss)
        + _normalize_radenet_loss_term(smooth_l1_loss)
    )
    total_loss = heatmap_loss + gwd_loss + smooth_l1_loss

    return backprop_loss, {
        "total_loss": total_loss.item(),
        "box_loss": (gwd_loss + smooth_l1_loss).item(),
        "cls_loss": heatmap_loss.item(),
        "heatmap_loss": heatmap_loss.item(),
        "gwd_loss": gwd_loss.item(),
        "l1_loss": smooth_l1_loss.item(),
        "ignore_pixels": int(object_ignore_mask.sum().item()),
    }
