"""Polar and Cartesian CenterPoint loss implementations."""

import torch
import torch.nn.functional as F

from data.geometry import (
    centerpoint_outputs_to_metric_regression,
    regression_cell_to_metric_box,
)
from training_utils.loss_components import DEFAULT_NUM_CLASSES
from training_utils.loss_components.common import (
    boxes_3d_to_ra_xyxy,
    build_normalized_ignore_mask,
    build_raw_ignore_mask,
    heatmap_focal_loss,
    masked_l1_loss,
    pairwise_box_iou_2d,
)
from training_utils.loss_components.gwd import (
    gaussian_wasserstein_distance_batch,
    normalized_rae_boxes_to_gwd_boxes,
)
from training_utils.loss_components.targets import (
    build_cartesian_centerpoint_targets,
    build_centerpoint_targets,
)


def cartesian_centerpoint_detection_loss(
        outputs,
        gt_boxes_raw_list,
        gt_metric_boxes_list,
        gt_labels_list,
        scope_modes,
        full_rae_shapes,
        gt_ignore_boxes_raw_list=None,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        gwd_loss_weight=2.0,
        heatmap_radius=3,
        num_classes=DEFAULT_NUM_CLASSES,
        ignore_mask_margin=1.0,
        ignore_mask_expand_ratio=1.0,
    ):
    """CenterPoint-style loss for metric Cartesian box branches.

    This branch intentionally does not use RADE-Net's detached-mean
    normalization.  Heatmap and each regression component use ordinary
    masked means over valid targets; GWD is added with its fixed configured
    weight.  The ``radenet_detection_loss`` function remains separate and
    preserves the original RADE-Net normalization for ``loss_mode``
    ``"radenet"``.
    """
    cls_logits = outputs["cls_logits"][:, :num_classes]
    regression_map = centerpoint_outputs_to_metric_regression(outputs)
    heatmap_targets, metric_targets, reg_mask = (
        build_cartesian_centerpoint_targets(
            gt_boxes_raw_list=gt_boxes_raw_list,
            gt_metric_boxes_list=gt_metric_boxes_list,
            gt_labels_list=gt_labels_list,
            cls_logits=cls_logits,
            regression_map=regression_map,
            scope_modes=scope_modes,
            full_rae_shapes=full_rae_shapes,
            num_classes=num_classes,
            radius=heatmap_radius,
        )
    )

    if gt_ignore_boxes_raw_list is None:
        ignore_mask = torch.zeros(
            (
                cls_logits.shape[0],
                1,
                cls_logits.shape[-2],
                cls_logits.shape[-1],
            ),
            device=cls_logits.device,
        )
    else:
        ignore_mask = build_raw_ignore_mask(
            gt_ignore_boxes_raw_list=gt_ignore_boxes_raw_list,
            scope_modes=scope_modes,
            full_rae_shapes=full_rae_shapes,
            height=cls_logits.shape[-2],
            width=cls_logits.shape[-1],
            device=cls_logits.device,
            ignore_margin=ignore_mask_margin,
            ignore_expand_ratio=ignore_mask_expand_ratio,
        )
    cls_valid_mask = 1.0 - ignore_mask
    cls_valid_mask = torch.maximum(
        cls_valid_mask,
        heatmap_targets.amax(dim=1, keepdim=True).gt(0.0).float(),
    )
    cls_loss = heatmap_focal_loss(
        logits=cls_logits,
        targets=heatmap_targets,
        valid_mask=cls_valid_mask,
    )

    center_losses = []
    height_losses = []
    size_losses = []
    yaw_losses = []
    gwd_losses = []
    positive_mask = reg_mask.squeeze(1).bool()

    for batch_idx in range(cls_logits.shape[0]):
        positive_b = positive_mask[batch_idx]
        if positive_b.sum() == 0:
            continue

        y_idx, x_idx = positive_b.nonzero(as_tuple=True)
        pred_reg = regression_map[batch_idx].permute(1, 2, 0)[positive_b]
        pred_metric = regression_cell_to_metric_box(
            pred_reg=pred_reg,
            y_idx=y_idx.to(pred_reg.dtype),
            x_idx=x_idx.to(pred_reg.dtype),
            feature_shape=regression_map.shape[-2:],
            scope_mode=scope_modes[batch_idx],
            full_rae_shape=full_rae_shapes[batch_idx],
        )
        gt_metric = metric_targets[batch_idx].permute(1, 2, 0)[positive_b]
        pred_yaw_vector = F.normalize(pred_reg[:, 6:8], dim=-1)
        gt_yaw_vector = torch.stack(
            [
                torch.sin(gt_metric[:, 6]),
                torch.cos(gt_metric[:, 6]),
            ],
            dim=-1,
        )

        center_losses.append(
            F.smooth_l1_loss(
                pred_metric[:, :2],
                gt_metric[:, :2],
                reduction="mean",
            )
        )
        height_losses.append(
            F.smooth_l1_loss(
                pred_metric[:, 2:3],
                gt_metric[:, 2:3],
                reduction="mean",
            )
        )
        size_losses.append(
            F.smooth_l1_loss(
                pred_metric[:, 3:6],
                gt_metric[:, 3:6],
                reduction="mean",
            )
        )
        yaw_losses.append(
            F.smooth_l1_loss(
                pred_yaw_vector,
                gt_yaw_vector,
                reduction="mean",
            )
        )

        pred_gwd_boxes = torch.cat(
            [pred_metric[:, :6], pred_yaw_vector],
            dim=-1,
        )
        gt_gwd_boxes = torch.cat(
            [gt_metric[:, :6], gt_yaw_vector],
            dim=-1,
        )
        _, gwd_values = gaussian_wasserstein_distance_batch(
            pred_gwd_boxes,
            gt_gwd_boxes,
        )
        gwd_losses.append(gwd_values.mean())

    zero = cls_logits.new_tensor(0.0)

    def mean_or_zero(values):
        return torch.stack(values).mean() if values else zero

    offset_loss = mean_or_zero(center_losses)
    height_loss = mean_or_zero(height_losses)
    size_loss = mean_or_zero(size_losses)
    yaw_loss = mean_or_zero(yaw_losses)
    gwd_loss = mean_or_zero(gwd_losses)
    box_loss = (
        offset_loss
        + height_loss
        + size_loss
        + yaw_loss
        + (gwd_loss_weight * gwd_loss)
    )
    total_loss = (
        (box_loss_weight * box_loss)
        + (cls_loss_weight * cls_loss)
    )

    return total_loss, {
        "total_loss": total_loss.item(),
        "box_loss": box_loss.item(),
        "cls_loss": cls_loss.item(),
        "heatmap_loss": cls_loss.item(),
        "offset_loss": offset_loss.item(),
        "height_loss": height_loss.item(),
        "size_loss": size_loss.item(),
        "yaw_loss": yaw_loss.item(),
        "gwd_loss": gwd_loss.item(),
        "num_center_targets": int(reg_mask.sum().item()),
        "ignore_pixels": int(ignore_mask.sum().item()),
    }


def dense_centerpoint_outputs_to_boxes(outputs):
    center_offset = outputs["center_offset"]
    B, _, H, W = center_offset.shape
    device = center_offset.device
    dtype = center_offset.dtype

    y_grid = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)
    x_grid = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)

    center_offset = center_offset.sigmoid()
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


def centerpoint_gwd_loss(
        outputs,
        target_boxes,
        mask,
        scope_modes=None,
        full_rae_shapes=None,
    ):
    positive_mask = mask.squeeze(1).bool()
    if positive_mask.sum() == 0:
        return outputs["cls_logits"].new_tensor(0.0)

    pred_box_map = dense_centerpoint_outputs_to_boxes(outputs)
    losses = []
    batch_size = pred_box_map.shape[0]
    for batch_idx in range(batch_size):
        positive_b = positive_mask[batch_idx]
        if positive_b.sum() == 0:
            continue

        pred_boxes = pred_box_map[batch_idx].permute(1, 2, 0)[positive_b]
        gt_boxes = target_boxes[batch_idx].permute(1, 2, 0)[positive_b]
        scope_mode = None if scope_modes is None else scope_modes[batch_idx]
        full_rae_shape = None if full_rae_shapes is None else full_rae_shapes[batch_idx]

        pred_gwd_boxes = normalized_rae_boxes_to_gwd_boxes(
            boxes=pred_boxes,
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
        )
        gt_gwd_boxes = normalized_rae_boxes_to_gwd_boxes(
            boxes=gt_boxes,
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
        )
        _, gwd_loss = gaussian_wasserstein_distance_batch(pred_gwd_boxes, gt_gwd_boxes)
        losses.append(gwd_loss.mean())

    if len(losses) == 0:
        return outputs["cls_logits"].new_tensor(0.0)
    return torch.stack(losses).mean()


def centerpoint_quality_loss(outputs, target_boxes, mask):
    if "objectness_logits" in outputs:
        objectness_logits = outputs["objectness_logits"]
        if objectness_logits.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                "objectness_logits and regression targets must have the same spatial size, "
                f"got objectness={tuple(objectness_logits.shape)} and mask={tuple(mask.shape)}"
            )
        return F.binary_cross_entropy_with_logits(
            input=objectness_logits,
            target=mask,
            reduction="mean",
        )

    if "quality_logits" not in outputs:
        return outputs["cls_logits"].new_tensor(0.0)

    quality_logits = outputs["quality_logits"]
    if quality_logits.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            "quality_logits and regression targets must have the same spatial size, "
            f"got quality={tuple(quality_logits.shape)} and mask={tuple(mask.shape)}"
        )

    positive_mask = mask.squeeze(1).bool()
    if positive_mask.sum() == 0:
        return outputs["cls_logits"].new_tensor(0.0)

    pred_box_map = dense_centerpoint_outputs_to_boxes(outputs)
    pred_boxes = pred_box_map.permute(0, 2, 3, 1)[positive_mask]
    gt_boxes = target_boxes.permute(0, 2, 3, 1)[positive_mask]

    pred_ra_boxes = boxes_3d_to_ra_xyxy(pred_boxes)
    gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)
    iou_targets = pairwise_box_iou_2d(pred_ra_boxes, gt_ra_boxes).diag()
    iou_targets = iou_targets.clamp(0.0, 1.0).detach()

    quality_pos = quality_logits.squeeze(1)[positive_mask]
    return F.binary_cross_entropy_with_logits(
        input=quality_pos,
        target=iou_targets,
        reduction="mean",
    )


def centerpoint_quality_focal_loss(
        outputs,
        target_boxes,
        target_labels,
        mask,
        valid_mask=None,
        beta=2.0,
    ):
    cls_logits = outputs["cls_logits"]
    pred_scores = cls_logits.sigmoid()
    qfl_targets = torch.zeros_like(cls_logits)
    if valid_mask is None:
        valid_mask = torch.ones_like(cls_logits)
    else:
        valid_mask = valid_mask.to(dtype=cls_logits.dtype).expand_as(cls_logits)

    positive_mask = mask.squeeze(1).bool()
    if positive_mask.sum() > 0:
        pred_box_map = dense_centerpoint_outputs_to_boxes(outputs)
        pred_boxes = pred_box_map.permute(0, 2, 3, 1)[positive_mask]
        gt_boxes = target_boxes.permute(0, 2, 3, 1)[positive_mask]
        labels = target_labels[positive_mask]

        pred_ra_boxes = boxes_3d_to_ra_xyxy(pred_boxes)
        gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)
        iou_targets = pairwise_box_iou_2d(pred_ra_boxes, gt_ra_boxes).diag()
        iou_targets = iou_targets.clamp(0.0, 1.0).detach()

        b_idx, y_idx, x_idx = positive_mask.nonzero(as_tuple=True)
        valid = (labels >= 0) & (labels < cls_logits.shape[1])
        if valid.any():
            qfl_targets[
                b_idx[valid],
                labels[valid],
                y_idx[valid],
                x_idx[valid],
            ] = iou_targets[valid]

    bce_loss = F.binary_cross_entropy_with_logits(
        cls_logits,
        qfl_targets,
        reduction="none",
    )
    modulating_factor = (qfl_targets - pred_scores).abs().pow(beta)
    num_pos = positive_mask.sum().to(cls_logits.dtype)
    return (bce_loss * modulating_factor * valid_mask).sum() / torch.clamp(num_pos, min=1.0)


def centerpoint_detection_loss(
        outputs,
        gt_boxes_list,
        gt_labels_list,
        gt_ignore_boxes_list=None,
        scope_modes=None,
        full_rae_shapes=None,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        gwd_loss_weight=2.0,
        quality_loss_weight=0.25,
        heatmap_radius=3,
        num_classes=DEFAULT_NUM_CLASSES,
        ignore_mask_margin=1.0,
        ignore_mask_expand_ratio=1.0,
    ):
    cls_logits = outputs["cls_logits"]
    heatmap_targets, reg_targets, reg_mask = build_centerpoint_targets(
        gt_boxes=gt_boxes_list,
        gt_labels=gt_labels_list,
        cls_logits=cls_logits,
        num_classes=num_classes,
        radius=heatmap_radius,
        reg_reference=outputs["center_offset"],
    )
    if gt_ignore_boxes_list is None:
        ignore_mask = torch.zeros(
            (cls_logits.shape[0], 1, cls_logits.shape[-2], cls_logits.shape[-1]),
            device=cls_logits.device,
        )
    else:
        ignore_mask = build_normalized_ignore_mask(
            gt_ignore_boxes_list=gt_ignore_boxes_list,
            height=cls_logits.shape[-2],
            width=cls_logits.shape[-1],
            device=cls_logits.device,
            ignore_margin=ignore_mask_margin,
            ignore_expand_ratio=ignore_mask_expand_ratio,
        )
    cls_valid_mask = 1.0 - ignore_mask
    cls_valid_mask = torch.maximum(
        cls_valid_mask,
        heatmap_targets.amax(dim=1, keepdim=True).gt(0.0).float(),
    )

    if "qfl_cls_logits" in outputs:
        cls_loss = centerpoint_quality_focal_loss(
            outputs=outputs,
            target_boxes=reg_targets["box"],
            target_labels=reg_targets["label"],
            mask=reg_mask,
            valid_mask=cls_valid_mask,
        )
    else:
        cls_loss = heatmap_focal_loss(
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
    gwd_loss = centerpoint_gwd_loss(
        outputs=outputs,
        target_boxes=reg_targets["box"],
        mask=reg_mask,
        scope_modes=scope_modes,
        full_rae_shapes=full_rae_shapes,
    )
    quality_loss_active = (
        "quality_logits" in outputs
        or "objectness_logits" in outputs
    )
    quality_loss = None
    if quality_loss_active:
        quality_loss = centerpoint_quality_loss(
            outputs=outputs,
            target_boxes=reg_targets["box"],
            mask=reg_mask
        )

    box_loss = (
        offset_loss
        + height_loss
        + size_loss
        + yaw_loss
        + (gwd_loss_weight * gwd_loss)
    )
    total_loss = (
        (box_loss_weight * box_loss)
        + (cls_loss_weight * cls_loss)
    )
    if quality_loss_active:
        total_loss = total_loss + (quality_loss_weight * quality_loss)

    loss_dict = {
        "total_loss": total_loss.item(),
        "box_loss": box_loss.item(),
        "cls_loss": cls_loss.item(),
        "heatmap_loss": cls_loss.item(),
        "offset_loss": offset_loss.item(),
        "height_loss": height_loss.item(),
        "size_loss": size_loss.item(),
        "yaw_loss": yaw_loss.item(),
        "gwd_loss": gwd_loss.item(),
        "num_center_targets": int(reg_mask.sum().item()),
        "ignore_pixels": int(ignore_mask.sum().item()),
    }
    if quality_loss_active:
        loss_dict["quality_loss"] = quality_loss.item()

    return total_loss, loss_dict
