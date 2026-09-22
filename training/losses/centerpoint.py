"""Cartesian CenterPoint loss implementation."""

import torch
import torch.nn.functional as F

from data.geometry import (
    centerpoint_outputs_to_metric_regression,
    regression_cell_to_metric_box,
)
from training.losses.common import DEFAULT_NUM_CLASSES, build_raw_ignore_mask, heatmap_focal_loss
from training.losses.gwd import gaussian_wasserstein_distance_batch
from training.losses.targets import build_cartesian_centerpoint_targets


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
