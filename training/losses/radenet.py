"""RADE-Net heatmap, regression, normalization, and result contract."""

import torch
import torch.nn.functional as F

from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    require_cartesian_data,
)
from data.coordinates import get_rae_scope_start_and_shape
from data.geometry import (
    feature_indices_to_cartesian_xy,
)
from training.losses import DEFAULT_NUM_CLASSES
from training.losses.common import build_raw_ignore_mask
from training.losses.gwd import gaussian_wasserstein_distance_batch
from training.losses.targets import build_radenet_gaussian_heatmap


def radenet_continuous_focal_loss(
        heatmap,
        gaussian_map,
        valid_mask=None,
        alpha=2.0,
        gamma=4.0,
    ):
    eps = 1e-6
    if valid_mask is None:
        valid_mask = 1.0
    else:
        valid_mask = valid_mask.to(dtype=gaussian_map.dtype)
    pos_weights = gaussian_map.eq(1.0).to(dtype=gaussian_map.dtype) * valid_mask
    neg_weights = torch.pow(1.0 - gaussian_map, gamma) * valid_mask

    heatmap = torch.clamp(heatmap, min=eps, max=1.0 - eps)
    heatmap_complement = torch.clamp(1.0 - heatmap, min=eps, max=1.0 - eps)

    pos_loss = -torch.log(heatmap) * torch.pow(heatmap_complement, alpha) * pos_weights
    neg_loss = -torch.log(heatmap_complement) * torch.pow(heatmap, alpha) * neg_weights
    loss = pos_loss + neg_loss

    num_pos = pos_weights.sum()
    if num_pos > 0:
        return torch.sum(loss) / (num_pos + eps)
    if torch.is_tensor(valid_mask):
        valid_count = valid_mask.sum()
        if valid_count > 0:
            return neg_loss.sum() / (valid_count + eps)
    return torch.mean(neg_loss)


def _normalize_radenet_loss_term(loss_term):
    """Apply the official RADE-Net per-term loss normalization.

    The official implementation uses ``loss / loss.detach().mean()`` for
    each scalar loss term.  The detached denominator keeps the normalization
    from changing the forward value's gradient dependence.  This project has
    empty radar frames, so retain a finite zero-term guard; the guard is only
    active when the term is effectively zero.
    """
    detached_mean = loss_term.detach().mean()
    if torch.abs(detached_mean) <= 1e-6:
        return loss_term
    return loss_term / detached_mean


def _gather_regression_at_centers(regression_map, center_indices):
    if center_indices.numel() == 0:
        return regression_map.new_zeros((0, regression_map.shape[0]))
    y_idx = center_indices[:, 0].long()
    x_idx = center_indices[:, 1].long()
    return regression_map[:, y_idx, x_idx].transpose(0, 1)


def radenet_detection_loss(
        outputs,
        gt_boxes_raw_list,
        gt_labels_list,
        scope_modes,
        full_rae_shapes,
        gt_metric_boxes_list=None,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
        gt_ignore_boxes_raw_list=None,
        num_classes=DEFAULT_NUM_CLASSES,
        gaussian_sigma=3.0,
        ignore_mask_margin=1.0,
        ignore_mask_expand_ratio=1.0,
    ):
    if "heatmap" not in outputs or "regression" not in outputs:
        raise KeyError("RADE-Net loss requires model outputs to contain 'heatmap' and 'regression'.")
    require_cartesian_data(box_coordinate_mode)
    if gt_metric_boxes_list is None:
        raise ValueError(
            "Cartesian RADE-Net loss requires exact gt_metric_boxes_list."
        )

    heatmap = outputs["heatmap"][:, :num_classes]
    regression = outputs["regression"]
    batch_size, _, height, width = heatmap.shape
    device = heatmap.device

    gaussian_targets = build_radenet_gaussian_heatmap(
        gt_boxes_raw_list=gt_boxes_raw_list,
        gt_labels_list=gt_labels_list,
        scope_modes=scope_modes,
        full_rae_shapes=full_rae_shapes,
        batch_size=batch_size,
        num_classes=num_classes,
        height=height,
        width=width,
        device=device,
        sigma=gaussian_sigma,
    )
    if gt_ignore_boxes_raw_list is None:
        ignore_mask = torch.zeros((batch_size, 1, height, width), device=device)
    else:
        ignore_mask = build_raw_ignore_mask(
            gt_ignore_boxes_raw_list=gt_ignore_boxes_raw_list,
            scope_modes=scope_modes,
            full_rae_shapes=full_rae_shapes,
            height=height,
            width=width,
            device=device,
            ignore_margin=ignore_mask_margin,
            ignore_expand_ratio=ignore_mask_expand_ratio,
        )
    valid_mask = 1.0 - ignore_mask
    valid_mask = torch.maximum(valid_mask, gaussian_targets.amax(dim=1, keepdim=True).gt(0.0).float())
    heatmap_loss = radenet_continuous_focal_loss(
        heatmap=heatmap,
        gaussian_map=gaussian_targets,
        valid_mask=valid_mask,
    )

    gwd_losses = []
    smooth_l1_losses = []
    for batch_idx in range(batch_size):
        raw_boxes = gt_boxes_raw_list[batch_idx].to(device)
        labels = gt_labels_list[batch_idx].to(device)
        if raw_boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                "RADE-Net raw GT box/label count mismatch at batch "
                f"{batch_idx}: {raw_boxes.shape[0]} vs {labels.shape[0]}"
            )
        valid = (labels >= 0) & (labels < num_classes)
        raw_boxes = raw_boxes[valid]
        if raw_boxes.numel() == 0:
            zero = regression.new_tensor(0.0)
            gwd_losses.append(zero)
            smooth_l1_losses.append(zero)
            continue

        metric_boxes = gt_metric_boxes_list[batch_idx].to(device)
        if metric_boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                "RADE-Net metric GT box/label count mismatch at batch "
                f"{batch_idx}: {metric_boxes.shape[0]} vs {labels.shape[0]}"
            )
        metric_boxes = metric_boxes[valid]
        yaw = metric_boxes[:, 6]
        gt_metric_boxes = torch.cat(
            [
                metric_boxes[:, :6],
                torch.sin(yaw).unsqueeze(-1),
                torch.cos(yaw).unsqueeze(-1),
            ],
            dim=-1,
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
        base_x, base_y = feature_indices_to_cartesian_xy(
            y_idx=center_y,
            x_idx=center_x,
            feature_shape=(height, width),
            scope_mode=scope_modes[batch_idx],
            full_rae_shape=full_rae_shapes[batch_idx],
        )
        # Original RADE-Net regression target:
        # [base_x + dx, base_y + dy, dz, l, w, h, sin(yaw), cos(yaw)].
        pred_gwd_boxes = torch.stack(
            [
                base_x + pred_reg[:, 0],
                base_y + pred_reg[:, 1],
                pred_reg[:, 2],
                pred_reg[:, 3],
                pred_reg[:, 4],
                pred_reg[:, 5],
                pred_reg[:, 6],
                pred_reg[:, 7],
            ],
            dim=-1,
        )
        gt_gwd_boxes = gt_metric_boxes
        _, gwd_loss = gaussian_wasserstein_distance_batch(pred_gwd_boxes, gt_gwd_boxes)
        gwd_losses.append(gwd_loss.mean())
        smooth_l1_losses.append(F.smooth_l1_loss(pred_gwd_boxes, gt_gwd_boxes, reduction="mean"))

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
        "ignore_pixels": int(ignore_mask.sum().item()),
    }
