"""Cartesian YOLOX loss with R-A cell localization and SimOTA assignment."""

import torch
import torch.nn.functional as F

from data.geometry import raw_local_centers_to_feature_indices
from training.losses import DEFAULT_NUM_CLASSES
from training.losses.common import build_raw_ignore_mask
from training.losses.gwd import gaussian_wasserstein_distance_batch
from training.losses.matching import simota_assign
from training.yolox_utils import decode_yolox_boxes, yolox_grid_centers


def _metric_boxes_to_gwd_boxes(boxes):
    """Pack metric box geometry as [x,y,z,l,w,h,sin(yaw),cos(yaw)]."""
    return torch.cat(
        (
            boxes[:, :3],
            boxes[:, 3:6].abs().clamp(min=1e-3),
            torch.sin(boxes[:, 6:7]),
            torch.cos(boxes[:, 6:7]),
        ),
        dim=1,
    )


def yolox_detection_loss(
        outputs,
        gt_metric_boxes_list,
        gt_boxes_raw_list,
        gt_labels_list,
        scope_modes,
        full_rae_shapes,
        gt_ignore_boxes_raw_list=None,
        num_classes=DEFAULT_NUM_CLASSES,
        box_loss_weight=5.0,
        obj_loss_weight=1.0,
        cls_loss_weight=1.0,
        l1_loss_weight=1.0,
        ignore_mask_margin=1.0,
        ignore_mask_expand_ratio=1.0,
    ):
    cls_logits_map = outputs["cls_logits"][:, :num_classes]
    objectness_logits_map = outputs["objectness_logits"]
    pred_boxes = decode_yolox_boxes(outputs, scope_modes, full_rae_shapes)
    grid_centers, grid_h, grid_w = yolox_grid_centers(outputs)
    # Keep the established SimOTA candidate window relative to feature-grid density.
    grid_scale = max(grid_h, grid_w) / 8.0
    center_radius = 2.5 * grid_scale
    candidate_topk = max(10, int(round(10 * grid_scale)))

    batch_size, num_preds, _ = pred_boxes.shape
    if not (
        len(gt_metric_boxes_list) == len(gt_boxes_raw_list)
        == len(gt_labels_list) == batch_size
    ):
        raise ValueError("YOLOX requires metric boxes, raw centers, and labels for each sample.")
    cls_logits = cls_logits_map.flatten(start_dim=2).transpose(1, 2)
    objectness_logits = objectness_logits_map.flatten(start_dim=2).squeeze(1)

    total_box_loss = pred_boxes.new_tensor(0.0)
    total_obj_loss = pred_boxes.new_tensor(0.0)
    total_cls_loss = pred_boxes.new_tensor(0.0)
    total_l1_loss = pred_boxes.new_tensor(0.0)
    total_fg = 0
    total_ignore_pixels = 0

    if gt_ignore_boxes_raw_list is None:
        obj_valid_masks = torch.ones((batch_size, num_preds), device=pred_boxes.device)
    else:
        ignore_mask = build_raw_ignore_mask(
            gt_ignore_boxes_raw_list=gt_ignore_boxes_raw_list,
            scope_modes=scope_modes,
            full_rae_shapes=full_rae_shapes,
            height=grid_h,
            width=grid_w,
            device=pred_boxes.device,
            ignore_margin=ignore_mask_margin,
            ignore_expand_ratio=ignore_mask_expand_ratio,
        )
        total_ignore_pixels = int(ignore_mask.sum().item())
        obj_valid_masks = 1.0 - ignore_mask.flatten(start_dim=2).squeeze(1)

    for batch_idx in range(batch_size):
        gt_boxes = gt_metric_boxes_list[batch_idx].to(pred_boxes.device)
        raw_boxes = gt_boxes_raw_list[batch_idx].to(pred_boxes.device)
        gt_labels = gt_labels_list[batch_idx].to(pred_boxes.device)
        if gt_boxes.shape[0] != raw_boxes.shape[0] or gt_boxes.shape[0] != gt_labels.shape[0]:
            raise ValueError(f"YOLOX GT box/center/label count mismatch at batch {batch_idx}.")
        gt_grid_centers = raw_local_centers_to_feature_indices(
            raw_boxes,
            feature_shape=(grid_h, grid_w),
            scope_mode=scope_modes[batch_idx],
            full_rae_shape=full_rae_shapes[batch_idx],
        )
        obj_targets = torch.zeros((num_preds,), device=pred_boxes.device)

        matched_pred_idx, matched_gt_idx, matched_labels, matched_ious = simota_assign(
            pred_boxes=pred_boxes[batch_idx],
            cls_logits=cls_logits[batch_idx],
            objectness_logits=objectness_logits[batch_idx],
            grid_centers=grid_centers[batch_idx],
            gt_grid_centers=gt_grid_centers,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            num_classes=num_classes,
            center_radius=center_radius,
            candidate_topk=candidate_topk,
        )

        num_fg = int(matched_pred_idx.numel())
        if num_fg > 0:
            total_fg += num_fg
            obj_targets[matched_pred_idx] = 1.0
            obj_valid_masks[batch_idx, matched_pred_idx] = 1.0

            matched_gt_boxes = gt_boxes[matched_gt_idx]
            pred_pos_boxes = pred_boxes[batch_idx, matched_pred_idx]
            _, gwd_loss = gaussian_wasserstein_distance_batch(
                _metric_boxes_to_gwd_boxes(pred_pos_boxes),
                _metric_boxes_to_gwd_boxes(matched_gt_boxes),
            )
            total_box_loss = total_box_loss + gwd_loss.sum()

            cls_targets = F.one_hot(matched_labels, num_classes=num_classes).float()
            cls_targets = cls_targets * matched_ious.detach().unsqueeze(1)
            total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                cls_logits[batch_idx, matched_pred_idx],
                cls_targets,
                reduction="sum",
            )
            total_l1_loss = total_l1_loss + F.l1_loss(
                pred_pos_boxes, matched_gt_boxes, reduction="sum"
            )

        obj_loss = F.binary_cross_entropy_with_logits(
            objectness_logits[batch_idx], obj_targets, reduction="none"
        )
        total_obj_loss = total_obj_loss + (obj_loss * obj_valid_masks[batch_idx]).sum()

    normalizer = max(total_fg, 1)
    box_loss = total_box_loss / normalizer
    obj_loss = total_obj_loss / normalizer
    cls_loss = total_cls_loss / normalizer
    l1_loss = total_l1_loss / normalizer
    total_loss = (
        (box_loss_weight * box_loss)
        + (obj_loss_weight * obj_loss)
        + (cls_loss_weight * cls_loss)
        + (l1_loss_weight * l1_loss)
    )

    return total_loss, {
        "total_loss": total_loss.item(),
        "box_loss": box_loss.item(),
        "cls_loss": cls_loss.item(),
        "obj_loss": obj_loss.item(),
        "l1_loss": l1_loss.item(),
        "gwd_loss": box_loss.item(),
        "num_center_targets": total_fg,
        "ignore_pixels": total_ignore_pixels,
    }
