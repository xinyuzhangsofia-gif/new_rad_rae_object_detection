"""SimOTA foreground assignment for YOLOX-style detector training."""

import torch
import torch.nn.functional as F

from data.rotated_bev import pairwise_rotated_bev_iou


def pairwise_center_candidate_mask(
        grid_centers,
        gt_grid_centers,
        center_radius=2.5,
    ):
    """Locate candidate cells using R-A feature indices, not box geometry."""
    if gt_grid_centers.numel() == 0:
        return torch.zeros((grid_centers.shape[0], 0), dtype=torch.bool, device=grid_centers.device)

    delta = (grid_centers[:, None, :] - gt_grid_centers[None, :, :]).abs()
    return (delta <= center_radius).all(dim=-1)


def simota_assign(
        pred_boxes,
        cls_logits,
        objectness_logits,
        grid_centers,
        gt_grid_centers,
        gt_boxes,
        gt_labels,
        num_classes,
        center_radius=2.5,
        candidate_topk=10,
    ):
    valid_gt = (gt_labels >= 0) & (gt_labels < num_classes)
    gt_boxes = gt_boxes[valid_gt]
    gt_grid_centers = gt_grid_centers[valid_gt]
    gt_labels = gt_labels[valid_gt]
    num_gt = gt_boxes.shape[0]
    if num_gt == 0:
        empty = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
        return empty, empty, empty, pred_boxes.new_empty(0)

    candidate_pair_mask = pairwise_center_candidate_mask(
        grid_centers=grid_centers,
        gt_grid_centers=gt_grid_centers,
        center_radius=center_radius,
    )
    candidate_mask = candidate_pair_mask.any(dim=1)
    if candidate_mask.sum() == 0:
        candidate_mask = torch.ones_like(candidate_mask)
        candidate_pair_mask = torch.ones(
            (pred_boxes.shape[0], num_gt),
            dtype=torch.bool,
            device=pred_boxes.device,
        )

    candidate_indices = candidate_mask.nonzero(as_tuple=False).squeeze(1)
    candidate_boxes = pred_boxes[candidate_indices]
    candidate_cls_logits = cls_logits[candidate_indices]
    candidate_obj_logits = objectness_logits[candidate_indices]
    pair_candidate_mask = candidate_pair_mask[candidate_indices]

    pair_ious = pairwise_rotated_bev_iou(candidate_boxes, gt_boxes)
    iou_cost = -torch.log(pair_ious.clamp(min=1e-8))

    gt_onehot = F.one_hot(gt_labels, num_classes=num_classes).float()
    cls_prob = (
        candidate_cls_logits.sigmoid().unsqueeze(1)
        * candidate_obj_logits.sigmoid().view(-1, 1, 1)
    ).sqrt().clamp(min=1e-4, max=1.0 - 1e-4)
    cls_prob = cls_prob.expand(-1, num_gt, -1)
    cls_targets = gt_onehot.unsqueeze(0).expand(candidate_boxes.shape[0], num_gt, num_classes)
    cls_cost = F.binary_cross_entropy(
        cls_prob,
        cls_targets,
        reduction="none",
    ).sum(dim=-1)

    cost = cls_cost + (3.0 * iou_cost)
    cost = cost + (~pair_candidate_mask).float() * 100000.0

    matching_matrix = torch.zeros_like(cost, dtype=torch.bool)
    dynamic_ks = torch.clamp(
        pair_ious.topk(k=min(candidate_topk, pair_ious.shape[0]), dim=0).values.sum(dim=0).int(),
        min=1,
    )
    for gt_idx in range(num_gt):
        num_match = int(dynamic_ks[gt_idx].item())
        _, pos_idx = torch.topk(
            cost[:, gt_idx],
            k=min(num_match, cost.shape[0]),
            largest=False,
        )
        matching_matrix[pos_idx, gt_idx] = True

    anchor_matching_gt = matching_matrix.sum(dim=1)
    if (anchor_matching_gt > 1).any():
        multi_match = anchor_matching_gt > 1
        _, min_cost_gt = cost[multi_match].min(dim=1)
        matching_matrix[multi_match] = False
        matching_matrix[multi_match, min_cost_gt] = True

    foreground = matching_matrix.sum(dim=1) > 0
    matched_pred_indices = candidate_indices[foreground]
    if matched_pred_indices.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
        return empty, empty, empty, pred_boxes.new_empty(0)

    matched_gt_indices = matching_matrix[foreground].float().argmax(dim=1)
    matched_labels = gt_labels[matched_gt_indices]
    matched_ious = (matching_matrix[foreground].float() * pair_ious[foreground]).sum(dim=1)
    return matched_pred_indices, matched_gt_indices, matched_labels, matched_ious
