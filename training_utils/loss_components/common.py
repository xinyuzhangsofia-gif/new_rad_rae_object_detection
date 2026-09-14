"""Common tensor operations shared by detector loss families."""

import math

import torch
import torch.nn.functional as F

from data.coordinates import get_rae_scope_start_and_shape


def boxes_3d_to_ra_xyxy(boxes):
    r = boxes[:, 0]
    a = boxes[:, 1]
    r_w = boxes[:, 3]
    a_w = boxes[:, 4]

    r_min = r - r_w / 2.0
    r_max = r + r_w / 2.0
    a_min = a - a_w / 2.0
    a_max = a + a_w / 2.0

    return torch.stack([r_min, a_min, r_max, a_max], dim=-1)


def pairwise_box_iou_2d(boxes1, boxes2):
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    left_top = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def gaussian2d(radius, sigma=None, device="cpu"):
    diameter = 2 * radius + 1
    if sigma is None:
        sigma = diameter / 6

    x = torch.arange(0, diameter, device=device).float()
    y = torch.arange(0, diameter, device=device).float()
    y, x = torch.meshgrid(y, x, indexing="ij")

    center = radius
    return torch.exp(
        -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)
    )


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


def build_normalized_ignore_mask(
        gt_ignore_boxes_list,
        height,
        width,
        device,
        ignore_margin=1.0,
        ignore_expand_ratio=1.0,
    ):
    mask = torch.zeros((len(gt_ignore_boxes_list), 1, height, width), device=device)

    for batch_idx, ignore_boxes in enumerate(gt_ignore_boxes_list):
        ignore_boxes = ignore_boxes.to(device)
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


def build_raw_ignore_mask(
        gt_ignore_boxes_raw_list,
        scope_modes,
        full_rae_shapes,
        height,
        width,
        device,
        ignore_margin=1.0,
        ignore_expand_ratio=1.0,
    ):
    mask = torch.zeros((len(gt_ignore_boxes_raw_list), 1, height, width), device=device)

    for batch_idx, ignore_boxes in enumerate(gt_ignore_boxes_raw_list):
        ignore_boxes = ignore_boxes.to(device)
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


def heatmap_focal_loss(logits, targets, valid_mask=None, alpha=2.0, beta=4.0):
    pred = logits.sigmoid().clamp(min=1e-4, max=1.0 - 1e-4)
    if valid_mask is None:
        valid_mask = 1.0
    else:
        valid_mask = valid_mask.to(dtype=targets.dtype)
    pos_inds = targets.eq(1.0).float()
    neg_inds = targets.lt(1.0).float() * valid_mask
    pos_inds = pos_inds * valid_mask
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


def masked_l1_loss(pred, target, mask):
    mask = mask.expand_as(pred)
    denom = torch.clamp(mask.sum(), min=1.0)
    return F.l1_loss(pred * mask, target * mask, reduction="sum") / denom

