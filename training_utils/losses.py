import math

import torch
import torch.nn.functional as F

from data.coordinates import (
    denormalize_rae_boxes_to_local_scope,
    get_rae_scope_start_and_shape,
)
from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    BOX_COORDINATE_POLAR,
    validate_box_coordinate_mode,
)
from training_utils.yolox_utils import (
    decode_yolox_boxes,
    simota_assign,
    yolox_grid_centers,
)
from data.geometry import (
    centerpoint_outputs_to_metric_regression,
    feature_indices_to_cartesian_xy,
    raw_local_rae_boxes_to_metric_boxes,
    regression_cell_to_metric_box,
)


DEFAULT_NUM_CLASSES = 2


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


def normalized_rae_boxes_to_gwd_boxes(boxes, scope_mode=None, full_rae_shape=None):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 8))

    if scope_mode is not None and full_rae_shape is not None:
        raw_boxes = denormalize_rae_boxes_to_local_scope(
            boxes=boxes,
            scope_mode=scope_mode,
            rae_shape=full_rae_shape,
        )
        return raw_local_rae_boxes_to_metric_boxes(
            raw_boxes=raw_boxes,
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
        )

    yaw = (boxes[:, 6] * 2.0 * torch.pi) - torch.pi
    return torch.stack(
        [
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2],
            boxes[:, 3].abs().clamp(min=1e-4),
            boxes[:, 4].abs().clamp(min=1e-4),
            boxes[:, 5].abs().clamp(min=1e-4),
            torch.sin(yaw),
            torch.cos(yaw),
        ],
        dim=-1,
    )


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


def build_radenet_gaussian_heatmap(
        gt_boxes_raw_list,
        gt_labels_list,
        scope_modes,
        full_rae_shapes,
        batch_size,
        num_classes,
        height,
        width,
        device,
        sigma=3.0,
    ):
    heatmap = torch.zeros((batch_size, num_classes, height, width), device=device)
    y_grid = torch.arange(0, height, device=device, dtype=torch.float32).view(height, 1).expand(height, width)
    x_grid = torch.arange(0, width, device=device, dtype=torch.float32).view(1, width).expand(height, width)
    default_factor = 2.0 * (sigma ** 2)

    for batch_idx in range(batch_size):
        raw_boxes = gt_boxes_raw_list[batch_idx].to(device)
        labels = gt_labels_list[batch_idx].to(device)
        if raw_boxes.numel() == 0 or labels.numel() == 0:
            continue

        valid = (labels >= 0) & (labels < num_classes)
        raw_boxes = raw_boxes[valid]
        labels = labels[valid]
        if raw_boxes.numel() == 0:
            continue

        _, scope_shape = get_rae_scope_start_and_shape(scope_modes[batch_idx], full_rae_shapes[batch_idx])
        scope_h = int(scope_shape[0])
        scope_w = int(scope_shape[1])
        scale_y = 0.0 if height <= 1 or scope_h <= 1 else float(height - 1) / float(scope_h - 1)
        scale_x = 0.0 if width <= 1 or scope_w <= 1 else float(width - 1) / float(scope_w - 1)

        for raw_box, cls_id in zip(raw_boxes, labels):
            cls_id = int(cls_id.item())
            center_y = torch.round(raw_box[0] * scale_y).clamp(min=0.0, max=float(height - 1))
            center_x = torch.round(raw_box[1] * scale_x).clamp(min=0.0, max=float(width - 1))
            gaussian = torch.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / default_factor)
            heatmap[batch_idx, cls_id] = torch.clamp(heatmap[batch_idx, cls_id] + gaussian, 0.0, 1.0)

    return heatmap


def radenet_continuous_focal_loss(heatmap, gaussian_map, valid_mask=None, alpha=2.0, gamma=4.0):
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


def _box_to_gaussian_batch(boxes):
    x, y, _, length, width, _, yaw_sin, yaw_cos = torch.unbind(boxes, dim=-1)
    width_half = width / 2.0
    length_half = length / 2.0
    cos_sq = yaw_cos ** 2
    sin_sq = yaw_sin ** 2
    cos_sin = yaw_cos * yaw_sin

    sigma_11 = width_half * cos_sq + length_half * sin_sq
    sigma_12 = (width_half - length_half) * cos_sin
    sigma_21 = sigma_12
    sigma_22 = width_half * sin_sq + length_half * cos_sq

    sigma = torch.stack(
        [
            torch.stack([sigma_11, sigma_12], dim=-1),
            torch.stack([sigma_21, sigma_22], dim=-1),
        ],
        dim=-2,
    )
    mu = torch.stack([x, y], dim=-1)
    return mu, sigma


def _matrix_sqrt_batch(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=1e-12))
    eigen_diag = torch.diag_embed(sqrt_eigenvalues)
    return eigenvectors @ eigen_diag @ eigenvectors.transpose(-1, -2)


def gaussian_wasserstein_distance_batch(pred_boxes, gt_boxes, tau=1.65):
    pred_mu, pred_sigma = _box_to_gaussian_batch(pred_boxes)
    gt_mu, gt_sigma = _box_to_gaussian_batch(gt_boxes)

    first_term = (pred_mu - gt_mu).pow(2).sum(dim=-1)
    pred_sigma_sq = pred_sigma @ pred_sigma
    gt_sigma_sq = gt_sigma @ gt_sigma
    intermediate = (-2.0) * _matrix_sqrt_batch(pred_sigma @ gt_sigma_sq @ pred_sigma)
    matrix_add = pred_sigma_sq + gt_sigma_sq + intermediate
    second_term = torch.diagonal(matrix_add, dim1=-2, dim2=-1).sum(dim=-1)
    gwd = first_term + second_term
    iou_like = 1.0 / (tau + torch.sqrt(torch.clamp(gwd, min=1e-12)))
    return iou_like, 1.0 - iou_like


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
        box_coordinate_mode=BOX_COORDINATE_POLAR,
        gt_ignore_boxes_raw_list=None,
        num_classes=DEFAULT_NUM_CLASSES,
        gaussian_sigma=3.0,
        ignore_mask_margin=1.0,
        ignore_mask_expand_ratio=1.0,
    ):
    if "heatmap" not in outputs or "regression" not in outputs:
        raise KeyError("RADE-Net loss requires model outputs to contain 'heatmap' and 'regression'.")
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    if (
        box_coordinate_mode == BOX_COORDINATE_CARTESIAN
        and gt_metric_boxes_list is None
    ):
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

        if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
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
        else:
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
        radius=3,
        reg_reference=None
    ):
    B, _, heatmap_h, heatmap_w = cls_logits.shape
    device = cls_logits.device
    if reg_reference is None:
        reg_reference = cls_logits
    _, _, reg_h, reg_w = reg_reference.shape

    heatmap_targets = torch.zeros((B, num_classes, heatmap_h, heatmap_w), device=device)
    reg_targets = {
        "center_offset": torch.zeros((B, 2, reg_h, reg_w), device=device),
        "center_height": torch.zeros((B, 1, reg_h, reg_w), device=device),
        "size": torch.zeros((B, 3, reg_h, reg_w), device=device),
        "yaw": torch.zeros((B, 2, reg_h, reg_w), device=device),
        "box": torch.zeros((B, 7, reg_h, reg_w), device=device),
        "label": torch.full((B, reg_h, reg_w), -1, dtype=torch.long, device=device),
    }
    reg_mask = torch.zeros((B, 1, reg_h, reg_w), device=device)

    for b in range(B):
        boxes_b = gt_boxes[b].to(device)
        labels_b = gt_labels[b].to(device)

        if boxes_b.numel() == 0:
            continue

        for box, cls_id in zip(boxes_b, labels_b):
            cls_id = int(cls_id.item())
            if cls_id < 0 or cls_id >= num_classes:
                continue

            heatmap_target = normalized_boxes_to_centerpoint_targets(
                box=box,
                H=heatmap_h,
                W=heatmap_w
            )
            draw_gaussian(
                heatmap=heatmap_targets[b, cls_id],
                center_y=heatmap_target["center_y"],
                center_x=heatmap_target["center_x"],
                radius=radius
            )

            reg_target = normalized_boxes_to_centerpoint_targets(
                box=box,
                H=reg_h,
                W=reg_w
            )
            reg_center_y = reg_target["center_y"]
            reg_center_x = reg_target["center_x"]

            reg_targets["center_offset"][b, :, reg_center_y, reg_center_x] = reg_target["center_offset"]
            reg_targets["center_height"][b, :, reg_center_y, reg_center_x] = reg_target["center_height"]
            reg_targets["size"][b, :, reg_center_y, reg_center_x] = reg_target["size"]
            reg_targets["yaw"][b, :, reg_center_y, reg_center_x] = reg_target["yaw"]
            reg_targets["box"][b, :, reg_center_y, reg_center_x] = box.clamp(0.0, 1.0)
            reg_targets["label"][b, reg_center_y, reg_center_x] = cls_id
            reg_mask[b, :, reg_center_y, reg_center_x] = 1.0

    return heatmap_targets, reg_targets, reg_mask


def _raw_index_to_feature_index(raw_index, scope_size, feature_size):
    if feature_size <= 1 or scope_size <= 1:
        return 0
    scaled = float(raw_index) * float(feature_size - 1) / float(scope_size - 1)
    return max(0, min(feature_size - 1, int(round(scaled))))


def build_cartesian_centerpoint_targets(
        gt_boxes_raw_list,
        gt_metric_boxes_list,
        gt_labels_list,
        cls_logits,
        regression_map,
        scope_modes,
        full_rae_shapes,
        num_classes,
        radius=3,
    ):
    """Build an R-A heatmap plus exact metric Cartesian box targets."""
    batch_size, _, heatmap_h, heatmap_w = cls_logits.shape
    _, _, reg_h, reg_w = regression_map.shape
    device = cls_logits.device
    heatmap_targets = torch.zeros(
        (batch_size, num_classes, heatmap_h, heatmap_w),
        device=device,
    )
    metric_targets = torch.zeros(
        (batch_size, 7, reg_h, reg_w),
        device=device,
    )
    reg_mask = torch.zeros((batch_size, 1, reg_h, reg_w), device=device)

    for batch_idx in range(batch_size):
        raw_boxes = gt_boxes_raw_list[batch_idx].to(device)
        metric_boxes = gt_metric_boxes_list[batch_idx].to(device)
        labels = gt_labels_list[batch_idx].to(device)
        if not (
            raw_boxes.shape[0] == metric_boxes.shape[0] == labels.shape[0]
        ):
            raise ValueError(
                "Cartesian GT raw/metric/label counts disagree for batch "
                f"{batch_idx}: raw={raw_boxes.shape[0]}, "
                f"metric={metric_boxes.shape[0]}, labels={labels.shape[0]}"
            )

        _, scope_shape = get_rae_scope_start_and_shape(
            scope_modes[batch_idx],
            full_rae_shapes[batch_idx],
        )
        scope_h = int(scope_shape[0])
        scope_w = int(scope_shape[1])

        for raw_box, metric_box, cls_id_tensor in zip(
                raw_boxes,
                metric_boxes,
                labels,
            ):
            cls_id = int(cls_id_tensor.item())
            if cls_id < 0 or cls_id >= num_classes:
                continue

            heatmap_y = _raw_index_to_feature_index(
                raw_box[0].item(),
                scope_h,
                heatmap_h,
            )
            heatmap_x = _raw_index_to_feature_index(
                raw_box[1].item(),
                scope_w,
                heatmap_w,
            )
            draw_gaussian(
                heatmap=heatmap_targets[batch_idx, cls_id],
                center_y=heatmap_y,
                center_x=heatmap_x,
                radius=radius,
            )

            reg_y = _raw_index_to_feature_index(
                raw_box[0].item(),
                scope_h,
                reg_h,
            )
            reg_x = _raw_index_to_feature_index(
                raw_box[1].item(),
                scope_w,
                reg_w,
            )
            metric_targets[batch_idx, :, reg_y, reg_x] = metric_box
            reg_mask[batch_idx, :, reg_y, reg_x] = 1.0

    return heatmap_targets, metric_targets, reg_mask


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
    weight.  The ``radenet_detection_loss`` function above remains unchanged
    and preserves the original RADE-Net normalization for ``loss_mode``
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


def masked_l1_loss(pred, target, mask):
    mask = mask.expand_as(pred)
    denom = torch.clamp(mask.sum(), min=1.0)
    return F.l1_loss(pred * mask, target * mask, reduction="sum") / denom


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


def centerpoint_gwd_loss(outputs, target_boxes, mask, scope_modes=None, full_rae_shapes=None):
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


def yolox_detection_loss(
        outputs,
        gt_boxes_list,
        gt_labels_list,
        gt_ignore_boxes_list=None,
        scope_modes=None,
        full_rae_shapes=None,
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
    pred_boxes = decode_yolox_boxes(outputs, clamp=True)
    grid_centers, grid_h, grid_w = yolox_grid_centers(outputs)
    # model12 was tuned around an 8x8 output grid. Scale the assignment window and
    # candidate budget with the actual grid size so denser YOLOX-style heads, such as
    # the Swin-based model14, keep a comparable normalized matching region.
    grid_scale = max(grid_h, grid_w) / 8.0
    center_radius = 2.5 * grid_scale
    candidate_topk = max(10, int(round(10 * grid_scale)))

    batch_size, num_preds, _ = pred_boxes.shape
    cls_logits = cls_logits_map.flatten(start_dim=2).transpose(1, 2)
    objectness_logits = objectness_logits_map.flatten(start_dim=2).squeeze(1)

    total_box_loss = pred_boxes.new_tensor(0.0)
    total_obj_loss = pred_boxes.new_tensor(0.0)
    total_cls_loss = pred_boxes.new_tensor(0.0)
    total_l1_loss = pred_boxes.new_tensor(0.0)
    total_fg = 0
    total_ignore_pixels = 0

    if gt_ignore_boxes_list is None:
        obj_valid_masks = torch.ones((batch_size, num_preds), device=pred_boxes.device)
    else:
        ignore_mask = build_normalized_ignore_mask(
            gt_ignore_boxes_list=gt_ignore_boxes_list,
            height=grid_h,
            width=grid_w,
            device=pred_boxes.device,
            ignore_margin=ignore_mask_margin,
            ignore_expand_ratio=ignore_mask_expand_ratio,
        )
        total_ignore_pixels = int(ignore_mask.sum().item())
        obj_valid_masks = 1.0 - ignore_mask.flatten(start_dim=2).squeeze(1)

    for batch_idx in range(batch_size):
        gt_boxes = gt_boxes_list[batch_idx].to(pred_boxes.device)
        gt_labels = gt_labels_list[batch_idx].to(pred_boxes.device)
        obj_targets = torch.zeros((num_preds,), device=pred_boxes.device)

        matched_pred_idx, matched_gt_idx, matched_labels, matched_ious = simota_assign(
            pred_boxes=pred_boxes[batch_idx],
            cls_logits=cls_logits[batch_idx],
            objectness_logits=objectness_logits[batch_idx],
            grid_centers=grid_centers[batch_idx],
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            num_classes=num_classes,
            height=grid_h,
            width=grid_w,
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
            scope_mode = None if scope_modes is None else scope_modes[batch_idx]
            full_rae_shape = None if full_rae_shapes is None else full_rae_shapes[batch_idx]
            pred_gwd_boxes = normalized_rae_boxes_to_gwd_boxes(
                boxes=pred_pos_boxes,
                scope_mode=scope_mode,
                full_rae_shape=full_rae_shape,
            )
            gt_gwd_boxes = normalized_rae_boxes_to_gwd_boxes(
                boxes=matched_gt_boxes,
                scope_mode=scope_mode,
                full_rae_shape=full_rae_shape,
            )
            _, gwd_loss = gaussian_wasserstein_distance_batch(pred_gwd_boxes, gt_gwd_boxes)
            total_box_loss = total_box_loss + gwd_loss.sum()

            cls_targets = F.one_hot(matched_labels, num_classes=num_classes).float()
            cls_targets = cls_targets * matched_ious.detach().unsqueeze(1)
            total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                cls_logits[batch_idx, matched_pred_idx],
                cls_targets,
                reduction="sum",
            )
            total_l1_loss = total_l1_loss + F.l1_loss(
                pred_pos_boxes,
                matched_gt_boxes.clamp(0.0, 1.0),
                reduction="sum",
            )

        obj_loss = F.binary_cross_entropy_with_logits(
            objectness_logits[batch_idx],
            obj_targets,
            reduction="none",
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
