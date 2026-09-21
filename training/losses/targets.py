"""Dense heatmap and regression target construction for detector losses."""

import torch

from data.coordinates import get_rae_scope_start_and_shape
from training.losses.common import draw_gaussian


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
