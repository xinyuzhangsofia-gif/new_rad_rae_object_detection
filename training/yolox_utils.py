"""Cartesian decoding and score preparation for YOLOX-style dense outputs."""

import torch

from data.geometry import (
    centerpoint_outputs_to_metric_regression,
    regression_cell_to_metric_box,
)


def _feature_grid(batch_size, height, width, device, dtype):
    y_grid = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
    x_grid = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
    y_grid = y_grid.expand(batch_size, height, width)
    x_grid = x_grid.expand(batch_size, height, width)
    return y_grid, x_grid


def decode_yolox_boxes(outputs, scope_modes, full_rae_shapes):
    """Decode each R-A feature cell directly into a metric Cartesian box."""
    center_offset = outputs["center_offset"]
    batch_size, _, height, width = center_offset.shape
    if len(scope_modes) != batch_size or len(full_rae_shapes) != batch_size:
        raise ValueError("YOLOX decoding requires scope and RAE shape for each sample.")

    regression = centerpoint_outputs_to_metric_regression(outputs)
    regression = regression.flatten(start_dim=2).transpose(1, 2)
    grid_centers, _, _ = yolox_grid_centers(outputs)
    boxes = []
    for batch_idx in range(batch_size):
        boxes.append(
            regression_cell_to_metric_box(
                pred_reg=regression[batch_idx],
                y_idx=grid_centers[batch_idx, :, 0],
                x_idx=grid_centers[batch_idx, :, 1],
                feature_shape=(height, width),
                scope_mode=scope_modes[batch_idx],
                full_rae_shape=full_rae_shapes[batch_idx],
                absolute_dimensions=False,
            )
        )
    return torch.stack(boxes, dim=0)


def yolox_grid_centers(outputs):
    """Return feature-cell indices in R-A order for candidate localization."""
    center_offset = outputs["center_offset"]
    batch_size, _, height, width = center_offset.shape
    y_grid, x_grid = _feature_grid(
        batch_size, height, width, center_offset.device, center_offset.dtype
    )
    centers = torch.stack((y_grid, x_grid), dim=-1)
    return centers.reshape(batch_size, height * width, 2), height, width


def yolox_outputs_to_detections(
        outputs,
        num_classes,
        scope_modes,
        full_rae_shapes,
        score_thresh=0.5,
    ):
    """Return scored Cartesian candidates; final rotated NMS runs after scope filtering."""
    boxes = decode_yolox_boxes(outputs, scope_modes, full_rae_shapes)
    cls_scores = outputs["cls_logits"][:, :num_classes].flatten(start_dim=2).transpose(1, 2).sigmoid()
    objectness_scores = outputs["objectness_logits"].flatten(start_dim=2).transpose(1, 2).sigmoid()
    final_scores = cls_scores * objectness_scores

    detections = []
    for batch_idx in range(boxes.shape[0]):
        image_boxes = []
        image_scores = []
        image_labels = []
        for class_id in range(num_classes):
            class_scores = final_scores[batch_idx, :, class_id]
            keep = (
                torch.ones_like(class_scores, dtype=torch.bool)
                if score_thresh is None
                else class_scores > score_thresh
            )
            if not keep.any():
                continue
            image_boxes.append(boxes[batch_idx, keep])
            image_scores.append(class_scores[keep])
            image_labels.append(
                torch.full(
                    (int(keep.sum().item()),), class_id,
                    dtype=torch.long, device=boxes.device,
                )
            )

        if image_boxes:
            detections.append({
                "boxes": torch.cat(image_boxes, dim=0),
                "scores": torch.cat(image_scores, dim=0),
                "labels": torch.cat(image_labels, dim=0),
            })
        else:
            detections.append({
                "boxes": boxes.new_zeros((0, 7)),
                "scores": boxes.new_zeros((0,)),
                "labels": torch.empty((0,), dtype=torch.long, device=boxes.device),
            })
    return detections
