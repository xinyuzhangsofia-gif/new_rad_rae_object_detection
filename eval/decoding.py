"""Prediction decoding, score processing, and NMS helpers."""

import torch
import torch.nn.functional as F

from data.coordinates import (
    AZIMUTH_AXIS,
    ELEVATION_AXIS,
    RANGE_AXIS,
    RDR_SP_CUBE,
    SCOPE_NARROW,
    denormalize_rae_boxes_for_scope,
    normalized_rae_box_centers_in_cartesian_roi,
)
from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    BOX_COORDINATE_POLAR,
    validate_box_coordinate_mode,
)
from eval.kitti_eval.rotate_iou_cpu import (
    rotate_iou_gpu_eval as rotate_iou_cpu_eval,
)
from data.geometry import (
    centerpoint_outputs_to_metric_regression,
    regression_cell_to_metric_box,
    regression_cell_to_normalized_rae_box,
)
from training_utils.yolox_utils import yolox_outputs_to_detections

HEATMAP_SCORE_MODES = ("peak_times_local_mean", "peak_only")
HEATMAP_PEAK_WEIGHT = 0.85
HEATMAP_LOCAL_MEAN_WEIGHT = 0.15
RADENET_ROTATED_NMS_IOU_THRESHOLD = 0.3


__all__ = [
    'normalized_rae_boxes_to_cartesian_metric_boxes',
    'centerpoint_heatmap_nms',
    'centerpoint_local_heatmap_mean',
    'build_heatmap_candidate_scores',
    'gather_dense_feature',
    'topk_heatmap_candidates',
    'cartesian_rotated_nms_indices',
    'apply_quality_score',
    'outputs_to_detections',
    'official_radenet_outputs_to_detections',
    'decode_batch_predictions',
    'filter_predictions_to_scope'
]


def normalized_rae_boxes_to_cartesian_metric_boxes(boxes, scope_mode, rae_shape):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 7))

    raw_boxes = denormalize_rae_boxes_for_scope(
        boxes=boxes,
        scope_mode=scope_mode,
        rae_shape=rae_shape,
    )

    radius = RANGE_AXIS.minimum + (raw_boxes[:, 0] * RANGE_AXIS.step)
    azimuth = torch.deg2rad(
        raw_boxes[:, 1].new_tensor(AZIMUTH_AXIS.minimum)
        + (raw_boxes[:, 1] * AZIMUTH_AXIS.step)
    )
    elevation = torch.deg2rad(
        raw_boxes[:, 2].new_tensor(ELEVATION_AXIS.minimum)
        + (raw_boxes[:, 2] * ELEVATION_AXIS.step)
    )

    r_xy = radius * torch.cos(elevation)
    x = r_xy * torch.cos(azimuth)
    y = -r_xy * torch.sin(azimuth)
    z = radius * torch.sin(elevation)

    length = (raw_boxes[:, 3].abs() * RANGE_AXIS.step).clamp(min=1e-3)
    width = (
        r_xy.abs()
        * torch.deg2rad(raw_boxes[:, 4].abs() * AZIMUTH_AXIS.step)
    ).clamp(min=1e-3)
    height = (
        (radius * torch.cos(elevation)).abs()
        * torch.deg2rad(raw_boxes[:, 5].abs() * ELEVATION_AXIS.step)
    ).clamp(min=1e-3)
    yaw = raw_boxes[:, 6]

    return torch.stack([x, y, z, length, width, height, yaw], dim=-1)


def centerpoint_heatmap_nms(heatmap, kernel_size=3):
    if kernel_size <= 1:
        return heatmap
    if kernel_size % 2 == 0:
        raise ValueError(f"Heatmap NMS kernel must be odd, got {kernel_size}")

    pad = (kernel_size - 1) // 2
    pooled = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)
    keep = pooled == heatmap
    return heatmap * keep.to(heatmap.dtype)


def centerpoint_local_heatmap_mean(heatmap, kernel_size=3):
    if kernel_size <= 1:
        return heatmap
    if kernel_size % 2 == 0:
        raise ValueError(f"Heatmap local-mean kernel must be odd, got {kernel_size}")

    pad = (kernel_size - 1) // 2
    return F.avg_pool2d(
        heatmap,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
        count_include_pad=False,
    )


def build_heatmap_candidate_scores(
        heatmap_scores,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        peak_scores=None,
    ):
    if peak_scores is None:
        peak_scores = centerpoint_heatmap_nms(
            heatmap=heatmap_scores,
            kernel_size=heatmap_nms_kernel,
        )

    if heatmap_score_mode == "peak_only":
        return peak_scores
    if heatmap_score_mode == "peak_times_local_mean":
        local_mean_scores = centerpoint_local_heatmap_mean(
            heatmap=heatmap_scores,
            kernel_size=heatmap_nms_kernel,
        )
        return (
            (HEATMAP_PEAK_WEIGHT * peak_scores)
            + (HEATMAP_LOCAL_MEAN_WEIGHT * local_mean_scores)
        )
    raise ValueError(
        f"Unknown heatmap_score_mode={heatmap_score_mode!r}. "
        f"Expected one of {HEATMAP_SCORE_MODES}."
    )


def gather_dense_feature(feature_map, indices):
    flat = feature_map.flatten(start_dim=2).transpose(1, 2)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
    return flat.gather(dim=1, index=gather_index)


def topk_heatmap_candidates(candidate_scores, max_detections, score_thresh=None):
    flat_scores = candidate_scores.flatten(start_dim=1)
    topk_count = min(max_detections, flat_scores.shape[1])
    scores, flat_indices = flat_scores.topk(topk_count, dim=1)
    keep = scores > 0.0
    if score_thresh is not None:
        keep = keep & (scores > float(score_thresh))
    return scores, flat_indices, keep


def cartesian_rotated_nms_indices(
        boxes,
        scores,
        iou_threshold=RADENET_ROTATED_NMS_IOU_THRESHOLD,
    ):
    """Class-agnostic rotated BEV NMS matching the original RADE-Net."""
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    bev_boxes = torch.stack(
        [
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 3].abs(),
            boxes[:, 4].abs(),
            boxes[:, 6],
        ],
        dim=-1,
    ).detach().cpu().numpy()
    keep = []

    while order.numel() > 0:
        current = order[0]
        keep.append(int(current.item()))
        if order.numel() == 1:
            break

        remaining = order[1:]
        current_index = int(current.item())
        remaining_indices = remaining.detach().cpu().numpy()
        ious = rotate_iou_cpu_eval(
            bev_boxes[current_index:current_index + 1],
            bev_boxes[remaining_indices],
            criterion=-1,
        )[0]
        keep_mask = torch.as_tensor(
            ious < float(iou_threshold),
            dtype=torch.bool,
            device=remaining.device,
        )
        order = remaining[keep_mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def apply_quality_score(heatmap_scores, outputs):
    if "quality_logits" in outputs:
        quality_scores = outputs["quality_logits"].sigmoid()
        if quality_scores.shape[-2:] != heatmap_scores.shape[-2:]:
            quality_scores = F.interpolate(
                quality_scores,
                size=heatmap_scores.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return heatmap_scores * quality_scores

    if "objectness_logits" not in outputs:
        return heatmap_scores

    objectness_scores = outputs["objectness_logits"].sigmoid()
    if objectness_scores.shape[-2:] != heatmap_scores.shape[-2:]:
        objectness_scores = F.interpolate(
            objectness_scores,
            size=heatmap_scores.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return heatmap_scores * objectness_scores


def outputs_to_detections(
        outputs,
        num_classes,
        max_detections=64,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        score_thresh=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
        scope_modes=None,
        full_rae_shapes=None,
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    dense_keys = {"cls_logits", "center_offset", "center_height", "size", "yaw"}
    missing_keys = sorted(dense_keys - set(outputs.keys()))
    if len(missing_keys) > 0:
        raise KeyError(
            "Dense CenterPoint evaluation requires output keys "
            f"{sorted(dense_keys)}, missing {missing_keys}."
        )

    cls_logits = outputs["cls_logits"][:, :num_classes]
    _, _, heatmap_h, heatmap_w = cls_logits.shape
    dtype = cls_logits.dtype

    heatmap_scores = apply_quality_score(cls_logits.sigmoid(), outputs)
    rescored_heatmap = build_heatmap_candidate_scores(
        heatmap_scores=heatmap_scores,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
    )

    scores, flat_indices, keep = topk_heatmap_candidates(
        candidate_scores=rescored_heatmap,
        max_detections=max_detections,
        score_thresh=score_thresh,
    )

    spatial_size = heatmap_h * heatmap_w
    labels = flat_indices // spatial_size
    spatial_indices = flat_indices % spatial_size

    heatmap_y_idx = spatial_indices // heatmap_w
    heatmap_x_idx = spatial_indices % heatmap_w
    _, _, box_h, box_w = outputs["center_offset"].shape
    box_y_idx_long = torch.div(
        heatmap_y_idx * box_h,
        max(heatmap_h, 1),
        rounding_mode="floor",
    ).clamp(max=box_h - 1)
    box_x_idx_long = torch.div(
        heatmap_x_idx * box_w,
        max(heatmap_w, 1),
        rounding_mode="floor",
    ).clamp(max=box_w - 1)
    box_indices = box_y_idx_long * box_w + box_x_idx_long

    y_idx = box_y_idx_long.to(dtype)
    x_idx = box_x_idx_long.to(dtype)
    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        if scope_modes is None or full_rae_shapes is None:
            raise ValueError(
                "Cartesian CenterPoint decoding requires scope_modes and "
                "full_rae_shapes."
            )
        regression_map = centerpoint_outputs_to_metric_regression(outputs)
        pred_reg = gather_dense_feature(regression_map, box_indices)
        metric_boxes = []
        for batch_index in range(pred_reg.shape[0]):
            metric_boxes.append(
                regression_cell_to_metric_box(
                    pred_reg=pred_reg[batch_index],
                    y_idx=y_idx[batch_index],
                    x_idx=x_idx[batch_index],
                    feature_shape=(box_h, box_w),
                    scope_mode=scope_modes[batch_index],
                    full_rae_shape=full_rae_shapes[batch_index],
                )
            )
        boxes = torch.stack(metric_boxes, dim=0)
    else:
        center_offset = gather_dense_feature(
            outputs["center_offset"],
            box_indices,
        ).sigmoid()
        center_height = gather_dense_feature(
            outputs["center_height"],
            box_indices,
        ).sigmoid()
        size = gather_dense_feature(outputs["size"], box_indices).sigmoid()
        yaw = gather_dense_feature(outputs["yaw"], box_indices)

        r_center = (y_idx + center_offset[..., 0]) / max(box_h, 1)
        a_center = (x_idx + center_offset[..., 1]) / max(box_w, 1)
        e_center = center_height[..., 0]
        yaw_angle = torch.atan2(yaw[..., 0], yaw[..., 1])
        yaw_norm = (yaw_angle + torch.pi) / (2.0 * torch.pi)

        boxes = torch.stack(
            [
                r_center,
                a_center,
                e_center,
                size[..., 0],
                size[..., 1],
                size[..., 2],
                yaw_norm,
            ],
            dim=-1,
        ).clamp(min=1e-4, max=1.0 - 1e-4)

    return boxes, scores, labels, keep


def official_radenet_outputs_to_detections(
        outputs,
        num_classes,
        scope_modes,
        full_rae_shapes,
        max_detections=64,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        score_thresh=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    if scope_modes is None or full_rae_shapes is None:
        raise ValueError("Official RADE-Net decoding requires scope_modes and full_rae_shapes.")

    heatmap_scores = outputs["heatmap"][:, :num_classes]
    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        # Original RADE-Net ranks detections by the heatmap peak itself.
        # The local-mean re-scoring option belongs to this project's
        # CenterPoint path and must not alter Cartesian RADE confidence.
        heatmap_score_mode = "peak_only"
    rescored_heatmap = build_heatmap_candidate_scores(
        heatmap_scores=heatmap_scores,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
    )
    regression = outputs["regression"]
    batch_size, _, heatmap_h, heatmap_w = heatmap_scores.shape

    scores, flat_indices, keep = topk_heatmap_candidates(
        candidate_scores=rescored_heatmap,
        max_detections=max_detections,
        score_thresh=score_thresh,
    )

    spatial_size = heatmap_h * heatmap_w
    labels = flat_indices // spatial_size
    spatial_indices = flat_indices % spatial_size
    y_idx = spatial_indices // heatmap_w
    x_idx = spatial_indices % heatmap_w
    reg = gather_dense_feature(regression, spatial_indices)

    boxes = []
    for batch_index in range(batch_size):
        decode_function = (
            regression_cell_to_metric_box
            if box_coordinate_mode == BOX_COORDINATE_CARTESIAN
            else regression_cell_to_normalized_rae_box
        )
        boxes.append(
            decode_function(
                pred_reg=reg[batch_index],
                y_idx=y_idx[batch_index].to(reg.dtype),
                x_idx=x_idx[batch_index].to(reg.dtype),
                feature_shape=(heatmap_h, heatmap_w),
                scope_mode=scope_modes[batch_index],
                full_rae_shape=full_rae_shapes[batch_index],
                **(
                    {"absolute_dimensions": False}
                    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN
                    else {}
                ),
            )
        )
    return torch.stack(boxes, dim=0), scores, labels, keep


def decode_batch_predictions(
        outputs,
        num_classes,
        max_detections,
        heatmap_nms_kernel,
        heatmap_score_mode,
        yolox_nms_iou,
        score_thresh=None,
        scope_modes=None,
        full_rae_shapes=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    if "objectness_logits" in outputs:
        if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
            raise ValueError(
                "Cartesian box mode is not implemented for YOLOX outputs."
            )
        return yolox_outputs_to_detections(
            outputs=outputs,
            num_classes=num_classes,
            score_thresh=score_thresh,
            max_detections=max_detections,
            nms_iou_thresh=yolox_nms_iou,
        )

    is_official_radenet_output = (
        "heatmap" in outputs and "regression" in outputs
    )
    if is_official_radenet_output:
        pred_boxes, pred_scores, pred_labels, pred_keep = official_radenet_outputs_to_detections(
            outputs=outputs,
            num_classes=num_classes,
            scope_modes=scope_modes,
            full_rae_shapes=full_rae_shapes,
            max_detections=max_detections,
            heatmap_nms_kernel=heatmap_nms_kernel,
            heatmap_score_mode=heatmap_score_mode,
            score_thresh=score_thresh,
            box_coordinate_mode=box_coordinate_mode,
        )
    else:
        pred_boxes, pred_scores, pred_labels, pred_keep = outputs_to_detections(
            outputs=outputs,
            num_classes=num_classes,
            max_detections=max_detections,
            heatmap_nms_kernel=heatmap_nms_kernel,
            heatmap_score_mode=heatmap_score_mode,
            score_thresh=score_thresh,
            box_coordinate_mode=box_coordinate_mode,
            scope_modes=scope_modes,
            full_rae_shapes=full_rae_shapes,
        )

    batch_predictions = []
    for batch_index in range(pred_boxes.shape[0]):
        keep = pred_keep[batch_index]
        boxes = pred_boxes[batch_index][keep]
        scores = pred_scores[batch_index][keep]
        labels = pred_labels[batch_index][keep]
        if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
            nms_keep = cartesian_rotated_nms_indices(
                boxes=boxes,
                scores=scores,
            )
            boxes = boxes[nms_keep]
            scores = scores[nms_keep]
            labels = labels[nms_keep]
        batch_predictions.append({
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "box_coordinate_mode": box_coordinate_mode,
        })
    return batch_predictions


def filter_predictions_to_scope(
        frame_predictions,
        scope_mode,
        full_rae_shape,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    if scope_mode != SCOPE_NARROW:
        return frame_predictions

    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        boxes = frame_predictions["boxes"]
        roi = RDR_SP_CUBE["ROI"]
        keep = (
            (boxes[:, 0] >= float(roi["x"][0]))
            & (boxes[:, 0] <= float(roi["x"][1]))
            & (boxes[:, 1] >= float(roi["y"][0]))
            & (boxes[:, 1] <= float(roi["y"][1]))
            & (boxes[:, 2] >= float(roi["z"][0]))
            & (boxes[:, 2] <= float(roi["z"][1]))
        )
    else:
        keep = normalized_rae_box_centers_in_cartesian_roi(
            frame_predictions["boxes"],
            scope_mode=scope_mode,
            rae_shape=full_rae_shape,
        )
    return {
        "boxes": frame_predictions["boxes"][keep],
        "scores": frame_predictions["scores"][keep],
        "labels": frame_predictions["labels"][keep],
        "box_coordinate_mode": box_coordinate_mode,
    }
