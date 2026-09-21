"""Convert canonical decoded detections into visualization coordinates."""

import torch

from configs.coordinates import BOX_COORDINATE_CARTESIAN, require_cartesian_data
from data.geometry import metric_boxes_to_raw_local_rae
from eval.decoding import decode_batch_predictions


def _normalize_class_names(class_names):
    if not class_names:
        return {}
    return {
        int(class_id): str(class_name)
        for class_id, class_name in dict(class_names).items()
    }


def resolve_visualization_classes(checkpoint_config):
    """Read current class metadata without reconstructing missing fields."""
    num_classes = int(checkpoint_config["num_classes"])
    class_names = _normalize_class_names(checkpoint_config["class_names"])
    class_to_idx = {
        str(class_name): int(class_id)
        for class_name, class_id in dict(checkpoint_config["class_to_idx"]).items()
    }
    expected_ids = set(range(num_classes))
    if set(class_names) != expected_ids:
        raise ValueError(
            "Checkpoint config.class_names does not match config.num_classes"
        )
    if set(class_to_idx.values()) != expected_ids:
        raise ValueError(
            "Checkpoint config.class_to_idx does not match config.num_classes"
        )
    return num_classes, class_names, class_to_idx


def format_visualization_predictions(
        frame_predictions,
        scope_mode,
        full_rae_shape,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
    ):
    """Convert canonical detections into polar-bin and metric plot formats."""
    pred_boxes = frame_predictions["boxes"]
    pred_labels = frame_predictions["labels"]
    pred_scores = frame_predictions["scores"]
    require_cartesian_data(box_coordinate_mode)
    pred_boxes_metric = pred_boxes.clone()
    pred_boxes_raw = metric_boxes_to_raw_local_rae(
        metric_boxes=pred_boxes,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
        use_planar_center_range=True,
    )

    return (
        pred_boxes_raw.cpu(),
        pred_labels.cpu(),
        pred_scores.cpu(),
        pred_boxes_metric.cpu(),
    )


def filter_predictions(
        outputs,
        num_classes,
        scope_mode,
        full_rae_shape,
        score_thresh,
        max_detections,
        pred_mode,
        heatmap_nms_kernel,
        heatmap_score_mode,
        yolox_nms_iou,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
    ):
    """Decode model outputs through the canonical evaluation decoder."""
    frame_predictions = decode_batch_predictions(
        outputs=outputs,
        num_classes=num_classes,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
        yolox_nms_iou=yolox_nms_iou,
        score_thresh=score_thresh,
        scope_modes=[scope_mode],
        full_rae_shapes=[full_rae_shape],
        box_coordinate_mode=box_coordinate_mode,
        prediction_mode=pred_mode,
        filter_to_scope_before_nms=True,
    )[0]
    return format_visualization_predictions(
        frame_predictions=frame_predictions,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
        box_coordinate_mode=box_coordinate_mode,
    )
