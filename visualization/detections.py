"""Convert canonical decoded detections into visualization coordinates."""

import torch

from configs.coordinates import BOX_COORDINATE_CARTESIAN, BOX_COORDINATE_POLAR
from data.coordinates import denormalize_rae_boxes_to_local_scope
from data.dataset import CLASS_NAMES
from data.geometry import metric_boxes_to_raw_local_rae
from eval.decoding import decode_batch_predictions


def _normalize_class_names(class_names):
    if not class_names:
        return {}
    return {
        int(class_id): str(class_name)
        for class_id, class_name in dict(class_names).items()
    }


def resolve_visualization_classes(checkpoint_config, inferred_num_classes=None):
    """Resolve class metadata, preferring state-dict-derived class counts."""
    class_names = _normalize_class_names(checkpoint_config.get("class_names"))
    config_num_classes = checkpoint_config.get("num_classes")
    if inferred_num_classes is not None:
        num_classes = int(inferred_num_classes)
    elif config_num_classes is not None:
        num_classes = int(config_num_classes)
    elif class_names:
        num_classes = len(class_names)
    else:
        num_classes = len(CLASS_NAMES)

    defaults = {
        class_id: CLASS_NAMES.get(class_id, f"Class {class_id}")
        for class_id in range(num_classes)
    }
    if not class_names:
        class_names = defaults
    else:
        class_names = {
            class_id: class_name
            for class_id, class_name in class_names.items()
            if 0 <= class_id < num_classes
        }
        for class_id, class_name in defaults.items():
            class_names.setdefault(class_id, class_name)

    configured = checkpoint_config.get("class_to_idx")
    if configured:
        class_to_idx = {
            str(class_name): int(class_id)
            for class_name, class_id in dict(configured).items()
            if int(class_id) in class_names
        }
    else:
        class_to_idx = {
            class_name: class_id
            for class_id, class_name in class_names.items()
        }
    return num_classes, class_names, class_to_idx


def normalized_boxes_to_raw_rae(boxes, scope_mode, full_rae_shape):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 7))
    return denormalize_rae_boxes_to_local_scope(
        boxes=boxes,
        scope_mode=scope_mode,
        rae_shape=full_rae_shape,
    )


def format_visualization_predictions(
        frame_predictions,
        scope_mode,
        full_rae_shape,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    """Convert canonical detections into polar-bin and metric plot formats."""
    pred_boxes = frame_predictions["boxes"]
    pred_labels = frame_predictions["labels"]
    pred_scores = frame_predictions["scores"]
    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        pred_boxes_metric = pred_boxes.clone()
        pred_boxes_raw = metric_boxes_to_raw_local_rae(
            metric_boxes=pred_boxes,
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
            use_planar_center_range=True,
        )
    else:
        pred_boxes_metric = None
        pred_boxes_raw = normalized_boxes_to_raw_rae(
            pred_boxes,
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
        )
    return (
        pred_boxes_raw.cpu(),
        pred_labels.cpu(),
        pred_scores.cpu(),
        None if pred_boxes_metric is None else pred_boxes_metric.cpu(),
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
        box_coordinate_mode=BOX_COORDINATE_POLAR,
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
