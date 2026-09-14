"""Canonical detector training loss API and implementations."""

from importlib import import_module


DEFAULT_NUM_CLASSES = 2

_EXPORT_MODULES = {
    "boxes_3d_to_ra_xyxy": "common",
    "build_normalized_ignore_mask": "common",
    "build_raw_ignore_mask": "common",
    "draw_gaussian": "common",
    "gaussian2d": "common",
    "heatmap_focal_loss": "common",
    "masked_l1_loss": "common",
    "pairwise_box_iou_2d": "common",
    "_box_to_gaussian_batch": "gwd",
    "_matrix_sqrt_batch": "gwd",
    "gaussian_wasserstein_distance_batch": "gwd",
    "normalized_rae_boxes_to_gwd_boxes": "gwd",
    "_raw_index_to_feature_index": "targets",
    "build_cartesian_centerpoint_targets": "targets",
    "build_centerpoint_targets": "targets",
    "build_radenet_gaussian_heatmap": "targets",
    "normalized_boxes_to_centerpoint_targets": "targets",
    "cartesian_centerpoint_detection_loss": "centerpoint",
    "centerpoint_detection_loss": "centerpoint",
    "centerpoint_gwd_loss": "centerpoint",
    "centerpoint_quality_focal_loss": "centerpoint",
    "centerpoint_quality_loss": "centerpoint",
    "dense_centerpoint_outputs_to_boxes": "centerpoint",
    "_gather_regression_at_centers": "radenet",
    "_normalize_radenet_loss_term": "radenet",
    "radenet_continuous_focal_loss": "radenet",
    "radenet_detection_loss": "radenet",
    "yolox_detection_loss": "yolox",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_NUM_CLASSES",
    "boxes_3d_to_ra_xyxy",
    "normalized_rae_boxes_to_gwd_boxes",
    "pairwise_box_iou_2d",
    "gaussian2d",
    "draw_gaussian",
    "build_normalized_ignore_mask",
    "build_raw_ignore_mask",
    "heatmap_focal_loss",
    "build_radenet_gaussian_heatmap",
    "radenet_continuous_focal_loss",
    "gaussian_wasserstein_distance_batch",
    "radenet_detection_loss",
    "normalized_boxes_to_centerpoint_targets",
    "build_centerpoint_targets",
    "build_cartesian_centerpoint_targets",
    "cartesian_centerpoint_detection_loss",
    "masked_l1_loss",
    "dense_centerpoint_outputs_to_boxes",
    "centerpoint_gwd_loss",
    "centerpoint_quality_loss",
    "centerpoint_quality_focal_loss",
    "centerpoint_detection_loss",
    "yolox_detection_loss",
]
