"""Canonical detector training loss API and implementations."""

DEFAULT_NUM_CLASSES = 2

from training.losses.centerpoint import (
    cartesian_centerpoint_detection_loss,
    centerpoint_detection_loss,
    centerpoint_gwd_loss,
    centerpoint_quality_focal_loss,
    centerpoint_quality_loss,
    dense_centerpoint_outputs_to_boxes,
)
from training.losses.common import (
    boxes_3d_to_ra_xyxy,
    build_normalized_ignore_mask,
    build_raw_ignore_mask,
    draw_gaussian,
    gaussian2d,
    heatmap_focal_loss,
    masked_l1_loss,
    pairwise_box_iou_2d,
)
from training.losses.gwd import (
    _box_to_gaussian_batch,
    _matrix_sqrt_batch,
    gaussian_wasserstein_distance_batch,
    normalized_rae_boxes_to_gwd_boxes,
)
from training.losses.radenet import (
    _gather_regression_at_centers,
    _normalize_radenet_loss_term,
    radenet_continuous_focal_loss,
    radenet_detection_loss,
)
from training.losses.targets import (
    _raw_index_to_feature_index,
    build_cartesian_centerpoint_targets,
    build_centerpoint_targets,
    build_radenet_gaussian_heatmap,
    normalized_boxes_to_centerpoint_targets,
)
from training.losses.yolox import yolox_detection_loss


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
