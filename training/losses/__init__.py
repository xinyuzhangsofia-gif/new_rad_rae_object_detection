"""Canonical detector training loss API and implementations."""

from .common import (
    DEFAULT_NUM_CLASSES,
    build_raw_ignore_mask,
    draw_gaussian,
    gaussian2d,
    heatmap_focal_loss,
    masked_l1_loss,
)
from .gwd import gaussian_wasserstein_distance_batch
from .targets import build_cartesian_centerpoint_targets, build_radenet_gaussian_heatmap
from .centerpoint import cartesian_centerpoint_detection_loss
from .radenet import radenet_continuous_focal_loss, radenet_detection_loss
from .yolox import yolox_detection_loss


__all__ = [
    "DEFAULT_NUM_CLASSES",
    "gaussian2d",
    "draw_gaussian",
    "build_raw_ignore_mask",
    "heatmap_focal_loss",
    "build_radenet_gaussian_heatmap",
    "radenet_continuous_focal_loss",
    "gaussian_wasserstein_distance_batch",
    "radenet_detection_loss",
    "build_cartesian_centerpoint_targets",
    "cartesian_centerpoint_detection_loss",
    "masked_l1_loss",
    "yolox_detection_loss",
]
