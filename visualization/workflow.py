"""Configuration parsing and dispatch for the unified visualizer."""

import argparse

from data.coordinates import SCOPE_CHOICES
from training.configuration import normalize_optional_path
from visualization.config import (
    MODE_MULTISENSOR,
    MODE_MULTISENSOR_VIDEO,
    MODE_RA_MAP,
    MODE_RA_MAP_VIDEO,
    RA_MAP_COORDINATES,
    VISUALIZATION_MODES,
    validate_visualization_config,
)
from visualize_cfg import VISUALIZE_CONFIG


HEATMAP_SCORE_MODES = ("peak_times_local_mean", "peak_only")
MODEL_TYPES = tuple(["auto"] + [f"model{idx}" for idx in range(1, 17)])


def parse_args(argv=None):
    """Apply optional CLI overrides to the canonical root configuration."""
    defaults = dict(VISUALIZE_CONFIG)
    parser = argparse.ArgumentParser(
        description="Render RA-map or Camera/LiDAR/Radar visualizations."
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--mode", choices=VISUALIZATION_MODES)
    parser.add_argument("--ra-map-coordinate", choices=RA_MAP_COORDINATES)
    parser.add_argument("--checkpoint-path", default=defaults["checkpoint_path"])
    parser.add_argument("--sequence", type=int, default=defaults["sequence"])
    parser.add_argument("--frame", type=int, default=defaults["frame"])
    parser.add_argument("--frame-step", type=int, default=defaults["frame_step"])
    parser.add_argument("--max-frames", type=int, default=defaults["max_frames"])
    parser.add_argument(
        "--show-gt",
        action=argparse.BooleanOptionalAction,
        default=defaults["show_gt"],
    )
    parser.add_argument(
        "--show-prediction",
        action=argparse.BooleanOptionalAction,
        default=defaults["show_prediction"],
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=defaults["score_thresh"],
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=defaults["max_detections"],
    )
    parser.add_argument(
        "--vis-scope",
        choices=SCOPE_CHOICES,
        default=defaults["vis_scope"],
    )
    parser.add_argument(
        "--pred-mode",
        choices=("raw", "final"),
        default=defaults["pred_mode"],
    )
    parser.add_argument(
        "--heatmap-nms-kernel",
        type=int,
        default=defaults["heatmap_nms_kernel"],
    )
    parser.add_argument(
        "--heatmap-score-mode",
        choices=HEATMAP_SCORE_MODES,
        default=defaults["heatmap_score_mode"],
    )
    parser.add_argument(
        "--yolox-nms-iou",
        type=float,
        default=defaults["yolox_nms_iou"],
    )
    parser.add_argument(
        "--box-coordinate-mode",
        choices=("auto", "cartesian"),
        default=defaults["box_coordinate_mode"],
    )
    parser.add_argument(
        "--model-type",
        choices=MODEL_TYPES,
        default=defaults["model_type"],
    )
    parser.add_argument(
        "--gt-object-ignore-override-path",
        default=defaults["gt_object_ignore_override_path"],
    )
    parser.add_argument("--fps", type=float, default=defaults["fps"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=defaults["display"],
    )
    args = parser.parse_args(argv)
    args.gt_object_ignore_override_path = normalize_optional_path(
        args.gt_object_ignore_override_path
    )
    return validate_visualization_config(vars(args))


def main():
    args = parse_args()
    if args.mode in {MODE_RA_MAP, MODE_RA_MAP_VIDEO}:
        from visualization.radar_workflow import run_ra_map

        run_ra_map(args)
        return
    if args.mode in {MODE_MULTISENSOR, MODE_MULTISENSOR_VIDEO}:
        from visualization.multisensor_workflow import run_multisensor

        run_multisensor(args)
        return
    raise AssertionError(f"Unhandled visualization mode: {args.mode}")
