"""Validation and semantic constants for the unified visualizer."""

from pathlib import Path
from types import SimpleNamespace


MODE_RA_MAP = "ra_map"
MODE_RA_MAP_VIDEO = "ra_map_video"
MODE_MULTISENSOR = "multisensor"
MODE_MULTISENSOR_VIDEO = "multisensor_video"
VISUALIZATION_MODES = (
    MODE_RA_MAP,
    MODE_RA_MAP_VIDEO,
    MODE_MULTISENSOR,
    MODE_MULTISENSOR_VIDEO,
)

RA_MAP_POLAR = "polar"
RA_MAP_CARTESIAN = "cartesian"
RA_MAP_COORDINATES = (RA_MAP_POLAR, RA_MAP_CARTESIAN)
RA_MAP_POLAR_TITLE = "RA map in Polar with bounding boxes"
RA_MAP_CARTESIAN_TITLE = "RA map in Cartesian with bounding boxes"

GROUND_TRUTH_COLOR = "green"
PREDICTION_COLOR = "red"

GT_KIND_OFFICIAL_KRADAR = "official_kradar_gt"
GT_KIND_CURRENT = "current_gt"

FRAME_OUTPUT_PICTURES = "pictures"
FRAME_OUTPUT_VIDEO = "video"
FRAME_OUTPUT_MODES = (FRAME_OUTPUT_PICTURES, FRAME_OUTPUT_VIDEO)

SENSOR_LAYOUT_CAMERA_RADAR = "camera_radar"
SENSOR_LAYOUT_CAMERA_LIDAR_RADAR = "camera_lidar_radar"
SENSOR_LAYOUTS = (
    SENSOR_LAYOUT_CAMERA_RADAR,
    SENSOR_LAYOUT_CAMERA_LIDAR_RADAR,
)


def validate_visualization_config(config):
    """Return a normalized namespace for the four supported workflows."""
    values = dict(config)
    mode = str(values.get("mode", MODE_RA_MAP)).strip().lower()
    if mode not in VISUALIZATION_MODES:
        raise ValueError(
            f"Unknown visualization mode {mode!r}; expected one of "
            f"{VISUALIZATION_MODES}."
        )

    coordinate = str(
        values.get("ra_map_coordinate", RA_MAP_POLAR)
    ).strip().lower()
    if coordinate not in RA_MAP_COORDINATES:
        raise ValueError(
            f"Unknown RA-map coordinate {coordinate!r}; expected one of "
            f"{RA_MAP_COORDINATES}."
        )

    values["mode"] = mode
    values["ra_map_coordinate"] = coordinate
    sensor_layout = str(
        values.get("sensor_layout", SENSOR_LAYOUT_CAMERA_LIDAR_RADAR)
    ).strip().lower()
    if sensor_layout not in SENSOR_LAYOUTS:
        raise ValueError(
            f"Unknown sensor layout {sensor_layout!r}; expected one of "
            f"{SENSOR_LAYOUTS}."
        )

    values["sensor_layout"] = sensor_layout
    values["show_gt"] = bool(values.get("show_gt", True))
    values["show_prediction"] = bool(values.get("show_prediction", True))
    values["ground_truth_color"] = GROUND_TRUTH_COLOR
    values["prediction_color"] = PREDICTION_COLOR
    return SimpleNamespace(**values)


def load_visualization_config(**overrides):
    """Load the canonical root configuration with optional test overrides."""
    from visualize_cfg import VISUALIZE_CONFIG

    values = dict(VISUALIZE_CONFIG)
    values.update(overrides)
    return validate_visualization_config(values)


def build_render_config(args):
    """Build the normalized runtime configuration shared by both workflows."""
    output_root = Path(args.output_dir).expanduser()
    is_single_frame = args.mode in {MODE_RA_MAP, MODE_MULTISENSOR}
    checkpoint_path = str(args.checkpoint_path or "")
    if args.show_prediction and not checkpoint_path:
        raise ValueError(
            "show_prediction=True requires checkpoint_path in visualize_cfg.py"
        )

    return SimpleNamespace(
        raw_sensor_root=args.raw_sensor_root,
        info_label_root=args.info_label_root,
        radar_npy_root=args.radar_npy_root,
        radar_view_source=args.radar_view_source,
        sensor_layout=args.sensor_layout,
        ra_map_coordinate=args.ra_map_coordinate,
        camera_rgb_root=args.camera_rgb_root,
        prediction_checkpoint_path=(
            checkpoint_path if args.show_prediction else ""
        ),
        prediction_device=args.prediction_device,
        prediction_score_thresh=args.score_thresh,
        prediction_max_detections=args.max_detections,
        prediction_mode=args.pred_mode,
        prediction_heatmap_nms_kernel=args.heatmap_nms_kernel,
        prediction_heatmap_score_mode=args.heatmap_score_mode,
        prediction_yolox_nms_iou=args.yolox_nms_iou,
        lidar2radar_calib_path=args.lidar2radar_calib_path,
        start_frame_idx=args.frame,
        sequence=args.sequence,
        step=args.frame_step,
        fps=args.fps,
        show_gt=args.show_gt,
        show_texts=args.show_texts,
        show_gt_texts=args.show_gt_texts,
        ground_truth_box_color=args.ground_truth_color,
        prediction_box_color=args.prediction_color,
        max_frames=1 if is_single_frame else args.max_frames,
        radar_mode=0 if args.ra_map_coordinate == RA_MAP_POLAR else 2,
        visualize_mode=(
            FRAME_OUTPUT_PICTURES
            if is_single_frame
            else FRAME_OUTPUT_VIDEO
        ),
        picture_save_dir=str(output_root / "multisensor"),
        picture_extension=".png",
        multisensor_video_path=str(
            output_root
            / "multisensor_video"
            / (
                f"sequence_{int(args.sequence):02d}_{args.sensor_layout}_"
                f"{args.ra_map_coordinate}.mp4"
            )
        ),
        display_window=args.display,
        camera_name=args.camera_name,
        lidar_type=args.lidar_type,
        camera_calibration_set=args.camera_calibration_set,
    )
