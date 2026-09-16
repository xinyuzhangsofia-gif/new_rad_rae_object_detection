"""Orchestrate one Camera/LiDAR/Radar frame or video."""

from pathlib import Path
from types import SimpleNamespace

from data.paths import get_camera_calib_path, get_label_files, get_lidar_dir
from visualization.config import (
    FRAME_OUTPUT_PICTURES,
    FRAME_OUTPUT_VIDEO,
    MODE_MULTISENSOR,
    MODE_RA_MAP,
    RA_MAP_POLAR,
    SENSOR_LAYOUT_CAMERA_LIDAR_RADAR,
)
from visualization.geometry import load_lidar2radar_calib
from visualization.multisensor import visualize_all_sensors
from visualization.paths import get_label_dir, get_visualization_camera_dir
from visualization.prediction import build_checkpoint_predictor
from visualization.radar_data import (
    build_current_radar_dataset,
    get_current_radar_axes,
)


def build_render_config(args):
    output_root = Path(args.output_dir).expanduser()
    is_single_frame = args.mode in {MODE_RA_MAP, MODE_MULTISENSOR}
    checkpoint_path = str(args.checkpoint_path or "")
    if args.show_prediction and not checkpoint_path:
        raise ValueError(
            "show_prediction=True requires checkpoint_path in visualize_cfg.py"
        )

    return SimpleNamespace(
        root_dir=args.raw_sensor_root,
        info_label_root=args.info_label_root,
        radar_npy_root=args.radar_npy_root,
        radar_view_source=args.radar_view_source,
        sensor_layout=SENSOR_LAYOUT_CAMERA_LIDAR_RADAR,
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
        show_radar_title=True,
        ground_truth_box_color=args.ground_truth_color,
        prediction_box_color=args.prediction_color,
        radar_ground_truth_linewidth=2.0,
        radar_prediction_linewidth=2.0,
        max_frames=1 if is_single_frame else args.max_frames,
        radar_mode=0 if args.ra_map_coordinate == RA_MAP_POLAR else 2,
        visualize_mode=(
            FRAME_OUTPUT_PICTURES
            if is_single_frame
            else FRAME_OUTPUT_VIDEO
        ),
        save_pictures=is_single_frame,
        picture_save_dir=str(output_root / "multisensor"),
        picture_extension=".png",
        all_sensors_save_path=str(
            output_root
            / "multisensor_video"
            / f"sequence_{int(args.sequence):02d}_{args.ra_map_coordinate}.mp4"
        ),
        display_window=args.display,
        choose_camera=args.camera_name,
        lidar_type=args.lidar_type,
        calib_seq=args.camera_calibration_set,
    )


def run_multisensor(args):
    """Render the configured three-sensor single frame or video."""
    if args.sequence is None:
        raise ValueError("Multi-sensor visualization requires sequence")
    cfg = build_render_config(args)
    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)
    camera_dir = get_visualization_camera_dir(cfg)
    path_calib = get_camera_calib_path(cfg)
    lidar_dir = get_lidar_dir(cfg)
    radar_dataset = build_current_radar_dataset(cfg)
    arr_range, arr_azimuth_deg, _ = get_current_radar_axes()
    checkpoint_predictor = build_checkpoint_predictor(cfg)
    rotation, translation = load_lidar2radar_calib(
        cfg.lidar2radar_calib_path
    )
    visualize_all_sensors(
        cfg=cfg,
        label_dir=label_dir,
        label_files=label_files,
        camera_dir=camera_dir,
        path_calib=path_calib,
        lidar_dir=lidar_dir,
        radar_dataset=radar_dataset,
        arr_range=arr_range,
        arr_azimuth_deg=arr_azimuth_deg,
        R_l2r=rotation,
        T_l2r=translation,
        checkpoint_predictor=checkpoint_predictor,
    )
