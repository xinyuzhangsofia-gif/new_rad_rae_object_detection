"""Orchestrate Camera/Radar and Camera/LiDAR/Radar outputs."""

from data.paths import get_camera_calib_path, get_label_files, get_lidar_dir
from visualization.config import (
    SENSOR_LAYOUT_CAMERA_LIDAR_RADAR,
    build_render_config,
)
from visualization.geometry import load_lidar2radar_calib
from visualization.multisensor import visualize_all_sensors
from visualization.paths import get_label_dir, get_visualization_camera_dir
from visualization.prediction import build_checkpoint_predictor
from visualization.radar_data import (
    build_current_radar_dataset,
    get_current_radar_axes,
)


def run_multisensor(args):
    """Render the configured two- or three-sensor image/video."""
    if args.sequence is None:
        raise ValueError("Multi-sensor visualization requires sequence")
    cfg = build_render_config(args)
    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)
    camera_dir = get_visualization_camera_dir(cfg)
    path_calib = get_camera_calib_path(cfg)
    lidar_dir = (
        get_lidar_dir(cfg)
        if cfg.sensor_layout == SENSOR_LAYOUT_CAMERA_LIDAR_RADAR
        else None
    )
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
