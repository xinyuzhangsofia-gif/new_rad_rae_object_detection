import path_setup

from data.labels import *
from data.paths import *
from sensor_transformation import *
from visualization import *
from visualization_cfg import DataConfig
from visualization_utils import get_label_dir, get_visualization_camera_dir
from radar_npy_reader import build_current_radar_dataset, get_current_radar_axes
from checkpoint_predictor import build_checkpoint_predictor


if __name__ == "__main__":

    cfg = DataConfig()

    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)

    camera_dir = get_visualization_camera_dir(cfg)
    path_calib = get_camera_calib_path(cfg)

    lidar_dir = get_lidar_dir(cfg)

    radar_dataset = build_current_radar_dataset(cfg)
    arr_range, arr_azimuth_deg, arr_elevation_deg = get_current_radar_axes()
    checkpoint_predictor = build_checkpoint_predictor(cfg)

    R_l2r, T_l2r = load_lidar2radar_calib(cfg.lidar2radar_calib_path)

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
        R_l2r=R_l2r,
        T_l2r=T_l2r,
        checkpoint_predictor=checkpoint_predictor
    )
