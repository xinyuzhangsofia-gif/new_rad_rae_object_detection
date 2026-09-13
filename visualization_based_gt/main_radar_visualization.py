import path_setup

from data.labels import *
from data.paths import *
from sensor_transformation import *
from visualization import *
from visualization_cfg import DataConfig
from visualization_utils import get_label_dir
from radar_npy_reader import build_current_radar_dataset, get_current_radar_axes

def get_radar_common_data(cfg):
    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)
    radar_dataset = build_current_radar_dataset(cfg)
    arr_range, arr_azimuth_deg, arr_elevation_deg = get_current_radar_axes()
    R_l2r, T_l2r = load_lidar2radar_calib(cfg.lidar2radar_calib_path)

    return (
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        R_l2r,
        T_l2r
    )

def get_radar_max_frames(cfg, label_files):
    if cfg.max_frames is None:
        return len(label_files)

    return min(cfg.max_frames, len(label_files))


if __name__ == "__main__":

    cfg = DataConfig()

    (
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        R_l2r,
        T_l2r
    ) = get_radar_common_data(cfg)

    max_frames = get_radar_max_frames(cfg, label_files)

    if cfg.radar_single_frame:
        play_ra_frames_by_step(
            label_dir,
            label_files,
            radar_dataset,
            arr_range,
            arr_azimuth_deg,
            max_frames,
            R_l2r,
            T_l2r,
            cfg.radar_mode,
            cfg.start_frame_idx,
            cfg.step,
            cfg.show_texts
        )

    else:
        if cfg.radar_mode == 0:
            frames = preload_ra_polar_frames(
                label_dir,
                label_files,
                radar_dataset,
                arr_range,
                arr_azimuth_deg,
                max_frames,
                R_l2r,
                T_l2r,
                cfg.start_frame_idx,
                cfg.show_texts
            )

            play_ra_polar_frames(
                frames=frames,
                fps=cfg.fps,
                save_path=cfg.radar_save_path
            )

        elif cfg.radar_mode == 1:
            frames = preload_ra_cartesian_frames(
                label_dir,
                label_files,
                radar_dataset,
                arr_range,
                arr_azimuth_deg,
                max_frames,
                R_l2r,
                T_l2r,
                cfg.start_frame_idx,
                show_texts=cfg.show_texts
            )

            play_ra_cartesian_frames(
                frames=frames,
                fps=cfg.fps,
                save_path=cfg.radar_save_path
            )

        elif cfg.radar_mode == 2:
            frames = preload_ra_cartesian_frames_with_yaw(
                label_dir,
                label_files,
                radar_dataset,
                arr_range,
                arr_azimuth_deg,
                max_frames,
                R_l2r,
                T_l2r,
                cfg.start_frame_idx,
                show_texts=cfg.show_texts
            )

            play_ra_cartesian_frames(
                frames=frames,
                fps=cfg.fps,
                save_path=cfg.radar_save_path
            )

        else:
            raise ValueError(f"Unknown radar_mode: {cfg.radar_mode}")
