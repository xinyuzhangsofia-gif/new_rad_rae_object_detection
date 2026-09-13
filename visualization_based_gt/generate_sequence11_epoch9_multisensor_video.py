"""Generate sequence-11 Camera + LiDAR + Cartesian RA video with epoch-9 predictions."""

import argparse
from pathlib import Path

import path_setup  # noqa: F401

from checkpoint_predictor import build_checkpoint_predictor
from radar_npy_reader import build_current_radar_dataset, get_current_radar_axes
from sensor_transformation import load_lidar2radar_calib
from visualization import visualize_all_sensors
from visualization_cfg import (
    DataConfig,
    SENSOR_LAYOUT_CAMERA_LIDAR_RADAR,
    VISUALIZATION_DIR,
    VISUALIZE_MODE_VIDEO,
)
from visualization_utils import (
    get_label_dir,
    get_visualization_camera_dir,
    resolve_smb_mount_path,
)
from data.paths import get_camera_calib_path, get_label_files


CHECKPOINT_PATH = (
    VISUALIZATION_DIR.parent
    / "checkpoints/object_detection/20260815_134341_021343__model_7__seq1-58"
    / "0815_epoch_009.pth"
)
RAW_SENSOR_ROOT = "smb://192.168.189.30/elab-share/Datasets/K-Radar"
DEFAULT_OUTPUT = (
    VISUALIZATION_DIR
    / "generated"
    / "sequence11_epoch009_camera_lidar_cartesian_ra_gt_pred.mp4"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(
        sequence=11,
        sequences=(11,),
        sensor_layout=SENSOR_LAYOUT_CAMERA_LIDAR_RADAR,
        visualize_mode=VISUALIZE_MODE_VIDEO,
        save_pictures=False,
        display_window=False,
        radar_mode=2,
        radar_single_frame=False,
        radar_view_source="rae",
        start_frame_idx=int(args.start_frame),
        max_frames=args.max_frames,
        step=max(1, int(args.step)),
        fps=max(1, int(args.fps)),
        prediction_checkpoint_path=str(CHECKPOINT_PATH),
        prediction_device=str(args.device),
        prediction_score_thresh=float(args.score_thresh),
        show_texts=True,
        show_gt_texts=False,
        show_radar_title=True,
        ground_truth_box_color="green",
        prediction_box_color="red",
        all_sensors_save_path=str(output_path),
    )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)
    camera_dir = get_visualization_camera_dir(cfg)
    lidar_dir = str(
        resolve_smb_mount_path(RAW_SENSOR_ROOT)
        / str(cfg.sequence)
        / cfg.lidar_type
    )
    if not Path(lidar_dir).is_dir():
        raise FileNotFoundError(f"LiDAR directory is unavailable: {lidar_dir}")

    path_calib = get_camera_calib_path(cfg)
    if not Path(path_calib).is_file():
        raise FileNotFoundError(f"Camera calibration is unavailable: {path_calib}")

    radar_dataset = build_current_radar_dataset(cfg)
    arr_range, arr_azimuth_deg, _ = get_current_radar_axes()
    checkpoint_predictor = build_checkpoint_predictor(cfg)
    rotation_lidar_to_radar, translation_lidar_to_radar = (
        load_lidar2radar_calib(cfg.lidar2radar_calib_path)
    )

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Sequence: {cfg.sequence}")
    print(f"Labels: {len(label_files)}")
    print(f"Radar frames: {len(radar_dataset)}")
    print(f"Camera: {camera_dir}")
    print(f"LiDAR: {lidar_dir}")
    print(f"Output: {output_path}")

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
        R_l2r=rotation_lidar_to_radar,
        T_l2r=translation_lidar_to_radar,
        checkpoint_predictor=checkpoint_predictor,
    )


if __name__ == "__main__":
    main()
