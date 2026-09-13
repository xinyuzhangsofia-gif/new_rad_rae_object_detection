"""Shared data, output, and raw-sensor locations.

Dataset locations have environment-variable overrides so source files do not
need machine-specific edits.  Relative output defaults intentionally preserve
the repository's existing command behavior.
"""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADAR_NPY_ROOT = os.environ.get(
    "MVRSS_RADAR_ROOT", "/home/local/xinyu/K-Radar-RAD"
)
CARTESIAN_GT_ROOT = os.environ.get(
    "MVRSS_CARTESIAN_GT_ROOT", "/home/local/xinyu/K-Radar-GT-cartesian-radar-v2"
)
RAW_KRADAR_ROOT = os.environ.get(
    "MVRSS_RAW_KRADAR_ROOT", "/home/local/xinyu/KRadar"
)
KRADAR_TOOLS_ROOT = os.environ.get(
    "MVRSS_KRADAR_TOOLS_ROOT", "/home/local/xinyu/K-Radar"
)
RAW_RADAR_ROOT = os.environ.get(
    "MVRSS_RAW_RADAR_ROOT",
    "/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar",
)
OFFICIAL_KRADAR_GT_ROOT = os.environ.get(
    "MVRSS_OFFICIAL_KRADAR_GT_ROOT",
    "/home/local/xinyu/kradar_revised_label_v2_1/KRadar_revised_visibility",
)
CAMERA_RGB_ROOT = os.environ.get(
    "MVRSS_CAMERA_RGB_ROOT",
    "smb://192.168.189.30/elab-share/Datasets/K-Radar-RGB",
)
LIDAR2RADAR_CALIB_PATH = os.environ.get(
    "MVRSS_LIDAR2RADAR_CALIB_PATH",
    str(PROJECT_ROOT / "lidar2radar_calib.yml"),
)
VISUALIZATION_LIDAR2RADAR_CALIB_PATH = os.environ.get(
    "MVRSS_LIDAR2RADAR_CALIB_PATH",
    str(PROJECT_ROOT / "visualization_based_gt" / "lidar2radar_calib.yml"),
)

# Shared relative output locations.
CHECKPOINT_BASE_DIR = "checkpoints"
LOG_BASE_DIR = "runs"
EVALUATION_PLOTS_BASE_DIR = "evaluation_plots"
EVALUATION_RESULTS_BASE_DIR = "evaluation_results"


@dataclass
class DataConfig:
    root_dir: str = RAW_KRADAR_ROOT
    raw_radar_root: str = RAW_RADAR_ROOT
    lidar2radar_calib_path: str = LIDAR2RADAR_CALIB_PATH

    start_frame_idx: int = 0
    sequence: int = 11
    #sequences: tuple[int, ...] | None =(2,3,4,9,10,11)
    sequences: tuple[int, ...] | None = tuple(range(1, 59))
    step: int = 1
    fps: int = 10
    show_texts: bool = True
    max_frames: int | None = None
    

    camera_mode: int = 1   # 0:visualize with bbx
                           # 1:video with bbx

    lidar_mode: int = 1   # 0:pcd_single
                          # 1:bev_video

    radar_single_frame: bool = True  # True: wait after each frame, False: play by fps
    radar_mode: int = 2   # 0:ra_map_polar
                          # 1:ra_map_cartesian
                          # 2:ra_map_cartesian_with_yaw
    radar_save_path: str = "ra_cartesian_video.mp4"

    all_sensors_mode: int = 1   # 0:single_frame
                                # 1:video
    all_sensors_save_path: str = "radar_lidar_camera_video_sequence_1.mp4"
    

    #maybe don't need to choose in the future
    
    # Raw-sensor tools only; training reads CARTESIAN_GT_ROOT/<sequence>/gt/gt.txt.
    choose_info_label: str = "info_label_rev2"
    choose_camera: str = "cam_1"      # cam_1, cam_2
    lidar_type: str = "os2-64"        # 0: os1-128, 1: os2-64
    calib_seq: str = "calib_seq_v2"          # 0: calib_seq, 1: calib_seq_v2, 2: calib_init
