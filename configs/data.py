"""Shared data locations and settings for raw-sensor visualization.

Set MVRSS_RADAR_ROOT and MVRSS_CARTESIAN_GT_ROOT before starting a command to
use another machine's data without editing consumers of these paths.
"""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADAR_NPY_ROOT = os.environ.get("MVRSS_RADAR_ROOT", "/home/local/xinyu/K-Radar-RAD")
CARTESIAN_GT_ROOT = os.environ.get(
    "MVRSS_CARTESIAN_GT_ROOT", "/home/local/xinyu/K-Radar-GT-cartesian-radar-v2"
)

@dataclass
class DataConfig:
    root_dir: str = "/home/local/xinyu/KRadar"
    lidar2radar_calib_path: str = str(PROJECT_ROOT / "lidar2radar_calib.yml")

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
