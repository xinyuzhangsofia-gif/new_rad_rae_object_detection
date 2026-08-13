"""Configuration used only by ``visualization_based_gt``.

The sensor data root and annotation root are intentionally separate.  The
two supported annotation roots contain the same per-frame text layout, but
their boxes use different coordinate frames.
"""

from dataclasses import dataclass
from pathlib import Path


VISUALIZATION_DIR = Path(__file__).resolve().parent

# Past visualization GT: the official K-Radar revised visibility labels.
OFFICIAL_KRADAR_GT_ROOT = (
    "/home/local/xinyu/kradar_revised_label_v2_1/"
    "KRadar_revised_visibility"
)

# Current GT: the labels currently configured and used by train_cfg.py.
CURRENT_GT_ROOT = (
    "/home/local/xinyu/K-Radar-GT-cartesian-radar-v2"
)

# Current Radar tensors: the same paired RAD/RAE npy files used by train.py.
CURRENT_RADAR_NPY_ROOT = "/home/local/xinyu/K-Radar-RAD"

# Front-camera images. The SMB URI is resolved to its local GVFS mount by
# visualization_utils.py before OpenCV reads it.
CAMERA_RGB_ROOT = "smb://192.168.189.30/elab-share/Datasets/K-Radar-RGB"

GT_KIND_OFFICIAL_KRADAR = "official_kradar_gt"
GT_KIND_CURRENT = "current_gt"

VISUALIZE_MODE_PICTURES = "pictures"
VISUALIZE_MODE_VIDEO = "video"
VISUALIZE_MODES = (VISUALIZE_MODE_PICTURES, VISUALIZE_MODE_VIDEO)

SENSOR_LAYOUT_CAMERA_RADAR = "camera_radar"
SENSOR_LAYOUT_CAMERA_LIDAR_RADAR = "camera_lidar_radar"
SENSOR_LAYOUTS = (
    SENSOR_LAYOUT_CAMERA_RADAR,
    SENSOR_LAYOUT_CAMERA_LIDAR_RADAR,
)

RADAR_VIEW_RAE = "rae"
RADAR_VIEW_RAD = "rad"
RADAR_VIEW_SOURCES = (RADAR_VIEW_RAE, RADAR_VIEW_RAD)

# Root -> parser/coordinate convention.  Adding a copied label root requires
# adding it here as well, because the two file formats look identical and
# therefore cannot be distinguished safely from file contents alone.
INFO_LABEL_KIND_BY_ROOT = {
    OFFICIAL_KRADAR_GT_ROOT: GT_KIND_OFFICIAL_KRADAR,
    CURRENT_GT_ROOT: GT_KIND_CURRENT,
}


@dataclass
class DataConfig:
    # Raw sensor/calibration data root.  It is independent of info_label_root.
    root_dir: str = "/home/local/xinyu/KRadar"

    # Choose one GT source:
    #   OFFICIAL_KRADAR_GT_ROOT -> GT used by the past visualization
    #   CURRENT_GT_ROOT         -> GT currently used by train_cfg.py
    info_label_root: str = CURRENT_GT_ROOT

    # Paired Radar npy input used by train.py.  "rae" collapses elevation to
    # make the RA image; "rad" collapses Doppler instead.
    rad_rae_root: str = CURRENT_RADAR_NPY_ROOT
    radar_view_source: str = RADAR_VIEW_RAE

    # "camera_radar" places Camera above Radar BEV and needs no LiDAR files.
    sensor_layout: str = SENSOR_LAYOUT_CAMERA_RADAR
    camera_rgb_root: str = CAMERA_RGB_ROOT

    # Leave empty for GT-only visualization.  Set a checkpoint to overlay its
    # predictions on the same Radar, LiDAR and Camera frames.
    prediction_checkpoint_path: str = str(
        VISUALIZATION_DIR.parent
        / "checkpoints/sleet/0808_train_seq9_12_first_11_first_3_10_first_test_seq53"
        / "0809_epoch_030.pth"
    )
    prediction_device: str = "auto"  # auto, cpu, cuda, cuda:0, ...
    prediction_score_thresh: float = 0.2
    prediction_max_detections: int = 64
    prediction_mode: str = "final"  # final or raw
    prediction_heatmap_nms_kernel: int = 3
    prediction_heatmap_score_mode: str = "peak_times_local_mean"
    prediction_yolox_nms_iou: float = 0.65

    lidar2radar_calib_path: str = str(
        VISUALIZATION_DIR / "lidar2radar_calib.yml"
    )

    start_frame_idx: int = 0
    sequence: int = 11
    sequences: tuple[int, ...] | None = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    step: int = 1
    fps: int = 10
    # Global text switch. GT boxes remain visible when show_gt_texts is False.
    show_texts: bool = True
    show_gt_texts: bool = False
    show_radar_title: bool = True
    # Overlay colors are names so each renderer can convert them to its own
    # color space (OpenCV BGR, Matplotlib, or Open3D RGB).
    ground_truth_box_color: str = "red"
    prediction_box_color: str = "green"
    radar_ground_truth_linewidth: float = 2.0
    radar_prediction_linewidth: float = 2.0
    max_frames: int | None = None

    camera_mode: int = 1  # 0: single frame, 1: video
    lidar_mode: int = 1  # 0: point cloud, 1: BEV video

    radar_single_frame: bool = True
    radar_mode: int = 2  # 0: polar, 1: Cartesian, 2: Cartesian with yaw
    radar_save_path: str = "ra_cartesian_video.mp4"

    # Combined Radar + LiDAR + Camera output.  Change only visualize_mode:
    #   "pictures" -> save every selected combined frame as an image
    #   "video"    -> save selected combined frames as one MP4
    visualize_mode: str = VISUALIZE_MODE_PICTURES
    save_pictures: bool = True
    picture_save_dir: str = str(VISUALIZATION_DIR / "generated" / "pictures")
    picture_extension: str = ".png"
    all_sensors_save_path: str = "radar_lidar_camera_video_sequence_1.mp4"
    display_window: bool = True

    choose_camera: str = "cam_1"
    lidar_type: str = "os2-64"
    calib_seq: str = "calib_seq_v2"
