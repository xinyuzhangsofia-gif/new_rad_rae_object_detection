"""Canonical configuration for ``python visualize.py``.

RAD and RAE are the paired current ``.npy`` inputs. ``polar`` and
``cartesian`` describe only how their range-azimuth map is displayed.
"""

from configs.data import (
    CAMERA_RGB_ROOT,
    LIDAR2RADAR_CALIB_PATH,
    OFFICIAL_KRADAR_GT_ROOT,
    RADAR_NPY_ROOT,
    RAW_KRADAR_ROOT,
)
from data.coordinates import SCOPE_FULL


VISUALIZE_CONFIG = {
    # ra_map | ra_map_video | multisensor | multisensor_video
    "mode": "ra_map",
    # polar | cartesian; also controls the radar panel in multi-sensor output.
    "ra_map_coordinate": "polar",

    "checkpoint_path": (
        "checkpoints/overcast/0728_train_seq9_13_test_seq22/"
        "0728_epoch_012.pth"
    ),
    "prediction_device": "auto",
    "sequence": 22,
    "frame": 0,
    "frame_step": 3,
    "max_frames": 0,  # 0 means every remaining frame in video modes.
    "show_gt": True,
    "show_prediction": True,
    "show_texts": True,
    "show_gt_texts": False,

    "score_thresh": 0.1,
    "max_detections": 64,
    "vis_scope": SCOPE_FULL,
    "pred_mode": "final",
    "heatmap_nms_kernel": 3,
    "heatmap_score_mode": "peak_only",
    "yolox_nms_iou": 0.5,
    "box_coordinate_mode": "auto",
    "model_type": "auto",
    "gt_object_ignore_override_path": None,
    "ignore_class_names": (
        "Bus or Truck",
        "Pedestrian",
        "Pedestrian Group",
        "Bicycle",
        "Bicycle Group",
        "Motorcycle",
    ),

    # Current RAD/RAE and multi-sensor inputs.
    "radar_npy_root": RADAR_NPY_ROOT,
    "radar_view_source": "rae",  # rae or rad
    # Multi-sensor projection needs the per-frame sensor indices stored here.
    "info_label_root": OFFICIAL_KRADAR_GT_ROOT,
    "raw_sensor_root": RAW_KRADAR_ROOT,
    "camera_rgb_root": CAMERA_RGB_ROOT,
    "lidar2radar_calib_path": LIDAR2RADAR_CALIB_PATH,
    "camera_name": "cam_1",
    "lidar_type": "os2-64",
    "camera_calibration_set": "calib_seq_v2",

    "fps": 10,
    "output_dir": "visualization_results",
    "display": True,
}
