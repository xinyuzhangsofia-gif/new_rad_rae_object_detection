"""Shared data, output, and raw-sensor locations.

Dataset locations have environment-variable overrides so source files do not
need machine-specific edits.  Relative output defaults intentionally preserve
the repository's existing command behavior.
"""

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
# Shared relative output locations.
CHECKPOINT_BASE_DIR = "checkpoints"
LOG_BASE_DIR = "runs"
EVALUATION_PLOTS_BASE_DIR = "evaluation_plots"
EVALUATION_RESULTS_BASE_DIR = "evaluation_results"

# Current K-Radar dataset coverage. These values also retain the existing
# sequence metadata written into training checkpoints.
DEFAULT_KRADAR_SEQUENCE = 11
KRADAR_SEQUENCE_IDS = tuple(range(1, 59))
