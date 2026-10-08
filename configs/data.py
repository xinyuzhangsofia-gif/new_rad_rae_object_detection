"""Shared data, output, and raw-sensor locations."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _environment_path(primary_name, legacy_name, default=None):
    """Read a public path setting, with one deprecated alias."""
    return (
        os.environ.get(primary_name)
        or os.environ.get(legacy_name)
        or default
    )


def require_data_path(value, environment_name):
    """Fail clearly when code actually needs an unconfigured data resource."""
    if value in (None, ""):
        raise RuntimeError(
            f"{environment_name} is not configured. "
            f"Set environment variable {environment_name}."
        )
    return str(value)


RADAR_NPY_ROOT = _environment_path(
    "KRADAR_RADAR_ROOT", "MVRSS_RADAR_ROOT"
)
CARTESIAN_GT_ROOT = _environment_path(
    "KRADAR_CARTESIAN_GT_ROOT", "MVRSS_CARTESIAN_GT_ROOT"
)
RAW_KRADAR_ROOT = _environment_path(
    "KRADAR_RAW_ROOT", "MVRSS_RAW_KRADAR_ROOT"
)
KRADAR_TOOLS_ROOT = _environment_path(
    "KRADAR_TOOLS_ROOT", "MVRSS_KRADAR_TOOLS_ROOT"
)
OFFICIAL_KRADAR_GT_ROOT = _environment_path(
    "KRADAR_OFFICIAL_GT_ROOT", "MVRSS_OFFICIAL_KRADAR_GT_ROOT"
)
CAMERA_RGB_ROOT = _environment_path(
    "KRADAR_CAMERA_RGB_ROOT", "MVRSS_CAMERA_RGB_ROOT"
)
LIDAR2RADAR_CALIB_PATH = _environment_path(
    "KRADAR_LIDAR2RADAR_CALIB_PATH",
    "MVRSS_LIDAR2RADAR_CALIB_PATH",
    str(PROJECT_ROOT / "lidar2radar_calib.yml"),
)
# Shared relative output locations.
CHECKPOINT_BASE_DIR = "checkpoints"
LOG_BASE_DIR = "runs"
EVALUATION_PLOTS_BASE_DIR = "evaluation_plots"
EVALUATION_RESULTS_BASE_DIR = "evaluation_results"

# Current K-Radar dataset coverage.
KRADAR_SEQUENCE_IDS = tuple(range(1, 59))
