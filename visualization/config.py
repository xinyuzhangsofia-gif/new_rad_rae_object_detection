"""Validation and semantic constants for the unified visualizer."""

from types import SimpleNamespace


MODE_RA_MAP = "ra_map"
MODE_RA_MAP_VIDEO = "ra_map_video"
MODE_MULTISENSOR = "multisensor"
MODE_MULTISENSOR_VIDEO = "multisensor_video"
VISUALIZATION_MODES = (
    MODE_RA_MAP,
    MODE_RA_MAP_VIDEO,
    MODE_MULTISENSOR,
    MODE_MULTISENSOR_VIDEO,
)

RA_MAP_POLAR = "polar"
RA_MAP_CARTESIAN = "cartesian"
RA_MAP_COORDINATES = (RA_MAP_POLAR, RA_MAP_CARTESIAN)
RA_MAP_POLAR_TITLE = "RA map in Polar with bounding boxes"
RA_MAP_CARTESIAN_TITLE = "RA map in Cartesian with bounding boxes"

GROUND_TRUTH_COLOR = "green"
PREDICTION_COLOR = "red"

GT_KIND_OFFICIAL_KRADAR = "official_kradar_gt"
GT_KIND_CURRENT = "current_gt"

FRAME_OUTPUT_PICTURES = "pictures"
FRAME_OUTPUT_VIDEO = "video"
FRAME_OUTPUT_MODES = (FRAME_OUTPUT_PICTURES, FRAME_OUTPUT_VIDEO)

SENSOR_LAYOUT_CAMERA_RADAR = "camera_radar"
SENSOR_LAYOUT_CAMERA_LIDAR_RADAR = "camera_lidar_radar"
SENSOR_LAYOUTS = (
    SENSOR_LAYOUT_CAMERA_RADAR,
    SENSOR_LAYOUT_CAMERA_LIDAR_RADAR,
)


def validate_visualization_config(config):
    """Return a normalized namespace for the four supported workflows."""
    values = dict(config)
    mode = str(values.get("mode", MODE_RA_MAP)).strip().lower()
    if mode not in VISUALIZATION_MODES:
        raise ValueError(
            f"Unknown visualization mode {mode!r}; expected one of "
            f"{VISUALIZATION_MODES}."
        )

    coordinate = str(
        values.get("ra_map_coordinate", RA_MAP_POLAR)
    ).strip().lower()
    if coordinate not in RA_MAP_COORDINATES:
        raise ValueError(
            f"Unknown RA-map coordinate {coordinate!r}; expected one of "
            f"{RA_MAP_COORDINATES}."
        )

    values["mode"] = mode
    values["ra_map_coordinate"] = coordinate
    values["show_gt"] = bool(values.get("show_gt", True))
    values["show_prediction"] = bool(values.get("show_prediction", True))
    values["ground_truth_color"] = GROUND_TRUTH_COLOR
    values["prediction_color"] = PREDICTION_COLOR
    return SimpleNamespace(**values)


def load_visualization_config(**overrides):
    """Load the canonical root configuration with optional test overrides."""
    from visualize_cfg import VISUALIZE_CONFIG

    values = dict(VISUALIZE_CONFIG)
    values.update(overrides)
    return validate_visualization_config(values)
