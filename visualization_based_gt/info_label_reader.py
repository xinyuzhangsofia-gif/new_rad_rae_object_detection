"""Read both label coordinate conventions used by the visualizer and train.py."""

from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import torch
import yaml

from visualization_cfg import (
    GT_KIND_CURRENT,
    GT_KIND_OFFICIAL_KRADAR,
    VISUALIZATION_DIR,
)
from visualization_utils import resolve_info_label_kind


_HEADER_INDEX_RE = re.compile(r"idx\([^)]*\)\s*=\s*([^,\s]+)")


def _read_per_frame_label(label_path):
    """Parse the common K-Radar per-frame text structure without conversion."""
    label_path = Path(label_path)
    with label_path.open("r") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        raise ValueError(f"Empty info_label file: {label_path}")

    match = _HEADER_INDEX_RE.search(lines[0])
    if match is None:
        raise ValueError(f"Cannot parse sensor indices from: {label_path}")
    index_parts = match.group(1).split("_")
    if len(index_parts) < 4:
        raise ValueError(
            f"Expected four sensor indices in {label_path}, got {match.group(1)!r}"
        )

    objects = []
    for line_number, line in enumerate(lines[1:], start=2):
        if line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 11:
            raise ValueError(
                f"Expected at least 11 object fields in "
                f"{label_path}:{line_number}, got {len(parts)}"
            )

        try:
            box = torch.tensor(
                [
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    2.0 * float(parts[8]),
                    2.0 * float(parts[9]),
                    2.0 * float(parts[10]),
                    np.deg2rad(float(parts[7])),
                ],
                dtype=torch.float32,
            )
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse object values in {label_path}:{line_number}"
            ) from exc

        objects.append(
            {
                "detec_sensor": parts[1],
                "label": parts[2],
                "cls": parts[3],
                "box": box,
            }
        )

    return {
        "objects": objects,
        "tesseract_idx": index_parts[0],
        "os2_64_idx": index_parts[1],
        "cam_front_idx": index_parts[2],
        "os1_128_idx": index_parts[3],
    }


@lru_cache(maxsize=8)
def _load_lidar_to_radar_calibration(calib_path):
    with Path(calib_path).open("r") as file:
        data = yaml.safe_load(file)

    calibration = data["calib_lidar2radar"]
    rotation = torch.tensor(calibration["R"], dtype=torch.float32)
    translation = torch.tensor(calibration["T"], dtype=torch.float32)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"Invalid LiDAR-to-Radar calibration: {calib_path}")
    if not torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-5):
        raise ValueError(f"LiDAR-to-Radar R is not orthonormal: {calib_path}")
    return rotation, translation


def read_official_kradar_gt(label_path):
    """Read the official K-Radar GT used by the past visualization."""
    frame_info = _read_per_frame_label(label_path)
    for obj in frame_info["objects"]:
        obj["lidar_box"] = obj["box"].clone()
        obj["source_coordinate_frame"] = "lidar"
    frame_info["source_coordinate_frame"] = "lidar"
    return frame_info


def read_current_gt(
    label_path,
    calib_path=VISUALIZATION_DIR / "lidar2radar_calib.yml",
):
    """Read the GT currently used by train_cfg.py.

    The generated training labels store boxes in Radar coordinates.  The
    visualizer historically expects ``obj['box']`` in LiDAR coordinates for
    camera projection and LiDAR overlays, so this reader applies the inverse
    calibration.  The unmodified training box remains available as
    ``obj['radar_box']``.
    """
    rotation, translation = _load_lidar_to_radar_calibration(str(calib_path))
    if not torch.allclose(rotation, torch.eye(3), atol=1e-6):
        raise ValueError(
            "Training Cartesian labels were generated only for identity "
            "LiDAR-to-Radar rotation; a general rotated box cannot be "
            "inverted by changing its center alone."
        )

    frame_info = _read_per_frame_label(label_path)
    for obj in frame_info["objects"]:
        radar_box = obj["box"].clone()
        lidar_box = radar_box.clone()
        # Builder convention: radar_center = lidar_center @ R.T + T.
        lidar_box[:3] = (radar_box[:3] - translation) @ rotation

        obj["radar_box"] = radar_box
        obj["lidar_box"] = lidar_box.clone()
        obj["box"] = lidar_box
        obj["source_coordinate_frame"] = "radar"

    frame_info["source_coordinate_frame"] = "radar"
    return frame_info


def read_info_label(label_path):
    """Dispatch to the correct reader according to the configured label root."""
    label_kind = resolve_info_label_kind(label_path)
    if label_kind == GT_KIND_OFFICIAL_KRADAR:
        return read_official_kradar_gt(label_path)
    if label_kind == GT_KIND_CURRENT:
        return read_current_gt(label_path)
    raise AssertionError(f"Unhandled info_label kind: {label_kind}")
