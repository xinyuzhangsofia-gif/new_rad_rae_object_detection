"""Readers for flat Cartesian GT and K-Radar per-frame Cartesian labels."""

import numpy as np
import torch
from collections import defaultdict
import glob
import os

from configs.data import CARTESIAN_GT_ROOT
from .paths import get_cartesian_gt_path


CARTESIAN_GT_COLUMNS = (
    "frame_idx", "object_label", "x", "y", "z", "x_width", "y_width",
    "z_width", "yaw_deg", "class",
)


def read_info_label(label_path):
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

        #read the first row and get the index relationship betweeen radar,camera,lidar
        header = lines[0]
        idx_str = header.split('=')[-2]
        idx_part = idx_str.split('_')

        tesseract_idx = idx_part[0]
        os2_64_idx = idx_part[1]
        cam_front_idx = idx_part[2]
        os1_128_idx = idx_part[3]

        #read the data about the bbx
        objects = []
        for line in lines[1:]:  # Skip the first line (header)
            parts = [p.strip() for p in line.split(',')]
            
            detec_sensor=parts[1]
            label=parts[2]
            cls=parts[3]
            x=float(parts[4])
            y=float(parts[5])
            z=float(parts[6])
            yaw=float(parts[7])*np.pi/180.0  # Convert yaw from degrees to radians
            l=2*float(parts[8])
            w=2*float(parts[9])
            h=2*float(parts[10])

            box=torch.tensor([x, y, z, l, w, h, yaw],dtype=torch.float32)

            objects.append({
                'detec_sensor':detec_sensor,
                'label':label,
                'cls': cls,
                'box': box
            })        
    return {
        'objects': objects,
        'tesseract_idx':tesseract_idx,
        'os2_64_idx':os2_64_idx,
        'cam_front_idx':cam_front_idx,
        'os1_128_idx': os1_128_idx
    }


def read_cartesian_gt_txt(gt_txt_path):
    """Read radar-aligned Cartesian GT with one-based frame indices.

    Each non-comment line is:
    ``frame_idx, object_label, x, y, z, x_width, y_width, z_width, yaw_deg, cls``.
    The first field is one-based radar dataset order. Dimensions are full
    metric dimensions in metres. A Cartesian column header is required to
    prevent silently interpreting legacy Polar rows as metric boxes.
    """
    gt_by_file_idx = defaultdict(list)

    with open(gt_txt_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    columns = (
        tuple(part.strip() for part in lines[0][1:].split(","))
        if lines and lines[0].startswith("#") else ()
    )
    if columns != CARTESIAN_GT_COLUMNS:
        raise ValueError(
            f"Cartesian GT requires the column header '# {','.join(CARTESIAN_GT_COLUMNS)}': "
            f"{gt_txt_path}. Polar or untyped GT input is not supported."
        )

    for line in lines:
        if line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 10:
            raise ValueError(
                f"Expected 10 values in Cartesian gt line, got {len(parts)}: {line}"
            )

        gt_frame_idx = int(parts[0])
        file_idx = gt_frame_idx - 1
        object_label = int(parts[1])

        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        x_width = float(parts[5])
        y_width = float(parts[6])
        z_width = float(parts[7])
        yaw = float(parts[8])
        yaw_rad = yaw * np.pi / 180.0
        cls = parts[9]

        if not np.all(np.isfinite([x, y, z, x_width, y_width, z_width, yaw])):
            raise ValueError(f"Non-finite Cartesian GT values: {line}")
        if min(x_width, y_width, z_width) <= 0.0:
            raise ValueError(
                "Cartesian GT dimensions must be positive full dimensions: "
                f"{line}"
            )

        box_metric = torch.tensor(
            [x, y, z, x_width, y_width, z_width, yaw_rad],
            dtype=torch.float32,
        )
        obj = {
            "gt_frame_idx": gt_frame_idx,
            "file_idx": file_idx,
            "object_label": object_label,
            "cls": cls,
            "class_id": object_label,
            "box_metric": box_metric,
            "raw": {
                "x": x,
                "y": y,
                "z": z,
                "x_width": x_width,
                "y_width": y_width,
                "z_width": z_width,
                "yaw": yaw,
                "yaw_rad": yaw_rad,
            },
        }
        gt_by_file_idx[file_idx].append(obj)

    return gt_by_file_idx


def read_kradar_revised_label_dir(
        label_root,
        sequence,
        radar_visibility_tokens=("R", "LR"),
    ):
    """Read official K-Radar v2.1 Cartesian labels keyed by tesseract frame name.

    The official files store ``l/2, w/2, h/2``. ``read_info_label`` doubles
    those values, so every returned ``box_metric`` is
    ``[x, y, z, length, width, height, yaw_radian]`` in meters/radians.
    """
    sequence_dir = os.path.join(str(label_root), str(int(sequence)))
    if not os.path.isdir(sequence_dir):
        raise FileNotFoundError(
            f"K-Radar Cartesian label sequence directory not found: {sequence_dir}"
        )

    label_paths = sorted(glob.glob(os.path.join(sequence_dir, "*.txt")))
    if len(label_paths) == 0:
        raise FileNotFoundError(
            f"No K-Radar Cartesian label txt files found in: {sequence_dir}"
        )

    allowed_tokens = {
        str(token).strip()
        for token in radar_visibility_tokens
    }
    gt_by_frame_name = defaultdict(list)
    seen_frame_names = set()

    for label_path in label_paths:
        frame_info = read_info_label(label_path)
        # Keep the original tesseract parsing path.  The header contains both
        # the label-index '=' and the later timestamp '=', so the original
        # ``header.split('=')[-2]`` correctly selects the label index.
        frame_name = str(frame_info["tesseract_idx"])
        filename_frame_name = os.path.basename(label_path).split("_", 1)[0]
        if frame_name != filename_frame_name:
            raise ValueError(
                "K-Radar label filename/header mismatch: "
                f"filename={os.path.basename(label_path)!r}, "
                f"header_tesseract_idx={frame_name!r}"
            )
        if frame_name in seen_frame_names:
            raise ValueError(
                "Duplicate K-Radar tesseract frame label "
                f"{frame_name!r} in {sequence_dir}"
            )
        seen_frame_names.add(frame_name)
        # Keep empty radar-visible frames in the mapping as an explicit empty
        # annotation instead of making them indistinguishable from missing files.
        gt_by_frame_name[frame_name]

        for obj in frame_info["objects"]:
            visibility = str(obj["detec_sensor"]).strip()
            if visibility not in allowed_tokens:
                continue

            try:
                object_label = int(obj["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid object label {obj['label']!r} in {label_path}"
                ) from exc

            gt_by_frame_name[frame_name].append({
                "object_label": object_label,
                "class_id": object_label,
                "cls": str(obj["cls"]),
                "detec_sensor": visibility,
                "box_metric": obj["box"].clone().to(torch.float32),
                "label_path": label_path,
            })

    return gt_by_frame_name


def load_cartesian_gt(sequence, cartesian_gt_root=None):
    """Load the active Cartesian label format and report its mapping key."""
    flat_path = get_cartesian_gt_path(sequence, cartesian_gt_root)
    if os.path.isfile(flat_path):
        return "file_idx", read_cartesian_gt_txt(flat_path)

    per_frame = read_kradar_revised_label_dir(
        label_root=cartesian_gt_root or CARTESIAN_GT_ROOT,
        sequence=sequence,
        radar_visibility_tokens=("R", "LR"),
    )
    return "frame_name", per_frame
