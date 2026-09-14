#!/usr/bin/env python3
"""Build a K-Radar Cartesian GT root keyed by radar tesseract frame names.

The output keeps radar-aligned per-frame revised labels for sensor visualization
and also writes the canonical training/evaluation file
``<sequence>/gt/gt.txt``, parsed by ``read_cartesian_gt_txt``.

Flat Cartesian format::

    frame_idx,object_label,x,y,z,x_width,y_width,z_width,yaw_deg,class

``frame_idx`` is the one-based ordinal of the shared, sorted radar/rae npy
files. The exact ``frame_idx`` to tesseract filename mapping is recorded in
``frame_manifest.csv``. This command does not produce Polar GT.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import (
    CARTESIAN_GT_ROOT,
    OFFICIAL_KRADAR_GT_ROOT,
    RADAR_NPY_ROOT,
    VISUALIZATION_LIDAR2RADAR_CALIB_PATH,
)


HEADER_INDEX_RE = re.compile(r"=\s*([^,\s]+)")
DEFAULT_CALIB_PATH = Path(VISUALIZATION_LIDAR2RADAR_CALIB_PATH)

# In the revised visibility labels, LR means that the object is visible to
# both LiDAR and radar.  A radar GT must therefore keep both R and LR.
RADAR_VISIBLE_TAGS = {"R", "LR"}


def numeric_name_key(name: str):
    return (0, int(name)) if name.isdigit() else (1, name)


def parse_revised_label(path: Path):
    """Return the tesseract frame name and radar-visible Cartesian objects.

    Both ``R`` and ``LR`` are radar-visible.  Lidar-only visibility tags
    (``L``, ``L1`` and other variants) are not included in radar GT.
    """
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty revised label file: {path}")

    match = HEADER_INDEX_RE.search(lines[0])
    if match is None:
        raise ValueError(f"Cannot parse frame index from header: {path}")
    index_parts = match.group(1).split("_")
    if len(index_parts) < 4:
        raise ValueError(f"Invalid frame index in header: {path}")

    frame_name = index_parts[0]
    filename_frame_name = path.name.split("_", 1)[0]
    if frame_name != filename_frame_name:
        raise ValueError(
            f"Filename/header mismatch in {path.name}: "
            f"filename={filename_frame_name}, header={frame_name}"
        )

    objects = []
    for line_number, line in enumerate(lines[1:], start=2):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 11:
            raise ValueError(
                f"Expected at least 11 fields in {path}:{line_number}, "
                f"got {len(parts)}"
            )
        if parts[1] not in RADAR_VISIBLE_TAGS:
            continue

        # -1 is a valid annotation identifier for this workflow.  It is a
        # tracking/association value, not a class label, so keep the bbox.
        object_label = int(parts[2])

        # Official v2.1 stores half dimensions in columns 8, 9 and 10.
        objects.append(
            {
                "object_label": object_label,
                "class_name": parts[3],
                "x_m": float(parts[4]),
                "y_m": float(parts[5]),
                "z_m": float(parts[6]),
                "length_m": 2.0 * float(parts[8]),
                "width_m": 2.0 * float(parts[9]),
                "height_m": 2.0 * float(parts[10]),
                "yaw_deg": float(parts[7]),
                "yaw_rad": math.radians(float(parts[7])),
            }
        )
    return frame_name, objects


def collect_names(directory: Path, suffix: str):
    return {
        path.stem: path
        for path in directory.glob(f"*{suffix}")
        if path.is_file()
    }


def load_lidar2radar_calibration(path: Path):
    """Load the same calibration used by visualization_based_gt."""
    with path.open("r") as file:
        data = yaml.safe_load(file)

    calibration = data["calib_lidar2radar"]
    rotation = np.asarray(calibration["R"], dtype=np.float64)
    translation = np.asarray(calibration["T"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"Invalid lidar-to-radar calibration in {path}")
    return rotation, translation


def transform_box_lidar_to_radar(obj, rotation, translation):
    """Apply the visualization_based_gt Cartesian transform to one box.

    The current calibration has identity rotation.  In that case, this is
    exactly equivalent to transforming all eight corners and reconstructing
    the same Cartesian box: only the center is translated; dimensions and
    yaw are unchanged.
    """
    center_lidar = np.asarray(
        [obj["x_m"], obj["y_m"], obj["z_m"]], dtype=np.float64
    )
    center_radar = center_lidar @ rotation.T + translation

    if not np.allclose(rotation, np.eye(3), atol=1e-8):
        raise ValueError(
            "The current Cartesian box format cannot represent a general "
            "3-D extrinsic rotation exactly. The visualization calibration "
            "must have identity R, or the box must be fitted from transformed "
            "corners first."
        )

    transformed = dict(obj)
    transformed["x_m"] = float(center_radar[0])
    transformed["y_m"] = float(center_radar[1])
    transformed["z_m"] = float(center_radar[2])
    return transformed


def transform_revised_label_file(label_path, output_path, rotation, translation):
    """Write a radar-aligned copy containing only R/ LR objects."""
    output_lines = []
    for line_number, raw_line in enumerate(
        label_path.read_text().splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("*") is False:
            output_lines.append(raw_line)
            continue

        parts = [part.strip() for part in line.split(",")]
        # The header starts with '* idx(...)=...' and is not an object row.
        if len(parts) < 11 or parts[0] != "*":
            output_lines.append(raw_line)
            continue

        if parts[1] not in RADAR_VISIBLE_TAGS:
            continue
        try:
            center_lidar = np.asarray(
                [float(parts[4]), float(parts[5]), float(parts[6])],
                dtype=np.float64,
            )
            center_radar = center_lidar @ rotation.T + translation
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse Cartesian object in {label_path}:{line_number}"
            ) from exc

        if not np.allclose(rotation, np.eye(3), atol=1e-8):
            raise ValueError(
                "The current Cartesian box format cannot represent a general "
                "3-D extrinsic rotation exactly; expected identity R."
            )

        parts[4] = f"{center_radar[0]:.10f}"
        parts[5] = f"{center_radar[1]:.10f}"
        parts[6] = f"{center_radar[2]:.10f}"
        output_lines.append(", ".join(parts))

    output_path.write_text("\n".join(output_lines) + "\n")


def build_sequence(
    sequence: str,
    source_root: Path,
    radar_root: Path,
    output_root: Path,
    rotation,
    translation,
):
    source_dir = source_root / sequence
    radar_dir = radar_root / sequence / "rad"
    rae_dir = radar_root / sequence / "rae"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing revised GT sequence directory: {source_dir}")
    if not radar_dir.is_dir() or not rae_dir.is_dir():
        raise FileNotFoundError(f"Missing radar/rae sequence directory for {sequence}")

    label_paths = sorted(source_dir.glob("*.txt"))
    labels_by_frame = {}
    for label_path in label_paths:
        frame_name, objects = parse_revised_label(label_path)
        if frame_name in labels_by_frame:
            raise ValueError(f"Duplicate label frame {frame_name} in {source_dir}")
        labels_by_frame[frame_name] = (label_path, objects)

    rad_by_name = collect_names(radar_dir, ".npy")
    rae_by_name = collect_names(rae_dir, ".npy")
    shared_names = sorted(set(rad_by_name) & set(rae_by_name), key=numeric_name_key)
    if not shared_names:
        raise ValueError(f"No shared rad/rae files for sequence {sequence}")

    sequence_output = output_root / sequence
    sequence_output.mkdir(parents=True)
    flat_output = sequence_output / "gt"
    flat_output.mkdir()

    # Keep per-frame files at the sequence root, but make their Cartesian
    # centers use the same LiDAR-to-radar transform as the visualization.
    for label_path, _ in labels_by_frame.values():
        transform_revised_label_file(
            label_path=label_path,
            output_path=sequence_output / label_path.name,
            rotation=rotation,
            translation=translation,
        )

    flat_path = flat_output / "gt.txt"
    with flat_path.open("w", newline="") as flat_file:
        flat_file.write(
            "# frame_idx,object_label,x,y,z,x_width,y_width,z_width,yaw_deg,class\n"
        )
        writer = csv.writer(flat_file, lineterminator="\n")
        shared_name_to_frame_idx = {
            frame_name: dataset_idx + 1
            for dataset_idx, frame_name in enumerate(shared_names)
        }
        for frame_name in shared_names:
            label_info = labels_by_frame.get(frame_name)
            if label_info is None:
                continue
            _, objects = label_info
            for obj in objects:
                obj = transform_box_lidar_to_radar(
                    obj=obj,
                    rotation=rotation,
                    translation=translation,
                )
                writer.writerow(
                    [
                        shared_name_to_frame_idx[frame_name],
                        obj["object_label"],
                        f'{obj["x_m"]:.10f}',
                        f'{obj["y_m"]:.10f}',
                        f'{obj["z_m"]:.10f}',
                        f'{obj["length_m"]:.10f}',
                        f'{obj["width_m"]:.10f}',
                        f'{obj["height_m"]:.10f}',
                        f'{obj["yaw_deg"]:.10f}',
                        obj["class_name"],
                    ]
                )

    manifest_path = sequence_output / "frame_manifest.csv"
    missing_gt = 0
    matched_empty = 0
    matched_with_objects = 0
    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.writer(manifest_file, lineterminator="\n")
        writer.writerow(
            [
                "dataset_idx",
                "frame_name",
                "rad_file",
                "rae_file",
                "label_file",
                "status",
                "radar_visible_object_count",
            ]
        )
        for dataset_idx, frame_name in enumerate(shared_names):
            label_info = labels_by_frame.get(frame_name)
            if label_info is None:
                label_name = ""
                status = "missing_gt_file"
                object_count = ""
                missing_gt += 1
            else:
                label_path, objects = label_info
                label_name = label_path.name
                object_count = len(objects)
                if objects:
                    status = "matched_with_objects"
                    matched_with_objects += 1
                else:
                    status = "matched_empty"
                    matched_empty += 1
            writer.writerow(
                [
                    dataset_idx,
                    frame_name,
                    f"rad/{rad_by_name[frame_name].name}",
                    f"rae/{rae_by_name[frame_name].name}",
                    label_name,
                    status,
                    object_count,
                ]
            )

    return {
        "sequence": sequence,
        "radar_frames": len(shared_names),
        "label_files": len(labels_by_frame),
        "missing_gt": missing_gt,
        "matched_empty": matched_empty,
        "matched_with_objects": matched_with_objects,
        "label_without_radar": len(set(labels_by_frame) - set(shared_names)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(OFFICIAL_KRADAR_GT_ROOT),
    )
    parser.add_argument(
        "--radar-root",
        type=Path,
        default=Path(RADAR_NPY_ROOT),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(CARTESIAN_GT_ROOT),
    )
    parser.add_argument(
        "--lidar2radar-calib",
        type=Path,
        default=DEFAULT_CALIB_PATH,
        help="The calibration YAML used by visualization_based_gt.",
    )
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output_root}. Use --overwrite only "
                "after checking its contents."
            )
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    rotation, translation = load_lidar2radar_calibration(
        args.lidar2radar_calib
    )

    if args.sequences is None:
        sequences = sorted(
            path.name
            for path in args.source_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
    else:
        sequences = sorted({str(int(sequence)) for sequence in args.sequences}, key=numeric_name_key)

    summary = []
    for sequence in sequences:
        summary.append(
            build_sequence(
                sequence=sequence,
                source_root=args.source_root,
                radar_root=args.radar_root,
                output_root=args.output_root,
                rotation=rotation,
                translation=translation,
            )
        )

    readme = args.output_root / "README.txt"
    readme.write_text(
        "K-Radar-GT-cartesian-radar-v2\n"
        "==============================\n\n"
        "Source: K-Radar revised v2.1 visibility labels.\n"
        "Cartesian centers are transformed from the source LiDAR/reference\n"
        "frame to radar coordinates using visualization_based_gt's\n"
        "lidar2radar_calib.yml: radar = lidar @ R.T + T.\n"
        f"Calibration file: {args.lidar2radar_calib}\n"
        "Objects with visibility token R or LR are included in the radar GT.\n"
        "L/L1/LL and other lidar-only visibility tokens are excluded.\n"
        "Objects with object_label=-1 are included.\n"
        "Radar-aligned per-frame label copies are written to each sequence directory.\n"
        "The filename first component and header tesseract_idx were checked.\n"
        "frame_manifest.csv is the authoritative radar-to-label matching audit.\n"
        "missing_gt_file means the radar npy exists but no revised label file exists;\n"
        "it must not be silently interpreted as a real empty annotated frame.\n\n"
        "gt/gt.txt columns:\n"
        "frame_idx, object_label, x, y, z, x_width, y_width, z_width, yaw_deg, class\n"
    )

    print("sequence,radar_frames,label_files,missing_gt,matched_empty,matched_with_objects,label_without_radar")
    for item in summary:
        print(
            ",".join(str(item[key]) for key in (
                "sequence",
                "radar_frames",
                "label_files",
                "missing_gt",
                "matched_empty",
                "matched_with_objects",
                "label_without_radar",
            ))
        )


if __name__ == "__main__":
    main()
