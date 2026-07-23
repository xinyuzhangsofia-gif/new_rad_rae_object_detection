#!/usr/bin/env python3
"""Build Polar GT from the radar-aligned Cartesian GT files.

The input Cartesian files contain full metric dimensions in radar Cartesian
coordinates::

    frame_idx,object_label,x,y,z,x_width,y_width,z_width,yaw_deg,class

The output keeps the Polar ``gt.txt`` layout used by this project::

    frame_idx,object_label,a_idx,r_idx,a_width,r_width,e_idx,e_width,yaw_deg,class

Each Cartesian box is converted by transforming all eight corners to RAE and
taking the clipped min/max envelope.  This preserves the full physical box
size; it does not multiply or divide the Cartesian dimensions again.
Rows with ``object_label == -1`` are deliberately excluded.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_CARTESIAN_ROOT = Path("/home/local/xinyu/K-Radar-GT-cartesian-radar-v2")
DEFAULT_RADAR_ROOT = Path("/home/local/xinyu/K-Radar-RAD")
DEFAULT_OUTPUT_ROOT = Path("/home/local/xinyu/K-Radar-GT-Polar-v2.9")

# These are the physical axes of the current 256 x 107 x 37 radar tensor.
R_MIN, R_MAX, R_SIZE = 0.0, 118.037109375, 256
A_MIN, A_MAX, A_SIZE = -53.0, 53.0, 107
E_MIN, E_MAX, E_SIZE = -18.0, 18.0, 37


def numeric_name_key(name: str):
    return (0, int(name)) if name.isdigit() else (1, name)


def axis_step(minimum: float, maximum: float, size: int) -> float:
    return (maximum - minimum) / max(size - 1, 1)


R_STEP = axis_step(R_MIN, R_MAX, R_SIZE)
A_STEP = axis_step(A_MIN, A_MAX, A_SIZE)
E_STEP = axis_step(E_MIN, E_MAX, E_SIZE)


def read_cartesian_gt(path: Path):
    rows = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 10:
            raise ValueError(
                f"Expected 10 Cartesian fields in {path}:{line_number}, "
                f"got {len(parts)}"
            )

        frame_idx = int(parts[0])
        object_label = int(parts[1])
        values = [float(value) for value in parts[2:9]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite Cartesian values in {path}:{line_number}")
        if min(values[3:6]) <= 0.0:
            raise ValueError(
                f"Cartesian dimensions must be positive in {path}:{line_number}"
            )
        rows.append(
            {
                "frame_idx": frame_idx,
                "object_label": object_label,
                "x": values[0],
                "y": values[1],
                "z": values[2],
                "length": values[3],
                "width": values[4],
                "height": values[5],
                "yaw_deg": values[6],
                "class_name": parts[9],
            }
        )
    return rows


def read_shared_radar_names(radar_root: Path, sequence: str):
    rad_dir = radar_root / sequence / "rad"
    rae_dir = radar_root / sequence / "rae"
    rad_names = {path.stem for path in rad_dir.glob("*.npy")}
    rae_names = {path.stem for path in rae_dir.glob("*.npy")}
    shared_names = sorted(rad_names & rae_names, key=numeric_name_key)
    if not shared_names:
        raise FileNotFoundError(f"No shared radar/rae files found for sequence {sequence}")
    return shared_names


def box_corners(row):
    """Return the eight radar-Cartesian corners of a full-size box."""
    half_length = row["length"] / 2.0
    half_width = row["width"] / 2.0
    half_height = row["height"] / 2.0
    yaw = math.radians(row["yaw_deg"])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    corners = []
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            for sign_z in (-1.0, 1.0):
                local_x = sign_x * half_length
                local_y = sign_y * half_width
                x = row["x"] + local_x * cos_yaw - local_y * sin_yaw
                y = row["y"] + local_x * sin_yaw + local_y * cos_yaw
                z = row["z"] + sign_z * half_height
                corners.append((x, y, z))
    return corners


def cartesian_corner_to_rae(x: float, y: float, z: float):
    r_xy = math.sqrt(x * x + y * y)
    radius = math.sqrt(r_xy * r_xy + z * z)
    azimuth = math.degrees(math.atan2(-y, x))
    elevation = math.degrees(math.atan2(z, r_xy))
    return radius, azimuth, elevation


def cartesian_box_to_polar(row):
    rae_corners = [cartesian_corner_to_rae(*corner) for corner in box_corners(row)]
    r_values = [corner[0] for corner in rae_corners]
    a_values = [corner[1] for corner in rae_corners]
    e_values = [corner[2] for corner in rae_corners]

    r_low = max(R_MIN, min(r_values))
    r_high = min(R_MAX, max(r_values))
    a_low = max(A_MIN, min(a_values))
    a_high = min(A_MAX, max(a_values))
    e_low = max(E_MIN, min(e_values))
    e_high = min(E_MAX, max(e_values))

    if r_high <= r_low or a_high <= a_low or e_high <= e_low:
        # This box is completely outside the radar tensor's visible R/A/E
        # range.  It must not become a Polar GT target because no radar voxel
        # can contain it.  Partially clipped boxes are kept below.
        return None

    r_center = (r_low + r_high) / 2.0
    a_center = (a_low + a_high) / 2.0
    e_center = (e_low + e_high) / 2.0
    return {
        "a_idx": (a_center - A_MIN) / A_STEP,
        "r_idx": (r_center - R_MIN) / R_STEP,
        "a_width": (a_high - a_low) / A_STEP,
        "r_width": (r_high - r_low) / R_STEP,
        "e_idx": (e_center - E_MIN) / E_STEP,
        "e_width": (e_high - e_low) / E_STEP,
    }


def build_sequence(
        sequence: str,
        cartesian_root: Path,
        radar_root: Path,
        output_root: Path,
        overwrite: bool,
    ):
    shared_names = read_shared_radar_names(radar_root, sequence)
    input_path = cartesian_root / sequence / "gt" / "gt.txt"
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing Cartesian GT file: {input_path}")

    output_dir = output_root / sequence / "gt"
    output_path = output_dir / "gt.txt"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}; use --overwrite to replace it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_cartesian_gt(input_path)
    kept_rows = []
    ignored_rows = 0
    ignored_out_of_view = 0
    for row in rows:
        if row["object_label"] == -1:
            ignored_rows += 1
            continue
        if row["frame_idx"] < 1 or row["frame_idx"] > len(shared_names):
            raise ValueError(
                f"Cartesian frame_idx {row['frame_idx']} is outside the "
                f"{len(shared_names)} shared radar frames in sequence {sequence}"
            )
        polar = cartesian_box_to_polar(row)
        if polar is None:
            ignored_out_of_view += 1
            continue
        kept_rows.append((row, polar))

    with output_path.open("w", newline="") as output_file:
        output_file.write(
            "# frame_idx,object_label,a_idx,r_idx,a_width,r_width,e_idx,e_width,yaw_deg,class\n"
        )
        writer = csv.writer(output_file, lineterminator="\n")
        for row, polar in kept_rows:
            writer.writerow(
                [
                    row["frame_idx"],
                    row["object_label"],
                    f"{polar['a_idx']:.10f}",
                    f"{polar['r_idx']:.10f}",
                    f"{polar['a_width']:.10f}",
                    f"{polar['r_width']:.10f}",
                    f"{polar['e_idx']:.10f}",
                    f"{polar['e_width']:.10f}",
                    f"{row['yaw_deg']:.10f}",
                    row["class_name"],
                ]
            )

    return {
        "sequence": sequence,
        "radar_frames": len(shared_names),
        "input_rows": len(rows),
        "ignored_label_minus_one": ignored_rows,
        "ignored_out_of_view": ignored_out_of_view,
        "output_rows": len(kept_rows),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartesian-root", type=Path, default=DEFAULT_CARTESIAN_ROOT)
    parser.add_argument("--radar-root", type=Path, default=DEFAULT_RADAR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sequences", default="1-58")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_sequences(value: str):
    result = []
    for token in value.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(token))
    return tuple(dict.fromkeys(result))


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        build_sequence(
            sequence=str(sequence),
            cartesian_root=args.cartesian_root,
            radar_root=args.radar_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        for sequence in parse_sequences(args.sequences)
    ]
    print("range_step_m_per_bin=", R_STEP)
    print("azimuth_step_deg_per_bin=", A_STEP)
    print("elevation_step_deg_per_bin=", E_STEP)
    print("total_input_rows=", sum(item["input_rows"] for item in summaries))
    print(
        "total_ignored_label_minus_one=",
        sum(item["ignored_label_minus_one"] for item in summaries),
    )
    print(
        "total_ignored_out_of_view=",
        sum(item["ignored_out_of_view"] for item in summaries),
    )
    print("total_output_rows=", sum(item["output_rows"] for item in summaries))
    for item in summaries:
        if item["ignored_label_minus_one"] or item["ignored_out_of_view"]:
            print(item)


if __name__ == "__main__":
    main()
