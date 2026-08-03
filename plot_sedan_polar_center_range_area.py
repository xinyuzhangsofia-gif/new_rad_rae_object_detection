#!/usr/bin/env python3
"""Plot Sedan bbox centers in the Polar ``(a, range)`` plane.

Each Sedan annotation contributes exactly one point at ``(a_idx, r_idx)``.
The point color is the Polar bbox area ``a_width * r_width``.  The default
axes match the Polar visualization convention used for this dataset:
azimuth bins 0--144, range bins 0--256, with azimuth bin 72 at the center.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


DEFAULT_GT_ROOT = Path("/home/local/xinyu/K-Radar-GT-Polar-v2.9")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/local/xinyu/MVRSS/mvrss/visualization_based_gt/generated/"
    "polar_center_range_area"
)


def read_sedan_centers(gt_path: Path):
    a_centers = []
    r_centers = []
    areas = []

    with gt_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 10:
                raise ValueError(
                    f"{gt_path}:{line_number}: expected 10 fields, got {len(parts)}"
                )
            if parts[9] != "Sedan":
                continue

            a_idx = float(parts[2])
            r_idx = float(parts[3])
            a_width = float(parts[4])
            r_width = float(parts[5])
            values = (a_idx, r_idx, a_width, r_width)
            if not all(np.isfinite(value) for value in values):
                continue
            if a_width <= 0.0 or r_width <= 0.0:
                continue

            a_centers.append(a_idx)
            r_centers.append(r_idx)
            areas.append(a_width * r_width)

    return (
        np.asarray(a_centers, dtype=np.float64),
        np.asarray(r_centers, dtype=np.float64),
        np.asarray(areas, dtype=np.float64),
    )


def make_plot(
    sequence: int,
    gt_root: Path,
    output_root: Path,
    a_max: float = 144.0,
    r_max: float = 256.0,
    point_size: float = 20.0,
    dpi: int = 200,
) -> Path:
    gt_path = gt_root / str(sequence) / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    a_center, r_center, area = read_sedan_centers(gt_path)
    if area.size == 0:
        raise ValueError(f"No valid Sedan annotations found in {gt_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / (
        f"seq_{sequence}_sedan_center_a_range_area.png"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    norm = LogNorm(
        vmin=max(float(area.min()), np.finfo(np.float64).tiny),
        vmax=float(area.max()),
    )
    scatter = ax.scatter(
        a_center,
        r_center,
        c=area,
        cmap="Blues",
        norm=norm,
        s=point_size,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.25,
    )

    a_mid = a_max / 2.0
    ax.axvline(
        a_mid,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"middle a bin = {a_mid:g}",
    )
    ax.set_xlim(0.0, a_max)
    ax.set_ylim(0.0, r_max)
    ax.set_xlabel("a center (azimuth bin)")
    ax.set_ylabel("range center (range bin)")
    ax.set_title(
        f"Sequence {sequence}: Sedan center distribution in Polar (a, range)\n"
        f"N = {area.size}; color = a_width × r_width"
    )
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="upper right")

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Polar bbox area: a_width × r_width (bin²)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Sedan centers in Polar a-range coordinates."
    )
    parser.add_argument("--sequence", type=int, default=11)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--a-max", type=float, default=144.0)
    parser.add_argument("--r-max", type=float, default=256.0)
    parser.add_argument("--point-size", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = make_plot(
        sequence=args.sequence,
        gt_root=args.gt_root,
        output_root=args.output_root,
        a_max=args.a_max,
        r_max=args.r_max,
        point_size=args.point_size,
        dpi=args.dpi,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
