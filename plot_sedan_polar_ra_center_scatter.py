#!/usr/bin/env python3
"""Plot the Polar ``(a, r)`` center distribution of Sedan GT boxes.

The input gt.txt format is:

    frame, object_id, a_idx, r_idx, a_width, r_width,
    e_idx, e_width, yaw, class

The scatter plot uses:
    x = a_idx
    y = r_idx
    color = a_width * r_width

Thus, the point position shows the bbox center in the Polar grid and the
color shows the Polar bbox size, matching the color logic of
plot_sedan_polar_bbox_scatter.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


DEFAULT_GT_ROOT = Path("/home/local/xinyu/K-Radar-GT-Polar-v2")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/local/xinyu/MVRSS/mvrss/visualization_based_gt/generated/polar_ra_center_scatter"
)


def read_sedan_polar_centers(
    gt_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a_idx, r_idx, and Polar bbox area for Sedan annotations."""

    a_centers: list[float] = []
    r_centers: list[float] = []
    polar_areas: list[float] = []

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
            polar_areas.append(a_width * r_width)

    return (
        np.asarray(a_centers, dtype=np.float64),
        np.asarray(r_centers, dtype=np.float64),
        np.asarray(polar_areas, dtype=np.float64),
    )


def make_plot(
    sequence: int,
    gt_root: Path,
    output_root: Path,
    point_size: float = 18.0,
    alpha: float = 0.75,
    dpi: int = 200,
    show: bool = False,
) -> Path:
    """Create and save one Sedan Polar center-distribution plot."""

    gt_path = gt_root / str(sequence) / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    a_idx, r_idx, polar_area = read_sedan_polar_centers(gt_path)
    if polar_area.size == 0:
        raise ValueError(f"No valid Sedan annotations found in {gt_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"seq_{sequence}_sedan_polar_ra_center_scatter.png"

    fig, ax = plt.subplots(figsize=(9, 7))
    color_norm = LogNorm(
        vmin=max(float(polar_area.min()), np.finfo(np.float64).tiny),
        vmax=float(polar_area.max()),
    )
    scatter = ax.scatter(
        a_idx,
        r_idx,
        c=polar_area,
        cmap="Blues",
        norm=color_norm,
        s=point_size,
        alpha=max(alpha, 0.85),
        edgecolors="white",
        linewidths=0.25,
    )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Polar bbox size: a_width × r_width (bin²)")

    ax.set_xlabel("a center (azimuth bin, a_idx)")
    ax.set_ylabel("r center (range bin, r_idx)")
    ax.set_title(
        f"Sequence {sequence}: Sedan Polar bbox center distribution\n"
        f"N = {polar_area.size}, color = a_width × r_width"
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Sedan Polar bbox center distribution colored by bbox size."
    )
    parser.add_argument("--sequence", type=int, default=11)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--dpi", type=int, default=200)
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show", dest="show", action="store_true", default=True)
    show_group.add_argument("--no-show", dest="show", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = make_plot(
        sequence=args.sequence,
        gt_root=args.gt_root,
        output_root=args.output_root,
        point_size=args.point_size,
        alpha=args.alpha,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
