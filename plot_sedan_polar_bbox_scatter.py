#!/usr/bin/env python3
"""Plot Sedan Polar bbox widths and color them by Polar bbox area.

The input gt.txt format is:

    frame, object_id, a_idx, r_idx, a_width, r_width,
    e_idx, e_width, yaw, class

The scatter plot uses:
    x = a_width
    y = r_width
    color = a_width * r_width
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

# Use a GUI backend when a display is available. On headless machines, keep
# saving support by falling back to a non-interactive backend.
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


DEFAULT_GT_ROOT = Path("/home/local/xinyu/K-Radar-GT-Polar-v2")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/local/xinyu/MVRSS/mvrss/visualization_based_gt/generated/polar_bbox_scatter"
)


def read_sedan_polar_widths(gt_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read Sedan a_width/r_width values and return a_width, r_width, area."""

    a_widths: list[float] = []
    r_widths: list[float] = []

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

            # gt.txt fields:
            # frame, object_id, a_idx, r_idx, a_width, r_width,
            # e_idx, e_width, yaw, class
            if parts[9] != "Sedan":
                continue

            a_width = float(parts[4])
            r_width = float(parts[5])
            if not np.isfinite(a_width) or not np.isfinite(r_width):
                continue
            if a_width <= 0 or r_width <= 0:
                continue

            a_widths.append(a_width)
            r_widths.append(r_width)

    a_values = np.asarray(a_widths, dtype=np.float64)
    r_values = np.asarray(r_widths, dtype=np.float64)
    areas = a_values * r_values
    return a_values, r_values, areas


def make_plot(
    sequence: int,
    gt_root: Path,
    output_root: Path,
    point_size: float = 18.0,
    alpha: float = 0.75,
    dpi: int = 200,
    show: bool = False,
) -> Path:
    """Create and save one Sedan Polar bbox scatter plot."""

    gt_path = gt_root / str(sequence) / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    a_width, r_width, area = read_sedan_polar_widths(gt_path)
    if area.size == 0:
        raise ValueError(f"No valid Sedan annotations found in {gt_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"seq_{sequence}_sedan_polar_bbox_scatter.png"

    fig, ax = plt.subplots(figsize=(9, 7))
    color_norm = LogNorm(
        vmin=max(float(area.min()), np.finfo(np.float64).tiny),
        vmax=float(area.max()),
    )
    scatter = ax.scatter(
        a_width,
        r_width,
        c=area,
        cmap="Blues",
        norm=color_norm,
        s=point_size,
        alpha=max(alpha, 0.85),
        edgecolors="white",
        linewidths=0.25,
    )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Polar bbox area: a_width × r_width (bin²)")

    ax.set_xlabel("a_width (azimuth bins)")
    ax.set_ylabel("r_width (range bins)")
    ax.set_title(
        f"Sequence {sequence}: Sedan Polar bbox widths\n"
        f"N = {area.size}, color = a_width × r_width"
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
        description="Plot Sedan Polar bbox widths colored by Polar bbox area."
    )
    parser.add_argument(
        "--sequence",
        type=int,
        default=12,
        help="Sequence number to plot (default: 11).",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=DEFAULT_GT_ROOT,
        help=f"Polar GT root (default: {DEFAULT_GT_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory for saved figures (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=18.0,
        help="Scatter point size (default: 18).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Point transparency from 0 to 1 (default: 0.75).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved image DPI (default: 200).",
    )
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Open the plot window after saving (default).",
    )
    show_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Only save the image; do not open a plot window.",
    )
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
