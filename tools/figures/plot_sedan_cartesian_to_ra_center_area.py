#!/usr/bin/env python3
"""Plot current Cartesian-radar GT after converting centers to Polar R-A bins.

The input is the same per-frame Cartesian radar GT used by the Cartesian
dataloader.  Only Sedan centers are plotted.  The point color is the metric
Cartesian BEV area ``length_m * width_m``; no Polar GT file is read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from data.coordinates import AZIMUTH_AXIS, RANGE_AXIS, cartesian_to_rae
from data.labels import read_kradar_revised_label_dir
from configs.data import CARTESIAN_GT_ROOT, PROJECT_ROOT


DEFAULT_CARTESIAN_ROOT = Path(CARTESIAN_GT_ROOT)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "visualization_based_gt/generated/cartesian_to_polar_center_range_area"
)


def read_sedan_cartesian_centers(gt_root: Path, sequence: int):
    """Read radar-visible Cartesian GT and convert center points to R-A bins."""

    gt_by_frame_name = read_kradar_revised_label_dir(
        label_root=gt_root,
        sequence=sequence,
        radar_visibility_tokens=("R", "LR"),
    )

    a_centers = []
    r_centers = []
    cartesian_areas = []
    for objects in gt_by_frame_name.values():
        for obj in objects:
            if str(obj.get("cls", "")) != "Sedan":
                continue

            x, y, z, length, width, _height, _yaw = (
                float(value) for value in obj["box_metric"].tolist()
            )
            radius, azimuth, _elevation = cartesian_to_rae(x, y, z)
            r_idx = (radius - RANGE_AXIS.minimum) / RANGE_AXIS.step
            a_idx = (azimuth - AZIMUTH_AXIS.minimum) / AZIMUTH_AXIS.step
            area = length * width
            if not np.all(np.isfinite([a_idx, r_idx, area])) or area <= 0.0:
                continue

            a_centers.append(a_idx)
            r_centers.append(r_idx)
            cartesian_areas.append(area)

    return (
        np.asarray(a_centers, dtype=np.float64),
        np.asarray(r_centers, dtype=np.float64),
        np.asarray(cartesian_areas, dtype=np.float64),
    )


def make_plot(
    sequence: int,
    gt_root: Path,
    output_root: Path,
    point_size: float = 20.0,
    dpi: int = 200,
) -> Path:
    a_center, r_center, cartesian_area = read_sedan_cartesian_centers(
        gt_root=gt_root,
        sequence=sequence,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / (
        f"seq_{sequence}_sedan_cartesian_to_polar_center.png"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = None
    if cartesian_area.size > 0:
        color_norm = LogNorm(
            vmin=max(float(cartesian_area.min()), np.finfo(np.float64).tiny),
            vmax=float(cartesian_area.max()),
        )
        scatter = ax.scatter(
            a_center,
            r_center,
            c=cartesian_area,
            cmap="Blues",
            norm=color_norm,
            s=point_size,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.25,
        )

    a_mid = (AZIMUTH_AXIS.size - 1) / 2.0
    a_last = AZIMUTH_AXIS.size - 1
    r_last = RANGE_AXIS.size - 1
    # The light band marks the current 107-bin azimuth tensor.  The limits
    # are slightly wider so a small number of out-of-scope GT centers remain
    # visible instead of silently disappearing from the figure.
    ax.axvspan(0.0, a_last, color="grey", alpha=0.06, label="current a scope")
    ax.axvline(
        a_mid,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"middle a bin = {a_mid:g}",
    )
    if cartesian_area.size > 0:
        ax.set_xlim(
            min(-2.0, float(a_center.min()) - 1.0),
            max(a_last + 2.0, float(a_center.max()) + 1.0),
        )
    else:
        ax.set_xlim(-2.0, a_last + 2.0)
    ax.set_ylim(0.0, r_last)
    ax.set_xlabel("a center after Cartesian → Polar conversion (azimuth bin)")
    ax.set_ylabel("r center after Cartesian → Polar conversion (range bin)")
    ax.set_title(
        f"Sequence {sequence}: current Cartesian radar GT converted to Polar centers\n"
        f"N = {cartesian_area.size}; color = Cartesian BEV area (length × width, m²)"
    )
    if cartesian_area.size == 0:
        ax.text(
            0.5,
            0.5,
            "No valid Sedan annotations",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="dimgray",
        )
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="upper right")

    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Cartesian BEV area: length × width (m²)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Cartesian-radar GT centers after conversion to Polar R-A bins."
    )
    parser.add_argument("--sequence", type=int, default=11)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_CARTESIAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--point-size", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = make_plot(
        sequence=args.sequence,
        gt_root=args.gt_root,
        output_root=args.output_root,
        point_size=args.point_size,
        dpi=args.dpi,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
