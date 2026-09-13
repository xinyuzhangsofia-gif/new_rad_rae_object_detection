#!/usr/bin/env python3
"""Render a complete K-Radar RAD tensor as a PyVista volume.

The input is a dense scalar field with axes [range, azimuth, Doppler].  Every
input value becomes one cell in a ``pyvista.ImageData`` volume; this script
does not threshold the tensor or convert it into a point cloud.

The project's 2D RA display computes ``log1p(abs(mean(RAD, axis=2)))``.  That
exact expression removes Doppler and therefore cannot produce a 3D RAD map.
The default 3D display instead applies the analogous final compression to
each voxel, ``log1p(abs(RAD))``, while preserving the Doppler dimension.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import RADAR_NPY_ROOT


RAD_SHAPE = (256, 107, 64)
RANGE_MIN_METERS = 0.0
RANGE_MAX_METERS = 118.037109375
AZIMUTH_MIN_DEGREES = -53.0
AZIMUTH_MAX_DEGREES = 53.0
DOPPLER_MIN_MPS = -1.932591218311502
DOPPLER_MAX_MPS = 1.8721977427392676

TRANSFORM_NONE = "none"
TRANSFORM_LOG1P = "log1p"
TRANSFORM_LOG10 = "log10"
TRANSFORMS = (TRANSFORM_NONE, TRANSFORM_LOG1P, TRANSFORM_LOG10)

AXES_BINS = "bins"
AXES_PHYSICAL = "physical"
AXIS_MODES = (AXES_BINS, AXES_PHYSICAL)


@dataclass(frozen=True)
class VolumeAxes:
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    titles: tuple[str, str, str]
    labels: tuple[str, str, str]


@dataclass(frozen=True)
class RadVolume:
    source_path: Path
    values: np.ndarray
    transform: str
    clim: tuple[float, float]
    axes: VolumeAxes

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(size) for size in self.values.shape)

    @property
    def voxel_count(self) -> int:
        return int(self.values.size)


def load_rad_tensor(path: Path | str) -> tuple[Path, np.ndarray]:
    """Load and validate one dense [range, azimuth, Doppler] tensor."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RAD tensor does not exist: {path}")

    rad = np.load(path, mmap_mode="r", allow_pickle=False)
    if rad.shape != RAD_SHAPE:
        raise ValueError(
            f"Expected RAD [R,A,D] shape {RAD_SHAPE}, got {rad.shape}. "
            "An RAE tensor is spatial [R,A,E] data, not a RAD map."
        )
    if not np.issubdtype(rad.dtype, np.number):
        raise TypeError(f"Expected numeric RAD values, got {rad.dtype}")
    if not np.isfinite(rad).all():
        raise ValueError("RAD tensor contains NaN or infinite values")
    return path, rad


def transform_volume(rad: np.ndarray, transform: str) -> np.ndarray:
    """Prepare voxel values for display without reducing any dimension."""
    if transform == TRANSFORM_NONE:
        values = np.asarray(rad, dtype=np.float32)
    elif transform == TRANSFORM_LOG1P:
        # This matches the final abs/log1p compression in the 2D RA renderer,
        # but deliberately omits its mean(axis=2), which would remove Doppler.
        values = np.log1p(np.abs(rad)).astype(np.float32)
    elif transform == TRANSFORM_LOG10:
        # Available to reproduce the transform in the user's PyVista example.
        values = np.log10(np.abs(rad) + 1e-6).astype(np.float32)
    else:
        raise ValueError(f"Unknown transform {transform!r}; choose from {TRANSFORMS}")

    if values.shape != RAD_SHAPE:
        raise AssertionError("The display transform changed the RAD shape")
    if not np.isfinite(values).all():
        raise ValueError("The display transform produced non-finite values")
    return values


def volume_axes(axis_mode: str, doppler_display_scale: float) -> VolumeAxes:
    """Return ImageData coordinates and labels for the selected axis mode."""
    if axis_mode == AXES_BINS:
        return VolumeAxes(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            titles=("Range bin", "Azimuth bin", "Doppler bin"),
            labels=("Range", "Azimuth", "Doppler"),
        )

    if axis_mode != AXES_PHYSICAL:
        raise ValueError(f"Unknown axis mode {axis_mode!r}; choose from {AXIS_MODES}")
    if doppler_display_scale <= 0.0:
        raise ValueError("doppler_display_scale must be positive")

    range_step = (RANGE_MAX_METERS - RANGE_MIN_METERS) / (RAD_SHAPE[0] - 1)
    azimuth_step = (
        AZIMUTH_MAX_DEGREES - AZIMUTH_MIN_DEGREES
    ) / (RAD_SHAPE[1] - 1)
    doppler_step = (DOPPLER_MAX_MPS - DOPPLER_MIN_MPS) / (RAD_SHAPE[2] - 1)
    return VolumeAxes(
        origin=(
            RANGE_MIN_METERS,
            AZIMUTH_MIN_DEGREES,
            DOPPLER_MIN_MPS * doppler_display_scale,
        ),
        spacing=(
            range_step,
            azimuth_step,
            doppler_step * doppler_display_scale,
        ),
        titles=(
            "Range (m)",
            "Azimuth (deg)",
            f"Doppler x{doppler_display_scale:g} (display)",
        ),
        labels=("Range", "Azimuth", "Doppler"),
    )


def percentile_color_limits(
    values: np.ndarray,
    percentiles: Sequence[float],
) -> tuple[float, float]:
    """Compute display limits; this does not filter or remove any voxel."""
    if len(percentiles) != 2:
        raise ValueError("Exactly two color-limit percentiles are required")
    low_q, high_q = (float(value) for value in percentiles)
    if not 0.0 <= low_q < high_q <= 100.0:
        raise ValueError(
            "Color-limit percentiles must satisfy 0 <= LOW < HIGH <= 100"
        )
    low, high = (float(value) for value in np.percentile(values, (low_q, high_q)))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid volume color limits: {(low, high)}")
    return low, high


def prepare_rad_volume(
    path: Path | str,
    *,
    transform: str,
    clim_percentiles: Sequence[float],
    axis_mode: str,
    doppler_display_scale: float,
) -> RadVolume:
    source_path, rad = load_rad_tensor(path)
    values = transform_volume(rad, transform)
    return RadVolume(
        source_path=source_path,
        values=values,
        transform=transform,
        clim=percentile_color_limits(values, clim_percentiles),
        axes=volume_axes(axis_mode, doppler_display_scale),
    )


def make_pyvista_grid(volume: RadVolume):
    """Build a cell-centered grid containing every RAD voxel."""
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError(
            "PyVista is not installed in this Python environment. In this "
            "workspace, run through the 'mvrss' conda environment."
        ) from error

    # Cell data with N cells per axis requires N+1 grid points per axis.
    grid = pv.ImageData()
    grid.dimensions = np.asarray(volume.shape, dtype=int) + 1
    grid.origin = volume.axes.origin
    grid.spacing = volume.axes.spacing
    grid.cell_data["RAD power"] = volume.values.ravel(order="F")

    if grid.n_cells != volume.voxel_count:
        raise AssertionError(
            f"Grid contains {grid.n_cells} cells for {volume.voxel_count} voxels"
        )
    if grid.cell_data["RAD power"].size != volume.voxel_count:
        raise AssertionError("Not every RAD voxel was assigned to the grid")
    return grid


def transform_label(transform: str) -> str:
    if transform == TRANSFORM_NONE:
        return "RAD value"
    if transform == TRANSFORM_LOG1P:
        return "log1p(abs(RAD))"
    if transform == TRANSFORM_LOG10:
        return "log10(abs(RAD) + 1e-6)"
    return transform


def _set_camera(plotter, grid) -> None:
    x_min, x_max, y_min, y_max, z_min, z_max = grid.bounds
    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min
    plotter.camera.focal_point = (
        0.5 * (x_min + x_max),
        0.5 * (y_min + y_max),
        0.5 * (z_min + z_max),
    )
    plotter.camera.position = (
        0.5 * (x_min + x_max) + x_size,
        y_min - 2.4 * y_size,
        0.5 * (z_min + z_max) + 3.1 * z_size,
    )
    plotter.camera.up = (0.0, 0.0, 1.0)


def render_volume(
    volume: RadVolume,
    *,
    screenshot: Path | None,
    cmap: str,
    opacity: str,
    shade: bool,
) -> None:
    """Volume-render all RAD cells interactively or to a screenshot."""
    if screenshot is not None:
        # DISPLAY may be set but inaccessible in remote sessions. EGL keeps
        # off-screen screenshots deterministic without affecting interaction.
        os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")

    import pyvista as pv

    grid = make_pyvista_grid(volume)
    plotter = pv.Plotter(
        off_screen=screenshot is not None,
        window_size=(1600, 1000),
    )
    plotter.set_background("#050a12", top="#17283c")
    plotter.add_volume(
        grid,
        scalars="RAD power",
        preference="cell",
        cmap=cmap,
        clim=volume.clim,
        opacity=opacity,
        blending="composite",
        shade=shade,
        scalar_bar_args={
            "title": transform_label(volume.transform),
            "color": "white",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.22,
            "height": 0.55,
            "title_font_size": 15,
            "label_font_size": 12,
        },
    )
    plotter.show_bounds(
        grid="back",
        location="outer",
        all_edges=True,
        use_2d=True,
        xtitle=volume.axes.titles[0],
        ytitle=volume.axes.titles[1],
        ztitle=volume.axes.titles[2],
        color="white",
        font_size=12,
    )
    plotter.add_axes(
        xlabel=volume.axes.labels[0],
        ylabel=volume.axes.labels[1],
        zlabel=volume.axes.labels[2],
        color="white",
        line_width=3,
    )
    plotter.add_text(
        f"K-Radar {volume.source_path.stem} | Full R-A-D volume",
        position=(405, 925),
        color="white",
        font_size=19,
    )
    plotter.add_text(
        f"All {volume.voxel_count:,} voxels | volume rendering "
        "(not a point cloud)",
        position=(24, 850),
        color="white",
        font_size=14,
    )
    _set_camera(plotter, grid)

    if screenshot is None:
        plotter.show()
        return

    screenshot = screenshot.expanduser().resolve()
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(screenshot), auto_close=True)
    print(f"Saved screenshot: {screenshot}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render every cell of a dense [range, azimuth, Doppler] NPY "
            "tensor as a PyVista volume."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path(RADAR_NPY_ROOT) / "1/rad/00033.npy",
        help="RAD .npy path (default: requested frame 00033)",
    )
    parser.add_argument(
        "--transform",
        choices=TRANSFORMS,
        default=TRANSFORM_LOG1P,
        help=(
            "Voxel display transform: log1p matches the final compression in "
            "the 2D RA code without collapsing Doppler; log10 reproduces the "
            "provided example; none uses stored values (default: log1p)"
        ),
    )
    parser.add_argument(
        "--clim-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(80.0, 99.9),
        help=(
            "Percentiles used only for color/opacity limits; all voxels remain "
            "in the volume (default: 80 99.9)"
        ),
    )
    parser.add_argument(
        "--axis-mode",
        choices=AXIS_MODES,
        default=AXES_BINS,
        help=(
            "Use tensor-bin axes like the provided code, or calibrated "
            "range/azimuth/Doppler coordinates (default: bins)"
        ),
    )
    parser.add_argument(
        "--doppler-display-scale",
        type=float,
        default=20.0,
        help=(
            "Visual stretch for the physical Doppler axis; has no effect in "
            "bin mode (default: 20)"
        ),
    )
    parser.add_argument(
        "--cmap",
        default="jet",
        help="Volume colormap (default: jet, matching the attached RA image)",
    )
    parser.add_argument(
        "--opacity",
        default="sigmoid",
        help="PyVista volume opacity transfer function (default: sigmoid)",
    )
    parser.add_argument(
        "--shade",
        action="store_true",
        help="Enable volume shading (disabled by default)",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Save a PNG off-screen instead of opening an interactive window",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    volume = prepare_rad_volume(
        args.input,
        transform=args.transform,
        clim_percentiles=args.clim_percentiles,
        axis_mode=args.axis_mode,
        doppler_display_scale=args.doppler_display_scale,
    )
    print(f"Source: {volume.source_path}")
    print(f"RAD shape: {volume.shape}")
    print(f"Volume cells: {volume.voxel_count:,} (all input voxels)")
    print(f"Display transform: {transform_label(volume.transform)}")
    print(f"Color/opacity limits: {volume.clim[0]:.6f} to {volume.clim[1]:.6f}")
    print("Selection: none (no top-k and no thresholded cells)")
    render_volume(
        volume,
        screenshot=args.screenshot,
        cmap=args.cmap,
        opacity=args.opacity,
        shade=args.shade,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
