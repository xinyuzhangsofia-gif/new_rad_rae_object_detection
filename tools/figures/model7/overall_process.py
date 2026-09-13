#!/usr/bin/env python3
"""Draw the simplified overall RADE -> RAD/RAE -> Model7 process.

The figure intentionally stays at overview level:

* the four-axis RADE input is an original neutral interlocking-cube glyph,
  following the visual language of the user-supplied 4DR P2T reference;
* the standalone projection cards are omitted;
* every visible RAD/RAE cube surface carries a real projected view, using one
  shared colormap: RA/RD/AD for RAD and RA/RE/AE for RAE;
* the detection output remains an editable placeholder for a later revision.

Only ``figures/multiview_rade_overview/rade_to_rad_rae_overall_process.*`` is
written. Earlier detailed architecture figures are not modified.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mvrss_rade_overall_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.transforms import Affine2D

from .architecture import (
    BLUE,
    DEFAULT_RADAR_ROOT,
    GREEN,
    INK,
    LIGHT,
    MID,
    NAVY,
    PALE_BLUE,
    PALE_GREEN,
    PALE_PURPLE,
    PURPLE,
    ROOT,
    WHITE,
    setup_style,
    validate_current_config,
)
from .architecture_3d import (
    arrow,
    cuboid,
    rounded_box,
    wireframe_box,
)


OUT_DIR = ROOT / "figures" / "multiview_rade_overview"
STEM = "rade_to_rad_rae_overall_process"
DEFAULT_SAMPLE_SEED = 42

CUBE_EDGE = "#33495c"
CUBE_LIGHT = "#e7e9ec"
CUBE_MID = "#d7dade"
CUBE_DARK = "#bfc4ca"
RA_CMAP = "viridis"


def numeric_path_key(path: Path) -> tuple[int | str, int | str]:
    sequence = path.parent.parent.name
    frame = path.stem
    return (
        int(sequence) if sequence.isdigit() else sequence,
        int(frame) if frame.isdigit() else frame,
    )


def paired_samples(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if not root.is_dir():
        return pairs
    for rad_path in sorted(root.glob("*/rad/*.npy"), key=numeric_path_key):
        rae_path = rad_path.parent.parent / "rae" / rad_path.name
        if rae_path.is_file():
            pairs.append((rad_path, rae_path))
    return pairs


def choose_paired_sample(
    root: Path,
    *,
    seed: int,
    sequence: str | None = None,
    frame: str | None = None,
) -> tuple[Path, Path, str]:
    """Choose one deterministic random pair, or one explicit paired frame."""
    if (sequence is None) != (frame is None):
        raise ValueError("--sequence and --frame must be supplied together")

    if sequence is not None and frame is not None:
        frame_name = frame if frame.endswith(".npy") else f"{frame}.npy"
        rad_path = root / sequence / "rad" / frame_name
        rae_path = root / sequence / "rae" / frame_name
        if not rad_path.is_file() or not rae_path.is_file():
            raise FileNotFoundError(f"Missing paired sample: {rad_path} / {rae_path}")
        note = f"K-Radar {sequence}/{Path(frame_name).stem} · explicit paired frame"
        return rad_path, rae_path, note

    pairs = paired_samples(root)
    if not pairs:
        raise FileNotFoundError(
            f"No paired RAD/RAE sample found under {root}; real cube textures are required"
        )
    rng = np.random.default_rng(seed)
    rad_path, rae_path = pairs[int(rng.integers(len(pairs)))]
    note = (
        f"K-Radar {rad_path.parent.parent.name}/{rad_path.stem} "
        f"· paired random selection, seed {seed}"
    )
    return rad_path, rae_path, note


def robust_normalize(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    low, high = np.nanpercentile(data, (2.0, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - low) / (high - low), 0.0, 1.0)


def log_power_mean(data: np.ndarray, axis: int) -> np.ndarray:
    """Average a base-10 log-power tensor in linear power, then return log10."""
    log_power = np.asarray(data, dtype=np.float64)
    peak = np.nanmax(log_power, axis=axis, keepdims=True)
    scaled_power = np.power(10.0, log_power - peak)
    mean_scaled_power = np.nanmean(scaled_power, axis=axis)
    projected = np.squeeze(peak, axis=axis) + np.log10(mean_scaled_power)
    if not np.isfinite(projected).all():
        raise ValueError("Log-power projection produced non-finite values")
    return projected


def load_projection_maps(
    root: Path,
    *,
    seed: int,
    sequence: str | None,
    frame: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    rad_path, rae_path, source_note = choose_paired_sample(
        root,
        seed=seed,
        sequence=sequence,
        frame=frame,
    )
    rad = np.load(rad_path, mmap_mode="r")
    rae = np.load(rae_path, mmap_mode="r")
    if rad.shape != (256, 107, 64):
        raise ValueError(f"Expected RAD [R,A,D]=(256,107,64), got {rad.shape}")
    if rae.shape != (256, 107, 37):
        raise ValueError(f"Expected RAE [R,A,E]=(256,107,37), got {rae.shape}")

    # These tensors store base-10 log power.  Projection must therefore happen
    # after returning to linear power; directly averaging logged values gives
    # a false mismatch between the two mathematically identical RA marginals.
    rad_ra = log_power_mean(rad, axis=2)
    rae_ra = log_power_mean(rae, axis=2)
    if not np.allclose(rad_ra, rae_ra, rtol=1e-10, atol=1e-10):
        max_error = float(np.nanmax(np.abs(rad_ra - rae_ra)))
        raise ValueError(f"Paired RAD/RAE tensors disagree on shared RA: {max_error}")
    shared_ra = robust_normalize(0.5 * (rad_ra + rae_ra))
    rad_maps = {
        "RA": shared_ra.copy(),
        # RAD is [R,A,D]: remove A for RD and remove R for AD.
        "RD": robust_normalize(log_power_mean(rad, axis=1)),
        "AD": robust_normalize(log_power_mean(rad, axis=0)),
    }
    rae_maps = {
        "RA": shared_ra.copy(),
        # RAE is [R,A,E]: remove A for RE and remove R for AE.
        "RE": robust_normalize(log_power_mean(rae, axis=1)),
        "AE": robust_normalize(log_power_mean(rae, axis=0)),
    }
    expected = {
        "RAD": {"RA": (256, 107), "RD": (256, 64), "AD": (107, 64)},
        "RAE": {"RA": (256, 107), "RE": (256, 37), "AE": (107, 37)},
    }
    actual = {
        "RAD": {name: image.shape for name, image in rad_maps.items()},
        "RAE": {name: image.shape for name, image in rae_maps.items()},
    }
    if actual != expected:
        raise AssertionError(f"Projection-map contract changed: {actual}")
    if not np.array_equal(rad_maps["RA"], rae_maps["RA"]):
        raise AssertionError("RAD and RAE must display one identical shared RA view")
    for cube_name, maps in (("RAD", rad_maps), ("RAE", rae_maps)):
        for view_name, image in maps.items():
            if not np.isfinite(image).all() or float(np.ptp(image)) <= 0.0:
                raise ValueError(f"Invalid real-data visualization for {cube_name} {view_name}")
    source_note = f"{source_note} · linear-power mean projections"
    return rad_maps, rae_maps, source_note


def draw_header(ax) -> None:
    ax.text(
        4.0,
        88.0,
        "Overall RADE dual-view processing and detection",
        ha="left",
        va="center",
        fontsize=18.0,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        4.1,
        84.2,
        "Four-axis radar tensor  →  three-view RAD and RAE cubes  →  independent encoders  →  encoded-feature fusion  →  shared detector",
        ha="left",
        va="center",
        fontsize=9.0,
        color=MID,
    )
    rounded_box(ax, 197.0, 85.7, 15.0, 3.0, face=PALE_BLUE, edge=BLUE, linewidth=0.8, radius=1.2, zorder=7)
    ax.text(204.5, 87.2, "CURRENT MODEL7", ha="center", va="center", fontsize=6.8, fontweight="bold", color=BLUE, zorder=8)
    ax.plot([4.0, 212.0], [80.5, 80.5], color=LIGHT, linewidth=1.0)


def stage_heading(ax, x: float, text: str, colour: str) -> None:
    ax.text(
        x,
        76.7,
        text,
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=colour,
    )


def cube_vertices(
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> list[np.ndarray]:
    """Return cube vertices indexed by a + 2*b + 4*c."""
    return [
        origin + a * u + b * v + c * w
        for c in (0, 1)
        for b in (0, 1)
        for a in (0, 1)
    ]


def draw_wire_cube(
    ax,
    vertices: list[np.ndarray],
    *,
    alpha: float,
    zorder: float,
) -> None:
    """Draw one translucent projected cube with an entirely solid outline."""
    face_specs = (
        ((0, 1, 5, 4), CUBE_LIGHT, 0.72),
        ((4, 5, 7, 6), CUBE_MID, 0.66),
        ((1, 3, 7, 5), CUBE_DARK, 0.70),
    )
    for indices, colour, face_alpha in face_specs:
        ax.add_patch(
            patches.Polygon(
                [vertices[index] for index in indices],
                closed=True,
                facecolor=colour,
                edgecolor="none",
                alpha=face_alpha * alpha,
                zorder=zorder,
            )
        )

    for index in range(8):
        a = index & 1
        b = (index >> 1) & 1
        c = (index >> 2) & 1
        for bit, flag in ((1, a), (2, b), (4, c)):
            if flag:
                continue
            other = index | bit
            ax.plot(
                [vertices[index][0], vertices[other][0]],
                [vertices[index][1], vertices[other][1]],
                color=CUBE_EDGE,
                linewidth=0.85,
                linestyle="-",
                alpha=alpha,
                zorder=zorder + 1,
            )


def axis_arrow(
    ax,
    start: np.ndarray,
    end: np.ndarray,
    *,
    label: str,
    label_xy: tuple[float, float],
    rotation: float = 0.0,
) -> None:
    ax.annotate(
        "",
        xy=tuple(end),
        xytext=tuple(start),
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=0.9),
        zorder=14,
    )
    ax.text(
        label_xy[0],
        label_xy[1],
        label,
        ha="center",
        va="center",
        fontsize=7.4,
        color="black",
        rotation=rotation,
        zorder=15,
    )


def draw_four_axis_tensor_glyph(ax, x: float, y: float) -> None:
    """Draw a neutral two-cube projection of a four-dimensional tensor."""
    origin = np.asarray((x + 7.0, y + 5.0), dtype=float)
    azimuth = np.asarray((12.5, 0.0), dtype=float)
    range_axis = np.asarray((4.2, 4.8), dtype=float)
    elevation = np.asarray((0.0, 12.5), dtype=float)
    doppler_offset = np.asarray((-4.2, 4.8), dtype=float)

    # Each ordinary cube spans range, azimuth, and elevation.  The second
    # homologous cube is displaced along Doppler, matching the reference.
    near_cube = cube_vertices(origin, azimuth, range_axis, elevation)
    far_cube = [point + doppler_offset for point in near_cube]
    draw_wire_cube(ax, far_cube, alpha=0.80, zorder=3.0)
    draw_wire_cube(ax, near_cube, alpha=0.96, zorder=5.0)

    # The fourth (Doppler) dimension is encoded only by dashed connections
    # between corresponding vertices of the two solid-outline cubes.
    for near, far in zip(near_cube, far_cube):
        ax.plot(
            [near[0], far[0]],
            [near[1], far[1]],
            color=CUBE_EDGE,
            linewidth=0.78,
            linestyle=(0, (4, 2)),
            zorder=7,
        )

    axis_arrow(
        ax,
        origin + np.asarray((0.0, -1.7)),
        origin + azimuth + np.asarray((0.0, -1.7)),
        label="Azimuth",
        label_xy=(origin[0] + azimuth[0] / 2, origin[1] - 3.4),
    )
    axis_arrow(
        ax,
        origin + np.asarray((-0.7, -0.2)),
        origin + 1.25 * doppler_offset + np.asarray((-0.7, -0.2)),
        label="Doppler",
        label_xy=(origin[0] - 5.4, origin[1] + 1.0),
    )
    range_start = origin + azimuth + np.asarray((1.1, -1.2))
    range_end = range_start + 1.35 * range_axis
    axis_arrow(
        ax,
        range_start,
        range_end,
        label="Range",
        label_xy=(range_start[0] + 3.8, range_start[1] + 0.5),
    )
    elevation_start = range_end + np.asarray((0.8, 0.0))
    axis_arrow(
        ax,
        elevation_start,
        elevation_start + 0.95 * elevation,
        label="Elevation",
        label_xy=(elevation_start[0] + 2.0, elevation_start[1] + 6.0),
        rotation=90.0,
    )

    ax.text(x + 12.0, y + 31.5, "4-D RADE tensor", ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)


def draw_textured_face(
    ax,
    image: np.ndarray,
    corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
    *,
    label: str,
    label_xy: tuple[float, float],
    label_rotation: float = 0.0,
    zorder: float,
) -> None:
    """Affine-map one real radar view onto a visible parallelogram face."""
    p00 = np.asarray(corners[0], dtype=float)
    p10 = np.asarray(corners[1], dtype=float)
    p11 = np.asarray(corners[2], dtype=float)
    p01 = np.asarray(corners[3], dtype=float)
    basis_u = p10 - p00
    basis_v = p01 - p00
    expected_p11 = p00 + basis_u + basis_v
    if not np.allclose(p11, expected_p11):
        raise ValueError("Tensor faces must be parallelograms")

    transform = Affine2D.from_values(
        basis_u[0],
        basis_u[1],
        basis_v[0],
        basis_v[1],
        p00[0],
        p00[1],
    ) + ax.transData
    ax.imshow(
        image,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        aspect="auto",
        cmap=RA_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        transform=transform,
        zorder=zorder,
    )
    ax.add_patch(
        patches.Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=CUBE_EDGE,
            linewidth=1.05,
            joinstyle="round",
            zorder=zorder + 0.4,
        )
    )
    ax.text(
        label_xy[0],
        label_xy[1],
        label,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=NAVY,
        rotation=label_rotation,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor=WHITE,
            edgecolor=CUBE_EDGE,
            linewidth=0.55,
            alpha=0.90,
        ),
        zorder=zorder + 0.8,
    )


def draw_tensor_cube(
    ax,
    x: float,
    y: float,
    *,
    maps: dict[str, np.ndarray],
    title: str,
    shape: str,
    side_view: str,
    top_view: str,
) -> None:
    """Draw a tensor cube whose three visible faces show real radar views."""
    width, height, depth, rise = 18.5, 18.0, 8.0, 4.8
    front_image = maps["RA"]
    side_image = maps[side_view]
    top_image = maps[top_view].T
    if side_image.shape[0] != front_image.shape[0]:
        raise ValueError(f"{side_view} range axis does not match the RA front")
    if top_image.shape[1] != front_image.shape[1]:
        raise ValueError(f"{top_view} azimuth axis does not match the RA front")
    if side_image.shape[1] != top_image.shape[0]:
        raise ValueError(f"{side_view}/{top_view} depth axes do not match")
    front = (
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    )
    top = (
        (x, y + height),
        (x + width, y + height),
        (x + width + depth, y + height + rise),
        (x + depth, y + height + rise),
    )
    side = (
        (x + width, y),
        (x + width + depth, y + rise),
        (x + width + depth, y + height + rise),
        (x + width, y + height),
    )

    # Top image rows must follow depth (D/E), while columns follow azimuth.
    draw_textured_face(
        ax,
        top_image,
        top,
        label=f"{top_view} view",
        label_xy=(x + width / 2 + depth / 2, y + height + rise / 2),
        zorder=3.0,
    )
    # Side image rows follow range, while columns follow depth (D/E).
    draw_textured_face(
        ax,
        side_image,
        side,
        label=f"{side_view} view",
        label_xy=(x + width + depth / 2, y + height / 2 + rise / 2),
        label_rotation=90.0,
        zorder=4.0,
    )
    # Front image rows follow range, while columns follow azimuth.
    draw_textured_face(
        ax,
        front_image,
        front,
        label="RA view",
        label_xy=(x + width / 2, y + 1.5),
        zorder=5.0,
    )
    ax.text(x + width / 2 + depth / 2, y + height + rise + 2.5, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(x + width / 2, y - 2.3, shape, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)


def draw_encoder(ax, x: float, y: float, *, title: str) -> None:
    width, height = 22.0, 18.0
    points = [(x, y), (x + width, y + 3.2), (x + width, y + height - 3.2), (x, y + height)]
    ax.add_patch(
        patches.Polygon(
            points,
            closed=True,
            facecolor=PALE_BLUE,
            edgecolor=BLUE,
            linewidth=1.15,
            joinstyle="round",
            zorder=3,
        )
    )
    for offset in (5.5, 11.0, 16.5):
        lower = y + offset * 3.2 / width
        upper = y + height - offset * 3.2 / width
        ax.plot([x + offset, x + offset], [lower, upper], color=BLUE, linewidth=0.5, alpha=0.42, zorder=4)
    ax.text(x + 10.4, y + 10.1, f"{title} encoder", ha="center", va="center", fontsize=9.4, fontweight="bold", color=BLUE, zorder=6)
    ax.text(x + 10.4, y + 6.7, "Swin–FPN", ha="center", va="center", fontsize=7.3, color=INK, zorder=6)


def draw_fusion(ax, x: float, y: float) -> None:
    cuboid(ax, x + 1.2, y + 1.5, 10.5, 22.0, depth=2.4, face=PALE_BLUE, edge=BLUE, linewidth=0.75, alpha=0.70, zorder=3)
    cuboid(ax, x + 0.6, y + 0.8, 10.5, 22.0, depth=2.4, face=PALE_BLUE, edge=BLUE, linewidth=0.75, alpha=0.82, zorder=4)
    cuboid(
        ax,
        x,
        y,
        10.5,
        22.0,
        depth=2.4,
        face=PALE_PURPLE,
        edge=PURPLE,
        title="encoded\nconcat + fusion",
        title_size=8.0,
        linewidth=1.0,
        zorder=5,
    )
    ax.text(x + 5.25, y - 2.4, "shared latent", ha="center", va="center", fontsize=6.8, fontweight="bold", color=PURPLE)


def draw_decoder(ax, x: float, y: float) -> None:
    width, height = 22.0, 25.0
    points = [(x, y + 4.5), (x + width, y), (x + width, y + height), (x, y + height - 4.5)]
    ax.add_patch(
        patches.Polygon(
            points,
            closed=True,
            facecolor=PALE_PURPLE,
            edgecolor=PURPLE,
            linewidth=1.2,
            joinstyle="round",
            zorder=3,
        )
    )
    for offset in (5.5, 11.0, 16.5):
        lower = y + 4.5 * (1.0 - offset / width)
        upper = y + height - 4.5 * (1.0 - offset / width)
        ax.plot([x + offset, x + offset], [lower, upper], color=PURPLE, linewidth=0.5, alpha=0.42, zorder=4)
    ax.text(x + width / 2, y + 15.3, "CenterPoint", ha="center", va="center", fontsize=10.2, fontweight="bold", color=PURPLE, zorder=6)
    ax.text(x + width / 2, y + 11.5, "decoder", ha="center", va="center", fontsize=9.3, fontweight="bold", color=PURPLE, zorder=6)
    ax.text(x + width / 2, y + 7.5, "heatmap + box regression", ha="center", va="center", fontsize=6.7, color=INK, zorder=6)


def draw_output_placeholder(ax, x: float, y: float) -> None:
    width, height = 21.0, 29.0
    rounded_box(ax, x, y, width, height, face=PALE_GREEN, edge=GREEN, linewidth=1.1, radius=1.0, zorder=2)
    ax.text(x + width / 2, y + height - 3.2, "Detection output", ha="center", va="center", fontsize=9.6, fontweight="bold", color=GREEN, zorder=10)
    ax.text(x + width / 2, y + height - 6.1, "placeholder", ha="center", va="center", fontsize=7.2, fontweight="bold", color=MID, zorder=10)

    origin = np.asarray((x + 10.7, y + 5.2), dtype=float)
    basis_u = np.asarray((0.46, 0.18))
    basis_v = np.asarray((-0.38, 0.27))
    basis_z = np.asarray((0.0, 0.68))

    def project(u: float, v: float, z: float = 0.0) -> tuple[float, float]:
        point = origin + u * basis_u + v * basis_v + z * basis_z
        return float(point[0]), float(point[1])

    plane = [project(0, 0), project(16, 0), project(16, 15), project(0, 15)]
    ax.add_patch(patches.Polygon(plane, closed=True, facecolor=WHITE, edgecolor=GREEN, linewidth=0.85, linestyle=(0, (4, 2)), zorder=3))
    wireframe_box(ax, project, u=6.0, v=6.0, length=4.0, width=2.6, height=2.1, colour=GREEN, linewidth=0.95)
    ax.text(x + width / 2, y + 2.0, "final visualization to be revised", ha="center", va="center", fontsize=6.4, color=MID, fontstyle="italic")


def labelled_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    text: str,
    label_xy: tuple[float, float],
) -> None:
    arrow(ax, start, end, color=BLUE, linewidth=1.25, mutation=9.5)
    ax.text(
        label_xy[0],
        label_xy[1],
        text,
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.23", facecolor=WHITE, edgecolor=BLUE, linewidth=0.65),
        zorder=12,
    )


def build_figure(
    rad_maps: dict[str, np.ndarray],
    rae_maps: dict[str, np.ndarray],
    source_note: str,
):
    setup_style()
    mpl.rcParams["font.serif"] = ["Liberation Serif"]
    fig, ax = plt.subplots(figsize=(21.6, 9.2))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 216)
    ax.set_ylim(0, 92)
    ax.axis("off")

    draw_header(ax)
    stage_heading(ax, 14.0, "4-D INPUT", NAVY)
    stage_heading(ax, 59.0, "MULTI-VIEW TENSORS", BLUE)
    stage_heading(ax, 89.0, "DUAL ENCODERS", BLUE)
    stage_heading(ax, 144.0, "FUSION + DECODING", PURPLE)
    stage_heading(ax, 187.0, "OUTPUT", GREEN)

    draw_four_axis_tensor_glyph(ax, 1.5, 24.5)

    labelled_arrow(ax, (29.5, 58.5), (45.5, 55.5), text="mean over E", label_xy=(37.5, 62.3))
    labelled_arrow(ax, (29.5, 34.0), (45.5, 25.5), text="mean over D", label_xy=(37.0, 36.7))
    draw_tensor_cube(
        ax,
        46.0,
        46.5,
        maps=rad_maps,
        title="RAD tensor",
        shape="[R, A, D]",
        side_view="RD",
        top_view="AD",
    )
    draw_tensor_cube(
        ax,
        46.0,
        16.5,
        maps=rae_maps,
        title="RAE tensor",
        shape="[R, A, E]",
        side_view="RE",
        top_view="AE",
    )

    arrow(ax, (72.8, 55.5), (77.5, 55.5), color=BLUE, linewidth=1.2, mutation=9.0)
    arrow(ax, (72.8, 25.5), (77.5, 25.5), color=BLUE, linewidth=1.2, mutation=9.0)
    draw_encoder(ax, 78.0, 46.5, title="RAD")
    draw_encoder(ax, 78.0, 16.5, title="RAE")
    ax.text(89.0, 43.0, "independent weights", ha="center", va="center", fontsize=6.8, color=MID, fontstyle="italic")

    arrow(ax, (100.3, 55.5), (112.5, 49.5), color=BLUE, linewidth=1.25, connection="arc3,rad=0.08", mutation=9.5)
    arrow(ax, (100.3, 25.5), (112.5, 39.5), color=BLUE, linewidth=1.25, connection="arc3,rad=-0.08", mutation=9.5)
    draw_fusion(ax, 113.0, 32.0)
    arrow(ax, (126.5, 43.0), (133.5, 43.0), color=PURPLE, linewidth=1.35, mutation=10.0)
    draw_decoder(ax, 134.0, 30.5)
    labelled_arrow(ax, (156.5, 43.0), (175.5, 43.0), text="decode", label_xy=(166.0, 46.0))
    draw_output_placeholder(ax, 176.0, 28.5)

    ax.plot([4.0, 212.0], [9.5, 9.5], color=LIGHT, linewidth=0.9)
    ax.text(
        4.0,
        6.6,
        f"Real paired data: {source_note}. Every visible cube surface uses the same colormap and display-normalization rule.",
        ha="left",
        va="center",
        fontsize=6.8,
        color=INK,
    )
    ax.text(
        4.0,
        3.8,
        "RAD surfaces: RA front · RD side · AD top. RAE surfaces: RA front · RE side · AE top. Four-axis glyph adapted from 4DR P2T Fig. 2; artwork is original.",
        ha="left",
        va="center",
        fontsize=6.5,
        color=MID,
        fontstyle="italic",
    )
    return fig


def validate_projection_contract() -> None:
    probe = np.empty((2, 3, 4, 5), dtype=np.uint8)  # conceptual [R,A,D,E]
    if probe.mean(axis=3).shape != (2, 3, 4):
        raise AssertionError("mean over E must produce [R,A,D]")
    if probe.mean(axis=2).shape != (2, 3, 5):
        raise AssertionError("mean over D must produce [R,A,E]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--radar-root", type=Path, default=DEFAULT_RADAR_ROOT)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--frame", type=str, default=None)
    parser.add_argument("--formats", nargs="+", choices=("png", "svg", "pdf"), default=("png", "svg", "pdf"))
    parser.add_argument("--dpi", type=int, default=360)
    parser.add_argument("--skip-config-check", action="store_true")
    return parser.parse_args()


def export_figure(fig, out_dir: Path, formats: Iterable[str], dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        path = out_dir / f"{STEM}.{extension}"
        # The vector formats retain vector text/geometry while embedding the
        # real radar-map textures at publication-quality resolution.
        fig.savefig(path, dpi=dpi)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    if not args.skip_config_check:
        validate_current_config()
    validate_projection_contract()
    rad_maps, rae_maps, source_note = load_projection_maps(
        args.radar_root,
        seed=args.sample_seed,
        sequence=args.sequence,
        frame=args.frame,
    )
    fig = build_figure(rad_maps, rae_maps, source_note)
    print(source_note)
    for path in export_figure(fig, args.out_dir, args.formats, args.dpi):
        print(path)


if __name__ == "__main__":
    main()
