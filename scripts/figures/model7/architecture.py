#!/usr/bin/env python3
"""Draw the exact model7 architecture selected by ``train.py``.

The figure is generated from vector primitives and project-owned radar data.
It deliberately follows the executed PyTorch graph rather than the stale
``SwinFPNEncoder`` stride-4 docstring: the active FPN returns P1 at stride 2.

Outputs are written to ``figures/`` as editable SVG, PDF, and high-resolution
PNG files.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Iterable


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mvrss_model7_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from configs.data import RADAR_NPY_ROOT


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = ROOT / "figures"
DEFAULT_RADAR_ROOT = Path(RADAR_NPY_ROOT)
STEM = "model7_swin_fpn_centerpoint_architecture"
OVERVIEW_STEM = "model7_swin_fpn_centerpoint_overview"

# Colour-blind-friendly palette shared by the Model7 figure family.
NAVY = "#17324D"
BLUE = "#2474B5"
CYAN = "#39A9C6"
TEAL = "#2D8C8C"
ORANGE = "#E17C35"
RED = "#C94C4C"
GREEN = "#3C8D73"
PURPLE = "#7566A8"
GOLD = "#C6922E"
INK = "#252B31"
MID = "#65727E"
LIGHT = "#DCE5EB"
PANEL = "#F7F9FB"
PALE_BLUE = "#EAF4FA"
PALE_CYAN = "#E8F7F8"
PALE_TEAL = "#E8F4F2"
PALE_ORANGE = "#FCF0E6"
PALE_RED = "#FBECEC"
PALE_GREEN = "#EAF4EF"
PALE_PURPLE = "#F0EDF8"
WHITE = "#FFFFFF"


EXPECTED_CONFIG = {
    "model_type": "model7",
    "box_coordinate_mode": "cartesian",
    "loss_mode": "centerpoint",
    "model7_decoder_hidden_channels": "64",
    "include_bus_as_target": False,
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.3,
            "axes.linewidth": 0.7,
            "text.color": INK,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _literal_train_config(path: Path) -> dict[str, object]:
    """Read literal values from TRAIN_CONFIG without importing training code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "TRAIN_CONFIG"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        values: dict[str, object] = {}
        for key_node, value_node in zip(node.value.keys, node.value.values):
            if key_node is None:
                continue
            try:
                key = ast.literal_eval(key_node)
                value = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                continue
            if isinstance(key, str):
                values[key] = value
        return values
    raise RuntimeError(f"Could not find a literal TRAIN_CONFIG dictionary in {path}")


def validate_current_config() -> None:
    """Fail loudly if the diagram no longer describes the selected model."""
    config = _literal_train_config(ROOT / "configs" / "training.py")
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (diagram expects {expected!r})"
            for key, (actual, expected) in mismatches.items()
        )
        raise RuntimeError(
            "configs/training.py has changed, so this model7 diagram must be "
            "updated: "
            + details
        )


def rounded_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = WHITE,
    edge: str = MID,
    linewidth: float = 0.9,
    radius: float = 0.35,
    linestyle: str | tuple = "-",
    zorder: float = 2,
):
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.10,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def panel(ax, x: float, y: float, width: float, height: float, title: str) -> None:
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        face=PANEL,
        edge=LIGHT,
        linewidth=1.0,
        radius=0.55,
        zorder=0.5,
    )
    ax.text(
        x + 1.0,
        y + height - 1.05,
        title,
        ha="left",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=NAVY,
        zorder=8,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    linewidth: float = 1.25,
    style: str = "-|>",
    mutation: float = 10,
    connection: str = "arc3",
    linestyle: str | tuple = "-",
    zorder: float = 4,
):
    item = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=linewidth,
        color=color,
        connectionstyle=connection,
        linestyle=linestyle,
        shrinkA=1.8,
        shrinkB=1.8,
        zorder=zorder,
    )
    ax.add_patch(item)
    return item


def badge(ax, x: float, y: float, text: str, *, face: str, edge: str) -> float:
    width = 1.7 + 0.50 * len(text)
    rounded_box(
        ax,
        x,
        y,
        width,
        1.9,
        face=face,
        edge=edge,
        linewidth=0.8,
        radius=0.8,
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + 0.95,
        text,
        ha="center",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color=edge,
        zorder=5,
    )
    return width


def add_node(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    face: str = WHITE,
    edge: str = NAVY,
    radius: float = 0.72,
) -> None:
    circle = patches.Circle(
        (x, y),
        radius,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
        zorder=6,
    )
    ax.add_patch(circle)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color=edge,
        zorder=7,
    )


def normalize_map(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    low, high = np.nanpercentile(data, (2.0, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - low) / (high - low), 0.0, 1.0)


def first_project_radar_pair(root: Path) -> tuple[Path, Path] | None:
    preferred_rad = root / "22" / "rad" / "00599.npy"
    preferred_rae = root / "22" / "rae" / "00599.npy"
    if preferred_rad.is_file() and preferred_rae.is_file():
        return preferred_rad, preferred_rae

    for rad_path in root.glob("*/rad/*.npy"):
        rae_path = rad_path.parent.parent / "rae" / rad_path.name
        if rae_path.is_file():
            return rad_path, rae_path
    return None


def load_input_tensors(
    radar_root: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    pair = first_project_radar_pair(radar_root) if radar_root.is_dir() else None
    if pair is None:
        raise FileNotFoundError(
            "A paired real RAD/RAE frame is required for figure export, "
            f"but none was found under {radar_root}."
        )

    rad_path, rae_path = pair
    rad = np.load(rad_path, mmap_mode="r")
    rae = np.load(rae_path, mmap_mode="r")
    if rad.ndim != 3 or rae.ndim != 3:
        raise ValueError(f"Expected 3-D RAD/RAE arrays, got {rad.shape} and {rae.shape}")
    if rad.shape[:2] != rae.shape[:2]:
        raise ValueError(
            "RAD and RAE range–azimuth dimensions must match, got "
            f"{rad.shape[:2]} and {rae.shape[:2]}"
        )
    return rad, rae, f"project frame {rad_path.parent.parent.name}/{rad_path.stem}"


def load_input_maps(radar_root: Path) -> tuple[np.ndarray, np.ndarray, str]:
    rad, rae, source_note = load_input_tensors(radar_root)
    rad_map = normalize_map(np.nanmax(rad, axis=2))
    rae_map = normalize_map(np.nanmax(rae, axis=2))
    return rad_map, rae_map, source_note


def representative_ra_slices(
    cube: np.ndarray,
    *,
    count: int = 5,
) -> tuple[np.ndarray, ...]:
    """Select and jointly normalize real R–A slices along the cube depth."""
    depth = cube.shape[2]
    if count < 2 or count > depth:
        raise ValueError(f"Slice count must be in [2, {depth}], got {count}")

    # Sample the full hidden axis, then place its strongest real slice at the
    # front of the visual stack so the visible plane remains informative.
    sampled = np.linspace(0, depth - 1, count - 1).round().astype(int).tolist()
    strength = np.nanpercentile(cube, 99.5, axis=(0, 1))
    strongest = int(np.nanargmax(strength))
    indices = [index for index in sampled if index != strongest]
    for index in range(depth):
        if len(indices) >= count - 1:
            break
        if index != strongest and index not in indices:
            indices.append(index)
    indices = indices[: count - 1] + [strongest]

    selected = np.stack(
        [np.asarray(cube[:, :, index], dtype=np.float32) for index in indices],
        axis=0,
    )
    low, high = np.nanpercentile(selected, (2.0, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Selected real radar slices have no finite display range")
    normalized = np.clip((selected - low) / (high - low), 0.0, 1.0)
    return tuple(normalized[index] for index in range(count))


def load_input_slice_stacks(
    radar_root: Path,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], str]:
    """Load real RA-plane stacks whose depth represents D for RAD and E for RAE."""
    rad, rae, source_note = load_input_tensors(radar_root)
    return (
        representative_ra_slices(rad),
        representative_ra_slices(rae),
        f"{source_note} · real RA slices",
    )


def draw_input_card(
    ax,
    x: float,
    y: float,
    *,
    title: str,
    subtitle: str,
    shape: str,
    image: np.ndarray,
    cmap: str,
    edge: str,
) -> None:
    width, height = 8.1, 6.7
    rounded_box(ax, x, y, width, height, face=WHITE, edge=edge, linewidth=1.0)
    ax.text(
        x + 0.55,
        y + height - 0.55,
        title,
        ha="left",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=edge,
        zorder=8,
    )
    ax.text(
        x + 0.57,
        y + height - 1.23,
        subtitle,
        ha="left",
        va="center",
        fontsize=5.2,
        color=MID,
        zorder=8,
    )
    ax.imshow(
        image,
        extent=(x + 0.55, x + width - 0.55, y + 1.35, y + height - 1.72),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bilinear",
        zorder=3,
    )
    ax.add_patch(
        patches.Rectangle(
            (x + 0.55, y + 1.35),
            width - 1.1,
            height - 3.07,
            fill=False,
            edgecolor=edge,
            linewidth=0.65,
            zorder=6,
        )
    )
    ax.text(
        x + width / 2,
        y + 0.65,
        shape,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
        zorder=8,
    )


def mini_stage(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    shape: str,
    face: str,
    edge: str,
) -> None:
    # Offset back planes make the feature tensor visually three-dimensional.
    for offset, alpha in ((0.38, 0.32), (0.19, 0.55)):
        ax.add_patch(
            patches.Rectangle(
                (x + offset, y + offset),
                width,
                height,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.55,
                alpha=alpha,
                zorder=2,
            )
        )
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        face=face,
        edge=edge,
        linewidth=0.85,
        radius=0.18,
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.61,
        title,
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=edge,
        zorder=6,
    )
    ax.text(
        x + width / 2,
        y + height * 0.24,
        shape,
        ha="center",
        va="center",
        fontsize=5.8,
        color=INK,
        zorder=6,
    )


def draw_encoder_summary(
    ax,
    x: float,
    y: float,
    *,
    modality: str,
    accent: str,
    pale: str,
) -> None:
    width, height = 26.2, 7.0
    rounded_box(ax, x, y, width, height, face=pale, edge=accent, linewidth=1.05)
    ax.text(
        x + 0.85,
        y + height - 0.72,
        f"{modality} Swin–FPN encoder",
        ha="left",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=accent,
        zorder=7,
    )
    ax.text(
        x + width - 0.75,
        y + height - 0.72,
        "independent weights",
        ha="right",
        va="center",
        fontsize=5.9,
        fontstyle="italic",
        color=MID,
        zorder=7,
    )

    mini_stage(
        ax,
        x + 0.9,
        y + 1.45,
        4.8,
        3.25,
        title="S1 · Swin ×2",
        shape="64 · 128×54",
        face=WHITE,
        edge=accent,
    )
    mini_stage(
        ax,
        x + 7.0,
        y + 1.75,
        4.5,
        2.75,
        title="S2 · Swin ×2",
        shape="128 · 64×27",
        face=WHITE,
        edge=accent,
    )
    mini_stage(
        ax,
        x + 12.7,
        y + 2.05,
        4.1,
        2.25,
        title="S3 · Swin ×2",
        shape="256 · 32×14",
        face=WHITE,
        edge=accent,
    )
    arrow(ax, (x + 5.85, y + 3.05), (x + 6.9, y + 3.05), color=accent, mutation=8)
    arrow(ax, (x + 11.65, y + 3.05), (x + 12.6, y + 3.05), color=accent, mutation=8)

    rounded_box(
        ax,
        x + 18.1,
        y + 1.20,
        7.0,
        3.8,
        face=PALE_ORANGE,
        edge=ORANGE,
        linewidth=0.9,
        radius=0.3,
    )
    ax.text(
        x + 21.6,
        y + 3.80,
        "Top-down FPN",
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=ORANGE,
        zorder=7,
    )
    ax.text(
        x + 21.6,
        y + 2.65,
        "lateral 1×1 · upsample · add",
        ha="center",
        va="center",
        fontsize=5.5,
        color=INK,
        zorder=7,
    )
    ax.text(
        x + 21.6,
        y + 1.78,
        "P1  [B, 128, 128, 54]",
        ha="center",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=NAVY,
        zorder=7,
    )
    arrow(
        ax,
        (x + 16.95, y + 3.05),
        (x + 18.0, y + 3.05),
        color=ORANGE,
        mutation=8,
    )


def draw_fusion_card(ax, x: float, y: float) -> None:
    rounded_box(ax, x, y, 14.2, 7.2, face=PALE_PURPLE, edge=PURPLE, linewidth=1.05)
    ax.text(
        x + 7.1,
        y + 6.25,
        "RAD / RAE feature fusion",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=PURPLE,
        zorder=7,
    )
    ax.text(
        x + 7.1,
        y + 4.85,
        "Concat: 128 + 128 = 256 ch",
        ha="center",
        va="center",
        fontsize=6.5,
        color=INK,
        zorder=7,
    )
    ax.text(
        x + 7.1,
        y + 3.55,
        "1×1 Conv–BN–LeakyReLU  256→128",
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
        zorder=7,
    )
    ax.text(
        x + 7.1,
        y + 2.25,
        "residual refine: 3×3 → 3×3 + skip",
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
        zorder=7,
    )
    ax.text(
        x + 7.1,
        y + 0.90,
        "fused  [B, 128, 128, 54]",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=NAVY,
        zorder=7,
    )


def draw_head_card(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    detail: str,
    face: str,
    edge: str,
) -> None:
    rounded_box(ax, x, y, width, height, face=face, edge=edge, linewidth=1.0)
    ax.text(
        x + 0.75,
        y + height - 0.85,
        title,
        ha="left",
        va="center",
        fontsize=7.9,
        fontweight="bold",
        color=edge,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + (height - 0.85) * 0.48,
        detail,
        ha="center",
        va="center",
        fontsize=6.2,
        linespacing=1.22,
        color=INK,
        zorder=7,
    )


def draw_dense_output_card(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    shape: str,
    detail: str,
    face: str,
    edge: str,
) -> None:
    rounded_box(ax, x, y, width, height, face=WHITE, edge=edge, linewidth=0.95)
    ax.add_patch(
        patches.Rectangle(
            (x + 0.55, y + 0.70),
            1.25,
            height - 1.40,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.65,
            zorder=3,
        )
    )
    for index in range(3):
        ax.add_patch(
            patches.Circle(
                (x + 1.17, y + 1.20 + index * (height - 2.4) / 2),
                0.16 + 0.035 * index,
                facecolor=edge,
                edgecolor=WHITE,
                linewidth=0.35,
                alpha=0.65 + 0.1 * index,
                zorder=4,
            )
        )
    ax.text(
        x + 2.25,
        y + height - 0.90,
        title,
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=edge,
        zorder=7,
    )
    ax.text(
        x + 2.25,
        y + height * 0.50,
        shape,
        ha="left",
        va="center",
        fontsize=6.3,
        fontweight="bold",
        color=INK,
        zorder=7,
    )
    ax.text(
        x + 2.25,
        y + 0.78,
        detail,
        ha="left",
        va="center",
        fontsize=5.7,
        color=MID,
        zorder=7,
    )


def draw_bev_icon(ax, x: float, y: float, width: float, height: float) -> None:
    rounded_box(ax, x, y, width, height, face="#F5FAF8", edge=GREEN, linewidth=0.9)
    # A compact forward-view Cartesian grid with two rotated detections.
    for fraction in (0.25, 0.50, 0.75):
        ax.plot(
            [x + width * fraction, x + width * fraction],
            [y + 0.55, y + height - 0.55],
            color=LIGHT,
            linewidth=0.45,
            zorder=2,
        )
        ax.plot(
            [x + 0.55, x + width - 0.55],
            [y + height * fraction, y + height * fraction],
            color=LIGHT,
            linewidth=0.45,
            zorder=2,
        )
    for cx, cy, angle, scale in (
        (x + width * 0.43, y + height * 0.55, 18, 1.0),
        (x + width * 0.70, y + height * 0.34, -28, 0.75),
    ):
        rect = patches.Rectangle(
            (cx - 0.72 * scale, cy - 0.36 * scale),
            1.44 * scale,
            0.72 * scale,
            angle=angle,
            rotation_point="center",
            facecolor="none",
            edgecolor=RED,
            linewidth=1.15,
            zorder=5,
        )
        ax.add_patch(rect)
        ax.add_patch(patches.Circle((cx, cy), 0.12, fc=RED, ec=WHITE, lw=0.3, zorder=6))
    ax.annotate(
        "",
        xy=(x + width / 2, y + height - 0.38),
        xytext=(x + width / 2, y + 0.40),
        arrowprops=dict(arrowstyle="-|>", mutation_scale=7, lw=0.8, color=NAVY),
        zorder=4,
    )


def draw_overview(
    ax,
    rad_map: np.ndarray,
    rae_map: np.ndarray,
    *,
    show_panel_title: bool = True,
) -> None:
    title = "A  |  Executed end-to-end graph" if show_panel_title else ""
    panel(ax, 1.0, 33.2, 98.0, 21.2, title)

    draw_input_card(
        ax,
        2.7,
        44.7,
        title="RAD",
        subtitle="range–azimuth–Doppler",
        shape="[B, 64, 256, 107]",
        image=rad_map,
        cmap="magma",
        edge=BLUE,
    )
    draw_input_card(
        ax,
        2.7,
        35.8,
        title="RAE",
        subtitle="range–azimuth–elevation",
        shape="[B, 37, 256, 107]",
        image=rae_map,
        cmap="viridis",
        edge=TEAL,
    )

    draw_encoder_summary(ax, 13.1, 44.55, modality="RAD", accent=BLUE, pale=PALE_BLUE)
    draw_encoder_summary(ax, 13.1, 35.65, modality="RAE", accent=TEAL, pale=PALE_TEAL)
    arrow(ax, (10.9, 48.05), (13.0, 48.05), color=BLUE)
    arrow(ax, (10.9, 39.15), (13.0, 39.15), color=TEAL)

    add_node(ax, 42.1, 43.60, "C", face=WHITE, edge=PURPLE, radius=0.73)
    ax.text(
        42.1,
        42.35,
        "concat",
        ha="center",
        va="top",
        fontsize=5.5,
        color=PURPLE,
        zorder=7,
    )
    arrow(
        ax,
        (39.35, 48.05),
        (41.47, 44.12),
        color=BLUE,
        connection="arc3,rad=-0.15",
    )
    arrow(
        ax,
        (39.35, 39.15),
        (41.47, 43.08),
        color=TEAL,
        connection="arc3,rad=0.15",
    )

    draw_fusion_card(ax, 44.2, 40.0)
    arrow(ax, (42.82, 43.60), (44.1, 43.60), color=PURPLE)

    add_node(ax, 60.2, 43.60, "Y", face=WHITE, edge=NAVY, radius=0.67)
    ax.text(
        60.2,
        42.42,
        "split",
        ha="center",
        va="top",
        fontsize=5.4,
        color=NAVY,
        zorder=7,
    )
    arrow(ax, (58.42, 43.60), (59.48, 43.60), color=NAVY)

    draw_head_card(
        ax,
        62.1,
        46.2,
        16.1,
        5.4,
        title="Sedan classification head",
        detail="3×3 Conv–BN–Act  128→64\n1×1 Conv  64→1",
        face=PALE_RED,
        edge=RED,
    )
    draw_head_card(
        ax,
        62.1,
        36.0,
        16.1,
        7.1,
        title="Cartesian box head",
        detail="shared 3×3 Conv–BN–Act ×2\n128→64→64\nparallel 1×1 branches: 2 + 1 + 3 + 2",
        face=PALE_GREEN,
        edge=GREEN,
    )
    arrow(
        ax,
        (60.72, 44.05),
        (62.0, 48.90),
        color=RED,
        connection="arc3,rad=-0.13",
    )
    arrow(
        ax,
        (60.72, 43.15),
        (62.0, 39.55),
        color=GREEN,
        connection="arc3,rad=0.13",
    )

    draw_dense_output_card(
        ax,
        80.5,
        46.2,
        11.3,
        5.4,
        title="center logits",
        shape="[B, 1, 128, 54]",
        detail="raw dense Sedan heatmap",
        face=PALE_RED,
        edge=RED,
    )
    draw_dense_output_card(
        ax,
        80.5,
        36.0,
        11.3,
        7.1,
        title="box_reg",
        shape="[B, 8, 128, 54]",
        detail="dx,dy,dz,l,w,h,sinθ,cosθ",
        face=PALE_GREEN,
        edge=GREEN,
    )
    arrow(ax, (78.3, 48.90), (80.4, 48.90), color=RED)
    arrow(ax, (78.3, 39.55), (80.4, 39.55), color=GREEN)

    add_node(ax, 94.3, 43.75, "D", face=WHITE, edge=NAVY, radius=0.58)
    ax.text(
        94.3,
        46.0,
        "top-K · decode\nrotated NMS",
        ha="center",
        va="center",
        fontsize=5.4,
        color=MID,
        linespacing=1.10,
        bbox=dict(boxstyle="round,pad=0.10", facecolor=PANEL, edgecolor="none", alpha=0.94),
        zorder=7,
    )
    arrow(
        ax,
        (91.9, 48.90),
        (93.85, 44.20),
        color=RED,
        connection="arc3,rad=-0.12",
        linestyle=(0, (3, 2)),
        mutation=8,
    )
    arrow(
        ax,
        (91.9, 39.55),
        (93.85, 43.30),
        color=GREEN,
        connection="arc3,rad=0.12",
        linestyle=(0, (3, 2)),
        mutation=8,
    )
    draw_bev_icon(ax, 96.0, 36.0, 2.3, 7.1)
    arrow(
        ax,
        (94.82, 43.55),
        (95.9, 40.05),
        color=NAVY,
        connection="arc3,rad=-0.16",
        mutation=8,
    )


def folded_input_card(
    ax,
    x: float,
    y: float,
    *,
    title: str,
    subtitle: str,
    shape: str,
    slices: tuple[np.ndarray, ...],
    depth_axis: str,
    total_slices: int,
    cmap: str,
    edge: str,
) -> None:
    """Show a tensor as a stack of real R–A planes along D or E."""
    width, height = 11.0, 11.5
    rounded_box(ax, x, y, width, height, face=WHITE, edge=edge, linewidth=1.25, radius=0.45)
    ax.text(
        x + 0.65,
        y + height - 0.85,
        title,
        ha="left",
        va="center",
        fontsize=12.6,
        fontweight="bold",
        color=edge,
        zorder=8,
    )
    ax.text(
        x + 0.68,
        y + height - 1.75,
        subtitle,
        ha="left",
        va="center",
        fontsize=7.5,
        color=MID,
        zorder=8,
    )

    if len(slices) < 2:
        raise ValueError("At least two R–A slices are required to show tensor depth")

    front_x, front_y = x + 0.72, y + 2.20
    plane_width, plane_height = 7.90, 5.65
    step_x, step_y = 0.34, 0.27
    for index, image in enumerate(slices):
        steps_back = len(slices) - index - 1
        layer_x = front_x + steps_back * step_x
        layer_y = front_y + steps_back * step_y
        zorder = 3.0 + index * 0.45
        ax.imshow(
            image,
            extent=(
                layer_x,
                layer_x + plane_width,
                layer_y,
                layer_y + plane_height,
            ),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            alpha=0.86 if steps_back else 1.0,
            zorder=zorder,
        )
        ax.add_patch(
            patches.Rectangle(
                (layer_x, layer_y),
                plane_width,
                plane_height,
                fill=False,
                edgecolor=edge,
                linewidth=0.56 if steps_back else 0.82,
                alpha=0.72 if steps_back else 1.0,
                zorder=zorder + 0.2,
            )
        )

    ax.text(
        front_x + 0.78,
        front_y + 0.55,
        "RA slice",
        ha="center",
        va="center",
        fontsize=5.0,
        fontweight="bold",
        color=edge,
        bbox=dict(
            boxstyle="round,pad=0.13",
            facecolor=WHITE,
            edgecolor=edge,
            linewidth=0.42,
            alpha=0.90,
        ),
        zorder=8,
    )

    back_steps = len(slices) - 1
    depth_start = (front_x + plane_width + 0.08, front_y + plane_height + 0.05)
    depth_end = (
        depth_start[0] + back_steps * step_x,
        depth_start[1] + back_steps * step_y,
    )
    ax.annotate(
        "",
        xy=depth_end,
        xytext=depth_start,
        arrowprops=dict(arrowstyle="-|>", color=edge, linewidth=0.75),
        zorder=9,
    )
    ax.text(
        (depth_start[0] + depth_end[0]) / 2 + 0.15,
        (depth_start[1] + depth_end[1]) / 2 + 0.22,
        f"{depth_axis} = {total_slices}",
        ha="center",
        va="center",
        fontsize=5.0,
        fontweight="bold",
        color=edge,
        rotation=35.0,
        bbox=dict(
            boxstyle="round,pad=0.10",
            facecolor=WHITE,
            edgecolor="none",
            alpha=0.86,
        ),
        zorder=9,
    )
    ax.text(
        x + width / 2,
        y + 0.95,
        shape,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        zorder=8,
    )


def folded_stage_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    shape: str,
    edge: str,
) -> None:
    for offset, alpha in ((0.36, 0.25), (0.18, 0.45)):
        ax.add_patch(
            patches.Rectangle(
                (x + offset, y + offset),
                width,
                height,
                facecolor=WHITE,
                edgecolor=edge,
                linewidth=0.65,
                alpha=alpha,
                zorder=2,
            )
        )
    rounded_box(ax, x, y, width, height, face=WHITE, edge=edge, linewidth=1.0, radius=0.22)
    ax.text(
        x + width / 2,
        y + height * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=9.7,
        fontweight="bold",
        color=edge,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + height * 0.25,
        shape,
        ha="center",
        va="center",
        fontsize=8.4,
        color=INK,
        zorder=7,
    )


def folded_encoder_card(
    ax,
    x: float,
    y: float,
    *,
    modality: str,
    accent: str,
    pale: str,
) -> None:
    width, height = 38.5, 11.5
    rounded_box(ax, x, y, width, height, face=pale, edge=accent, linewidth=1.25, radius=0.48)
    ax.text(
        x + 0.9,
        y + height - 0.85,
        f"{modality} Swin–FPN encoder",
        ha="left",
        va="center",
        fontsize=12.4,
        fontweight="bold",
        color=accent,
        zorder=7,
    )
    ax.text(
        x + width - 0.8,
        y + height - 0.88,
        "independent weights",
        ha="right",
        va="center",
        fontsize=8.0,
        fontstyle="italic",
        color=MID,
        zorder=7,
    )

    stage_y = y + 2.15
    folded_stage_box(
        ax,
        x + 1.0,
        stage_y,
        6.5,
        5.6,
        title="S1 · Swin ×2",
        shape="64 · 128×54",
        edge=accent,
    )
    folded_stage_box(
        ax,
        x + 8.9,
        stage_y + 0.25,
        6.1,
        5.1,
        title="S2 · Swin ×2",
        shape="128 · 64×27",
        edge=accent,
    )
    folded_stage_box(
        ax,
        x + 16.3,
        stage_y + 0.50,
        5.8,
        4.6,
        title="S3 · Swin ×2",
        shape="256 · 32×14",
        edge=accent,
    )
    arrow(ax, (x + 7.65, y + 4.95), (x + 8.8, y + 4.95), color=accent, mutation=9)
    arrow(ax, (x + 15.15, y + 4.95), (x + 16.2, y + 4.95), color=accent, mutation=9)

    rounded_box(
        ax,
        x + 23.3,
        y + 1.85,
        14.0,
        6.7,
        face=PALE_ORANGE,
        edge=ORANGE,
        linewidth=1.05,
        radius=0.35,
    )
    ax.text(
        x + 30.3,
        y + 7.35,
        "Top-down FPN",
        ha="center",
        va="center",
        fontsize=11.3,
        fontweight="bold",
        color=ORANGE,
        zorder=7,
    )
    ax.text(
        x + 30.3,
        y + 5.25,
        "lateral 1×1 · bilinear ↑ · add",
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK,
        zorder=7,
    )
    ax.text(
        x + 30.3,
        y + 3.35,
        "P1  [B, 128, 128, 54]",
        ha="center",
        va="center",
        fontsize=9.7,
        fontweight="bold",
        color=NAVY,
        zorder=7,
    )
    arrow(ax, (x + 22.25, y + 4.95), (x + 23.2, y + 4.95), color=ORANGE, mutation=9)


def folded_fusion_card(ax, x: float, y: float) -> None:
    width, height = 18.0, 9.0
    rounded_box(ax, x, y, width, height, face=PALE_PURPLE, edge=PURPLE, linewidth=1.25, radius=0.45)
    ax.text(
        x + width / 2,
        y + 7.80,
        "RAD / RAE feature fusion",
        ha="center",
        va="center",
        fontsize=11.7,
        fontweight="bold",
        color=PURPLE,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + 5.95,
        "Concat 256 ch → 1×1 Conv–BN–Act → 128 ch",
        ha="center",
        va="center",
        fontsize=8.6,
        color=INK,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + 4.10,
        "residual refine: 3×3 → 3×3 + skip",
        ha="center",
        va="center",
        fontsize=8.6,
        color=INK,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + 1.85,
        "fused  [B, 128, 128, 54]",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=NAVY,
        zorder=7,
    )


def folded_head_card(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    lines: tuple[str, ...],
    face: str,
    edge: str,
) -> None:
    rounded_box(ax, x, y, width, height, face=face, edge=edge, linewidth=1.2, radius=0.42)
    ax.text(
        x + 0.85,
        y + height - 1.05,
        title,
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=edge,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + (height - 1.15) * 0.45,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=9.1,
        linespacing=1.18,
        color=INK,
        zorder=7,
    )


def folded_output_card(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    shape: str,
    detail: str,
    face: str,
    edge: str,
) -> None:
    rounded_box(ax, x, y, width, height, face=WHITE, edge=edge, linewidth=1.15, radius=0.42)
    ax.add_patch(
        patches.Rectangle(
            (x + 0.7, y + 0.8),
            1.35,
            height - 1.6,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.75,
            zorder=3,
        )
    )
    for index in range(3):
        ax.add_patch(
            patches.Circle(
                (x + 1.38, y + 1.25 + index * (height - 2.5) / 2),
                0.17,
                facecolor=edge,
                edgecolor=WHITE,
                linewidth=0.35,
                alpha=0.65 + 0.1 * index,
                zorder=4,
            )
        )
    ax.text(
        x + 2.55,
        y + height - 1.0,
        title,
        ha="left",
        va="center",
        fontsize=11.1,
        fontweight="bold",
        color=edge,
        zorder=7,
    )
    ax.text(
        x + 2.55,
        y + height * 0.50,
        shape,
        ha="left",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color=INK,
        zorder=7,
    )
    ax.text(
        x + 2.55,
        y + 1.20,
        detail,
        ha="left",
        va="center",
        fontsize=7.8,
        linespacing=1.05,
        color=MID,
        zorder=7,
    )


def draw_folded_overview(
    ax,
    rad_slices: tuple[np.ndarray, ...],
    rae_slices: tuple[np.ndarray, ...],
) -> None:
    """Draw the compact two-row overview requested for the standalone export."""
    # Keep both physical input views visually equal: RAE's established teal
    # accent and viridis scale are shared by the complete RAD and RAE streams.
    stream_accent = TEAL
    stream_pale = PALE_TEAL
    stream_cmap = "viridis"

    rounded_box(
        ax,
        1.0,
        16.0,
        72.0,
        55.0,
        face=PANEL,
        edge=LIGHT,
        linewidth=1.1,
        radius=0.60,
        zorder=0.5,
    )

    folded_input_card(
        ax,
        3.0,
        57.5,
        title="RAD",
        subtitle="",
        shape="[B, D=64, R=256, A=107]",
        slices=rad_slices,
        depth_axis="D",
        total_slices=64,
        cmap=stream_cmap,
        edge=stream_accent,
    )
    folded_input_card(
        ax,
        3.0,
        43.5,
        title="RAE",
        subtitle="",
        shape="[B, E=37, R=256, A=107]",
        slices=rae_slices,
        depth_axis="E",
        total_slices=37,
        cmap=stream_cmap,
        edge=stream_accent,
    )
    folded_encoder_card(
        ax,
        17.0,
        57.5,
        modality="RAD",
        accent=stream_accent,
        pale=stream_pale,
    )
    folded_encoder_card(
        ax,
        17.0,
        43.5,
        modality="RAE",
        accent=stream_accent,
        pale=stream_pale,
    )
    arrow(
        ax,
        (14.1, 63.25),
        (16.9, 63.25),
        color=stream_accent,
        mutation=11,
        linewidth=1.5,
    )
    arrow(
        ax,
        (14.1, 49.25),
        (16.9, 49.25),
        color=stream_accent,
        mutation=11,
        linewidth=1.5,
    )

    add_node(ax, 61.0, 55.9, "C", face=WHITE, edge=PURPLE, radius=0.85)
    ax.text(
        61.0,
        54.35,
        "concat",
        ha="center",
        va="top",
        fontsize=8.6,
        color=PURPLE,
        zorder=7,
    )
    arrow(
        ax,
        (55.6, 63.25),
        (60.28, 56.42),
        color=stream_accent,
        connection="arc3,rad=-0.11",
        mutation=11,
        linewidth=1.5,
    )
    arrow(
        ax,
        (55.6, 49.25),
        (60.28, 55.38),
        color=stream_accent,
        connection="arc3,rad=0.11",
        mutation=11,
        linewidth=1.5,
    )

    # The fold: concatenate first, then move vertically down into fusion.
    folded_fusion_card(ax, 52.0, 27.5)
    arrow(
        ax,
        (61.0, 55.0),
        (61.0, 36.6),
        color=PURPLE,
        mutation=12,
        linewidth=1.7,
    )

    add_node(ax, 48.0, 32.0, "Y", face=WHITE, edge=NAVY, radius=0.78)
    ax.text(48.0, 30.55, "split", ha="center", va="top", fontsize=8.4, color=NAVY, zorder=7)
    arrow(ax, (51.9, 32.0), (48.85, 32.0), color=NAVY, mutation=11, linewidth=1.5)

    folded_head_card(
        ax,
        29.0,
        34.5,
        16.0,
        7.5,
        title="Sedan classification head",
        lines=("3×3 Conv–BN–Act 128→64", "1×1 Conv 64→1"),
        face=PALE_RED,
        edge=RED,
    )
    folded_head_card(
        ax,
        29.0,
        19.0,
        16.0,
        10.0,
        title="Cartesian box head",
        lines=("shared 3×3 Conv–BN–Act ×2", "128→64→64", "parallel branches: 2 + 1 + 3 + 2"),
        face=PALE_GREEN,
        edge=GREEN,
    )
    arrow(
        ax,
        (47.35, 32.48),
        (45.1, 38.25),
        color=RED,
        connection="arc3,rad=0.12",
        mutation=10,
        linewidth=1.35,
    )
    arrow(
        ax,
        (47.35, 31.52),
        (45.1, 24.0),
        color=GREEN,
        connection="arc3,rad=-0.12",
        mutation=10,
        linewidth=1.35,
    )

    folded_output_card(
        ax,
        14.0,
        34.5,
        12.0,
        7.5,
        title="center logits",
        shape="[B,1,128,54]",
        detail="Sedan heatmap",
        face=PALE_RED,
        edge=RED,
    )
    folded_output_card(
        ax,
        14.0,
        19.0,
        12.0,
        10.0,
        title="box_reg",
        shape="[B,8,128,54]",
        detail="dx,dy,dz,l,w,h\nsinθ,cosθ",
        face=PALE_GREEN,
        edge=GREEN,
    )
    arrow(ax, (28.9, 38.25), (26.1, 38.25), color=RED, mutation=11, linewidth=1.5)
    arrow(ax, (28.9, 24.0), (26.1, 24.0), color=GREEN, mutation=11, linewidth=1.5)

    add_node(ax, 10.2, 31.5, "D", face=WHITE, edge=NAVY, radius=0.78)
    ax.text(
        9.6,
        35.2,
        "top-K · decode\nrotated NMS",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MID,
        linespacing=1.12,
        bbox=dict(boxstyle="round,pad=0.12", facecolor=PANEL, edgecolor="none", alpha=0.94),
        zorder=8,
    )
    arrow(
        ax,
        (13.9, 38.25),
        (10.73, 32.10),
        color=RED,
        connection="arc3,rad=-0.12",
        linestyle=(0, (3, 2)),
        mutation=9,
        linewidth=1.25,
    )
    arrow(
        ax,
        (13.9, 24.0),
        (10.73, 30.90),
        color=GREEN,
        connection="arc3,rad=0.12",
        linestyle=(0, (3, 2)),
        mutation=9,
        linewidth=1.25,
    )
    draw_bev_icon(ax, 3.0, 26.3, 4.7, 10.4)
    arrow(ax, (9.35, 31.5), (7.8, 31.5), color=NAVY, mutation=10, linewidth=1.35)
    ax.text(
        5.35,
        24.95,
        "metric Cartesian boxes",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=GREEN,
        zorder=7,
    )


def draw_stage_detail(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    detail: str,
    shape: str,
    face: str,
    edge: str,
) -> None:
    rounded_box(ax, x, y, width, height, face=face, edge=edge, linewidth=0.9)
    ax.text(
        x + width / 2,
        y + height - 0.83,
        title,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=edge,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + height * 0.50,
        detail,
        ha="center",
        va="center",
        fontsize=5.7,
        color=INK,
        linespacing=1.10,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + 0.63,
        shape,
        ha="center",
        va="center",
        fontsize=6.1,
        fontweight="bold",
        color=NAVY,
        zorder=7,
    )


def draw_fpn_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    shape: str,
) -> None:
    rounded_box(ax, x, y, width, height, face=PALE_ORANGE, edge=ORANGE, linewidth=0.9)
    ax.text(
        x + width / 2,
        y + height * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=6.9,
        fontweight="bold",
        color=ORANGE,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + height * 0.26,
        shape,
        ha="center",
        va="center",
        fontsize=5.8,
        color=INK,
        zorder=7,
    )


def draw_encoder_detail(ax) -> None:
    panel(
        ax,
        1.0,
        8.2,
        63.1,
        23.5,
        "B  |  Inside one Swin–FPN encoder  (instantiated twice; weights are not shared)",
    )

    draw_stage_detail(
        ax,
        2.8,
        12.7,
        7.2,
        5.2,
        title="Patch embedding",
        detail="pad A: 107→108\nConv 2×2, stride 2 + LN",
        shape="Cin→64",
        face=PALE_CYAN,
        edge=CYAN,
    )
    draw_stage_detail(
        ax,
        13.1,
        12.0,
        9.0,
        6.4,
        title="Stage 1 · 2 Swin blocks",
        detail="dim 64 · 2 heads\nW-MSA / SW-MSA",
        shape="C1  [B,64,128,54]",
        face=PALE_BLUE,
        edge=BLUE,
    )
    draw_stage_detail(
        ax,
        28.0,
        12.7,
        8.1,
        5.2,
        title="Stage 2 · 2 blocks",
        detail="dim 128 · 4 heads\nW-MSA / SW-MSA",
        shape="C2  [B,128,64,27]",
        face=PALE_BLUE,
        edge=BLUE,
    )
    draw_stage_detail(
        ax,
        42.0,
        13.3,
        7.5,
        4.0,
        title="Stage 3 · 2 blocks",
        detail="dim 256 · 8 heads",
        shape="C3  [B,256,32,14]",
        face=PALE_BLUE,
        edge=BLUE,
    )
    arrow(ax, (10.1, 15.3), (13.0, 15.3), color=BLUE)
    arrow(ax, (22.2, 15.3), (27.9, 15.3), color=BLUE)
    arrow(ax, (36.2, 15.3), (41.9, 15.3), color=BLUE)
    ax.text(25.0, 16.0, "patch merge", ha="center", va="bottom", fontsize=5.6, color=MID)
    ax.text(39.0, 16.0, "patch merge\n(pad 27→28)", ha="center", va="bottom", fontsize=5.4, color=MID)

    draw_fpn_box(ax, 43.0, 23.5, 6.5, 3.4, title="P3 · lateral + smooth", shape="128 · 32×14")
    draw_fpn_box(ax, 29.0, 23.0, 7.1, 4.0, title="P2 · ⊕ + smooth", shape="128 · 64×27")
    draw_fpn_box(ax, 14.0, 22.5, 8.0, 4.6, title="P1 · ⊕ + smooth", shape="128 · 128×54")
    draw_fpn_box(ax, 2.8, 22.5, 7.4, 4.6, title="output refine", shape="2× 3×3 · map stride 2")

    arrow(ax, (45.75, 17.4), (45.75, 23.4), color=ORANGE)
    arrow(ax, (32.05, 18.0), (32.05, 22.9), color=ORANGE)
    arrow(ax, (17.6, 18.5), (17.6, 22.4), color=ORANGE)
    ax.text(46.65, 20.3, "1×1", ha="left", va="center", fontsize=5.4, color=ORANGE)
    ax.text(32.95, 20.3, "1×1", ha="left", va="center", fontsize=5.4, color=ORANGE)
    ax.text(18.50, 20.3, "1×1", ha="left", va="center", fontsize=5.4, color=ORANGE)
    arrow(ax, (42.9, 25.2), (36.2, 25.2), color=ORANGE)
    arrow(ax, (28.9, 25.2), (22.1, 25.2), color=ORANGE)
    arrow(ax, (13.9, 24.8), (10.3, 24.8), color=ORANGE)
    ax.text(39.6, 26.0, "bilinear ↑", ha="center", va="bottom", fontsize=5.5, color=ORANGE)
    ax.text(25.5, 26.0, "bilinear ↑", ha="center", va="bottom", fontsize=5.5, color=ORANGE)

    rounded_box(ax, 51.4, 12.0, 11.0, 15.0, face=WHITE, edge=LIGHT, linewidth=0.85)
    ax.text(
        56.9,
        25.75,
        "Swin block pair",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=NAVY,
        zorder=7,
    )
    steps = (
        ("LN + W-MSA", BLUE, PALE_BLUE),
        ("residual + LN + MLP", PURPLE, PALE_PURPLE),
        ("LN + SW-MSA", TEAL, PALE_TEAL),
        ("residual + LN + MLP", PURPLE, PALE_PURPLE),
    )
    y_positions = (22.6, 19.8, 17.0, 14.2)
    for (label, edge, face), yy in zip(steps, y_positions):
        rounded_box(ax, 52.5, yy, 8.8, 1.9, face=face, edge=edge, linewidth=0.75, radius=0.25)
        ax.text(
            56.9,
            yy + 0.95,
            label,
            ha="center",
            va="center",
            fontsize=5.9,
            fontweight="bold",
            color=edge,
            zorder=7,
        )
        if yy != y_positions[-1]:
            arrow(ax, (56.9, yy), (56.9, yy - 0.8), color=MID, mutation=7, linewidth=0.8)
    ax.text(
        56.9,
        12.70,
        "window 4×4 · shift 0/2\nMLP ratio 4 · drop-path 0→0.1",
        ha="center",
        va="center",
        fontsize=5.6,
        color=MID,
        linespacing=1.15,
        zorder=7,
    )


def branch_output(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    channels: str,
    edge: str,
    face: str,
) -> None:
    rounded_box(ax, x, y, width, height, face=face, edge=edge, linewidth=0.75, radius=0.25)
    ax.text(
        x + width / 2,
        y + height * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=5.9,
        fontweight="bold",
        color=edge,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + height * 0.25,
        channels,
        ha="center",
        va="center",
        fontsize=5.2,
        color=INK,
        zorder=7,
    )


def draw_decoder_detail(ax) -> None:
    panel(ax, 65.3, 8.2, 33.7, 23.5, "C  |  Dense CenterPoint decoder and Cartesian output")

    rounded_box(ax, 67.0, 26.3, 10.5, 2.7, face=PALE_PURPLE, edge=PURPLE, linewidth=0.9)
    ax.text(
        72.25,
        27.65,
        "fused feature  [B,128,128,54]",
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=PURPLE,
        zorder=7,
    )
    add_node(ax, 79.2, 27.65, "Y", face=WHITE, edge=NAVY, radius=0.58)
    arrow(ax, (77.6, 27.65), (78.55, 27.65), color=NAVY, mutation=8)

    draw_head_card(
        ax,
        67.0,
        20.5,
        12.0,
        4.3,
        title="classification",
        detail="3×3: 128→64  ·  1×1: 64→1",
        face=PALE_RED,
        edge=RED,
    )
    rounded_box(ax, 81.6, 20.5, 15.3, 4.3, face=WHITE, edge=RED, linewidth=0.85)
    ax.text(
        89.25,
        23.45,
        "Sedan center logits",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=RED,
        zorder=7,
    )
    ax.text(
        89.25,
        21.75,
        "[B,1,128,54]  ·  output bias −2.19",
        ha="center",
        va="center",
        fontsize=5.8,
        color=INK,
        zorder=7,
    )
    arrow(
        ax,
        (79.65, 27.25),
        (73.0, 24.9),
        color=RED,
        connection="arc3,rad=0.12",
        mutation=8,
    )
    arrow(ax, (79.1, 22.65), (81.5, 22.65), color=RED, mutation=8)

    draw_head_card(
        ax,
        67.0,
        13.2,
        12.0,
        5.2,
        title="box shared trunk",
        detail="3×3 Conv–BN–Act ×2\n128→64→64",
        face=PALE_GREEN,
        edge=GREEN,
    )
    arrow(
        ax,
        (79.65, 27.15),
        (73.0, 18.5),
        color=GREEN,
        connection="arc3,rad=0.18",
        mutation=8,
    )
    add_node(ax, 80.8, 15.8, "Y", face=WHITE, edge=GREEN, radius=0.55)
    arrow(ax, (79.1, 15.8), (80.2, 15.8), color=GREEN, mutation=8)

    branch_output(ax, 82.3, 16.1, 6.7, 2.7, title="center offset", channels="dx, dy · 2 ch", edge=GREEN, face=PALE_GREEN)
    branch_output(ax, 90.2, 16.1, 6.7, 2.7, title="center height", channels="dz · 1 ch", edge=GREEN, face=PALE_GREEN)
    branch_output(ax, 82.3, 12.7, 6.7, 2.7, title="box size", channels="l, w, h · 3 ch", edge=GREEN, face=PALE_GREEN)
    branch_output(ax, 90.2, 12.7, 6.7, 2.7, title="orientation", channels="sinθ, cosθ · 2 ch", edge=GREEN, face=PALE_GREEN)
    for endpoint in ((82.2, 17.45), (90.1, 17.45), (82.2, 14.05), (90.1, 14.05)):
        arrow(ax, (81.35, 15.8), endpoint, color=GREEN, mutation=6, linewidth=0.8)

    rounded_box(
        ax,
        67.0,
        9.2,
        29.9,
        2.5,
        face="#F5FAF8",
        edge=GREEN,
        linewidth=0.85,
        linestyle=(0, (4, 2)),
    )
    ax.text(
        81.95,
        10.45,
        "concat → box_reg [B,8,128,54]    ·    R–A cell + offsets → [x,y,z,l,w,h,θ]",
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=GREEN,
        zorder=7,
    )


def draw_footer(ax, source_note: str) -> None:
    rounded_box(
        ax,
        1.0,
        2.4,
        61.8,
        3.6,
        face=PALE_ORANGE,
        edge=ORANGE,
        linewidth=0.9,
        linestyle=(0, (4, 2)),
    )
    ax.text(
        2.0,
        4.80,
        "TRAINING OBJECTIVE  (not a model layer)",
        ha="left",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=ORANGE,
        zorder=7,
    )
    ax.text(
        2.0,
        3.45,
        "Gaussian-focal center heatmap  +  Smooth-L1 offset / height / size / yaw  +  2.0 × GWD    ·    heatmap radius = 3",
        ha="left",
        va="center",
        fontsize=6.4,
        color=INK,
        zorder=7,
    )

    rounded_box(ax, 64.2, 2.4, 34.8, 3.6, face=WHITE, edge=LIGHT, linewidth=0.9)
    ax.text(
        65.2,
        4.80,
        "WHAT IS NOT IN THIS ACTIVE GRAPH",
        ha="left",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=MID,
        zorder=7,
    )
    ax.text(
        65.2,
        3.45,
        "No CBAM · no deformable convolution · no object-query Transformer · no absolute positional embedding",
        ha="left",
        va="center",
        fontsize=6.1,
        color=INK,
        zorder=7,
    )

    ax.text(
        1.0,
        1.15,
        "Tensor order: [batch, channels, range, azimuth]  ·  full scope: R=256 (0–118.0 m), A=107 (−53°…+53°)",
        ha="left",
        va="center",
        fontsize=5.8,
        color=MID,
        zorder=7,
    )
    ax.text(
        99.0,
        1.15,
        f"6,620,419 registered parameters  ·  torchvision classifier tail bypassed  ·  inputs: {source_note}",
        ha="right",
        va="center",
        fontsize=5.8,
        color=MID,
        zorder=7,
    )


def build_figure(rad_map: np.ndarray, rae_map: np.ndarray, source_note: str):
    setup_style()
    fig, ax = plt.subplots(figsize=(17.2, 10.2))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    draw_header(ax)
    draw_overview(ax, rad_map, rae_map)
    draw_encoder_detail(ax)
    draw_decoder_detail(ax)
    draw_footer(ax, source_note)

    return fig


def draw_header(ax) -> None:
    ax.text(
        1.0,
        58.4,
        "Dual-view Swin–FPN CenterPoint radar detector",
        ha="left",
        va="center",
        fontsize=17.0,
        fontweight="bold",
        color=NAVY,
        zorder=10,
    )
    ax.text(
        1.0,
        56.55,
        "Exact model instantiated by train.py with the current train_cfg.py",
        ha="left",
        va="center",
        fontsize=8.6,
        color=MID,
        zorder=10,
    )

    x = 62.0
    for text, face, edge in (
        ("model7", PALE_BLUE, BLUE),
        ("Cartesian", PALE_TEAL, TEAL),
        ("CenterPoint", PALE_PURPLE, PURPLE),
        ("Sedan only", PALE_RED, RED),
        ("decoder 64", PALE_GREEN, GREEN),
    ):
        width = badge(ax, x, 57.3, text, face=face, edge=edge)
        x += width + 0.55


def build_overview_figure(
    rad_slices: tuple[np.ndarray, ...],
    rae_slices: tuple[np.ndarray, ...],
):
    setup_style()
    # Compact folded export: the upper streams end at concat, the graph drops
    # into fusion, then prediction proceeds right-to-left on the lower row.
    fig, ax = plt.subplots(figsize=(11.8, 9.0))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0.5, 73.5)
    ax.set_ylim(15.5, 71.5)
    ax.axis("off")
    draw_folded_overview(ax, rad_slices, rae_slices)
    return fig


def save_figure(
    fig,
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
    *,
    stem: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        path = out_dir / f"{stem}.{extension}"
        kwargs = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(path, **kwargs)
        written.append(path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--radar-root",
        type=Path,
        default=DEFAULT_RADAR_ROOT,
        help="RAD/RAE root used only for project-owned input thumbnails",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=("png", "svg", "pdf"),
    )
    parser.add_argument("--dpi", type=int, default=360, help="PNG resolution")
    parser.add_argument(
        "--skip-config-check",
        action="store_true",
        help="draw even if train_cfg.py no longer matches this architecture",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_config_check:
        validate_current_config()
    rad_map, rae_map, source_note = load_input_maps(args.radar_root)
    rad_slices, rae_slices, _ = load_input_slice_stacks(args.radar_root)
    fig = build_figure(rad_map, rae_map, source_note)
    written = save_figure(
        fig,
        args.out_dir,
        args.formats,
        args.dpi,
        stem=STEM,
    )
    plt.close(fig)
    overview_fig = build_overview_figure(rad_slices, rae_slices)
    written.extend(
        save_figure(
            overview_fig,
            args.out_dir,
            args.formats,
            args.dpi,
            stem=OVERVIEW_STEM,
        )
    )
    plt.close(overview_fig)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
