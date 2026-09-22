#!/usr/bin/env python3
"""Draw thesis-ready 2.5-D architecture figures for model7.

This generator is intentionally non-destructive: it writes two new figure
families and leaves the two-dimensional architecture exports untouched.
The first figure documents the currently implemented model7 graph.  The second
is an explicitly labelled research concept that adapts a query-based
Transformer decoder to the same dual-view radar inputs.

The shallow cuboid depth is a visual encoding of feature channels.  It does
not imply that the network uses 3-D convolutions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patches

# Reuse the existing exact-model figure as a Hilfsmittel for configuration
# validation, project-owned thumbnails, typography, palette, and export rules.
# Importing it does not draw or overwrite anything.
from .architecture import (
    BLUE,
    CYAN,
    DEFAULT_OUT_DIR,
    DEFAULT_RADAR_ROOT,
    GREEN,
    INK,
    LIGHT,
    MID,
    NAVY,
    ORANGE,
    PALE_BLUE,
    PALE_CYAN,
    PALE_GREEN,
    PALE_ORANGE,
    PALE_PURPLE,
    PALE_RED,
    PALE_TEAL,
    PANEL,
    PURPLE,
    RED,
    TEAL,
    WHITE,
    load_input_maps,
    save_figure,
    setup_style,
    validate_current_config,
)


CURRENT_STEM = "model7_swin_fpn_centerpoint_architecture_3d"
CONCEPT_STEM = "model7_transformer_decoder_concept_3d"


def tint(colour: str, amount: float) -> tuple[float, float, float]:
    """Mix *colour* toward white (positive) or black (negative)."""
    rgb = np.asarray(mcolors.to_rgb(colour), dtype=float)
    target = np.ones(3) if amount >= 0 else np.zeros(3)
    return tuple(rgb + (target - rgb) * abs(amount))


def channel_depth(channels: int) -> float:
    """A compact, monotone visual depth encoding for common channel counts."""
    return 0.75 + 2.9 * min(max(channels, 1), 256) / 256.0


def rounded_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = WHITE,
    edge: str = LIGHT,
    linewidth: float = 0.9,
    linestyle: str | tuple = "-",
    radius: float = 0.7,
    zorder: float = 1.0,
):
    item = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.16,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(item)
    return item


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    linewidth: float = 1.15,
    linestyle: str | tuple = "-",
    connection: str = "arc3",
    mutation: float = 9.0,
    zorder: float = 5.0,
):
    item = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=connection,
        shrinkA=2.0,
        shrinkB=2.0,
        zorder=zorder,
    )
    ax.add_patch(item)
    return item


def cuboid(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    depth: float,
    face: str,
    edge: str,
    title: str = "",
    detail: str = "",
    title_size: float = 7.0,
    detail_size: float = 5.5,
    linewidth: float = 0.85,
    alpha: float = 1.0,
    zorder: float = 3.0,
):
    """Draw a shallow vector cuboid with a frontal label."""
    dx = depth
    dy = 0.54 * depth
    top = patches.Polygon(
        [(x, y + height), (x + dx, y + height + dy),
         (x + width + dx, y + height + dy), (x + width, y + height)],
        closed=True,
        facecolor=tint(face, 0.34),
        edgecolor=edge,
        linewidth=linewidth,
        alpha=alpha,
        joinstyle="round",
        zorder=zorder,
    )
    side = patches.Polygon(
        [(x + width, y), (x + width + dx, y + dy),
         (x + width + dx, y + height + dy), (x + width, y + height)],
        closed=True,
        facecolor=tint(face, -0.13),
        edgecolor=edge,
        linewidth=linewidth,
        alpha=alpha,
        joinstyle="round",
        zorder=zorder + 0.1,
    )
    front = patches.Rectangle(
        (x, y),
        width,
        height,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        alpha=alpha,
        joinstyle="round",
        zorder=zorder + 0.2,
    )
    ax.add_patch(top)
    ax.add_patch(side)
    ax.add_patch(front)
    if title:
        ax.text(
            x + width / 2,
            y + height * (0.61 if detail else 0.5),
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=edge,
            linespacing=1.0,
            zorder=zorder + 1.0,
        )
    if detail:
        ax.text(
            x + width / 2,
            y + height * 0.25,
            detail,
            ha="center",
            va="center",
            fontsize=detail_size,
            color=INK,
            linespacing=1.05,
            zorder=zorder + 1.0,
        )
    return front


def image_cuboid(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    image: np.ndarray,
    depth: float,
    cmap: str,
    edge: str,
    title: str,
    subtitle: str,
    shape: str,
) -> None:
    """Draw a radar-map front plane with vector cuboid sides and live labels."""
    cuboid(
        ax,
        x,
        y,
        width,
        height,
        depth=depth,
        face=WHITE,
        edge=edge,
        linewidth=1.0,
    )
    inset = 0.45
    ax.imshow(
        image,
        extent=(x + inset, x + width - inset, y + 1.55, y + height - 2.35),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bilinear",
        zorder=4.0,
    )
    ax.add_patch(
        patches.Rectangle(
            (x + inset, y + 1.55),
            width - 2 * inset,
            height - 3.90,
            fill=False,
            edgecolor=edge,
            linewidth=0.55,
            zorder=5.0,
        )
    )
    ax.text(
        x + 0.48,
        y + height - 0.72,
        title,
        ha="left",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color=edge,
        zorder=7,
    )
    ax.text(
        x + 0.48,
        y + height - 1.53,
        subtitle,
        ha="left",
        va="center",
        fontsize=4.8,
        color=MID,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + 0.70,
        shape,
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=INK,
        zorder=7,
    )


def pill(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    face: str,
    edge: str,
    width: float | None = None,
    fontsize: float = 6.6,
) -> float:
    width = width if width is not None else 2.2 + 0.50 * len(text)
    rounded_box(
        ax,
        x,
        y,
        width,
        2.25,
        face=face,
        edge=edge,
        linewidth=0.75,
        radius=1.05,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + 1.12,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=edge,
        zorder=8,
    )
    return width


def section_label(ax, x: float, y: float, number: str, text: str) -> None:
    ax.add_patch(
        patches.Circle(
            (x, y),
            1.30,
            facecolor=NAVY,
            edgecolor=NAVY,
            linewidth=0.8,
            zorder=8,
        )
    )
    ax.text(
        x,
        y,
        number,
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=WHITE,
        zorder=9,
    )
    ax.text(
        x + 2.0,
        y,
        text.upper(),
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=NAVY,
        zorder=9,
    )


def encoder_lane(
    ax,
    y: float,
    *,
    name: str,
    accent: str,
    pale: str,
) -> None:
    """Draw the exact three-stage Swin hierarchy and P1 FPN output."""
    rounded_box(
        ax,
        21.5,
        y - 2.4,
        68.8,
        16.8,
        face=pale,
        edge=tint(accent, 0.30),
        linewidth=0.9,
        linestyle=(0, (4, 2)),
        radius=1.0,
        zorder=0.5,
    )
    ax.text(
        23.0,
        y + 13.1,
        f"{name} Swin–FPN encoder",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=accent,
        zorder=7,
    )
    ax.text(
        88.5,
        y + 13.1,
        "independent parameters",
        ha="right",
        va="center",
        fontsize=5.7,
        fontstyle="italic",
        color=MID,
        zorder=7,
    )

    stages = (
        (25.0, y, 9.4, 9.7, 64, "S1 · Swin ×2", "64 · 128×54"),
        (41.4, y + 0.9, 8.6, 8.0, 128, "S2 · Swin ×2", "128 · 64×27"),
        (57.0, y + 1.7, 7.7, 6.4, 256, "S3 · Swin ×2", "256 · 32×14"),
        (77.3, y + 0.2, 10.3, 9.3, 128, "P1 · FPN", "128 · 128×54"),
    )
    for x, sy, width, height, channels, title, detail in stages:
        cuboid(
            ax,
            x,
            sy,
            width,
            height,
            depth=channel_depth(channels),
            face=WHITE,
            edge=accent,
            title=title,
            detail=detail,
            title_size=6.6,
            detail_size=5.2,
        )

    arrow(ax, (19.6, y + 4.9), (24.7, y + 4.9), color=accent)
    ax.text(
        22.0,
        y - 1.15,
        "2×2 patch /2",
        ha="center",
        va="center",
        fontsize=4.9,
        color=MID,
        zorder=8,
    )
    arrow(ax, (37.0, y + 4.9), (41.1, y + 4.9), color=accent)
    arrow(ax, (53.3, y + 4.9), (56.7, y + 4.9), color=accent)
    ax.text(45.6, y - 1.15, "patch merge /2", ha="center", va="center", fontsize=4.8, color=MID)
    ax.text(60.7, y - 1.15, "patch merge /2", ha="center", va="center", fontsize=4.8, color=MID)

    # Three lateral taps feed a compact aggregation bus above the feature
    # blocks.  This makes the FPN relation explicit without crossing labels.
    bus_y = y + 10.65
    tap_points = ((34.0, y + 10.40), (49.6, y + 10.05), (64.4, y + 10.02))
    ax.plot([34.0, 69.0], [bus_y, bus_y], color=accent, linewidth=0.75, zorder=5)
    for tap_x, tap_y in tap_points:
        ax.plot([tap_x, tap_x], [tap_y, bus_y], color=accent, linewidth=0.75, zorder=5)
        ax.add_patch(
            patches.Circle(
                (tap_x, bus_y),
                0.16,
                facecolor=accent,
                edgecolor=accent,
                linewidth=0.4,
                zorder=6,
            )
        )
    arrow(
        ax,
        (69.0, bus_y),
        (77.0, y + 8.7),
        color=accent,
        linewidth=0.85,
        mutation=7.0,
    )
    ax.text(
        55.5,
        y + 11.55,
        "FPN: 1×1 lateral · bilinear top-down · 3×3 smooth + 2× refine",
        ha="center",
        va="center",
        fontsize=4.8,
        color=accent,
        zorder=8,
    )


def wireframe_box(
    ax,
    project,
    *,
    u: float,
    v: float,
    length: float,
    width: float,
    height: float,
    colour: str,
    linewidth: float = 1.0,
) -> None:
    corners = [
        (u, v, 0),
        (u + length, v, 0),
        (u + length, v + width, 0),
        (u, v + width, 0),
        (u, v, height),
        (u + length, v, height),
        (u + length, v + width, height),
        (u, v + width, height),
    ]
    points = [project(*corner) for corner in corners]
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    ax.add_patch(
        patches.Polygon(
            [points[index] for index in (4, 5, 6, 7)],
            closed=True,
            facecolor=mcolors.to_rgba(colour, 0.12),
            edgecolor="none",
            zorder=8,
        )
    )
    for start, end in edges:
        ax.plot(
            [points[start][0], points[end][0]],
            [points[start][1], points[end][1]],
            color=colour,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=9,
        )


def bev_output(ax, x: float, y: float, *, concept: bool = False) -> None:
    """Draw an isometric BEV plane with genuine wireframe 3-D boxes."""
    accent = PURPLE if concept else GREEN
    pale = PALE_PURPLE if concept else PALE_GREEN
    rounded_box(ax, x, y, 19.0, 34.5, face=pale, edge=accent, linewidth=1.0, radius=1.0)
    ax.text(
        x + 9.5,
        y + 31.7,
        "Cartesian 3-D detections",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=accent,
        zorder=10,
    )
    ax.text(
        x + 9.5,
        y + 29.6,
        "[x, y, z, l, w, h, yaw]",
        ha="center",
        va="center",
        fontsize=5.5,
        color=INK,
        zorder=10,
    )

    origin = np.asarray((x + 9.0, y + 5.0), dtype=float)
    basis_u = np.asarray((0.52, 0.22))
    basis_v = np.asarray((-0.43, 0.31))
    basis_z = np.asarray((0.0, 0.80))

    def project(u: float, v: float, z: float = 0.0) -> tuple[float, float]:
        point = origin + u * basis_u + v * basis_v + z * basis_z
        return float(point[0]), float(point[1])

    plane = [project(0, 0), project(17, 0), project(17, 18), project(0, 18)]
    ax.add_patch(
        patches.Polygon(
            plane,
            closed=True,
            facecolor=WHITE,
            edgecolor=accent,
            linewidth=0.9,
            zorder=3,
        )
    )
    for value in (4.25, 8.5, 12.75):
        a, b = project(value, 0), project(value, 18)
        c, d = project(0, value), project(17, value)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=LIGHT, linewidth=0.45, zorder=4)
        ax.plot([c[0], d[0]], [c[1], d[1]], color=LIGHT, linewidth=0.45, zorder=4)

    # Ego footprint and three detected objects.
    ego = [project(7.2, 0.8), project(10.0, 0.8), project(10.0, 5.1), project(7.2, 5.1)]
    ax.add_patch(
        patches.Polygon(
            ego,
            closed=True,
            facecolor=NAVY,
            edgecolor=NAVY,
            linewidth=0.6,
            zorder=6,
        )
    )
    ax.text(*project(8.6, 2.9), "ego", ha="center", va="center", fontsize=4.6, color=WHITE, zorder=7)
    wireframe_box(ax, project, u=2.0, v=9.2, length=4.0, width=2.4, height=2.0, colour=ORANGE)
    wireframe_box(ax, project, u=10.0, v=11.5, length=3.8, width=2.2, height=1.8, colour=accent)
    wireframe_box(ax, project, u=4.5, v=15.0, length=3.0, width=1.9, height=1.6, colour=RED)
    ax.text(x + 9.5, y + 1.65, "metric BEV plane + box height", ha="center", va="center", fontsize=5.0, color=MID)


def depth_legend(ax, x: float, y: float, *, width: float = 60.0) -> None:
    rounded_box(ax, x, y, width, 8.0, face=WHITE, edge=LIGHT, linewidth=0.8, radius=0.8)
    ax.text(
        x + 1.3,
        y + 6.25,
        "2.5-D drawing convention",
        ha="left",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=NAVY,
        zorder=8,
    )
    ax.text(
        x + 1.3,
        y + 3.45,
        "front face = spatial feature map",
        ha="left",
        va="center",
        fontsize=5.5,
        color=INK,
        zorder=8,
    )
    ax.text(
        x + 1.3,
        y + 1.55,
        "depth = channels · perspective is illustrative · no 3-D convolution",
        ha="left",
        va="center",
        fontsize=5.1,
        color=MID,
        fontstyle="italic",
        zorder=8,
    )
    icon_x = x + width - 20.0
    for offset, channels in ((0.0, 64), (6.1, 128), (12.3, 256)):
        cuboid(
            ax,
            icon_x + offset,
            y + 2.70,
            3.0,
            2.0,
            depth=0.36 * channel_depth(channels),
            face=PALE_BLUE,
            edge=BLUE,
            linewidth=0.55,
            zorder=4,
        )
        ax.text(
            icon_x + offset + 1.5,
            y + 1.15,
            str(channels),
            ha="center",
            va="center",
            fontsize=4.5,
            color=BLUE,
            zorder=8,
        )


def figure_header(ax, title: str, subtitle: str, *, concept: bool) -> None:
    ax.text(
        2.0,
        81.2,
        title,
        ha="left",
        va="center",
        fontsize=16.0,
        fontweight="bold",
        color=NAVY,
        zorder=20,
    )
    ax.text(
        2.1,
        77.9,
        subtitle,
        ha="left",
        va="center",
        fontsize=7.7,
        color=MID,
        zorder=20,
    )
    if concept:
        pill(
            ax,
            137.0,
            79.4,
            "PROPOSED CONCEPT · NOT CURRENT IMPLEMENTATION",
            face=PALE_RED,
            edge=RED,
            width=40.0,
            fontsize=6.5,
        )
    else:
        cursor = 137.0
        for text, face, edge, width in (
            ("model7", PALE_BLUE, BLUE, 8.3),
            ("Cartesian", PALE_TEAL, TEAL, 10.4),
            ("CenterPoint", PALE_PURPLE, PURPLE, 12.5),
            ("Sedan", PALE_ORANGE, ORANGE, 8.5),
        ):
            pill(ax, cursor, 79.4, text, face=face, edge=edge, width=width)
            cursor += width + 0.7
    ax.plot([2.0, 178.0], [75.6, 75.6], color=LIGHT, linewidth=0.9, zorder=1)


def build_current_figure(rad_map: np.ndarray, rae_map: np.ndarray, source_note: str):
    setup_style()
    fig, ax = plt.subplots(figsize=(18.0, 8.4))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 84)
    ax.axis("off")
    figure_header(
        ax,
        "Dual-view Swin–FPN CenterPoint radar detector",
        "Implemented graph selected by train.py / train_cfg.py · exact tensor shapes · independent RAD and RAE encoders",
        concept=False,
    )

    section_label(ax, 3.2, 71.5, "1", "dual-view input")
    section_label(ax, 23.0, 71.5, "2", "hierarchical encoders + FPN")
    section_label(ax, 93.2, 71.5, "3", "fusion")
    section_label(ax, 111.5, 71.5, "4", "dense heads")
    section_label(ax, 147.8, 71.5, "5", "decode")
    section_label(ax, 161.7, 71.5, "6", "output")

    image_cuboid(
        ax,
        2.7,
        51.3,
        14.0,
        12.0,
        image=rad_map,
        depth=channel_depth(64),
        cmap="magma",
        edge=BLUE,
        title="RAD cube",
        subtitle="range–azimuth–Doppler",
        shape="[B, 64, 256, 107]",
    )
    image_cuboid(
        ax,
        2.7,
        26.4,
        14.0,
        12.0,
        image=rae_map,
        depth=channel_depth(37),
        cmap="viridis",
        edge=TEAL,
        title="RAE cube",
        subtitle="range–azimuth–elevation",
        shape="[B, 37, 256, 107]",
    )
    encoder_lane(ax, 51.7, name="RAD", accent=BLUE, pale=PALE_BLUE)
    encoder_lane(ax, 26.8, name="RAE", accent=TEAL, pale=PALE_TEAL)

    # The two P1 feature maps converge only after their non-shared encoders.
    arrow(ax, (90.1, 56.5), (94.0, 49.7), color=BLUE, connection="arc3,rad=0.10")
    arrow(ax, (90.1, 31.6), (94.0, 42.3), color=TEAL, connection="arc3,rad=-0.10")
    cuboid(
        ax,
        94.4,
        37.0,
        11.8,
        18.0,
        depth=channel_depth(128),
        face=PALE_CYAN,
        edge=CYAN,
        title="RAD / RAE\nfeature fusion",
        detail="concat: 256 ch\n1×1 Conv–BN–Act → 128\nresidual refinement",
        title_size=7.1,
        detail_size=5.2,
    )
    ax.text(
        100.3,
        34.6,
        "fused [B,128,128,54]",
        ha="center",
        va="center",
        fontsize=5.7,
        fontweight="bold",
        color=CYAN,
    )

    # Classification path.
    arrow(ax, (108.6, 48.0), (111.3, 56.0), color=ORANGE, connection="arc3,rad=-0.08")
    cuboid(
        ax,
        111.6,
        51.0,
        13.8,
        10.0,
        depth=channel_depth(64),
        face=PALE_ORANGE,
        edge=ORANGE,
        title="Sedan heatmap",
        detail="3×3 Conv–BN–Act\n128→64 · 1×1 → 1",
        title_size=6.8,
        detail_size=5.0,
    )
    cuboid(
        ax,
        130.0,
        52.1,
        8.3,
        7.8,
        depth=channel_depth(1),
        face=WHITE,
        edge=ORANGE,
        title="logits",
        detail="[B,1,128,54]",
        title_size=6.2,
        detail_size=4.8,
    )
    arrow(ax, (127.6, 56.0), (129.7, 56.0), color=ORANGE, mutation=7.0)

    # Regression path with an explicit four-way parameter decomposition.
    arrow(ax, (108.6, 44.3), (111.3, 33.6), color=GREEN, connection="arc3,rad=0.08")
    cuboid(
        ax,
        111.6,
        28.2,
        13.8,
        10.0,
        depth=channel_depth(64),
        face=PALE_GREEN,
        edge=GREEN,
        title="box shared trunk",
        detail="3×3 Conv–BN–Act ×2\n128→64→64",
        title_size=6.6,
        detail_size=5.0,
    )
    branch_specs = (
        (129.4, 35.0, "offset", "dx,dy · 2"),
        (138.2, 35.0, "height", "dz · 1"),
        (129.4, 27.4, "size", "l,w,h · 3"),
        (138.2, 27.4, "yaw", "sin,cos · 2"),
    )
    for bx, by, title, detail in branch_specs:
        cuboid(
            ax,
            bx,
            by,
            6.8,
            5.2,
            depth=0.9,
            face=WHITE,
            edge=GREEN,
            title=title,
            detail=detail,
            title_size=5.5,
            detail_size=4.4,
            linewidth=0.7,
        )
        arrow(ax, (127.8, 33.2), (bx - 0.2, by + 2.6), color=GREEN, linewidth=0.7, mutation=6.0)
    ax.text(
        136.9,
        25.1,
        "concat → [B,8,128,54]",
        ha="center",
        va="center",
        fontsize=5.3,
        fontweight="bold",
        color=GREEN,
    )

    # Algorithmic post-processing is intentionally a flat card rather than a
    # feature cuboid: it is not a learned tensor stage.
    rounded_box(ax, 146.8, 35.1, 11.4, 20.8, face=PANEL, edge=NAVY, linewidth=1.0, radius=0.9)
    ax.text(152.5, 52.4, "CenterPoint\ndecode", ha="center", va="center", fontsize=7.0, fontweight="bold", color=NAVY)
    ax.plot([148.1, 156.9], [48.7, 48.7], color=LIGHT, linewidth=0.65)
    ax.text(
        152.5,
        43.8,
        "sigmoid + local peaks\ntop-K centers\nR–A cell + offsets\n→ metric Cartesian boxes",
        ha="center",
        va="center",
        fontsize=5.2,
        color=INK,
        linespacing=1.25,
    )
    pill(ax, 148.8, 36.3, "rotated NMS", face=PALE_RED, edge=RED, width=7.5, fontsize=5.0)
    arrow(ax, (140.0, 56.0), (146.5, 50.0), color=ORANGE, connection="arc3,rad=0.12")
    arrow(ax, (146.0, 34.1), (146.5, 41.2), color=GREEN, connection="arc3,rad=-0.10")

    bev_output(ax, 160.4, 25.4, concept=False)
    arrow(ax, (158.3, 45.4), (160.1, 45.4), color=NAVY, mutation=7.0)

    depth_legend(ax, 2.1, 3.8, width=72.0)
    rounded_box(ax, 76.0, 3.8, 102.0, 8.0, face=PANEL, edge=LIGHT, linewidth=0.8, radius=0.8)
    ax.text(77.5, 9.95, "Implemented-model notes", ha="left", va="center", fontsize=6.5, fontweight="bold", color=NAVY)
    ax.text(
        77.5,
        7.2,
        "The two Swin–FPN modules do not share weights. Dense prediction is evaluated at stride 2 (128×54).",
        ha="left",
        va="center",
        fontsize=5.5,
        color=INK,
    )
    ax.text(
        77.5,
        5.15,
        "This active graph contains no object-query Transformer decoder; the separate concept figure shows that possible redesign.",
        ha="left",
        va="center",
        fontsize=5.3,
        color=MID,
        fontstyle="italic",
    )
    ax.text(
        178.0,
        1.65,
        f"Inputs: {source_note} · generated from vector primitives; raster used only on input-map faces",
        ha="right",
        va="center",
        fontsize=4.9,
        color=MID,
    )
    return fig


def query_stack(ax, x: float, y: float) -> None:
    for index in range(3, -1, -1):
        offset = index * 0.62
        ax.add_patch(
            patches.Rectangle(
                (x + offset, y + offset * 0.45),
                9.6,
                4.2,
                facecolor=tint(PALE_ORANGE, 0.06 * index),
                edgecolor=ORANGE,
                linewidth=0.65,
                zorder=4 + index * 0.01,
            )
        )
    ax.text(x + 4.8, y + 2.1, "learned\nobject queries", ha="center", va="center", fontsize=5.0, fontweight="bold", color=ORANGE, linespacing=0.95, zorder=8)
    ax.text(x + 4.8, y - 1.15, "N × d query embeddings", ha="center", va="center", fontsize=4.8, color=MID, zorder=8)


def pyramid_lane(
    ax,
    y: float,
    *,
    name: str,
    accent: str,
    pale: str,
) -> None:
    rounded_box(
        ax,
        21.0,
        y - 2.2,
        51.0,
        16.5,
        face=pale,
        edge=tint(accent, 0.30),
        linewidth=0.9,
        linestyle=(0, (4, 2)),
        radius=1.0,
        zorder=0.5,
    )
    ax.text(22.5, y + 12.1, f"{name} hierarchical Transformer pyramid", ha="left", va="center", fontsize=6.9, fontweight="bold", color=accent)
    ax.text(70.5, y + 12.1, "separate view encoder", ha="right", va="center", fontsize=5.4, fontstyle="italic", color=MID)
    specs = (
        (25.0, y, 10.0, 9.8, 64, "F1 · fine", "64 · 128×54"),
        (42.0, y + 1.0, 8.8, 7.9, 128, "F2 · mid", "128 · 64×27"),
        (58.0, y + 1.9, 7.5, 6.0, 256, "F3 · coarse", "256 · 32×14"),
    )
    for x, sy, width, height, channels, title, detail in specs:
        cuboid(
            ax,
            x,
            sy,
            width,
            height,
            depth=channel_depth(channels),
            face=WHITE,
            edge=accent,
            title=title,
            detail=detail,
            title_size=6.4,
            detail_size=5.0,
        )
    arrow(ax, (19.4, y + 4.9), (24.7, y + 4.9), color=accent)
    arrow(ax, (37.6, y + 4.9), (41.7, y + 4.9), color=accent)
    arrow(ax, (54.3, y + 4.9), (57.7, y + 4.9), color=accent)


def build_concept_figure(rad_map: np.ndarray, rae_map: np.ndarray, source_note: str):
    setup_style()
    fig, ax = plt.subplots(figsize=(18.0, 8.4))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 84)
    ax.axis("off")
    figure_header(
        ax,
        "Dual-view radar detector with a conditional Transformer decoder",
        "Architecture study inspired by pyramid-token fusion and RADE-style radar diagrams · conceptual only",
        concept=True,
    )

    section_label(ax, 3.2, 71.5, "1", "dual-view input")
    section_label(ax, 22.6, 71.5, "2", "feature pyramids")
    section_label(ax, 76.6, 71.5, "3", "pyramid token fusion")
    section_label(ax, 110.2, 71.5, "4", "queries")
    section_label(ax, 125.0, 71.5, "5", "decoder ×3")
    section_label(ax, 149.0, 71.5, "6", "set heads")
    section_label(ax, 163.0, 71.5, "7", "output")

    image_cuboid(
        ax,
        2.6,
        51.2,
        13.6,
        11.8,
        image=rad_map,
        depth=channel_depth(64),
        cmap="magma",
        edge=BLUE,
        title="RAD cube",
        subtitle="R–A–D",
        shape="[B, 64, 256, 107]",
    )
    image_cuboid(
        ax,
        2.6,
        26.4,
        13.6,
        11.8,
        image=rae_map,
        depth=channel_depth(37),
        cmap="viridis",
        edge=TEAL,
        title="RAE cube",
        subtitle="R–A–E",
        shape="[B, 37, 256, 107]",
    )
    pyramid_lane(ax, 51.6, name="RAD", accent=BLUE, pale=PALE_BLUE)
    pyramid_lane(ax, 26.8, name="RAE", accent=TEAL, pale=PALE_TEAL)

    rounded_box(
        ax,
        75.5,
        24.1,
        29.8,
        42.7,
        face=PANEL,
        edge=PURPLE,
        linewidth=1.05,
        linestyle=(0, (4, 2)),
        radius=1.0,
        zorder=0.8,
    )
    ax.text(90.4, 63.8, "Pyramid Token Fusion (PTF)", ha="center", va="center", fontsize=7.4, fontweight="bold", color=PURPLE)
    ax.text(
        90.4,
        61.5,
        "six inputs: {RAD, RAE} × {F1, F2, F3}",
        ha="center",
        va="center",
        fontsize=5.1,
        color=MID,
    )
    cuboid(
        ax,
        78.2,
        48.2,
        9.0,
        8.5,
        depth=1.25,
        face=PALE_PURPLE,
        edge=PURPLE,
        title="project",
        detail="1×1 → d\nper level/view",
        title_size=6.1,
        detail_size=4.8,
    )
    cuboid(
        ax,
        90.3,
        48.2,
        10.1,
        8.5,
        depth=1.25,
        face=PALE_PURPLE,
        edge=PURPLE,
        title="flatten + encode",
        detail="2-D position\nlevel + view ID",
        title_size=5.8,
        detail_size=4.7,
    )
    arrow(ax, (88.5, 52.4), (90.0, 52.4), color=PURPLE, mutation=7.0)
    cuboid(
        ax,
        82.2,
        31.4,
        14.7,
        9.7,
        depth=2.5,
        face=WHITE,
        edge=PURPLE,
        title="unified token memory",
        detail="concat {RAD, RAE}\n× {F1, F2, F3}",
        title_size=6.2,
        detail_size=5.0,
    )
    arrow(ax, (95.4, 48.0), (92.0, 41.5), color=PURPLE, connection="arc3,rad=0.10")
    # Both view pyramids route their three scales into PTF.
    for start, end, colour, rad in (
        ((69.2, 58.0), (77.9, 53.3), BLUE, 0.08),
        ((69.2, 33.2), (77.9, 51.0), TEAL, -0.12),
    ):
        arrow(ax, start, end, color=colour, connection=f"arc3,rad={rad}")

    query_stack(ax, 109.3, 53.7)
    arrow(ax, (117.8, 55.2), (124.3, 52.0), color=ORANGE, connection="arc3,rad=0.08")
    cuboid(
        ax,
        109.5,
        31.8,
        9.2,
        8.4,
        depth=2.0,
        face=PALE_PURPLE,
        edge=PURPLE,
        title="memory",
        detail="M × d tokens",
        title_size=6.2,
        detail_size=4.9,
    )
    arrow(ax, (100.5, 36.2), (109.2, 36.2), color=PURPLE)

    # Three offset plates make the repeated conditional decoder explicit.
    for offset, alpha in ((2.2, 0.42), (1.1, 0.70)):
        cuboid(
            ax,
            124.5 + offset,
            36.0 + offset * 0.32,
            16.3,
            19.0,
            depth=1.1,
            face=PALE_CYAN,
            edge=CYAN,
            alpha=alpha,
            linewidth=0.65,
        )
    cuboid(
        ax,
        124.5,
        36.0,
        16.3,
        19.0,
        depth=1.5,
        face=PALE_CYAN,
        edge=CYAN,
        title="Conditional\nTransformer decoder",
        detail="query self-attention\ncross-attention to memory\nFFN + iterative refinement",
        title_size=6.7,
        detail_size=4.9,
        linewidth=1.0,
        zorder=5,
    )
    pill(ax, 129.0, 37.2, "×3 layers", face=WHITE, edge=CYAN, width=7.3, fontsize=5.1)
    arrow(ax, (120.8, 36.1), (124.2, 42.3), color=PURPLE, connection="arc3,rad=-0.10")

    cuboid(
        ax,
        146.0,
        51.0,
        10.5,
        9.0,
        depth=1.15,
        face=PALE_ORANGE,
        edge=ORANGE,
        title="class head",
        detail="object / no-object\nper query",
        title_size=6.3,
        detail_size=4.8,
    )
    cuboid(
        ax,
        146.0,
        31.5,
        10.5,
        11.0,
        depth=1.15,
        face=PALE_GREEN,
        edge=GREEN,
        title="3-D box head",
        detail="x,y,z,l,w,h,yaw\nper query",
        title_size=6.3,
        detail_size=4.8,
    )
    arrow(ax, (143.3, 49.2), (145.7, 55.3), color=ORANGE, connection="arc3,rad=-0.08")
    arrow(ax, (143.3, 43.0), (145.7, 37.0), color=GREEN, connection="arc3,rad=0.08")
    pill(
        ax,
        143.7,
        25.8,
        "Hungarian set loss · no NMS",
        face=PALE_RED,
        edge=RED,
        width=15.6,
        fontsize=4.8,
    )

    bev_output(ax, 160.6, 25.4, concept=True)
    arrow(ax, (157.8, 55.0), (160.2, 46.2), color=ORANGE, connection="arc3,rad=0.12", mutation=7.0)
    arrow(ax, (157.8, 37.0), (160.2, 43.8), color=GREEN, connection="arc3,rad=-0.12", mutation=7.0)

    depth_legend(ax, 2.1, 3.8, width=68.0)
    rounded_box(ax, 72.0, 3.8, 106.0, 8.0, face=PALE_RED, edge=tint(RED, 0.35), linewidth=0.8, radius=0.8)
    ax.text(73.5, 9.95, "Concept boundary and provenance", ha="left", va="center", fontsize=6.5, fontweight="bold", color=RED)
    ax.text(
        73.5,
        7.15,
        "Proposed adaptation: Zhang et al. (2026), pyramid-token fusion + conditional Transformer decoder; RADE-Net-inspired visual language.",
        ha="left",
        va="center",
        fontsize=5.35,
        color=INK,
    )
    ax.text(
        73.5,
        5.10,
        "This is not the current model7 implementation. One-to-one Hungarian assignment is assumed; token count and embedding width remain design choices.",
        ha="left",
        va="center",
        fontsize=5.15,
        color=RED,
        fontstyle="italic",
    )
    ax.text(
        178.0,
        1.65,
        f"Inputs: {source_note} · references: arXiv:2601.13386 and arXiv:2602.19994",
        ha="right",
        va="center",
        fontsize=4.9,
        color=MID,
    )
    return fig


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
        help="draw even if train_cfg.py no longer matches the exact current-model figure",
    )
    return parser.parse_args()


def export_all(
    figures: Iterable[tuple[object, str]],
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    written: list[Path] = []
    for fig, stem in figures:
        written.extend(save_figure(fig, out_dir, formats, dpi, stem=stem))
        plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    if not args.skip_config_check:
        validate_current_config()
    rad_map, rae_map, source_note = load_input_maps(args.radar_root)
    current = build_current_figure(rad_map, rae_map, source_note)
    concept = build_concept_figure(rad_map, rae_map, source_note)
    written = export_all(
        ((current, CURRENT_STEM), (concept, CONCEPT_STEM)),
        args.out_dir,
        args.formats,
        args.dpi,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
