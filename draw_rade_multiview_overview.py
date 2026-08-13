#!/usr/bin/env python3
"""Draw the RADE-to-dual-view preprocessing and active Model7 overview.

The figure separates two parts that live at different points in the data
pipeline:

* upstream projection of a conceptual [R, A, D, E] radar tensor into RAD and
  RAE tensors; and
* the implemented Model7 path, which loads those precomputed views, encodes
  them independently, concatenates the encoded features, and performs dense
  Cartesian CenterPoint detection.

Only a new output family under ``figures/multiview_rade_overview`` is written.
Existing architecture figures are imported as drawing aids and are never
modified.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mvrss_rade_multiview_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from draw_model7_architecture import (
    BLUE,
    CYAN,
    DEFAULT_RADAR_ROOT,
    GOLD,
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
    ROOT,
    TEAL,
    WHITE,
    load_input_maps,
    save_figure,
    setup_style,
    validate_current_config,
)
from draw_model7_architecture_3d import (
    arrow,
    bev_output,
    channel_depth,
    cuboid,
    image_cuboid,
    pill,
    rounded_box,
    section_label,
)


OUT_DIR = ROOT / "figures" / "multiview_rade_overview"
STEM = "rade_to_rad_rae_model7_processing_overview"


def draw_header(ax) -> None:
    ax.text(
        2.2,
        88.6,
        "RADE-to-dual-view processing and Model7 detection overview",
        ha="left",
        va="center",
        fontsize=15.8,
        fontweight="bold",
        color=NAVY,
        zorder=30,
    )
    ax.text(
        2.3,
        85.25,
        "Upstream tensor projection  →  independent Swin–FPN encoders  →  encoded-feature fusion  →  dense Cartesian detection",
        ha="left",
        va="center",
        fontsize=7.7,
        color=MID,
        zorder=30,
    )
    cursor = 169.5
    for text, face, edge, width in (
        ("model7", PALE_BLUE, BLUE, 8.0),
        ("precomputed views", PALE_TEAL, TEAL, 15.5),
        ("CenterPoint", PALE_PURPLE, PURPLE, 12.2),
        ("3-D boxes", PALE_ORANGE, ORANGE, 10.0),
    ):
        pill(ax, cursor, 87.0, text, face=face, edge=edge, width=width, fontsize=5.9)
        cursor += width + 0.7
    ax.plot([2.2, 218.0], [82.3, 82.3], color=LIGHT, linewidth=0.9, zorder=1)


def draw_rade_tensor(ax, x: float, y: float) -> None:
    """Represent four axes using an RA-front/D-depth cube stacked over E."""
    width, height = 13.7, 18.5
    # Farther elevation slices are drawn first; the front slice remains crisp.
    for index in reversed(range(5)):
        # Elevation slices rise slightly to the left, whereas the cuboid's
        # Doppler depth recedes to the right. Keeping those directions
        # distinct prevents the two hidden dimensions from being conflated.
        offset_x = -0.28 * index
        offset_y = 0.78 * index
        cuboid(
            ax,
            x + offset_x,
            y + offset_y,
            width,
            height,
            depth=2.7,
            face=PALE_PURPLE,
            edge=PURPLE if index == 0 else GOLD,
            linewidth=1.0 if index == 0 else 0.60,
            alpha=1.0 if index == 0 else 0.55,
            zorder=3.0 + 0.05 * (4 - index),
        )

    # A restrained voxel grid makes the R-A front plane legible without
    # pretending that a true four-dimensional object can be drawn in 3-D.
    for fraction in (0.25, 0.50, 0.75):
        ax.plot(
            [x + fraction * width, x + fraction * width],
            [y, y + height],
            color=PURPLE,
            linewidth=0.35,
            alpha=0.34,
            zorder=6,
        )
    for fraction in (0.20, 0.40, 0.60, 0.80):
        ax.plot(
            [x, x + width],
            [y + fraction * height, y + fraction * height],
            color=PURPLE,
            linewidth=0.35,
            alpha=0.34,
            zorder=6,
        )
    for cx, cy, size, colour in (
        (x + 3.0, y + 4.0, 1.5, ORANGE),
        (x + 8.8, y + 8.9, 1.2, TEAL),
        (x + 5.8, y + 13.8, 1.0, RED),
    ):
        ax.add_patch(
            patches.Rectangle(
                (cx, cy),
                size,
                0.62 * size,
                facecolor=colour,
                edgecolor="none",
                alpha=0.64,
                zorder=7,
            )
        )

    ax.text(
        x + width / 2,
        y + height * 0.60,
        "4-D RADE",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color=PURPLE,
        zorder=9,
    )
    ax.text(
        x + width / 2,
        y + height * 0.46,
        "radar tensor",
        ha="center",
        va="center",
        fontsize=6.4,
        color=INK,
        zorder=9,
    )
    ax.text(
        x + width / 2,
        y + 1.25,
        "[R=256, A=107, D=64, E=37]",
        ha="center",
        va="center",
        fontsize=5.2,
        fontweight="bold",
        color=INK,
        zorder=9,
    )

    # Axis cues: front plane is R-A, oblique cube depth is D, repeated cubes E.
    ax.annotate(
        "",
        xy=(x - 0.85, y + height),
        xytext=(x - 0.85, y),
        arrowprops=dict(arrowstyle="-|>", color=NAVY, linewidth=0.75),
        zorder=10,
    )
    ax.text(x - 1.45, y + height / 2, "R", ha="center", va="center", fontsize=5.6, fontweight="bold", color=NAVY)
    ax.annotate(
        "",
        xy=(x + width, y - 0.82),
        xytext=(x, y - 0.82),
        arrowprops=dict(arrowstyle="-|>", color=NAVY, linewidth=0.75),
        zorder=10,
    )
    ax.text(x + width / 2, y - 1.55, "A", ha="center", va="center", fontsize=5.6, fontweight="bold", color=NAVY)
    ax.annotate(
        "",
        xy=(x + width + 2.7, y + height + 1.55),
        xytext=(x + width + 0.15, y + height + 0.12),
        arrowprops=dict(arrowstyle="-|>", color=PURPLE, linewidth=0.7),
        zorder=12,
    )
    ax.text(x + width + 2.95, y + height + 1.65, "D = 64 bins", ha="left", va="center", fontsize=5.0, color=PURPLE, zorder=12)
    ax.annotate(
        "",
        xy=(x - 1.0, y + height + 3.55),
        xytext=(x + 0.1, y + height + 0.25),
        arrowprops=dict(arrowstyle="-|>", color=GOLD, linewidth=0.7),
        zorder=12,
    )
    ax.text(x + 2.7, y + height + 4.1, "E = 37 stacked slices", ha="center", va="center", fontsize=5.0, color=GOLD, fontweight="bold", zorder=12)
    ax.text(x + width / 2, y - 4.0, "conceptual upstream representation", ha="center", va="center", fontsize=4.9, color=MID, fontstyle="italic")


def projection_operator(
    ax,
    x: float,
    y: float,
    *,
    title: str,
    formula: str,
    detail: str,
    face: str,
    edge: str,
) -> None:
    rounded_box(ax, x, y, 14.8, 10.0, face=face, edge=edge, linewidth=0.9, radius=0.85, zorder=4)
    ax.text(
        x + 7.4,
        y + 7.8,
        title,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=edge,
        zorder=8,
    )
    ax.text(
        x + 7.4,
        y + 4.6,
        formula,
        ha="center",
        va="center",
        fontsize=5.15,
        color=INK,
        linespacing=1.0,
        zorder=8,
    )
    ax.text(
        x + 7.4,
        y + 1.7,
        detail,
        ha="center",
        va="center",
        fontsize=4.8,
        color=MID,
        fontstyle="italic",
        zorder=8,
    )


def view_cube(
    ax,
    x: float,
    y: float,
    *,
    name: str,
    subtitle: str,
    stored_shape: str,
    model_shape: str,
    image: np.ndarray,
    channels: int,
    cmap: str,
    edge: str,
) -> None:
    image_cuboid(
        ax,
        x,
        y,
        13.4,
        12.1,
        image=image,
        depth=channel_depth(channels),
        cmap=cmap,
        edge=edge,
        title=name,
        subtitle=subtitle,
        shape=stored_shape,
    )
    ax.text(
        x + 6.7,
        y - 1.55,
        model_shape,
        ha="center",
        va="center",
        fontsize=5.0,
        fontweight="bold",
        color=edge,
        zorder=9,
    )


def compact_encoder(
    ax,
    x: float,
    y: float,
    *,
    name: str,
    accent: str,
    pale: str,
) -> None:
    width, height = 49.0, 17.5
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        face=pale,
        edge=accent,
        linewidth=0.9,
        linestyle=(0, (4, 2)),
        radius=0.9,
        zorder=1,
    )
    ax.text(x + 1.4, y + 15.4, f"{name} Swin–FPN encoder", ha="left", va="center", fontsize=6.8, fontweight="bold", color=accent, zorder=8)
    ax.text(x + width - 1.2, y + 15.4, "independent weights", ha="right", va="center", fontsize=4.8, color=MID, fontstyle="italic", zorder=8)

    specs = (
        (1.7, 8.0, 7.8, 64, "S1 · Swin ×2", "64 · 128×54"),
        (13.1, 7.7, 6.8, 128, "S2 · Swin ×2", "128 · 64×27"),
        (24.5, 7.4, 5.8, 256, "S3 · Swin ×2", "256 · 32×14"),
        (37.1, 8.0, 7.8, 128, "P1 · FPN", "128 · 128×54"),
    )
    previous_end: tuple[float, float] | None = None
    for sx, block_w, block_h, channels, title, detail in specs:
        block_x = x + sx
        block_y = y + 3.0 + (7.8 - block_h) / 2
        if previous_end is not None:
            arrow(
                ax,
                previous_end,
                (block_x - 0.35, block_y + block_h / 2),
                color=accent,
                linewidth=0.75,
                mutation=6.5,
            )
        cuboid(
            ax,
            block_x,
            block_y,
            block_w,
            block_h,
            depth=0.56 * channel_depth(channels),
            face=WHITE,
            edge=accent,
            title=title,
            detail=detail,
            title_size=5.3,
            detail_size=4.35,
            linewidth=0.72,
            zorder=3,
        )
        previous_end = (block_x + block_w + 0.56 * channel_depth(channels), block_y + block_h / 2)


def concat_node(ax, x: float, y: float) -> None:
    ax.add_patch(patches.Circle((x, y), 1.65, facecolor=WHITE, edgecolor=PURPLE, linewidth=1.1, zorder=10))
    ax.text(x, y + 0.12, "C", ha="center", va="center", fontsize=8.0, fontweight="bold", color=PURPLE, zorder=11)
    ax.text(x, y - 2.9, "channel concat", ha="center", va="center", fontsize=5.0, color=PURPLE, fontweight="bold", zorder=11)
    ax.text(x, y - 4.7, "[B,256,128,54]", ha="center", va="center", fontsize=4.9, color=INK, zorder=11)


def draw_decoder_and_heads(ax) -> None:
    # CenterPointDecoder is the pair of parallel heads below; it is not a
    # learned serial block in front of them.  The dashed enclosure makes that
    # implementation boundary explicit.
    rounded_box(
        ax,
        143.2,
        27.2,
        34.4,
        37.0,
        face=PALE_PURPLE,
        edge=PURPLE,
        linewidth=0.9,
        linestyle=(0, (4, 2)),
        radius=0.9,
        zorder=1,
    )
    ax.text(
        145.0,
        61.8,
        "CenterPoint decoder",
        ha="left",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=PURPLE,
        zorder=9,
    )
    ax.text(
        145.0,
        59.6,
        "parallel heads · grid stride 2",
        ha="left",
        va="center",
        fontsize=5.0,
        color=MID,
        zorder=9,
    )
    ax.add_patch(
        patches.Circle(
            (149.2, 45.4),
            1.35,
            facecolor=WHITE,
            edgecolor=PURPLE,
            linewidth=0.9,
            zorder=8,
        )
    )
    ax.text(149.2, 45.4, "split", ha="center", va="center", fontsize=4.8, fontweight="bold", color=PURPLE, zorder=9)

    cuboid(
        ax,
        162.3,
        51.0,
        12.0,
        9.0,
        depth=channel_depth(1),
        face=PALE_ORANGE,
        edge=ORANGE,
        title="Sedan heatmap",
        detail="[B,1,128,54]\ncenter logits",
        title_size=6.2,
        detail_size=4.6,
    )
    cuboid(
        ax,
        162.3,
        29.8,
        12.0,
        10.5,
        depth=channel_depth(8),
        face=PALE_GREEN,
        edge=GREEN,
        title="box regression",
        detail="[B,8,128,54]\ndx,dy,dz,l,w,h\nsin(yaw), cos(yaw)",
        title_size=6.1,
        detail_size=4.25,
    )
    arrow(ax, (150.7, 46.1), (162.0, 55.4), color=ORANGE, connection="arc3,rad=-0.10", mutation=7.0)
    arrow(ax, (150.7, 44.7), (162.0, 35.2), color=GREEN, connection="arc3,rad=0.10", mutation=7.0)


def draw_postprocess(ax) -> None:
    rounded_box(ax, 180.3, 35.4, 12.2, 20.6, face=PANEL, edge=NAVY, linewidth=1.0, radius=0.85, zorder=3)
    ax.text(186.4, 52.7, "CenterPoint\ndecode", ha="center", va="center", fontsize=6.8, fontweight="bold", color=NAVY, zorder=8)
    ax.plot([181.7, 191.1], [48.9, 48.9], color=LIGHT, linewidth=0.65, zorder=8)
    ax.text(
        186.4,
        44.0,
        "sigmoid + local peaks\ntop-K centers\nR–A cells + offsets\nmetric box conversion",
        ha="center",
        va="center",
        fontsize=4.9,
        color=INK,
        linespacing=1.22,
        zorder=8,
    )
    pill(ax, 182.3, 36.9, "rotated NMS", face=PALE_RED, edge=RED, width=8.2, fontsize=4.8)
    arrow(ax, (176.1, 55.4), (180.0, 50.0), color=ORANGE, connection="arc3,rad=0.10", mutation=7.0)
    arrow(ax, (176.1, 35.2), (180.0, 41.5), color=GREEN, connection="arc3,rad=-0.10", mutation=7.0)


def draw_bottom_notes(ax, source_note: str) -> None:
    rounded_box(ax, 2.2, 2.8, 69.0, 11.2, face=WHITE, edge=LIGHT, linewidth=0.8, radius=0.8, zorder=1)
    ax.text(3.6, 11.75, "Shallow 3-D drawing convention", ha="left", va="center", fontsize=6.4, fontweight="bold", color=NAVY, zorder=8)
    ax.text(3.6, 8.7, "RADE: front = R×A · cube depth = D · repeated slices = E", ha="left", va="center", fontsize=5.4, color=INK, zorder=8)
    ax.text(3.6, 6.25, "Features: front = spatial grid · depth = channels", ha="left", va="center", fontsize=5.4, color=INK, zorder=8)
    ax.text(3.6, 4.25, "Perspective is illustrative; it does not denote 3-D convolution.", ha="left", va="center", fontsize=5.0, color=MID, fontstyle="italic", zorder=8)
    for offset, channels in ((0.0, 64), (7.0, 128), (14.4, 256)):
        cuboid(
            ax,
            49.4 + offset,
            6.4,
            3.2,
            2.2,
            depth=0.38 * channel_depth(channels),
            face=PALE_BLUE,
            edge=BLUE,
            linewidth=0.55,
            zorder=3,
        )
        ax.text(51.0 + offset, 4.65, str(channels), ha="center", va="center", fontsize=4.4, color=BLUE, zorder=8)

    rounded_box(ax, 73.3, 2.8, 144.7, 11.2, face=PANEL, edge=LIGHT, linewidth=0.8, radius=0.8, zorder=1)
    ax.text(74.9, 11.75, "Implementation boundary and provenance", ha="left", va="center", fontsize=6.4, fontweight="bold", color=NAVY, zorder=8)
    ax.text(
        74.9,
        8.9,
        "Upstream: RADE is projected into views. Active loader: paired precomputed RAD/RAE .npy tensors are read directly.",
        ha="left",
        va="center",
        fontsize=5.35,
        color=INK,
        zorder=8,
    )
    ax.text(
        74.9,
        6.55,
        "Model7 concatenates encoded Swin–FPN features—not raw views. The endpoint here is 3-D detection, not semantic segmentation.",
        ha="left",
        va="center",
        fontsize=5.35,
        color=INK,
        zorder=8,
    )
    ax.text(
        74.9,
        4.25,
        "Overview convention inspired by Ouaknine et al., Multi-View Radar Semantic Segmentation, ICCV 2021 Fig. 1 / thesis Fig. 5.1; graphics are original.",
        ha="left",
        va="center",
        fontsize=5.05,
        color=MID,
        fontstyle="italic",
        zorder=8,
    )
    ax.text(217.7, 1.25, f"Input thumbnails: {source_note} · raster only on RAD/RAE faces; all architecture elements are vector", ha="right", va="center", fontsize=4.65, color=MID, zorder=8)


def validate_projection_contract() -> None:
    """Guard the illustrated dimension reductions against an axis swap."""
    probe = np.empty((2, 3, 4, 5), dtype=np.uint8)  # [R, A, D, E]
    if probe.mean(axis=3).shape != (2, 3, 4):
        raise AssertionError("mean over E must produce an [R,A,D] RAD tensor")
    if probe.mean(axis=2).shape != (2, 3, 5):
        raise AssertionError("mean over D must produce an [R,A,E] RAE tensor")


def build_figure(rad_map: np.ndarray, rae_map: np.ndarray, source_note: str):
    setup_style()
    # Liberation Serif preserves the established compact thesis styling while
    # producing a clean embedded TrueType subset in the PDF on this host.
    mpl.rcParams["font.serif"] = ["Liberation Serif"]
    fig, ax = plt.subplots(figsize=(20.4, 8.8))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 220)
    ax.set_ylim(0, 92)
    ax.axis("off")

    draw_header(ax)
    section_label(ax, 3.4, 78.3, "1", "4-D radar tensor")
    section_label(ax, 23.2, 78.3, "2", "mean projections + stored views")
    section_label(ax, 57.2, 78.3, "3", "independent Swin–FPN encoders")
    section_label(ax, 112.0, 78.3, "4", "encoded concat + fusion")
    section_label(ax, 142.3, 78.3, "5", "dense decoder heads")
    section_label(ax, 179.5, 78.3, "6", "decode")
    section_label(ax, 198.0, 78.3, "7", "output")

    draw_rade_tensor(ax, 3.6, 36.7)

    # The split and projection operators describe the upstream data-preparation
    # logic. The active loader begins at the two stored-view cubes to the right.
    ax.plot([20.5, 22.2], [47.0, 47.0], color=NAVY, linewidth=0.95, zorder=5)
    ax.plot([22.2, 22.2], [34.2, 60.2], color=NAVY, linewidth=0.95, zorder=5)
    arrow(ax, (22.2, 60.2), (24.0, 60.2), color=BLUE, linewidth=1.0, mutation=7.5)
    arrow(ax, (22.2, 34.2), (24.0, 34.2), color=TEAL, linewidth=1.0, mutation=7.5)
    ax.text(22.2, 48.8, "split", ha="center", va="bottom", fontsize=4.8, color=NAVY, fontweight="bold")

    projection_operator(
        ax,
        24.4,
        55.2,
        title="Mean over elevation E",
        formula="RAD(r,a,d) = (1/N_E) Σ_e\nX(r,a,d,e)",
        detail="collapse E · retain Doppler",
        face=PALE_BLUE,
        edge=BLUE,
    )
    projection_operator(
        ax,
        24.4,
        29.2,
        title="Mean over Doppler D",
        formula="RAE(r,a,e) = (1/N_D) Σ_d\nX(r,a,d,e)",
        detail="collapse D · retain elevation",
        face=PALE_TEAL,
        edge=TEAL,
    )
    arrow(ax, (39.4, 60.2), (42.0, 58.2), color=BLUE, mutation=7.0)
    arrow(ax, (39.4, 34.2), (42.0, 32.2), color=TEAL, mutation=7.0)

    view_cube(
        ax,
        42.3,
        51.8,
        name="RAD view",
        subtitle="range–azimuth–Doppler",
        stored_shape="stored [256,107,64]",
        model_shape="model [B,64,256,107]",
        image=rad_map,
        channels=64,
        cmap="magma",
        edge=BLUE,
    )
    view_cube(
        ax,
        42.3,
        25.8,
        name="RAE view",
        subtitle="range–azimuth–elevation",
        stored_shape="stored [256,107,37]",
        model_shape="model [B,37,256,107]",
        image=rae_map,
        channels=37,
        cmap="viridis",
        edge=TEAL,
    )
    # One loader sample is a paired RAD/RAE item, so the marker is centered
    # between the two stored-view cubes and connected to both.
    pill(ax, 42.0, 43.4, "PAIRED LOADER: RAD + RAE", face=PALE_RED, edge=RED, width=15.8, fontsize=4.7)
    ax.plot([49.9, 49.9], [45.75, 51.45], color=RED, linewidth=0.65, linestyle=(0, (2, 2)), zorder=5)
    ax.plot([49.9, 49.9], [38.25, 43.35], color=RED, linewidth=0.65, linestyle=(0, (2, 2)), zorder=5)

    arrow(ax, (58.4, 57.8), (60.0, 58.7), color=BLUE, mutation=7.0)
    arrow(ax, (58.4, 31.8), (60.0, 32.7), color=TEAL, mutation=7.0)
    compact_encoder(ax, 60.3, 49.7, name="RAD", accent=BLUE, pale=PALE_BLUE)
    compact_encoder(ax, 60.3, 23.7, name="RAE", accent=TEAL, pale=PALE_TEAL)

    # Convergence happens after—not before—the two non-shared encoders.
    arrow(ax, (110.2, 58.5), (116.5, 47.8), color=BLUE, connection="arc3,rad=0.12", mutation=7.5)
    arrow(ax, (110.2, 32.5), (116.5, 44.4), color=TEAL, connection="arc3,rad=-0.12", mutation=7.5)
    concat_node(ax, 118.1, 46.0)
    arrow(ax, (119.9, 46.0), (122.1, 46.0), color=PURPLE, mutation=7.5)

    cuboid(
        ax,
        122.5,
        36.8,
        14.1,
        18.2,
        depth=channel_depth(128),
        face=PALE_CYAN,
        edge=CYAN,
        title="learned feature\nfusion",
        detail="1×1 Conv–BN–Act\n256 → 128 channels\nresidual 3×3 refine",
        title_size=6.7,
        detail_size=4.8,
    )
    ax.text(129.55, 34.25, "fused [B,128,128,54]", ha="center", va="center", fontsize=5.0, fontweight="bold", color=CYAN, zorder=9)
    arrow(ax, (140.1, 46.0), (147.7, 45.5), color=PURPLE, mutation=7.5)

    draw_decoder_and_heads(ax)
    draw_postprocess(ax)
    bev_output(ax, 198.3, 25.0, concept=False)
    arrow(ax, (192.7, 45.5), (198.0, 45.5), color=NAVY, mutation=7.5)

    draw_bottom_notes(ax, source_note)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--radar-root",
        type=Path,
        default=DEFAULT_RADAR_ROOT,
        help="RAD/RAE root used for the project-owned input thumbnails",
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
        help="draw even if train_cfg.py no longer selects the illustrated Model7 graph",
    )
    return parser.parse_args()


def export_figure(fig, out_dir: Path, formats: Iterable[str], dpi: int) -> list[Path]:
    written = save_figure(fig, out_dir, formats, dpi, stem=STEM)
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    if not args.skip_config_check:
        validate_current_config()
    validate_projection_contract()
    rad_map, rae_map, source_note = load_input_maps(args.radar_root)
    fig = build_figure(rad_map, rae_map, source_note)
    for path in export_figure(fig, args.out_dir, args.formats, args.dpi):
        print(path)


if __name__ == "__main__":
    main()
