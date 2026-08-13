#!/usr/bin/env python3
"""Draw an original, publication-ready overview of FMCW radar processing.

The figure is deliberately generated from vector primitives and synthetic data,
so it can be edited and reused without relying on third-party artwork.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mvrss_fmcw_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


OUT_DIR = Path(__file__).resolve().parent / "figures"

# Colour-blind-friendly palette that remains distinguishable in print.
NAVY = "#17324D"
BLUE = "#2474B5"
CYAN = "#39A9C6"
ORANGE = "#E17C35"
RED = "#C94C4C"
GREEN = "#3C8D73"
INK = "#252B31"
MID = "#65727E"
LIGHT = "#E8EEF2"
PALE_BLUE = "#EAF4FA"
PALE_ORANGE = "#FCF0E6"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            # Nimbus Roman is the metrically compatible Times-family font
            # available on this system. STIX supplies Times-style mathematical
            # glyphs closely matching the appearance of newtxmath.
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.2,
            "font.weight": "bold",
            "axes.titlesize": 10.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.2,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.linewidth": 0.7,
            "axes.edgecolor": MID,
            "xtick.color": MID,
            "ytick.color": MID,
            "axes.labelcolor": INK,
            "text.color": INK,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_title(ax, title: str, *, centered: bool = False) -> None:
    ax.text(
        0.5 if centered else 0.0,
        1.045,
        title,
        transform=ax.transAxes,
        ha="center" if centered else "left",
        va="bottom",
        fontsize=11.4,
        fontweight="bold",
        color=INK,
    )


def clean_axes(ax, *, left=True, bottom=True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=2.5, width=0.6)


def double_arrow(ax, p0, p1, text, *, color=INK, text_offset=(0, 0), lw=0.9) -> None:
    arrow = patches.FancyArrowPatch(
        p0,
        p1,
        arrowstyle="<->",
        mutation_scale=8,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)
    ax.text(
        (p0[0] + p1[0]) / 2 + text_offset[0],
        (p0[1] + p1[1]) / 2 + text_offset[1],
        text,
        ha="center",
        va="center",
        color=color,
    )


def draw_chirp_panel(ax) -> None:
    """Draw chirps and IF processing on visibly independent coordinate axes."""
    panel_title(ax, "FMCW chirp and beat signal", centered=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Upper coordinate system: transmitted and delayed received chirps.
    chirp_ax = ax.inset_axes([0.055, 0.43, 0.70, 0.50])
    chirp_ax.set_xlim(-0.04, 1.18)
    chirp_ax.set_ylim(-0.34, 1.05)
    # Keep the chirp geometry from appearing horizontally stretched.
    chirp_ax.set_aspect(0.72, adjustable="box", anchor="W")
    chirp_ax.axis("off")
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=9, lw=0.9, color=INK)
    chirp_ax.annotate("", xy=(1.16, 0), xytext=(0.04, 0), arrowprops=arrow_kw)
    chirp_ax.annotate("", xy=(0.06, 1.01), xytext=(0.06, -0.02), arrowprops=arrow_kw)
    chirp_ax.text(1.17, -0.02, r"$t$", ha="left", va="top")
    chirp_ax.text(0.025, 1.02, "frequency", ha="right", va="bottom")

    x0, x1, tau = 0.06, 0.91, 0.18
    y0, y1 = 0.0, 0.88
    slope = (y1 - y0) / (x1 - x0)
    chirp_ax.plot([x0, x1], [y0, y1], color=BLUE, lw=2.0, solid_capstyle="round")
    chirp_ax.plot([x0 + tau, x1 + tau], [y0, y1], color=ORANGE, lw=2.0, solid_capstyle="round")

    # A compact legend keeps text clear of the chirps and construction lines.
    chirp_ax.plot([0.12, 0.23], [0.76, 0.76], color=BLUE, lw=2.0, solid_capstyle="round")
    legend_box = dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=0.96)
    chirp_ax.text(0.255, 0.76, "Tx chirp", color=BLUE, ha="left", va="center", bbox=legend_box, zorder=6)
    chirp_ax.plot([0.61, 0.72], [0.76, 0.76], color=ORANGE, lw=2.0, solid_capstyle="round")
    chirp_ax.text(0.745, 0.76, "Rx echo", color=ORANGE, ha="left", va="center", bbox=legend_box, zorder=6)

    # Bandwidth, chirp duration, propagation delay, and beat-frequency gap.
    double_arrow(chirp_ax, (-0.005, y0), (-0.005, y1), r"$B$", text_offset=(-0.025, 0))
    double_arrow(chirp_ax, (x0, -0.09), (x1, -0.09), r"$T_c$", text_offset=(0, -0.045))
    double_arrow(
        chirp_ax,
        (x0, -0.22),
        (x0 + tau, -0.22),
        r"$\tau=2R/c$",
        color=GREEN,
        text_offset=(0, -0.075),
    )

    x_fb = 0.64
    y_tx = y0 + slope * (x_fb - x0)
    y_rx = y0 + slope * (x_fb - x0 - tau)
    chirp_ax.plot([x_fb, 1.10], [y_tx, y_tx], color=MID, lw=0.7, ls=(0, (3, 2)))
    chirp_ax.plot([x_fb, 1.10], [y_rx, y_rx], color=MID, lw=0.7, ls=(0, (3, 2)))
    double_arrow(
        chirp_ax,
        (x_fb, y_rx),
        (x_fb, y_tx),
        r"$f_b$",
        color=RED,
        text_offset=(0.070, 0),
        lw=1.0,
    )
    chirp_ax.text(0.42, 0.94, r"$S=B/T_c$", color=NAVY, ha="center")

    # The equations use the notation introduced in the thesis text.
    ax.text(
        0.865,
        0.680,
        r"$f_b=S\tau$" "\n" r"$\tau=\dfrac{2R}{c}$" "\n\n" r"$R=\dfrac{c f_b}{2S}$",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        linespacing=1.35,
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.42", fc=PALE_BLUE, ec="#BDD5E3", lw=0.7),
    )

    # Lower coordinate system: IF beat samples. It is independent of the
    # frequency--time coordinate above.
    ax.plot([0.045, 0.955], [0.365, 0.365], color=LIGHT, lw=0.8, transform=ax.transAxes)
    if_ax = ax.inset_axes([0.075, 0.075, 0.84, 0.205])
    if_ax.set_xlim(0, 1.05)
    if_ax.set_ylim(-1.30, 1.32)
    if_ax.axis("off")
    if_ax.annotate("", xy=(1.04, 0), xytext=(0, 0), arrowprops=arrow_kw)
    if_ax.annotate("", xy=(0, 1.28), xytext=(0, -1.25), arrowprops=arrow_kw)
    t = np.linspace(0.05, 0.98, 500)
    beat = 0.78 * np.sin(2 * np.pi * 8.3 * t)
    if_ax.plot(t, beat, color=RED, lw=1.25)
    if_ax.text(0.02, 1.20, "IF beat signal", ha="left", va="bottom")
    if_ax.text(1.04, -0.12, r"$t$", ha="left", va="top")

def draw_cube_panel(ax) -> None:
    panel_title(ax, "Radar data cube and FFT dimensions")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Three offset sampling planes suggest the antenna-channel dimension.
    rows, cols = 5, 9
    x0, y0 = 0.8, 1.25
    w, h = 5.1, 3.25
    offsets = [(0.85, 0.72), (0.43, 0.36), (0.0, 0.0)]
    for layer, (dx, dy) in enumerate(offsets):
        alpha = 0.38 + 0.25 * (layer == 2)
        face = PALE_BLUE if layer < 2 else "#F7FAFC"
        rect = patches.Rectangle(
            (x0 + dx, y0 + dy),
            w,
            h,
            facecolor=face,
            edgecolor=BLUE,
            linewidth=0.9,
            alpha=alpha,
        )
        ax.add_patch(rect)
        if layer == 2:
            for j in range(1, cols):
                xx = x0 + w * j / cols
                ax.plot([xx, xx], [y0, y0 + h], color=LIGHT, lw=0.45)
            for i in range(1, rows):
                yy = y0 + h * i / rows
                ax.plot([x0, x0 + w], [yy, yy], color=LIGHT, lw=0.45)

    # Synthetic reflections in the sample matrix.
    cells = [(2, 1, 0.45), (5, 2, 0.72), (7, 3, 0.95), (4, 4, 0.58)]
    for col, row, strength in cells:
        cx = x0 + w * (col + 0.5) / cols
        cy = y0 + h * (row + 0.5) / rows
        ax.add_patch(
            patches.Circle(
                (cx, cy),
                0.10 + 0.08 * strength,
                facecolor=ORANGE,
                edgecolor="white",
                lw=0.5,
                alpha=0.45 + 0.5 * strength,
            )
        )

    # Axis arrows around the cube.
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=10, lw=1.25)
    ax.annotate("", xy=(6.45, 0.88), xytext=(0.8, 0.88), arrowprops={**arrow_kw, "color": BLUE})
    ax.text(3.6, 0.48, "ADC samples within a chirp (fast time)", ha="center", color=BLUE)
    ax.annotate("", xy=(0.46, 4.82), xytext=(0.46, 1.25), arrowprops={**arrow_kw, "color": GREEN})
    ax.text(0.07, 3.05, "Chirps\n(slow time)", rotation=90, ha="center", va="center", color=GREEN)
    ax.annotate("", xy=(6.80, 5.48), xytext=(6.02, 4.52), arrowprops={**arrow_kw, "color": ORANGE})
    ax.text(5.72, 6.15, "Antenna channels", ha="center", color=ORANGE)

    # FFT labels on the right, connected to their dimensions.
    labels = [
        (5.42, "1", "Range FFT", BLUE, "fast time", r"$f_b \rightarrow R$"),
        (3.45, "2", "Doppler FFT", GREEN, "slow time", r"$f_D \rightarrow v_r$"),
        (1.48, "3", "Angle processing", ORANGE, "channels", r"$\Delta\phi \rightarrow \theta$"),
    ]
    for y, number, title, color, dim, mapping in labels:
        ax.add_patch(patches.Circle((7.25, y), 0.25, fc=color, ec="none"))
        ax.text(
            7.25,
            y,
            number,
            color="white",
            ha="center",
            va="center",
            fontsize=8.7,
            fontfamily="Nimbus Roman",
            fontstyle="normal",
            fontweight="bold",
        )
        ax.text(7.67, y + 0.13, title, ha="left", va="center", fontsize=9.0, fontweight="bold", color=INK)
        ax.text(7.67, y - 0.16, f"{dim}:  {mapping}", ha="left", va="center", fontsize=7.8, fontweight="bold", color=MID)


def main() -> None:
    setup_style()
    fig = plt.figure(figsize=(7.25, 3.65), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        left=0.045,
        right=0.985,
        bottom=0.075,
        top=0.935,
        wspace=0.27,
        width_ratios=[1.06, 0.94],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    draw_chirp_panel(ax_a)
    draw_cube_panel(ax_b)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "fmcw_range_velocity_angle_principle"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=450, facecolor="white")
    plt.close(fig)
    print(f"Wrote {stem}.pdf, {stem}.svg, and {stem}.png")


if __name__ == "__main__":
    main()
