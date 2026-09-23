#!/usr/bin/env python3
"""Plot distance-quartile relative AP-drop trends by test sequence."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Nimbus Roman",
            "Times New Roman",
            "Times",
            "Liberation Serif",
        ],
        "mathtext.fontset": "stix",
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "distance_quartiles"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "target_drop_plots"
DEFAULT_TEST_SEQUENCES = (24, 25, 50, 51, 52)
QUARTILES = ("q1", "q2", "q3", "q4")
DISPLAY_QUARTILES = (
    "q0\n(0–25%)",
    "q1\n(25–50%)",
    "q2\n(50–75%)",
    "q3\n(75–100%)",
)
METRICS = ("BEV", "3D")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate BEV and 3D relative AP-drop trend plots from the "
            "distance-quartile rain and sleet tables."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Directory containing rain_experiments.txt and sleet_experiments.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which PNG files are written.",
    )
    parser.add_argument(
        "--test-seqs",
        type=int,
        nargs="+",
        default=list(DEFAULT_TEST_SEQUENCES),
        help="Test sequence ids to plot.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def load_experiment_table(path: Path, weather: str) -> pd.DataFrame:
    """Load completed group rows from a whitespace-aligned experiment table."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty experiment table: {path}")

    headers = lines[0].split()
    rows = []
    for line in lines[2:]:
        tokens = line.split()
        if not tokens or not tokens[0].startswith("group"):
            continue
        # Incomplete placeholder rows contain only the six metadata fields.
        if len(tokens) != len(headers):
            continue
        rows.append(dict(zip(headers, tokens)))

    if not rows:
        raise ValueError(f"No completed group rows found in {path}")

    frame = pd.DataFrame(rows)
    frame["weather"] = weather
    frame["test_seq"] = pd.to_numeric(frame["test_seq"], errors="raise").astype(int)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)

    for metric in METRICS:
        for quartile in QUARTILES:
            for branch in ("src", "tgt"):
                column = f"{metric}_{branch}_{quartile}"
                frame[column] = pd.to_numeric(frame[column], errors="raise")

    return frame


def load_experiments(experiment_dir: Path) -> pd.DataFrame:
    frames = [
        load_experiment_table(experiment_dir / "rain_experiments.txt", "rain"),
        load_experiment_table(experiment_dir / "sleet_experiments.txt", "sleet"),
    ]
    return pd.concat(frames, ignore_index=True)


def add_relative_ap_drop(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate 100 * (AP_tgt - AP_src) / AP_tgt for each metric/quartile."""
    frame = frame.copy()
    for metric in METRICS:
        for quartile in QUARTILES:
            source = frame[f"{metric}_src_{quartile}"]
            target = frame[f"{metric}_tgt_{quartile}"].replace(0.0, float("nan"))
            frame[f"{metric}_drop_{quartile}"] = (target - source) / target * 100.0
    return frame


def add_absolute_td(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the signed target-domain difference AP_tgt - AP_src."""
    frame = frame.copy()
    for metric in METRICS:
        for quartile in QUARTILES:
            source = frame[f"{metric}_src_{quartile}"]
            target = frame[f"{metric}_tgt_{quartile}"]
            frame[f"{metric}_absolute_td_{quartile}"] = target - source
    return frame


def plot_test_sequence_metric(
    frame: pd.DataFrame,
    test_seq: int,
    metric: str,
    output_dir: Path,
    dpi: int = 300,
    show_title: bool = True,
) -> Path:
    """Create one relative AP-drop trend PNG for a metric and test sequence."""
    metric = metric.upper()
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}; expected one of {METRICS}")

    subset = frame.loc[frame["test_seq"] == test_seq].copy()
    if subset.empty:
        raise ValueError(f"No completed rows found for test sequence {test_seq}")

    subset = subset.sort_values(["seed", "group"], kind="stable")
    weather = ", ".join(sorted(subset["weather"].unique()))
    colors = plt.get_cmap("tab10").colors
    x_values = list(range(len(QUARTILES)))

    fig, axis = plt.subplots(figsize=(13.5, 8.8))
    columns = [f"{metric}_drop_{quartile}" for quartile in QUARTILES]
    for color_index, (_, row) in enumerate(subset.iterrows()):
        axis.plot(
            x_values,
            [row[column] for column in columns],
            color=colors[color_index % len(colors)],
            marker="o",
            markersize=6,
            linewidth=2.0,
            alpha=0.78,
            label=(
                f"group{color_index + 1} "
                f"(seed {row['seed']}, source {row['source_seq']})"
            ),
        )
    average = subset[columns].mean(axis=0, skipna=True)
    axis.plot(
        x_values,
        average.to_numpy(),
        color="black",
        marker="D",
        markersize=7,
        linewidth=3.2,
        linestyle="-",
        label="average",
        zorder=10,
    )
    if test_seq in (24, 25):
        axis.set_ylim(bottom=20.0)
    else:
        values = subset[columns].to_numpy(dtype=float)
        data_min = float(pd.DataFrame(values).min().min())
        data_max = float(pd.DataFrame(values).max().max())
        padding = max((data_max - data_min) * 0.08, 1.0)
        lower = math.floor((data_min - padding) / 5.0) * 5.0
        upper = math.ceil((data_max + padding) / 5.0) * 5.0
        axis.set_ylim(bottom=lower, top=upper)
        if lower <= 0.0 <= upper:
            axis.axhline(0.0, color="0.35", linewidth=1.0, linestyle=":")
    axis.set_xticks(x_values, DISPLAY_QUARTILES)
    axis.set_xlabel(
        "GT boxes ranked by distance percentile (near → far)",
        fontsize=17,
    )
    axis.set_ylabel("Relative target drop (%)", fontsize=17)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=17, frameon=True, loc="best")

    if show_title:
        fig.suptitle(
            f"{metric} relative target drop — test sequence {test_seq} ({weather})",
            fontsize=24,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    else:
        fig.tight_layout()

    relative_dir = output_dir / "relative"
    relative_dir.mkdir(parents=True, exist_ok=True)
    output_path = relative_dir / (
        f"{metric.lower()}_relative_target_drop_test_seq_{test_seq}"
        f"{'' if show_title else '_no_title'}.png"
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_test_sequence_absolute_td(
    frame: pd.DataFrame,
    test_seq: int,
    metric: str,
    output_dir: Path,
    dpi: int = 300,
    show_title: bool = True,
) -> Path:
    """Create one signed absolute-TD trend PNG for a metric/test sequence."""
    metric = metric.upper()
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}; expected one of {METRICS}")

    subset = frame.loc[frame["test_seq"] == test_seq].copy()
    if subset.empty:
        raise ValueError(f"No completed rows found for test sequence {test_seq}")

    subset = subset.sort_values(["seed", "group"], kind="stable")
    weather = ", ".join(sorted(subset["weather"].unique()))
    colors = plt.get_cmap("tab10").colors
    x_values = list(range(len(QUARTILES)))
    columns = [f"{metric}_absolute_td_{quartile}" for quartile in QUARTILES]

    fig, axis = plt.subplots(figsize=(13.5, 8.8))
    for color_index, (_, row) in enumerate(subset.iterrows()):
        axis.plot(
            x_values,
            [row[column] for column in columns],
            color=colors[color_index % len(colors)],
            marker="o",
            markersize=6,
            linewidth=2.0,
            alpha=0.78,
            label=(
                f"group{color_index + 1} "
                f"(seed {row['seed']}, source {row['source_seq']})"
            ),
        )

    average = subset[columns].mean(axis=0, skipna=True)
    axis.plot(
        x_values,
        average.to_numpy(),
        color="black",
        marker="D",
        markersize=7,
        linewidth=3.2,
        linestyle="-",
        label="average",
        zorder=10,
    )

    axis.axhline(0.0, color="0.35", linewidth=1.0, linestyle=":")
    axis.set_xticks(x_values, DISPLAY_QUARTILES)
    axis.set_xlabel(
        "GT boxes ranked by distance percentile (near → far)",
        fontsize=17,
    )
    axis.set_ylabel("Absolute TD (AP points)", fontsize=17)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=17, frameon=True, loc="best")
    if show_title:
        fig.suptitle(
            f"{metric} absolute TD — test sequence {test_seq} ({weather})",
            fontsize=24,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    else:
        fig.tight_layout()

    absolute_dir = output_dir / "absolute"
    absolute_dir.mkdir(parents=True, exist_ok=True)
    output_path = absolute_dir / (
        f"{metric.lower()}_absolute_td_test_seq_{test_seq}"
        f"{'' if show_title else '_no_title'}.png"
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_overall_relative_target_drop(
    frame: pd.DataFrame,
    test_sequences: list[int],
    metric: str,
    output_dir: Path,
    dpi: int = 300,
    show_title: bool = True,
    destination_dir: Path | None = None,
) -> Path:
    """Overlay the mean relative-target-drop trend of each test sequence."""
    metric = metric.upper()
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}; expected one of {METRICS}")

    colors = plt.get_cmap("tab10").colors
    x_values = list(range(len(QUARTILES)))
    columns = [f"{metric}_drop_{quartile}" for quartile in QUARTILES]
    figure_size = (13.5, 8.8) if len(test_sequences) >= 5 else (10.4, 7.2)
    fig, axis = plt.subplots(figsize=figure_size)

    for color_index, test_seq in enumerate(test_sequences):
        subset = frame.loc[frame["test_seq"] == test_seq].copy()
        if subset.empty:
            raise ValueError(f"No completed rows found for test sequence {test_seq}")
        weather = ", ".join(sorted(subset["weather"].unique()))
        trend = subset[columns].mean(axis=0, skipna=True)
        axis.plot(
            x_values,
            trend.to_numpy(),
            color=colors[color_index % len(colors)],
            marker="o",
            markersize=7,
            linewidth=2.6,
            label=f"test {test_seq} ({weather})",
        )

    axis.set_xticks(x_values, DISPLAY_QUARTILES)
    axis.set_xlabel(
        "GT boxes ranked by distance percentile (near → far)",
        fontsize=20,
    )
    axis.set_ylabel("Mean relative target drop (%)", fontsize=20)
    axis.tick_params(axis="both", labelsize=18)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=16, frameon=True, loc="best")
    sequence_title = ", ".join(str(value) for value in test_sequences)
    if show_title:
        fig.suptitle(
            f"{metric} overall relative target-drop trend — tests {sequence_title}",
            fontsize=26,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    else:
        fig.tight_layout()

    relative_dir = destination_dir or (output_dir / "relative")
    relative_dir.mkdir(parents=True, exist_ok=True)
    sequence_text = "_".join(str(value) for value in test_sequences)
    output_path = relative_dir / (
        f"{metric.lower()}_overall_relative_target_drop_test_sequences_"
        f"{sequence_text}{'' if show_title else '_no_title'}.png"
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main(argv=None):
    args = parse_args(argv)
    frame = load_experiments(args.experiment_dir.resolve())
    frame = add_absolute_td(add_relative_ap_drop(frame))
    for test_seq in args.test_seqs:
        for metric in METRICS:
            for show_title in (True, False):
                relative_path = plot_test_sequence_metric(
                    frame,
                    test_seq,
                    metric,
                    args.output_dir.resolve(),
                    dpi=args.dpi,
                    show_title=show_title,
                )
                absolute_path = plot_test_sequence_absolute_td(
                    frame,
                    test_seq,
                    metric,
                    args.output_dir.resolve(),
                    dpi=args.dpi,
                    show_title=show_title,
                )
                print(relative_path)
                print(absolute_path)
    requested_sequences = set(args.test_seqs)
    for overview_group in ((24, 25), (50, 51, 52)):
        selected = [value for value in overview_group if value in requested_sequences]
        if not selected:
            continue
        for metric in METRICS:
            for show_title in (True, False):
                overview_path = plot_overall_relative_target_drop(
                    frame,
                    selected,
                    metric,
                    args.output_dir.resolve(),
                    dpi=args.dpi,
                    show_title=show_title,
                )
                print(overview_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
