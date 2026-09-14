#!/usr/bin/env python3
"""Generate the weather experiment data-composition table.

The table follows the experiment sheets in ``experiments/``.  Dataset
partitions do not depend on the random seed, so the first seed (42) is used
once per group instead of repeating identical statistics for seeds 43/44.

Target-train and target-test counts are computed from the Cartesian Sedan GT
used by the current experiments.  Normal-train counts come from the exact
controlled splits recorded in the corresponding source-branch training logs.
Shared sequences are intentionally omitted to match the requested table.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from data.splits import (
    DEFAULT_RANGE_M_BINS,
    _build_frame_infos,
    _select_sequence_part,
    _summarize_frames,
)
from configs.training import TRAIN_CONFIG
from training.experiments.tables import load_domain_shift_experiments


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments"
RUN_DIR = ROOT / "runs" / "experiment_queue"
OUTPUT_PATH = EXPERIMENT_DIR / "weather_experiment_data_summary.txt"
CONTROL_CLASSES = ("Sedan",)
HALF_RATIO = 0.5

WEATHER_TABLES = (
    ("Heavy Snow", "heavy_snow_experiments.txt", "heavy_snow"),
    ("Light Snow", "light_snow_experiments.txt", "light_snow"),
    ("Overcast", "overcast_experiments.txt", "overcast"),
    ("Rain", "rain_experiments.txt", "rain"),
    ("Sleet", "sleet_experiments.txt", "sleet"),
)


def format_part(sequence: int, position: str) -> str:
    return str(sequence) if position == "full" else f"{sequence}_{position}"


def format_parts(parts: tuple[tuple[int, str], ...]) -> str:
    return ",".join(format_part(int(sequence), str(position)) for sequence, position in parts)


@lru_cache(maxsize=None)
def sequence_part_stats(sequence: int, position: str) -> tuple[int, int]:
    frame_infos = _build_frame_infos(
        sequence=int(sequence),
        range_m_bins=DEFAULT_RANGE_M_BINS,
        box_coordinate_mode="cartesian",
        cartesian_gt_root=TRAIN_CONFIG["cartesian_gt_root"],
        control_class_names=CONTROL_CLASSES,
    )
    selected_infos = _select_sequence_part(
        frame_infos,
        position=str(position),
        ratio=HALF_RATIO,
        complementary=False,
    )
    summary = _summarize_frames(selected_infos)
    return int(summary["frames"]), int(summary["all_target_objects"])


def aggregate_original_parts(parts: tuple[tuple[int, str], ...]) -> tuple[int, int]:
    frames = 0
    bboxes = 0
    for sequence, position in parts:
        part_frames, part_bboxes = sequence_part_stats(int(sequence), str(position))
        frames += part_frames
        bboxes += part_bboxes
    return frames, bboxes


def find_controlled_split(weather_slug: str, group: str, seed: int) -> Path:
    pattern = f"{weather_slug}_*_{group}_seed{seed}_source.train.log"
    candidates = sorted(RUN_DIR.glob(f"**/{pattern}"))
    if not candidates:
        raise FileNotFoundError(
            f"No source training log found for {weather_slug}/{group}/seed{seed}"
        )

    split_pattern = re.compile(
        r"(?:Generated|Reusing) controlled training split:\s*(\S+)"
    )
    for log_path in reversed(candidates):
        match = split_pattern.search(log_path.read_text(encoding="utf-8", errors="replace"))
        if match is None:
            continue
        split_path = Path(match.group(1))
        if not split_path.is_absolute():
            split_path = ROOT / split_path
        if (split_path / "stats.json").is_file():
            return split_path
    raise FileNotFoundError(
        f"No valid controlled split recorded for {weather_slug}/{group}/seed{seed}"
    )


def controlled_source_stats(split_path: Path) -> tuple[int, int]:
    stats = json.loads((split_path / "stats.json").read_text(encoding="utf-8"))
    frames = 0
    bboxes = 0
    for pair in stats["pairs"]:
        after = pair["after_summary"]
        frames += int(after["frames"])
        # This is the number of Sedan targets remaining after the actual
        # experiment's object-ignore control has been applied.
        bboxes += int(
            after.get(
                "selected_target_objects",
                after["effective_target_objects"],
            )
        )
    return frames, bboxes


def group_label(name: str) -> str:
    match = re.fullmatch(r"group(\d+)", name.strip().lower())
    return f"G{match.group(1)}" if match else name


def combined_cell(parts: tuple[tuple[int, str], ...], frames: int) -> str:
    return f"{format_parts(parts)} ({frames:,})"


def render_table(rows: list[list[str]]) -> str:
    headers = [
        "Weather",
        "Group",
        "Target Train (Sequences, Frames)",
        "Target Train BBoxes",
        "Normal Train (Sequences, Frames)",
        "Normal Train BBoxes",
        "Target Test (Sequences, Frames)",
        "Target Test BBoxes",
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def line(values: list[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths)).rstrip()

    separator = "  ".join("-" * width for width in widths)
    notes = [
        "# Weather experiment data composition",
        "# Seed-independent partitions: each group is listed once using its seed-42 definition; seeds 43/44 repeat the same data.",
        "# BBoxes are the effective Sedan target boxes used by the current Cartesian experiments.",
        "# Normal Train uses the exact logged controlled split; shared sequences are omitted to match the requested table.",
        "",
    ]
    return "\n".join(notes + [line(headers), separator] + [line(row) for row in rows]) + "\n"


def main() -> None:
    rows: list[list[str]] = []
    for weather_name, table_name, weather_slug in WEATHER_TABLES:
        experiments = load_domain_shift_experiments(EXPERIMENT_DIR / table_name)
        seed = min(experiment.seed for experiment in experiments)
        experiments = [experiment for experiment in experiments if experiment.seed == seed]

        for experiment in experiments:
            target_frames, target_bboxes = aggregate_original_parts(experiment.target_parts)
            test_frames, test_bboxes = aggregate_original_parts(experiment.test_parts)
            controlled_split = find_controlled_split(
                weather_slug=weather_slug,
                group=experiment.name,
                seed=experiment.seed,
            )
            normal_frames, normal_bboxes = controlled_source_stats(controlled_split)
            rows.append(
                [
                    weather_name,
                    group_label(experiment.name),
                    combined_cell(experiment.target_parts, target_frames),
                    f"{target_bboxes:,}",
                    combined_cell(experiment.source_parts, normal_frames),
                    f"{normal_bboxes:,}",
                    combined_cell(experiment.test_parts, test_frames),
                    f"{test_bboxes:,}",
                ]
            )

    OUTPUT_PATH.write_text(render_table(rows), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
