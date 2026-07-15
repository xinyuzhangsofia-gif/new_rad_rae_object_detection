from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import KRadarRADRAEDataset
from zxy_data_path import get_gt_txt_path, get_rad_rae_npy_root_dir
from zxy_label_utils import read_gt_txt


RAD_ROOT = get_rad_rae_npy_root_dir()
OUTPUT_DIR = Path("analysis_plots/domain_shift_stats")
DEFAULT_SOURCE_TRAIN = (1, 5, 6, 14, 15, 18, 20)
DEFAULT_TARGET_TRAIN = (3, 9, 11, 12)
DEFAULT_TARGET_TEST = (4, 10)


def parse_sequence_list(text: str) -> tuple[int, ...]:
    values = []
    for token in text.replace(" ", "").split(","):
        if token == "":
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid sequence range: {token!r}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if len(values) == 0:
        raise ValueError("Sequence list must not be empty.")
    return tuple(values)


def format_sequence_slug(sequences: tuple[int, ...]) -> str:
    return "".join(str(sequence) for sequence in sequences)


def sequence_stats(sequence: int) -> dict[str, int]:
    dataset = KRadarRADRAEDataset(RAD_ROOT, sequence)
    gt = read_gt_txt(get_gt_txt_path(None, sequence=sequence))

    frames = len(dataset.frame_names)
    empty = 0
    sedan = 0
    bus = 0
    bbox = 0

    for file_idx in range(frames):
        objects = gt.get(file_idx, [])
        sedan_count = sum(1 for obj in objects if obj["cls"] == "Sedan")
        bus_count = sum(1 for obj in objects if obj["cls"] == "Bus or Truck")
        bbox_count = sedan_count + bus_count

        sedan += sedan_count
        bus += bus_count
        bbox += bbox_count
        if bbox_count == 0:
            empty += 1

    return {
        "frames": frames,
        "empty": empty,
        "sedan": sedan,
        "bus": bus,
        "bbox": bbox,
    }


def aggregate_group(sequences: tuple[int, ...]) -> dict[str, float]:
    total = {"frames": 0, "empty": 0, "sedan": 0, "bus": 0, "bbox": 0}
    for sequence in sequences:
        stats = sequence_stats(sequence)
        for key, value in stats.items():
            total[key] += value
    total["empty_rate"] = total["empty"] / total["frames"] if total["frames"] else 0.0
    return total


def split_line_to_sequence_and_frame_names(line: str):
    line = line.strip()
    if line == "" or line.startswith("#"):
        return None

    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Invalid split line: {line!r}")

    sequence = int(parts[0])
    frame_token = os.path.splitext(parts[1])[0]
    if frame_token == "":
        raise ValueError(f"Invalid split frame token: {line!r}")

    frame_name_candidates = []
    if "_" in frame_token:
        frame_name_candidates.append(frame_token.split("_")[0])
        frame_name_candidates.append(frame_token)
    else:
        frame_name_candidates.append(frame_token)
    return sequence, frame_name_candidates


def aggregate_group_from_split(split_dir: str | Path, split_name: str = "train.txt") -> tuple[dict[str, float], tuple[int, ...]]:
    split_dir = Path(split_dir)
    split_path = split_dir / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    entries_by_sequence = {}
    with split_path.open("r", encoding="utf-8") as split_file:
        for line in split_file:
            parsed = split_line_to_sequence_and_frame_names(line)
            if parsed is None:
                continue
            sequence, frame_name_candidates = parsed
            entries_by_sequence.setdefault(sequence, []).append(frame_name_candidates)

    total = {"frames": 0, "empty": 0, "sedan": 0, "bus": 0, "bbox": 0}
    for sequence, frame_name_candidates_list in sorted(entries_by_sequence.items()):
        dataset = KRadarRADRAEDataset(RAD_ROOT, sequence)
        gt = read_gt_txt(get_gt_txt_path(None, sequence=sequence))
        frame_name_to_local_idx = {
            frame_name: local_idx
            for local_idx, frame_name in enumerate(dataset.frame_names)
        }

        seen_local_indices = set()
        for frame_name_candidates in frame_name_candidates_list:
            local_idx = None
            for frame_name in frame_name_candidates:
                local_idx = frame_name_to_local_idx.get(frame_name)
                if local_idx is not None:
                    break
            if local_idx is None or local_idx in seen_local_indices:
                continue
            seen_local_indices.add(local_idx)

            objects = gt.get(local_idx, [])
            sedan_count = sum(1 for obj in objects if obj["cls"] == "Sedan")
            bus_count = sum(1 for obj in objects if obj["cls"] == "Bus or Truck")
            bbox_count = sedan_count + bus_count

            total["frames"] += 1
            total["sedan"] += sedan_count
            total["bus"] += bus_count
            total["bbox"] += bbox_count
            if bbox_count == 0:
                total["empty"] += 1

    total["empty_rate"] = total["empty"] / total["frames"] if total["frames"] else 0.0
    return total, tuple(sorted(entries_by_sequence.keys()))


def ratio_text(train_value: int, test_value: int) -> str:
    total = train_value + test_value
    train_pct = train_value / total * 100 if total else 0.0
    test_pct = test_value / total * 100 if total else 0.0
    test_on_7_scale = test_value / train_value * 7 if train_value else 0.0
    return (
        f"{train_value}:{test_value} = "
        f"{train_pct:.2f}%:{test_pct:.2f}%  (~7:{test_on_7_scale:.2f})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-split-dir",
        default=None,
        help="Optional split directory whose train.txt should be used as source-domain train.",
    )
    parser.add_argument(
        "--source-train",
        default="1,5,6,14,15,18,20",
        help="Comma-separated source-domain train sequences.",
    )
    parser.add_argument(
        "--target-train",
        default="3,9,11,12",
        help="Comma-separated target-domain train sequences.",
    )
    parser.add_argument(
        "--target-test",
        default="4,10",
        help="Comma-separated target-domain test sequences.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_train = parse_sequence_list(args.target_train)
    target_test = parse_sequence_list(args.target_test)

    source_display_text = ""
    source_table_text = ""
    if args.source_split_dir:
        source_train_stats, source_split_sequences = aggregate_group_from_split(
            split_dir=args.source_split_dir,
            split_name="train.txt",
        )
        source_train = source_split_sequences
        source_display_text = f"split:{Path(args.source_split_dir).name}/train.txt"
        source_table_text = "file split"
    else:
        source_train = parse_sequence_list(args.source_train)
        source_train_stats = aggregate_group(source_train)
        source_display_text = ",".join(str(sequence) for sequence in source_train)
        source_table_text = source_display_text

    groups = [
        ("Source Train", source_train),
        ("Target Train", target_train),
        ("Target Test", target_test),
    ]

    stats_by_group = {
        "Target Train": aggregate_group(target_train),
        "Target Test": aggregate_group(target_test),
    }
    stats_by_group["Source Train"] = source_train_stats

    source = stats_by_group["Source Train"]
    target_train_stats = stats_by_group["Target Train"]
    target_test_stats = stats_by_group["Target Test"]

    rows = []
    for name, sequences in groups:
        stats = stats_by_group[name]
        rows.append(
            [
                name,
                source_table_text if name == "Source Train" else ",".join(str(sequence) for sequence in sequences),
                f"{stats['frames']}",
                f"{stats['empty']}",
                f"{stats['empty_rate'] * 100:.2f}%",
                f"{stats['bbox']}",
                f"{stats['sedan']}",
                f"{stats['bus']}",
            ]
        )

    fig = plt.figure(figsize=(16, 10.5), dpi=180)
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.suptitle("Domain Shift Split Summary", fontsize=20, y=0.97)

    col_labels = [
        "Set",
        "Sequences",
        "Frames",
        "Empty",
        "Empty Rate",
        "Total BBox",
        "Sedan BBox",
        "Bus BBox",
    ]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.48, 0.96, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dbeafe")
        elif row == 1:
            cell.set_facecolor("#f8fafc")
        elif row == 2:
            cell.set_facecolor("#eefbf3")
        elif row == 3:
            cell.set_facecolor("#fff7ed")

    target_ratio_lines = [
        "Target train : target test ratios",
        f"Frames      : {ratio_text(target_train_stats['frames'], target_test_stats['frames'])}",
        f"Total BBox  : {ratio_text(target_train_stats['bbox'], target_test_stats['bbox'])}",
        f"Sedan BBox  : {ratio_text(target_train_stats['sedan'], target_test_stats['sedan'])}",
        f"Bus BBox    : {ratio_text(target_train_stats['bus'], target_test_stats['bus'])}",
        f"Empty Frames: {ratio_text(target_train_stats['empty'], target_test_stats['empty'])}",
    ]

    source_ratio_lines = [
        "Source train : target test ratios",
        f"Frames      : {ratio_text(source['frames'], target_test_stats['frames'])}",
        f"Total BBox  : {ratio_text(source['bbox'], target_test_stats['bbox'])}",
        f"Sedan BBox  : {ratio_text(source['sedan'], target_test_stats['sedan'])}",
        f"Bus BBox    : {ratio_text(source['bus'], target_test_stats['bus'])}",
        f"Empty Frames: {ratio_text(source['empty'], target_test_stats['empty'])}",
    ]

    quick_view_lines = [
        "Source vs Target-train quick view",
        f"Frames      : {source['frames']} vs {target_train_stats['frames']}",
        f"Total BBox  : {source['bbox']} vs {target_train_stats['bbox']}",
        f"Sedan BBox  : {source['sedan']} vs {target_train_stats['sedan']}",
        f"Bus BBox    : {source['bus']} vs {target_train_stats['bus']}",
        f"Empty Rate  : {source['empty_rate'] * 100:.2f}% vs {target_train_stats['empty_rate'] * 100:.2f}%",
    ]

    def add_text_box(x: float, y: float, w: float, h: float, lines: list[str], fontsize: int = 11) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            linewidth=1.0,
            edgecolor="#cbd5e1",
            facecolor="#f8fafc",
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.012,
            y + h - 0.012,
            "\n".join(lines),
            fontsize=fontsize,
            family="monospace",
            transform=ax.transAxes,
            va="top",
        )

    add_text_box(
        0.03,
        0.34,
        0.44,
        0.10,
        [
            "Group Definition",
            f"Source train = {source_display_text}",
            f"Target train = {target_train}",
            f"Target test  = {target_test}",
        ],
        fontsize=12,
    )
    add_text_box(0.03, 0.15, 0.44, 0.15, target_ratio_lines, fontsize=10)
    add_text_box(0.52, 0.15, 0.44, 0.15, source_ratio_lines, fontsize=10)
    add_text_box(0.03, 0.03, 0.93, 0.09, quick_view_lines, fontsize=10)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / (
        f"{timestamp}_source_{Path(args.source_split_dir).name if args.source_split_dir else format_sequence_slug(source_train)}"
        f"_target_{format_sequence_slug(target_train)}"
        f"_test_{format_sequence_slug(target_test)}_summary.png"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
