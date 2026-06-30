from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from cfg_model import AZIMUTH_AXIS, ELEVATION_AXIS, RANGE_AXIS, is_rae_center_in_gt_scope
from zxy_label_utils import read_gt_txt


OFFICIAL_TAG_PATH = Path("/home/local/xinyu/K-Radar/tools/tag_generator/tag_generation.py")
GT_ROOT = Path("/home/local/xinyu/K-Radar-GT-Polar-v2")
OUTPUT_DIR = Path("analysis_plots/road_type_stats")

SEDAN_CLASS = "Sedan"
BUS_CLASS = "Bus or Truck"
WEATHER_ORDER = ["normal", "overcast", "fog", "rain", "sleet", "lightsnow", "heavysnow"]


def load_official_sequence_tags(tag_path: Path) -> dict[int, dict[str, str]]:
    tree = ast.parse(tag_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "dict_tag":
                raw_tags = ast.literal_eval(node.value)
                return {
                    int(sequence): {
                        "road": values[0],
                        "time": values[1],
                        "weather": values[2],
                    }
                    for sequence, values in raw_tags.items()
                }
    raise RuntimeError(f"Could not find dict_tag in {tag_path}")


def is_in_full_fov(raw: dict[str, float]) -> bool:
    return (
        0 <= raw["r_idx"] < RANGE_AXIS.size
        and 0 <= raw["a_idx"] < AZIMUTH_AXIS.size
        and 0 <= raw["e_idx"] < ELEVATION_AXIS.size
    )


def empty_count_dict() -> Counter:
    return Counter(
        {
            "all_total": 0,
            "narrow_total": 0,
            "all_sedan": 0,
            "narrow_sedan": 0,
            "all_bus": 0,
            "narrow_bus": 0,
        }
    )


def safe_ratio(kept: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return kept / total


def attach_ignore_metrics(item: dict[str, int]) -> dict[str, int | float]:
    result = dict(item)
    result["ignored_total"] = result["all_total"] - result["narrow_total"]
    result["ignored_sedan"] = result["all_sedan"] - result["narrow_sedan"]
    result["ignored_bus"] = result["all_bus"] - result["narrow_bus"]
    result["keep_rate_total"] = safe_ratio(result["narrow_total"], result["all_total"])
    result["keep_rate_sedan"] = safe_ratio(result["narrow_sedan"], result["all_sedan"])
    result["keep_rate_bus"] = safe_ratio(result["narrow_bus"], result["all_bus"])
    result["ignore_rate_total"] = 1.0 - result["keep_rate_total"] if result["all_total"] > 0 else 0.0
    result["ignore_rate_sedan"] = 1.0 - result["keep_rate_sedan"] if result["all_sedan"] > 0 else 0.0
    result["ignore_rate_bus"] = 1.0 - result["keep_rate_bus"] if result["all_bus"] > 0 else 0.0
    return result


def count_sequence_objects(sequence: int) -> dict[str, int]:
    gt_path = GT_ROOT / str(sequence) / "gt" / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT file: {gt_path}")

    gt_by_file_idx = read_gt_txt(str(gt_path))
    counts = empty_count_dict()
    frame_ids = set()

    for file_idx, objects in gt_by_file_idx.items():
        frame_ids.add(file_idx)
        for obj in objects:
            raw = obj["raw"]
            if not is_in_full_fov(raw):
                continue

            cls = obj["cls"]
            counts["all_total"] += 1
            if cls == SEDAN_CLASS:
                counts["all_sedan"] += 1
            elif cls == BUS_CLASS:
                counts["all_bus"] += 1

            if is_rae_center_in_gt_scope(
                r_idx=raw["r_idx"],
                a_idx=raw["a_idx"],
                e_idx=raw["e_idx"],
            ):
                counts["narrow_total"] += 1
                if cls == SEDAN_CLASS:
                    counts["narrow_sedan"] += 1
                elif cls == BUS_CLASS:
                    counts["narrow_bus"] += 1

    counts["frames_with_gt"] = len(frame_ids)
    return dict(counts)


def aggregate_road_stats(road_type: str) -> dict:
    tags = load_official_sequence_tags(OFFICIAL_TAG_PATH)
    selected_sequences = sorted(
        sequence for sequence, meta in tags.items() if meta["road"] == road_type
    )

    weather_to_sequences = defaultdict(list)
    by_sequence = {}
    total_counts = empty_count_dict()
    weather_counts = {weather: empty_count_dict() for weather in WEATHER_ORDER}
    total_frames_with_gt = 0

    for sequence in selected_sequences:
        meta = tags[sequence]
        weather = meta["weather"]
        counts = count_sequence_objects(sequence)
        by_sequence[sequence] = attach_ignore_metrics({
            "road": meta["road"],
            "time": meta["time"],
            "weather": weather,
            **counts,
        })
        weather_to_sequences[weather].append(sequence)
        total_frames_with_gt += counts["frames_with_gt"]

        for key, value in counts.items():
            if key == "frames_with_gt":
                continue
            total_counts[key] += value
            weather_counts[weather][key] += value

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "official_tag_path": str(OFFICIAL_TAG_PATH),
            "gt_root": str(GT_ROOT),
        },
        "road_type": road_type,
        "scope_definition": {
            "all": "Full valid GT tensor FOV: 0<=r<256, 0<=a<107, 0<=e<37.",
            "narrow": (
                "Current project narrow scope: Cartesian ROI "
                "x:[0,72], y:[-6.4,6.4], z:[-2,6] plus azimuth [-50deg, 50deg]."
            ),
        },
        "selected_sequences": selected_sequences,
        "weather_to_sequences": {
            weather: weather_to_sequences[weather]
            for weather in WEATHER_ORDER
            if weather_to_sequences[weather]
        },
        "summary": {
            **attach_ignore_metrics(dict(total_counts)),
            "frames_with_gt": total_frames_with_gt,
            "num_sequences": len(selected_sequences),
        },
        "by_weather": {
            weather: attach_ignore_metrics(dict(weather_counts[weather]))
            for weather in WEATHER_ORDER
            if weather_to_sequences[weather]
        },
        "by_sequence": by_sequence,
    }


def _draw_table(
    ax,
    title: str,
    col_labels: list[str],
    cell_text: list[list[str]],
    col_widths: list[float] | None = None,
    font_size: int = 12,
    row_scale: float = 1.45,
) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=10)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, row_scale)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#6B7280")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#DCE6F2")
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("#EEF4FB")


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def plot_road_stats(stats: dict, output_path: Path) -> None:
    summary = stats["summary"]
    weather_to_sequences = stats["weather_to_sequences"]
    by_weather = stats["by_weather"]
    road_type = stats["road_type"]
    road_label = road_type.title()

    fig = plt.figure(figsize=(26, 18), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.25, 2.1, 4.3])
    ax_info = fig.add_subplot(grid[0, 0])
    ax_summary = fig.add_subplot(grid[1, 0])
    ax_weather = fig.add_subplot(grid[2, 0])
    ax_sequence = fig.add_subplot(grid[3, 0])

    fig.suptitle(f"K-Radar {road_label} Object Statistics", fontsize=20)

    ax_info.axis("off")
    info_lines = [
        f"{road_label} sequences ({summary['num_sequences']}): "
        + ", ".join(str(sequence) for sequence in stats["selected_sequences"]),
        "Weather split: "
        + " | ".join(
            f"{weather}: {', '.join(str(sequence) for sequence in weather_to_sequences[weather])}"
            for weather in WEATHER_ORDER
            if weather in weather_to_sequences
        ),
        "Scope: all = valid GT tensor FOV | narrow = ROI(x:0-72m, y:-6.4~6.4m, z:-2~6m) + azimuth[-50deg, 50deg]",
        "Ignore rate definition: (all - narrow) / all",
    ]
    ax_info.text(0.01, 0.98, "\n".join(info_lines), ha="left", va="top", fontsize=13)

    summary_table = [[
        str(summary["all_total"]),
        str(summary["narrow_total"]),
        str(summary["all_sedan"]),
        str(summary["narrow_sedan"]),
        str(summary["all_bus"]),
        str(summary["narrow_bus"]),
        fmt_pct(summary["ignore_rate_sedan"]),
        fmt_pct(summary["ignore_rate_bus"]),
        str(summary["frames_with_gt"]),
    ]]
    _draw_table(
        ax_summary,
        "Overall Summary",
        [
            "All Total",
            "Narrow Total",
            "Sedan All",
            "Sedan Narrow",
            "Bus All",
            "Bus Narrow",
            "Sedan Ignore Rate",
            "Bus Ignore Rate",
            "Frames with GT",
        ],
        summary_table,
        col_widths=[0.09, 0.10, 0.09, 0.10, 0.09, 0.10, 0.12, 0.12, 0.11],
        font_size=12,
        row_scale=1.9,
    )

    weather_rows = []
    for weather in WEATHER_ORDER:
        if weather not in by_weather:
            continue
        counts = by_weather[weather]
        sequences = ", ".join(str(sequence) for sequence in weather_to_sequences[weather])
        weather_rows.append([
            weather,
            sequences,
            str(counts["all_total"]),
            str(counts["narrow_total"]),
            str(counts["all_sedan"]),
            str(counts["narrow_sedan"]),
            str(counts["all_bus"]),
            str(counts["narrow_bus"]),
            fmt_pct(counts["ignore_rate_sedan"]),
            fmt_pct(counts["ignore_rate_bus"]),
        ])
    _draw_table(
        ax_weather,
        "Weather Breakdown",
        [
            "Weather",
            "Sequences",
            "All Total",
            "Narrow Total",
            "Sedan All",
            "Sedan Narrow",
            "Bus All",
            "Bus Narrow",
            "Sedan Ignore Rate",
            "Bus Ignore Rate",
        ],
        weather_rows,
        col_widths=[0.08, 0.17, 0.08, 0.09, 0.08, 0.09, 0.08, 0.09, 0.12, 0.12],
        font_size=11,
        row_scale=1.65,
    )

    sequence_rows = []
    for sequence in stats["selected_sequences"]:
        item = stats["by_sequence"][sequence]
        sequence_rows.append([
            str(sequence),
            item["time"],
            item["weather"],
            str(item["all_total"]),
            str(item["narrow_total"]),
            str(item["all_sedan"]),
            str(item["narrow_sedan"]),
            str(item["all_bus"]),
            str(item["narrow_bus"]),
            fmt_pct(item["ignore_rate_sedan"]),
            fmt_pct(item["ignore_rate_bus"]),
            str(item["frames_with_gt"]),
        ])
    _draw_table(
        ax_sequence,
        "Per-Sequence Breakdown",
        [
            "Seq",
            "Time",
            "Weather",
            "All Total",
            "Narrow Total",
            "Sedan All",
            "Sedan Narrow",
            "Bus All",
            "Bus Narrow",
            "Sedan Ignore Rate",
            "Bus Ignore Rate",
            "Frames",
        ],
        sequence_rows,
        col_widths=[0.04, 0.06, 0.08, 0.08, 0.09, 0.08, 0.09, 0.07, 0.08, 0.11, 0.11, 0.07],
        font_size=10,
        row_scale=1.28,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-type", default="highway")
    args = parser.parse_args()

    stats = aggregate_road_stats(args.road_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{timestamp}_kradar_{args.road_type}_weather_counts.png"
    json_path = OUTPUT_DIR / f"{timestamp}_kradar_{args.road_type}_weather_counts.json"
    plot_road_stats(stats, png_path)
    json_path.write_text(json.dumps(stats, indent=2))

    print(f"PNG: {png_path}")
    print(f"JSON: {json_path}")
    print(f"{args.road_type} sequences:", stats["selected_sequences"])
    print("Summary:", stats["summary"])


if __name__ == "__main__":
    main()
