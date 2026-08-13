"""Plot the weather/road-type joint distribution using actual radar frames."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE_CSV = PROJECT_ROOT / "sequence_information.csv"
DEFAULT_RADAR_ROOT = Path(
    "/run/user/1000/gvfs/"
    "smb-share:server=192.168.189.30,share=elab-share/"
    "Datasets/K-Radar-RAD"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

WEATHER_ORDER = (
    "normal",
    "overcast",
    "rain",
    "fog",
    "sleet",
    "lightsnow",
    "heavysnow",
)
WEATHER_LABELS = {
    "normal": "Normal",
    "overcast": "Overcast",
    "rain": "Rain",
    "fog": "Fog",
    "sleet": "Sleet",
    "lightsnow": "Light Snow",
    "heavysnow": "Heavy Snow",
}
WEATHER_COLORS = {
    "normal": "#E6A15A",
    "overcast": "#7FA6D8",
    "rain": "#6FBFA5",
    "fog": "#A987C7",
    "sleet": "#D9828E",
    "lightsnow": "#7EC5D8",
    "heavysnow": "#D6B65C",
}

ROAD_ORDER = (
    "urban",
    "highway",
    "alleyway",
    "suburban",
    "university",
    "mountain",
    "parking_lot",
    "shoulder",
)
ROAD_LABELS = {
    "urban": "Urban",
    "highway": "Highway",
    "alleyway": "Alleyway",
    "suburban": "Suburban",
    "university": "University",
    "mountain": "Mountain",
    "parking_lot": "Parking Lot",
    "shoulder": "Shoulder",
}
ROAD_ALIASES = {
    "urban": "urban",
    "highway": "highway",
    "alleyway": "alleyway",
    "countryside": "suburban",
    "suburban": "suburban",
    "university": "university",
    "mountain": "mountain",
    "parkinglots": "parking_lot",
    "parking_lot": "parking_lot",
    "parking lot": "parking_lot",
    "shoulder": "shoulder",
}
ROAD_COLORS = {
    "urban": "#E6A15A",
    "highway": "#7FA6D8",
    "alleyway": "#74B878",
    "suburban": "#D9828E",
    "university": "#6EBEC6",
    "mountain": "#A987C7",
    "parking_lot": "#C98AB8",
    "shoulder": "#D6B65C",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count matched RAD/RAE frame files and plot their joint "
            "distribution by weather and road type."
        )
    )
    parser.add_argument("--sequence-csv", type=Path, default=DEFAULT_SEQUENCE_CSV)
    parser.add_argument("--radar-root", type=Path, default=DEFAULT_RADAR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def normalize_weather(value):
    weather = str(value).strip().lower().replace(" ", "").replace("_", "")
    if weather not in WEATHER_LABELS:
        raise ValueError(f"Unsupported weather condition: {value!r}")
    return weather


def normalize_road(value):
    road = str(value).strip().lower()
    try:
        return ROAD_ALIASES[road]
    except KeyError as error:
        raise ValueError(f"Unsupported road type: {value!r}") from error


def npy_stems(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {directory}")
    return {path.stem for path in directory.glob("*.npy") if path.is_file()}


def count_sequence_frames(radar_root, sequence_id):
    sequence_dir = radar_root / str(sequence_id)
    rad_names = npy_stems(sequence_dir / "rad")
    rae_names = npy_stems(sequence_dir / "rae")
    matched_names = rad_names & rae_names
    if not matched_names:
        raise ValueError(f"Sequence {sequence_id} has no matched RAD/RAE frames")
    return {
        "rad_files": len(rad_names),
        "rae_files": len(rae_names),
        "matched_frames": len(matched_names),
        "rad_only": len(rad_names - rae_names),
        "rae_only": len(rae_names - rad_names),
    }


def load_and_count(sequence_csv, radar_root):
    with sequence_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"Sequence information CSV is empty: {sequence_csv}")

    joint_counts = defaultdict(lambda: defaultdict(int))
    audit_rows = []
    for row in rows:
        sequence_id = int(row["sequence_id"])
        weather = normalize_weather(row["weather"])
        road = normalize_road(row["environment"])
        counts = count_sequence_frames(radar_root, sequence_id)
        joint_counts[weather][road] += counts["matched_frames"]
        audit_rows.append({
            "sequence_id": sequence_id,
            "weather": WEATHER_LABELS[weather],
            "road_type": ROAD_LABELS[road],
            "csv_frames": int(row["frames"]),
            **counts,
            "matched_minus_csv": counts["matched_frames"] - int(row["frames"]),
        })

    present_weather = set(joint_counts)
    weather_order = tuple(
        weather for weather in WEATHER_ORDER if weather in present_weather
    )
    unexpected_weather = present_weather - set(weather_order)
    if unexpected_weather:
        raise ValueError(f"Weather order is missing: {sorted(unexpected_weather)}")
    return weather_order, joint_counts, audit_rows


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(output_dir, weather_order, joint_counts, audit_rows):
    joint_rows = []
    for weather in weather_order:
        row = {"weather": WEATHER_LABELS[weather]}
        total = 0
        for road in ROAD_ORDER:
            count = int(joint_counts[weather][road])
            row[ROAD_LABELS[road]] = count
            total += count
        row["Total"] = total
        joint_rows.append(row)

    joint_fields = [
        "weather",
        *(ROAD_LABELS[road] for road in ROAD_ORDER),
        "Total",
    ]
    joint_path = output_dir / "weather_road_frame_joint_distribution.csv"
    audit_path = output_dir / "sequence_frame_count_audit.csv"
    write_csv(joint_path, joint_fields, joint_rows)
    write_csv(
        audit_path,
        (
            "sequence_id",
            "weather",
            "road_type",
            "csv_frames",
            "rad_files",
            "rae_files",
            "matched_frames",
            "rad_only",
            "rae_only",
            "matched_minus_csv",
        ),
        audit_rows,
    )
    return joint_path, audit_path, joint_rows


def plot_distribution(output_dir, weather_order, joint_counts, dpi):
    x_positions = list(range(len(weather_order)))
    bottoms = [0] * len(weather_order)
    fig, ax = plt.subplots(figsize=(13.5, 8.0), constrained_layout=True)

    for road in ROAD_ORDER:
        values = [int(joint_counts[weather][road]) for weather in weather_order]
        bars = ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            width=0.72,
            label=ROAD_LABELS[road],
            color=ROAD_COLORS[road],
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value, bottom in zip(bars, values, bottoms):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value >= 900 else "black",
                    fontweight="semibold",
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for x_position, total in zip(x_positions, bottoms):
        ax.text(
            x_position,
            total + max(bottoms) * 0.012,
            f"{total:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(
        x_positions,
        [WEATHER_LABELS[weather] for weather in weather_order],
    )
    ax.set_xlabel("Weather Condition", fontsize=12)
    ax.set_ylabel("Number of Matched RAD/RAE Frames", fontsize=12)
    ax.set_title(
        "Joint Distribution of Weather Conditions and Road Types",
        fontsize=15,
        pad=16,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(bottoms) * 1.10)
    ax.legend(
        title="Road Type",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )

    output_path = output_dir / "weather_road_frame_stacked_bar.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def category_summary(audit_rows, field_name, category_order, category_labels):
    summary = []
    for category in category_order:
        display_name = category_labels[category]
        category_rows = [
            row for row in audit_rows
            if row[field_name] == display_name
        ]
        if not category_rows:
            continue
        summary.append({
            "category": category,
            "display_name": display_name,
            "sequence_count": len(category_rows),
            "frame_count": sum(
                int(row["matched_frames"]) for row in category_rows
            ),
        })
    return summary


def plot_category_pie(output_path, summary, colors, dpi):
    values = [row["frame_count"] for row in summary]
    slice_colors = [colors[row["category"]] for row in summary]
    total_frames = sum(values)

    def format_slice(row):
        return (
            f"{row['display_name']}\n"
            f"{row['frame_count']:,} frames"
        )

    fig, ax = plt.subplots(figsize=(17.0, 14.0), constrained_layout=True)
    wedges, _label_texts = ax.pie(
        values,
        colors=slice_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )

    outside_labels = {-1: [], 1: []}
    for wedge, row in zip(wedges, summary):
        percentage = 100.0 * row["frame_count"] / total_frames
        middle_angle = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
        x_direction = math.cos(middle_angle)
        y_direction = math.sin(middle_angle)
        label = format_slice(row)
        if percentage < 8.0:
            side = 1 if x_direction >= 0 else -1
            outside_labels[side].append({
                "label": label,
                "natural_y": 1.12 * y_direction,
                "arrow_xy": (0.94 * x_direction, 0.94 * y_direction),
                "color": wedge.get_facecolor(),
            })
            continue

        ax.text(
            0.64 * x_direction,
            0.64 * y_direction,
            label,
            fontsize=22,
            fontweight="normal",
            color="black",
            ha="center",
            va="center",
            linespacing=1.35,
        )

    for side, items in outside_labels.items():
        items.sort(key=lambda item: item["natural_y"])
        minimum_gap = 0.40
        y_positions = []
        for item in items:
            y_position = item["natural_y"]
            if y_positions:
                y_position = max(y_position, y_positions[-1] + minimum_gap)
            y_positions.append(y_position)
        if y_positions and y_positions[-1] > 1.16:
            shift = y_positions[-1] - 1.16
            y_positions = [position - shift for position in y_positions]
        if y_positions and y_positions[0] < -1.16:
            shift = -1.16 - y_positions[0]
            y_positions = [position + shift for position in y_positions]

        for item, y_position in zip(items, y_positions):
            ax.annotate(
                item["label"],
                xy=item["arrow_xy"],
                xytext=(1.18 * side, y_position),
                fontsize=22,
                fontweight="normal",
                ha="left" if side > 0 else "right",
                va="center",
                linespacing=1.35,
                arrowprops={
                    "arrowstyle": "->",
                    "color": item["color"],
                    "linewidth": 2.0,
                    "shrinkA": 4,
                    "shrinkB": 2,
                },
            )

    ax.axis("equal")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weather_order, joint_counts, audit_rows = load_and_count(
        sequence_csv=args.sequence_csv,
        radar_root=args.radar_root,
    )
    joint_path, audit_path, joint_rows = write_tables(
        output_dir=args.output_dir,
        weather_order=weather_order,
        joint_counts=joint_counts,
        audit_rows=audit_rows,
    )
    plot_path = plot_distribution(
        output_dir=args.output_dir,
        weather_order=weather_order,
        joint_counts=joint_counts,
        dpi=args.dpi,
    )
    weather_summary = category_summary(
        audit_rows=audit_rows,
        field_name="weather",
        category_order=weather_order,
        category_labels=WEATHER_LABELS,
    )
    road_summary = category_summary(
        audit_rows=audit_rows,
        field_name="road_type",
        category_order=ROAD_ORDER,
        category_labels=ROAD_LABELS,
    )
    weather_pie_path = plot_category_pie(
        output_path=args.output_dir / "weather_frame_distribution_pie.png",
        summary=weather_summary,
        colors=WEATHER_COLORS,
        dpi=args.dpi,
    )
    road_pie_path = plot_category_pie(
        output_path=args.output_dir / "road_type_frame_distribution_pie.png",
        summary=road_summary,
        colors=ROAD_COLORS,
        dpi=args.dpi,
    )

    print("weather totals:")
    for row in joint_rows:
        print(f"  {row['weather']}: {row['Total']}")
    print(f"plot={plot_path}")
    print(f"weather_pie={weather_pie_path}")
    print(f"road_pie={road_pie_path}")
    print(f"joint_table={joint_path}")
    print(f"audit_table={audit_path}")


if __name__ == "__main__":
    main()
