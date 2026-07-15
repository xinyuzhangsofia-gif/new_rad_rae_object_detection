from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import KRadarRADRAEDataset
from zxy_data_path import get_gt_txt_path, get_rad_rae_npy_root_dir
from zxy_label_utils import read_gt_txt


SEDAN_CLASS_NAME = "Sedan"
BUS_CLASS_NAME = "Bus or Truck"
CATEGORY_KEYS = (
    "sedan_ridx_0_80",
    "sedan_ridx_80_144",
    "bus_ridx_0_80",
    "bus_ridx_80_144",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a simple random seq9 control split: choose one continuous window, "
            "match seq13 ridx-bin counts by random keep/ignore sampling, and pick "
            "the trial whose empty-frame count is closest."
        )
    )
    parser.add_argument("--source-sequence", type=int, default=9)
    parser.add_argument("--target-sequence", type=int, default=13)
    parser.add_argument("--window-length", type=int, default=1190)
    parser.add_argument(
        "--window-position",
        default="last",
        choices=("first", "last"),
    )
    parser.add_argument("--bin1-upper", type=float, default=80.0)
    parser.add_argument("--bin2-upper", type=float, default=144.0)
    parser.add_argument("--num-trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="split/seq9_simple_random_to_seq13_control",
    )
    return parser.parse_args()


def object_category_key(obj: dict, bin1_upper: float, bin2_upper: float) -> str | None:
    cls = str(obj["cls"])
    if cls not in {SEDAN_CLASS_NAME, BUS_CLASS_NAME}:
        return None

    r_idx = float(obj["raw"]["r_idx"])
    if 0.0 <= r_idx < float(bin1_upper):
        suffix = "ridx_0_80"
    elif float(bin1_upper) <= r_idx < float(bin2_upper):
        suffix = "ridx_80_144"
    else:
        return None

    prefix = "sedan" if cls == SEDAN_CLASS_NAME else "bus"
    return f"{prefix}_{suffix}"


def build_frame_infos(
        sequence: int,
        bin1_upper: float,
        bin2_upper: float,
    ) -> list[dict]:
    radar_dataset = KRadarRADRAEDataset(get_rad_rae_npy_root_dir(), sequence)
    gt_by_file_idx = read_gt_txt(get_gt_txt_path(None, sequence=sequence))
    frame_infos = []

    for file_idx, frame_name in enumerate(radar_dataset.frame_names):
        category_object_labels = {
            key: []
            for key in CATEGORY_KEYS
        }
        target_object_labels_all = []
        target_object_labels_outside_bins = []
        for obj in gt_by_file_idx.get(file_idx, []):
            cls = str(obj["cls"])
            if cls not in {SEDAN_CLASS_NAME, BUS_CLASS_NAME}:
                continue

            object_label = int(obj["object_label"])
            target_object_labels_all.append(object_label)
            category_key = object_category_key(
                obj,
                bin1_upper=bin1_upper,
                bin2_upper=bin2_upper,
            )
            if category_key is None:
                target_object_labels_outside_bins.append(object_label)
                continue
            category_object_labels[category_key].append(object_label)

        for labels in category_object_labels.values():
            labels.sort()
        target_object_labels_all.sort()
        target_object_labels_outside_bins.sort()

        category_counts = {
            key: len(category_object_labels[key])
            for key in CATEGORY_KEYS
        }
        total_in_bins = sum(category_counts.values())
        frame_infos.append(
            {
                "file_idx": int(file_idx),
                "frame_name": str(frame_name),
                "category_object_labels": category_object_labels,
                "category_counts": category_counts,
                "target_object_labels_all": target_object_labels_all,
                "target_object_labels_outside_bins": target_object_labels_outside_bins,
                "total_in_bins": int(total_in_bins),
            }
        )

    return frame_infos


def summarize_frames(frame_infos: list[dict]) -> dict:
    totals = Counter()
    empty_frames = 0
    histogram = Counter()
    for frame_info in frame_infos:
        frame_total = int(frame_info["total_in_bins"])
        histogram[frame_total] += 1
        if frame_total == 0:
            empty_frames += 1
        for key, count in frame_info["category_counts"].items():
            totals[key] += int(count)
    return {
        "frames": len(frame_infos),
        "empty_frames": int(empty_frames),
        "nonempty_frames": int(len(frame_infos) - empty_frames),
        "category_totals": {
            key: int(totals[key])
            for key in CATEGORY_KEYS
        },
        "total_boxes_in_bins": int(sum(totals.values())),
        "frame_total_histogram": {
            str(total): int(count)
            for total, count in sorted(histogram.items())
        },
    }


def build_population_by_category(window_frame_infos: list[dict]) -> dict[str, list[tuple[int, int]]]:
    population = {
        key: []
        for key in CATEGORY_KEYS
    }
    for window_frame_idx, frame_info in enumerate(window_frame_infos):
        for key in CATEGORY_KEYS:
            for object_label in frame_info["category_object_labels"][key]:
                population[key].append((window_frame_idx, int(object_label)))
    return population


def run_random_trial(
        window_frame_infos: list[dict],
        target_summary: dict,
        population_by_category: dict[str, list[tuple[int, int]]],
        seed: int,
    ) -> dict:
    rng = random.Random(int(seed))
    keep_by_frame = {
        frame_idx: set()
        for frame_idx in range(len(window_frame_infos))
    }

    for key in CATEGORY_KEYS:
        target_count = int(target_summary["category_totals"][key])
        population = population_by_category[key]
        if target_count > len(population):
            raise ValueError(
                f"Not enough source objects for {key}: need {target_count}, have {len(population)}"
            )
        for frame_idx, object_label in rng.sample(population, target_count):
            keep_by_frame[frame_idx].add(int(object_label))

    frame_after_counts = {}
    empty_frames = 0
    histogram = Counter()
    override_frames = {}
    override_object_count = 0

    for frame_idx, frame_info in enumerate(window_frame_infos):
        kept_counts = Counter()
        keep_set = keep_by_frame[frame_idx]
        for key in CATEGORY_KEYS:
            for object_label in frame_info["category_object_labels"][key]:
                if int(object_label) in keep_set:
                    kept_counts[key] += 1
        total_kept = sum(kept_counts.values())
        histogram[total_kept] += 1
        if total_kept == 0:
            empty_frames += 1

        ignore_object_labels = [
            int(object_label)
            for object_label in frame_info["target_object_labels_all"]
            if int(object_label) not in keep_set
        ]
        if len(ignore_object_labels) > 0:
            override_frames[frame_info["frame_name"]] = {
                "ignore_object_labels": ignore_object_labels,
            }
            override_object_count += len(ignore_object_labels)

        frame_after_counts[frame_info["frame_name"]] = {
            "total_kept": int(total_kept),
            "category_counts": {
                key: int(kept_counts[key])
                for key in CATEGORY_KEYS
            },
        }

    score = (
        abs(int(empty_frames) - int(target_summary["empty_frames"])),
        int(override_object_count),
        int(seed),
    )
    return {
        "seed": int(seed),
        "score": score,
        "empty_frames": int(empty_frames),
        "frame_total_histogram": {
            str(total): int(count)
            for total, count in sorted(histogram.items())
        },
        "frame_after_counts": frame_after_counts,
        "override_frames": override_frames,
        "override_object_count": int(override_object_count),
    }


def write_split_file(path: Path, sequence: int, frame_names: list[str]) -> None:
    path.write_text(
        "".join(f"{int(sequence)},{frame_name}.txt\n" for frame_name in frame_names),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_frame_infos = build_frame_infos(
        sequence=int(args.target_sequence),
        bin1_upper=float(args.bin1_upper),
        bin2_upper=float(args.bin2_upper),
    )
    target_summary = summarize_frames(target_frame_infos)

    source_frame_infos = build_frame_infos(
        sequence=int(args.source_sequence),
        bin1_upper=float(args.bin1_upper),
        bin2_upper=float(args.bin2_upper),
    )
    if int(args.window_length) > len(source_frame_infos):
        raise ValueError(
            f"window_length={args.window_length} exceeds source sequence length={len(source_frame_infos)}"
        )

    if args.window_position == "first":
        start_file_idx = 0
    else:
        start_file_idx = len(source_frame_infos) - int(args.window_length)
    end_file_idx = start_file_idx + int(args.window_length) - 1
    window_frame_infos = source_frame_infos[start_file_idx:end_file_idx + 1]
    window_summary_before = summarize_frames(window_frame_infos)
    population_by_category = build_population_by_category(window_frame_infos)

    best_trial = None
    for trial_idx in range(int(args.num_trials)):
        trial = run_random_trial(
            window_frame_infos=window_frame_infos,
            target_summary=target_summary,
            population_by_category=population_by_category,
            seed=int(args.seed) + trial_idx,
        )
        if best_trial is None or trial["score"] < best_trial["score"]:
            best_trial = trial

    if best_trial is None:
        raise RuntimeError("Failed to create any random trial.")

    selected_frame_names = [
        frame_info["frame_name"]
        for frame_info in window_frame_infos
    ]
    selected_frame_name_set = set(selected_frame_names)
    excluded_frame_names = [
        frame_info["frame_name"]
        for frame_info in source_frame_infos
        if frame_info["frame_name"] not in selected_frame_name_set
    ]

    achieved_summary = {
        "frames": int(args.window_length),
        "empty_frames": int(best_trial["empty_frames"]),
        "nonempty_frames": int(int(args.window_length) - int(best_trial["empty_frames"])),
        "category_totals": dict(target_summary["category_totals"]),
        "total_boxes_in_bins": int(target_summary["total_boxes_in_bins"]),
        "frame_total_histogram": dict(best_trial["frame_total_histogram"]),
    }

    override_payload = {
        "schema_version": 1,
        "experiment_name": "seq9_simple_random_to_seq13_control",
        "notes": (
            "Simple random control: one fixed continuous seq9 window, then random keep "
            "sampling to match seq13 ridx-bin counts. Raw gt.txt was not modified."
        ),
        "ridx_bins": {
            "bin1": [0.0, float(args.bin1_upper)],
            "bin2": [float(args.bin1_upper), float(args.bin2_upper)],
        },
        "window_position": str(args.window_position),
        "selected_seed": int(best_trial["seed"]),
        "target_summary": target_summary,
        "selected_window_summary_before_override": {
            **window_summary_before,
            "start_file_idx": int(start_file_idx),
            "end_file_idx": int(end_file_idx),
        },
        "selected_window_summary_after_override": achieved_summary,
        "sequences": {
            str(int(args.source_sequence)): {
                "matched_to_sequence": int(args.target_sequence),
                "frame_overrides": best_trial["override_frames"],
                "frame_after_counts": best_trial["frame_after_counts"],
            }
        },
    }

    stats_payload = {
        "source_sequence": int(args.source_sequence),
        "target_sequence": int(args.target_sequence),
        "window_length": int(args.window_length),
        "window_position": str(args.window_position),
        "selected_seed": int(best_trial["seed"]),
        "selected_window": {
            "start_file_idx": int(start_file_idx),
            "end_file_idx": int(end_file_idx),
            "start_frame_name": selected_frame_names[0],
            "end_frame_name": selected_frame_names[-1],
        },
        "target_summary": target_summary,
        "selected_window_summary_before_override": {
            **window_summary_before,
            "start_file_idx": int(start_file_idx),
            "end_file_idx": int(end_file_idx),
        },
        "selected_window_summary_after_override": achieved_summary,
        "best_trial_score": list(best_trial["score"]),
    }

    readme_text = "\n".join([
        "# seq9 simple random to seq13 control",
        "",
        "This is the simplified version:",
        "- choose one continuous seq9 window",
        "- count seq13 sedan/bus bbox in ridx bins `(0-80)` and `(80-144)`",
        "- randomly keep seq9 objects to match those counts",
        "- convert all other seq9 sedan/bus GT to ignore",
        "- choose the random seed whose empty-frame count is closest to seq13",
        "",
        "## Selected window",
        "",
        f"- source sequence: `{int(args.source_sequence)}`",
        f"- target sequence: `{int(args.target_sequence)}`",
        f"- window position: `{args.window_position}`",
        f"- start_file_idx: `{int(start_file_idx)}`",
        f"- end_file_idx: `{int(end_file_idx)}`",
        f"- start_frame_name: `{selected_frame_names[0]}`",
        f"- end_frame_name: `{selected_frame_names[-1]}`",
        f"- selected random seed: `{int(best_trial['seed'])}`",
        "",
        "## Use",
        "",
        "```python",
        "\"split_mode\": \"file\",",
        f"\"split_dir\": \"{output_dir.as_posix()}\",",
        f"\"gt_object_ignore_override_path\": \"{(output_dir / 'object_ignore_override.json').as_posix()}\",",
        "```",
        "",
        "## Summary",
        "",
        f"- target empty/non-empty: `{target_summary['empty_frames']}/{target_summary['nonempty_frames']}`",
        f"- achieved empty/non-empty: `{achieved_summary['empty_frames']}/{achieved_summary['nonempty_frames']}`",
        f"- target boxes in bins: `{target_summary['total_boxes_in_bins']}`",
        f"- achieved boxes in bins: `{achieved_summary['total_boxes_in_bins']}`",
    ]) + "\n"

    (output_dir / "object_ignore_override.json").write_text(
        json.dumps(override_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "stats.json").write_text(
        json.dumps(stats_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(readme_text, encoding="utf-8")
    write_split_file(output_dir / "train.txt", int(args.source_sequence), selected_frame_names)
    write_split_file(output_dir / "test.txt", int(args.source_sequence), excluded_frame_names)

    print(f"Selected window: start={start_file_idx} end={end_file_idx} ({args.window_position})")
    print(f"Selected random seed: {best_trial['seed']}")
    print(
        f"Target empty/non-empty: {target_summary['empty_frames']}/{target_summary['nonempty_frames']}"
    )
    print(
        f"Achieved empty/non-empty: {achieved_summary['empty_frames']}/{achieved_summary['nonempty_frames']}"
    )
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
