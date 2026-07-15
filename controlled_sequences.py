"""Generate non-destructive, distribution-controlled training sequence splits."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from dataset import KRadarRADRAEDataset
from zxy_data_path import get_gt_txt_path, get_rad_rae_npy_root_dir
from zxy_label_utils import read_gt_txt


SEDAN_CLASS_NAME = "Sedan"
BUS_CLASS_NAME = "Bus or Truck"
TARGET_CLASS_NAMES = (SEDAN_CLASS_NAME, BUS_CLASS_NAME)
DEFAULT_RIDX_BINS = ((0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 120.0), (120.0, 144.0))
CONTROL_SCHEMA_VERSION = 3
REQUIRED_CONTROL_OUTPUT_FILENAMES = (
    "train.txt",
    "test.txt",
    "object_ignore_override.json",
    "stats.json",
    "comparison.txt",
)


def _normalize_sequences(value, name):
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        values = []
        for token in value.replace(" ", "").split(","):
            if token:
                values.append(int(token))
        return tuple(values)
    return tuple(int(sequence) for sequence in value)


def _normalize_bins(value):
    if value is None:
        return DEFAULT_RIDX_BINS

    bins = tuple((float(pair[0]), float(pair[1])) for pair in value)
    if len(bins) == 0:
        raise ValueError("control_ridx_bins must not be empty")

    previous_upper = None
    for lower, upper in bins:
        if lower < 0.0 or upper <= lower:
            raise ValueError(f"Invalid control r_idx bin: {(lower, upper)!r}")
        if previous_upper is not None and lower != previous_upper:
            raise ValueError("control_ridx_bins must be contiguous")
        previous_upper = upper
    return bins


def _format_number(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _bin_key(class_name, lower, upper):
    return f"{class_name.lower().replace(' ', '_')}_ridx_{_format_number(lower)}_{_format_number(upper)}"


def _category_keys(ridx_bins):
    return tuple(
        _bin_key(class_name, lower, upper)
        for class_name in TARGET_CLASS_NAMES
        for lower, upper in ridx_bins
    )


def _category_key(obj, ridx_bins):
    class_name = str(obj["cls"])
    if class_name not in TARGET_CLASS_NAMES:
        return None

    r_idx = float(obj["raw"]["r_idx"])
    for lower, upper in ridx_bins:
        if lower <= r_idx < upper:
            return _bin_key(class_name, lower, upper)
    return None


def _build_frame_infos(sequence, ridx_bins):
    radar_dataset = KRadarRADRAEDataset(
        get_rad_rae_npy_root_dir(),
        int(sequence),
    )
    gt_by_file_idx = read_gt_txt(get_gt_txt_path(None, sequence=int(sequence)))
    category_keys = _category_keys(ridx_bins)
    frame_infos = []

    for file_idx, frame_name in enumerate(radar_dataset.frame_names):
        category_object_labels = {key: [] for key in category_keys}
        all_target_object_labels = []
        outside_bin_object_labels = []
        for obj in gt_by_file_idx.get(file_idx, []):
            class_name = str(obj["cls"])
            if class_name not in TARGET_CLASS_NAMES:
                continue

            object_label = int(obj["object_label"])
            all_target_object_labels.append(object_label)
            category_key = _category_key(obj, ridx_bins)
            if category_key is None:
                outside_bin_object_labels.append(object_label)
            else:
                category_object_labels[category_key].append(object_label)

        for labels in category_object_labels.values():
            labels.sort()
        all_target_object_labels.sort()
        outside_bin_object_labels.sort()
        category_counts = {
            key: len(labels)
            for key, labels in category_object_labels.items()
        }
        frame_infos.append(
            {
                "file_idx": int(file_idx),
                "frame_name": str(frame_name),
                "category_object_labels": category_object_labels,
                "category_counts": category_counts,
                "all_target_object_labels": all_target_object_labels,
                "outside_bin_object_labels": outside_bin_object_labels,
                "total_boxes_in_bins": int(sum(category_counts.values())),
            }
        )

    return frame_infos


def _summarize_frames(frame_infos, kept_by_frame=None):
    category_keys = tuple(
        key
        for frame_info in frame_infos
        for key in frame_info["category_counts"]
    )
    category_keys = tuple(dict.fromkeys(category_keys))
    category_totals = Counter()
    frame_histogram = Counter()
    empty_frames = 0
    all_target_objects = 0
    effective_target_objects = 0
    outside_bin_objects = 0

    for frame_idx, frame_info in enumerate(frame_infos):
        if kept_by_frame is None:
            frame_counts = frame_info["category_counts"]
            effective_target_objects += int(frame_info["total_boxes_in_bins"])
            outside_bin_objects += len(frame_info["outside_bin_object_labels"])
        else:
            kept_labels = set(kept_by_frame.get(frame_idx, set()))
            frame_counts = {
                key: sum(
                    int(label) in kept_labels
                    for label in labels
                )
                for key, labels in frame_info["category_object_labels"].items()
            }
            effective_target_objects += len(kept_labels)

        all_target_objects += len(frame_info["all_target_object_labels"])

        frame_total = int(sum(frame_counts.values()))
        frame_histogram[frame_total] += 1
        if frame_total == 0:
            empty_frames += 1
        for key, count in frame_counts.items():
            category_totals[key] += int(count)

    frames = len(frame_infos)
    return {
        "frames": int(frames),
        "empty_frames": int(empty_frames),
        "nonempty_frames": int(frames - empty_frames),
        "empty_rate": float(empty_frames / frames) if frames else 0.0,
        "category_totals": {
            key: int(category_totals[key])
            for key in category_keys
        },
        "total_boxes_in_bins": int(sum(category_totals.values())),
        "all_target_objects": int(all_target_objects),
        "effective_target_objects": int(effective_target_objects),
        "outside_bin_objects": int(outside_bin_objects),
        "frame_total_histogram": {
            str(total): int(count)
            for total, count in sorted(frame_histogram.items())
        },
    }


def _build_population(frame_infos):
    population = {key: [] for key in _category_keys_from_frames(frame_infos)}
    for frame_idx, frame_info in enumerate(frame_infos):
        for key, labels in frame_info["category_object_labels"].items():
            population.setdefault(key, [])
            population[key].extend(
                (int(frame_idx), int(object_label))
                for object_label in labels
            )
    return population


def _category_keys_from_frames(frame_infos):
    if not frame_infos:
        return ()
    return tuple(frame_infos[0]["category_counts"].keys())


def _run_trial(window_frame_infos, reference_summary, population, seed):
    rng = random.Random(int(seed))
    keep_by_frame = {
        frame_idx: set()
        for frame_idx in range(len(window_frame_infos))
    }

    requested_counts = reference_summary["category_totals"]
    actual_requested_counts = {}
    for category_key, objects in population.items():
        requested = int(requested_counts.get(category_key, 0))
        keep_count = min(requested, len(objects))
        actual_requested_counts[category_key] = keep_count
        for frame_idx, object_label in rng.sample(objects, keep_count):
            keep_by_frame[frame_idx].add(int(object_label))

    after_summary = _summarize_frames(
        window_frame_infos,
        kept_by_frame=keep_by_frame,
    )
    empty_delta = abs(
        int(after_summary["empty_frames"])
        - int(reference_summary["empty_frames"])
    )
    bbox_delta = sum(
        abs(
            int(after_summary["category_totals"].get(key, 0))
            - int(reference_summary["category_totals"].get(key, 0))
        )
        for key in reference_summary["category_totals"]
    )
    return {
        "seed": int(seed),
        "score": (int(empty_delta), int(bbox_delta), int(seed)),
        "keep_by_frame": keep_by_frame,
        "actual_requested_counts": actual_requested_counts,
        "after_summary": after_summary,
    }


def _build_override_frames(window_frame_infos, keep_by_frame):
    override_frames = {}
    frame_after_counts = {}
    for frame_idx, frame_info in enumerate(window_frame_infos):
        keep_set = set(keep_by_frame.get(frame_idx, set()))
        ignore_labels = [
            int(label)
            for label in frame_info["all_target_object_labels"]
            if int(label) not in keep_set
        ]
        if ignore_labels:
            override_frames[frame_info["frame_name"]] = {
                "ignore_object_labels": ignore_labels,
            }
        frame_after_counts[frame_info["frame_name"]] = {
            "total_kept": int(len(keep_set)),
            "category_counts": {
                key: int(
                    sum(
                        int(label) in keep_set
                        for label in labels
                    )
                )
                for key, labels in frame_info["category_object_labels"].items()
            },
        }
    return override_frames, frame_after_counts


def _format_rate(rate):
    return f"{float(rate) * 100.0:.2f}%"


def _comparison_text(pair_results, ridx_bins):
    lines = [
        "Controlled sequence comparison",
        "",
        "The before and after values refer to the selected continuous source window.",
        "Ignored bboxes are not removed from gt.txt; they are applied through the training ignore override.",
        "",
        "r_idx bins: " + ", ".join(
            f"[{_format_number(lower)},{_format_number(upper)})"
            for lower, upper in ridx_bins
        ),
    ]

    for result in pair_results:
        source = result["source_sequence"]
        reference = result["reference_sequence"]
        before = result["before_summary"]
        after = result["after_summary"]
        target = result["reference_summary"]
        lines.extend([
            "",
            f"Sequence pair: controlled seq{source} -> reference seq{reference}",
            f"Selected source window: file_idx {result['start_file_idx']} - {result['end_file_idx']}",
            f"Selected random seed: {result['selected_seed']}",
            "",
            "Metric | Before control | After control | Reference",
            "--- | ---: | ---: | ---:",
            f"Frames | {before['frames']} | {after['frames']} | {target['frames']}",
            f"Empty frames | {before['empty_frames']} | {after['empty_frames']} | {target['empty_frames']}",
            f"Empty rate | {_format_rate(before['empty_rate'])} | {_format_rate(after['empty_rate'])} | {_format_rate(target['empty_rate'])}",
        ])
        for class_name in TARGET_CLASS_NAMES:
            for lower, upper in ridx_bins:
                key = _bin_key(class_name, lower, upper)
                lines.append(
                    f"{class_name} bbox [{_format_number(lower)},{_format_number(upper)}) | "
                    f"{before['category_totals'].get(key, 0)} | "
                    f"{after['category_totals'].get(key, 0)} | "
                    f"{target['category_totals'].get(key, 0)}"
                )
        lines.extend([
            f"Total bbox in bins | {before['total_boxes_in_bins']} | {after['total_boxes_in_bins']} | {target['total_boxes_in_bins']}",
            f"Effective target bbox in bins | {before['effective_target_objects']} | {after['effective_target_objects']} | {target['effective_target_objects']}",
            f"Original target bbox in GT | {before['all_target_objects']} | {after['all_target_objects']} | {target['all_target_objects']}",
        ])

    return "\n".join(lines) + "\n"


def _pair_sequences(controlled_sequences, reference_sequences):
    controlled = _normalize_sequences(controlled_sequences, "controled_sequences")
    reference = _normalize_sequences(reference_sequences, "reference_sequences")
    if not controlled:
        raise ValueError("controled_sequences must contain at least one sequence")
    if not reference:
        raise ValueError("reference_sequences must contain at least one sequence")
    if len(reference) == 1:
        return tuple((source, reference[0]) for source in controlled)
    if len(controlled) != len(reference):
        raise ValueError(
            "reference_sequences must contain one sequence or the same number of "
            "sequences as controled_sequences"
        )
    return tuple(zip(controlled, reference))


def _requested_config(args):
    pairs = _pair_sequences(
        getattr(args, "controled_sequences", None),
        getattr(args, "reference_sequences", None),
    )
    bins = _normalize_bins(getattr(args, "control_ridx_bins", None))
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "pairs": [[int(source), int(reference)] for source, reference in pairs],
        "ridx_bins": [[float(lower), float(upper)] for lower, upper in bins],
        "window_position": str(getattr(args, "control_window_position", "last")),
        "seed": int(getattr(args, "seed", 42)),
        "num_trials": int(getattr(args, "control_num_trials", 300)),
    }, pairs, bins


def _request_signature(request):
    """Only compare settings that change the generated controlled data."""
    return {
        "pairs": request.get("pairs"),
        "ridx_bins": request.get("ridx_bins"),
        "window_position": request.get("window_position"),
        "seed": request.get("seed"),
        "num_trials": request.get("num_trials"),
    }


def _is_complete_matching_control_dir(candidate, request):
    if not candidate.is_dir():
        return False
    if any(
        not (candidate / filename).is_file()
        for filename in REQUIRED_CONTROL_OUTPUT_FILENAMES
    ):
        return False

    config_path = candidate / "control_config.json"
    if not config_path.is_file():
        return False
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return _request_signature(existing) == _request_signature(request)


def _find_matching_control_dir(base_dir, request):
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return None

    candidates = sorted(
        (config_path.parent for config_path in base_dir.rglob("control_config.json")),
        key=lambda path: str(path),
    )
    for candidate in candidates:
        if _is_complete_matching_control_dir(candidate, request):
            return candidate
    return None


def _select_output_dir(base_dir, request):
    matching_dir = _find_matching_control_dir(base_dir, request)
    if matching_dir is not None:
        return matching_dir, False

    pairs_text = "__".join(
        f"seq{source}_ref{reference}"
        for source, reference in request["pairs"]
    )
    base_dir = Path(base_dir)
    desired = base_dir / f"controled_{pairs_text}"
    candidate = desired
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{desired}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate, True


def prepare_controlled_train_data(args):
    """Create or reuse an automatic controlled split and attach its paths to args."""
    if not bool(getattr(args, "train_control_split_enabled", False)):
        return args

    train_sequences = _normalize_sequences(
        getattr(args, "train_sequences", None),
        "train_sequences",
    )
    request, pairs, ridx_bins = _requested_config(args)
    controlled_sequences = tuple(source for source, _ in pairs)
    missing = sorted(set(controlled_sequences) - set(train_sequences))
    if missing:
        raise ValueError(
            "controled_sequences must be included in train_sequences; "
            f"missing={missing}"
        )
    if request["window_position"] not in {"first", "last"}:
        raise ValueError("control_window_position must be 'first' or 'last'")
    if request["num_trials"] <= 0:
        raise ValueError("control_num_trials must be greater than 0")

    output_dir, should_generate = _select_output_dir(
        getattr(args, "controlled_split_base_dir", "split"),
        request,
    )
    if should_generate:
        pair_results = []
        override_sequences = {}
        for source_sequence, reference_sequence in pairs:
            source_infos = _build_frame_infos(source_sequence, ridx_bins)
            reference_infos = _build_frame_infos(reference_sequence, ridx_bins)
            if not source_infos or not reference_infos:
                raise ValueError(
                    f"Cannot control seq{source_sequence} -> seq{reference_sequence}: "
                    "one sequence has no frames"
                )

            window_length = min(len(source_infos), len(reference_infos))
            if request["window_position"] == "first":
                start_file_idx = 0
            else:
                start_file_idx = len(source_infos) - window_length
            end_file_idx = start_file_idx + window_length - 1
            window_infos = source_infos[start_file_idx:end_file_idx + 1]
            before_summary = _summarize_frames(window_infos)
            reference_summary = _summarize_frames(reference_infos)
            population = _build_population(window_infos)

            best_trial = None
            for trial_idx in range(request["num_trials"]):
                trial = _run_trial(
                    window_frame_infos=window_infos,
                    reference_summary=reference_summary,
                    population=population,
                    seed=request["seed"] + trial_idx,
                )
                if best_trial is None or trial["score"] < best_trial["score"]:
                    best_trial = trial

            override_frames, frame_after_counts = _build_override_frames(
                window_infos,
                best_trial["keep_by_frame"],
            )
            after_summary = best_trial["after_summary"]
            selected_frame_names = [
                frame_info["frame_name"]
                for frame_info in window_infos
            ]
            excluded_frame_names = [
                frame_info["frame_name"]
                for frame_info in source_infos
                if frame_info["frame_name"] not in set(selected_frame_names)
            ]
            override_sequences[str(int(source_sequence))] = {
                "matched_to_sequence": int(reference_sequence),
                "frame_overrides": override_frames,
                "frame_after_counts": frame_after_counts,
            }
            pair_results.append({
                "source_sequence": int(source_sequence),
                "reference_sequence": int(reference_sequence),
                "start_file_idx": int(start_file_idx),
                "end_file_idx": int(end_file_idx),
                "start_frame_name": selected_frame_names[0],
                "end_frame_name": selected_frame_names[-1],
                "selected_frame_names": selected_frame_names,
                "selected_seed": int(best_trial["seed"]),
                "before_summary": before_summary,
                "after_summary": after_summary,
                "reference_summary": reference_summary,
                "score": list(best_trial["score"]),
                "excluded_frame_names": excluded_frame_names,
            })

        override_payload = {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "experiment_name": "automatic_controled_sequences",
            "notes": (
                "Automatically generated training-only object ignore control. "
                "Original gt.txt files are not modified."
            ),
            "ridx_bins": [
                [float(lower), float(upper)]
                for lower, upper in ridx_bins
            ],
            "control_config": request,
            "sequences": override_sequences,
        }
        stats_payload = {
            "control_config": request,
            "pairs": pair_results,
        }
        (output_dir / "control_config.json").write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "object_ignore_override.json").write_text(
            json.dumps(override_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "stats.json").write_text(
            json.dumps(stats_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "comparison.txt").write_text(
            _comparison_text(pair_results, ridx_bins),
            encoding="utf-8",
        )
        train_lines = []
        test_lines = []
        for result in pair_results:
            selected_names = result["selected_frame_names"]
            excluded_names = result["excluded_frame_names"]
            train_lines.extend(
                f"{result['source_sequence']},{frame_name}.txt\n"
                for frame_name in selected_names
            )
            test_lines.extend(
                f"{result['source_sequence']},{frame_name}.txt\n"
                for frame_name in excluded_names
            )
        (output_dir / "train.txt").write_text("".join(train_lines), encoding="utf-8")
        (output_dir / "test.txt").write_text("".join(test_lines), encoding="utf-8")

        print(f"Generated controlled training split: {output_dir}")
        print(f"Comparison report: {output_dir / 'comparison.txt'}")
        for result in pair_results:
            after = result["after_summary"]
            reference = result["reference_summary"]
            print(
                f"  seq{result['source_sequence']} -> seq{result['reference_sequence']}: "
                f"frames {after['frames']}/{reference['frames']}, "
                f"empty rate {_format_rate(after['empty_rate'])}/{_format_rate(reference['empty_rate'])}"
            )
    else:
        print(f"Reusing controlled training split: {output_dir}")

    args.train_control_split_dir = str(output_dir)
    args.gt_object_ignore_override_path = str(
        output_dir / "object_ignore_override.json"
    )
    args.controlled_sequences = tuple(source for source, _ in pairs)
    args.reference_sequences = tuple(reference for _, reference in pairs)
    args.control_ridx_bins = tuple(tuple(pair) for pair in ridx_bins)
    return args
