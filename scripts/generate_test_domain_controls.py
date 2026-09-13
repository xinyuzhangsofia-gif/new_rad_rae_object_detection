#!/usr/bin/env python3
"""Build deterministic normal-weather controls for source-domain testing.

The generated controls are evaluation-only.  They never edit the K-Radar GT
files.  A strict ``test.txt`` selects the source frames and an object-ignore
override marks surplus *eligible Sedan* boxes as neutral regions.  The target
domain's fixed GT-distance quartile boundaries are retained so controlled
source and target AP can later be compared in exactly the same distance bins.

The experiment design is discovered from all five adverse-weather tables:
heavy snow, light snow, overcast, rain, and sleet.  For each distinct target
test set, every locally available same-road normal sequence is considered.
Frames used by *any* source-trained model in that weather group are removed,
and the remaining continuous candidates are ranked with the same count/range/
empty-frame priorities used by ``train_cfg`` controls.

Selection and masking are deterministic for seed 42.  If the source candidate
has more frames than the target, the closest continuous window is selected.
If it has more eligible Sedan boxes than the target, surplus source boxes are
masked.  A source shortage is reported and no box is invented or duplicated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.coordinates import ELEVATION_AXIS, RANGE_AXIS, AZIMUTH_AXIS, SCOPE_FULL
from configs.coordinates import BOX_COORDINATE_CARTESIAN
from configs.data import CARTESIAN_GT_ROOT
from data.dataset import KRadarGTDetectionDataset, KRadarRADRAEDataset
from data.geometry import (
    cartesian_box_overlaps_rae_fov,
    object_center_in_scope,
    prepare_cartesian_objects,
)
from eval.distance_quartiles import derive_gt_distance_quartile_bins
from data.paths import get_rad_rae_npy_root_dir
from configs.experiment_paths import (
    DISTANCE_QUARTILE_EXPERIMENT_DIR,
    SOURCE_DROP_EXPERIMENT_DIR,
    TARGET_DROP_EXPERIMENT_DIR,
)


SCHEMA_VERSION = 2
DEFAULT_SEED = 42
DEFAULT_CARTESIAN_GT_ROOT = Path(CARTESIAN_GT_ROOT)
DEFAULT_OUTPUT_DIR = SOURCE_DROP_EXPERIMENT_DIR / "control_specs"
DEFAULT_SEQUENCE_CSV = PROJECT_ROOT / "sequence_information.csv"
DEFAULT_QUARTILE_REPORT_ROOT = (
    DISTANCE_QUARTILE_EXPERIMENT_DIR / "evaluation_reports"
)

WEATHERS = ("heavy_snow", "light_snow", "overcast", "rain", "sleet")
CSV_WEATHER_BY_GROUP = {
    "heavy_snow": "heavysnow",
    "light_snow": "lightsnow",
    "overcast": "overcast",
    "rain": "rain",
    "sleet": "sleet",
}


def _sequence_tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    return tuple(int(token) for token in re.findall(r"\d+", str(value)))


def _test_set_tag(sequences):
    values = _sequence_tuple(sequences)
    if not values:
        raise ValueError(f"Empty target test sequence specification: {sequences!r}")
    return "_".join(str(value) for value in values)


def _control_id(weather_group, target_sequences):
    return f"{weather_group}_test_{_test_set_tag(target_sequences)}"

# Exact train_cfg control ranges.  The outside category preserves total-bbox
# control if an evaluation-eligible box center falls outside the configured
# [0, 120) metre histogram.
CONTROL_RANGE_BINS = (
    {"tag": "range_0_20", "lower_m": 0.0, "upper_m": 20.0},
    {"tag": "range_20_40", "lower_m": 20.0, "upper_m": 40.0},
    {"tag": "range_40_60", "lower_m": 40.0, "upper_m": 60.0},
    {"tag": "range_60_80", "lower_m": 60.0, "upper_m": 80.0},
    {"tag": "range_80_120", "lower_m": 80.0, "upper_m": 120.0},
    {"tag": "outside_120_inf", "lower_m": 120.0, "upper_m": math.inf},
)


def _json_safe(value):
    """Return a strict-JSON representation (infinity becomes ``"inf"``)."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_sequence_information(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = int(row["sequence_id"])
            rows[sequence] = {
                "sequence": sequence,
                "road_type": str(row["environment"]).strip().lower(),
                "weather": str(row["weather"]).strip().lower(),
                "time": str(row["time"]).strip().lower(),
                "csv_frame_count": int(row["frames"]),
            }
    return rows


def _full_rae_shape(radar_dataset):
    """Read only the first NPY header when possible; validate project axes."""
    try:
        shape = tuple(np.load(radar_dataset.rae_files[0], mmap_mode="r").shape)
    except (OSError, ValueError):
        shape = (RANGE_AXIS.size, AZIMUTH_AXIS.size, ELEVATION_AXIS.size)
    expected = (RANGE_AXIS.size, AZIMUTH_AXIS.size, ELEVATION_AXIS.size)
    if shape != expected:
        raise RuntimeError(
            f"Unexpected RAE tensor shape {shape}; evaluation expects {expected}."
        )
    return shape


def load_eligible_sedan_frames(sequence, cartesian_gt_root):
    """Load exactly the Sedan GT that reaches full-scope evaluation.

    This deliberately calls the dataset's Cartesian preparation and FOV
    predicates instead of maintaining a second approximation of those rules.
    Radar tensor payloads are not loaded.
    """
    radar_dataset = KRadarRADRAEDataset(
        get_rad_rae_npy_root_dir(), int(sequence), scope_mode=SCOPE_FULL
    )
    gt_dataset = KRadarGTDetectionDataset(
        radar_dataset=radar_dataset,
        sequence=int(sequence),
        class_to_idx={"Sedan": 0},
        ignore_unmapped_classes=True,
        ignore_class_names=(
            "Bus or Truck", "Pedestrian", "Pedestrian Group", "Bicycle",
            "Bicycle Group", "Motorcycle",
        ),
        gt_object_ignore_override_path=None,
        scope_mode=SCOPE_FULL,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
        cartesian_gt_root=str(cartesian_gt_root),
        ignore_object_label_minus_one=False,
        ignore_out_of_scope_gt=True,
    )
    rae_shape = _full_rae_shape(radar_dataset)
    frames = []
    for file_idx, frame_name in enumerate(radar_dataset.frame_names):
        if gt_dataset.gt_by_file_idx is not None:
            raw_objects = gt_dataset.gt_by_file_idx.get(file_idx, [])
        else:
            raw_objects = gt_dataset.gt_by_frame_name.get(frame_name, [])
        prepared = prepare_cartesian_objects(raw_objects, rae_shape)
        eligible = []
        for obj in prepared:
            if str(obj["cls"]) != "Sedan":
                continue
            if not cartesian_box_overlaps_rae_fov(obj):
                continue
            if not object_center_in_scope(obj, gt_dataset.scope_mode):
                continue
            metric_box = np.asarray(
                obj["box_metric"].detach().cpu(), dtype=np.float64
            ).reshape(7)
            eligible.append({
                "object_label": int(obj["object_label"]),
                "distance_m": float(np.linalg.norm(metric_box[:3])),
                "metric_box": metric_box,
            })
        frames.append({
            "sequence": int(sequence),
            "file_idx": int(file_idx),
            "frame_name": str(frame_name),
            "eligible_objects": eligible,
        })
    return frames


def load_eligible_sedan_frames_cached(args, sequence):
    cache = getattr(args, "_eligible_frame_cache", None)
    if cache is None:
        cache = {}
        args._eligible_frame_cache = cache
    sequence = int(sequence)
    if sequence not in cache:
        cache[sequence] = load_eligible_sedan_frames(
            sequence, args.cartesian_gt_root
        )
    return cache[sequence]


def frames_to_quartile_state(frames):
    metric_frames = []
    for frame in frames:
        boxes = [obj["metric_box"] for obj in frame["eligible_objects"]]
        metric_frames.append({
            "gt_boxes": (
                np.stack(boxes, axis=0)
                if boxes else np.zeros((0, 7), dtype=np.float64)
            ),
            "gt_labels": np.zeros((len(boxes),), dtype=np.int64),
            "dt_boxes": np.zeros((0, 7), dtype=np.float64),
            "dt_labels": np.zeros((0,), dtype=np.int64),
            "dt_scores": np.zeros((0,), dtype=np.float64),
        })
    return {"metric_frames": metric_frames}


def _normalize_bins(bins):
    normalized = []
    for index, item in enumerate(bins):
        upper = item["upper_m"]
        if isinstance(upper, str) and upper.lower() in {"inf", "infinity"}:
            upper = math.inf
        normalized.append({
            "tag": f"q{index + 1}",
            "lower_m": float(item["lower_m"]),
            "upper_m": float(upper),
            "bbox_count": int(item.get("bbox_count", item.get("num_gt", 0))),
        })
    return normalized


def load_report_quartile_bins(report_root, weather_group, target_sequences):
    target_tag = _test_set_tag(target_sequences)
    report_dir = (
        Path(report_root) / str(weather_group) / f"test_set_{target_tag}"
    )
    found = []
    for report_path in sorted(report_dir.rglob("*_result.txt")):
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("distance_quartile_bins:"):
                raw = line.split(":", 1)[1].strip()
                found.append((report_path, _normalize_bins(json.loads(raw))))
                break
    if not found:
        return None, []
    reference = found[0][1]
    for report_path, bins in found[1:]:
        for expected, actual in zip(reference, bins):
            same_bounds = math.isclose(
                expected["lower_m"], actual["lower_m"], abs_tol=1e-9
            ) and (
                (math.isinf(expected["upper_m"]) and math.isinf(actual["upper_m"]))
                or math.isclose(
                    expected["upper_m"], actual["upper_m"], abs_tol=1e-9
                )
            )
            if not same_bounds or expected["bbox_count"] != actual["bbox_count"]:
                raise RuntimeError(
                    f"Quartile metadata disagrees between reports for target "
                    f"test set {target_tag}: {report_path}"
                )
    return reference, [str(path.relative_to(PROJECT_ROOT)) for path, _ in found]


def crosscheck_quartile_bins(derived, reported, target_sequences):
    target_tag = _test_set_tag(target_sequences)
    if reported is None:
        return
    if len(derived) != len(reported):
        raise RuntimeError(f"Quartile count mismatch for target {target_tag}.")
    for expected, actual in zip(derived, reported):
        same_upper = (
            math.isinf(expected["upper_m"]) and math.isinf(actual["upper_m"])
        ) or math.isclose(expected["upper_m"], actual["upper_m"], abs_tol=1e-9)
        if not (
            math.isclose(expected["lower_m"], actual["lower_m"], abs_tol=1e-9)
            and same_upper
            and int(expected["bbox_count"]) == int(actual["bbox_count"])
        ):
            raise RuntimeError(
                f"Derived and recorded quartiles differ for target "
                f"test set {target_tag}: derived={derived}, reported={reported}"
            )


def bin_index(distance_m, bins):
    distance_m = float(distance_m)
    for index, item in enumerate(bins):
        if float(item["lower_m"]) <= distance_m < float(item["upper_m"]):
            return index
    raise RuntimeError(f"Distance {distance_m} is outside the quartile bins.")


def summarize_frames(frames, bins, kept_keys=None):
    kept_keys = None if kept_keys is None else set(kept_keys)
    counts = [0 for _ in bins]
    frame_histogram = Counter()
    for frame in frames:
        frame_count = 0
        for obj in frame["eligible_objects"]:
            key = (int(frame["file_idx"]), int(obj["object_label"]))
            if kept_keys is not None and key not in kept_keys:
                continue
            counts[bin_index(obj["distance_m"], bins)] += 1
            frame_count += 1
        frame_histogram[frame_count] += 1
    return {
        "frame_count": len(frames),
        "eligible_bbox_count": int(sum(counts)),
        "bin_counts": {
            str(item["tag"]): int(value)
            for item, value in zip(bins, counts)
        },
        "empty_frame_count": int(frame_histogram[0]),
        "frame_bbox_histogram": {
            str(key): int(value) for key, value in sorted(frame_histogram.items())
        },
    }


def select_continuous_window(candidate_frames, target_summary, bins, seed):
    """Select the closest target-length continuous source window."""
    desired_length = int(target_summary["frame_count"])
    if len(candidate_frames) <= desired_length:
        return list(candidate_frames), 0, "candidate_not_longer_than_target"

    rng = random.Random(int(seed))
    starts = list(range(0, len(candidate_frames) - desired_length + 1))
    tie_rank = {start: rank for rank, start in enumerate(rng.sample(starts, len(starts)))}
    tags = [str(item["tag"]) for item in bins]
    target_counts = [target_summary["bin_counts"][tag] for tag in tags]
    scored = []
    for start in starts:
        window = candidate_frames[start:start + desired_length]
        summary = summarize_frames(window, bins)
        source_counts = [summary["bin_counts"][tag] for tag in tags]
        controlled_counts = allocate_keep_counts(source_counts, target_counts)
        deltas = [
            abs(a - b) for a, b in zip(controlled_counts, target_counts)
        ]
        score = (
            abs(sum(controlled_counts) - target_summary["eligible_bbox_count"]),
            sum(deltas[:4]),
            sum(deltas[4:]),
            tuple(deltas),
            abs(summary["empty_frame_count"] - target_summary["empty_frame_count"]),
            tie_rank[start],
            start,
        )
        scored.append((score, start, window))
    _score, start, window = min(scored, key=lambda item: item[0])
    return list(window), int(start), "closest_continuous_window"


def allocate_keep_counts(source_counts, target_counts):
    """Allocate an exact keep total while staying closest to target bins."""
    source_counts = [int(value) for value in source_counts]
    target_counts = [int(value) for value in target_counts]
    source_total = sum(source_counts)
    target_total = sum(target_counts)
    if source_total <= target_total:
        return source_counts

    keep = [min(source, target) for source, target in zip(source_counts, target_counts)]
    remaining = target_total - sum(keep)
    # Match controlled_sequences.py: once reference-bin reservations are made,
    # fill any shortage strictly near-to-far, followed by the outside bin.
    for index in range(len(source_counts)):
        available = source_counts[index] - keep[index]
        added = min(remaining, available)
        keep[index] += added
        remaining -= added
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError("Source allocation cannot reach the requested keep total.")
    if sum(keep) != target_total:
        raise RuntimeError("Control allocation did not preserve the target total.")
    return keep


def allocate_keep_counts_with_minimum(source_counts, target_counts, minimum_counts):
    """Allocate keep counts while preserving GT boxes that cannot be masked.

    Negative or duplicate object labels cannot be targeted unambiguously by an
    object-label override.  They are therefore mandatory positives, not a
    reason to discard an otherwise better source sequence.
    """
    source_counts = [int(value) for value in source_counts]
    target_counts = [int(value) for value in target_counts]
    minimum_counts = [int(value) for value in minimum_counts]
    if not (
        len(source_counts) == len(target_counts) == len(minimum_counts)
    ):
        raise ValueError("source, target, and minimum counts must align")
    if any(
        minimum < 0 or minimum > source
        for source, minimum in zip(source_counts, minimum_counts)
    ):
        raise ValueError("minimum keep counts must lie within source counts")
    source_total = sum(source_counts)
    target_total = sum(target_counts)
    desired_total = min(source_total, target_total)
    if sum(minimum_counts) > desired_total:
        raise RuntimeError(
            "Unmaskable source GT count exceeds the controlled target total."
        )
    if source_total <= target_total:
        return source_counts

    keep = list(minimum_counts)
    remaining = desired_total - sum(keep)
    # First reserve each target bin above its mandatory floor.
    for index, (source, target) in enumerate(zip(source_counts, target_counts)):
        requested = max(0, target - keep[index])
        added = min(remaining, source - keep[index], requested)
        keep[index] += added
        remaining -= added
    # Match train_cfg's deterministic near-to-far fill for any shortages.
    for index, source in enumerate(source_counts):
        added = min(remaining, source - keep[index])
        keep[index] += added
        remaining -= added
        if remaining == 0:
            break
    if remaining or sum(keep) != desired_total:
        raise RuntimeError("Constrained source allocation cannot reach target total.")
    return keep


def _frame_histogram_distance(source_summary, target_summary):
    keys = set(source_summary["frame_bbox_histogram"]) | set(
        target_summary["frame_bbox_histogram"]
    )
    return sum(
        abs(
            int(source_summary["frame_bbox_histogram"].get(key, 0))
            - int(target_summary["frame_bbox_histogram"].get(key, 0))
        )
        for key in keys
    )


def select_kept_objects(source_frames, target_summary, bins, seed, num_trials=300):
    """Choose deterministic object labels to keep; return keys and audit data."""
    objects_by_bin = [[] for _ in bins]
    mandatory_keys = set()
    mandatory_counts = [0 for _ in bins]
    for frame in source_frames:
        labels = [int(obj["object_label"]) for obj in frame["eligible_objects"]]
        label_counts = Counter(labels)
        for obj in frame["eligible_objects"]:
            label = int(obj["object_label"])
            key = (int(frame["file_idx"]), label)
            bin_id = bin_index(obj["distance_m"], bins)
            if label < 0 or label_counts[label] != 1:
                mandatory_keys.add(key)
                mandatory_counts[bin_id] += 1
            else:
                objects_by_bin[bin_id].append(key)

    source_counts = [
        len(items) + mandatory
        for items, mandatory in zip(objects_by_bin, mandatory_counts)
    ]
    tags = [str(item["tag"]) for item in bins]
    target_counts = [target_summary["bin_counts"][tag] for tag in tags]
    allocations = allocate_keep_counts_with_minimum(
        source_counts, target_counts, mandatory_counts
    )
    if allocations == source_counts:
        keep = set(mandatory_keys)
        keep.update(key for items in objects_by_bin for key in items)
        return keep, {
            "selected_seed": int(seed),
            "num_trials": 0,
            "source_control_bin_counts": source_counts,
            "target_control_bin_counts": target_counts,
            "allocated_keep_counts": allocations,
            "mandatory_unmaskable_counts": mandatory_counts,
        }

    best = None
    for trial_offset in range(int(num_trials)):
        trial_seed = int(seed) + trial_offset
        rng = random.Random(trial_seed)
        keep = set(mandatory_keys)
        for objects, keep_count, mandatory_count in zip(
            objects_by_bin, allocations, mandatory_counts
        ):
            candidates = list(objects)
            rng.shuffle(candidates)
            keep.update(candidates[:keep_count - mandatory_count])
        summary = summarize_frames(source_frames, bins, kept_keys=keep)
        score = (
            abs(summary["empty_frame_count"] - target_summary["empty_frame_count"]),
            _frame_histogram_distance(summary, target_summary),
            trial_seed,
        )
        if best is None or score < best[0]:
            best = (score, keep, trial_seed)
    return best[1], {
        "selected_seed": int(best[2]),
        "num_trials": int(num_trials),
        "source_control_bin_counts": source_counts,
        "target_control_bin_counts": target_counts,
        "allocated_keep_counts": allocations,
        "mandatory_unmaskable_counts": mandatory_counts,
    }


def build_override(source_sequence, source_frames, kept_keys):
    """Build and validate a standard object-ignore override payload."""
    kept_keys = set(kept_keys)
    frame_overrides = {}
    masked_count = 0
    for frame in source_frames:
        labels = [int(obj["object_label"]) for obj in frame["eligible_objects"]]
        ignored = sorted(
            label for label in labels
            if (int(frame["file_idx"]), label) not in kept_keys
        )
        if any(label < 0 for label in ignored) or len(ignored) != len(set(ignored)):
            raise RuntimeError(
                f"Fail closed: invalid ignore labels for {source_sequence},"
                f"{frame['frame_name']}."
            )
        if ignored:
            frame_overrides[str(frame["frame_name"])] = {
                "ignore_object_labels": ignored,
            }
            masked_count += len(ignored)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "test_domain_control",
        "notes": (
            "Evaluation-only neutral regions. Surplus eligible source-domain "
            "Sedan boxes are ignored; original GT files are unchanged."
        ),
        "sequences": {
            str(int(source_sequence)): {
                "frame_overrides": frame_overrides,
            }
        },
    }
    return payload, int(masked_count)


def validate_override_and_frame_counts(source_frames, kept_keys, override_payload):
    """Fail closed unless every requested mask resolves to one eligible Sedan."""
    kept_keys = set(kept_keys)
    selected_by_name = {str(frame["frame_name"]): frame for frame in source_frames}
    if len(selected_by_name) != len(source_frames):
        raise RuntimeError("Selected source frames are not unique.")
    sequence_ids = {int(frame["sequence"]) for frame in source_frames}
    if len(sequence_ids) != 1:
        raise RuntimeError(f"Control contains multiple source sequences: {sequence_ids}")
    sequence = str(next(iter(sequence_ids)))
    sequence_payload = override_payload.get("sequences", {}).get(sequence)
    if sequence_payload is None:
        raise RuntimeError(f"Override is missing selected source sequence {sequence}.")
    frame_overrides = sequence_payload.get("frame_overrides")
    if not isinstance(frame_overrides, dict):
        raise RuntimeError("Override frame_overrides must be an object.")
    missing_override_frames = sorted(set(frame_overrides) - set(selected_by_name))
    if missing_override_frames:
        raise RuntimeError(
            f"Override references frames absent from the manifest: {missing_override_frames[:5]}"
        )

    total_masked = 0
    per_frame = {}
    for frame_name, frame in selected_by_name.items():
        file_idx = int(frame["file_idx"])
        eligible_labels = [
            int(obj["object_label"]) for obj in frame["eligible_objects"]
        ]
        ignored = list(
            frame_overrides.get(frame_name, {}).get("ignore_object_labels", ())
        )
        ignored = [int(label) for label in ignored]
        if len(ignored) != len(set(ignored)) or any(label < 0 for label in ignored):
            raise RuntimeError(
                f"Override labels are duplicate or negative in {sequence},{frame_name}."
            )
        for label in ignored:
            # Only uniquely labelled, non-negative Sedan boxes are maskable;
            # negative/duplicate labels were forced into the keep set.
            if eligible_labels.count(label) != 1:
                raise RuntimeError(
                    f"Override label {label} does not resolve uniquely to an "
                    f"eligible Sedan in {sequence},{frame_name}."
                )
        expected_ignored = sorted(
            label for label in eligible_labels if (file_idx, label) not in kept_keys
        )
        if ignored != expected_ignored:
            raise RuntimeError(
                f"Override labels do not equal the selected surplus in "
                f"{sequence},{frame_name}."
            )
        kept_count = len(eligible_labels) - len(ignored)
        per_frame[frame_name] = {
            "eligible_post_fov_before_control": len(eligible_labels),
            "kept_post_fov": int(kept_count),
            "masked_post_fov": len(ignored),
        }
        total_masked += len(ignored)
    return per_frame, int(total_masked)


def _parse_experiment_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("group ", "---", "average")):
            continue
        parts = stripped.split()
        if len(parts) < 6 or not re.fullmatch(r"group\d+", parts[0]):
            continue
        if parts[2] == "-" or parts[3] == "-" or parts[5] == "-":
            continue
        rows.append({
            "group": parts[0],
            "seed": int(parts[1]),
            "shared": parts[2],
            "source": parts[3],
            "test": _sequence_tuple(parts[5]),
        })
    return rows


def _part_frame_names(token, frame_names):
    match = re.fullmatch(r"(\d+)(?:_(first|last))?", token.strip())
    if match is None:
        raise ValueError(f"Invalid experiment sequence token: {token!r}")
    position = match.group(2) or "full"
    keep_size = max(1, int(math.ceil(len(frame_names) * 0.5)))
    if position == "first":
        selected = frame_names[:keep_size]
    elif position == "last":
        selected = frame_names[-keep_size:]
    else:
        selected = frame_names
    return int(match.group(1)), set(selected), position


def verify_no_training_overlap(weather_group, target_sequences, source_sequence,
                               source_frame_names):
    """Audit candidate frames against every source-trained row for the target."""
    rows = _parse_experiment_rows(
        TARGET_DROP_EXPERIMENT_DIR / f"{weather_group}_experiments.txt"
    )
    target_sequences = _sequence_tuple(target_sequences)
    relevant = [row for row in rows if row["test"] == target_sequences]
    if not relevant:
        raise RuntimeError(
            f"No experiment rows found for {weather_group} target "
            f"{_test_set_tag(target_sequences)}."
        )
    source_frame_names = set(source_frame_names)
    source_all_names = [
        path.stem for path in sorted(
            (Path(get_rad_rae_npy_root_dir()) / str(source_sequence) / "rad").glob("*.npy")
        )
        if (Path(get_rad_rae_npy_root_dir()) / str(source_sequence) / "rae" / path.name).is_file()
    ]
    audits = []
    for row in relevant:
        trained_from_source_sequence = set()
        trained_parts = []
        for cell in (row["shared"], row["source"]):
            for token in cell.split(","):
                sequence_token = int(token.split("_", 1)[0])
                if sequence_token != int(source_sequence):
                    continue
                _sequence, names, position = _part_frame_names(token, source_all_names)
                trained_from_source_sequence.update(names)
                trained_parts.append(position)
        overlap = sorted(source_frame_names & trained_from_source_sequence)
        if overlap:
            raise RuntimeError(
                f"Source test leakage for {weather_group} {row['group']} seed "
                f"{row['seed']}: {len(overlap)} overlapping frames, first={overlap[:5]}"
            )
        audits.append({
            "group": row["group"],
            "seed": row["seed"],
            "trained_parts_of_control_sequence": trained_parts,
            "overlap_frame_count": 0,
        })
    return audits


def _paired_radar_frame_names(sequence):
    sequence_root = Path(get_rad_rae_npy_root_dir()) / str(int(sequence))
    rad_names = {path.stem for path in (sequence_root / "rad").glob("*.npy")}
    rae_names = {path.stem for path in (sequence_root / "rae").glob("*.npy")}
    return sorted(rad_names & rae_names)


def _continuous_available_runs(frame_names, blocked_names):
    blocked_names = set(blocked_names)
    runs = []
    start = None
    for index, frame_name in enumerate(frame_names):
        available = frame_name not in blocked_names
        if available and start is None:
            start = index
        if start is not None and (not available or index == len(frame_names) - 1):
            end = index if available and index == len(frame_names) - 1 else index - 1
            runs.append((start, end))
            start = None
    return runs


def _weather_source_training_tokens(weather_group):
    rows = _parse_experiment_rows(
        TARGET_DROP_EXPERIMENT_DIR / f"{weather_group}_experiments.txt"
    )
    tokens_by_sequence = {}
    for row in rows:
        for cell in (row["shared"], row["source"]):
            for token in cell.split(","):
                sequence = int(token.split("_", 1)[0])
                tokens_by_sequence.setdefault(sequence, set()).add(token)
    return rows, tokens_by_sequence


def audit_same_road_normal_candidates(
        weather_group,
        target_meta,
        target_control_summary,
        sequence_info,
        args,
        selected_source_sequence=None,
        selected_candidate_start=None,
        selected_candidate_end=None,
        ):
    """Rank every locally available, leakage-free same-road normal candidate."""
    _rows, training_tokens = _weather_source_training_tokens(weather_group)
    audit_rows = []
    viable = []
    for sequence, metadata in sorted(sequence_info.items()):
        if metadata["weather"] != "normal" or metadata["road_type"] != target_meta["road_type"]:
            continue
        frame_names = _paired_radar_frame_names(sequence)
        row = {
            "sequence": int(sequence),
            "weather": metadata["weather"],
            "road_type": metadata["road_type"],
            "time": metadata["time"],
            "paired_radar_frame_count": len(frame_names),
            "trained_part_tokens_across_weather_group": sorted(
                training_tokens.get(sequence, ())
            ),
        }
        if not frame_names:
            row.update({
                "status": "rejected",
                "rejection_reason": "paired_RAD_RAE_data_not_available_locally",
                "available_held_out_frame_count": 0,
            })
            audit_rows.append(row)
            continue

        blocked = set()
        for token in training_tokens.get(sequence, ()):
            _sequence, names, _position = _part_frame_names(token, frame_names)
            blocked.update(names)
        runs = _continuous_available_runs(frame_names, blocked)
        row["source_trained_frame_count"] = len(blocked)
        row["available_held_out_frame_count"] = len(frame_names) - len(blocked)
        if not runs:
            row.update({
                "status": "rejected",
                "rejection_reason": "all_frames_overlap_source_training",
            })
            audit_rows.append(row)
            continue

        # A control is one continuous domain sample. Prefer the longest
        # leakage-free run; use its earlier start only as a stable final tie.
        run_start, run_end = min(
            runs, key=lambda pair: (-(pair[1] - pair[0] + 1), pair[0])
        )
        all_frames = load_eligible_sedan_frames_cached(args, sequence)
        if [frame["frame_name"] for frame in all_frames] != frame_names:
            raise RuntimeError(
                f"Frame-order mismatch while auditing source candidate seq{sequence}."
            )
        held_out_frames = all_frames[run_start:run_end + 1]
        controlled_window, relative_start, window_reason = select_continuous_window(
            held_out_frames,
            target_control_summary,
            CONTROL_RANGE_BINS,
            args.seed,
        )
        window_start = run_start + relative_start
        window_end = window_start + len(controlled_window) - 1
        before = summarize_frames(controlled_window, CONTROL_RANGE_BINS)
        try:
            kept, _selection = select_kept_objects(
                controlled_window,
                target_control_summary,
                CONTROL_RANGE_BINS,
                args.seed,
                args.num_trials,
            )
        except RuntimeError as exc:
            # Object-label overrides must identify a unique non-negative label.
            # A candidate that cannot be masked safely is ineligible; it must
            # not abort the audit of other same-road normal candidates.
            row.update({
                "status": "rejected",
                "rejection_reason": f"unsafe_object_override_labels: {exc}",
                "held_out_run_index_start": int(run_start),
                "held_out_run_index_end": int(run_end),
                "held_out_run_frame_count": int(run_end - run_start + 1),
            })
            audit_rows.append(row)
            continue
        after = summarize_frames(
            controlled_window, CONTROL_RANGE_BINS, kept_keys=kept
        )
        tags = [item["tag"] for item in CONTROL_RANGE_BINS]
        deltas = [
            abs(
                int(after["bin_counts"][tag])
                - int(target_control_summary["bin_counts"][tag])
            )
            for tag in tags
        ]
        score = (
            abs(
                int(after["eligible_bbox_count"])
                - int(target_control_summary["eligible_bbox_count"])
            ),
            sum(deltas[:4]),
            sum(deltas[4:]),
            tuple(deltas),
            abs(
                int(after["empty_frame_count"])
                - int(target_control_summary["empty_frame_count"])
            ),
            abs(len(controlled_window) - int(target_control_summary["frame_count"])),
            int(sequence),
        )
        row.update({
            "status": "viable",
            "rejection_reason": None,
            "held_out_run_index_start": int(run_start),
            "held_out_run_index_end": int(run_end),
            "held_out_run_frame_count": int(run_end - run_start + 1),
            "controlled_window_index_start": int(window_start),
            "controlled_window_index_end": int(window_end),
            "controlled_window_frame_count": len(controlled_window),
            "window_selection_reason": window_reason,
            "eligible_bbox_before_control": before["eligible_bbox_count"],
            "eligible_bbox_after_control": after["eligible_bbox_count"],
            "masked_bbox_count": (
                before["eligible_bbox_count"] - after["eligible_bbox_count"]
            ),
            "bbox_deficit": max(
                0,
                target_control_summary["eligible_bbox_count"]
                - after["eligible_bbox_count"],
            ),
            "control_range_counts_before": before["bin_counts"],
            "control_range_counts_after": after["bin_counts"],
            "control_score": [
                score[0], score[1], score[2], list(score[3]),
                score[4], score[5], score[6],
            ],
        })
        audit_rows.append(row)
        viable.append((score, row))

    if not viable:
        raise RuntimeError(
            f"No leakage-free normal {target_meta['road_type']} source candidate."
        )
    viable.sort(key=lambda item: item[0])
    for rank, (_score, row) in enumerate(viable, start=1):
        row["rank"] = rank
        row["selected"] = rank == 1
        if rank > 1:
            row["status"] = "rejected"
            row["rejection_reason"] = "higher_train_cfg_control_score"
    winner = viable[0][1]
    actual = (
        int(winner["sequence"]),
        int(winner["held_out_run_index_start"]),
        int(winner["held_out_run_index_end"]),
    )
    if selected_source_sequence is not None:
        expected = (
            int(selected_source_sequence),
            int(selected_candidate_start),
            int(selected_candidate_end),
        )
        if actual != expected:
            raise RuntimeError(
                f"Configured source candidate {expected} is not the deterministic "
                f"best same-road normal candidate {actual}."
            )
    return {
        "ranking_objective": (
            "train_cfg-compatible lexicographic score: absolute controlled total "
            "bbox delta; 0-80 m histogram delta; 80-120/outside delta; "
            "individual range deltas; empty-frame delta; frame-count delta"
        ),
        "scope": "all normal-weather sequences of the same road type in sequence_information.csv",
        "weather_group_training_union_used_for_leakage": str(weather_group),
        "selected_sequence": int(winner["sequence"]),
        "selected_candidate_start_index": int(actual[1]),
        "selected_candidate_end_index": int(actual[2]),
        "selected_rank": 1,
        "candidates": audit_rows,
    }


def _comparison_text(stats):
    target = stats["target"]
    source = stats["source"]
    lines = [
        "Test-domain control comparison",
        "",
        f"Weather experiment: {stats['weather_group']}",
        f"Target test: seq{target['sequence_tag']} ({target['weather']}, "
        f"{target['road_type']}, {target['time']})",
        f"Controlled source test: seq{source['sequence']} ({source['weather']}, "
        f"{source['road_type']}, {source['time']})",
        f"Road type match: {stats['validation']['same_road_type']}",
        f"Source normal weather: {stats['validation']['source_is_normal_weather']}",
        f"Training-frame overlap: {stats['validation']['training_frame_overlap_count']}",
        "",
        "Surplus source GT boxes are neutral ignore regions: they are neither "
        "positive GT nor false-positive background. Original GT is unchanged.",
        "",
        "Metric | Target | Source before | Source after",
        "--- | ---: | ---: | ---:",
        f"Frames | {target['eligible_frame_count']} | {source['selected_frame_count']} | "
        f"{source['selected_frame_count']}",
        f"Eligible Sedan bbox | {target['eligible_bbox_count']} | "
        f"{source['eligible_bbox_before_control']} | {source['eligible_bbox_after_control']}",
    ]
    for item in CONTROL_RANGE_BINS:
        tag = item["tag"]
        lines.append(
            f"Control-bin Sedan {tag} | {target['control_range_counts'][tag]} | "
            f"{source['control_range_counts_before'][tag]} | "
            f"{source['control_range_counts_after'][tag]}"
        )
    lines.append("")
    lines.append("Reporting-only counts under the target's fixed AP quartiles:")
    for tag in ("q1", "q2", "q3", "q4"):
        lines.append(
            f"Sedan bbox {tag.upper()} | {target['quartile_counts'][tag]} | "
            f"{source['quartile_counts_before'][tag]} | "
            f"{source['quartile_counts_after'][tag]}"
        )
    lines.extend([
        f"Masked surplus bbox | 0 | 0 | {source['masked_eligible_bbox_count']}",
        f"Unrecoverable bbox deficit | 0 | 0 | {source['deficit_bbox_count']}",
        "",
        "Target-fixed quartile bounds:",
    ])
    for item in target["quartile_bins"]:
        upper = "inf" if math.isinf(float(item["upper_m"])) else f"{item['upper_m']:.6f}"
        lines.append(
            f"- {item['tag'].upper()}: [{item['lower_m']:.6f}, {upper}) m; "
            f"target N={item['bbox_count']}"
        )
    return "\n".join(lines) + "\n"


def discover_control_definitions():
    """Return one fixed-control definition for each distinct weather test set."""
    definitions = []
    for weather_group in WEATHERS:
        rows = _parse_experiment_rows(
            TARGET_DROP_EXPERIMENT_DIR / f"{weather_group}_experiments.txt"
        )
        seen = set()
        for row in rows:
            target_sequences = tuple(row["test"])
            if target_sequences in seen:
                continue
            seen.add(target_sequences)
            definitions.append({
                "control_id": _control_id(weather_group, target_sequences),
                "weather_group": weather_group,
                "target_sequences": target_sequences,
            })
    return definitions


def _combined_target_metadata(weather_group, target_sequences, sequence_info):
    target_sequences = _sequence_tuple(target_sequences)
    metadata = [sequence_info[sequence] for sequence in target_sequences]
    expected_weather = CSV_WEATHER_BY_GROUP[weather_group]
    bad_weather = [
        (item["sequence"], item["weather"])
        for item in metadata if item["weather"] != expected_weather
    ]
    if bad_weather:
        raise RuntimeError(
            f"Target weather mismatch for {weather_group}: {bad_weather!r}."
        )
    road_types = {item["road_type"] for item in metadata}
    if len(road_types) != 1:
        raise RuntimeError(
            f"A controlled test set must have one road type, got {road_types!r}."
        )
    times = sorted({item["time"] for item in metadata})
    return {
        "sequences": list(target_sequences),
        "sequence_tag": _test_set_tag(target_sequences),
        "road_type": next(iter(road_types)),
        "weather": expected_weather,
        "weather_group": weather_group,
        "time": times[0] if len(times) == 1 else "+".join(times),
        "csv_frame_count": sum(item["csv_frame_count"] for item in metadata),
        "per_sequence_metadata": metadata,
    }


def generate_control(definition, sequence_info, args):
    weather_group = str(definition["weather_group"])
    target_sequences = _sequence_tuple(definition["target_sequences"])
    target_tag = _test_set_tag(target_sequences)
    control_id = str(
        definition.get("control_id", _control_id(weather_group, target_sequences))
    )
    target_meta = _combined_target_metadata(
        weather_group, target_sequences, sequence_info
    )

    target_frames = []
    for target_sequence in target_sequences:
        target_frames.extend(
            load_eligible_sedan_frames_cached(args, target_sequence)
        )

    derived_bins = [dict(item) for item in derive_gt_distance_quartile_bins(
        frames_to_quartile_state(target_frames)
    )]
    reported_bins, report_paths = load_report_quartile_bins(
        args.quartile_report_root, weather_group, target_sequences
    )
    crosscheck_quartile_bins(derived_bins, reported_bins, target_sequences)
    quartile_bins = derived_bins
    target_quartile_summary = summarize_frames(target_frames, quartile_bins)
    if [target_quartile_summary["bin_counts"][f"q{i}"] for i in range(1, 5)] != [
        item["bbox_count"] for item in quartile_bins
    ]:
        raise RuntimeError(f"Target quartile recount failed for seq{target_tag}.")
    target_control_summary = summarize_frames(target_frames, CONTROL_RANGE_BINS)
    candidate_selection_audit = audit_same_road_normal_candidates(
        weather_group=weather_group,
        target_meta=target_meta,
        target_control_summary=target_control_summary,
        sequence_info=sequence_info,
        args=args,
    )

    source_sequence = int(candidate_selection_audit["selected_sequence"])
    candidate_start = int(
        candidate_selection_audit["selected_candidate_start_index"]
    )
    candidate_end = int(
        candidate_selection_audit["selected_candidate_end_index"]
    )
    source_meta = sequence_info[source_sequence]
    if source_meta["weather"] != "normal":
        raise RuntimeError(f"Control source seq{source_sequence} is not normal weather.")
    if source_meta["road_type"] != target_meta["road_type"]:
        raise RuntimeError(
            f"Road mismatch: target={target_meta['road_type']}, "
            f"source={source_meta['road_type']}."
        )
    source_all = load_eligible_sedan_frames_cached(args, source_sequence)
    if (
        candidate_start < 0
        or candidate_end >= len(source_all)
        or candidate_end < candidate_start
    ):
        raise RuntimeError(
            f"Invalid source candidate indices {candidate_start}..{candidate_end} "
            f"for seq{source_sequence} with {len(source_all)} frames."
        )
    candidate_frames = source_all[candidate_start:candidate_end + 1]

    selected_frames, relative_start, selection_reason = select_continuous_window(
        candidate_frames, target_control_summary, CONTROL_RANGE_BINS, args.seed
    )
    selected_start_index = candidate_start + relative_start
    selected_end_index = selected_start_index + len(selected_frames) - 1
    source_before = summarize_frames(selected_frames, CONTROL_RANGE_BINS)
    kept_keys, selection_audit = select_kept_objects(
        selected_frames,
        target_control_summary,
        CONTROL_RANGE_BINS,
        args.seed,
        args.num_trials,
    )
    source_after = summarize_frames(
        selected_frames, CONTROL_RANGE_BINS, kept_keys=kept_keys
    )
    source_before_quartile = summarize_frames(selected_frames, quartile_bins)
    source_after_quartile = summarize_frames(
        selected_frames, quartile_bins, kept_keys=kept_keys
    )
    override, masked_count = build_override(source_sequence, selected_frames, kept_keys)
    per_frame_control_counts, validated_masked_count = (
        validate_override_and_frame_counts(selected_frames, kept_keys, override)
    )
    if validated_masked_count != masked_count:
        raise RuntimeError(
            "Validated override object count differs from generated mask count."
        )
    if source_before["eligible_bbox_count"] - source_after["eligible_bbox_count"] != masked_count:
        raise RuntimeError("Override masked-count validation failed.")
    expected_after_total = min(
        source_before["eligible_bbox_count"],
        target_control_summary["eligible_bbox_count"],
    )
    if source_after["eligible_bbox_count"] != expected_after_total:
        raise RuntimeError("Controlled source total does not equal min(source, target).")

    selected_names = [frame["frame_name"] for frame in selected_frames]
    if len(selected_names) != len(set(selected_names)):
        raise RuntimeError("Duplicate source frame names in test control manifest.")
    overlap_audits = verify_no_training_overlap(
        weather_group, target_sequences, source_sequence, selected_names
    )
    source_frame_name_set = {frame["frame_name"] for frame in source_all}
    missing_manifest_frames = sorted(set(selected_names) - source_frame_name_set)
    override_frame_names = set(
        override["sequences"][str(source_sequence)]["frame_overrides"]
    )
    missing_override_frames = sorted(override_frame_names - source_frame_name_set)
    if missing_manifest_frames or missing_override_frames:
        raise RuntimeError(
            "Control contains missing frames: "
            f"manifest={missing_manifest_frames[:5]}, "
            f"override={missing_override_frames[:5]}"
        )
    output_dir = Path(args.output_dir) / control_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "test.txt"
    manifest_path.write_text(
        "".join(f"{source_sequence},{frame_name}.txt\n" for frame_name in selected_names),
        encoding="utf-8",
    )
    override_path = output_dir / "object_ignore_override.json"
    _write_json(override_path, override)

    deficit = max(
        0,
        target_control_summary["eligible_bbox_count"]
        - source_after["eligible_bbox_count"],
    )
    rel = lambda path: str(Path(path).resolve().relative_to(PROJECT_ROOT))
    stats = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(args.seed),
        "control_id": control_id,
        "weather_group": weather_group,
        "status": "exact_bbox_total" if deficit == 0 else "source_bbox_deficit",
        "manifest_path": rel(manifest_path),
        "object_override_path": rel(override_path),
        "expected_override_object_count": masked_count,
        "control_range_bins": CONTROL_RANGE_BINS,
        "target_quartile_bins": quartile_bins,
        "normal_same_road_candidate_selection": candidate_selection_audit,
        "target": {
            **target_meta,
            "eligible_frame_count": target_control_summary["frame_count"],
            "eligible_bbox_count": target_control_summary["eligible_bbox_count"],
            "empty_frame_count": target_control_summary["empty_frame_count"],
            "control_range_counts": target_control_summary["bin_counts"],
            "quartile_bins": quartile_bins,
            "quartile_counts": target_quartile_summary["bin_counts"],
            "quartile_crosscheck_reports": report_paths,
            "quartile_crosscheck_status": (
                "matched" if reported_bins is not None else "not_yet_available_derived_identically"
            ),
        },
        "source": {
            **source_meta,
            "candidate_index_start": candidate_start,
            "candidate_index_end": candidate_end,
            "candidate_frame_count": len(candidate_frames),
            "selected_index_start": selected_start_index,
            "selected_index_end": selected_end_index,
            "selected_frame_start": selected_names[0],
            "selected_frame_end": selected_names[-1],
            "selected_frame_count": len(selected_frames),
            "selection_reason": selection_reason,
            "eligible_bbox_before_control": source_before["eligible_bbox_count"],
            "eligible_bbox_after_control": source_after["eligible_bbox_count"],
            "masked_eligible_bbox_count": masked_count,
            "expected_override_object_count": masked_count,
            "deficit_bbox_count": int(deficit),
            "empty_frame_count_before": source_before["empty_frame_count"],
            "empty_frame_count_after": source_after["empty_frame_count"],
            "control_range_counts_before": source_before["bin_counts"],
            "control_range_counts_after": source_after["bin_counts"],
            "quartile_counts_before": source_before_quartile["bin_counts"],
            "quartile_counts_after": source_after_quartile["bin_counts"],
            "per_frame_post_fov_counts": per_frame_control_counts,
            "mask_selection": selection_audit,
        },
        "validation": {
            "same_road_type": True,
            "source_is_normal_weather": True,
            "strict_manifest_format": "sequence,frame.txt",
            "manifest_duplicate_count": 0,
            "missing_manifest_frame_count": 0,
            "missing_override_frame_count": 0,
            "training_frame_overlap_count": 0,
            "expected_override_object_count": masked_count,
            "validated_override_object_count": validated_masked_count,
            "source_training_overlap_audits": overlap_audits,
            "eligible_gt_rule": (
                "Sedan; Cartesian GT; bbox overlaps full RAE FOV; center in full scope; "
                "same rule as evaluation dataset"
            ),
            "override_labels": (
                "only unique nonnegative Sedan labels are maskable; negative or "
                "duplicate labels are mandatory retained positives"
            ),
        },
    }
    stats_path = output_dir / "stats.json"
    comparison_path = output_dir / "comparison.txt"
    readme_path = output_dir / "README.txt"
    _write_json(stats_path, stats)
    comparison_path.write_text(_comparison_text(stats), encoding="utf-8")
    readme_path.write_text(
        "Evaluation-only test-domain control.\n\n"
        "Use test.txt as the strict file split and object_ignore_override.json "
        "as the evaluation GT neutral-region override. Extra eligible source "
        "Sedan boxes are ignored, not counted as positives or negatives.\n",
        encoding="utf-8",
    )
    return stats


def build_index(all_stats, output_dir):
    controls = {}
    for stats in all_stats:
        target = stats["target"]
        source = stats["source"]
        control_id = stats["control_id"]
        stats_path = Path(output_dir) / control_id / "stats.json"
        controls[control_id] = {
            "control_id": control_id,
            "weather_group": stats["weather_group"],
            "target_sequences": target["sequences"],
            "target_sequence_tag": target["sequence_tag"],
            "source_sequence": source["sequence"],
            "source_sequences": [source["sequence"]],
            "manifest_path": stats["manifest_path"],
            "object_override_path": stats["object_override_path"],
            "stats_path": str(stats_path.resolve().relative_to(PROJECT_ROOT)),
            "status": stats["status"],
            "source_frames_selected": source["selected_frame_count"],
            "target_eligible_bbox_count": target["eligible_bbox_count"],
            "source_kept_bbox_count": source["eligible_bbox_after_control"],
            "source_masked_bbox_count": source["masked_eligible_bbox_count"],
            "expected_override_object_count": stats[
                "expected_override_object_count"
            ],
            "source_bbox_deficit": source["deficit_bbox_count"],
            "target_quartile_bins": stats["target_quartile_bins"],
            "selection_rationale": stats[
                "normal_same_road_candidate_selection"
            ]["ranking_objective"],
            "normal_same_road_candidates": [
                {
                    key: candidate.get(key)
                    for key in (
                        "sequence", "paired_radar_frame_count",
                        "available_held_out_frame_count", "status",
                        "rejection_reason", "rank", "selected",
                        "eligible_bbox_before_control",
                        "eligible_bbox_after_control", "bbox_deficit",
                        "control_score",
                    )
                }
                for candidate in stats[
                    "normal_same_road_candidate_selection"
                ]["candidates"]
            ],
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": DEFAULT_SEED,
        "description": (
            "Fixed normal-weather test-domain controls shared across all seeds/groups."
        ),
        "selection_method": (
            "Enumerate all sequence_information.csv normal-weather sequences "
            "with the target road type; reject missing data and every frame "
            "used by any source-trained checkpoint in the weather group; rank "
            "remaining continuous candidates with train_cfg total/range/empty "
            "control priorities."
        ),
        "controls": controls,
    }
    _write_json(Path(output_dir) / "index.json", payload)
    lines = [
        "Test-domain control specifications",
        "",
        payload["selection_method"],
        "",
        "Selected fixed controls:",
    ]
    for control_id, item in sorted(controls.items()):
        lines.append(
            f"- {control_id}: target seq{item['target_sequence_tag']} <- normal source "
            f"seq{item['source_sequence']}: frames={item['source_frames_selected']}, "
            f"kept_bbox={item['source_kept_bbox_count']}, "
            f"masked_bbox={item['source_masked_bbox_count']}, "
            f"deficit={item['source_bbox_deficit']}"
        )
    lines.extend([
        "",
        "Each weather_test_*/test.txt uses strict sequence,frame.txt syntax. "
        "Each object_ignore_override.json makes surplus source Sedan boxes "
        "neutral (neither positives nor false-positive background).",
        "",
        "See index.json and each stats.json/comparison.txt for the complete "
        "candidate ranking, post-FOV counts, target quartiles, and leakage audit.",
    ])
    (Path(output_dir) / "README.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-information", type=Path, default=DEFAULT_SEQUENCE_CSV)
    parser.add_argument("--cartesian-gt-root", type=Path, default=DEFAULT_CARTESIAN_GT_ROOT)
    parser.add_argument("--quartile-report-root", type=Path, default=DEFAULT_QUARTILE_REPORT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-trials", type=int, default=300)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seed != DEFAULT_SEED:
        raise ValueError("This experiment design requires deterministic seed 42.")
    if args.num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_info = load_sequence_information(args.sequence_information)
    stats = []
    definitions = discover_control_definitions()
    if len(definitions) != 17:
        raise RuntimeError(
            f"Expected 17 distinct all-weather target test sets, got "
            f"{len(definitions)}: {definitions!r}"
        )
    for definition in definitions:
        result = generate_control(definition, sequence_info, args)
        stats.append(result)
        source = result["source"]
        target = result["target"]
        print(
            f"{result['control_id']}: target seq{target['sequence_tag']} "
            f"<- source seq{source['sequence']}: "
            f"frames={source['selected_frame_count']}, "
            f"bbox {source['eligible_bbox_before_control']} -> "
            f"{source['eligible_bbox_after_control']} "
            f"(target={target['eligible_bbox_count']}, "
            f"masked={source['masked_eligible_bbox_count']}, "
            f"deficit={source['deficit_bbox_count']})"
        )
    build_index(stats, args.output_dir)
    print(f"Wrote test-domain control specs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
