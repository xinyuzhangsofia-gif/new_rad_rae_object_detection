"""Generate non-destructive, distribution-controlled training sequence splits."""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

from ..coordinates import cartesian_to_rae
from configs.coordinates import BOX_COORDINATE_CARTESIAN, require_cartesian_data
from ..dataset import KRadarRADRAEDataset
from ..paths import get_rad_rae_npy_root_dir
from ..labels import load_cartesian_gt
from .manifests import (
    build_sequence_index_lookup,
    read_split_file,
    split_entries_to_indices,
)
from .sequences import (
    _normalize_sequence_parts,
    _normalize_sequences,
    _pair_sequence_parts,
    _select_sequence_part,
    _sequence_part_label,
)


SEDAN_CLASS_NAME = "Sedan"
BUS_CLASS_NAME = "Bus or Truck"
SUPPORTED_CONTROL_CLASS_NAMES = (SEDAN_CLASS_NAME, BUS_CLASS_NAME)
DEFAULT_CONTROL_CLASS_NAMES = (SEDAN_CLASS_NAME,)
DEFAULT_RANGE_M_BINS = (
    (0.0, 20.0),
    (20.0, 40.0),
    (40.0, 60.0),
    (60.0, 80.0),
    (80.0, 120.0),
)
CONTROL_SCHEMA_VERSION = 10
OUTSIDE_RANGE_CATEGORY_KEY = "__outside_range__"
REQUIRED_CONTROL_OUTPUT_FILENAMES = (
    "train.txt",
    "test.txt",
    "object_ignore_override.json",
    "stats.json",
    "comparison.txt",
)


def _normalize_bins(value):
    if value is None:
        return DEFAULT_RANGE_M_BINS

    bins = tuple((float(pair[0]), float(pair[1])) for pair in value)
    if len(bins) == 0:
        raise ValueError("control_range_m_bins must not be empty")

    previous_upper = None
    for lower, upper in bins:
        if lower < 0.0 or upper <= lower:
            raise ValueError(f"Invalid control range bin: {(lower, upper)!r}")
        if previous_upper is not None and lower != previous_upper:
            raise ValueError("control_range_m_bins must be contiguous")
        previous_upper = upper
    return bins


def _format_number(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _bin_key(class_name, lower, upper):
    return f"{class_name.lower().replace(' ', '_')}_range_m_{_format_number(lower)}_{_format_number(upper)}"


def _normalize_control_class_names(value):
    if value is None:
        return DEFAULT_CONTROL_CLASS_NAMES
    if isinstance(value, str):
        names = tuple(
            token.strip()
            for token in value.split(",")
            if token.strip()
        )
    else:
        names = tuple(str(name).strip() for name in value if str(name).strip())
    if not names:
        raise ValueError("control_class_names must not be empty")
    invalid = [
        name
        for name in names
        if name not in SUPPORTED_CONTROL_CLASS_NAMES
    ]
    if invalid:
        raise ValueError(
            "control_class_names contains unsupported classes: "
            f"{invalid}; supported={SUPPORTED_CONTROL_CLASS_NAMES}"
        )
    return tuple(dict.fromkeys(names))


def _category_keys(
        range_m_bins,
        control_class_names=DEFAULT_CONTROL_CLASS_NAMES,
    ):
    return tuple(
        _bin_key(class_name, lower, upper)
        for class_name in control_class_names
        for lower, upper in range_m_bins
    )


def _category_key(
        obj,
        range_m_bins,
        control_class_names=DEFAULT_CONTROL_CLASS_NAMES,
    ):
    class_name = str(obj["cls"])
    if class_name not in control_class_names:
        return None

    range_m = float(obj["range_m"])
    for lower, upper in range_m_bins:
        if lower <= range_m < upper:
            return _bin_key(class_name, lower, upper)
    return None


def _load_cartesian_control_gt(sequence, cartesian_gt_root):
    """Load Cartesian GT using the same two formats as the training dataset."""
    if cartesian_gt_root in (None, ""):
        raise ValueError(
            "Cartesian controlled training requires cartesian_gt_root."
        )

    return load_cartesian_gt(sequence, cartesian_gt_root)


def _cartesian_object_range_m(obj):
    box_metric = obj.get("box_metric")
    if box_metric is None or len(box_metric) < 3:
        raise ValueError("Cartesian controlled GT object has no valid box_metric")
    x = float(box_metric[0])
    y = float(box_metric[1])
    z = float(box_metric[2])
    range_m, _azimuth, _elevation = cartesian_to_rae(x, y, z)
    return float(range_m)


def _build_frame_infos(
        sequence,
        range_m_bins,
        box_coordinate_mode,
        cartesian_gt_root=None,
        control_class_names=DEFAULT_CONTROL_CLASS_NAMES,
    ):
    require_cartesian_data(box_coordinate_mode)
    radar_dataset = KRadarRADRAEDataset(
        get_rad_rae_npy_root_dir(), int(sequence)
    )
    gt_key_mode, cartesian_gt = _load_cartesian_control_gt(
        sequence=sequence, cartesian_gt_root=cartesian_gt_root
    )

    category_keys = _category_keys(
        range_m_bins,
        control_class_names=control_class_names,
    )
    frame_infos = []

    for file_idx, frame_name in enumerate(radar_dataset.frame_names):
        category_object_labels = {key: [] for key in category_keys}
        all_target_object_labels = []
        outside_bin_object_labels = []
        if gt_key_mode == "frame_name":
            frame_objects = cartesian_gt.get(frame_name, [])
        else:
            frame_objects = cartesian_gt.get(file_idx, [])

        for obj in frame_objects:
            class_name = str(obj["cls"])
            if class_name not in control_class_names:
                continue

            object_label = int(obj["object_label"])
            all_target_object_labels.append(object_label)
            obj_with_range = dict(obj)
            obj_with_range["range_m"] = _cartesian_object_range_m(obj)
            category_key = _category_key(
                obj_with_range,
                range_m_bins,
                control_class_names=control_class_names,
            )
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
            outside_bin_objects += sum(
                int(label) in kept_labels
                for label in frame_info["outside_bin_object_labels"]
            )
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
        "total_target_objects": int(all_target_objects),
        "selected_target_objects": int(
            sum(
                len(set(kept_by_frame.get(frame_idx, set())))
                for frame_idx in range(len(frame_infos))
            )
            if kept_by_frame is not None
            else all_target_objects
        ),
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
        # Keep out-of-bin objects available to the primary total-bbox
        # decision.  A distance-bin mismatch must not delete them when the
        # overall bbox count is already close to the reference.
        population.setdefault(OUTSIDE_RANGE_CATEGORY_KEY, [])
        population[OUTSIDE_RANGE_CATEGORY_KEY].extend(
            (int(frame_idx), int(object_label))
            for object_label in frame_info["outside_bin_object_labels"]
        )
    return population


def _category_keys_from_frames(frame_infos):
    if not frame_infos:
        return ()
    return tuple(frame_infos[0]["category_counts"].keys())


def _ordered_control_categories(
        range_m_bins,
        control_class_names=DEFAULT_CONTROL_CLASS_NAMES,
    ):
    """Return distance categories in the configured range-bin order."""
    ordered = [
        _bin_key(class_name, lower, upper)
        for lower, upper in range_m_bins
        for class_name in control_class_names
    ]
    ordered.append(OUTSIDE_RANGE_CATEGORY_KEY)
    return ordered


def _range_priority_delta(
        after_summary,
        reference_summary,
        range_m_bins,
        control_class_names=DEFAULT_CONTROL_CLASS_NAMES,
    ):
    """Compare 0-80 m first, then 80-120 m, using old bin differences."""
    deltas = []
    for lower, upper in range_m_bins:
        deltas.append(
            sum(
                abs(
                    int(after_summary["category_totals"].get(key, 0))
                    - int(reference_summary["category_totals"].get(key, 0))
                )
                for key in (
                    _bin_key(class_name, lower, upper)
                    for class_name in control_class_names
                )
            )
        )
    outside_delta = (
        abs(
            int(after_summary["outside_bin_objects"])
            - int(reference_summary.get("outside_bin_objects", 0))
        )
    )
    near_deltas = [
        delta
        for (lower, _upper), delta in zip(range_m_bins, deltas)
        if lower < 80.0
    ]
    far_deltas = [
        delta
        for (lower, _upper), delta in zip(range_m_bins, deltas)
        if lower >= 80.0
    ]
    # The first two values preserve the old absolute-difference behavior,
    # but make the 0-80 m group more important than 80-120 m.  The remaining
    # values keep the individual bins visible for tie-breaking and reporting.
    return tuple(
        [
            int(sum(near_deltas)),
            int(sum(far_deltas) + outside_delta),
        ]
        + [int(delta) for delta in deltas]
        + [int(outside_delta)]
    )


def _run_trial(
    window_frame_infos,
    reference_summary,
    population,
    seed,
    range_m_bins=DEFAULT_RANGE_M_BINS,
    total_bbox_tolerance_ratio=0.0,
    control_class_names=DEFAULT_CONTROL_CLASS_NAMES,
):
    rng = random.Random(int(seed))
    keep_by_frame = {
        frame_idx: set()
        for frame_idx in range(len(window_frame_infos))
    }

    requested_counts = dict(reference_summary["category_totals"])
    requested_counts[OUTSIDE_RANGE_CATEGORY_KEY] = int(
        reference_summary.get("outside_bin_objects", 0)
    )
    source_total = sum(len(objects) for objects in population.values())
    reference_total = int(
        reference_summary.get(
            "total_target_objects",
            reference_summary.get("all_target_objects", 0),
        )
    )
    tolerance_ratio = float(total_bbox_tolerance_ratio)
    tolerance = (
        0
        if not reference_total or tolerance_ratio <= 0.0
        else max(1, int(round(reference_total * tolerance_ratio)))
    )

    # Total bbox count is the primary control target.  If the source is below
    # the reference count, or is already within the configured tolerance,
    # keep every bbox.  In that case a different range distribution is not a
    # reason to delete data.
    keep_all = source_total <= reference_total + tolerance
    actual_requested_counts = {}
    if keep_all:
        for frame_idx, frame_info in enumerate(window_frame_infos):
            keep_by_frame[frame_idx] = {
                int(object_label)
                for object_label in frame_info["all_target_object_labels"]
            }
        actual_requested_counts = {
            category_key: len(objects)
            for category_key, objects in population.items()
        }
    else:
        desired_total = min(source_total, reference_total)
        selected_counts = {
            category_key: min(
                int(requested_counts.get(category_key, 0)),
                len(objects),
            )
            for category_key, objects in population.items()
        }
        selected_total = sum(selected_counts.values())

        # Only when total-count control requires deletion do we use the
        # distance bins as a secondary objective.  First reserve as many boxes
        # as the corresponding reference bin contains.  If more source boxes
        # must be kept to reach the reference total, keep the remaining boxes
        # strictly from near to far: 0-20, 20-40, 40-60, 60-80, then 80-120 m.
        ordered_categories = _ordered_control_categories(
            range_m_bins,
            control_class_names=control_class_names,
        )
        category_order = {
            category_key: index
            for index, category_key in enumerate(ordered_categories)
        }

        while selected_total < desired_total:
            candidates = [
                category_key
                for category_key, objects in population.items()
                if selected_counts[category_key] < len(objects)
            ]
            if not candidates:
                break
            chosen_category = min(
                candidates,
                key=lambda category_key: category_order.get(
                    category_key,
                    len(category_order),
                ),
            )
            selected_counts[chosen_category] += 1
            selected_total += 1

        for category_key, objects in population.items():
            keep_count = int(selected_counts[category_key])
            actual_requested_counts[category_key] = keep_count
            for frame_idx, object_label in rng.sample(objects, keep_count):
                keep_by_frame[frame_idx].add(int(object_label))

    after_summary = _summarize_frames(
        window_frame_infos,
        kept_by_frame=keep_by_frame,
    )
    total_bbox_delta = abs(
        int(after_summary["selected_target_objects"])
        - reference_total
    )
    range_priority_delta = _range_priority_delta(
        after_summary,
        reference_summary,
        range_m_bins,
        control_class_names=control_class_names,
    )
    empty_delta = abs(
        int(after_summary["empty_frames"])
        - int(reference_summary["empty_frames"])
    )
    return {
        "seed": int(seed),
        "keep_all": bool(keep_all),
        "total_bbox_tolerance": int(tolerance),
        # Primary: total bbox count. Secondary: largest bin difference in
        # 0-80 m, then 80-120 m. Tertiary: empty-frame count, then seed.
        "score": (
            int(total_bbox_delta),
            range_priority_delta,
            int(empty_delta),
            int(seed),
        ),
        "range_priority_delta": list(range_priority_delta),
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


def _comparison_text(
        pair_results,
        range_m_bins,
        box_coordinate_mode,
        control_class_names=DEFAULT_CONTROL_CLASS_NAMES,
    ):
    controlled_class_text = ", ".join(control_class_names)
    lines = [
        "Controlled sequence comparison",
        "",
        "The before and after values refer to the selected continuous source window.",
        "Ignored bboxes are not removed from gt.txt; they are applied through the training ignore override.",
        f"Controlled classes: {controlled_class_text}",
        "Control priority: controlled-class bbox count first, largest distance-bin difference within 0-80 m second, 80-120 m third, empty-frame count fourth.",
        "",
        f"Coordinate mode: {box_coordinate_mode}",
        "Center range bins (m): " + ", ".join(
            f"[{_format_number(lower)},{_format_number(upper)})"
            for lower, upper in range_m_bins
        ),
    ]

    for result in pair_results:
        source = _sequence_part_label(
            result["source_sequence"],
            result.get("source_position", "full"),
        )
        reference = _sequence_part_label(
            result["reference_sequence"],
            result.get("reference_position", "full"),
        )
        before = result["before_summary"]
        after = result["after_summary"]
        target = result["reference_summary"]
        lines.extend([
            "",
            f"Sequence pair: controlled {source} -> reference {reference}",
            f"Selected source window: file_idx {result['start_file_idx']} - {result['end_file_idx']}",
            f"Selected random seed: {result['selected_seed']}",
            "",
            "Metric | Before control | After control | Reference",
            "--- | ---: | ---: | ---:",
            f"Frames | {before['frames']} | {after['frames']} | {target['frames']}",
            f"Empty frames | {before['empty_frames']} | {after['empty_frames']} | {target['empty_frames']}",
            f"Empty rate | {_format_rate(before['empty_rate'])} | {_format_rate(after['empty_rate'])} | {_format_rate(target['empty_rate'])}",
            f"Controlled-class bbox | {before['total_target_objects']} | {after['selected_target_objects']} | {target['total_target_objects']}",
        ])
        for class_name in control_class_names:
            for lower, upper in range_m_bins:
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


def _requested_config(args):
    pairs = _pair_sequence_parts(args)
    control_class_names = _normalize_control_class_names(
        getattr(args, "control_class_names", None)
    )
    box_coordinate_mode = require_cartesian_data(
        getattr(args, "box_coordinate_mode", BOX_COORDINATE_CARTESIAN)
    )
    bins = _normalize_bins(getattr(args, "control_range_m_bins", None))
    half_ratio = float(getattr(args, "train_sequence_half_ratio", 0.5))
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "pairs": [
            {
                "source": [int(source), str(source_position)],
                "reference": [int(reference), str(reference_position)],
            }
            for source, source_position, reference, reference_position in pairs
        ],
        "box_coordinate_mode": box_coordinate_mode,
        "control_class_names": list(control_class_names),
        "range_m_bins": [[float(lower), float(upper)] for lower, upper in bins],
        "half_ratio": half_ratio,
        "window_position": str(getattr(args, "control_window_position", "last")),
        "seed": int(getattr(args, "seed", 42)),
        "num_trials": int(getattr(args, "control_num_trials", 300)),
        "total_bbox_tolerance_ratio": float(
            getattr(args, "control_total_bbox_tolerance_ratio", 0.0)
        ),
    }, pairs, bins


def _request_signature(request):
    """Only compare settings that change the generated controlled data."""
    return {
        "schema_version": request.get("schema_version"),
        "pairs": request.get("pairs"),
        "box_coordinate_mode": request.get("box_coordinate_mode"),
        "control_class_names": request.get(
            "control_class_names",
            list(SUPPORTED_CONTROL_CLASS_NAMES),
        ),
        "range_m_bins": request.get("range_m_bins"),
        "half_ratio": request.get("half_ratio", 0.5),
        "window_position": request.get("window_position"),
        "seed": request.get("seed"),
        "num_trials": request.get("num_trials"),
        "total_bbox_tolerance_ratio": request.get(
            "total_bbox_tolerance_ratio"
        ),
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
        f"{_sequence_part_label(*pair['source'])}_ref"
        f"{_sequence_part_label(*pair['reference']).removeprefix('seq')}"
        for pair in request["pairs"]
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
    request, pairs, range_m_bins = _requested_config(args)
    control_class_names = tuple(request["control_class_names"])
    controlled_sequences = tuple(source for source, _part, _reference, _ref_part in pairs)
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
    if not 0.0 <= request["total_bbox_tolerance_ratio"] <= 1.0:
        raise ValueError(
            "control_total_bbox_tolerance_ratio must be between 0 and 1"
        )

    source_positions = {}
    reference_positions = {}
    for source, source_position, reference, reference_position in pairs:
        source_positions.setdefault(int(source), set()).add(source_position)
        reference_positions.setdefault(int(reference), set()).add(
            reference_position
        )
    for sequence, positions in (
        list(source_positions.items()) + list(reference_positions.items())
    ):
        if {"first", "last"}.issubset(positions) and abs(
            float(request["half_ratio"]) - 0.5
        ) > 1e-12:
            raise ValueError(
                f"Sequence {sequence} uses both first and last parts; "
                "train_sequence_half_ratio must be 0.5."
            )

    output_dir, should_generate = _select_output_dir(
        getattr(args, "controlled_split_base_dir", "split"),
        request,
    )
    if should_generate:
        pair_results = []
        override_sequences = {}
        for (
            source_sequence,
            source_position,
            reference_sequence,
            reference_position,
        ) in pairs:
            all_source_infos = _build_frame_infos(
                sequence=source_sequence,
                range_m_bins=range_m_bins,
                box_coordinate_mode=request["box_coordinate_mode"],
                cartesian_gt_root=getattr(args, "cartesian_gt_root", None),
                control_class_names=control_class_names,
            )
            all_reference_infos = _build_frame_infos(
                sequence=reference_sequence,
                range_m_bins=range_m_bins,
                box_coordinate_mode=request["box_coordinate_mode"],
                cartesian_gt_root=getattr(args, "cartesian_gt_root", None),
                control_class_names=control_class_names,
            )
            source_infos = _select_sequence_part(
                all_source_infos,
                source_position,
                request["half_ratio"],
                complementary={"first", "last"}.issubset(
                    source_positions[int(source_sequence)]
                ),
            )
            reference_infos = _select_sequence_part(
                all_reference_infos,
                reference_position,
                request["half_ratio"],
                complementary={"first", "last"}.issubset(
                    reference_positions[int(reference_sequence)]
                ),
            )
            if not source_infos or not reference_infos:
                raise ValueError(
                    "Cannot control "
                    f"{_sequence_part_label(source_sequence, source_position)} "
                    "-> "
                    f"{_sequence_part_label(reference_sequence, reference_position)}: "
                    "one sequence has no frames"
                )

            window_length = min(len(source_infos), len(reference_infos))
            if request["window_position"] == "first":
                start_file_idx = 0
            else:
                start_file_idx = len(source_infos) - window_length
            end_file_idx = start_file_idx + window_length - 1
            window_infos = source_infos[start_file_idx:end_file_idx + 1]
            window_start_file_idx = int(window_infos[0]["file_idx"])
            window_end_file_idx = int(window_infos[-1]["file_idx"])
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
                    range_m_bins=range_m_bins,
                    total_bbox_tolerance_ratio=request[
                        "total_bbox_tolerance_ratio"
                    ],
                    control_class_names=control_class_names,
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
            sequence_payload = override_sequences.setdefault(
                str(int(source_sequence)),
                {
                    "matched_parts": [],
                    "frame_overrides": {},
                    "frame_after_counts": {},
                },
            )
            sequence_payload["matched_parts"].append({
                "source_position": str(source_position),
                "reference_sequence": int(reference_sequence),
                "reference_position": str(reference_position),
            })
            for frame_name, frame_payload in override_frames.items():
                existing_payload = sequence_payload["frame_overrides"].setdefault(
                    frame_name,
                    {"ignore_object_labels": []},
                )
                merged_labels = set(existing_payload.get("ignore_object_labels", ()))
                merged_labels.update(frame_payload.get("ignore_object_labels", ()))
                existing_payload["ignore_object_labels"] = sorted(
                    int(label) for label in merged_labels
                )
            sequence_payload["frame_after_counts"].update(frame_after_counts)
            pair_results.append({
                "source_sequence": int(source_sequence),
                "source_position": str(source_position),
                "reference_sequence": int(reference_sequence),
                "reference_position": str(reference_position),
                "start_file_idx": window_start_file_idx,
                "end_file_idx": window_end_file_idx,
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
            "range_m_bins": [
                [float(lower), float(upper)]
                for lower, upper in range_m_bins
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
            _comparison_text(
                pair_results,
                range_m_bins,
                request["box_coordinate_mode"],
                control_class_names=control_class_names,
            ),
            encoding="utf-8",
        )
        train_entries = set()
        test_entries = set()
        for result in pair_results:
            selected_names = result["selected_frame_names"]
            excluded_names = result["excluded_frame_names"]
            train_entries.update(
                (int(result["source_sequence"]), frame_name)
                for frame_name in selected_names
            )
            test_entries.update(
                (int(result["source_sequence"]), frame_name)
                for frame_name in excluded_names
            )
        overlap_entries = train_entries & test_entries
        if overlap_entries:
            raise RuntimeError(
                "Controlled sequence parts produced overlapping train/test "
                f"entries: {sorted(overlap_entries)[:10]}"
            )
        train_lines = [
            f"{sequence},{frame_name}.txt\n"
            for sequence, frame_name in sorted(train_entries)
        ]
        test_lines = [
            f"{sequence},{frame_name}.txt\n"
            for sequence, frame_name in sorted(test_entries)
        ]
        (output_dir / "train.txt").write_text("".join(train_lines), encoding="utf-8")
        (output_dir / "test.txt").write_text("".join(test_lines), encoding="utf-8")

        print(f"Generated controlled training split: {output_dir}")
        print(f"Comparison report: {output_dir / 'comparison.txt'}")
        for result in pair_results:
            after = result["after_summary"]
            reference = result["reference_summary"]
            print(
                "  "
                f"{_sequence_part_label(result['source_sequence'], result['source_position'])} "
                "-> "
                f"{_sequence_part_label(result['reference_sequence'], result['reference_position'])}: "
                f"frames {after['frames']}/{reference['frames']}, "
                f"empty rate {_format_rate(after['empty_rate'])}/{_format_rate(reference['empty_rate'])}"
            )
    else:
        print(f"Reusing controlled training split: {output_dir}")

    args.train_control_split_dir = str(output_dir)
    args.gt_object_ignore_override_path = str(
        output_dir / "object_ignore_override.json"
    )
    args.controlled_sequences = tuple(dict.fromkeys(
        source for source, _part, _reference, _ref_part in pairs
    ))
    args.reference_sequences = tuple(dict.fromkeys(
        reference for _source, _part, reference, _ref_part in pairs
    ))
    args.controlled_sequence_parts = tuple(
        (source, source_position)
        for source, source_position, _reference, _reference_position in pairs
    )
    args.reference_sequence_parts = tuple(
        (reference, reference_position)
        for _source, _source_position, reference, reference_position in pairs
    )
    args.control_range_m_bins = tuple(tuple(pair) for pair in range_m_bins)
    args.control_class_names = tuple(control_class_names)
    return args


def apply_train_control_split_indices(
    full_dataset,
    train_indices,
    control_split_dir,
):
    """Filter controlled sequences to the generated training manifest."""
    if control_split_dir is None:
        raise ValueError(
            "train_control_split_enabled=True requires train_control_split_dir."
        )

    control_train_split_path = os.path.join(control_split_dir, "train.txt")
    if not os.path.exists(control_train_split_path):
        raise FileNotFoundError(
            f"Controlled train split file not found: {control_train_split_path}"
        )

    sequence_lookup = build_sequence_index_lookup(full_dataset)
    allowed_sequences = tuple(sorted(sequence_lookup.keys()))
    control_train_by_sequence = read_split_file(
        control_train_split_path,
        allowed_sequences=allowed_sequences,
    )
    controlled_sequences = {
        int(sequence)
        for sequence, entries in control_train_by_sequence.items()
        if len(entries) > 0
    }
    if len(controlled_sequences) == 0:
        print(
            f"Train control split enabled but {control_train_split_path} contains no usable entries. "
            "Keeping the original train split."
        )
        return train_indices

    controlled_index_set = set(
        split_entries_to_indices(
            split_by_sequence=control_train_by_sequence,
            sequence_lookup=sequence_lookup,
            split_name=os.path.join(control_split_dir, "train.txt"),
        )
    )
    sequence_ranges = [
        {
            "sequence": int(sequence_range["sequence"]),
            "start": int(sequence_range["start"]),
            "end": int(sequence_range["end"]),
        }
        for sequence_range in full_dataset.get_sequence_ranges()
    ]

    def sequence_for_index(index):
        for sequence_range in sequence_ranges:
            if sequence_range["start"] <= index < sequence_range["end"]:
                return sequence_range["sequence"]
        raise ValueError(f"Could not resolve sequence for global index {index}")

    filtered_train_indices = []
    per_sequence_before = {}
    per_sequence_after = {}
    for index in train_indices:
        sequence = sequence_for_index(int(index))
        per_sequence_before[sequence] = per_sequence_before.get(sequence, 0) + 1
        if sequence in controlled_sequences and index not in controlled_index_set:
            continue
        filtered_train_indices.append(index)
        per_sequence_after[sequence] = per_sequence_after.get(sequence, 0) + 1

    controlled_sequences_in_train = sorted(
        int(sequence)
        for sequence in controlled_sequences
        if per_sequence_before.get(int(sequence), 0) > 0
    )
    if len(controlled_sequences_in_train) == 0:
        print(
            f"Train control split {control_split_dir} does not overlap the current train split. "
            "Keeping the original train split."
        )
        return train_indices

    summary_parts = []
    for sequence in controlled_sequences_in_train:
        before_count = int(per_sequence_before.get(sequence, 0))
        after_count = int(per_sequence_after.get(sequence, 0))
        summary_parts.append(f"seq{sequence}: {before_count}->{after_count}")
    print(
        "Applied train control split:",
        f"dir={control_split_dir}",
        f"sequences={controlled_sequences_in_train}",
        f"counts={', '.join(summary_parts)}",
    )

    return filtered_train_indices
