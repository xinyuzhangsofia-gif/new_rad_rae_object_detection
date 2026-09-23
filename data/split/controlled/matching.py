"""Scientific matching policy for generated Controlled Splits."""

from __future__ import annotations

import random
from collections import Counter

from configs.coordinates import require_cartesian_data
from configs.data import RADAR_NPY_ROOT
from ...coordinates import cartesian_to_rae
from ...dataset import KRadarRADRAEDataset
from ...labels import load_cartesian_gt


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
OUTSIDE_RANGE_CATEGORY_KEY = "__outside_range__"


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
    """Load Cartesian GT from the canonical ``<sequence>/gt/gt.txt`` file."""
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
        RADAR_NPY_ROOT, int(sequence)
    )
    cartesian_gt = _load_cartesian_control_gt(
        sequence=sequence, cartesian_gt_root=cartesian_gt_root
    )[1]

    category_keys = _category_keys(
        range_m_bins,
        control_class_names=control_class_names,
    )
    frame_infos = []

    for file_idx, frame_name in enumerate(radar_dataset.frame_names):
        category_object_labels = {key: [] for key in category_keys}
        all_target_object_labels = []
        outside_bin_object_labels = []
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
