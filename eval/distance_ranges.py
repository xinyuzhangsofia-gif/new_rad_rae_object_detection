"""Distance-bin helpers for Cartesian K-Radar evaluation.

Distance bins use the Euclidean norm of each box center ``(x, y, z)`` and
follow left-closed, right-open semantics: ``lower_m <= distance < upper_m``.
Ground-truth boxes and detections are filtered independently so that neither
side can influence which objects remain on the other side.
"""

import math

import numpy as np

from eval.adapter import metric_boxes_to_kitti_anno


DEFAULT_DISTANCE_RANGES = (
    (0.0, 30.0),
    (30.0, 60.0),
    (60.0, 90.0),
    (90.0, 120.0),
)


__all__ = [
    "DEFAULT_DISTANCE_RANGES",
    "distance_range_tag",
    "normalize_distance_ranges",
    "filter_metric_frame_by_distance",
    "filter_kradar_eval_state_by_distance",
]


def _format_distance_bound(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def distance_range_tag(lower_m, upper_m):
    """Return a stable key suffix such as ``0_30m``."""
    return f"{_format_distance_bound(lower_m)}_{_format_distance_bound(upper_m)}m"


def normalize_distance_ranges(value=None):
    """Normalize configured distance bins to validated ``(lower, upper)`` pairs.

    Strings accept either ``0-30,30-60`` or ``0:30,30:60`` syntax. ``None``
    selects :data:`DEFAULT_DISTANCE_RANGES`.
    """
    if value is None:
        raw_ranges = DEFAULT_DISTANCE_RANGES
    elif isinstance(value, str):
        raw_ranges = []
        for raw_item in value.split(","):
            item = raw_item.strip()
            if item == "":
                continue
            separator = ":" if ":" in item else "-"
            parts = [part.strip() for part in item.split(separator)]
            if len(parts) != 2 or any(part == "" for part in parts):
                raise ValueError(
                    "Distance ranges must use lower-upper pairs, for example "
                    f"'0-30,30-60'; got {item!r}."
                )
            raw_ranges.append((float(parts[0]), float(parts[1])))
    elif isinstance(value, np.ndarray):
        raw_ranges = value.tolist()
    elif isinstance(value, (list, tuple)):
        raw_ranges = value
    else:
        raise ValueError(f"Unsupported distance-range value: {value!r}")

    normalized = []
    for raw_range in raw_ranges:
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raise ValueError(
                "Each distance range must contain exactly two bounds; "
                f"got {raw_range!r}."
            )
        lower_m, upper_m = (float(raw_range[0]), float(raw_range[1]))
        if not math.isfinite(lower_m) or not math.isfinite(upper_m):
            raise ValueError(
                f"Distance-range bounds must be finite, got {raw_range!r}."
            )
        if lower_m < 0.0:
            raise ValueError(
                f"Distance-range lower bounds must be non-negative, got {lower_m}."
            )
        if upper_m <= lower_m:
            raise ValueError(
                "Distance-range upper bounds must be greater than lower bounds, "
                f"got ({lower_m}, {upper_m})."
            )
        normalized.append((lower_m, upper_m))

    if len(normalized) == 0:
        raise ValueError("At least one distance range is required.")

    for previous, current in zip(normalized, normalized[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                "Distance ranges must be ordered and non-overlapping, got "
                f"{previous!r} followed by {current!r}."
            )

    tags = [distance_range_tag(*distance_range) for distance_range in normalized]
    if len(set(tags)) != len(tags):
        raise ValueError(f"Distance ranges produce duplicate tags: {normalized!r}")
    return tuple(normalized)


def _distance_mask(boxes, lower_m, upper_m):
    boxes = np.asarray(boxes)
    boxes = boxes.reshape(-1, 7)
    distances_m = np.linalg.norm(boxes[:, :3], axis=1)
    return (distances_m >= float(lower_m)) & (distances_m < float(upper_m))


def filter_metric_frame_by_distance(frame, lower_m, upper_m):
    """Filter one metric frame without mutating it.

    GT and detection masks are deliberately calculated separately. Empty
    arrays remain in the returned frame, allowing callers to retain every
    original frame for official evaluation.
    """
    lower_m, upper_m = normalize_distance_ranges([(lower_m, upper_m)])[0]
    gt_boxes = np.asarray(frame["gt_boxes"]).reshape(-1, 7)
    gt_labels = np.asarray(frame["gt_labels"]).reshape(-1)
    dt_boxes = np.asarray(frame["dt_boxes"]).reshape(-1, 7)
    dt_labels = np.asarray(frame["dt_labels"]).reshape(-1)
    dt_scores = np.asarray(frame["dt_scores"]).reshape(-1)

    if gt_boxes.shape[0] != gt_labels.shape[0]:
        raise ValueError(
            "GT box/label count mismatch: "
            f"{gt_boxes.shape[0]} boxes vs {gt_labels.shape[0]} labels."
        )
    if not (
        dt_boxes.shape[0] == dt_labels.shape[0] == dt_scores.shape[0]
    ):
        raise ValueError(
            "Detection box/label/score count mismatch: "
            f"{dt_boxes.shape[0]} boxes, {dt_labels.shape[0]} labels, "
            f"{dt_scores.shape[0]} scores."
        )

    gt_mask = _distance_mask(gt_boxes, lower_m, upper_m)
    dt_mask = _distance_mask(dt_boxes, lower_m, upper_m)
    return {
        **frame,
        "gt_boxes": gt_boxes[gt_mask],
        "gt_labels": gt_labels[gt_mask],
        "dt_boxes": dt_boxes[dt_mask],
        "dt_labels": dt_labels[dt_mask],
        "dt_scores": dt_scores[dt_mask],
    }


def filter_kradar_eval_state_by_distance(
        state,
        lower_m,
        upper_m,
        official_class_name_map,
    ):
    """Rebuild official annotations for one distance bin.

    Exactly one filtered annotation pair is produced per input metric frame,
    including frames where the selected bin contains no GT or detections.
    """
    filtered_frames = [
        filter_metric_frame_by_distance(frame, lower_m, upper_m)
        for frame in state.get("metric_frames", [])
    ]
    filtered_state = dict(state)
    filtered_state["metric_frames"] = filtered_frames
    filtered_state["official_gt_annos"] = [
        metric_boxes_to_kitti_anno(
            boxes=frame["gt_boxes"],
            labels=frame["gt_labels"],
            is_prediction=False,
            class_name_map=official_class_name_map,
        )
        for frame in filtered_frames
    ]
    filtered_state["official_dt_annos"] = [
        metric_boxes_to_kitti_anno(
            boxes=frame["dt_boxes"],
            labels=frame["dt_labels"],
            scores=frame["dt_scores"],
            is_prediction=True,
            class_name_map=official_class_name_map,
        )
        for frame in filtered_frames
    ]
    return filtered_state
