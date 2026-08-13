"""GT-derived, tie-preserving distance quartiles for Cartesian evaluation.

Quartile thresholds are derived only from eligible ground-truth boxes in the
evaluation state.  A threshold is placed at the midpoint of a gap between two
distinct GT ranges, as close as possible to the requested cumulative rank.
Consequently equal-range GT boxes are never split between quartiles.  When an
exact 25% split is impossible, the actual GT count is reported for every bin.
"""

import ast
import json
import math
from pathlib import Path

import numpy as np

from eval.adapter import metric_boxes_to_kitti_anno


QUARTILE_TAGS = ("q1", "q2", "q3", "q4")


__all__ = [
    "QUARTILE_TAGS",
    "derive_gt_distance_quartile_bins",
    "filter_kradar_eval_state_by_quartile",
    "normalize_distance_quartile_bins",
]


def _load_quartile_payload(value):
    if isinstance(value, (list, tuple, dict)):
        return value
    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return None
    candidate_path = Path(text)
    try:
        is_file = candidate_path.is_file()
    except OSError:
        is_file = False
    if is_file:
        text = candidate_path.read_text(encoding="utf-8").strip()

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            payload = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            payload = None

    if payload is None:
        # Evaluation TXT reports store the JSON payload after this metadata
        # prefix.  Accepting the report itself avoids hand-copying boundaries.
        prefix = "distance_quartile_bins:"
        for line in text.splitlines():
            if line.strip().startswith(prefix):
                payload_text = line.split(":", 1)[1].strip()
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    payload = ast.literal_eval(payload_text)
                break

    if payload is None and "," in text and "-" in text:
        tokens = text.split(",")
        if len(tokens) != 4:
            raise ValueError(
                "distance_quartile_bins text must contain exactly four bins."
            )
        payload = []
        for tag, token in zip(QUARTILE_TAGS, tokens):
            lower_text, upper_text = token.strip().split("-", 1)
            upper_normalized = upper_text.strip().lower()
            payload.append({
                "tag": tag,
                "lower_m": float(lower_text),
                "upper_m": (
                    math.inf
                    if upper_normalized in {"inf", "+inf", "infinity", "+infinity"}
                    else float(upper_text)
                ),
            })
    if payload is None:
        raise ValueError(
            "distance_quartile_bins must be a JSON/list value, an evaluation "
            "report or JSON file, or 'lower-upper,...' bin text."
        )
    return payload


def normalize_distance_quartile_bins(value):
    """Normalize four externally supplied, contiguous quartile boundaries.

    Only boundaries are retained. Counts in a target-domain report are not
    copied because the controlled source test has its own valid-GT counts.
    """
    payload = _load_quartile_payload(value)
    if payload is None:
        return None
    if isinstance(payload, dict):
        for key in ("distance_quartile_bins", "quartile_bins", "bins"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)) or len(payload) != 4:
        raise ValueError("distance_quartile_bins must contain exactly four bins.")

    bins = []
    previous_upper = None
    for index, item in enumerate(payload):
        if isinstance(item, dict):
            lower_m = item.get("lower_m", item.get("lower"))
            upper_m = item.get("upper_m", item.get("upper"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lower_m, upper_m = item[:2]
        else:
            raise ValueError(f"Invalid distance quartile bin: {item!r}")
        lower_m = float(lower_m)
        if isinstance(upper_m, str) and upper_m.strip().lower() in {
            "inf", "+inf", "infinity", "+infinity",
        }:
            upper_m = math.inf
        else:
            upper_m = float(upper_m)
        if not math.isfinite(lower_m) or lower_m < 0.0:
            raise ValueError(f"Invalid quartile lower bound: {lower_m!r}")
        if not (math.isfinite(upper_m) or math.isinf(upper_m)):
            raise ValueError(f"Invalid quartile upper bound: {upper_m!r}")
        if upper_m <= lower_m:
            raise ValueError(f"Quartile upper bound must exceed lower bound: {item!r}")
        if previous_upper is not None and not math.isclose(
            lower_m, previous_upper, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("distance_quartile_bins must be contiguous.")
        bins.append({
            "tag": QUARTILE_TAGS[index],
            "lower_m": lower_m,
            "upper_m": upper_m,
        })
        previous_upper = upper_m

    if not math.isclose(bins[0]["lower_m"], 0.0, abs_tol=1e-9):
        raise ValueError("The first distance quartile must start at 0 metres.")
    if not math.isinf(bins[-1]["upper_m"]):
        raise ValueError("The fourth distance quartile must end at infinity.")
    return tuple(bins)


def _metric_distances(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    distances = np.linalg.norm(boxes[:, :3], axis=1)
    if not np.all(np.isfinite(distances)):
        raise ValueError("GT/detection box centers contain non-finite distances.")
    return distances


def _nearest_gap_index(cumulative_counts, target_count, min_index, max_index):
    """Choose a gap closest to a target rank; ties prefer the larger rank."""
    candidates = np.arange(min_index, max_index + 1, dtype=np.int64)
    candidate_counts = cumulative_counts[candidates]
    errors = np.abs(candidate_counts.astype(np.float64) - float(target_count))
    minimum_error = float(errors.min())
    tied = candidates[np.isclose(errors, minimum_error, rtol=0.0, atol=1e-12)]
    return int(tied[-1])


def derive_gt_distance_quartile_bins(state):
    """Derive four contiguous distance bins from pooled eligible GT centers.

    Returns four dictionaries with ``tag``, ``lower_m``, ``upper_m``, and
    ``bbox_count``.  Bins are left-closed/right-open; Q4 ends at infinity.
    """
    distance_parts = [
        _metric_distances(frame["gt_boxes"])
        for frame in state.get("metric_frames", [])
        if np.asarray(frame["gt_boxes"]).reshape(-1, 7).shape[0] > 0
    ]
    if not distance_parts:
        raise ValueError("Cannot derive distance quartiles: evaluation GT is empty.")
    distances = np.sort(np.concatenate(distance_parts))
    unique_values, unique_counts = np.unique(distances, return_counts=True)
    if unique_values.shape[0] < 4:
        raise ValueError(
            "Cannot construct four non-empty tie-preserving distance quartiles "
            f"from only {unique_values.shape[0]} distinct GT range value(s)."
        )

    cumulative = np.cumsum(unique_counts)
    selected_gap_indices = []
    previous_index = -1
    num_unique = int(unique_values.shape[0])
    num_gt = int(distances.shape[0])
    for quartile_number in (1, 2, 3):
        remaining_cuts = 3 - quartile_number
        minimum_index = previous_index + 1
        maximum_index = num_unique - 2 - remaining_cuts
        gap_index = _nearest_gap_index(
            cumulative_counts=cumulative,
            target_count=(num_gt * quartile_number) / 4.0,
            min_index=minimum_index,
            max_index=maximum_index,
        )
        selected_gap_indices.append(gap_index)
        previous_index = gap_index

    edges = [
        float(
            unique_values[index]
            + ((unique_values[index + 1] - unique_values[index]) * 0.5)
        )
        for index in selected_gap_indices
    ]
    bounds = ((0.0, edges[0]), (edges[0], edges[1]), (edges[1], edges[2]), (edges[2], math.inf))
    bins = []
    for tag, (lower_m, upper_m) in zip(QUARTILE_TAGS, bounds):
        mask = (distances >= lower_m) & (distances < upper_m)
        bins.append(
            {
                "tag": tag,
                "lower_m": float(lower_m),
                "upper_m": float(upper_m),
                "bbox_count": int(mask.sum()),
            }
        )
    if sum(quartile["bbox_count"] for quartile in bins) != num_gt:
        raise RuntimeError("Derived quartile bins do not cover every GT box exactly once.")
    if any(quartile["bbox_count"] <= 0 for quartile in bins):
        raise RuntimeError(f"Derived an empty distance quartile: {bins!r}")
    return tuple(bins)


def _filter_metric_frame(frame, lower_m, upper_m):
    gt_boxes = np.asarray(frame["gt_boxes"]).reshape(-1, 7)
    gt_labels = np.asarray(frame["gt_labels"]).reshape(-1)
    dt_boxes = np.asarray(frame["dt_boxes"]).reshape(-1, 7)
    dt_labels = np.asarray(frame["dt_labels"]).reshape(-1)
    dt_scores = np.asarray(frame["dt_scores"]).reshape(-1)
    neutral_gt_boxes = np.asarray(
        frame.get("neutral_gt_boxes", np.zeros((0, 7), dtype=np.float64))
    ).reshape(-1, 7)
    neutral_gt_labels = np.asarray(
        frame.get("neutral_gt_labels", np.zeros((0,), dtype=np.int64))
    ).reshape(-1)
    if gt_boxes.shape[0] != gt_labels.shape[0]:
        raise ValueError("GT box and label counts differ.")
    if not (dt_boxes.shape[0] == dt_labels.shape[0] == dt_scores.shape[0]):
        raise ValueError("Detection box, label, and score counts differ.")
    if neutral_gt_boxes.shape[0] != neutral_gt_labels.shape[0]:
        raise ValueError("Neutral GT box and label counts differ.")
    gt_distances = _metric_distances(gt_boxes)
    dt_distances = _metric_distances(dt_boxes)
    gt_mask = (gt_distances >= lower_m) & (gt_distances < upper_m)
    dt_mask = (dt_distances >= lower_m) & (dt_distances < upper_m)
    neutral_distances = _metric_distances(neutral_gt_boxes)
    neutral_mask = (
        (neutral_distances >= lower_m) & (neutral_distances < upper_m)
    )
    return {
        **frame,
        "gt_boxes": gt_boxes[gt_mask],
        "gt_labels": gt_labels[gt_mask],
        "dt_boxes": dt_boxes[dt_mask],
        "dt_labels": dt_labels[dt_mask],
        "dt_scores": dt_scores[dt_mask],
        "neutral_gt_boxes": neutral_gt_boxes[neutral_mask],
        "neutral_gt_labels": neutral_gt_labels[neutral_mask],
    }


def _metric_frame_to_official_gt_anno(frame, official_class_name_map):
    valid_anno = metric_boxes_to_kitti_anno(
        boxes=frame["gt_boxes"],
        labels=frame["gt_labels"],
        is_prediction=False,
        class_name_map=official_class_name_map,
    )
    neutral_boxes = np.asarray(
        frame.get("neutral_gt_boxes", np.zeros((0, 7), dtype=np.float64))
    ).reshape(-1, 7)
    if neutral_boxes.shape[0] == 0:
        return valid_anno
    neutral_anno = metric_boxes_to_kitti_anno(
        boxes=neutral_boxes,
        labels=frame["neutral_gt_labels"],
        is_prediction=False,
        class_name_map=official_class_name_map,
    )
    # The revised official evaluator treats same-class GT with occlusion 3 as
    # ignored for every difficulty. Predictions matching it are neutral.
    neutral_anno["occluded"][:] = 3
    return {
        key: np.concatenate([valid_anno[key], neutral_anno[key]], axis=0)
        for key in valid_anno
    }


def filter_kradar_eval_state_by_quartile(
        state,
        quartile_bin,
        official_class_name_map,
    ):
    """Filter GT/detections independently and retain every evaluation frame."""
    lower_m = float(quartile_bin["lower_m"])
    upper_m = float(quartile_bin["upper_m"])
    if lower_m < 0.0 or upper_m <= lower_m or not math.isfinite(lower_m):
        raise ValueError(f"Invalid quartile bounds: {quartile_bin!r}")
    if not (math.isfinite(upper_m) or math.isinf(upper_m)):
        raise ValueError(f"Invalid quartile upper bound: {upper_m!r}")

    filtered_frames = [
        _filter_metric_frame(frame, lower_m, upper_m)
        for frame in state.get("metric_frames", [])
    ]
    filtered_state = dict(state)
    filtered_state["metric_frames"] = filtered_frames
    filtered_state["official_gt_annos"] = [
        _metric_frame_to_official_gt_anno(frame, official_class_name_map)
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
