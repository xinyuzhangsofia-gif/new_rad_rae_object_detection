"""Average precision for axis-aligned Polar/RAE BEV rectangles.

This metric is intentionally separate from the official Cartesian K-Radar
metric.  A Polar box is represented as
``[r_center, a_center, e_center, r_width, a_width, e_width, yaw]`` in bin
coordinates.  Polar BEV IoU uses only the ``r``/``a`` rectangle and ignores
elevation and yaw, matching the rectangle drawn on the RA map.

Detection matching, score-threshold sampling, the 41-point precision envelope,
and percentage units follow K-Radar ``eval_revised.py``. The overlap function
is the only intentional metric difference.
"""

import numpy as np


POLAR_NUM_SAMPLE_POINTS = 41


def format_polar_iou_suffix(iou_value):
    return f"{float(iou_value):.1f}"


def polar_bev_overlaps(gt_boxes, dt_boxes):
    """Return axis-aligned IoU in Polar ``(range-bin, azimuth-bin)`` space."""
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)
    dt_boxes = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 7)
    if gt_boxes.shape[0] == 0 or dt_boxes.shape[0] == 0:
        return np.zeros((gt_boxes.shape[0], dt_boxes.shape[0]), dtype=np.float64)

    gt_r_half = np.abs(gt_boxes[:, 3])[:, None] * 0.5
    gt_a_half = np.abs(gt_boxes[:, 4])[:, None] * 0.5
    dt_r_half = np.abs(dt_boxes[:, 3])[None, :] * 0.5
    dt_a_half = np.abs(dt_boxes[:, 4])[None, :] * 0.5

    gt_r_min = gt_boxes[:, 0][:, None] - gt_r_half
    gt_r_max = gt_boxes[:, 0][:, None] + gt_r_half
    gt_a_min = gt_boxes[:, 1][:, None] - gt_a_half
    gt_a_max = gt_boxes[:, 1][:, None] + gt_a_half
    dt_r_min = dt_boxes[:, 0][None, :] - dt_r_half
    dt_r_max = dt_boxes[:, 0][None, :] + dt_r_half
    dt_a_min = dt_boxes[:, 1][None, :] - dt_a_half
    dt_a_max = dt_boxes[:, 1][None, :] + dt_a_half

    intersection_r = np.maximum(
        0.0,
        np.minimum(gt_r_max, dt_r_max) - np.maximum(gt_r_min, dt_r_min),
    )
    intersection_a = np.maximum(
        0.0,
        np.minimum(gt_a_max, dt_a_max) - np.maximum(gt_a_min, dt_a_min),
    )
    intersection = intersection_r * intersection_a

    gt_area = np.maximum(gt_r_max - gt_r_min, 0.0) * np.maximum(gt_a_max - gt_a_min, 0.0)
    dt_area = np.maximum(dt_r_max - dt_r_min, 0.0) * np.maximum(dt_a_max - dt_a_min, 0.0)
    union = gt_area + dt_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _get_score_thresholds(scores, num_gt):
    """Match K-Radar ``get_thresholds(..., num_sample_pts=41)``."""
    if num_gt <= 0 or len(scores) == 0:
        return []

    scores = sorted((float(score) for score in scores), reverse=True)
    current_recall = 0.0
    thresholds = []
    for index, score in enumerate(scores):
        left_recall = float(index + 1) / float(num_gt)
        if index < len(scores) - 1:
            right_recall = float(index + 2) / float(num_gt)
        else:
            right_recall = left_recall
        if (
            (right_recall - current_recall)
            < (current_recall - left_recall)
            and index < len(scores) - 1
        ):
            continue
        thresholds.append(score)
        current_recall += 1.0 / float(POLAR_NUM_SAMPLE_POINTS - 1)
    return thresholds


def _matched_tp_scores(overlaps, dt_scores, min_iou):
    """First K-Radar pass: choose the highest-score detection for each GT."""
    gt_count, dt_count = overlaps.shape
    assigned_detection = np.zeros((dt_count,), dtype=bool)
    matched_scores = []
    for gt_index in range(gt_count):
        best_detection = -1
        best_score = -np.inf
        for dt_index in range(dt_count):
            if assigned_detection[dt_index]:
                continue
            if (
                float(overlaps[gt_index, dt_index]) > float(min_iou)
                and float(dt_scores[dt_index]) > best_score
            ):
                best_detection = dt_index
                best_score = float(dt_scores[dt_index])
        if best_detection >= 0:
            assigned_detection[best_detection] = True
            matched_scores.append(best_score)
    return matched_scores


def _statistics_at_score_threshold(
        overlaps,
        dt_scores,
        min_iou,
        score_threshold,
    ):
    """Second K-Radar pass: maximize IoU per GT, then count remaining FP."""
    gt_count, dt_count = overlaps.shape
    assigned_detection = np.zeros((dt_count,), dtype=bool)
    ignored_by_score = dt_scores < float(score_threshold)
    tp = 0
    fn = 0

    for gt_index in range(gt_count):
        best_detection = -1
        best_overlap = 0.0
        for dt_index in range(dt_count):
            if assigned_detection[dt_index] or ignored_by_score[dt_index]:
                continue
            overlap = float(overlaps[gt_index, dt_index])
            if overlap > float(min_iou) and overlap > best_overlap:
                best_detection = dt_index
                best_overlap = overlap
        if best_detection < 0:
            fn += 1
        else:
            assigned_detection[best_detection] = True
            tp += 1

    fp = int(
        np.sum(
            (~assigned_detection)
            & (~ignored_by_score)
        )
    )
    return tp, fp, fn


def _evaluate_class(polar_frames, class_id, iou_thresholds):
    class_frames = []
    num_gt = 0

    for frame in polar_frames:
        gt_boxes = np.asarray(frame["gt_boxes"], dtype=np.float64).reshape(-1, 7)
        gt_labels = np.asarray(frame["gt_labels"], dtype=np.int64).reshape(-1)
        dt_boxes = np.asarray(frame["dt_boxes"], dtype=np.float64).reshape(-1, 7)
        dt_labels = np.asarray(frame["dt_labels"], dtype=np.int64).reshape(-1)
        dt_scores = np.asarray(frame["dt_scores"], dtype=np.float64).reshape(-1)

        gt_boxes = gt_boxes[gt_labels == int(class_id)]
        dt_mask = dt_labels == int(class_id)
        dt_boxes = dt_boxes[dt_mask]
        dt_scores = dt_scores[dt_mask]
        num_gt += int(gt_boxes.shape[0])
        class_frames.append(
            {
                "overlaps": polar_bev_overlaps(gt_boxes, dt_boxes),
                "dt_scores": dt_scores,
            }
        )

    ap_values = np.zeros((len(iou_thresholds),), dtype=np.float64)
    if num_gt <= 0:
        return ap_values, int(num_gt)

    for threshold_index, min_iou in enumerate(iou_thresholds):
        matched_scores = []
        for frame in class_frames:
            matched_scores.extend(
                _matched_tp_scores(
                    overlaps=frame["overlaps"],
                    dt_scores=frame["dt_scores"],
                    min_iou=min_iou,
                )
            )
        score_thresholds = _get_score_thresholds(
            scores=matched_scores,
            num_gt=num_gt,
        )
        precision = np.zeros(
            (POLAR_NUM_SAMPLE_POINTS,),
            dtype=np.float64,
        )
        for score_index, score_threshold in enumerate(
                score_thresholds[:POLAR_NUM_SAMPLE_POINTS]
            ):
            tp = 0
            fp = 0
            for frame in class_frames:
                frame_tp, frame_fp, _ = _statistics_at_score_threshold(
                    overlaps=frame["overlaps"],
                    dt_scores=frame["dt_scores"],
                    min_iou=min_iou,
                    score_threshold=score_threshold,
                )
                tp += frame_tp
                fp += frame_fp
            precision[score_index] = float(tp) / max(float(tp + fp), 1.0)

        for precision_index in range(POLAR_NUM_SAMPLE_POINTS - 2, -1, -1):
            precision[precision_index] = max(
                precision[precision_index],
                precision[precision_index + 1],
            )
        ap_values[threshold_index] = float(np.mean(precision) * 100.0)

    return ap_values, int(num_gt)


def compute_polar_ap_metrics(
        polar_frames,
        class_ids,
        class_name_map,
        iou_thresholds,
):
    """Compute Polar BEV AP and return flat metrics for logging/export."""
    iou_thresholds = np.asarray(iou_thresholds, dtype=np.float64).reshape(-1)
    if iou_thresholds.size == 0:
        return {}

    per_class = {}
    class_ap_values = []
    total_gt = 0
    for class_id in sorted(int(value) for value in class_ids):
        ap_values, num_gt = _evaluate_class(
            polar_frames=polar_frames,
            class_id=class_id,
            iou_thresholds=iou_thresholds,
        )
        class_name = str(class_name_map.get(class_id, class_id))
        class_ap_values.append(ap_values)
        total_gt += num_gt
        class_metrics = {
            "mAP": float(np.mean(ap_values)) if ap_values.size else 0.0,
            "num_gt": int(num_gt),
        }
        for threshold, ap_value in zip(iou_thresholds, ap_values):
            suffix = format_polar_iou_suffix(threshold)
            class_metrics[f"AP_{suffix}"] = float(ap_value)
        per_class[class_name] = class_metrics

    ap_matrix = np.asarray(class_ap_values, dtype=np.float64)
    mean_ap = np.mean(ap_matrix, axis=0) if ap_matrix.size else np.zeros((iou_thresholds.size,))
    metrics = {
        "polar_eval_enabled": True,
        "polar_iou_thresholds": [float(value) for value in iou_thresholds],
        "polar_recall_points": int(POLAR_NUM_SAMPLE_POINTS),
        "polar_bev_mAP": float(np.mean(mean_ap)) if mean_ap.size else 0.0,
        "polar_bev_num_gt": int(total_gt),
        "polar_per_class": per_class,
    }
    for threshold, ap_value in zip(iou_thresholds, mean_ap):
        suffix = format_polar_iou_suffix(threshold)
        metrics[f"polar_bev_mAP_{suffix}"] = float(ap_value)
        metrics[f"polar_bev_AP_{suffix}"] = float(ap_value)
        for class_name, class_metrics in per_class.items():
            metrics[f"polar_{class_name}_bev_AP_{suffix}"] = float(
                class_metrics[f"AP_{suffix}"]
            )
    return metrics
