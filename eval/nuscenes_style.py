"""nuScenes-style detection metrics adapted to this project's 3D boxes.

This keeps the nuScenes detection core ideas:
1. class-wise AP based on BEV center-distance matching,
2. AP averaged over distance thresholds {0.5, 1.0, 2.0, 4.0} meters,
3. TP error metrics computed at the 2.0 m match threshold.

This project does not provide velocity / attribute targets, so the adapted
implementation reports mAP, mATE, mASE, and mAOE only.
"""

import math

import numpy as np

from .adapter import OFFICIAL_CLASS_NAMES


NUSCENES_DIST_THRESHOLDS = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
NUSCENES_TP_DIST_THRESHOLD = 2.0
NUSCENES_MIN_RECALL = 0.1
NUSCENES_MIN_PRECISION = 0.1
NUSCENES_RECALL_GRID = np.linspace(0.0, 1.0, 101, dtype=np.float64)
NUSCENES_CLASS_RANGES = {
    "sed": 50.0,
    "bus": 50.0,
}


def format_distance_suffix(distance_value):
    return f"{float(distance_value):.1f}m"


def nanmean_or_default(values, default=np.nan):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float(default)
    return float(valid.mean())


def cumulative_mean(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    if np.all(np.isnan(values)):
        return np.ones((values.size,), dtype=np.float64)

    sum_values = np.nancumsum(values.astype(np.float64))
    count_values = np.cumsum(~np.isnan(values))
    return np.divide(
        sum_values,
        count_values,
        out=np.zeros_like(sum_values),
        where=count_values != 0,
    )


def center_distance_xy(gt_box, dt_box):
    return float(np.linalg.norm(np.asarray(dt_box[:2]) - np.asarray(gt_box[:2])))


def scale_iou_aligned(gt_box, dt_box):
    gt_size = np.asarray(gt_box[3:6], dtype=np.float64)
    dt_size = np.asarray(dt_box[3:6], dtype=np.float64)
    if np.any(gt_size <= 0.0) or np.any(dt_size <= 0.0):
        return 0.0

    min_size = np.minimum(gt_size, dt_size)
    intersection = float(np.prod(min_size))
    gt_volume = float(np.prod(gt_size))
    dt_volume = float(np.prod(dt_size))
    union = gt_volume + dt_volume - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def yaw_diff_rad(gt_box, dt_box, period=2.0 * math.pi):
    yaw_gt = float(gt_box[6])
    yaw_dt = float(dt_box[6])
    diff = (yaw_gt - yaw_dt + (period / 2.0)) % period - (period / 2.0)
    if diff > math.pi:
        diff -= 2.0 * math.pi
    return abs(float(diff))


def boxes_in_range(boxes, max_range):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    if boxes.shape[0] == 0 or max_range is None or not np.isfinite(max_range):
        return boxes, np.ones((boxes.shape[0],), dtype=bool)
    ranges = np.linalg.norm(boxes[:, :2], axis=1)
    keep = ranges <= float(max_range)
    return boxes[keep], keep


def no_predictions_metric_data(recall_grid):
    recall_grid = np.asarray(recall_grid, dtype=np.float64).reshape(-1)
    zeros = np.zeros((recall_grid.size,), dtype=np.float64)
    ones = np.ones((recall_grid.size,), dtype=np.float64)
    return {
        "recall": recall_grid.copy(),
        "precision": zeros.copy(),
        "confidence": zeros.copy(),
        "trans_err": ones.copy(),
        "scale_err": ones.copy(),
        "orient_err": ones.copy(),
    }


def prepare_nuscenes_class_data(metric_frames, class_id, class_name, class_ranges):
    gt_by_frame = []
    predictions = []
    num_gt = 0
    max_range = class_ranges.get(class_name, np.inf)

    for frame_idx, frame in enumerate(metric_frames):
        gt_boxes = np.asarray(frame["gt_boxes"], dtype=np.float64).reshape(-1, 7)
        gt_labels = np.asarray(frame["gt_labels"], dtype=np.int64).reshape(-1)
        dt_boxes = np.asarray(frame["dt_boxes"], dtype=np.float64).reshape(-1, 7)
        dt_labels = np.asarray(frame["dt_labels"], dtype=np.int64).reshape(-1)
        dt_scores = np.asarray(frame["dt_scores"], dtype=np.float64).reshape(-1)

        gt_boxes = gt_boxes[gt_labels == int(class_id)]
        dt_mask = dt_labels == int(class_id)
        dt_boxes = dt_boxes[dt_mask]
        dt_scores = dt_scores[dt_mask]

        gt_boxes, _ = boxes_in_range(gt_boxes, max_range)
        dt_boxes, dt_keep = boxes_in_range(dt_boxes, max_range)
        dt_scores = dt_scores[dt_keep]

        gt_by_frame.append(gt_boxes)
        num_gt += int(gt_boxes.shape[0])

        if dt_boxes.shape[0] == 0:
            continue

        order = np.argsort(-dt_scores, kind="mergesort")
        for det_index in order:
            predictions.append(
                (
                    float(dt_scores[det_index]),
                    int(frame_idx),
                    dt_boxes[det_index].astype(np.float64, copy=False),
                )
            )

    predictions.sort(key=lambda item: item[0], reverse=True)
    return gt_by_frame, predictions, int(num_gt)


def accumulate_nuscenes_class(gt_by_frame, predictions, num_gt, dist_threshold, recall_grid):
    recall_grid = np.asarray(recall_grid, dtype=np.float64).reshape(-1)
    if num_gt <= 0:
        return None
    if len(predictions) == 0:
        return no_predictions_metric_data(recall_grid)

    taken_by_frame = [
        np.zeros((gt_boxes.shape[0],), dtype=bool)
        for gt_boxes in gt_by_frame
    ]

    tp = []
    fp = []
    conf = []
    match_data = {
        "trans_err": [],
        "scale_err": [],
        "orient_err": [],
        "conf": [],
    }

    for score, frame_idx, pred_box in predictions:
        gt_boxes = gt_by_frame[frame_idx]
        taken = taken_by_frame[frame_idx]

        min_dist = np.inf
        match_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes):
            if taken[gt_idx]:
                continue
            distance_value = center_distance_xy(gt_box, pred_box)
            if distance_value < min_dist:
                min_dist = distance_value
                match_gt_idx = gt_idx

        if match_gt_idx is not None and min_dist < float(dist_threshold):
            taken[match_gt_idx] = True
            gt_box = gt_boxes[match_gt_idx]
            tp.append(1.0)
            fp.append(0.0)
            conf.append(float(score))
            match_data["trans_err"].append(center_distance_xy(gt_box, pred_box))
            match_data["scale_err"].append(1.0 - scale_iou_aligned(gt_box, pred_box))
            match_data["orient_err"].append(yaw_diff_rad(gt_box, pred_box))
            match_data["conf"].append(float(score))
        else:
            tp.append(0.0)
            fp.append(1.0)
            conf.append(float(score))

    if len(match_data["conf"]) == 0:
        return no_predictions_metric_data(recall_grid)

    tp = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp = np.cumsum(np.asarray(fp, dtype=np.float64))
    conf = np.asarray(conf, dtype=np.float64)

    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / float(num_gt)
    precision_interp = np.interp(recall_grid, recall, precision, right=0.0)
    confidence_interp = np.interp(recall_grid, recall, conf, right=0.0)

    metric_data = {
        "recall": recall_grid.copy(),
        "precision": precision_interp,
        "confidence": confidence_interp,
    }

    match_conf = np.asarray(match_data["conf"], dtype=np.float64)
    for key in ("trans_err", "scale_err", "orient_err"):
        cumulative = cumulative_mean(np.asarray(match_data[key], dtype=np.float64))
        metric_data[key] = np.interp(
            confidence_interp[::-1],
            match_conf[::-1],
            cumulative[::-1],
        )[::-1]

    return metric_data


def calc_ap(metric_data, min_recall, min_precision):
    if metric_data is None:
        return np.nan
    precision = np.copy(metric_data["precision"])
    precision = precision[int(round(100 * float(min_recall))) + 1:]
    precision -= float(min_precision)
    precision[precision < 0.0] = 0.0
    return float(np.mean(precision)) / (1.0 - float(min_precision))


def calc_tp(metric_data, min_recall, metric_name):
    if metric_data is None:
        return np.nan

    first_ind = int(round(100 * float(min_recall))) + 1
    nonzero_confidence = np.flatnonzero(metric_data["confidence"] > 0.0)
    last_ind = int(nonzero_confidence[-1]) if nonzero_confidence.size > 0 else -1
    if last_ind < first_ind:
        return 1.0
    return float(np.mean(metric_data[metric_name][first_ind:last_ind + 1]))


def build_nuscenes_result_text(per_class, overall, dist_thresholds):
    overall_ap_text = " ".join(
        f"AP@{format_distance_suffix(dist_threshold)}="
        f"{overall[f'AP_{format_distance_suffix(dist_threshold)}']:.4f}"
        for dist_threshold in dist_thresholds
    )
    lines = [
        "nuScenes-style (adapted): center-distance AP with TP metrics mATE/mASE/mAOE.",
        (
            f"Overall: mAP={overall['mAP']:.4f} "
            f"{overall_ap_text} "
            f"| mATE={overall['mATE']:.4f} mASE={overall['mASE']:.4f} mAOE={overall['mAOE']:.4f}"
        ),
    ]
    for class_name in sorted(per_class.keys()):
        class_metrics = per_class[class_name]
        class_ap_text = " ".join(
            f"AP@{format_distance_suffix(dist_threshold)}="
            f"{class_metrics[f'AP_{format_distance_suffix(dist_threshold)}']:.4f}"
            for dist_threshold in dist_thresholds
        )
        lines.append(
            (
                f"{class_name}: mAP={class_metrics['mAP']:.4f} "
                f"{class_ap_text} "
                f"| ATE={class_metrics['ATE']:.4f} ASE={class_metrics['ASE']:.4f} AOE={class_metrics['AOE']:.4f} "
                f"| gt={class_metrics['num_gt']}"
            )
        )
    lines.append("Velocity and attribute metrics are omitted because this project does not provide them.")
    return "\n".join(lines)


def compute_nuscenes_style_metrics(
        state,
        class_ranges=None,
        dist_thresholds=None,
        tp_dist_threshold=NUSCENES_TP_DIST_THRESHOLD,
        min_recall=NUSCENES_MIN_RECALL,
        min_precision=NUSCENES_MIN_PRECISION,
    ):
    metric_frames = state.get("metric_frames", [])
    if len(metric_frames) == 0:
        raise ValueError("No frames were collected for nuScenes-style evaluation.")

    if class_ranges is None:
        class_ranges = dict(NUSCENES_CLASS_RANGES)
    else:
        class_ranges = {
            str(key): float(value)
            for key, value in dict(class_ranges).items()
        }
    if dist_thresholds is None:
        dist_thresholds = NUSCENES_DIST_THRESHOLDS
    dist_thresholds = np.asarray(dist_thresholds, dtype=np.float64).reshape(-1)
    if dist_thresholds.size == 0:
        raise ValueError("nuScenes-style dist_thresholds must not be empty.")
    if not np.any(np.isclose(dist_thresholds, float(tp_dist_threshold))):
        raise ValueError(
            f"nuScenes-style tp_dist_threshold={tp_dist_threshold} must be in dist_thresholds={dist_thresholds.tolist()}."
        )

    per_class = {}
    overall_ap_by_threshold = {format_distance_suffix(dist): [] for dist in dist_thresholds}
    overall_ate = []
    overall_ase = []
    overall_aoe = []

    for class_id, class_name in sorted(OFFICIAL_CLASS_NAMES.items()):
        gt_by_frame, predictions, num_gt = prepare_nuscenes_class_data(
            metric_frames=metric_frames,
            class_id=class_id,
            class_name=class_name,
            class_ranges=class_ranges,
        )

        class_metrics = {"num_gt": int(num_gt)}
        ap_values = []
        tp_metric_data = None
        for dist_threshold in dist_thresholds:
            metric_data = accumulate_nuscenes_class(
                gt_by_frame=gt_by_frame,
                predictions=predictions,
                num_gt=num_gt,
                dist_threshold=dist_threshold,
                recall_grid=NUSCENES_RECALL_GRID,
            )
            dist_suffix = format_distance_suffix(dist_threshold)
            ap_value = calc_ap(
                metric_data=metric_data,
                min_recall=min_recall,
                min_precision=min_precision,
            )
            class_metrics[f"AP_{dist_suffix}"] = ap_value
            ap_values.append(ap_value)
            if np.isclose(dist_threshold, float(tp_dist_threshold)):
                tp_metric_data = metric_data

        class_metrics["mAP"] = nanmean_or_default(ap_values, default=np.nan)
        class_metrics["ATE"] = calc_tp(tp_metric_data, min_recall, "trans_err")
        class_metrics["ASE"] = calc_tp(tp_metric_data, min_recall, "scale_err")
        class_metrics["AOE"] = calc_tp(tp_metric_data, min_recall, "orient_err")
        per_class[class_name] = class_metrics

        if num_gt > 0:
            for dist_threshold in dist_thresholds:
                dist_suffix = format_distance_suffix(dist_threshold)
                overall_ap_by_threshold[dist_suffix].append(class_metrics[f"AP_{dist_suffix}"])
            overall_ate.append(class_metrics["ATE"])
            overall_ase.append(class_metrics["ASE"])
            overall_aoe.append(class_metrics["AOE"])

    overall = {}
    for dist_threshold in dist_thresholds:
        dist_suffix = format_distance_suffix(dist_threshold)
        overall[f"AP_{dist_suffix}"] = nanmean_or_default(
            overall_ap_by_threshold[dist_suffix],
            default=0.0,
        )
    overall["mAP"] = nanmean_or_default(list(overall.values()), default=0.0)
    overall["mATE"] = nanmean_or_default(overall_ate, default=0.0)
    overall["mASE"] = nanmean_or_default(overall_ase, default=0.0)
    overall["mAOE"] = nanmean_or_default(overall_aoe, default=0.0)

    flat_metrics = {
        "nuscenes_num_eval_frames": len(metric_frames),
        "nuscenes_distance_thresholds": [float(value) for value in dist_thresholds.tolist()],
        "nuscenes_tp_distance_threshold": float(tp_dist_threshold),
        "nuscenes_min_recall": float(min_recall),
        "nuscenes_min_precision": float(min_precision),
        "nuscenes_class_ranges": {
            class_name: float(class_ranges.get(class_name, np.inf))
            for class_name in sorted(set(OFFICIAL_CLASS_NAMES.values()))
        },
        "nuscenes_per_class": per_class,
        "nuscenes_mAP": overall["mAP"],
        "nuscenes_mATE": overall["mATE"],
        "nuscenes_mASE": overall["mASE"],
        "nuscenes_mAOE": overall["mAOE"],
    }

    for dist_threshold in dist_thresholds:
        dist_suffix = format_distance_suffix(dist_threshold)
        flat_metrics[f"nuscenes_AP_{dist_suffix}"] = overall[f"AP_{dist_suffix}"]

    for class_name, class_metrics in per_class.items():
        flat_metrics[f"nuscenes_{class_name}_mAP"] = class_metrics["mAP"]
        flat_metrics[f"nuscenes_{class_name}_ATE"] = class_metrics["ATE"]
        flat_metrics[f"nuscenes_{class_name}_ASE"] = class_metrics["ASE"]
        flat_metrics[f"nuscenes_{class_name}_AOE"] = class_metrics["AOE"]
        flat_metrics[f"nuscenes_{class_name}_num_gt"] = int(class_metrics["num_gt"])
        for dist_threshold in dist_thresholds:
            dist_suffix = format_distance_suffix(dist_threshold)
            flat_metrics[f"nuscenes_{class_name}_AP_{dist_suffix}"] = class_metrics[f"AP_{dist_suffix}"]

    flat_metrics["nuscenes_result_text"] = build_nuscenes_result_text(
        per_class=per_class,
        overall=overall,
        dist_thresholds=dist_thresholds,
    )
    return flat_metrics
