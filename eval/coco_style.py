"""COCO-style AP evaluation for this project's rotated BEV / 3D boxes."""

import numpy as np

from .adapter import (
    OFFICIAL_CLASS_NAMES,
    load_official_overlap_functions,
    metric_boxes_to_official_3d_boxes,
    metric_boxes_to_official_bev_rbbox,
)


COCO_IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05, dtype=np.float64)
COCO_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101, dtype=np.float64)


def format_coco_iou_suffix(iou_value):
    return f"{float(iou_value):.2f}"


def _empty_overlap(gt_boxes, dt_boxes):
    return np.zeros((len(gt_boxes), len(dt_boxes)), dtype=np.float64)


def compute_bev_overlaps(gt_boxes, dt_boxes, bev_overlap_fn):
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)
    dt_boxes = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 7)
    if gt_boxes.shape[0] == 0 or dt_boxes.shape[0] == 0:
        return _empty_overlap(gt_boxes, dt_boxes)
    return bev_overlap_fn(
        metric_boxes_to_official_bev_rbbox(gt_boxes),
        metric_boxes_to_official_bev_rbbox(dt_boxes),
        -1,
    ).astype(np.float64)


def compute_3d_overlaps(gt_boxes, dt_boxes, d3_overlap_fn):
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)
    dt_boxes = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 7)
    if gt_boxes.shape[0] == 0 or dt_boxes.shape[0] == 0:
        return _empty_overlap(gt_boxes, dt_boxes)
    return d3_overlap_fn(
        metric_boxes_to_official_3d_boxes(gt_boxes),
        metric_boxes_to_official_3d_boxes(dt_boxes),
        -1,
    ).astype(np.float64)


def match_detections_for_threshold(overlaps, min_iou):
    overlaps = np.asarray(overlaps, dtype=np.float64)
    gt_count = int(overlaps.shape[0])
    dt_count = int(overlaps.shape[1])
    matched_gt = np.zeros((gt_count,), dtype=bool)
    tp = np.zeros((dt_count,), dtype=np.float64)
    fp = np.zeros((dt_count,), dtype=np.float64)

    for dt_idx in range(dt_count):
        best_gt_idx = -1
        best_iou = float(min_iou)
        for gt_idx in range(gt_count):
            if matched_gt[gt_idx]:
                continue
            iou_value = float(overlaps[gt_idx, dt_idx])
            if iou_value >= best_iou:
                best_iou = iou_value
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matched_gt[best_gt_idx] = True
            tp[dt_idx] = 1.0
        else:
            fp[dt_idx] = 1.0

    return tp, fp


def interpolated_ap_from_pr(recalls, precisions, recall_thresholds):
    recalls = np.asarray(recalls, dtype=np.float64).reshape(-1)
    precisions = np.asarray(precisions, dtype=np.float64).reshape(-1)
    recall_thresholds = np.asarray(recall_thresholds, dtype=np.float64).reshape(-1)

    if recalls.size == 0 or precisions.size == 0:
        return 0.0

    precision_envelope = precisions.copy()
    for idx in range(precision_envelope.size - 1, 0, -1):
        if precision_envelope[idx] > precision_envelope[idx - 1]:
            precision_envelope[idx - 1] = precision_envelope[idx]

    sampled_precisions = np.zeros((recall_thresholds.size,), dtype=np.float64)
    for idx, recall_threshold in enumerate(recall_thresholds):
        pos = np.searchsorted(recalls, recall_threshold, side="left")
        if pos < precision_envelope.size:
            sampled_precisions[idx] = precision_envelope[pos]

    return float(sampled_precisions.mean())


def summarize_class_ap(ap_by_iou, num_gt):
    ap_by_iou = np.asarray(ap_by_iou, dtype=np.float64)
    if int(num_gt) <= 0:
        return {
            "mAP": 0.0,
            "AP_0.50": 0.0,
            "AP_0.75": 0.0,
            "num_gt": 0,
        }

    return {
        "mAP": float(np.nanmean(ap_by_iou)),
        "AP_0.50": float(ap_by_iou[0]),
        "AP_0.75": float(ap_by_iou[5]),
        "num_gt": int(num_gt),
    }


def evaluate_coco_style_class(
        metric_frames,
        class_id,
        overlap_fn,
        iou_thresholds,
        recall_thresholds,
    ):
    iou_thresholds = np.asarray(iou_thresholds, dtype=np.float64).reshape(-1)
    recall_thresholds = np.asarray(recall_thresholds, dtype=np.float64).reshape(-1)
    detections = []
    num_gt = 0

    for frame in metric_frames:
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
        if dt_boxes.shape[0] == 0:
            continue

        order = np.argsort(-dt_scores, kind="mergesort")
        dt_boxes = dt_boxes[order]
        dt_scores = dt_scores[order]

        overlaps = overlap_fn(gt_boxes, dt_boxes)
        for threshold_index, threshold_value in enumerate(iou_thresholds):
            tp, fp = match_detections_for_threshold(overlaps, threshold_value)
            for det_index in range(dt_boxes.shape[0]):
                detections.append(
                    (
                        float(dt_scores[det_index]),
                        int(threshold_index),
                        float(tp[det_index]),
                        float(fp[det_index]),
                    )
                )

    if num_gt <= 0:
        return np.full((iou_thresholds.size,), np.nan, dtype=np.float64), 0

    if len(detections) == 0:
        return np.zeros((iou_thresholds.size,), dtype=np.float64), int(num_gt)

    detections.sort(key=lambda item: item[0], reverse=True)
    ap_by_iou = np.zeros((iou_thresholds.size,), dtype=np.float64)

    for threshold_index in range(iou_thresholds.size):
        dt_scores = np.array(
            [item[0] for item in detections if item[1] == threshold_index],
            dtype=np.float64,
        )
        tp = np.array(
            [item[2] for item in detections if item[1] == threshold_index],
            dtype=np.float64,
        )
        fp = np.array(
            [item[3] for item in detections if item[1] == threshold_index],
            dtype=np.float64,
        )
        if dt_scores.size == 0:
            continue

        order = np.argsort(-dt_scores, kind="mergesort")
        tp = tp[order]
        fp = fp[order]
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(float(num_gt), 1.0)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        ap_by_iou[threshold_index] = interpolated_ap_from_pr(
            recalls=recalls,
            precisions=precisions,
            recall_thresholds=recall_thresholds,
        )

    return ap_by_iou, int(num_gt)


def build_coco_result_text(per_class):
    lines = [
        "COCO-style AP (IoU=0.50:0.05:0.95, recall=101 points)",
    ]
    for class_name in sorted(per_class.keys()):
        class_metrics = per_class[class_name]
        bev = class_metrics["bev"]
        d3 = class_metrics["3d"]
        lines.append(
            f"{class_name}: "
            f"BEV mAP={bev['mAP']:.4f} AP50={bev['AP_0.50']:.4f} AP75={bev['AP_0.75']:.4f} "
            f"| 3D mAP={d3['mAP']:.4f} AP50={d3['AP_0.50']:.4f} AP75={d3['AP_0.75']:.4f} "
            f"| gt={bev['num_gt']}"
        )
    return "\n".join(lines)


def compute_coco_style_metrics(
        state,
        iou_backend="auto",
        iou_thresholds=None,
        recall_thresholds=None,
    ):
    metric_frames = state.get("metric_frames", [])
    if len(metric_frames) == 0:
        raise ValueError("No frames were collected for COCO-style evaluation.")

    if iou_thresholds is None:
        iou_thresholds = COCO_IOU_THRESHOLDS
    if recall_thresholds is None:
        recall_thresholds = COCO_RECALL_THRESHOLDS

    bev_overlap_fn, d3_overlap_fn, backend_used = load_official_overlap_functions(
        iou_backend
    )

    per_class = {}
    for class_id, class_name in sorted(OFFICIAL_CLASS_NAMES.items()):
        bev_ap_by_iou, bev_num_gt = evaluate_coco_style_class(
            metric_frames=metric_frames,
            class_id=class_id,
            overlap_fn=lambda gt_boxes, dt_boxes: compute_bev_overlaps(
                gt_boxes, dt_boxes, bev_overlap_fn
            ),
            iou_thresholds=iou_thresholds,
            recall_thresholds=recall_thresholds,
        )
        d3_ap_by_iou, d3_num_gt = evaluate_coco_style_class(
            metric_frames=metric_frames,
            class_id=class_id,
            overlap_fn=lambda gt_boxes, dt_boxes: compute_3d_overlaps(
                gt_boxes, dt_boxes, d3_overlap_fn
            ),
            iou_thresholds=iou_thresholds,
            recall_thresholds=recall_thresholds,
        )
        per_class[class_name] = {
            "bev": {
                **summarize_class_ap(bev_ap_by_iou, bev_num_gt),
                "iou": [float(value) for value in np.asarray(iou_thresholds, dtype=np.float64)],
                "ap": [0.0 if np.isnan(value) else float(value) for value in bev_ap_by_iou],
            },
            "3d": {
                **summarize_class_ap(d3_ap_by_iou, d3_num_gt),
                "iou": [float(value) for value in np.asarray(iou_thresholds, dtype=np.float64)],
                "ap": [0.0 if np.isnan(value) else float(value) for value in d3_ap_by_iou],
            },
        }

    valid_class_names = [
        class_name
        for class_name, metrics in per_class.items()
        if metrics["bev"]["num_gt"] > 0 or metrics["3d"]["num_gt"] > 0
    ]
    if len(valid_class_names) == 0:
        valid_class_names = sorted(per_class.keys())

    results = {
        "coco_style_eval_enabled": True,
        "coco_iou_backend_used": backend_used,
        "coco_num_eval_frames": len(metric_frames),
        "coco_iou_thresholds": [float(value) for value in np.asarray(iou_thresholds, dtype=np.float64)],
        "coco_recall_thresholds": [float(value) for value in np.asarray(recall_thresholds, dtype=np.float64)],
        "coco_per_class": per_class,
    }

    for metric_name in ("bev", "3d"):
        metric_maps = [
            float(per_class[class_name][metric_name]["mAP"])
            for class_name in valid_class_names
            if per_class[class_name][metric_name]["num_gt"] > 0
        ]
        metric_ap50 = [
            float(per_class[class_name][metric_name]["AP_0.50"])
            for class_name in valid_class_names
            if per_class[class_name][metric_name]["num_gt"] > 0
        ]
        metric_ap75 = [
            float(per_class[class_name][metric_name]["AP_0.75"])
            for class_name in valid_class_names
            if per_class[class_name][metric_name]["num_gt"] > 0
        ]

        results[f"coco_{metric_name}_mAP"] = (
            float(sum(metric_maps) / len(metric_maps)) if len(metric_maps) > 0 else 0.0
        )
        results[f"coco_{metric_name}_AP_0.50"] = (
            float(sum(metric_ap50) / len(metric_ap50)) if len(metric_ap50) > 0 else 0.0
        )
        results[f"coco_{metric_name}_AP_0.75"] = (
            float(sum(metric_ap75) / len(metric_ap75)) if len(metric_ap75) > 0 else 0.0
        )

        for class_name in sorted(per_class.keys()):
            class_metrics = per_class[class_name][metric_name]
            results[f"coco_{class_name}_{metric_name}_mAP"] = float(class_metrics["mAP"])
            results[f"coco_{class_name}_{metric_name}_AP_0.50"] = float(class_metrics["AP_0.50"])
            results[f"coco_{class_name}_{metric_name}_AP_0.75"] = float(class_metrics["AP_0.75"])
            for iou_value, ap_value in zip(class_metrics["iou"], class_metrics["ap"]):
                iou_suffix = format_coco_iou_suffix(iou_value)
                results[f"coco_{class_name}_{metric_name}_AP_{iou_suffix}"] = float(ap_value)

    results["coco_result_text"] = build_coco_result_text(per_class)
    return results
