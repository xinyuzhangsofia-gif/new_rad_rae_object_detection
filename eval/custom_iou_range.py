"""Custom IoU-range AP evaluation for this project's rotated BEV / 3D boxes."""

import numpy as np

from .adapter import (
    OFFICIAL_CLASS_NAMES,
    load_official_overlap_functions,
    load_rotate_iou_eval_function,
    match_frame_detections,
)
from .coco_style import (
    COCO_RECALL_THRESHOLDS,
    compute_bev_overlaps,
    compute_3d_overlaps,
    evaluate_coco_style_class,
)


DEFAULT_CUSTOM_IOU_THRESHOLDS = np.arange(0.25, 0.51, 0.05, dtype=np.float64)


def format_custom_iou_suffix(iou_value):
    return f"{float(iou_value):.2f}"


def summarize_dynamic_class_ap(ap_by_iou, num_gt, iou_thresholds):
    ap_by_iou = np.asarray(ap_by_iou, dtype=np.float64).reshape(-1)

    if int(num_gt) <= 0:
        return {
            "mAP": 0.0,
            "num_gt": 0,
        }

    return {
        "mAP": float(np.nanmean(ap_by_iou)),
        "num_gt": int(num_gt),
    }


def build_custom_iou_result_text(per_class, iou_thresholds):
    threshold_text = ", ".join(
        format_custom_iou_suffix(iou_value)
        for iou_value in np.asarray(iou_thresholds, dtype=np.float64).reshape(-1)
    )
    lines = [
        f"Custom IoU-range AP (IoU={threshold_text}, recall=101 points)",
    ]
    for class_name in sorted(per_class.keys()):
        class_metrics = per_class[class_name]
        bev = class_metrics["bev"]
        d3 = class_metrics["3d"]
        lines.append(
            f"{class_name}: "
            f"BEV mAP={bev['mAP']:.4f} | 3D mAP={d3['mAP']:.4f} | "
            f"P={class_metrics['precision']:.4f} R={class_metrics['recall']:.4f} F1={class_metrics['f1']:.4f} | "
            f"gt={bev['num_gt']}"
        )
    return "\n".join(lines)


def compute_custom_iou_detection_metrics(
        metric_frames,
        iou_backend,
        iou_thresholds,
        detection_score_thresh,
    ):
    rotate_iou_eval_fn, matching_backend_used = load_rotate_iou_eval_function(
        iou_backend
    )

    overall_precision_values = []
    overall_recall_values = []
    overall_f1_values = []
    per_class_metric_values = {
        class_name: {
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for class_name in OFFICIAL_CLASS_NAMES.values()
    }

    for min_overlap in np.asarray(iou_thresholds, dtype=np.float64).reshape(-1):
        totals = {"tp": 0, "fp": 0, "fn": 0}
        per_class_totals = {}
        for frame in metric_frames:
            frame_totals, frame_per_class = match_frame_detections(
                gt_boxes=frame["gt_boxes"],
                gt_labels=frame["gt_labels"],
                dt_boxes=frame["dt_boxes"],
                dt_labels=frame["dt_labels"],
                dt_scores=frame["dt_scores"],
                min_overlap=float(min_overlap),
                detection_score_thresh=detection_score_thresh,
                rotate_iou_eval_fn=rotate_iou_eval_fn,
            )
            for key in totals:
                totals[key] += int(frame_totals[key])
            for class_name, class_totals in frame_per_class.items():
                if class_name not in per_class_totals:
                    per_class_totals[class_name] = {"tp": 0, "fp": 0, "fn": 0}
                for key in totals:
                    per_class_totals[class_name][key] += int(class_totals[key])

        tp = int(totals["tp"])
        fp = int(totals["fp"])
        fn = int(totals["fn"])
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float((2.0 * precision * recall) / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        overall_precision_values.append(precision)
        overall_recall_values.append(recall)
        overall_f1_values.append(f1)

        for class_name in OFFICIAL_CLASS_NAMES.values():
            class_totals = per_class_totals.get(class_name, {"tp": 0, "fp": 0, "fn": 0})
            class_tp = int(class_totals["tp"])
            class_fp = int(class_totals["fp"])
            class_fn = int(class_totals["fn"])
            class_precision = (
                float(class_tp / (class_tp + class_fp))
                if (class_tp + class_fp) > 0
                else 0.0
            )
            class_recall = (
                float(class_tp / (class_tp + class_fn))
                if (class_tp + class_fn) > 0
                else 0.0
            )
            class_f1 = (
                float((2.0 * class_precision * class_recall) / (class_precision + class_recall))
                if (class_precision + class_recall) > 0
                else 0.0
            )
            per_class_metric_values[class_name]["precision"].append(class_precision)
            per_class_metric_values[class_name]["recall"].append(class_recall)
            per_class_metric_values[class_name]["f1"].append(class_f1)

    per_class_metrics = {}
    for class_name, class_values in per_class_metric_values.items():
        per_class_metrics[class_name] = {
            "precision": float(np.mean(class_values["precision"])) if len(class_values["precision"]) > 0 else 0.0,
            "recall": float(np.mean(class_values["recall"])) if len(class_values["recall"]) > 0 else 0.0,
            "f1": float(np.mean(class_values["f1"])) if len(class_values["f1"]) > 0 else 0.0,
        }

    return {
        "custom_iou_detection_score_threshold": float(detection_score_thresh),
        "custom_iou_detection_match_backend_used": matching_backend_used,
        "custom_iou_precision": float(np.mean(overall_precision_values)) if len(overall_precision_values) > 0 else 0.0,
        "custom_iou_recall": float(np.mean(overall_recall_values)) if len(overall_recall_values) > 0 else 0.0,
        "custom_iou_f1": float(np.mean(overall_f1_values)) if len(overall_f1_values) > 0 else 0.0,
        "custom_iou_detection_per_class": per_class_metrics,
    }


def compute_custom_iou_range_metrics(
        state,
        iou_backend="auto",
        iou_thresholds=None,
        recall_thresholds=None,
        detection_score_thresh=0.3,
    ):
    metric_frames = state.get("metric_frames", [])
    if len(metric_frames) == 0:
        raise ValueError("No frames were collected for custom IoU-range evaluation.")

    if iou_thresholds is None:
        iou_thresholds = DEFAULT_CUSTOM_IOU_THRESHOLDS
    if recall_thresholds is None:
        recall_thresholds = COCO_RECALL_THRESHOLDS

    iou_thresholds = np.asarray(iou_thresholds, dtype=np.float64).reshape(-1)
    recall_thresholds = np.asarray(recall_thresholds, dtype=np.float64).reshape(-1)

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
            "bev": summarize_dynamic_class_ap(
                bev_ap_by_iou,
                bev_num_gt,
                iou_thresholds,
            ),
            "3d": summarize_dynamic_class_ap(
                d3_ap_by_iou,
                d3_num_gt,
                iou_thresholds,
            ),
        }

    detection_metrics = compute_custom_iou_detection_metrics(
        metric_frames=metric_frames,
        iou_backend=iou_backend,
        iou_thresholds=iou_thresholds,
        detection_score_thresh=detection_score_thresh,
    )

    valid_class_names = [
        class_name
        for class_name, metrics in per_class.items()
        if metrics["bev"]["num_gt"] > 0 or metrics["3d"]["num_gt"] > 0
    ]
    if len(valid_class_names) == 0:
        valid_class_names = sorted(per_class.keys())

    results = {
        "custom_iou_range_eval_enabled": True,
        "custom_iou_backend_used": backend_used,
        "custom_iou_num_eval_frames": len(metric_frames),
        "custom_iou_thresholds": [float(value) for value in iou_thresholds],
        "custom_iou_recall_thresholds": [float(value) for value in recall_thresholds],
        "custom_iou_per_class": per_class,
    }
    results.update(detection_metrics)

    for metric_name in ("bev", "3d"):
        metric_maps = [
            float(per_class[class_name][metric_name]["mAP"])
            for class_name in valid_class_names
            if per_class[class_name][metric_name]["num_gt"] > 0
        ]
        results[f"custom_iou_{metric_name}_mAP"] = (
            float(sum(metric_maps) / len(metric_maps)) if len(metric_maps) > 0 else 0.0
        )

        for class_name in sorted(per_class.keys()):
            class_metrics = per_class[class_name][metric_name]
            results[f"custom_iou_{class_name}_{metric_name}_mAP"] = float(
                class_metrics["mAP"]
            )
        for class_name in sorted(detection_metrics["custom_iou_detection_per_class"].keys()):
            class_detection_metrics = detection_metrics["custom_iou_detection_per_class"][class_name]
            results[f"custom_iou_{class_name}_precision"] = float(class_detection_metrics["precision"])
            results[f"custom_iou_{class_name}_recall"] = float(class_detection_metrics["recall"])
            results[f"custom_iou_{class_name}_f1"] = float(class_detection_metrics["f1"])

    results["custom_iou_result_text"] = build_custom_iou_result_text(
        {
            class_name: {
                **class_metrics,
                **detection_metrics["custom_iou_detection_per_class"].get(class_name, {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }),
            }
            for class_name, class_metrics in per_class.items()
        },
        iou_thresholds,
    )
    return results
