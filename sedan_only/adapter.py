"""Official K-Radar evaluator wrapper for sedan-only metrics."""

import numpy as np
import torch

from eval.adapter import (
    compute_supplementary_detection_metrics,
    filter_official_per_class,
    filter_official_result_text,
    format_iou_suffix,
    load_official_eval_function,
    normalize_official_eval_version,
    official_metrics_for_classes,
)


SEDAN_ONLY_CLASS_NAMES = {
    0: "sed",
}


def empty_kitti_anno():
    return {
        "name": np.array([], dtype=str),
        "truncated": np.zeros((0,), dtype=np.float64),
        "occluded": np.zeros((0,), dtype=np.int64),
        "alpha": np.zeros((0,), dtype=np.float64),
        "bbox": np.zeros((0, 4), dtype=np.float64),
        "dimensions": np.zeros((0, 3), dtype=np.float64),
        "location": np.zeros((0, 3), dtype=np.float64),
        "rotation_y": np.zeros((0,), dtype=np.float64),
        "score": np.zeros((0,), dtype=np.float64),
    }


def metric_boxes_to_kitti_anno(boxes, labels, scores=None, is_prediction=False):
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if scores is not None and torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores is None:
        scores = np.zeros((boxes.shape[0],), dtype=np.float64)
    else:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    if boxes.shape[0] == 0:
        return empty_kitti_anno()

    names = np.array([SEDAN_ONLY_CLASS_NAMES[int(label)] for label in labels])
    x = boxes[:, 0]
    y = boxes[:, 1]
    z = boxes[:, 2]
    length = boxes[:, 3]
    width = boxes[:, 4]
    height = boxes[:, 5]
    yaw = boxes[:, 6]

    location = np.stack([y, z, x], axis=1)
    dimensions = np.stack([length, height, width], axis=1)

    if is_prediction:
        truncated = np.full((boxes.shape[0],), -1.0, dtype=np.float64)
        occluded = np.full((boxes.shape[0],), -1, dtype=np.int64)
    else:
        truncated = np.zeros((boxes.shape[0],), dtype=np.float64)
        occluded = np.zeros((boxes.shape[0],), dtype=np.int64)

    return {
        "name": names,
        "truncated": truncated,
        "occluded": occluded,
        "alpha": np.full((boxes.shape[0],), -10.0, dtype=np.float64),
        "bbox": np.tile(np.array([[50.0, 50.0, 150.0, 150.0]], dtype=np.float64), (boxes.shape[0], 1)),
        "dimensions": dimensions,
        "location": location,
        "rotation_y": yaw.astype(np.float64),
        "score": scores.astype(np.float64),
    }


def flatten_official_metrics(per_class):
    if len(per_class) == 0:
        return {}

    class_order = [
        class_name
        for _, class_name in sorted(SEDAN_ONLY_CLASS_NAMES.items())
        if class_name in per_class
    ]
    if len(class_order) == 0:
        class_order = sorted(per_class.keys())

    sample_metrics = per_class[class_order[0]]
    iou_values = list(sample_metrics.get("iou", []))
    flat_metrics = {}

    for class_name in class_order:
        class_metrics = per_class[class_name]
        for metric_name in ("bev", "3d"):
            metric_values = class_metrics.get(metric_name, [])
            for idx, metric_value in enumerate(metric_values):
                iou_suffix = format_iou_suffix(iou_values[idx])
                flat_metrics[f"official_{class_name}_{metric_name}_AP_{iou_suffix}"] = float(metric_value)

    for metric_name in ("bev", "3d"):
        for idx, iou_value in enumerate(iou_values):
            values = [float(per_class[class_name][metric_name][idx]) for class_name in class_order]
            iou_suffix = format_iou_suffix(iou_value)
            mean_value = sum(values) / len(values)
            flat_metrics[f"official_{metric_name}_mAP_{iou_suffix}"] = mean_value

    return flat_metrics


def compute_official_sedan_only_metrics(
        state,
        official_eval_enabled,
        official_eval_version,
        official_eval_iou_backend,
        official_eval_iou_mode,
        official_detection_metrics_enabled=True,
        detection_score_thresh=0.3,
    ):
    if not official_eval_enabled:
        return {}

    if len(state["official_gt_annos"]) == 0:
        raise ValueError("No frames were collected for sedan-only official evaluation.")

    official_eval_fn, official_iou_backend_used = load_official_eval_function(
        official_eval_version,
        official_eval_iou_backend,
    )
    official_result_text, official_per_class = official_metrics_for_classes(
        eval_fn=official_eval_fn,
        gt_annos=state["official_gt_annos"],
        dt_annos=state["official_dt_annos"],
        classes=[0],
        iou_mode=official_eval_iou_mode,
        class_name_map=SEDAN_ONLY_CLASS_NAMES,
    )
    official_result_text = filter_official_result_text(official_result_text)
    official_per_class = filter_official_per_class(official_per_class)
    official_flat_metrics = flatten_official_metrics(official_per_class)

    main_iou_suffix = {
        "easy": "0.3",
        "mod": "0.5",
        "hard": "0.7",
        "all": "0.3",
    }[official_eval_iou_mode]
    main_metric_key = f"official_bev_mAP_{main_iou_suffix}"

    official_eval_metrics = {
        "official_eval_version": normalize_official_eval_version(official_eval_version),
        "official_iou_mode": official_eval_iou_mode,
        "official_iou_backend_used": official_iou_backend_used,
        "official_num_eval_frames": len(state["official_gt_annos"]),
        "official_result_text": official_result_text,
        "official_per_class": official_per_class,
        "official_main_metric_key": main_metric_key,
        "official_main_metric_value": official_flat_metrics.get(main_metric_key, 0.0),
        "mAP": official_flat_metrics.get(main_metric_key, 0.0),
    }
    official_eval_metrics.update(official_flat_metrics)
    if official_detection_metrics_enabled:
        official_eval_metrics.update(
            compute_supplementary_detection_metrics(
                state=state,
                official_eval_iou_backend=official_eval_iou_backend,
                official_eval_iou_mode=official_eval_iou_mode,
                detection_score_thresh=detection_score_thresh,
            )
        )
    return official_eval_metrics
