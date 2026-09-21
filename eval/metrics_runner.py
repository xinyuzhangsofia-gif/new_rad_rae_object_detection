"""K-Radar annotation collection and metric execution."""

import numpy as np
import torch
import tqdm

from data.coordinates import (
    SCOPE_FULL,
)
from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    require_cartesian_data,
)
from data.dataloader import prepare_model_inputs
from eval.adapter import (
    compute_official_kradar_style_metrics,
    load_official_eval_function,
    metric_boxes_to_kitti_anno,
)
from eval.coco_style import compute_coco_style_metrics
from eval.custom_iou_range import compute_custom_iou_range_metrics
from eval.distance_quartiles import (
    derive_gt_distance_quartile_bins,
    filter_kradar_eval_state_by_quartile,
    normalize_distance_quartile_bins,
)
from eval.nuscenes_style import compute_nuscenes_style_metrics

from eval.decoding import (
    filter_predictions_to_scope,
)
from eval.inference import predict_batch as predict_detection_batch

__all__ = [
    'init_kradar_eval_state',
    'suppress_predictions_near_ignore_boxes',
    'append_frame_annos_for_kradar_eval',
    'collect_kradar_annos',
    'run_kradar_eval_revised',
    'evaluate_checkpoint_with_kradar_revised',
    'evaluate_train_val_iou'
]


def init_kradar_eval_state():
    return {
        "official_gt_annos": [],
        "official_dt_annos": [],
        "metric_frames": [],
    }


def _append_neutral_gt_to_official_anno(
        valid_anno,
        neutral_boxes,
        neutral_labels,
        official_class_name_map,
    ):
    """Append same-class neutral GT after valid GT for the official evaluator."""
    neutral_anno = metric_boxes_to_kitti_anno(
        boxes=neutral_boxes,
        labels=neutral_labels,
        is_prediction=False,
        class_name_map=official_class_name_map,
    )
    if neutral_anno["name"].shape[0] == 0:
        return valid_anno
    neutral_anno["occluded"][:] = 3
    return {
        key: np.concatenate([valid_anno[key], neutral_anno[key]], axis=0)
        for key in valid_anno
    }


def suppress_predictions_near_ignore_boxes(
        pred_boxes,
        pred_scores,
        pred_labels,
        gt_ignore_boxes_raw,
        gt_ignore_metric_boxes,
        scope_mode,
        full_rae_shape,
        expand_ratio=1.5,
        margin=1.0,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
    ):
    box_coordinate_mode = require_cartesian_data(box_coordinate_mode)
    ignore_source = gt_ignore_metric_boxes
    if ignore_source is None or pred_boxes.numel() == 0:
        return pred_boxes, pred_scores, pred_labels, 0

    ignore_boxes = ignore_source.to(pred_boxes.device)
    if ignore_boxes.numel() == 0:
        return pred_boxes, pred_scores, pred_labels, 0

    pred_center_y = pred_boxes[:, 0]
    pred_center_x = pred_boxes[:, 1]
    keep = torch.ones((pred_boxes.shape[0],), dtype=torch.bool, device=pred_boxes.device)

    for ignore_box in ignore_boxes:
        center_y = ignore_box[0]
        center_x = ignore_box[1]
        half_h = (
            ignore_box[3].abs().clamp(min=1e-4)
            * 0.5
            * float(expand_ratio)
        ) + float(margin)
        half_w = (
            ignore_box[4].abs().clamp(min=1e-4)
            * 0.5
            * float(expand_ratio)
        ) + float(margin)
        inside = (
            (pred_center_y >= (center_y - half_h))
            & (pred_center_y <= (center_y + half_h))
            & (pred_center_x >= (center_x - half_w))
            & (pred_center_x <= (center_x + half_w))
        )
        keep &= ~inside

    suppressed_predictions = int((~keep).sum().item())
    return (
        pred_boxes[keep],
        pred_scores[keep],
        pred_labels[keep],
        suppressed_predictions,
    )


def append_frame_annos_for_kradar_eval(
        state,
        batch,
        batch_index,
        frame_predictions,
        device,
        num_classes,
        scope_mode,
        official_class_name_map,
        eval_ignore_suppress_enabled=False,
        eval_ignore_expand_ratio=1.5,
        eval_ignore_suppress_margin=1.0,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
    ):
    box_coordinate_mode = require_cartesian_data(box_coordinate_mode)
    full_rae_shape = batch["full_rae_shape"][batch_index]
    frame_predictions = filter_predictions_to_scope(
        frame_predictions=frame_predictions,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
        box_coordinate_mode=box_coordinate_mode,
    )
    if eval_ignore_suppress_enabled:
        gt_ignore_boxes_raw_list = batch.get("gt_ignore_boxes_raw")
        gt_ignore_metric_boxes_list = batch.get("gt_ignore_metric_boxes")
        gt_ignore_boxes_raw = None
        gt_ignore_metric_boxes = None
        if gt_ignore_boxes_raw_list is not None:
            gt_ignore_boxes_raw = gt_ignore_boxes_raw_list[batch_index]
        if gt_ignore_metric_boxes_list is not None:
            gt_ignore_metric_boxes = gt_ignore_metric_boxes_list[batch_index]
        (
            frame_predictions["boxes"],
            frame_predictions["scores"],
            frame_predictions["labels"],
            suppressed_predictions,
        ) = suppress_predictions_near_ignore_boxes(
            pred_boxes=frame_predictions["boxes"],
            pred_scores=frame_predictions["scores"],
            pred_labels=frame_predictions["labels"],
            gt_ignore_boxes_raw=gt_ignore_boxes_raw,
            gt_ignore_metric_boxes=gt_ignore_metric_boxes,
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
            expand_ratio=eval_ignore_expand_ratio,
            margin=eval_ignore_suppress_margin,
            box_coordinate_mode=box_coordinate_mode,
        )
        state["eval_ignore_suppressed_predictions"] += suppressed_predictions

    gt_labels_all = batch["gt_labels"][batch_index].to(device)
    valid_gt = gt_labels_all < num_classes
    gt_labels = gt_labels_all[valid_gt]
    neutral_labels_list = batch.get("gt_override_ignore_labels")
    neutral_metric_boxes_list = batch.get("gt_override_ignore_metric_boxes")
    neutral_labels = (
        torch.zeros((0,), dtype=torch.long, device=device)
        if neutral_labels_list is None
        else neutral_labels_list[batch_index].to(device)
    )
    valid_neutral = neutral_labels < num_classes
    neutral_labels = neutral_labels[valid_neutral]

    gt_metric_boxes_all = batch["gt_metric_boxes"][batch_index].to(device)
    gt_metric_boxes = gt_metric_boxes_all[valid_gt]
    pred_metric_boxes = frame_predictions["boxes"]
    neutral_metric_boxes = (
        torch.zeros((0, 7), dtype=torch.float32, device=device)
        if neutral_metric_boxes_list is None
        else neutral_metric_boxes_list[batch_index].to(device)[valid_neutral]
    )

    valid_official_gt_anno = metric_boxes_to_kitti_anno(
        boxes=gt_metric_boxes.detach().cpu(),
        labels=gt_labels.detach().cpu(),
        is_prediction=False,
        class_name_map=official_class_name_map,
    )
    state["official_gt_annos"].append(
        _append_neutral_gt_to_official_anno(
            valid_anno=valid_official_gt_anno,
            neutral_boxes=neutral_metric_boxes.detach().cpu(),
            neutral_labels=neutral_labels.detach().cpu(),
            official_class_name_map=official_class_name_map,
        )
    )
    state["official_dt_annos"].append(
        metric_boxes_to_kitti_anno(
            boxes=pred_metric_boxes.detach().cpu(),
            labels=frame_predictions["labels"].detach().cpu(),
            scores=frame_predictions["scores"].detach().cpu(),
            is_prediction=True,
            class_name_map=official_class_name_map,
        )
    )
    state["metric_frames"].append(
        {
            "gt_boxes": gt_metric_boxes.detach().cpu().numpy(),
            "gt_labels": gt_labels.detach().cpu().numpy(),
            "dt_boxes": pred_metric_boxes.detach().cpu().numpy(),
            "dt_labels": frame_predictions["labels"].detach().cpu().numpy(),
            "dt_scores": frame_predictions["scores"].detach().cpu().numpy(),
            "neutral_gt_boxes": neutral_metric_boxes.detach().cpu().numpy(),
            "neutral_gt_labels": neutral_labels.detach().cpu().numpy(),
        }
    )


@torch.no_grad()
def collect_kradar_annos(
        model,
        dataloader,
        device,
        num_classes,
        official_class_name_map,
        prepare_model_inputs,
        max_detections=64,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        yolox_nms_iou=0.65,
        ap_score_thresh=0.01,
        score_thresh=0.3,
        scope_mode=SCOPE_FULL,
        eval_ignore_suppress_enabled=False,
        eval_ignore_expand_ratio=1.5,
        eval_ignore_suppress_margin=1.0,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
    ):
    box_coordinate_mode = require_cartesian_data(box_coordinate_mode)
    model.eval()
    state = init_kradar_eval_state()
    state["eval_ignore_suppressed_predictions"] = 0
    state["official_neutral_gt_count"] = 0

    for batch in tqdm.tqdm(dataloader, desc="Evaluation", ncols=120, leave=False):
        batch_modes = {
            require_cartesian_data(value)
            for value in batch.get(
                "box_coordinate_mode",
                [box_coordinate_mode],
            )
        }
        if batch_modes != {box_coordinate_mode}:
            raise ValueError(
                "Evaluation coordinate mode does not match the dataset: "
                f"requested={box_coordinate_mode!r}, batch={sorted(batch_modes)}"
            )
        batch_predictions = predict_detection_batch(
            model=model,
            batch=batch,
            device=device,
            num_classes=num_classes,
            max_detections=max_detections,
            heatmap_nms_kernel=heatmap_nms_kernel,
            heatmap_score_mode=heatmap_score_mode,
            yolox_nms_iou=yolox_nms_iou,
            score_thresh=ap_score_thresh,
            box_coordinate_mode=box_coordinate_mode,
            prepare_model_inputs=prepare_model_inputs,
        )

        for batch_index, frame_predictions in enumerate(batch_predictions):
            append_frame_annos_for_kradar_eval(
                state=state,
                batch=batch,
                batch_index=batch_index,
                frame_predictions=frame_predictions,
                device=device,
                num_classes=num_classes,
                scope_mode=scope_mode,
                official_class_name_map=official_class_name_map,
                eval_ignore_suppress_enabled=eval_ignore_suppress_enabled,
                eval_ignore_expand_ratio=eval_ignore_expand_ratio,
                eval_ignore_suppress_margin=eval_ignore_suppress_margin,
                box_coordinate_mode=box_coordinate_mode,
            )
            state["official_neutral_gt_count"] += int(
                batch.get(
                    "gt_override_ignore_labels",
                    [torch.zeros((0,), dtype=torch.long)] * len(batch_predictions),
                )[batch_index].numel()
            )

    return state


def run_kradar_eval_revised(
        kradar_eval_state,
        official_eval_enabled=True,
        official_eval_version="revised",
        official_eval_iou_backend="auto",
        official_eval_iou_mode="easy",
        official_detection_metrics_enabled=True,
        custom_iou_range_eval_enabled=False,
        custom_iou_thresholds=None,
        coco_style_eval_enabled=False,
        nuscenes_style_eval_enabled=False,
        score_thresh=0.3,
        official_eval_class_ids=None,
        official_class_name_map=None,
        distance_quartile_eval_enabled=False,
        distance_quartile_bins=None,
    ):
    if official_eval_version not in ("revised", "kradar"):
        raise ValueError(
            f"evaluation.py only supports K-Radar revised evaluation, got {official_eval_version!r}."
        )

    frame_count = len(kradar_eval_state["official_gt_annos"])
    print(
        "Finished model inference. "
        f"Collected {frame_count} eval frames. "
        "Running selected metrics now...",
        flush=True,
    )
    if distance_quartile_eval_enabled and not official_eval_enabled:
        raise ValueError(
            "Distance quartile evaluation requires official Cartesian "
            "evaluation."
        )

    shared_official_eval_fn = None
    shared_official_iou_backend_used = None
    if distance_quartile_eval_enabled:
        metric_frame_count = len(kradar_eval_state.get("metric_frames", []))
        if metric_frame_count != len(kradar_eval_state["official_gt_annos"]):
            raise ValueError(
                "Distance-quartile evaluation requires one Cartesian metric frame "
                "for every official annotation frame, got "
                f"{metric_frame_count} metric frames and "
                f"{len(kradar_eval_state['official_gt_annos'])} official frames."
            )
        (
            shared_official_eval_fn,
            shared_official_iou_backend_used,
        ) = load_official_eval_function(
            "revised",
            official_eval_iou_backend,
        )

    metrics = {}
    if official_eval_enabled:
        official_metric_kwargs = dict(
            state=kradar_eval_state,
            official_eval_enabled=True,
            official_eval_version="revised",
            official_eval_iou_backend=official_eval_iou_backend,
            official_eval_iou_mode=official_eval_iou_mode,
            official_detection_metrics_enabled=official_detection_metrics_enabled,
            score_thresh=score_thresh,
            official_eval_class_ids=official_eval_class_ids,
            official_class_name_map=official_class_name_map,
        )
        if shared_official_eval_fn is not None:
            official_metric_kwargs.update({
                "official_eval_fn": shared_official_eval_fn,
                "official_iou_backend_used": shared_official_iou_backend_used,
            })
        metrics.update(
            compute_official_kradar_style_metrics(**official_metric_kwargs)
        )
    if distance_quartile_eval_enabled:
        fixed_quartile_bins = normalize_distance_quartile_bins(
            distance_quartile_bins
        )
        quartile_bins = (
            derive_gt_distance_quartile_bins(kradar_eval_state)
            if fixed_quartile_bins is None
            else fixed_quartile_bins
        )
        quartile_iou_mode = (
            "all" if official_eval_iou_mode == "all" else "easy"
        )
        metrics["distance_quartile_eval_enabled"] = True
        metrics["distance_quartile_bins_mode"] = (
            "derived" if fixed_quartile_bins is None else "fixed"
        )
        metrics["distance_quartile_bins"] = [dict(item) for item in quartile_bins]
        for quartile in quartile_bins:
            tag = str(quartile["tag"])
            quartile_state = filter_kradar_eval_state_by_quartile(
                state=kradar_eval_state,
                quartile_bin=quartile,
                official_class_name_map=official_class_name_map,
            )
            quartile_metrics = compute_official_kradar_style_metrics(
                state=quartile_state,
                official_eval_enabled=True,
                official_eval_version="revised",
                official_eval_iou_backend=official_eval_iou_backend,
                official_eval_iou_mode=quartile_iou_mode,
                official_detection_metrics_enabled=False,
                score_thresh=score_thresh,
                official_eval_class_ids=official_eval_class_ids,
                official_class_name_map=official_class_name_map,
                official_eval_fn=shared_official_eval_fn,
                official_iou_backend_used=shared_official_iou_backend_used,
            )
            for geometry in ("bev", "3d"):
                source_key = f"official_{geometry}_mAP_0.3"
                if source_key not in quartile_metrics:
                    raise RuntimeError(
                        "Official distance-quartile evaluation did not return "
                        f"the required metric {source_key!r}."
                    )
                metrics[
                    f"official_{geometry}_mAP_0.3_quartile_{tag}"
                ] = float(quartile_metrics[source_key])
            num_gt = sum(
                int(frame["gt_boxes"].shape[0])
                for frame in quartile_state["metric_frames"]
            )
            num_detections = sum(
                int(frame["dt_boxes"].shape[0])
                for frame in quartile_state["metric_frames"]
            )
            metrics[f"distance_quartile_num_gt_{tag}"] = num_gt
            metrics[f"distance_quartile_num_detections_{tag}"] = num_detections
            # Make the runtime-derived metadata self-contained and robust to
            # either helper spelling used by downstream table writers.
            quartile["num_gt"] = int(num_gt)
            quartile["bbox_count"] = int(num_gt)
        metrics["distance_quartile_bins"] = [dict(item) for item in quartile_bins]
    if official_eval_enabled and custom_iou_range_eval_enabled:
        metrics.update(
            compute_custom_iou_range_metrics(
                state=kradar_eval_state,
                iou_backend=official_eval_iou_backend,
                iou_thresholds=custom_iou_thresholds,
                score_thresh=score_thresh,
                class_ids=official_eval_class_ids,
                class_name_map=official_class_name_map,
            )
        )
    if official_eval_enabled and coco_style_eval_enabled:
        metrics.update(
            compute_coco_style_metrics(
                state=kradar_eval_state,
                iou_backend=official_eval_iou_backend,
                class_ids=official_eval_class_ids,
                class_name_map=official_class_name_map,
            )
        )
    if official_eval_enabled and nuscenes_style_eval_enabled:
        metrics.update(
            compute_nuscenes_style_metrics(
                state=kradar_eval_state,
                class_ids=official_eval_class_ids,
                class_name_map=official_class_name_map,
            )
        )
    metrics["evaluation_num_eval_frames"] = int(frame_count)
    print("Metric computation finished.", flush=True)
    metrics["mAP"] = float(
        metrics.get("official_main_metric_value", 0.0)
        if official_eval_enabled
        else 0.0
    )
    return metrics


@torch.no_grad()
def evaluate_checkpoint_with_kradar_revised(
        model,
        dataloader,
        device,
        num_classes,
        official_class_name_map,
        prepare_model_inputs,
        max_detections=64,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        yolox_nms_iou=0.65,
        scope_mode=SCOPE_FULL,
        official_eval_enabled=True,
        official_eval_version="revised",
        official_eval_iou_backend="auto",
        official_eval_iou_mode="easy",
        official_detection_metrics_enabled=True,
        custom_iou_range_eval_enabled=False,
        custom_iou_thresholds=None,
        coco_style_eval_enabled=False,
        nuscenes_style_eval_enabled=False,
        ap_score_thresh=0.01,
        score_thresh=0.3,
        eval_ignore_suppress_enabled=False,
        eval_ignore_expand_ratio=1.5,
        eval_ignore_suppress_margin=1.0,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
        distance_quartile_eval_enabled=False,
        distance_quartile_bins=None,
    ):
    box_coordinate_mode = require_cartesian_data(box_coordinate_mode)
    if distance_quartile_eval_enabled and not official_eval_enabled:
        raise ValueError(
            "Distance quartile evaluation requires official Cartesian "
            "evaluation."
        )
    if not official_eval_enabled:
        return {"mAP": 0.0}

    kradar_eval_state = collect_kradar_annos(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=num_classes,
        official_class_name_map=official_class_name_map,
        prepare_model_inputs=prepare_model_inputs,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
        yolox_nms_iou=yolox_nms_iou,
        ap_score_thresh=ap_score_thresh,
        score_thresh=score_thresh,
        scope_mode=scope_mode,
        eval_ignore_suppress_enabled=eval_ignore_suppress_enabled,
        eval_ignore_expand_ratio=eval_ignore_expand_ratio,
        eval_ignore_suppress_margin=eval_ignore_suppress_margin,
        box_coordinate_mode=box_coordinate_mode,
    )
    metrics = run_kradar_eval_revised(
        kradar_eval_state=kradar_eval_state,
        official_eval_enabled=official_eval_enabled,
        official_eval_version=official_eval_version,
        official_eval_iou_backend=official_eval_iou_backend,
        official_eval_iou_mode=official_eval_iou_mode,
        official_detection_metrics_enabled=official_detection_metrics_enabled,
        custom_iou_range_eval_enabled=custom_iou_range_eval_enabled,
        custom_iou_thresholds=custom_iou_thresholds,
        coco_style_eval_enabled=coco_style_eval_enabled,
        nuscenes_style_eval_enabled=nuscenes_style_eval_enabled,
        score_thresh=score_thresh,
        official_eval_class_ids=sorted(official_class_name_map.keys()),
        official_class_name_map=official_class_name_map,
        distance_quartile_eval_enabled=distance_quartile_eval_enabled,
        distance_quartile_bins=distance_quartile_bins,
    )
    metrics["ap_score_thresh"] = float(ap_score_thresh)
    metrics["eval_ignore_suppressed_predictions"] = int(
        kradar_eval_state.get("eval_ignore_suppressed_predictions", 0)
    )
    metrics["official_neutral_gt_count"] = int(
        kradar_eval_state.get("official_neutral_gt_count", 0)
    )
    return metrics


def evaluate_train_val_iou(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_classes,
        official_class_name_map,
        prepare_model_inputs,
        max_detections=64,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        yolox_nms_iou=0.65,
        scope_mode=SCOPE_FULL,
        evaluate_train=False,
        official_eval_enabled=True,
        official_eval_version="revised",
        official_eval_iou_backend="auto",
        official_eval_iou_mode="easy",
        official_detection_metrics_enabled=True,
        custom_iou_range_eval_enabled=False,
        custom_iou_thresholds=None,
        coco_style_eval_enabled=False,
        nuscenes_style_eval_enabled=False,
        ap_score_thresh=0.01,
        score_thresh=0.3,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
    ):
    evaluation_kwargs = {
        "model": model,
        "device": device,
        "num_classes": num_classes,
        "official_class_name_map": official_class_name_map,
        "prepare_model_inputs": prepare_model_inputs,
        "max_detections": max_detections,
        "heatmap_nms_kernel": heatmap_nms_kernel,
        "heatmap_score_mode": heatmap_score_mode,
        "yolox_nms_iou": yolox_nms_iou,
        "scope_mode": scope_mode,
        "official_eval_enabled": official_eval_enabled,
        "official_eval_version": official_eval_version,
        "official_eval_iou_backend": official_eval_iou_backend,
        "official_eval_iou_mode": official_eval_iou_mode,
        "official_detection_metrics_enabled": official_detection_metrics_enabled,
        "custom_iou_range_eval_enabled": custom_iou_range_eval_enabled,
        "custom_iou_thresholds": custom_iou_thresholds,
        "coco_style_eval_enabled": coco_style_eval_enabled,
        "nuscenes_style_eval_enabled": nuscenes_style_eval_enabled,
        "ap_score_thresh": ap_score_thresh,
        "score_thresh": score_thresh,
        "box_coordinate_mode": box_coordinate_mode,
    }
    train_eval_metrics = None
    if evaluate_train:
        train_eval_metrics = evaluate_checkpoint_with_kradar_revised(
            dataloader=train_dataloader,
            **evaluation_kwargs,
        )

    val_eval_metrics = evaluate_checkpoint_with_kradar_revised(
        dataloader=val_dataloader,
        **evaluation_kwargs,
    )

    return {
        "train_eval_metrics": train_eval_metrics,
        "val_eval_metrics": val_eval_metrics,
    }
