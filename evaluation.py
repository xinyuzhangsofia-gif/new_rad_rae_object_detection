"""Evaluation entrypoint for this project using K-Radar official eval_revised.py.

This file does not implement mAP itself. It only:
1. runs the project model,
2. converts predictions and GT into K-Radar KITTI-style annos,
3. calls eval/kitti_eval/eval_revised.py.
"""

import argparse
import os
import re
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml

from cfg_model import (
    AZIMUTH_AXIS,
    ELEVATION_AXIS,
    RANGE_AXIS,
    SCOPE_CHOICES,
    SCOPE_FULL,
    SCOPE_NARROW,
    denormalize_rae_boxes_for_scope,
    normalized_rae_box_centers_in_cartesian_roi,
)
from dataloader import (
    build_train_val_dataloaders,
    normalize_sequence_list,
    prepare_model_inputs,
)
from dataset import CLASS_NAMES, CLASS_TO_IDX
from eval.adapter import (
    compute_official_kradar_style_metrics,
    metric_boxes_to_kitti_anno,
)
from eval.coco_style import compute_coco_style_metrics
from eval.custom_iou_range import (
    DEFAULT_CUSTOM_IOU_THRESHOLDS,
    compute_custom_iou_range_metrics,
    format_custom_iou_suffix,
)
from eval.nuscenes_style import compute_nuscenes_style_metrics
from models import MODEL_TYPES, build_model
from training_utils.checkpoints import format_sequence_run_name
from training_utils.radenet_utils import regression_cell_to_normalized_rae_box
from training_utils.yolox_utils import yolox_outputs_to_detections
from zxy_config import DataConfig

try:
    from eval_cfg import EVAL_CONFIG
except ImportError:
    EVAL_CONFIG = {}


NUM_CLASSES = len(CLASS_NAMES)


def load_torch_checkpoint(checkpoint_path, map_location="cpu"):
    # PyTorch 2.6 changed torch.load default weights_only to True.
    # Our local training checkpoints store config/history objects too.
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def parse_gpu_ids(gpu_ids_text):
    return [
        int(gpu_id.strip())
        for gpu_id in gpu_ids_text.split(",")
        if gpu_id.strip() != ""
    ]


def parse_cuda_choice(cuda_text, fallback_gpu_ids_text):
    if cuda_text is None:
        return parse_gpu_ids(fallback_gpu_ids_text)

    cuda_text = cuda_text.strip().lower()
    if cuda_text in ("", "cpu", "none"):
        return []

    gpu_ids = []
    for cuda_part in cuda_text.split(","):
        cuda_part = cuda_part.strip().lower()
        if cuda_part.startswith("cuda:"):
            cuda_part = cuda_part.removeprefix("cuda:")
        if cuda_part.startswith("gpu"):
            gpu_number = int(cuda_part.removeprefix("gpu"))
            if gpu_number <= 0:
                raise ValueError(f"GPU names start from gpu1, got {cuda_part!r}")
            gpu_ids.append(gpu_number - 1)
        else:
            gpu_ids.append(int(cuda_part))
    return gpu_ids


def select_evaluation_device(cuda_text, gpu_ids_text):
    gpu_ids = parse_cuda_choice(cuda_text, gpu_ids_text)
    requested_cuda = cuda_text is not None and cuda_text.strip().lower() not in ("", "cpu", "none")
    if requested_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was explicitly requested via cuda={cuda_text!r}, "
            "but torch.cuda.is_available() is False."
        )
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        available_gpu_count = torch.cuda.device_count()
        invalid_gpu_ids = [
            gpu_id
            for gpu_id in gpu_ids
            if gpu_id < 0 or gpu_id >= available_gpu_count
        ]
        if len(invalid_gpu_ids) > 0:
            raise ValueError(
                f"Requested GPU ids {invalid_gpu_ids}, "
                f"but only {available_gpu_count} CUDA device(s) are available."
            )
        return torch.device(f"cuda:{gpu_ids[0]}")
    if requested_cuda:
        raise RuntimeError(
            f"CUDA was explicitly requested via cuda={cuda_text!r}, "
            "but no CUDA device could be selected."
        )
    return torch.device("cpu")


def parse_args():
    cfg_defaults = {
        "checkpoint_root": (
            "checkpoints/object_detection/20260619_155520_209652__model_12__seq1_4-6_11_14_20_3_18/"
            "20260620_040729_mAP_0p4741_model_12_global_best_epoch_059_seq1-11.pth"
        ),
        "epoch_step": 1,
        "batch_size": 100,
        "train_ratio": 0.7,
        "split_mode": "file",
        "split_dir": "split",
        "train_sequences": None,
        "val_sequences": None,
        "seed": 42,
        "num_workers": 0,
        "limit_samples": None,
        "eval_scope": None,
        "max_detections": 64,
        "heatmap_nms_kernel": 3,
        "yolox_nms_iou": 0.65,
        "model_type": "auto",
        "gpu_ids": "0,1,2",
        "cuda": None,
        "official_eval_version": "revised",
        "official_eval_iou_backend": "auto",
        "official_eval_iou_mode": "easy",
        "custom_iou_range_eval_enabled": False,
        "custom_iou_thresholds": DEFAULT_CUSTOM_IOU_THRESHOLDS.tolist(),
        "coco_style_eval_enabled": False,
        "nuscenes_style_eval_enabled": False,
        "detection_score_thresh": 0.3,
        "plot_output": None,
    }
    cfg_defaults.update(EVAL_CONFIG)

    parser = argparse.ArgumentParser(
        description="Run official K-Radar KITTI-style evaluation."
    )
    parser.add_argument("--checkpoint-root", default=cfg_defaults["checkpoint_root"])
    parser.add_argument("--epoch-step", type=int, default=cfg_defaults["epoch_step"])
    parser.add_argument("--batch-size", type=int, default=cfg_defaults["batch_size"])
    parser.add_argument("--train-ratio", type=float, default=cfg_defaults["train_ratio"])
    parser.add_argument("--split-mode", default=cfg_defaults["split_mode"], choices=["random", "order", "file", "sequence"])
    parser.add_argument("--split-dir", default=cfg_defaults["split_dir"])
    parser.add_argument("--train-sequences", default=cfg_defaults["train_sequences"])
    parser.add_argument("--val-sequences", default=cfg_defaults["val_sequences"])
    parser.add_argument("--seed", type=int, default=cfg_defaults["seed"])
    parser.add_argument("--num-workers", type=int, default=cfg_defaults["num_workers"])
    parser.add_argument("--limit-samples", type=int, default=cfg_defaults["limit_samples"])
    parser.add_argument("--eval-scope", default=cfg_defaults["eval_scope"], choices=SCOPE_CHOICES)
    parser.add_argument("--max-detections", type=int, default=cfg_defaults["max_detections"])
    parser.add_argument("--heatmap-nms-kernel", type=int, default=cfg_defaults["heatmap_nms_kernel"])
    parser.add_argument("--yolox-nms-iou", type=float, default=cfg_defaults["yolox_nms_iou"])
    parser.add_argument(
        "--model-type",
        default=cfg_defaults["model_type"],
        choices=["auto"] + sorted(MODEL_TYPES),
    )
    parser.add_argument("--gpu-ids", default=cfg_defaults["gpu_ids"])
    parser.add_argument("--cuda", default=cfg_defaults["cuda"])
    parser.add_argument(
        "--official-eval-version",
        default=cfg_defaults["official_eval_version"],
        choices=["revised", "kradar"],
    )
    parser.add_argument(
        "--official-eval-iou-backend",
        default=cfg_defaults["official_eval_iou_backend"],
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--official-eval-iou-mode",
        default=cfg_defaults["official_eval_iou_mode"],
        choices=["easy", "mod", "hard", "all"],
    )
    parser.add_argument(
        "--custom-iou-range-eval-enabled",
        default=cfg_defaults["custom_iou_range_eval_enabled"],
    )
    parser.add_argument(
        "--custom-iou-thresholds",
        default=cfg_defaults["custom_iou_thresholds"],
    )
    parser.add_argument(
        "--coco-style-eval-enabled",
        default=cfg_defaults["coco_style_eval_enabled"],
    )
    parser.add_argument(
        "--nuscenes-style-eval-enabled",
        default=cfg_defaults["nuscenes_style_eval_enabled"],
    )
    parser.add_argument(
        "--detection-score-thresh",
        type=float,
        default=cfg_defaults["detection_score_thresh"],
    )
    parser.add_argument("--plot-output", default=cfg_defaults["plot_output"])
    args = parser.parse_args()
    args.custom_iou_range_eval_enabled = normalize_bool_flag(
        args.custom_iou_range_eval_enabled,
        name="custom_iou_range_eval_enabled",
    )
    args.custom_iou_thresholds = normalize_float_thresholds(
        args.custom_iou_thresholds,
        name="custom_iou_thresholds",
    )
    args.coco_style_eval_enabled = normalize_bool_flag(
        args.coco_style_eval_enabled,
        name="coco_style_eval_enabled",
    )
    args.nuscenes_style_eval_enabled = normalize_bool_flag(
        args.nuscenes_style_eval_enabled,
        name="nuscenes_style_eval_enabled",
    )
    if args.custom_iou_range_eval_enabled and len(args.custom_iou_thresholds) == 0:
        raise ValueError(
            "custom_iou_range_eval_enabled is True, but custom_iou_thresholds is empty."
        )
    return args


def normalize_bool_flag(value, name):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    raise ValueError(f"Invalid boolean-like value for {name}: {value!r}")


def normalize_float_thresholds(value, name):
    if value is None:
        return []

    values = None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized == "":
            return []
        if normalized.count(":") == 2:
            start_text, step_text, end_text = [part.strip() for part in normalized.split(":")]
            start = float(start_text)
            step = float(step_text)
            end = float(end_text)
            if step <= 0:
                raise ValueError(f"{name} step must be > 0, got {step}")
            values = []
            current = start
            while current <= end + 1e-9:
                values.append(float(round(current, 6)))
                current += step
        else:
            values = [
                float(item.strip())
                for item in normalized.split(",")
                if item.strip() != ""
            ]
    elif isinstance(value, np.ndarray):
        values = [float(item) for item in value.reshape(-1).tolist()]
    elif isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        raise ValueError(f"Unsupported threshold list value for {name}: {value!r}")

    if len(values) == 0:
        return []
    for threshold in values:
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError(f"{name} values must be in (0, 1], got {threshold}")
    return [float(round(threshold, 6)) for threshold in values]


def normalized_rae_boxes_to_cartesian_metric_boxes(boxes, scope_mode, rae_shape):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 7))

    raw_boxes = denormalize_rae_boxes_for_scope(
        boxes=boxes,
        scope_mode=scope_mode,
        rae_shape=rae_shape,
    )

    radius = RANGE_AXIS.minimum + (raw_boxes[:, 0] * RANGE_AXIS.step)
    azimuth = torch.deg2rad(
        raw_boxes[:, 1].new_tensor(AZIMUTH_AXIS.minimum)
        + (raw_boxes[:, 1] * AZIMUTH_AXIS.step)
    )
    elevation = torch.deg2rad(
        raw_boxes[:, 2].new_tensor(ELEVATION_AXIS.minimum)
        + (raw_boxes[:, 2] * ELEVATION_AXIS.step)
    )

    r_xy = radius * torch.cos(elevation)
    x = r_xy * torch.cos(azimuth)
    y = -r_xy * torch.sin(azimuth)
    z = radius * torch.sin(elevation)

    length = (raw_boxes[:, 3].abs() * RANGE_AXIS.step).clamp(min=1e-3)
    width = (
        r_xy.abs()
        * torch.deg2rad(raw_boxes[:, 4].abs() * AZIMUTH_AXIS.step)
    ).clamp(min=1e-3)
    height = (
        (radius * torch.cos(elevation)).abs()
        * torch.deg2rad(raw_boxes[:, 5].abs() * ELEVATION_AXIS.step)
    ).clamp(min=1e-3)
    yaw = raw_boxes[:, 6]

    return torch.stack([x, y, z, length, width, height, yaw], dim=-1)


def centerpoint_heatmap_nms(heatmap, kernel_size=3):
    if kernel_size <= 1:
        return heatmap
    if kernel_size % 2 == 0:
        raise ValueError(f"Heatmap NMS kernel must be odd, got {kernel_size}")

    pad = (kernel_size - 1) // 2
    pooled = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)
    keep = pooled == heatmap
    return heatmap * keep.to(heatmap.dtype)


def gather_dense_feature(feature_map, indices):
    flat = feature_map.flatten(start_dim=2).transpose(1, 2)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
    return flat.gather(dim=1, index=gather_index)


def apply_quality_score(heatmap_scores, outputs):
    if "quality_logits" in outputs:
        quality_scores = outputs["quality_logits"].sigmoid()
        if quality_scores.shape[-2:] != heatmap_scores.shape[-2:]:
            quality_scores = F.interpolate(
                quality_scores,
                size=heatmap_scores.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return heatmap_scores * quality_scores

    if "objectness_logits" not in outputs:
        return heatmap_scores

    objectness_scores = outputs["objectness_logits"].sigmoid()
    if objectness_scores.shape[-2:] != heatmap_scores.shape[-2:]:
        objectness_scores = F.interpolate(
            objectness_scores,
            size=heatmap_scores.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return heatmap_scores * objectness_scores


def outputs_to_detections(
        outputs,
        num_classes,
        max_detections=64,
        heatmap_nms_kernel=3,
    ):
    dense_keys = {"cls_logits", "center_offset", "center_height", "size", "yaw"}
    missing_keys = sorted(dense_keys - set(outputs.keys()))
    if len(missing_keys) > 0:
        raise KeyError(
            "Dense CenterPoint evaluation requires output keys "
            f"{sorted(dense_keys)}, missing {missing_keys}."
        )

    cls_logits = outputs["cls_logits"][:, :num_classes]
    _, _, heatmap_h, heatmap_w = cls_logits.shape
    dtype = cls_logits.dtype

    heatmap_scores = apply_quality_score(cls_logits.sigmoid(), outputs)
    heatmap_scores = centerpoint_heatmap_nms(
        heatmap=heatmap_scores,
        kernel_size=heatmap_nms_kernel,
    )

    flat_scores = heatmap_scores.flatten(start_dim=1)
    topk_count = min(max_detections, flat_scores.shape[1])
    scores, flat_indices = flat_scores.topk(topk_count, dim=1)

    spatial_size = heatmap_h * heatmap_w
    labels = flat_indices // spatial_size
    spatial_indices = flat_indices % spatial_size

    heatmap_y_idx = spatial_indices // heatmap_w
    heatmap_x_idx = spatial_indices % heatmap_w
    _, _, box_h, box_w = outputs["center_offset"].shape
    box_y_idx_long = torch.div(
        heatmap_y_idx * box_h,
        max(heatmap_h, 1),
        rounding_mode="floor",
    ).clamp(max=box_h - 1)
    box_x_idx_long = torch.div(
        heatmap_x_idx * box_w,
        max(heatmap_w, 1),
        rounding_mode="floor",
    ).clamp(max=box_w - 1)
    box_indices = box_y_idx_long * box_w + box_x_idx_long

    y_idx = box_y_idx_long.to(dtype)
    x_idx = box_x_idx_long.to(dtype)
    center_offset = gather_dense_feature(outputs["center_offset"], box_indices).sigmoid()
    center_height = gather_dense_feature(outputs["center_height"], box_indices).sigmoid()
    size = gather_dense_feature(outputs["size"], box_indices).sigmoid()
    yaw = gather_dense_feature(outputs["yaw"], box_indices)

    r_center = (y_idx + center_offset[..., 0]) / max(box_h, 1)
    a_center = (x_idx + center_offset[..., 1]) / max(box_w, 1)
    e_center = center_height[..., 0]
    yaw_angle = torch.atan2(yaw[..., 0], yaw[..., 1])
    yaw_norm = (yaw_angle + torch.pi) / (2.0 * torch.pi)

    boxes = torch.stack(
        [
            r_center,
            a_center,
            e_center,
            size[..., 0],
            size[..., 1],
            size[..., 2],
            yaw_norm,
        ],
        dim=-1,
    ).clamp(min=1e-4, max=1.0 - 1e-4)

    return boxes, scores, labels


def official_radenet_outputs_to_detections(
        outputs,
        num_classes,
        scope_modes,
        full_rae_shapes,
        max_detections=64,
        heatmap_nms_kernel=3,
    ):
    if scope_modes is None or full_rae_shapes is None:
        raise ValueError("Official RADE-Net decoding requires scope_modes and full_rae_shapes.")

    heatmap = outputs["heatmap"][:, :num_classes]
    heatmap_scores = centerpoint_heatmap_nms(
        heatmap=heatmap,
        kernel_size=heatmap_nms_kernel,
    )
    regression = outputs["regression"]
    batch_size, _, heatmap_h, heatmap_w = heatmap_scores.shape

    flat_scores = heatmap_scores.flatten(start_dim=1)
    topk_count = min(max_detections, flat_scores.shape[1])
    scores, flat_indices = flat_scores.topk(topk_count, dim=1)

    spatial_size = heatmap_h * heatmap_w
    labels = flat_indices // spatial_size
    spatial_indices = flat_indices % spatial_size
    y_idx = spatial_indices // heatmap_w
    x_idx = spatial_indices % heatmap_w
    reg = gather_dense_feature(regression, spatial_indices)

    boxes = []
    for batch_index in range(batch_size):
        boxes.append(
            regression_cell_to_normalized_rae_box(
                pred_reg=reg[batch_index],
                y_idx=y_idx[batch_index].to(reg.dtype),
                x_idx=x_idx[batch_index].to(reg.dtype),
                feature_shape=(heatmap_h, heatmap_w),
                scope_mode=scope_modes[batch_index],
                full_rae_shape=full_rae_shapes[batch_index],
            )
        )
    return torch.stack(boxes, dim=0), scores, labels


def decode_batch_predictions(
        outputs,
        num_classes,
        max_detections,
        heatmap_nms_kernel,
        yolox_nms_iou,
        scope_modes=None,
        full_rae_shapes=None,
    ):
    if "objectness_logits" in outputs:
        return yolox_outputs_to_detections(
            outputs=outputs,
            num_classes=num_classes,
            score_thresh=None,
            max_detections=max_detections,
            nms_iou_thresh=yolox_nms_iou,
        )

    if "heatmap" in outputs and "regression" in outputs:
        pred_boxes, pred_scores, pred_labels = official_radenet_outputs_to_detections(
            outputs=outputs,
            num_classes=num_classes,
            scope_modes=scope_modes,
            full_rae_shapes=full_rae_shapes,
            max_detections=max_detections,
            heatmap_nms_kernel=heatmap_nms_kernel,
        )
    else:
        pred_boxes, pred_scores, pred_labels = outputs_to_detections(
            outputs=outputs,
            num_classes=num_classes,
            max_detections=max_detections,
            heatmap_nms_kernel=heatmap_nms_kernel,
        )

    batch_predictions = []
    for batch_index in range(pred_boxes.shape[0]):
        batch_predictions.append({
            "boxes": pred_boxes[batch_index],
            "scores": pred_scores[batch_index],
            "labels": pred_labels[batch_index],
        })
    return batch_predictions


def filter_predictions_to_scope(frame_predictions, scope_mode, full_rae_shape):
    if scope_mode != SCOPE_NARROW:
        return frame_predictions

    keep = normalized_rae_box_centers_in_cartesian_roi(
        frame_predictions["boxes"],
        scope_mode=scope_mode,
        rae_shape=full_rae_shape,
    )
    return {
        "boxes": frame_predictions["boxes"][keep],
        "scores": frame_predictions["scores"][keep],
        "labels": frame_predictions["labels"][keep],
    }


def init_kradar_eval_state():
    return {
        "official_gt_annos": [],
        "official_dt_annos": [],
        "metric_frames": [],
    }


def append_frame_annos_for_kradar_eval(
        state,
        batch,
        batch_index,
        frame_predictions,
        device,
        num_classes,
        scope_mode,
    ):
    full_rae_shape = batch["full_rae_shape"][batch_index]
    frame_predictions = filter_predictions_to_scope(
        frame_predictions=frame_predictions,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
    )

    gt_boxes_all = batch["gt_boxes"][batch_index].to(device)
    gt_labels_all = batch["gt_labels"][batch_index].to(device)
    valid_gt = gt_labels_all < num_classes
    gt_boxes = gt_boxes_all[valid_gt]
    gt_labels = gt_labels_all[valid_gt]

    gt_metric_boxes = normalized_rae_boxes_to_cartesian_metric_boxes(
        gt_boxes,
        scope_mode=scope_mode,
        rae_shape=full_rae_shape,
    )
    pred_metric_boxes = normalized_rae_boxes_to_cartesian_metric_boxes(
        frame_predictions["boxes"],
        scope_mode=scope_mode,
        rae_shape=full_rae_shape,
    )

    state["official_gt_annos"].append(
        metric_boxes_to_kitti_anno(
            boxes=gt_metric_boxes.detach().cpu(),
            labels=gt_labels.detach().cpu(),
            is_prediction=False,
        )
    )
    state["official_dt_annos"].append(
        metric_boxes_to_kitti_anno(
            boxes=pred_metric_boxes.detach().cpu(),
            labels=frame_predictions["labels"].detach().cpu(),
            scores=frame_predictions["scores"].detach().cpu(),
            is_prediction=True,
        )
    )
    state["metric_frames"].append(
        {
            "gt_boxes": gt_metric_boxes.detach().cpu().numpy(),
            "gt_labels": gt_labels.detach().cpu().numpy(),
            "dt_boxes": pred_metric_boxes.detach().cpu().numpy(),
            "dt_labels": frame_predictions["labels"].detach().cpu().numpy(),
            "dt_scores": frame_predictions["scores"].detach().cpu().numpy(),
        }
    )


@torch.no_grad()
def collect_kradar_annos(
        model,
        dataloader,
        device,
        num_classes,
        prepare_model_inputs,
        max_detections=64,
        heatmap_nms_kernel=3,
        yolox_nms_iou=0.65,
        scope_mode=SCOPE_FULL,
    ):
    model.eval()
    state = init_kradar_eval_state()

    for batch in tqdm.tqdm(dataloader, desc="Evaluation", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)
        batch_predictions = decode_batch_predictions(
            outputs=outputs,
            num_classes=num_classes,
            max_detections=max_detections,
            heatmap_nms_kernel=heatmap_nms_kernel,
            yolox_nms_iou=yolox_nms_iou,
            scope_modes=batch["scope_mode"],
            full_rae_shapes=batch["full_rae_shape"],
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
            )

    return state


def run_kradar_eval_revised(
        kradar_eval_state,
        official_eval_version="revised",
        official_eval_iou_backend="auto",
        official_eval_iou_mode="easy",
        custom_iou_range_eval_enabled=False,
        custom_iou_thresholds=None,
        coco_style_eval_enabled=False,
        nuscenes_style_eval_enabled=False,
        detection_score_thresh=0.3,
    ):
    if official_eval_version not in ("revised", "kradar"):
        raise ValueError(
            f"evaluation.py only supports K-Radar revised evaluation, got {official_eval_version!r}."
        )

    print(
        "Finished model inference. "
        f"Collected {len(kradar_eval_state['official_gt_annos'])} eval frames. "
        "Running official K-Radar metrics now...",
        flush=True,
    )
    official_metrics = compute_official_kradar_style_metrics(
        state=kradar_eval_state,
        official_eval_enabled=True,
        official_eval_version="revised",
        official_eval_iou_backend=official_eval_iou_backend,
        official_eval_iou_mode=official_eval_iou_mode,
        detection_score_thresh=detection_score_thresh,
    )
    if custom_iou_range_eval_enabled:
        official_metrics.update(
            compute_custom_iou_range_metrics(
                state=kradar_eval_state,
                iou_backend=official_eval_iou_backend,
                iou_thresholds=custom_iou_thresholds,
            )
        )
    if coco_style_eval_enabled:
        official_metrics.update(
            compute_coco_style_metrics(
                state=kradar_eval_state,
                iou_backend=official_eval_iou_backend,
            )
        )
    if nuscenes_style_eval_enabled:
        official_metrics.update(
            compute_nuscenes_style_metrics(
                state=kradar_eval_state,
            )
        )
    print("Official K-Radar metric computation finished.", flush=True)
    official_metrics["mAP"] = float(
        official_metrics.get("official_main_metric_value", 0.0)
    )
    return official_metrics


@torch.no_grad()
def evaluate_checkpoint_with_kradar_revised(
        model,
        dataloader,
        device,
        num_classes,
        prepare_model_inputs,
        max_detections=64,
        heatmap_nms_kernel=3,
        yolox_nms_iou=0.65,
        scope_mode=SCOPE_FULL,
        official_eval_enabled=True,
        official_eval_version="revised",
        official_eval_iou_backend="auto",
        official_eval_iou_mode="easy",
        custom_iou_range_eval_enabled=False,
        custom_iou_thresholds=None,
        coco_style_eval_enabled=False,
        nuscenes_style_eval_enabled=False,
        detection_score_thresh=0.3,
    ):
    if not official_eval_enabled:
        return {"mAP": 0.0}

    kradar_eval_state = collect_kradar_annos(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=num_classes,
        prepare_model_inputs=prepare_model_inputs,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        yolox_nms_iou=yolox_nms_iou,
        scope_mode=scope_mode,
    )
    return run_kradar_eval_revised(
        kradar_eval_state=kradar_eval_state,
        official_eval_version=official_eval_version,
        official_eval_iou_backend=official_eval_iou_backend,
        official_eval_iou_mode=official_eval_iou_mode,
        custom_iou_range_eval_enabled=custom_iou_range_eval_enabled,
        custom_iou_thresholds=custom_iou_thresholds,
        coco_style_eval_enabled=coco_style_eval_enabled,
        nuscenes_style_eval_enabled=nuscenes_style_eval_enabled,
        detection_score_thresh=detection_score_thresh,
    )


@torch.no_grad()
def evaluate_train_val_iou(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_classes,
        prepare_model_inputs,
        max_detections=64,
        heatmap_nms_kernel=3,
        yolox_nms_iou=0.65,
        scope_mode=SCOPE_FULL,
        evaluate_train=False,
        official_eval_enabled=True,
        official_eval_version="revised",
        official_eval_iou_backend="auto",
        official_eval_iou_mode="easy",
        custom_iou_range_eval_enabled=False,
        custom_iou_thresholds=None,
        coco_style_eval_enabled=False,
        nuscenes_style_eval_enabled=False,
        detection_score_thresh=0.3,
    ):
    del train_dataloader
    del evaluate_train

    val_eval_metrics = evaluate_checkpoint_with_kradar_revised(
        model=model,
        dataloader=val_dataloader,
        device=device,
        num_classes=num_classes,
        prepare_model_inputs=prepare_model_inputs,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        yolox_nms_iou=yolox_nms_iou,
        scope_mode=scope_mode,
        official_eval_enabled=official_eval_enabled,
        official_eval_version=official_eval_version,
        official_eval_iou_backend=official_eval_iou_backend,
        official_eval_iou_mode=official_eval_iou_mode,
        custom_iou_range_eval_enabled=custom_iou_range_eval_enabled,
        custom_iou_thresholds=custom_iou_thresholds,
        coco_style_eval_enabled=coco_style_eval_enabled,
        nuscenes_style_eval_enabled=nuscenes_style_eval_enabled,
        detection_score_thresh=detection_score_thresh,
    )

    return {
        "train_eval_metrics": None,
        "val_eval_metrics": val_eval_metrics,
    }


def checkpoint_epoch(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"epoch_(\d+)", filename)
    if match is None:
        return None
    return int(match.group(1))


def find_epoch_checkpoints(checkpoint_root, epoch_step):
    if epoch_step <= 0:
        raise ValueError(f"--epoch-step must be greater than 0, got {epoch_step}")

    if os.path.isfile(checkpoint_root):
        epoch = checkpoint_epoch(checkpoint_root)
        if epoch is None:
            checkpoint = load_torch_checkpoint(checkpoint_root, map_location="cpu")
            epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
        return [(epoch, checkpoint_root)]

    checkpoint_by_epoch = {}
    for filename in os.listdir(checkpoint_root):
        if not filename.endswith(".pth"):
            continue

        is_global_best = (
            filename.startswith("global_best_epoch_")
            or "_global_best_epoch_" in filename
        )
        is_candidate = (
            filename.startswith("candidate_epoch_")
            or "_candidate_epoch_" in filename
        )
        is_epoch = filename.startswith("epoch_")
        if not (is_global_best or is_candidate or is_epoch):
            continue

        checkpoint_path = os.path.join(checkpoint_root, filename)
        epoch = checkpoint_epoch(checkpoint_path)
        if epoch is None:
            continue
        if not is_global_best and epoch % epoch_step != 0:
            continue

        existing = checkpoint_by_epoch.get(epoch)
        if existing is None or is_global_best:
            checkpoint_by_epoch[epoch] = (epoch, checkpoint_path)

    checkpoint_paths = list(checkpoint_by_epoch.values())
    checkpoint_paths.sort(key=lambda item: item[0])
    return checkpoint_paths


def get_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def infer_model_type_from_checkpoint(checkpoint_path):
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = get_checkpoint_state_dict(checkpoint)
    if isinstance(checkpoint, dict):
        model_type = checkpoint.get("config", {}).get("model_type")
        if model_type:
            return model_type

    if "_qfl_model_marker" in state_dict:
        return "model11"
    if "_model14_swin_yolox_marker" in state_dict:
        return "model14"
    if "_model15_radenet_official_marker" in state_dict:
        return "model15"
    if "_model13_radenet_marker" in state_dict:
        return "model13"
    if "_model12_yolox_marker" in state_dict:
        return "model12"
    if any(".cls_feature_mixer." in key or ".reg_feature_mixer." in key for key in state_dict.keys()):
        return "model10"

    has_bifpn = any(".bifpn_blocks." in key for key in state_dict.keys())
    has_cfe = any(".cfe1." in key or ".cfe2." in key or ".cfe3." in key for key in state_dict.keys())
    if has_bifpn and has_cfe:
        return "model9"
    if has_bifpn:
        return "model2"
    if has_cfe:
        return "model8"
    if any(".attn.relative_position_bias_table" in key for key in state_dict.keys()):
        return "model7"
    if any(".quality_decoder." in key for key in state_dict.keys()):
        return "model6"

    has_fpn_lateral = any(
        key.startswith("backbone.encoder.rad_encoder.lateral")
        for key in state_dict.keys()
    )
    has_deform_conv = any(
        ".offset_conv." in key or ".deform_conv." in key
        for key in state_dict.keys()
    )
    if has_fpn_lateral:
        return "model5" if has_deform_conv else "model3"
    if has_deform_conv:
        return "model4"
    if any(key.startswith("backbone.encoder.") for key in state_dict.keys()):
        return "model1"

    raise ValueError(f"Unsupported old model checkpoint: {checkpoint_path}")


def resolve_model_type(args, checkpoint_paths):
    if args.model_type != "auto":
        return args.model_type

    _, first_checkpoint_path = checkpoint_paths[0]
    model_type = infer_model_type_from_checkpoint(first_checkpoint_path)
    print(f"Auto-detected model type: {model_type}")
    return model_type


def apply_checkpoint_config_defaults(args, checkpoint_paths):
    _, first_checkpoint_path = checkpoint_paths[0]
    checkpoint = load_torch_checkpoint(first_checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return

    config = checkpoint.get("config", {})
    if config.get("max_detections") is not None:
        args.max_detections = int(config["max_detections"])
    elif config.get("num_boxes") is not None:
        args.max_detections = int(config["num_boxes"])
    if config.get("train_ratio") is not None:
        args.train_ratio = float(config["train_ratio"])
    if config.get("custom_iou_range_eval_enabled") is not None:
        args.custom_iou_range_eval_enabled = bool(config["custom_iou_range_eval_enabled"])
    if config.get("custom_iou_thresholds") is not None:
        args.custom_iou_thresholds = normalize_float_thresholds(
            config["custom_iou_thresholds"],
            name="checkpoint.config.custom_iou_thresholds",
        )
    if config.get("coco_style_eval_enabled") is not None:
        args.coco_style_eval_enabled = bool(config["coco_style_eval_enabled"])
    if config.get("nuscenes_style_eval_enabled") is not None:
        args.nuscenes_style_eval_enabled = bool(config["nuscenes_style_eval_enabled"])
    if config.get("split_mode") is not None:
        args.split_mode = config["split_mode"]
    if config.get("split_dir") is not None:
        args.split_dir = config["split_dir"]
    if config.get("train_sequences") is not None:
        args.train_sequences = config["train_sequences"]
    if config.get("val_sequences") is not None:
        args.val_sequences = config["val_sequences"]
    if config.get("seed") is not None:
        args.seed = int(config["seed"])
    if args.eval_scope is None:
        args.eval_scope = config.get("train_scope", SCOPE_FULL)
    if args.eval_scope not in SCOPE_CHOICES:
        raise ValueError(
            f"Invalid evaluation scope {args.eval_scope!r}; expected one of {SCOPE_CHOICES}."
        )


def metric_text(value):
    if value is None:
        return "-"
    numeric_value = float(value)
    if np.isnan(numeric_value):
        return "-"
    return f"{numeric_value:.4f}"


def official_ap_value(value):
    if value is None:
        return None
    numeric_value = float(value)
    if np.isnan(numeric_value):
        return np.nan
    return numeric_value / 100.0


def official_ap_text(value):
    return metric_text(official_ap_value(value))


def sequence_name_for_filename(sequences, empty_name):
    sequences = normalize_sequence_list(sequences, name=empty_name)
    if sequences is None or len(sequences) == 0:
        return empty_name
    return format_sequence_run_name(sequences)


def build_plot_metadata(args, model_type):
    return {
        "model_type": str(model_type or "model_unknown"),
        "split_mode": str(args.split_mode),
        "train_sequences": sequence_name_for_filename(args.train_sequences, "train_unknown"),
        "val_sequences": sequence_name_for_filename(args.val_sequences, "val_unknown"),
        "eval_scope": str(args.eval_scope),
        "detection_score_thresh": float(args.detection_score_thresh),
        "official_iou_mode": str(args.official_eval_iou_mode),
        "custom_iou_range_eval_enabled": bool(args.custom_iou_range_eval_enabled),
        "custom_iou_thresholds": [float(value) for value in args.custom_iou_thresholds],
        "coco_style_eval_enabled": bool(args.coco_style_eval_enabled),
        "nuscenes_style_eval_enabled": bool(args.nuscenes_style_eval_enabled),
    }


def resolve_plot_output_path(args, checkpoint_paths, model_type):
    plot_output = args.plot_output
    if plot_output is None:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = str(model_type or "model_unknown")
    train_tag = sequence_name_for_filename(args.train_sequences, "train_unknown")
    val_tag = sequence_name_for_filename(args.val_sequences, "val_unknown")

    if isinstance(plot_output, str):
        normalized = plot_output.strip()
        if normalized == "":
            return None
        if normalized.lower() in {"no", "none", "false", "0", "null"}:
            return None
        if normalized.lower() in {"yes", "true", "1", "auto"}:
            checkpoint_root = args.checkpoint_root
            if os.path.isfile(checkpoint_root):
                checkpoint_name = os.path.splitext(
                    os.path.basename(checkpoint_root)
                )[0]
            else:
                _, first_checkpoint_path = checkpoint_paths[0]
                checkpoint_name = os.path.splitext(
                    os.path.basename(first_checkpoint_path)
                )[0]
            filename = (
                f"{timestamp}_{model_tag}_{train_tag}_{val_tag}_{checkpoint_name}_{args.official_eval_iou_mode}_"
                f"{args.official_eval_iou_backend}.png"
            )
            return os.path.join("evaluation_plots", "png_photos", filename)
        return normalized

    if bool(plot_output):
        checkpoint_root = args.checkpoint_root
        if os.path.isfile(checkpoint_root):
            checkpoint_name = os.path.splitext(
                os.path.basename(checkpoint_root)
            )[0]
        else:
            _, first_checkpoint_path = checkpoint_paths[0]
            checkpoint_name = os.path.splitext(
                os.path.basename(first_checkpoint_path)
            )[0]
        filename = (
            f"{timestamp}_{model_tag}_{train_tag}_{val_tag}_{checkpoint_name}_{args.official_eval_iou_mode}_"
            f"{args.official_eval_iou_backend}.png"
        )
        return os.path.join("evaluation_plots", "png_photos", filename)

    return None


def metric_label(metric_key):
    if metric_key.startswith("official_bev_mAP_"):
        return f"BEV mAP@{metric_key.rsplit('_', 1)[-1]}"
    if metric_key.startswith("official_3d_mAP_"):
        return f"3D mAP@{metric_key.rsplit('_', 1)[-1]}"
    return metric_key


def display_class_name(class_name):
    return {
        "sed": "Sedan",
        "bus": "Bus",
    }.get(class_name, class_name)


def pick_plot_metric_keys(results):
    preferred_order = [
        "official_bev_mAP_0.3",
        "official_bev_mAP_0.5",
        "official_bev_mAP_0.7",
        "official_3d_mAP_0.3",
        "official_3d_mAP_0.5",
        "official_3d_mAP_0.7",
    ]
    metric_keys = []
    for metric_key in preferred_order:
        if any(result.get(metric_key) is not None for result in results):
            metric_keys.append(metric_key)
    return metric_keys


def available_plot_iou_suffixes(results):
    preferred_suffixes = ["0.3", "0.5"]
    available_metric_keys = set(pick_plot_metric_keys(results))
    suffixes = []
    for suffix in preferred_suffixes:
        bev_key = f"official_bev_mAP_{suffix}"
        d3_key = f"official_3d_mAP_{suffix}"
        if bev_key in available_metric_keys or d3_key in available_metric_keys:
            suffixes.append(suffix)
    return suffixes


def main_plot_iou_suffix(results):
    for result in results:
        main_key = result.get("official_main_metric_key")
        if isinstance(main_key, str) and main_key.startswith("official_bev_mAP_"):
            return main_key.rsplit("_", 1)[-1]
    return "0.3"


def detection_table_columns():
    return [
        ("Precision", "official_detection_precision", "metric"),
        ("Recall", "official_detection_recall", "metric"),
        ("F1", "official_detection_f1", "metric"),
        ("TP", "official_detection_tp", "int"),
        ("FP", "official_detection_fp", "int"),
        ("FN", "official_detection_fn", "int"),
    ]


def result_row_label(result, object_name, multi_epoch):
    if multi_epoch:
        return f"Epoch {result['epoch']} / {object_name}"
    return object_name


def style_metric_table(table, row_meta, best_epoch, font_size=9.5):
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor("#9ca3af")
        if row_idx == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold", color="#111827")
            continue

        meta = row_meta[row_idx - 1]
        is_best_overall = (
            int(meta["epoch"]) == int(best_epoch)
            and meta.get("row_kind") == "overall"
        )
        if meta.get("row_kind") == "overall":
            cell.set_facecolor("#dbeafe" if is_best_overall else "#eef2ff")
        else:
            stripe_index = int(meta.get("stripe_index", row_idx - 1))
            cell.set_facecolor("#f8fafc" if stripe_index % 2 == 0 else "#ffffff")

        if col_idx == 0 and is_best_overall:
            cell.set_text_props(weight="bold", color="#1d4ed8")


def build_official_plot_section(results, iou_suffixes, multi_epoch):
    columns = ["Object"]
    for iou_suffix in iou_suffixes:
        columns.extend([f"BEV AP@{iou_suffix}", f"3D AP@{iou_suffix}"])
    columns.extend(label for label, _, _ in detection_table_columns())

    rows = []
    row_meta = []
    stripe_index = 0
    for result in results:
        row = [result_row_label(result, "Overall", multi_epoch)]
        for iou_suffix in iou_suffixes:
            row.extend([
                official_ap_text(result.get(f"official_bev_mAP_{iou_suffix}")),
                official_ap_text(result.get(f"official_3d_mAP_{iou_suffix}")),
            ])
        for _, result_key, value_type in detection_table_columns():
            value = result.get(result_key)
            if value_type == "metric":
                row.append(metric_text(value))
            else:
                row.append("-" if value is None else str(int(value)))
        rows.append(row)
        row_meta.append({
            "epoch": int(result["epoch"]),
            "row_kind": "overall",
            "stripe_index": stripe_index,
        })
        stripe_index += 1

        per_class = result.get("official_detection_per_class", {})
        for class_name in ("sed", "bus"):
            class_stats = per_class.get(class_name, {})
            row = [result_row_label(result, display_class_name(class_name), multi_epoch)]
            for iou_suffix in iou_suffixes:
                row.extend([
                    official_ap_text(result.get(f"official_{class_name}_bev_AP_{iou_suffix}")),
                    official_ap_text(result.get(f"official_{class_name}_3d_AP_{iou_suffix}")),
                ])
            for label, _, value_type in detection_table_columns():
                class_value = class_stats.get(label.lower())
                if value_type == "metric":
                    row.append(metric_text(class_value))
                else:
                    row.append("-" if class_value is None else str(int(class_value)))
            rows.append(row)
            row_meta.append({
                "epoch": int(result["epoch"]),
                "row_kind": class_name,
                "stripe_index": stripe_index,
            })
            stripe_index += 1

    return {
        "title": "Official K-Radar",
        "columns": columns,
        "rows": rows,
        "row_meta": row_meta,
    }


def build_coco_plot_section(results, multi_epoch):
    if not any("coco_bev_mAP" in result for result in results):
        return None

    columns = [
        "Object",
        "BEV mAP",
        "BEV AP@0.50",
        "BEV AP@0.75",
        "3D mAP",
        "3D AP@0.50",
        "3D AP@0.75",
    ]

    rows = []
    row_meta = []
    stripe_index = 0
    include_per_class = not multi_epoch
    for result in results:
        rows.append([
            result_row_label(result, "Overall", multi_epoch),
            metric_text(result.get("coco_bev_mAP")),
            metric_text(result.get("coco_bev_AP_0.50")),
            metric_text(result.get("coco_bev_AP_0.75")),
            metric_text(result.get("coco_3d_mAP")),
            metric_text(result.get("coco_3d_AP_0.50")),
            metric_text(result.get("coco_3d_AP_0.75")),
        ])
        row_meta.append({
            "epoch": int(result["epoch"]),
            "row_kind": "overall",
            "stripe_index": stripe_index,
        })
        stripe_index += 1

        if not include_per_class:
            continue

        for class_name in ("sed", "bus"):
            rows.append([
                result_row_label(result, display_class_name(class_name), multi_epoch),
                metric_text(result.get(f"coco_{class_name}_bev_mAP")),
                metric_text(result.get(f"coco_{class_name}_bev_AP_0.50")),
                metric_text(result.get(f"coco_{class_name}_bev_AP_0.75")),
                metric_text(result.get(f"coco_{class_name}_3d_mAP")),
                metric_text(result.get(f"coco_{class_name}_3d_AP_0.50")),
                metric_text(result.get(f"coco_{class_name}_3d_AP_0.75")),
            ])
            row_meta.append({
                "epoch": int(result["epoch"]),
                "row_kind": class_name,
                "stripe_index": stripe_index,
            })
            stripe_index += 1

    return {
        "title": "COCO-Style",
        "columns": columns,
        "rows": rows,
        "row_meta": row_meta,
    }


def build_custom_iou_plot_section(results, multi_epoch):
    if not any("custom_iou_bev_mAP" in result for result in results):
        return None

    columns = ["Object", "BEV mAP", "3D mAP", "Precision", "Recall", "F1"]

    rows = []
    row_meta = []
    stripe_index = 0
    include_per_class = not multi_epoch
    for result in results:
        row = [
            result_row_label(result, "Overall", multi_epoch),
            metric_text(result.get("custom_iou_bev_mAP")),
            metric_text(result.get("custom_iou_3d_mAP")),
            metric_text(result.get("custom_iou_precision")),
            metric_text(result.get("custom_iou_recall")),
            metric_text(result.get("custom_iou_f1")),
        ]
        rows.append(row)
        row_meta.append({
            "epoch": int(result["epoch"]),
            "row_kind": "overall",
            "stripe_index": stripe_index,
        })
        stripe_index += 1

        if not include_per_class:
            continue

        for class_name in ("sed", "bus"):
            row = [
                result_row_label(result, display_class_name(class_name), multi_epoch),
                metric_text(result.get(f"custom_iou_{class_name}_bev_mAP")),
                metric_text(result.get(f"custom_iou_{class_name}_3d_mAP")),
                metric_text(result.get(f"custom_iou_{class_name}_precision")),
                metric_text(result.get(f"custom_iou_{class_name}_recall")),
                metric_text(result.get(f"custom_iou_{class_name}_f1")),
            ]
            rows.append(row)
            row_meta.append({
                "epoch": int(result["epoch"]),
                "row_kind": class_name,
                "stripe_index": stripe_index,
            })
            stripe_index += 1

    return {
        "title": "Custom IoU Range",
        "columns": columns,
        "rows": rows,
        "row_meta": row_meta,
    }


def build_nuscenes_plot_section(results, multi_epoch):
    if not any("nuscenes_mAP" in result for result in results):
        return None

    columns = [
        "Object",
        "mAP",
        "AP@0.5m",
        "AP@1.0m",
        "AP@2.0m",
        "AP@4.0m",
        "ATE",
        "ASE",
        "AOE",
    ]

    rows = []
    row_meta = []
    stripe_index = 0
    include_per_class = not multi_epoch
    for result in results:
        rows.append([
            result_row_label(result, "Overall", multi_epoch),
            metric_text(result.get("nuscenes_mAP")),
            metric_text(result.get("nuscenes_AP_0.5m")),
            metric_text(result.get("nuscenes_AP_1.0m")),
            metric_text(result.get("nuscenes_AP_2.0m")),
            metric_text(result.get("nuscenes_AP_4.0m")),
            metric_text(result.get("nuscenes_mATE")),
            metric_text(result.get("nuscenes_mASE")),
            metric_text(result.get("nuscenes_mAOE")),
        ])
        row_meta.append({
            "epoch": int(result["epoch"]),
            "row_kind": "overall",
            "stripe_index": stripe_index,
        })
        stripe_index += 1

        if not include_per_class:
            continue

        for class_name in ("sed", "bus"):
            rows.append([
                result_row_label(result, display_class_name(class_name), multi_epoch),
                metric_text(result.get(f"nuscenes_{class_name}_mAP")),
                metric_text(result.get(f"nuscenes_{class_name}_AP_0.5m")),
                metric_text(result.get(f"nuscenes_{class_name}_AP_1.0m")),
                metric_text(result.get(f"nuscenes_{class_name}_AP_2.0m")),
                metric_text(result.get(f"nuscenes_{class_name}_AP_4.0m")),
                metric_text(result.get(f"nuscenes_{class_name}_ATE")),
                metric_text(result.get(f"nuscenes_{class_name}_ASE")),
                metric_text(result.get(f"nuscenes_{class_name}_AOE")),
            ])
            row_meta.append({
                "epoch": int(result["epoch"]),
                "row_kind": class_name,
                "stripe_index": stripe_index,
            })
            stripe_index += 1

    return {
        "title": "nuScenes-Style",
        "columns": columns,
        "rows": rows,
        "row_meta": row_meta,
    }


def save_evaluation_plot(results, plot_output_path, plot_metadata=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iou_suffixes = available_plot_iou_suffixes(results)
    if len(iou_suffixes) == 0:
        iou_suffixes = [main_plot_iou_suffix(results)]

    best_result = max(
        results,
        key=lambda item: float(item.get("official_main_metric_value", 0.0)),
    )
    multi_epoch = len(results) > 1

    sections = [
        build_official_plot_section(results, iou_suffixes, multi_epoch),
    ]
    custom_iou_section = build_custom_iou_plot_section(results, multi_epoch)
    if custom_iou_section is not None:
        sections.append(custom_iou_section)
    coco_section = build_coco_plot_section(results, multi_epoch)
    if coco_section is not None:
        sections.append(coco_section)
    nuscenes_section = build_nuscenes_plot_section(results, multi_epoch)
    if nuscenes_section is not None:
        sections.append(nuscenes_section)

    max_columns = max(len(section["columns"]) for section in sections)
    total_rows = sum(len(section["rows"]) for section in sections)
    fig_width = max(13.0, 1.15 * max_columns)
    fig_height = max(7.0, 3.4 + 0.46 * total_rows + 0.8 * len(sections))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    fig.suptitle("Evaluation Results", fontsize=15, y=0.975)
    summary_lines = [
        (
            f"Best epoch: {best_result['epoch']} | "
            f"{best_result.get('official_main_metric_key', 'official_bev_mAP_0.3')} = "
            f"{official_ap_text(best_result.get('official_main_metric_value'))}"
        )
    ]
    if plot_metadata is not None:
        summary_lines.append(
            (
                f"Model: {plot_metadata['model_type']} | "
                f"Split: {plot_metadata['split_mode']} | "
                f"Scope: {plot_metadata['eval_scope']}"
            )
        )
        summary_lines.append(
            (
                f"Train: {plot_metadata['train_sequences']} | "
                f"Val: {plot_metadata['val_sequences']}"
            )
        )
        summary_lines.append(
            (
                f"Frames: {best_result.get('official_num_eval_frames', 0)} | "
                f"Backend: {best_result.get('official_iou_backend_used', '-')} | "
                f"Score threshold: {plot_metadata['detection_score_thresh']:.2f} | "
                f"Det IoU: {best_result.get('official_detection_iou_threshold', 0.0):.2f} | "
                f"Shown AP IoU: 0.3, 0.5"
            )
        )
    if "custom_iou_bev_mAP" in best_result:
        custom_iou_text = ", ".join(
            format_custom_iou_suffix(value)
            for value in best_result.get("custom_iou_thresholds", [])
        )
        summary_lines.append(
            (
                f"Custom IoU range: {custom_iou_text} | "
                f"BEV mAP={metric_text(best_result.get('custom_iou_bev_mAP'))} | "
                f"3D mAP={metric_text(best_result.get('custom_iou_3d_mAP'))} | "
                f"P={metric_text(best_result.get('custom_iou_precision'))} | "
                f"R={metric_text(best_result.get('custom_iou_recall'))} | "
                f"F1={metric_text(best_result.get('custom_iou_f1'))}"
            )
        )
    if "nuscenes_mAP" in best_result:
        summary_lines.append(
            (
                f"nuScenes-style: mAP={metric_text(best_result.get('nuscenes_mAP'))} | "
                f"AP@0.5m={metric_text(best_result.get('nuscenes_AP_0.5m'))} | "
                f"AP@1.0m={metric_text(best_result.get('nuscenes_AP_1.0m'))} | "
                f"AP@2.0m={metric_text(best_result.get('nuscenes_AP_2.0m'))} | "
                f"AP@4.0m={metric_text(best_result.get('nuscenes_AP_4.0m'))} | "
                f"mATE={metric_text(best_result.get('nuscenes_mATE'))} | "
                f"mASE={metric_text(best_result.get('nuscenes_mASE'))} | "
                f"mAOE={metric_text(best_result.get('nuscenes_mAOE'))}"
            )
        )
    ax.text(
        0.5,
        0.93,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#374151",
        linespacing=1.55,
    )

    summary_bottom = 0.80
    section_gap = 0.03
    title_height = 0.028
    raw_table_heights = [
        0.030 + 0.027 * (len(section["rows"]) + 1)
        for section in sections
    ]
    total_raw_height = sum(raw_table_heights) + len(sections) * title_height + (len(sections) - 1) * section_gap
    available_height = summary_bottom - 0.05
    scale = min(1.0, available_height / max(total_raw_height, 1e-6))
    current_top = summary_bottom

    for section, raw_table_height in zip(sections, raw_table_heights):
        ax.text(
            0.03,
            current_top,
            section["title"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#111827",
        )
        scaled_title_height = title_height * scale
        table_height = raw_table_height * scale
        table_bottom = current_top - scaled_title_height - table_height
        table = ax.table(
            cellText=section["rows"],
            colLabels=section["columns"],
            bbox=[0.03, table_bottom, 0.94, table_height],
            cellLoc="center",
            colLoc="center",
        )
        style_metric_table(
            table,
            row_meta=section["row_meta"],
            best_epoch=int(best_result["epoch"]),
            font_size=max(7.8, 9.5 * scale),
        )
        table.scale(1.0, max(1.0, 1.2 * scale))
        current_top = table_bottom - (section_gap * scale)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    fig.savefig(plot_output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def yaml_safe_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): yaml_safe_value(child_value)
            for key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [yaml_safe_value(item) for item in value]
    return str(value)


def collect_method_summary(result):
    summary = {
        "official": {
            "main_metric_key": result.get("official_main_metric_key"),
            "main_metric_value": official_ap_value(result.get("official_main_metric_value")),
            "bev_mAP_0.3": official_ap_value(result.get("official_bev_mAP_0.3")),
            "bev_mAP_0.5": official_ap_value(result.get("official_bev_mAP_0.5")),
            "bev_mAP_0.7": official_ap_value(result.get("official_bev_mAP_0.7")),
            "3d_mAP_0.3": official_ap_value(result.get("official_3d_mAP_0.3")),
            "3d_mAP_0.5": official_ap_value(result.get("official_3d_mAP_0.5")),
            "3d_mAP_0.7": official_ap_value(result.get("official_3d_mAP_0.7")),
        }
    }
    if "coco_bev_mAP" in result:
        summary["coco_style"] = {
            "bev_mAP": result.get("coco_bev_mAP"),
            "bev_AP_0.50": result.get("coco_bev_AP_0.50"),
            "bev_AP_0.75": result.get("coco_bev_AP_0.75"),
            "3d_mAP": result.get("coco_3d_mAP"),
            "3d_AP_0.50": result.get("coco_3d_AP_0.50"),
            "3d_AP_0.75": result.get("coco_3d_AP_0.75"),
        }
    if "custom_iou_bev_mAP" in result:
        summary["custom_iou_range"] = {
            "iou_thresholds": result.get("custom_iou_thresholds"),
            "bev_mAP": result.get("custom_iou_bev_mAP"),
            "3d_mAP": result.get("custom_iou_3d_mAP"),
            "precision": result.get("custom_iou_precision"),
            "recall": result.get("custom_iou_recall"),
            "f1": result.get("custom_iou_f1"),
        }
    if "nuscenes_mAP" in result:
        summary["nuscenes_style"] = {
            "mAP": result.get("nuscenes_mAP"),
            "AP_0.5m": result.get("nuscenes_AP_0.5m"),
            "AP_1.0m": result.get("nuscenes_AP_1.0m"),
            "AP_2.0m": result.get("nuscenes_AP_2.0m"),
            "AP_4.0m": result.get("nuscenes_AP_4.0m"),
            "mATE": result.get("nuscenes_mATE"),
            "mASE": result.get("nuscenes_mASE"),
            "mAOE": result.get("nuscenes_mAOE"),
        }
    return summary


def build_yaml_export(results, plot_metadata=None):
    if len(results) == 0:
        raise ValueError("Cannot export YAML because evaluation results are empty.")

    best_result = max(
        results,
        key=lambda item: float(item.get("official_main_metric_value", 0.0)),
    )

    checkpoint_entries = []
    for result in results:
        checkpoint_entries.append({
            "epoch": int(result["epoch"]),
            "checkpoint_path": str(result.get("checkpoint_path", "")),
            "method_summary": collect_method_summary(result),
            "all_metrics": yaml_safe_value(result),
        })

    export_data = {
        "generated_at": datetime.now().isoformat(),
        "plot_metadata": yaml_safe_value(plot_metadata or {}),
        "best_result": {
            "epoch": int(best_result["epoch"]),
            "checkpoint_path": str(best_result.get("checkpoint_path", "")),
            "method_summary": collect_method_summary(best_result),
        },
        "checkpoints": checkpoint_entries,
    }
    return export_data


def resolve_yaml_output_path(plot_output_path):
    base, _ = os.path.splitext(plot_output_path)
    return f"{base}.yml"


def save_evaluation_yaml(results, yaml_output_path, plot_metadata=None):
    export_data = yaml_safe_value(
        build_yaml_export(results, plot_metadata=plot_metadata)
    )
    with open(yaml_output_path, "w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(
            export_data,
            yaml_file,
            sort_keys=False,
            allow_unicode=True,
        )


def print_checkpoint_metrics(epoch, metrics):
    main_key = metrics.get("official_main_metric_key", "official_bev_mAP_0.3")
    print(
        f"epoch={epoch} "
        f"{main_key}={official_ap_text(metrics.get('official_main_metric_value'))} "
        f"bev@0.3={official_ap_text(metrics.get('official_bev_mAP_0.3'))} "
        f"bev@0.5={official_ap_text(metrics.get('official_bev_mAP_0.5'))} "
        f"bev@0.7={official_ap_text(metrics.get('official_bev_mAP_0.7'))} "
        f"3d@0.3={official_ap_text(metrics.get('official_3d_mAP_0.3'))} "
        f"3d@0.5={official_ap_text(metrics.get('official_3d_mAP_0.5'))} "
        f"3d@0.7={official_ap_text(metrics.get('official_3d_mAP_0.7'))} "
        f"score_thr={metric_text(metrics.get('official_detection_score_threshold'))} "
        f"p={metric_text(metrics.get('official_detection_precision'))} "
        f"r={metric_text(metrics.get('official_detection_recall'))} "
        f"f1={metric_text(metrics.get('official_detection_f1'))} "
        f"tp={metrics.get('official_detection_tp', 0)} "
        f"fp={metrics.get('official_detection_fp', 0)} "
        f"fn={metrics.get('official_detection_fn', 0)} "
        f"frames={metrics.get('official_num_eval_frames', 0)}"
    )
    if "coco_bev_mAP" in metrics:
        print(
            f"  coco-style "
            f"bev_mAP={metric_text(metrics.get('coco_bev_mAP'))} "
            f"bev@0.50={metric_text(metrics.get('coco_bev_AP_0.50'))} "
            f"bev@0.75={metric_text(metrics.get('coco_bev_AP_0.75'))} "
            f"3d_mAP={metric_text(metrics.get('coco_3d_mAP'))} "
            f"3d@0.50={metric_text(metrics.get('coco_3d_AP_0.50'))} "
            f"3d@0.75={metric_text(metrics.get('coco_3d_AP_0.75'))}"
        )
    if "custom_iou_bev_mAP" in metrics:
        custom_iou_text = ", ".join(
            format_custom_iou_suffix(value)
            for value in metrics.get("custom_iou_thresholds", [])
        )
        print(
            f"  custom-iou-range "
            f"iou=[{custom_iou_text}] "
            f"bev_mAP={metric_text(metrics.get('custom_iou_bev_mAP'))} "
            f"3d_mAP={metric_text(metrics.get('custom_iou_3d_mAP'))} "
            f"p={metric_text(metrics.get('custom_iou_precision'))} "
            f"r={metric_text(metrics.get('custom_iou_recall'))} "
            f"f1={metric_text(metrics.get('custom_iou_f1'))}"
        )
    if "nuscenes_mAP" in metrics:
        print(
            f"  nuscenes-style "
            f"mAP={metric_text(metrics.get('nuscenes_mAP'))} "
            f"AP@0.5m={metric_text(metrics.get('nuscenes_AP_0.5m'))} "
            f"AP@1.0m={metric_text(metrics.get('nuscenes_AP_1.0m'))} "
            f"AP@2.0m={metric_text(metrics.get('nuscenes_AP_2.0m'))} "
            f"AP@4.0m={metric_text(metrics.get('nuscenes_AP_4.0m'))} "
            f"mATE={metric_text(metrics.get('nuscenes_mATE'))} "
            f"mASE={metric_text(metrics.get('nuscenes_mASE'))} "
            f"mAOE={metric_text(metrics.get('nuscenes_mAOE'))}"
        )


def build_eval_context(args):
    device = select_evaluation_device(args.cuda, args.gpu_ids)
    cfg = DataConfig()
    checkpoint_paths = find_epoch_checkpoints(args.checkpoint_root, args.epoch_step)
    if len(checkpoint_paths) == 0:
        raise ValueError(f"No epoch checkpoints found in {args.checkpoint_root}")

    apply_checkpoint_config_defaults(args, checkpoint_paths)
    model_type = resolve_model_type(args, checkpoint_paths)
    plot_output_path = resolve_plot_output_path(args, checkpoint_paths, model_type)

    _, validation_dataset, _, validation_loader = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        class_to_idx=CLASS_TO_IDX,
        ignore_unmapped_classes=True,
        split_mode=args.split_mode,
        split_dir=args.split_dir,
        scope_mode=args.eval_scope,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
    )
    if len(validation_dataset) == 0:
        raise ValueError("Validation split is empty.")

    model = build_model(
        model_type=model_type,
        device=device,
        num_classes=NUM_CLASSES,
    )

    return {
        "device": device,
        "checkpoint_paths": checkpoint_paths,
        "model_type": model_type,
        "plot_output_path": plot_output_path,
        "plot_metadata": build_plot_metadata(args, model_type),
        "validation_loader": validation_loader,
        "model": model,
    }


def main():
    args = parse_args()
    context = build_eval_context(args)
    device = context["device"]
    checkpoint_paths = context["checkpoint_paths"]
    model_type = context["model_type"]
    plot_output_path = context["plot_output_path"]
    plot_metadata = context["plot_metadata"]
    validation_loader = context["validation_loader"]
    model = context["model"]

    print(f"Evaluation classes: {CLASS_NAMES}")
    print(f"Using evaluation device: {device}")
    print(f"Evaluation scope: {args.eval_scope}")
    print(f"Official evaluator: {args.official_eval_version}")
    print(
        f"Evaluating {len(checkpoint_paths)} checkpoint(s) from {args.checkpoint_root}",
        flush=True,
    )
    if plot_output_path is not None:
        print(f"Plot output path: {plot_output_path}", flush=True)

    results = []
    for epoch, checkpoint_path in tqdm.tqdm(
        checkpoint_paths,
        desc="Checkpoints",
        ncols=120,
    ):
        load_model_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
        metrics = evaluate_checkpoint_with_kradar_revised(
            model=model,
            dataloader=validation_loader,
            device=device,
            num_classes=NUM_CLASSES,
            prepare_model_inputs=prepare_model_inputs,
            max_detections=args.max_detections,
            heatmap_nms_kernel=args.heatmap_nms_kernel,
            yolox_nms_iou=args.yolox_nms_iou,
            scope_mode=args.eval_scope,
            official_eval_enabled=True,
            official_eval_version=args.official_eval_version,
            official_eval_iou_backend=args.official_eval_iou_backend,
            official_eval_iou_mode=args.official_eval_iou_mode,
            custom_iou_range_eval_enabled=args.custom_iou_range_eval_enabled,
            custom_iou_thresholds=args.custom_iou_thresholds,
            coco_style_eval_enabled=args.coco_style_eval_enabled,
            nuscenes_style_eval_enabled=args.nuscenes_style_eval_enabled,
            detection_score_thresh=args.detection_score_thresh,
        )
        result = {
            "epoch": epoch,
            "checkpoint_path": checkpoint_path,
            **metrics,
        }
        results.append(result)
        print_checkpoint_metrics(epoch, metrics)

    if len(results) > 0:
        best_result = max(
            results,
            key=lambda item: float(item.get("official_main_metric_value", 0.0)),
        )
        print(
            "best_epoch:",
            f"epoch={best_result['epoch']}",
            f"{best_result.get('official_main_metric_key', 'official_bev_mAP_0.3')}="
            f"{official_ap_text(best_result.get('official_main_metric_value'))}",
        )

    if plot_output_path is not None:
        output_dir = os.path.dirname(plot_output_path)
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
        save_evaluation_plot(results, plot_output_path, plot_metadata=plot_metadata)
        yaml_output_path = resolve_yaml_output_path(plot_output_path)
        save_evaluation_yaml(results, yaml_output_path, plot_metadata=plot_metadata)
        print(f"Saved evaluation plot: {plot_output_path}")
        print(f"Saved evaluation YAML: {yaml_output_path}")


if __name__ == "__main__":
    main()
