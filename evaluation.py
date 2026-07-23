"""Coordinate-aware evaluation entrypoint for this project.

Cartesian checkpoints use K-Radar official rotated BEV/3D evaluation.
Polar checkpoints use direct axis-aligned RA BEV evaluation.
This file:
1. runs the project model,
2. keeps predictions/GT in the checkpoint's direct geometry by default,
3. calls the matching AP evaluator.
"""

import argparse
import json
from numbers import Real
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter

from cfg_model import (
    AZIMUTH_AXIS,
    ELEVATION_AXIS,
    RANGE_AXIS,
    RDR_SP_CUBE,
    SCOPE_CHOICES,
    SCOPE_FULL,
    SCOPE_NARROW,
    denormalize_rae_boxes_to_local_scope,
    denormalize_rae_boxes_for_scope,
    normalized_rae_box_centers_in_cartesian_roi,
)
from coordinate_modes import (
    BOX_COORDINATE_CARTESIAN,
    BOX_COORDINATE_POLAR,
    EVAL_COORDINATE_AUTO,
    EVAL_COORDINATE_BOTH,
    EVAL_COORDINATE_CHOICES,
    resolve_evaluation_coordinate_mode,
    validate_box_coordinate_mode,
)
from dataloader import (
    build_train_val_dataloaders,
    normalize_sequence_list,
    prepare_model_inputs,
)
from domain_shift_tables import (
    build_model_configuration,
    update_domain_shift_tables,
)
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
from eval.kitti_eval.rotate_iou_cpu import (
    rotate_iou_gpu_eval as rotate_iou_cpu_eval,
)
from eval.nuscenes_style import compute_nuscenes_style_metrics
from eval.polar_ap import compute_polar_ap_metrics
from models import MODEL_TYPES, build_model
from train_mode_utils import (
    apply_task_configuration,
    infer_include_bus_as_target_from_checkpoint_config,
    initialize_model_from_checkpoint,
    normalize_optional_path,
    resolve_loss_mode,
)
from training_utils.checkpoints import format_sequence_run_name
from training_utils.radenet_utils import (
    centerpoint_outputs_to_metric_regression,
    metric_boxes_to_raw_local_rae,
    regression_cell_to_metric_box,
    regression_cell_to_normalized_rae_box,
)
from training_utils.training_loop import validate_loss
from training_utils.yolox_utils import yolox_outputs_to_detections
from zxy_config import DataConfig

try:
    from eval_cfg import EVAL_CONFIG
except ImportError:
    EVAL_CONFIG = {}

_EVAL_CFG_MISSING = object()
HEATMAP_SCORE_MODES = ("peak_times_local_mean", "peak_only")
HEATMAP_PEAK_WEIGHT = 0.85
HEATMAP_LOCAL_MEAN_WEIGHT = 0.15
RADENET_ROTATED_NMS_IOU_THRESHOLD = 0.3
OFFICIAL_CLASS_TOKEN_BY_DATASET_NAME = {
    "Sedan": "sed",
    "Bus or Truck": "bus",
}
SPLIT_BBOX_COUNT_CLASS_NAMES = ("Sedan", "Bus or Truck")


def eval_cfg_value(key):
    return EVAL_CONFIG.get(key, _EVAL_CFG_MISSING)


def should_inherit_from_checkpoint(key):
    value = eval_cfg_value(key)
    return value is _EVAL_CFG_MISSING or value is None


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


def load_model_checkpoint(model, checkpoint_path, device, include_bus_as_target=True):
    checkpoint = initialize_model_from_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        map_location=device,
        include_bus_as_target=include_bus_as_target,
    )
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
            "0620_model_12_global_best_epoch_059_seq1-11.pth"
        ),
        "epoch_step": 1,
        "end_epoch": None,
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
        "eval_coordinate_mode": EVAL_COORDINATE_AUTO,
        "box_coordinate_mode": None,
        "cartesian_gt_root": None,
        "polar_gt_root": "/home/local/xinyu/K-Radar-GT-Polar-v2.9",
        "ignore_object_label_minus_one": False,
        "include_bus_as_target": True,
        "gt_object_ignore_override_path": None,
        "train_control_split_enabled": False,
        "train_control_split_dir": None,
        "ignore_class_names": (
            "Pedestrian",
            "Pedestrian Group",
            "Bicycle",
            "Bicycle Group",
            "Motorcycle",
        ),
        "ignore_mask_margin": 1.0,
        "ignore_mask_expand_ratio": 1.0,
        "eval_ignore_suppress_enabled": False,
        "eval_ignore_expand_ratio": 1.5,
        "eval_ignore_suppress_margin": 1.0,
        "loss_eval_enabled": False,
        "heatmap_radius": 3,
        "centerpoint_gwd_loss_weight": 2.0,
        "quality_loss_weight": 0.25,
        "table_txt_enabled": False,
        "table_output_base_dir": "evaluation_plots",
        "domain_comparison_enabled": True,
        "domain_comparison_output_dir": "evaluation_results",
        "domain_comparison_sequence_info_path": "sequence_information.csv",
        "evaluation_tensorboard_log_dir": "runs",
        "max_detections": 64,
        "heatmap_nms_kernel": 3,
        "heatmap_score_mode": "peak_times_local_mean",
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
        "official_detection_metrics_enabled": True,
        "polar_iou_thresholds": [0.3, 0.5],
        "official_ap03_only": False,
        "terminal_epoch_table_enabled": False,
        "group_checkpoint_plot_best_only": False,
        "ap_score_thresh": 0.01,
        "score_thresh": 0.3,
        "plot_output": None,
    }
    eval_config = dict(EVAL_CONFIG)
    if (
        "centerpoint_gwd_loss_weight" not in eval_config
        and "centerpoint_giou_loss_weight" in eval_config
    ):
        eval_config["centerpoint_gwd_loss_weight"] = eval_config[
            "centerpoint_giou_loss_weight"
        ]
    cfg_defaults.update(eval_config)
    if "score_thresh" not in EVAL_CONFIG and "detection_score_thresh" in EVAL_CONFIG:
        cfg_defaults["score_thresh"] = EVAL_CONFIG["detection_score_thresh"]

    parser = argparse.ArgumentParser(
        description="Run official K-Radar KITTI-style evaluation."
    )
    parser.add_argument("--checkpoint-root", default=cfg_defaults["checkpoint_root"])
    parser.add_argument("--epoch-step", type=int, default=cfg_defaults["epoch_step"])
    parser.add_argument("--end-epoch", type=int, default=cfg_defaults["end_epoch"])
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
    parser.add_argument(
        "--eval-coordinate-mode",
        default=cfg_defaults["eval_coordinate_mode"],
        choices=list(EVAL_COORDINATE_CHOICES),
        help=(
            "auto uses the checkpoint's direct geometry; both also computes "
            "the converted auxiliary geometry."
        ),
    )
    parser.add_argument(
        "--box-coordinate-mode",
        default=cfg_defaults["box_coordinate_mode"],
        choices=["auto", BOX_COORDINATE_POLAR, BOX_COORDINATE_CARTESIAN],
    )
    parser.add_argument(
        "--cartesian-gt-root",
        default=cfg_defaults["cartesian_gt_root"],
    )
    parser.add_argument(
        "--polar-gt-root",
        default=cfg_defaults["polar_gt_root"],
    )
    parser.add_argument(
        "--ignore-object-label-minus-one",
        default=cfg_defaults["ignore_object_label_minus_one"],
    )
    parser.add_argument(
        "--include-bus-as-target",
        default=cfg_defaults["include_bus_as_target"],
    )
    parser.add_argument(
        "--gt-object-ignore-override-path",
        default=cfg_defaults["gt_object_ignore_override_path"],
    )
    parser.add_argument(
        "--train-control-split-enabled",
        default=cfg_defaults["train_control_split_enabled"],
    )
    parser.add_argument(
        "--train-control-split-dir",
        default=cfg_defaults["train_control_split_dir"],
    )
    parser.add_argument("--ignore-mask-margin", type=float, default=cfg_defaults["ignore_mask_margin"])
    parser.add_argument(
        "--ignore-mask-expand-ratio",
        type=float,
        default=cfg_defaults["ignore_mask_expand_ratio"],
    )
    parser.add_argument(
        "--eval-ignore-suppress-enabled",
        default=cfg_defaults["eval_ignore_suppress_enabled"],
    )
    parser.add_argument(
        "--eval-ignore-expand-ratio",
        type=float,
        default=cfg_defaults["eval_ignore_expand_ratio"],
    )
    parser.add_argument(
        "--eval-ignore-suppress-margin",
        type=float,
        default=cfg_defaults["eval_ignore_suppress_margin"],
    )
    parser.add_argument(
        "--loss-eval-enabled",
        default=cfg_defaults["loss_eval_enabled"],
    )
    parser.add_argument("--heatmap-radius", type=int, default=cfg_defaults["heatmap_radius"])
    parser.add_argument(
        "--centerpoint-gwd-loss-weight",
        "--centerpoint-giou-loss-weight",
        dest="centerpoint_gwd_loss_weight",
        type=float,
        default=cfg_defaults["centerpoint_gwd_loss_weight"],
        help="GWD loss weight; the GIoU spelling is accepted only for legacy commands.",
    )
    parser.add_argument(
        "--quality-loss-weight",
        type=float,
        default=cfg_defaults["quality_loss_weight"],
    )
    parser.add_argument("--max-detections", type=int, default=cfg_defaults["max_detections"])
    parser.add_argument("--heatmap-nms-kernel", type=int, default=cfg_defaults["heatmap_nms_kernel"])
    parser.add_argument(
        "--heatmap-score-mode",
        default=cfg_defaults["heatmap_score_mode"],
        choices=list(HEATMAP_SCORE_MODES),
        help=(
            "peak_only uses pure local peaks; peak_times_local_mean uses "
            "0.85 * peak_score + 0.15 * local_mean."
        ),
    )
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
        "--official-detection-metrics-enabled",
        default=cfg_defaults["official_detection_metrics_enabled"],
    )
    parser.add_argument(
        "--polar-iou-thresholds",
        default=cfg_defaults["polar_iou_thresholds"],
    )
    parser.add_argument(
        "--official-ap03-only",
        default=cfg_defaults["official_ap03_only"],
    )
    parser.add_argument(
        "--terminal-epoch-table-enabled",
        default=cfg_defaults["terminal_epoch_table_enabled"],
    )
    parser.add_argument(
        "--group-checkpoint-plot-best-only",
        default=cfg_defaults["group_checkpoint_plot_best_only"],
    )
    parser.add_argument(
        "--ap-score-thresh",
        type=float,
        default=cfg_defaults["ap_score_thresh"],
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=cfg_defaults["score_thresh"],
    )
    parser.add_argument(
        "--detection-score-thresh",
        dest="score_thresh",
        type=float,
        default=cfg_defaults["score_thresh"],
    )
    parser.add_argument("--plot-output", default=cfg_defaults["plot_output"])
    parser.add_argument(
        "--table-txt-enabled",
        default=cfg_defaults["table_txt_enabled"],
    )
    parser.add_argument(
        "--table-output-base-dir",
        default=cfg_defaults["table_output_base_dir"],
    )
    parser.add_argument(
        "--domain-comparison-enabled",
        default=cfg_defaults["domain_comparison_enabled"],
    )
    parser.add_argument(
        "--domain-comparison-output-dir",
        default=cfg_defaults["domain_comparison_output_dir"],
    )
    parser.add_argument(
        "--domain-comparison-sequence-info-path",
        default=cfg_defaults["domain_comparison_sequence_info_path"],
    )
    parser.add_argument(
        "--evaluation-tensorboard-log-dir",
        default=cfg_defaults["evaluation_tensorboard_log_dir"],
    )
    args = parser.parse_args()
    args.ignore_class_names = tuple(cfg_defaults.get("ignore_class_names", ()))
    if args.end_epoch is not None:
        args.end_epoch = int(args.end_epoch)
    args.ap_score_thresh = float(args.ap_score_thresh)
    args.detection_score_thresh = float(args.score_thresh)
    if args.ap_score_thresh < 0.0:
        raise ValueError(
            f"ap_score_thresh must be non-negative, got {args.ap_score_thresh!r}"
        )
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
    args.official_detection_metrics_enabled = normalize_bool_flag(
        args.official_detection_metrics_enabled,
        name="official_detection_metrics_enabled",
    )
    args.polar_iou_thresholds = normalize_float_thresholds(
        args.polar_iou_thresholds,
        name="polar_iou_thresholds",
    )
    args.official_ap03_only = normalize_bool_flag(
        args.official_ap03_only,
        name="official_ap03_only",
    )
    args.terminal_epoch_table_enabled = normalize_bool_flag(
        args.terminal_epoch_table_enabled,
        name="terminal_epoch_table_enabled",
    )
    args.group_checkpoint_plot_best_only = normalize_bool_flag(
        args.group_checkpoint_plot_best_only,
        name="group_checkpoint_plot_best_only",
    )
    args.eval_ignore_suppress_enabled = normalize_bool_flag(
        args.eval_ignore_suppress_enabled,
        name="eval_ignore_suppress_enabled",
    )
    args.loss_eval_enabled = normalize_bool_flag(
        args.loss_eval_enabled,
        name="loss_eval_enabled",
    )
    args.table_txt_enabled = normalize_bool_flag(
        args.table_txt_enabled,
        name="table_txt_enabled",
    )
    args.domain_comparison_enabled = normalize_bool_flag(
        args.domain_comparison_enabled,
        name="domain_comparison_enabled",
    )
    if args.custom_iou_range_eval_enabled and len(args.custom_iou_thresholds) == 0:
        raise ValueError(
            "custom_iou_range_eval_enabled is True, but custom_iou_thresholds is empty."
        )
    if args.eval_ignore_expand_ratio <= 0.0:
        raise ValueError(
            f"eval_ignore_expand_ratio must be greater than 0, got {args.eval_ignore_expand_ratio!r}"
        )
    if args.eval_ignore_suppress_margin < 0.0:
        raise ValueError(
            f"eval_ignore_suppress_margin must be non-negative, got {args.eval_ignore_suppress_margin!r}"
        )
    if args.ignore_mask_expand_ratio <= 0.0:
        raise ValueError(
            f"ignore_mask_expand_ratio must be greater than 0, got {args.ignore_mask_expand_ratio!r}"
        )
    apply_eval_profile(args)
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


def apply_eval_profile(args):
    if getattr(args, "official_ap03_only", False):
        args.official_eval_iou_mode = "easy"
        args.custom_iou_range_eval_enabled = False
        args.coco_style_eval_enabled = False
        args.nuscenes_style_eval_enabled = False
        args.official_detection_metrics_enabled = False
        args.terminal_epoch_table_enabled = True


def apply_standalone_evaluation_coordinate_mode(args):
    settings = resolve_evaluation_coordinate_mode(
        eval_coordinate_mode=getattr(
            args,
            "eval_coordinate_mode",
            EVAL_COORDINATE_AUTO,
        ),
        box_coordinate_mode=args.box_coordinate_mode,
    )
    args.eval_coordinate_mode = settings["requested_mode"]
    args.effective_eval_coordinate_mode = settings["effective_mode"]
    args.official_eval_enabled = settings["official_eval_enabled"]
    args.polar_eval_enabled = settings["polar_eval_enabled"]
    args.evaluation_primary_geometry = settings["primary_geometry"]
    args.official_geometry_source = settings["official_geometry_source"]
    args.polar_geometry_source = settings["polar_geometry_source"]
    return args


def resolve_official_eval_class_name_map(class_names):
    class_name_map = {}
    display_name_map = {}
    for class_id, dataset_class_name in sorted(class_names.items()):
        dataset_class_name = str(dataset_class_name)
        if dataset_class_name not in OFFICIAL_CLASS_TOKEN_BY_DATASET_NAME:
            raise KeyError(
                f"Unsupported evaluation class name {dataset_class_name!r}. "
                f"Known names: {sorted(OFFICIAL_CLASS_TOKEN_BY_DATASET_NAME)}"
            )
        official_name = OFFICIAL_CLASS_TOKEN_BY_DATASET_NAME[dataset_class_name]
        class_name_map[int(class_id)] = official_name
        display_name_map[official_name] = dataset_class_name
    return class_name_map, display_name_map


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


def centerpoint_local_heatmap_mean(heatmap, kernel_size=3):
    if kernel_size <= 1:
        return heatmap
    if kernel_size % 2 == 0:
        raise ValueError(f"Heatmap local-mean kernel must be odd, got {kernel_size}")

    pad = (kernel_size - 1) // 2
    return F.avg_pool2d(
        heatmap,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
        count_include_pad=False,
    )


def build_heatmap_candidate_scores(
        heatmap_scores,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        peak_scores=None,
    ):
    if peak_scores is None:
        peak_scores = centerpoint_heatmap_nms(
            heatmap=heatmap_scores,
            kernel_size=heatmap_nms_kernel,
        )

    if heatmap_score_mode == "peak_only":
        return peak_scores
    if heatmap_score_mode == "peak_times_local_mean":
        local_mean_scores = centerpoint_local_heatmap_mean(
            heatmap=heatmap_scores,
            kernel_size=heatmap_nms_kernel,
        )
        return (
            (HEATMAP_PEAK_WEIGHT * peak_scores)
            + (HEATMAP_LOCAL_MEAN_WEIGHT * local_mean_scores)
        )
    raise ValueError(
        f"Unknown heatmap_score_mode={heatmap_score_mode!r}. "
        f"Expected one of {HEATMAP_SCORE_MODES}."
    )


def gather_dense_feature(feature_map, indices):
    flat = feature_map.flatten(start_dim=2).transpose(1, 2)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
    return flat.gather(dim=1, index=gather_index)


def topk_heatmap_candidates(candidate_scores, max_detections, score_thresh=None):
    flat_scores = candidate_scores.flatten(start_dim=1)
    topk_count = min(max_detections, flat_scores.shape[1])
    scores, flat_indices = flat_scores.topk(topk_count, dim=1)
    keep = scores > 0.0
    if score_thresh is not None:
        keep = keep & (scores > float(score_thresh))
    return scores, flat_indices, keep


def cartesian_rotated_nms_indices(
        boxes,
        scores,
        iou_threshold=RADENET_ROTATED_NMS_IOU_THRESHOLD,
    ):
    """Class-agnostic rotated BEV NMS matching the original RADE-Net."""
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    bev_boxes = torch.stack(
        [
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 3].abs(),
            boxes[:, 4].abs(),
            boxes[:, 6],
        ],
        dim=-1,
    ).detach().cpu().numpy()
    keep = []

    while order.numel() > 0:
        current = order[0]
        keep.append(int(current.item()))
        if order.numel() == 1:
            break

        remaining = order[1:]
        current_index = int(current.item())
        remaining_indices = remaining.detach().cpu().numpy()
        ious = rotate_iou_cpu_eval(
            bev_boxes[current_index:current_index + 1],
            bev_boxes[remaining_indices],
            criterion=-1,
        )[0]
        keep_mask = torch.as_tensor(
            ious < float(iou_threshold),
            dtype=torch.bool,
            device=remaining.device,
        )
        order = remaining[keep_mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


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
        heatmap_score_mode="peak_times_local_mean",
        score_thresh=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
        scope_modes=None,
        full_rae_shapes=None,
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
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
    rescored_heatmap = build_heatmap_candidate_scores(
        heatmap_scores=heatmap_scores,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
    )

    scores, flat_indices, keep = topk_heatmap_candidates(
        candidate_scores=rescored_heatmap,
        max_detections=max_detections,
        score_thresh=score_thresh,
    )

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
    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        if scope_modes is None or full_rae_shapes is None:
            raise ValueError(
                "Cartesian CenterPoint decoding requires scope_modes and "
                "full_rae_shapes."
            )
        regression_map = centerpoint_outputs_to_metric_regression(outputs)
        pred_reg = gather_dense_feature(regression_map, box_indices)
        metric_boxes = []
        for batch_index in range(pred_reg.shape[0]):
            metric_boxes.append(
                regression_cell_to_metric_box(
                    pred_reg=pred_reg[batch_index],
                    y_idx=y_idx[batch_index],
                    x_idx=x_idx[batch_index],
                    feature_shape=(box_h, box_w),
                    scope_mode=scope_modes[batch_index],
                    full_rae_shape=full_rae_shapes[batch_index],
                )
            )
        boxes = torch.stack(metric_boxes, dim=0)
    else:
        center_offset = gather_dense_feature(
            outputs["center_offset"],
            box_indices,
        ).sigmoid()
        center_height = gather_dense_feature(
            outputs["center_height"],
            box_indices,
        ).sigmoid()
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

    return boxes, scores, labels, keep


def official_radenet_outputs_to_detections(
        outputs,
        num_classes,
        scope_modes,
        full_rae_shapes,
        max_detections=64,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        score_thresh=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    if scope_modes is None or full_rae_shapes is None:
        raise ValueError("Official RADE-Net decoding requires scope_modes and full_rae_shapes.")

    heatmap_scores = outputs["heatmap"][:, :num_classes]
    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        # Original RADE-Net ranks detections by the heatmap peak itself.
        # The local-mean re-scoring option belongs to this project's
        # CenterPoint path and must not alter Cartesian RADE confidence.
        heatmap_score_mode = "peak_only"
    rescored_heatmap = build_heatmap_candidate_scores(
        heatmap_scores=heatmap_scores,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
    )
    regression = outputs["regression"]
    batch_size, _, heatmap_h, heatmap_w = heatmap_scores.shape

    scores, flat_indices, keep = topk_heatmap_candidates(
        candidate_scores=rescored_heatmap,
        max_detections=max_detections,
        score_thresh=score_thresh,
    )

    spatial_size = heatmap_h * heatmap_w
    labels = flat_indices // spatial_size
    spatial_indices = flat_indices % spatial_size
    y_idx = spatial_indices // heatmap_w
    x_idx = spatial_indices % heatmap_w
    reg = gather_dense_feature(regression, spatial_indices)

    boxes = []
    for batch_index in range(batch_size):
        decode_function = (
            regression_cell_to_metric_box
            if box_coordinate_mode == BOX_COORDINATE_CARTESIAN
            else regression_cell_to_normalized_rae_box
        )
        boxes.append(
            decode_function(
                pred_reg=reg[batch_index],
                y_idx=y_idx[batch_index].to(reg.dtype),
                x_idx=x_idx[batch_index].to(reg.dtype),
                feature_shape=(heatmap_h, heatmap_w),
                scope_mode=scope_modes[batch_index],
                full_rae_shape=full_rae_shapes[batch_index],
                **(
                    {"absolute_dimensions": False}
                    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN
                    else {}
                ),
            )
        )
    return torch.stack(boxes, dim=0), scores, labels, keep


def decode_batch_predictions(
        outputs,
        num_classes,
        max_detections,
        heatmap_nms_kernel,
        heatmap_score_mode,
        yolox_nms_iou,
        score_thresh=None,
        scope_modes=None,
        full_rae_shapes=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    if "objectness_logits" in outputs:
        if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
            raise ValueError(
                "Cartesian box mode is not implemented for YOLOX outputs."
            )
        return yolox_outputs_to_detections(
            outputs=outputs,
            num_classes=num_classes,
            score_thresh=score_thresh,
            max_detections=max_detections,
            nms_iou_thresh=yolox_nms_iou,
        )

    is_official_radenet_output = (
        "heatmap" in outputs and "regression" in outputs
    )
    if is_official_radenet_output:
        pred_boxes, pred_scores, pred_labels, pred_keep = official_radenet_outputs_to_detections(
            outputs=outputs,
            num_classes=num_classes,
            scope_modes=scope_modes,
            full_rae_shapes=full_rae_shapes,
            max_detections=max_detections,
            heatmap_nms_kernel=heatmap_nms_kernel,
            heatmap_score_mode=heatmap_score_mode,
            score_thresh=score_thresh,
            box_coordinate_mode=box_coordinate_mode,
        )
    else:
        pred_boxes, pred_scores, pred_labels, pred_keep = outputs_to_detections(
            outputs=outputs,
            num_classes=num_classes,
            max_detections=max_detections,
            heatmap_nms_kernel=heatmap_nms_kernel,
            heatmap_score_mode=heatmap_score_mode,
            score_thresh=score_thresh,
            box_coordinate_mode=box_coordinate_mode,
            scope_modes=scope_modes,
            full_rae_shapes=full_rae_shapes,
        )

    batch_predictions = []
    for batch_index in range(pred_boxes.shape[0]):
        keep = pred_keep[batch_index]
        boxes = pred_boxes[batch_index][keep]
        scores = pred_scores[batch_index][keep]
        labels = pred_labels[batch_index][keep]
        if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
            nms_keep = cartesian_rotated_nms_indices(
                boxes=boxes,
                scores=scores,
            )
            boxes = boxes[nms_keep]
            scores = scores[nms_keep]
            labels = labels[nms_keep]
        batch_predictions.append({
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "box_coordinate_mode": box_coordinate_mode,
        })
    return batch_predictions


def filter_predictions_to_scope(
        frame_predictions,
        scope_mode,
        full_rae_shape,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    if scope_mode != SCOPE_NARROW:
        return frame_predictions

    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        boxes = frame_predictions["boxes"]
        roi = RDR_SP_CUBE["ROI"]
        keep = (
            (boxes[:, 0] >= float(roi["x"][0]))
            & (boxes[:, 0] <= float(roi["x"][1]))
            & (boxes[:, 1] >= float(roi["y"][0]))
            & (boxes[:, 1] <= float(roi["y"][1]))
            & (boxes[:, 2] >= float(roi["z"][0]))
            & (boxes[:, 2] <= float(roi["z"][1]))
        )
    else:
        keep = normalized_rae_box_centers_in_cartesian_roi(
            frame_predictions["boxes"],
            scope_mode=scope_mode,
            rae_shape=full_rae_shape,
        )
    return {
        "boxes": frame_predictions["boxes"][keep],
        "scores": frame_predictions["scores"][keep],
        "labels": frame_predictions["labels"][keep],
        "box_coordinate_mode": box_coordinate_mode,
    }


def init_kradar_eval_state():
    return {
        "official_gt_annos": [],
        "official_dt_annos": [],
        "metric_frames": [],
        "polar_frames": [],
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
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    ignore_source = (
        gt_ignore_metric_boxes
        if box_coordinate_mode == BOX_COORDINATE_CARTESIAN
        else gt_ignore_boxes_raw
    )
    if ignore_source is None or pred_boxes.numel() == 0:
        return pred_boxes, pred_scores, pred_labels, 0

    ignore_boxes = ignore_source.to(pred_boxes.device)
    if ignore_boxes.numel() == 0:
        return pred_boxes, pred_scores, pred_labels, 0

    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        pred_center_y = pred_boxes[:, 0]
        pred_center_x = pred_boxes[:, 1]
    else:
        pred_boxes_raw = denormalize_rae_boxes_to_local_scope(
            boxes=pred_boxes,
            scope_mode=scope_mode,
            rae_shape=full_rae_shape,
        )
        pred_center_y = pred_boxes_raw[:, 0]
        pred_center_x = pred_boxes_raw[:, 1]
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
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
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

    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        gt_metric_boxes_all = batch["gt_metric_boxes"][batch_index].to(device)
        gt_metric_boxes = gt_metric_boxes_all[valid_gt]
        pred_metric_boxes = frame_predictions["boxes"]
        gt_polar_boxes = metric_boxes_to_raw_local_rae(
            metric_boxes=gt_metric_boxes,
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
        )
        pred_polar_boxes = metric_boxes_to_raw_local_rae(
            metric_boxes=pred_metric_boxes,
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
        )
    else:
        gt_boxes_all = batch["gt_boxes"][batch_index].to(device)
        gt_boxes = gt_boxes_all[valid_gt]
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
        gt_polar_boxes = denormalize_rae_boxes_for_scope(
            boxes=gt_boxes,
            scope_mode=scope_mode,
            rae_shape=full_rae_shape,
        )
        pred_polar_boxes = denormalize_rae_boxes_for_scope(
            boxes=frame_predictions["boxes"],
            scope_mode=scope_mode,
            rae_shape=full_rae_shape,
        )

    state["official_gt_annos"].append(
        metric_boxes_to_kitti_anno(
            boxes=gt_metric_boxes.detach().cpu(),
            labels=gt_labels.detach().cpu(),
            is_prediction=False,
            class_name_map=official_class_name_map,
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
        }
    )
    state["polar_frames"].append(
        {
            "gt_boxes": gt_polar_boxes.detach().cpu().numpy(),
            "gt_labels": gt_labels.detach().cpu().numpy(),
            "dt_boxes": pred_polar_boxes.detach().cpu().numpy(),
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
        official_class_name_map,
        prepare_model_inputs,
        max_detections=64,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        yolox_nms_iou=0.65,
        ap_score_thresh=0.01,
        detection_score_thresh=0.3,
        scope_mode=SCOPE_FULL,
        eval_ignore_suppress_enabled=False,
        eval_ignore_expand_ratio=1.5,
        eval_ignore_suppress_margin=1.0,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    model.eval()
    state = init_kradar_eval_state()
    state["eval_ignore_suppressed_predictions"] = 0

    for batch in tqdm.tqdm(dataloader, desc="Evaluation", ncols=120, leave=False):
        batch_modes = {
            validate_box_coordinate_mode(value)
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
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)
        batch_predictions = decode_batch_predictions(
            outputs=outputs,
            num_classes=num_classes,
            max_detections=max_detections,
            heatmap_nms_kernel=heatmap_nms_kernel,
            heatmap_score_mode=heatmap_score_mode,
            yolox_nms_iou=yolox_nms_iou,
            score_thresh=ap_score_thresh,
            scope_modes=batch["scope_mode"],
            full_rae_shapes=batch["full_rae_shape"],
            box_coordinate_mode=box_coordinate_mode,
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
        detection_score_thresh=0.3,
        official_eval_class_ids=None,
        official_class_name_map=None,
        polar_eval_enabled=False,
        polar_iou_thresholds=None,
    ):
    if official_eval_version not in ("revised", "kradar"):
        raise ValueError(
            f"evaluation.py only supports K-Radar revised evaluation, got {official_eval_version!r}."
        )

    frame_count = max(
        len(kradar_eval_state["official_gt_annos"]),
        len(kradar_eval_state.get("polar_frames", [])),
    )
    print(
        "Finished model inference. "
        f"Collected {frame_count} eval frames. "
        "Running selected metrics now...",
        flush=True,
    )
    metrics = {}
    if official_eval_enabled:
        metrics.update(
            compute_official_kradar_style_metrics(
                state=kradar_eval_state,
                official_eval_enabled=True,
                official_eval_version="revised",
                official_eval_iou_backend=official_eval_iou_backend,
                official_eval_iou_mode=official_eval_iou_mode,
                official_detection_metrics_enabled=official_detection_metrics_enabled,
                detection_score_thresh=detection_score_thresh,
                official_eval_class_ids=official_eval_class_ids,
                official_class_name_map=official_class_name_map,
            )
        )
    if official_eval_enabled and custom_iou_range_eval_enabled:
        metrics.update(
            compute_custom_iou_range_metrics(
                state=kradar_eval_state,
                iou_backend=official_eval_iou_backend,
                iou_thresholds=custom_iou_thresholds,
                detection_score_thresh=detection_score_thresh,
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
    if polar_eval_enabled:
        metrics.update(
            compute_polar_ap_metrics(
                polar_frames=kradar_eval_state.get("polar_frames", []),
                class_ids=official_eval_class_ids,
                class_name_map=official_class_name_map,
                iou_thresholds=polar_iou_thresholds,
            )
        )
    metrics["evaluation_num_eval_frames"] = int(frame_count)
    print("Metric computation finished.", flush=True)
    if official_eval_enabled:
        metrics["mAP"] = float(
            metrics.get("official_main_metric_value", 0.0)
        )
    else:
        metrics["mAP"] = float(metrics.get("polar_bev_mAP", 0.0))
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
        detection_score_thresh=0.3,
        eval_ignore_suppress_enabled=False,
        eval_ignore_expand_ratio=1.5,
        eval_ignore_suppress_margin=1.0,
        polar_eval_enabled=False,
        polar_iou_thresholds=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    if not official_eval_enabled and not polar_eval_enabled:
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
        detection_score_thresh=detection_score_thresh,
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
        detection_score_thresh=detection_score_thresh,
        official_eval_class_ids=sorted(official_class_name_map.keys()),
        official_class_name_map=official_class_name_map,
        polar_eval_enabled=polar_eval_enabled,
        polar_iou_thresholds=polar_iou_thresholds,
    )
    metrics["ap_score_thresh"] = float(ap_score_thresh)
    metrics["eval_ignore_suppressed_predictions"] = int(
        kradar_eval_state.get("eval_ignore_suppressed_predictions", 0)
    )
    return metrics


@torch.no_grad()
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
        detection_score_thresh=0.3,
        polar_eval_enabled=False,
        polar_iou_thresholds=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    del train_dataloader
    del evaluate_train

    val_eval_metrics = evaluate_checkpoint_with_kradar_revised(
        model=model,
        dataloader=val_dataloader,
        device=device,
        num_classes=num_classes,
        official_class_name_map=official_class_name_map,
        prepare_model_inputs=prepare_model_inputs,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
        yolox_nms_iou=yolox_nms_iou,
        scope_mode=scope_mode,
        official_eval_enabled=official_eval_enabled,
        official_eval_version=official_eval_version,
        official_eval_iou_backend=official_eval_iou_backend,
        official_eval_iou_mode=official_eval_iou_mode,
        official_detection_metrics_enabled=official_detection_metrics_enabled,
        custom_iou_range_eval_enabled=custom_iou_range_eval_enabled,
        custom_iou_thresholds=custom_iou_thresholds,
        coco_style_eval_enabled=coco_style_eval_enabled,
        nuscenes_style_eval_enabled=nuscenes_style_eval_enabled,
        ap_score_thresh=ap_score_thresh,
        detection_score_thresh=detection_score_thresh,
        polar_eval_enabled=polar_eval_enabled,
        polar_iou_thresholds=polar_iou_thresholds,
        box_coordinate_mode=box_coordinate_mode,
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


def find_epoch_checkpoints(checkpoint_root, epoch_step, end_epoch=None):
    if epoch_step <= 0:
        raise ValueError(f"--epoch-step must be greater than 0, got {epoch_step}")
    if end_epoch is not None and end_epoch <= 0:
        raise ValueError(f"--end-epoch must be greater than 0, got {end_epoch}")

    if os.path.isfile(checkpoint_root):
        epoch = checkpoint_epoch(checkpoint_root)
        if epoch is None:
            checkpoint = load_torch_checkpoint(checkpoint_root, map_location="cpu")
            epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
        if end_epoch is not None and epoch > end_epoch:
            return []
        return [(epoch, checkpoint_root)]

    checkpoint_by_epoch = {}
    for filename in os.listdir(checkpoint_root):
        if not filename.endswith(".pth"):
            continue

        checkpoint_path = os.path.join(checkpoint_root, filename)
        epoch = checkpoint_epoch(checkpoint_path)
        if epoch is None:
            continue

        is_global_best = (
            filename.startswith("global_best_epoch_")
            or "_global_best_epoch_" in filename
        )
        if end_epoch is not None and epoch > end_epoch:
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


def _list_matching_conv_weights(state_dict, prefixes):
    matches = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value) or value.ndim != 4:
            continue
        if not key.endswith(".weight"):
            continue
        if any(key.startswith(prefix) for prefix in prefixes):
            matches.append((key, tuple(value.shape)))
    matches.sort(key=lambda item: item[0])
    return matches


def infer_checkpoint_decoder_overrides(checkpoint):
    state_dict = get_checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        return {}

    decoder_prefixes = (
        "decoder.cls_decoder.decoder",
        "decoder.box_decoder.shared",
        "decoder.heatmap_head.head",
        "decoder.regression_head.head",
    )
    class_output_prefixes = (
        "decoder.cls_decoder.decoder",
        "decoder.heatmap_head.head",
    )

    decoder_convs = _list_matching_conv_weights(state_dict, decoder_prefixes)
    class_output_convs = _list_matching_conv_weights(state_dict, class_output_prefixes)

    overrides = {}
    if decoder_convs:
        first_key, first_shape = decoder_convs[0]
        overrides["decoder_hidden_channels"] = int(first_shape[0])
        overrides["feature_channels"] = int(first_shape[1])
        overrides["decoder_channels_key"] = first_key

    class_output_1x1 = [
        (key, shape)
        for key, shape in class_output_convs
        if shape[2:] == (1, 1)
    ]
    if class_output_1x1:
        class_key, class_shape = class_output_1x1[-1]
        overrides["num_classes"] = int(class_shape[0])
        overrides["num_classes_key"] = class_key

    return overrides


def print_checkpoint_override_summary(checkpoint_path, overrides):
    if not overrides:
        return

    details = []
    if "decoder_hidden_channels" in overrides:
        details.append(f"decoder_hidden_channels={overrides['decoder_hidden_channels']}")
    if "feature_channels" in overrides:
        details.append(f"feature_channels={overrides['feature_channels']}")
    if "num_classes" in overrides:
        details.append(f"inferred_num_classes={overrides['num_classes']}")
    print(
        f"Checkpoint overrides for {Path(checkpoint_path).name}: "
        + ", ".join(details)
    )


def format_sequence_label(sequences, prefix):
    normalized = normalize_sequence_list(sequences, name=prefix)
    if normalized in (None, "", ()):
        return None
    values = [str(int(sequence)) for sequence in normalized]
    return f"{prefix}_{','.join(values)}"


def normalize_checkpoint_sequences(sequences, name):
    if sequences in (None, "", ()):
        return None
    return normalize_sequence_list(sequences, name=name)


def infer_source_controlled_from_config(config):
    if not isinstance(config, dict):
        return False
    override_path = normalize_optional_path(
        config.get("gt_object_ignore_override_path")
    )
    if override_path is not None:
        return True
    return bool(config.get("train_control_split_enabled", False))


def learning_rate_name(value):
    if value is None:
        return None
    numeric_value = float(value)
    if numeric_value == 0.0:
        return "lr0"
    text = f"{numeric_value:.8g}"
    if "e" not in text and abs(numeric_value) < 0.01:
        text = f"{numeric_value:.8e}"
    if "e" in text:
        mantissa, exponent = text.split("e", 1)
        mantissa = mantissa.rstrip("0").rstrip(".")
        exponent_sign = "+" if exponent.startswith("+") else "-"
        exponent_digits = exponent.lstrip("+-").lstrip("0") or "0"
        text = f"{mantissa}e{exponent_sign if exponent_sign == '+' else '-'}{exponent_digits}"
    return f"lr{text}"


def extract_checkpoint_source_metadata(checkpoint):
    if not isinstance(checkpoint, dict):
        return {
            "train_sequences": None,
            "val_sequences": None,
            "include_bus_as_target": True,
            "gt_object_ignore_override_path": None,
            "train_control_split_enabled": False,
        }

    config = checkpoint.get("config", {})
    inferred_include_bus_as_target = infer_include_bus_as_target_from_checkpoint_config(
        config
    )
    include_bus_as_target = True
    if inferred_include_bus_as_target is not None:
        include_bus_as_target = bool(inferred_include_bus_as_target)

    return {
        "train_sequences": normalize_checkpoint_sequences(
            config.get("train_sequences"),
            name="checkpoint.config.train_sequences",
        ),
        "val_sequences": normalize_checkpoint_sequences(
            config.get("val_sequences"),
            name="checkpoint.config.val_sequences",
        ),
        "include_bus_as_target": include_bus_as_target,
        "gt_object_ignore_override_path": normalize_optional_path(
            config.get("gt_object_ignore_override_path")
        ),
        "train_control_split_enabled": infer_source_controlled_from_config(config),
    }


def build_model_variant_name(
        model_type,
        overrides,
        include_bus_as_target=True,
        train_sequences=None,
        train_control_split_enabled=False,
        learning_rate=None,
    ):
    parts = [str(model_type)]
    decoder_hidden_channels = overrides.get("decoder_hidden_channels")
    feature_channels = overrides.get("feature_channels")
    if decoder_hidden_channels is not None:
        parts.append(str(int(decoder_hidden_channels)))
    if feature_channels is not None:
        parts.append(str(int(feature_channels)))
    if not bool(include_bus_as_target):
        parts.append("ig")
    train_label = format_sequence_label(train_sequences, prefix="train")
    if train_label is not None:
        parts.append(train_label)
    if bool(train_control_split_enabled):
        parts.append("controlled")
    learning_rate_label = learning_rate_name(learning_rate)
    if learning_rate_label is not None:
        parts.append(learning_rate_label)
    return "_".join(parts)


def infer_model_variant_name(
        model_type,
        checkpoint_or_state_dict,
        include_bus_as_target=True,
        train_sequences=None,
        train_control_split_enabled=False,
    ):
    overrides = infer_checkpoint_decoder_overrides(checkpoint_or_state_dict)
    checkpoint_config = (
        checkpoint_or_state_dict.get("config", {})
        if isinstance(checkpoint_or_state_dict, dict)
        else {}
    )
    return build_model_variant_name(
        model_type=model_type,
        overrides=overrides,
        include_bus_as_target=include_bus_as_target,
        train_sequences=train_sequences,
        train_control_split_enabled=train_control_split_enabled,
        learning_rate=checkpoint_config.get(
            "lr",
            checkpoint_config.get("learning_rate"),
        ),
    )


def build_model_for_checkpoint(
        model_type,
        device,
        num_classes,
        checkpoint_path,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
        loss_mode="auto",
    ):
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    overrides = infer_checkpoint_decoder_overrides(checkpoint)
    build_kwargs = {
        "model_type": model_type,
        "device": device,
        "num_classes": num_classes,
        "box_coordinate_mode": box_coordinate_mode,
        "loss_mode": loss_mode,
    }
    if "decoder_hidden_channels" in overrides:
        build_kwargs["decoder_hidden_channels"] = overrides["decoder_hidden_channels"]
    if "feature_channels" in overrides:
        build_kwargs["feature_channels"] = overrides["feature_channels"]

    print_checkpoint_override_summary(checkpoint_path, overrides)
    model = build_model(**build_kwargs)
    return model, overrides


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
    if "_model16_swin_radenet_official_marker" in state_dict:
        return "model16"
    if "_model7_cartesian_radenet_marker" in state_dict:
        return "model7"
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
    state_dict = get_checkpoint_state_dict(checkpoint)
    inferred_box_coordinate_mode = (
        BOX_COORDINATE_CARTESIAN
        if (
            "_model7_cartesian_radenet_marker" in state_dict
            or "_model7_cartesian_centerpoint_marker" in state_dict
        )
        else BOX_COORDINATE_POLAR
    )
    inferred_include_bus_as_target = infer_include_bus_as_target_from_checkpoint_config(
        config
    )
    if should_inherit_from_checkpoint("max_detections") and config.get("max_detections") is not None:
        args.max_detections = int(config["max_detections"])
    elif should_inherit_from_checkpoint("max_detections") and config.get("num_boxes") is not None:
        args.max_detections = int(config["num_boxes"])
    if should_inherit_from_checkpoint("train_ratio") and config.get("train_ratio") is not None:
        args.train_ratio = float(config["train_ratio"])
    if (
        should_inherit_from_checkpoint("include_bus_as_target")
        and inferred_include_bus_as_target is not None
    ):
        args.include_bus_as_target = inferred_include_bus_as_target
        if config.get("include_bus_as_target") is None:
            print(
                "Checkpoint config missing include_bus_as_target; "
                f"inferred {args.include_bus_as_target} "
                f"from stored num_classes/class_names for {Path(first_checkpoint_path).name}"
            )
    if should_inherit_from_checkpoint("ignore_class_names") and config.get("ignore_class_names") is not None:
        args.ignore_class_names = tuple(config["ignore_class_names"])
    if (
        should_inherit_from_checkpoint("gt_object_ignore_override_path")
        and config.get("gt_object_ignore_override_path") is not None
    ):
        args.gt_object_ignore_override_path = config["gt_object_ignore_override_path"]
    if (
        should_inherit_from_checkpoint("ignore_object_label_minus_one")
        and config.get("ignore_object_label_minus_one") is not None
    ):
        args.ignore_object_label_minus_one = bool(
            config["ignore_object_label_minus_one"]
        )
    if (
        should_inherit_from_checkpoint("train_control_split_enabled")
        and config.get("train_control_split_enabled") is not None
    ):
        args.train_control_split_enabled = bool(config["train_control_split_enabled"])
    if (
        should_inherit_from_checkpoint("train_control_split_dir")
        and config.get("train_control_split_dir") is not None
    ):
        args.train_control_split_dir = config["train_control_split_dir"]
    if should_inherit_from_checkpoint("ignore_mask_margin") and config.get("ignore_mask_margin") is not None:
        args.ignore_mask_margin = float(config["ignore_mask_margin"])
    if (
        should_inherit_from_checkpoint("ignore_mask_expand_ratio")
        and config.get("ignore_mask_expand_ratio") is not None
    ):
        args.ignore_mask_expand_ratio = float(config["ignore_mask_expand_ratio"])
    if should_inherit_from_checkpoint("loss_mode") and config.get("loss_mode") is not None:
        args.loss_mode = config["loss_mode"]
    elif "loss_mode" not in config:
        if "_model7_cartesian_radenet_marker" in state_dict:
            args.loss_mode = "radenet"
        elif "_model7_cartesian_centerpoint_marker" in state_dict:
            args.loss_mode = "centerpoint"
        elif inferred_box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
            # Older Cartesian model7 checkpoints used the RADE-Net head.
            args.loss_mode = "radenet"
        else:
            args.loss_mode = "auto"
    if (
        should_inherit_from_checkpoint("custom_iou_range_eval_enabled")
        and config.get("custom_iou_range_eval_enabled") is not None
    ):
        args.custom_iou_range_eval_enabled = bool(config["custom_iou_range_eval_enabled"])
    if should_inherit_from_checkpoint("custom_iou_thresholds") and config.get("custom_iou_thresholds") is not None:
        args.custom_iou_thresholds = normalize_float_thresholds(
            config["custom_iou_thresholds"],
            name="checkpoint.config.custom_iou_thresholds",
        )
    if should_inherit_from_checkpoint("coco_style_eval_enabled") and config.get("coco_style_eval_enabled") is not None:
        args.coco_style_eval_enabled = bool(config["coco_style_eval_enabled"])
    if should_inherit_from_checkpoint("nuscenes_style_eval_enabled") and config.get("nuscenes_style_eval_enabled") is not None:
        args.nuscenes_style_eval_enabled = bool(config["nuscenes_style_eval_enabled"])
    if should_inherit_from_checkpoint("split_mode") and config.get("split_mode") is not None:
        args.split_mode = config["split_mode"]
    if should_inherit_from_checkpoint("split_dir") and config.get("split_dir") is not None:
        args.split_dir = config["split_dir"]
    if should_inherit_from_checkpoint("train_sequences") and config.get("train_sequences") is not None:
        args.train_sequences = config["train_sequences"]
    if should_inherit_from_checkpoint("val_sequences") and config.get("val_sequences") is not None:
        args.val_sequences = config["val_sequences"]
    if should_inherit_from_checkpoint("seed") and config.get("seed") is not None:
        args.seed = int(config["seed"])
    if args.box_coordinate_mode in (None, "auto"):
        args.box_coordinate_mode = config.get(
            "box_coordinate_mode",
            inferred_box_coordinate_mode,
        )
    args.box_coordinate_mode = validate_box_coordinate_mode(
        args.box_coordinate_mode
    )
    if (
        should_inherit_from_checkpoint("cartesian_gt_root")
        and config.get("cartesian_gt_root") is not None
    ):
        args.cartesian_gt_root = config["cartesian_gt_root"]
    args.cartesian_gt_root = normalize_optional_path(
        args.cartesian_gt_root
    )
    if (
        args.box_coordinate_mode == BOX_COORDINATE_CARTESIAN
        and args.cartesian_gt_root is None
    ):
        raise ValueError(
            "Cartesian checkpoint evaluation requires cartesian_gt_root "
            "in eval_cfg.py or checkpoint config."
        )
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


def result_main_metric_key(result):
    return result.get(
        "evaluation_main_metric_key",
        result.get("official_main_metric_key", "official_bev_mAP_0.3"),
    )


def result_main_metric_value(result):
    value = result.get("evaluation_main_metric_value")
    if value is None:
        value = result.get("official_main_metric_value", 0.0)
    return float(value)


def attach_evaluation_main_metric(metrics, primary_geometry):
    primary_geometry = validate_box_coordinate_mode(primary_geometry)
    if primary_geometry == BOX_COORDINATE_POLAR:
        primary_key = (
            "polar_bev_mAP_0.3"
            if "polar_bev_mAP_0.3" in metrics
            else "polar_bev_mAP"
        )
    else:
        primary_key = metrics.get(
            "official_main_metric_key",
            "official_bev_mAP_0.3",
        )
    primary_value = float(metrics.get(primary_key, 0.0))
    metrics["evaluation_main_metric_key"] = primary_key
    metrics["evaluation_main_metric_value"] = primary_value
    metrics["mAP"] = primary_value
    return metrics


def sequence_name_for_filename(sequences, empty_name):
    sequences = normalize_sequence_list(sequences, name=empty_name)
    if sequences is None or len(sequences) == 0:
        return empty_name
    return format_sequence_run_name(sequences)


def build_plot_metadata(args, model_variant_name, source_metadata):
    source_train_sequences = source_metadata.get("train_sequences")
    source_val_sequences = source_metadata.get("val_sequences")
    return {
        "model_type": str(model_variant_name or "model_unknown"),
        "base_model_type": str(source_metadata.get("base_model_type", "model_unknown")),
        "model_configuration_name": source_metadata.get("model_configuration_name"),
        "model_configuration": source_metadata.get("model_configuration", {}),
        "split_mode": str(args.split_mode),
        "train_sequences": sequence_name_for_filename(source_train_sequences, "train_unknown"),
        "checkpoint_train_sequences": sequence_name_for_filename(source_train_sequences, "train_unknown"),
        "checkpoint_val_sequences": sequence_name_for_filename(source_val_sequences, "val_unknown"),
        "val_sequences": sequence_name_for_filename(args.val_sequences, "val_unknown"),
        "eval_scope": str(args.eval_scope),
        "eval_coordinate_mode": str(args.eval_coordinate_mode),
        "effective_eval_coordinate_mode": str(args.effective_eval_coordinate_mode),
        "evaluation_primary_geometry": str(args.evaluation_primary_geometry),
        "box_coordinate_mode": str(args.box_coordinate_mode),
        "include_bus_as_target": bool(args.include_bus_as_target),
        "checkpoint_include_bus_as_target": bool(
            source_metadata.get("include_bus_as_target", args.include_bus_as_target)
        ),
        "gt_object_ignore_override_path": source_metadata.get(
            "gt_object_ignore_override_path",
            args.gt_object_ignore_override_path,
        ),
        "train_control_split_enabled": bool(
            source_metadata.get("train_control_split_enabled", False)
        ),
        "end_epoch": None if args.end_epoch is None else int(args.end_epoch),
        "heatmap_score_mode": str(args.heatmap_score_mode),
        "ap_score_thresh": float(args.ap_score_thresh),
        "score_thresh": float(args.score_thresh),
        "official_iou_mode": str(args.official_eval_iou_mode),
        "official_ap03_only": bool(args.official_ap03_only),
        "official_detection_metrics_enabled": bool(args.official_detection_metrics_enabled),
        "group_checkpoint_plot_best_only": bool(args.group_checkpoint_plot_best_only),
        "custom_iou_range_eval_enabled": bool(args.custom_iou_range_eval_enabled),
        "custom_iou_thresholds": [float(value) for value in args.custom_iou_thresholds],
        "coco_style_eval_enabled": bool(args.coco_style_eval_enabled),
        "nuscenes_style_eval_enabled": bool(args.nuscenes_style_eval_enabled),
        "official_eval_enabled": bool(args.official_eval_enabled),
        "official_geometry_source": str(args.official_geometry_source),
        "polar_eval_enabled": bool(args.polar_eval_enabled),
        "polar_geometry_source": str(args.polar_geometry_source),
        "polar_iou_thresholds": [float(value) for value in args.polar_iou_thresholds],
    }


def plot_output_requested(plot_output):
    if plot_output is None:
        return False
    if isinstance(plot_output, str):
        normalized = plot_output.strip()
        if normalized == "":
            return False
        if normalized.lower() in {"no", "none", "false", "0", "null"}:
            return False
        return True
    return bool(plot_output)


def plot_output_is_auto(plot_output):
    if not isinstance(plot_output, str):
        return bool(plot_output)
    normalized = plot_output.strip().lower()
    return normalized in {"yes", "true", "1", "auto"}


def plot_checkpoint_name(checkpoint_path):
    return os.path.splitext(os.path.basename(checkpoint_path))[0]


def checkpoint_epoch_number(checkpoint_path):
    checkpoint_name = plot_checkpoint_name(checkpoint_path)
    match = re.search(r"_epoch_(\d+)_", checkpoint_name)
    if match is None:
        return None
    return int(match.group(1))


def append_filename_tag(output_path, tag):
    if tag in (None, ""):
        return output_path
    base, suffix = os.path.splitext(output_path)
    return f"{base}__{tag}{suffix}"


def resolve_plot_output_path(args, checkpoint_paths, model_type, checkpoint_path=None, selection_tag=None):
    plot_output = args.plot_output
    if not plot_output_requested(plot_output):
        return None

    model_tag = str(model_type or "model_unknown")
    val_tag = format_sequence_tag(args.val_sequences, "val_seq")
    if checkpoint_path is None:
        checkpoint_root = args.checkpoint_root
        if os.path.isfile(checkpoint_root):
            checkpoint_path = checkpoint_root
        else:
            _, checkpoint_path = checkpoint_paths[0]
    epoch_number = checkpoint_epoch_number(checkpoint_path)

    def auto_plot_output_path():
        # Keep the model identity in the directory, matching evaluation_plots.
        # The filename only identifies the validation set and selected epoch.
        output_dir = (
            resolve_output_base_dir("evaluation_plots/png_photos")
            / sanitize_filename(model_tag)
        )
        stem_parts = [
            sanitize_filename(val_tag),
        ]
        if selection_tag:
            stem_parts.append(sanitize_filename(selection_tag))
        if epoch_number is not None:
            stem_parts.append(f"e{int(epoch_number):03d}")
        if args.official_ap03_only:
            stem_parts.append("ap03only")
        stem = "__".join(part for part in stem_parts if part not in {"", None})
        return str(
            next_available_output_path(
                output_dir=output_dir,
                stem=stem,
                suffix=".png",
            )
        )

    if isinstance(plot_output, str):
        normalized = plot_output.strip()
        if plot_output_is_auto(normalized):
            return auto_plot_output_path()
        return append_filename_tag(normalized, selection_tag)

    if bool(plot_output):
        return auto_plot_output_path()

    return None


def metric_label(metric_key):
    if metric_key.startswith("official_bev_mAP_"):
        return f"BEV mAP@{metric_key.rsplit('_', 1)[-1]}"
    if metric_key.startswith("official_3d_mAP_"):
        return f"3D mAP@{metric_key.rsplit('_', 1)[-1]}"
    return metric_key


def metric_class_names_for_result(result):
    class_names = result.get("metric_class_names")
    if isinstance(class_names, (list, tuple)) and len(class_names) > 0:
        return [str(class_name) for class_name in class_names]

    for key in (
        "official_per_class",
        "official_detection_per_class",
        "custom_iou_per_class",
        "custom_iou_detection_per_class",
        "coco_per_class",
        "nuscenes_per_class",
        "polar_per_class",
    ):
        values = result.get(key)
        if isinstance(values, dict) and len(values) > 0:
            return [str(class_name) for class_name in values.keys()]
    return []


def merged_metric_class_names(results):
    merged = []
    seen = set()
    for result in results:
        for class_name in metric_class_names_for_result(result):
            if class_name in seen:
                continue
            merged.append(class_name)
            seen.add(class_name)
    return merged


def class_display_name_map_for_results(results):
    merged = {}
    for result in results:
        values = result.get("class_display_name_map", {})
        if isinstance(values, dict):
            for class_name, display_name in values.items():
                merged[str(class_name)] = str(display_name)
    return merged


def display_class_name(class_name, display_name_map=None):
    if display_name_map is not None and class_name in display_name_map:
        return display_name_map[class_name]
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
    class_names = merged_metric_class_names(results)
    display_name_map = class_display_name_map_for_results(results)
    columns = ["Object"]
    for iou_suffix in iou_suffixes:
        columns.extend([f"BEV AP@{iou_suffix}", f"3D AP@{iou_suffix}"])
    include_polar = any("polar_bev_mAP_0.3" in result for result in results)
    if include_polar:
        columns.extend(["Polar AP@0.3", "Polar AP@0.5"])
    include_detection_metrics = any(
        isinstance(result.get("official_detection_precision"), (int, float))
        for result in results
    )
    if include_detection_metrics:
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
        if include_polar:
            row.extend([
                official_ap_text(result.get("polar_bev_mAP_0.3")),
                official_ap_text(result.get("polar_bev_mAP_0.5")),
            ])
        if include_detection_metrics:
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
        for class_name in class_names:
            class_stats = per_class.get(class_name, {})
            row = [result_row_label(result, display_class_name(class_name, display_name_map), multi_epoch)]
            for iou_suffix in iou_suffixes:
                row.extend([
                    official_ap_text(result.get(f"official_{class_name}_bev_AP_{iou_suffix}")),
                    official_ap_text(result.get(f"official_{class_name}_3d_AP_{iou_suffix}")),
                ])
            if include_polar:
                row.extend([
                    official_ap_text(result.get(f"polar_{class_name}_bev_AP_0.3")),
                    official_ap_text(result.get(f"polar_{class_name}_bev_AP_0.5")),
                ])
            if include_detection_metrics:
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
        "title": (
            "Official K-Radar"
            if len(iou_suffixes) > 0
            else "Polar BEV"
        ),
        "columns": columns,
        "rows": rows,
        "row_meta": row_meta,
    }


def build_coco_plot_section(results, multi_epoch):
    if not any("coco_bev_mAP" in result for result in results):
        return None
    class_names = merged_metric_class_names(results)
    display_name_map = class_display_name_map_for_results(results)

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

        for class_name in class_names:
            rows.append([
                result_row_label(result, display_class_name(class_name, display_name_map), multi_epoch),
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
    class_names = merged_metric_class_names(results)
    display_name_map = class_display_name_map_for_results(results)

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

        for class_name in class_names:
            row = [
                result_row_label(result, display_class_name(class_name, display_name_map), multi_epoch),
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
    class_names = merged_metric_class_names(results)
    display_name_map = class_display_name_map_for_results(results)

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

        for class_name in class_names:
            rows.append([
                result_row_label(result, display_class_name(class_name, display_name_map), multi_epoch),
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

    best_result = max(
        results,
        key=result_main_metric_value,
    )
    best_metric_key = result_main_metric_key(best_result)
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

    summary_lines = [
        (
            f"Best epoch: {best_result['epoch']} | "
            f"{best_metric_key} = "
            f"{official_ap_text(result_main_metric_value(best_result))}"
        )
    ]
    if plot_metadata is not None:
        shown_ap_iou_text = ", ".join(iou_suffixes)
        summary_lines.append(
            (
                f"Model: {plot_metadata.get('model_type', '-')} | "
                f"Split: {plot_metadata.get('split_mode', '-')} | "
                f"Scope: {plot_metadata.get('eval_scope', '-')}"
            )
        )
        summary_lines.append(
            (
                f"Train: {plot_metadata.get('train_sequences', '-')} | "
                f"Val: {plot_metadata.get('val_sequences', '-')}"
            )
        )
        summary_lines.append(
            (
                f"Frames: {best_result.get('evaluation_num_eval_frames', best_result.get('official_num_eval_frames', 0))} | "
                f"Backend: {best_result.get('official_iou_backend_used', '-')} | "
                f"Shown AP IoU: {shown_ap_iou_text or 'polar'}"
            )
        )
        ap_score_thresh = plot_metadata.get("ap_score_thresh")
        if ap_score_thresh is not None:
            summary_lines.append(f"AP score threshold: {ap_score_thresh:.2f}")
        if plot_metadata.get("official_detection_metrics_enabled", False):
            score_thresh = plot_metadata.get(
                "score_thresh",
                plot_metadata.get("detection_score_thresh"),
            )
            if score_thresh is not None:
                summary_lines.append(
                    (
                        f"Det score threshold: {score_thresh:.2f} | "
                        f"Det IoU: {best_result.get('official_detection_iou_threshold', 0.0):.2f}"
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
                f"3D mAP={metric_text(best_result.get('custom_iou_3d_mAP'))}"
            )
        )
        summary_lines.append(
            (
                "Custom IoU detection: "
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
                f"AP@4.0m={metric_text(best_result.get('nuscenes_AP_4.0m'))}"
            )
        )
        summary_lines.append(
            (
                "nuScenes-style errors: "
                f"mATE={metric_text(best_result.get('nuscenes_mATE'))} | "
                f"mASE={metric_text(best_result.get('nuscenes_mASE'))} | "
                f"mAOE={metric_text(best_result.get('nuscenes_mAOE'))}"
            )
        )

    max_columns = max(len(section["columns"]) for section in sections)
    fig_width = max(13.0, 1.15 * max_columns)
    summary_height = 0.62 + 0.22 * len(summary_lines)
    section_heights = [
        0.42 + 0.36 * (len(section["rows"]) + 1)
        for section in sections
    ]
    fig_height = max(
        7.0,
        summary_height + sum(section_heights) + 0.22 * (len(sections) - 1) + 0.35,
    )
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        nrows=len(sections) + 1,
        ncols=1,
        height_ratios=[summary_height, *section_heights],
        left=0.035,
        right=0.965,
        top=0.98,
        bottom=0.025,
        hspace=0.22,
    )

    summary_ax = fig.add_subplot(grid[0])
    summary_ax.axis("off")
    summary_ax.text(
        0.5,
        0.98,
        "Evaluation Results",
        transform=summary_ax.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )
    summary_ax.text(
        0.5,
        0.78,
        "\n".join(summary_lines),
        transform=summary_ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.0,
        color="#374151",
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#f8fafc",
            "edgecolor": "#d1d5db",
            "linewidth": 0.8,
        },
    )

    for section_index, (section, section_height) in enumerate(
        zip(sections, section_heights),
        start=1,
    ):
        section_ax = fig.add_subplot(grid[section_index])
        section_ax.axis("off")
        section_ax.text(
            0.0,
            0.98,
            section["title"],
            transform=section_ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#111827",
        )
        title_fraction = min(0.24, 0.34 / section_height)
        table_top = 1.0 - title_fraction
        table = section_ax.table(
            cellText=section["rows"],
            colLabels=section["columns"],
            bbox=[0.0, 0.0, 1.0, table_top],
            cellLoc="center",
            colLoc="center",
        )
        style_metric_table(
            table,
            row_meta=section["row_meta"],
            best_epoch=int(best_result["epoch"]),
            font_size=9.5,
        )

    fig.savefig(plot_output_path, dpi=200, bbox_inches="tight", pad_inches=0.12)
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


def available_official_iou_suffixes(result):
    suffixes = []
    for suffix in ("0.3", "0.5"):
        bev_key = f"official_bev_mAP_{suffix}"
        d3_key = f"official_3d_mAP_{suffix}"
        if bev_key in result or d3_key in result:
            suffixes.append(suffix)
    return suffixes


def collect_method_summary(result):
    official_summary = {
        "main_metric_key": result.get("official_main_metric_key"),
        "main_metric_value": official_ap_value(result.get("official_main_metric_value")),
    }
    for suffix in available_official_iou_suffixes(result):
        official_summary[f"bev_mAP_{suffix}"] = official_ap_value(
            result.get(f"official_bev_mAP_{suffix}")
        )
        official_summary[f"3d_mAP_{suffix}"] = official_ap_value(
            result.get(f"official_3d_mAP_{suffix}")
        )

    summary = {
        "evaluation": {
            "coordinate_mode": result.get("effective_eval_coordinate_mode"),
            "primary_geometry": result.get("evaluation_primary_geometry"),
            "main_metric_key": result_main_metric_key(result),
            "main_metric_value": official_ap_value(
                result_main_metric_value(result)
            ),
        },
        "official": official_summary,
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
    if "polar_bev_mAP" in result:
        summary["polar_bev"] = {
            "iou_thresholds": result.get("polar_iou_thresholds"),
            "mAP": result.get("polar_bev_mAP"),
            "AP_0.3": result.get("polar_bev_mAP_0.3"),
            "AP_0.5": result.get("polar_bev_mAP_0.5"),
            "num_gt": result.get("polar_bev_num_gt"),
        }
    return summary


def build_yaml_export(results, plot_metadata=None):
    if len(results) == 0:
        raise ValueError("Cannot export YAML because evaluation results are empty.")

    best_result = max(
        results,
        key=result_main_metric_value,
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
    main_key = result_main_metric_key(metrics)
    parts = [
        f"epoch={epoch}",
        f"{main_key}={official_ap_text(result_main_metric_value(metrics))}",
        f"ap_score_thr={metric_text(metrics.get('ap_score_thresh'))}",
    ]
    for suffix in available_official_iou_suffixes(metrics):
        parts.append(
            f"bev@{suffix}={official_ap_text(metrics.get(f'official_bev_mAP_{suffix}'))}"
        )
    for suffix in available_official_iou_suffixes(metrics):
        parts.append(
            f"3d@{suffix}={official_ap_text(metrics.get(f'official_3d_mAP_{suffix}'))}"
        )
    if "official_detection_precision" in metrics:
        parts.extend([
            f"score_thr={metric_text(metrics.get('official_detection_score_threshold'))}",
            f"p={metric_text(metrics.get('official_detection_precision'))}",
            f"r={metric_text(metrics.get('official_detection_recall'))}",
            f"f1={metric_text(metrics.get('official_detection_f1'))}",
            f"tp={metrics.get('official_detection_tp', 0)}",
            f"fp={metrics.get('official_detection_fp', 0)}",
            f"fn={metrics.get('official_detection_fn', 0)}",
        ])
    if "eval_ignore_suppressed_predictions" in metrics:
        parts.append(
            f"ign_sup={int(metrics.get('eval_ignore_suppressed_predictions', 0))}"
        )
    parts.append(
        f"frames={metrics.get('evaluation_num_eval_frames', metrics.get('official_num_eval_frames', 0))}"
    )
    print(" ".join(parts))
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
            f"ap_score_thr={metric_text(metrics.get('ap_score_thresh'))} "
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
    if "polar_bev_mAP" in metrics:
        threshold_text = ", ".join(
            f"{float(value):.2f}" for value in metrics.get("polar_iou_thresholds", [])
        )
        print(
            f"  polar-bev-ap "
            f"iou=[{threshold_text}] "
            f"mAP={official_ap_text(metrics.get('polar_bev_mAP'))} "
            f"bev@0.3={official_ap_text(metrics.get('polar_bev_mAP_0.3'))} "
            f"bev@0.5={official_ap_text(metrics.get('polar_bev_mAP_0.5'))} "
            f"gt={int(metrics.get('polar_bev_num_gt', 0))}"
        )


def _plain_text_table(headers, rows):
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in string_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(row_values):
        return "| " + " | ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(row_values)
        ) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [separator, format_row(headers), separator]
    for row in string_rows:
        lines.append(format_row(row))
    lines.append(separator)
    return "\n".join(lines)


def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9.,_-]+", "_", str(text)).strip("_")


def format_sequence_tag(sequences, prefix):
    if sequences in (None, "", ()):
        return f"{prefix}_unknown"
    values = [str(int(sequence)) for sequence in sequences]
    return f"{prefix}_{'_'.join(values)}"


def resolve_output_base_dir(base_dir):
    output_dir = Path(base_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    return output_dir


def create_evaluation_tensorboard_writer(args, model_variant_name, source_metadata):
    base_dir = resolve_output_base_dir(args.evaluation_tensorboard_log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_tag = sanitize_filename(model_variant_name or "model_unknown")
    val_tag = sanitize_filename(format_sequence_tag(args.val_sequences, "val_seq"))
    log_dir = base_dir / "evaluation" / f"{timestamp}__{model_tag}__{val_tag}"
    writer = SummaryWriter(log_dir=str(log_dir))
    config = {
        "model_variant": model_variant_name,
        "checkpoint_root": str(args.checkpoint_root),
        "train_sequences": source_metadata.get("train_sequences"),
        "checkpoint_val_sequences": source_metadata.get("val_sequences"),
        "val_sequences": args.val_sequences,
        "include_bus_as_target": bool(args.include_bus_as_target),
        "eval_scope": args.eval_scope,
        "eval_coordinate_mode": args.eval_coordinate_mode,
        "effective_eval_coordinate_mode": args.effective_eval_coordinate_mode,
        "box_coordinate_mode": args.box_coordinate_mode,
        "evaluation_primary_geometry": args.evaluation_primary_geometry,
        "official_eval_version": args.official_eval_version,
        "official_eval_iou_mode": args.official_eval_iou_mode,
        "official_eval_enabled": bool(args.official_eval_enabled),
        "official_geometry_source": args.official_geometry_source,
        "ap_score_thresh": float(args.ap_score_thresh),
        "score_thresh": float(args.score_thresh),
        "custom_iou_range_eval_enabled": bool(args.custom_iou_range_eval_enabled),
        "custom_iou_thresholds": [float(value) for value in args.custom_iou_thresholds],
        "group_checkpoint_plot_best_only": bool(args.group_checkpoint_plot_best_only),
        "polar_eval_enabled": bool(args.polar_eval_enabled),
        "polar_geometry_source": args.polar_geometry_source,
        "polar_iou_thresholds": [float(value) for value in args.polar_iou_thresholds],
    }
    writer.add_text(
        "run/config",
        json.dumps(config, indent=2, default=str),
        global_step=0,
    )
    writer.flush()
    return writer, log_dir


def write_evaluation_tensorboard_result(writer, result, namespace="evaluation"):
    if writer is None:
        return

    metric_prefixes = (
        "evaluation_",
        "official_",
        "custom_iou_",
        "coco_",
        "nuscenes_",
        "polar_",
        "val_",
    )
    epoch = int(result["epoch"])
    for key, value in result.items():
        if not isinstance(value, Real) or isinstance(value, bool):
            continue
        if not key.startswith(metric_prefixes):
            continue
        writer.add_scalar(f"{namespace}/metrics/{key}", float(value), epoch)
    writer.flush()


def next_available_output_path(output_dir, stem, suffix):
    candidate = output_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = output_dir / f"{stem}__{index:02d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def format_custom_iou_range_text(thresholds):
    values = [float(value) for value in thresholds]
    if len(values) == 0:
        return "iou=[]"
    if len(values) == 1:
        return f"iou={format_custom_iou_suffix(values[0])}"

    step = values[1] - values[0]
    is_uniform = all(
        abs((values[index] - values[index - 1]) - step) <= 1e-6
        for index in range(1, len(values))
    )
    if is_uniform and step > 0.0:
        return (
            f"iou={format_custom_iou_suffix(values[0])}"
            f" -> {format_custom_iou_suffix(values[-1])}"
            f" (step={format_custom_iou_suffix(step)})"
        )

    threshold_text = ", ".join(
        format_custom_iou_suffix(value)
        for value in values
    )
    return f"iou=[{threshold_text}]"


def format_eval_table(rows):
    if len(rows) == 0:
        return ""

    columns = [("epoch", "epoch", 5, "int")]
    if any("val_loss" in row for row in rows):
        columns.extend(
            [
                ("val_loss", "val_loss", 10, "float"),
                ("val_box", "val_box_loss", 10, "float"),
                ("val_cls", "val_cls_loss", 10, "float"),
            ]
        )
        if any("val_heatmap_loss" in row for row in rows):
            columns.append(("val_hm", "val_heatmap_loss", 10, "float"))
        if any("val_quality_loss" in row for row in rows):
            columns.append(("val_q", "val_quality_loss", 10, "float"))
        if any("val_obj_loss" in row for row in rows):
            columns.append(("val_obj", "val_obj_loss", 10, "float"))
        if any("val_l1_loss" in row for row in rows):
            columns.append(("val_l1", "val_l1_loss", 10, "float"))
        if any("val_gwd_loss" in row for row in rows):
            columns.append(("val_gwd", "val_gwd_loss", 10, "float"))
    if any("official_bev_mAP_0.3" in row for row in rows):
        columns.extend([
            ("bev@0.3", "official_bev_mAP_0.3", 9, "float"),
            ("bev@0.5", "official_bev_mAP_0.5", 9, "float"),
            ("3d@0.3", "official_3d_mAP_0.3", 9, "float"),
            ("3d@0.5", "official_3d_mAP_0.5", 9, "float"),
            ("p", "official_detection_precision", 8, "float"),
            ("r", "official_detection_recall", 8, "float"),
            ("f1", "official_detection_f1", 8, "float"),
        ])
    if any("polar_bev_mAP_0.3" in row for row in rows):
        columns.extend(
            [
                ("pbev@0.3", "polar_bev_mAP_0.3", 10, "float"),
                ("pbev@0.5", "polar_bev_mAP_0.5", 10, "float"),
            ]
        )
    if any("custom_iou_bev_mAP" in row for row in rows):
        columns.extend(
            [
                ("c_bev_mAP", "custom_iou_bev_mAP", 11, "float"),
                ("c_3d_mAP", "custom_iou_3d_mAP", 10, "float"),
            ]
        )
    if any("eval_ignore_suppressed_predictions" in row for row in rows):
        columns.append(("ign_sup", "eval_ignore_suppressed_predictions", 8, "int"))

    header = " ".join(f"{label:>{width}}" for label, _, width, _ in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        row_text = []
        for _, key, width, kind in columns:
            if kind == "int":
                row_text.append(f"{int(row.get(key, 0)):>{width}d}")
            else:
                row_text.append(f"{float(row.get(key, 0.0)):>{width}.4f}")
        lines.append(" ".join(row_text))
    return "\n".join(lines)


def format_best_epoch_summary(rows):
    summary_specs = (
        ("official_bev_mAP_0.3", "bev@0.3"),
        ("official_3d_mAP_0.3", "3d@0.3"),
    )
    lines = []
    thresholds = next(
        (
            row.get("custom_iou_thresholds")
            for row in rows
            if isinstance(row.get("custom_iou_thresholds"), (list, tuple))
            and len(row.get("custom_iou_thresholds")) > 0
        ),
        None,
    )
    if thresholds is not None:
        lines.append(f"custom_iou_range: {format_custom_iou_range_text(thresholds)}")
        lines.append("custom_iou_range_metrics: c_bev_mAP=BEV mAP, c_3d_mAP=3D mAP")
        summary_specs = summary_specs + (
            ("custom_iou_bev_mAP", "custom_bev_mAP"),
            ("custom_iou_3d_mAP", "custom_3d_mAP"),
        )
    if any("polar_bev_mAP" in row for row in rows):
        summary_specs = summary_specs + (
            ("polar_bev_mAP_0.3", "pbev@0.3"),
            ("polar_bev_mAP_0.5", "pbev@0.5"),
            ("polar_bev_mAP", "polar_bev_mAP"),
        )
    for metric_key, metric_label in summary_specs:
        best_row = select_best_result_by_metric(rows, metric_key)
        if best_row is None:
            continue
        lines.append(
            f"best_epoch_{metric_label}: "
            f"epoch {int(best_row['epoch'])} "
            f"({metric_label}={float(best_row.get(metric_key, 0.0)):.4f})"
        )
    return lines


def default_eval_table_txt_path(
        model_variant_name,
        val_sequences=None,
        base_dir="evaluation_plots",
    ):
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = resolve_output_base_dir(base_dir) / sanitize_filename(model_variant_name)
    filename_parts = [timestamp]
    if val_sequences is not None:
        filename_parts.append(format_sequence_tag(val_sequences, "val_seq"))
    return next_available_output_path(
        output_dir=output_dir,
        stem="__".join(filename_parts),
        suffix=".txt",
    )


def save_eval_table_txt(rows, output_path, metadata=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []
    if metadata:
        for key, value in metadata.items():
            sections.append(f"{key}: {value}")
        sections.append("")

    summary_lines = format_best_epoch_summary(rows)
    if summary_lines:
        sections.extend(summary_lines)
        sections.append("")

    table_text = format_eval_table(rows)
    if table_text != "":
        sections.append(table_text)

    output_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return output_path


def init_split_bbox_count_summary():
    return {
        "frames": 0,
        "bbox_total": 0,
        "bbox_by_class": {
            class_name: 0
            for class_name in SPLIT_BBOX_COUNT_CLASS_NAMES
        },
    }


def iter_subset_global_indices(dataset_subset):
    if hasattr(dataset_subset, "indices") and hasattr(dataset_subset, "dataset"):
        return [int(index) for index in dataset_subset.indices], dataset_subset.dataset
    return list(range(len(dataset_subset))), dataset_subset


def compute_subset_bbox_count_summary(dataset_subset):
    subset_indices, base_dataset = iter_subset_global_indices(dataset_subset)
    if not hasattr(base_dataset, "_resolve_index") or not hasattr(
        base_dataset,
        "sequence_datasets",
    ):
        raise TypeError(
            "Expected KRadarMultiSequenceGTDetectionDataset or its Subset for bbox stats."
        )

    summary = init_split_bbox_count_summary()
    summary["frames"] = len(subset_indices)

    for global_index in subset_indices:
        dataset_idx, sample_idx = base_dataset._resolve_index(int(global_index))
        sequence_dataset = base_dataset.sequence_datasets[dataset_idx]
        all_objects = sequence_dataset.gt_by_file_idx.get(int(sample_idx), [])
        for obj in all_objects:
            class_name = str(obj.get("cls", ""))
            if class_name not in summary["bbox_by_class"]:
                continue
            if not sequence_dataset._object_center_in_scope(obj):
                continue
            summary["bbox_by_class"][class_name] += 1
            summary["bbox_total"] += 1

    return summary


def build_split_statistics_metadata(train_dataset, test_dataset):
    train_summary = compute_subset_bbox_count_summary(train_dataset)
    test_summary = compute_subset_bbox_count_summary(test_dataset)

    total_frames = train_summary["frames"] + test_summary["frames"]
    total_bboxes = train_summary["bbox_total"] + test_summary["bbox_total"]

    return {
        "train_frames": train_summary["frames"],
        "test_frames": test_summary["frames"],
        "train_ratio_frames": (
            float(train_summary["frames"] / total_frames)
            if total_frames > 0
            else 0.0
        ),
        "train_bbox_total": train_summary["bbox_total"],
        "train_bbox_sedan": train_summary["bbox_by_class"]["Sedan"],
        "train_bbox_bus": train_summary["bbox_by_class"]["Bus or Truck"],
        "test_bbox_total": test_summary["bbox_total"],
        "test_bbox_sedan": test_summary["bbox_by_class"]["Sedan"],
        "test_bbox_bus": test_summary["bbox_by_class"]["Bus or Truck"],
        "train_ratio_bboxes": (
            float(train_summary["bbox_total"] / total_bboxes)
            if total_bboxes > 0
            else 0.0
        ),
    }


def print_epoch_ap03_table(results):
    if len(results) == 0:
        return

    class_names = merged_metric_class_names(results)
    polar_primary = any(
        result.get("evaluation_primary_geometry") == BOX_COORDINATE_POLAR
        for result in results
    )
    if polar_primary:
        headers = ["epoch", "pbev@0.3", "pbev@0.5"]
        for class_name in class_names:
            headers.extend([
                f"{class_name}_pbev@0.3",
                f"{class_name}_pbev@0.5",
            ])
        rows = []
        for result in sorted(results, key=lambda item: int(item["epoch"])):
            row = [
                str(int(result["epoch"])),
                official_ap_text(result.get("polar_bev_mAP_0.3")),
                official_ap_text(result.get("polar_bev_mAP_0.5")),
            ]
            for class_name in class_names:
                row.extend([
                    official_ap_text(
                        result.get(f"polar_{class_name}_bev_AP_0.3")
                    ),
                    official_ap_text(
                        result.get(f"polar_{class_name}_bev_AP_0.5")
                    ),
                ])
            rows.append(row)
        print("Epoch-wise direct Polar BEV AP summary:")
        print(_plain_text_table(headers, rows))
        return

    headers = ["epoch", "bev@0.3", "3d@0.3"]
    for class_name in class_names:
        headers.extend([f"{class_name}_bev@0.3", f"{class_name}_3d@0.3"])
    rows = []
    for result in sorted(results, key=lambda item: int(item["epoch"])):
        row = [
            str(int(result["epoch"])),
            official_ap_text(result.get("official_bev_mAP_0.3")),
            official_ap_text(result.get("official_3d_mAP_0.3")),
        ]
        for class_name in class_names:
            row.extend([
                official_ap_text(result.get(f"official_{class_name}_bev_AP_0.3")),
                official_ap_text(result.get(f"official_{class_name}_3d_AP_0.3")),
            ])
        rows.append(row)

    print("Epoch-wise official AP@0.3 summary:")
    print(_plain_text_table(headers, rows))


def select_best_result_by_metric(results, metric_key):
    best_result = None
    best_value = float("-inf")
    best_epoch = None

    for result in results:
        if metric_key not in result or "epoch" not in result:
            continue
        value = float(result.get(metric_key, 0.0))
        epoch = int(result.get("epoch", 0))
        if (
            best_result is None
            or value > best_value
            or (value == best_value and epoch < best_epoch)
        ):
            best_result = result
            best_value = value
            best_epoch = epoch
    return best_result


def selection_iou_mode_for_group_plot(args):
    if args.official_eval_iou_mode in {"easy", "all"}:
        return args.official_eval_iou_mode
    return "easy"


def group_checkpoint_plot_best_only_active(args, checkpoint_paths):
    return (
        bool(args.group_checkpoint_plot_best_only)
        and plot_output_requested(args.plot_output)
        and os.path.isdir(args.checkpoint_root)
        and len(checkpoint_paths) > 1
    )


def evaluate_checkpoint_result(
        model,
        checkpoint_path,
        epoch,
        validation_loader,
        device,
        model_type,
        args,
        official_eval_iou_mode=None,
        official_detection_metrics_enabled=None,
        custom_iou_range_eval_enabled=None,
        coco_style_eval_enabled=None,
        nuscenes_style_eval_enabled=None,
        loss_eval_enabled=None,
        polar_eval_enabled=None,
    ):
    load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        include_bus_as_target=args.include_bus_as_target,
    )

    metrics = {}
    effective_loss_eval_enabled = (
        args.loss_eval_enabled
        if loss_eval_enabled is None
        else bool(loss_eval_enabled)
    )
    if effective_loss_eval_enabled:
        loss_metrics = validate_loss(
            model=model,
            dataloader=validation_loader,
            device=device,
            heatmap_radius=args.heatmap_radius,
            centerpoint_gwd_loss_weight=args.centerpoint_gwd_loss_weight,
            quality_loss_weight=args.quality_loss_weight,
            ignore_mask_margin=args.ignore_mask_margin,
            ignore_mask_expand_ratio=args.ignore_mask_expand_ratio,
            loss_mode=resolve_loss_mode(
                model_type,
                box_coordinate_mode=args.box_coordinate_mode,
            ),
            num_classes=args.num_classes,
            box_coordinate_mode=args.box_coordinate_mode,
        )
        metrics.update(loss_metrics)

    eval_metrics = evaluate_checkpoint_with_kradar_revised(
        model=model,
        dataloader=validation_loader,
        device=device,
        num_classes=args.num_classes,
        official_class_name_map=args.official_class_name_map,
        prepare_model_inputs=prepare_model_inputs,
        max_detections=args.max_detections,
        heatmap_nms_kernel=args.heatmap_nms_kernel,
        heatmap_score_mode=args.heatmap_score_mode,
        yolox_nms_iou=args.yolox_nms_iou,
        scope_mode=args.eval_scope,
        official_eval_enabled=args.official_eval_enabled,
        official_eval_version=args.official_eval_version,
        official_eval_iou_backend=args.official_eval_iou_backend,
        official_eval_iou_mode=(
            args.official_eval_iou_mode
            if official_eval_iou_mode is None
            else official_eval_iou_mode
        ),
        official_detection_metrics_enabled=(
            args.official_detection_metrics_enabled
            if official_detection_metrics_enabled is None
            else official_detection_metrics_enabled
        ),
        custom_iou_range_eval_enabled=(
            args.custom_iou_range_eval_enabled
            if custom_iou_range_eval_enabled is None
            else custom_iou_range_eval_enabled
        ),
        custom_iou_thresholds=args.custom_iou_thresholds,
        coco_style_eval_enabled=(
            args.coco_style_eval_enabled
            if coco_style_eval_enabled is None
            else coco_style_eval_enabled
        ),
        nuscenes_style_eval_enabled=(
            args.nuscenes_style_eval_enabled
            if nuscenes_style_eval_enabled is None
            else nuscenes_style_eval_enabled
        ),
        polar_eval_enabled=(
            args.polar_eval_enabled
            if polar_eval_enabled is None
            else polar_eval_enabled
        ),
        polar_iou_thresholds=args.polar_iou_thresholds,
        ap_score_thresh=args.ap_score_thresh,
        detection_score_thresh=args.detection_score_thresh,
        eval_ignore_suppress_enabled=args.eval_ignore_suppress_enabled,
        eval_ignore_expand_ratio=args.eval_ignore_expand_ratio,
        eval_ignore_suppress_margin=args.eval_ignore_suppress_margin,
        box_coordinate_mode=args.box_coordinate_mode,
    )
    metrics.update(eval_metrics)
    attach_evaluation_main_metric(
        metrics,
        primary_geometry=args.evaluation_primary_geometry,
    )
    return {
        "epoch": int(epoch),
        "checkpoint_path": checkpoint_path,
        "eval_coordinate_mode": args.eval_coordinate_mode,
        "effective_eval_coordinate_mode": args.effective_eval_coordinate_mode,
        "box_coordinate_mode": args.box_coordinate_mode,
        "evaluation_primary_geometry": args.evaluation_primary_geometry,
        "official_geometry_source": args.official_geometry_source,
        "polar_geometry_source": args.polar_geometry_source,
        "ap_score_thresh": float(args.ap_score_thresh),
        "metric_class_names": list(args.metric_class_names),
        "class_display_name_map": dict(args.class_display_name_map),
        **metrics,
    }


def group_plot_selection_specs(args):
    if args.evaluation_primary_geometry == BOX_COORDINATE_POLAR:
        return [("polar_bev_mAP_0.3", "best_pbev03")]
    return [
        ("official_bev_mAP_0.3", "best_bev03"),
        ("official_3d_mAP_0.3", "best_3d03"),
    ]


def build_group_plot_selection_entries(results, args):
    selection_specs = group_plot_selection_specs(args)
    selected_by_path = {}
    for metric_key, selection_tag in selection_specs:
        best_result = select_best_result_by_metric(results, metric_key)
        if best_result is None:
            continue
        checkpoint_path = str(best_result.get("checkpoint_path", ""))
        entry = selected_by_path.setdefault(
            checkpoint_path,
            {
                "result": best_result,
                "selection_tags": [],
            },
        )
        if selection_tag not in entry["selection_tags"]:
            entry["selection_tags"].append(selection_tag)
    return list(selected_by_path.values())


def save_group_best_only_plot_exports(
        selected_entries,
        checkpoint_paths,
        model_type,
        args,
        plot_metadata,
    ):
    for entry in selected_entries:
        result = entry["result"]
        selection_tags = list(entry["selection_tags"])
        selection_tag = "_".join(selection_tags)
        export_metadata = dict(plot_metadata)
        export_metadata.update({
            "group_checkpoint_plot_best_only": True,
            "group_checkpoint_plot_selection": selection_tags,
            "group_checkpoint_plot_source_num_checkpoints": len(checkpoint_paths),
            "group_checkpoint_plot_source_checkpoint_root": str(args.checkpoint_root),
        })
        plot_output_path = resolve_plot_output_path(
            args=args,
            checkpoint_paths=checkpoint_paths,
            model_type=model_type,
            checkpoint_path=result["checkpoint_path"],
            selection_tag=selection_tag,
        )
        if plot_output_path is None:
            continue
        output_dir = os.path.dirname(plot_output_path)
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
        save_evaluation_plot([result], plot_output_path, plot_metadata=export_metadata)
        yaml_output_path = resolve_yaml_output_path(plot_output_path)
        save_evaluation_yaml([result], yaml_output_path, plot_metadata=export_metadata)
        print(
            "Saved evaluation plot:",
            plot_output_path,
            f"(selection={selection_tag}, epoch={int(result['epoch'])})",
        )
        print(f"Saved evaluation YAML: {yaml_output_path}")


def build_eval_context(args):
    device = select_evaluation_device(args.cuda, args.gpu_ids)
    cfg = DataConfig()
    checkpoint_paths = find_epoch_checkpoints(
        args.checkpoint_root,
        args.epoch_step,
        end_epoch=args.end_epoch,
    )
    if len(checkpoint_paths) == 0:
        raise ValueError(f"No epoch checkpoints found in {args.checkpoint_root}")

    apply_checkpoint_config_defaults(args, checkpoint_paths)
    if args.box_coordinate_mode in (None, "auto"):
        args.box_coordinate_mode = BOX_COORDINATE_POLAR
    args.box_coordinate_mode = validate_box_coordinate_mode(
        args.box_coordinate_mode
    )
    apply_standalone_evaluation_coordinate_mode(args)
    apply_eval_profile(args)
    args = apply_task_configuration(args)
    print(
        "Ignore object_label=-1: "
        f"{args.ignore_object_label_minus_one}"
    )
    (
        args.official_class_name_map,
        args.class_display_name_map,
    ) = resolve_official_eval_class_name_map(args.class_names)
    args.metric_class_names = [
        args.official_class_name_map[class_id]
        for class_id in sorted(args.official_class_name_map.keys())
    ]
    model_type = resolve_model_type(args, checkpoint_paths)
    resolved_loss_mode = resolve_loss_mode(
        model_type,
        box_coordinate_mode=args.box_coordinate_mode,
        loss_mode=getattr(args, "loss_mode", "auto"),
    )
    reference_checkpoint = load_torch_checkpoint(
        checkpoint_paths[0][1],
        map_location="cpu",
    )
    source_metadata = extract_checkpoint_source_metadata(reference_checkpoint)
    reference_overrides = infer_checkpoint_decoder_overrides(reference_checkpoint)
    model_variant_name = infer_model_variant_name(
        model_type=model_type,
        checkpoint_or_state_dict=reference_checkpoint,
        include_bus_as_target=source_metadata["include_bus_as_target"],
        train_sequences=source_metadata["train_sequences"],
        train_control_split_enabled=source_metadata["train_control_split_enabled"],
    )
    checkpoint_config = (
        reference_checkpoint.get("config", {})
        if isinstance(reference_checkpoint, dict)
        else {}
    )
    model_configuration_name, model_configuration = build_model_configuration(
        model_type=model_type,
        model_variant_name=model_variant_name,
        checkpoint_config=checkpoint_config,
        model_overrides=reference_overrides,
    )
    source_metadata.update({
        "base_model_type": model_type,
        "model_configuration_name": model_configuration_name,
        "model_configuration": model_configuration,
    })

    train_dataset, validation_dataset, _, validation_loader = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        class_to_idx=args.class_to_idx,
        ignore_class_names=args.ignore_class_names,
        gt_object_ignore_override_path=args.gt_object_ignore_override_path,
        ignore_unmapped_classes=True,
        split_mode=args.split_mode,
        split_dir=args.split_dir,
        scope_mode=args.eval_scope,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
        train_control_split_enabled=getattr(args, "train_control_split_enabled", False),
        train_control_split_dir=getattr(args, "train_control_split_dir", None),
        box_coordinate_mode=args.box_coordinate_mode,
        cartesian_gt_root=args.cartesian_gt_root,
        polar_gt_root=args.polar_gt_root,
        ignore_object_label_minus_one=args.ignore_object_label_minus_one,
    )
    if len(validation_dataset) == 0:
        raise ValueError("Validation split is empty.")
    split_statistics_metadata = build_split_statistics_metadata(
        train_dataset=train_dataset,
        test_dataset=validation_dataset,
    )

    model, _ = build_model_for_checkpoint(
        model_type=model_type,
        device=device,
        num_classes=args.num_classes,
        checkpoint_path=checkpoint_paths[0][1],
        box_coordinate_mode=args.box_coordinate_mode,
        loss_mode=resolved_loss_mode,
    )

    return {
        "args": args,
        "device": device,
        "checkpoint_paths": checkpoint_paths,
        "model_type": model_type,
        "model_variant_name": model_variant_name,
        "source_metadata": source_metadata,
        "plot_metadata": build_plot_metadata(args, model_variant_name, source_metadata),
        "split_statistics_metadata": split_statistics_metadata,
        "validation_loader": validation_loader,
        "model": model,
    }


def update_domain_comparison_outputs(
        results,
        args,
        model_type,
        model_variant_name,
        source_metadata,
    ):
    if not args.domain_comparison_enabled or len(results) == 0:
        return None
    if not args.official_eval_enabled:
        print(
            "Domain-shift comparison tables skipped: the current table format "
            "requires direct official Cartesian BEV/3D metrics."
        )
        return None

    output_dir = resolve_output_base_dir(args.domain_comparison_output_dir)
    sequence_info_path = resolve_output_base_dir(
        args.domain_comparison_sequence_info_path
    )
    metadata = {
        "model_type": model_variant_name,
        "base_model_type": model_type,
        "model_configuration_name": source_metadata.get(
            "model_configuration_name"
        ),
        "model_configuration": source_metadata.get("model_configuration", {}),
        "train_sequences": source_metadata.get("train_sequences"),
        "checkpoint_val_sequences": source_metadata.get("val_sequences"),
        "val_sequences": args.val_sequences,
        "include_bus_as_target": bool(args.include_bus_as_target),
        "checkpoint_include_bus_as_target": bool(
            source_metadata.get("include_bus_as_target", args.include_bus_as_target)
        ),
        "train_control_split_enabled": bool(
            source_metadata.get("train_control_split_enabled", False)
        ),
        "gt_object_ignore_override_path": source_metadata.get(
            "gt_object_ignore_override_path"
        ),
        "eval_scope": args.eval_scope,
        "official_eval_version": args.official_eval_version,
        "official_eval_iou_mode": args.official_eval_iou_mode,
        "ap_score_thresh": float(args.ap_score_thresh),
        "score_thresh": float(args.score_thresh),
        "checkpoint_root": str(args.checkpoint_root),
        "evaluated_checkpoint_count": len(results),
    }
    summary = update_domain_shift_tables(
        results,
        metadata,
        output_dir=output_dir,
        sequence_info_path=sequence_info_path,
    )
    print(
        "Updated domain-shift comparison tables:",
        f"configuration={summary['model_configuration_name']}",
        f"group={summary['evaluation_group']}",
    )
    print(f"  Source: {summary['source_domain']}")
    print(f"  Target: {summary['target_domain']}")
    if summary["target_recorded_in_table"]:
        for criterion, table_path in summary["table_paths"].items():
            selection = summary["selections"][criterion]
            print(
                f"  {criterion}: epoch={selection['epoch']} -> {table_path}"
            )
    else:
        print("  CSV tables: skipped because target weather is normal")
    print(f"  Structured record: {summary['record_path']}")
    return summary


def main():
    args = parse_args()
    context = build_eval_context(args)
    args = context["args"]
    device = context["device"]
    checkpoint_paths = context["checkpoint_paths"]
    model_type = context["model_type"]
    model_variant_name = context["model_variant_name"]
    source_metadata = context["source_metadata"]
    plot_metadata = context["plot_metadata"]
    split_statistics_metadata = context["split_statistics_metadata"]
    validation_loader = context["validation_loader"]
    model = context["model"]
    evaluation_tensorboard_writer, evaluation_tensorboard_log_dir = (
        create_evaluation_tensorboard_writer(
            args=args,
            model_variant_name=model_variant_name,
            source_metadata=source_metadata,
        )
    )
    print(f"Evaluation TensorBoard log dir: {evaluation_tensorboard_log_dir}")
    group_plot_best_only_mode = group_checkpoint_plot_best_only_active(
        args, checkpoint_paths
    )
    default_plot_output_path = None
    default_table_txt_path = None

    print(f"Evaluation classes: {args.class_names}")
    print(f"Bus target enabled: {args.include_bus_as_target}")
    if getattr(args, "train_control_split_enabled", False):
        print(f"Train control split: {args.train_control_split_dir}")
    if args.gt_object_ignore_override_path is not None:
        print(f"GT object ignore override: {args.gt_object_ignore_override_path}")
    print(f"Ignore-mask classes: {args.ignore_class_names}")
    print(
        f"Ignore-mask region: GT box * {args.ignore_mask_expand_ratio} + margin {args.ignore_mask_margin}"
    )
    if args.eval_ignore_suppress_enabled:
        print(
            "Eval ignore suppression: enabled "
            f"(expand_ratio={args.eval_ignore_expand_ratio}, margin={args.eval_ignore_suppress_margin})"
        )
    print(f"Using evaluation device: {device}")
    print(f"Evaluation scope: {args.eval_scope}")
    print(f"Box coordinate mode: {args.box_coordinate_mode}")
    print(
        "Evaluation coordinate mode: "
        f"requested={args.eval_coordinate_mode}, "
        f"effective={args.effective_eval_coordinate_mode}, "
        f"primary={args.evaluation_primary_geometry}"
    )
    if args.end_epoch is not None:
        print(f"Evaluation end epoch: {args.end_epoch}")
    print(
        f"Official K-Radar AP: {'enabled' if args.official_eval_enabled else 'disabled'}"
        f" ({args.official_geometry_source})"
    )
    if args.official_eval_enabled:
        print(f"Official evaluator: {args.official_eval_version}")
        print(f"Official IoU mode: {args.official_eval_iou_mode}")
    print(
        f"Polar BEV AP: {'enabled' if args.polar_eval_enabled else 'disabled'}"
        f" ({args.polar_geometry_source}, IoU={args.polar_iou_thresholds})"
    )
    print(f"AP score threshold: {args.ap_score_thresh}")
    print(f"Detection score threshold: {args.score_thresh}")
    if args.domain_comparison_enabled:
        print(
            "Domain-shift table auto-update: enabled "
            f"({resolve_output_base_dir(args.domain_comparison_output_dir)})"
        )
    if args.official_ap03_only:
        print("Official AP@0.3 only mode: enabled")
    if group_plot_best_only_mode:
        selection_labels = ", ".join(
            metric_key for metric_key, _ in group_plot_selection_specs(args)
        )
        print(
            "Group checkpoint best-only plot mode: enabled "
            f"(select by {selection_labels})."
        )
    print(
        f"Evaluating {len(checkpoint_paths)} checkpoint(s) from {args.checkpoint_root}",
        flush=True,
    )
    if not group_plot_best_only_mode and plot_output_requested(args.plot_output):
        default_plot_output_path = resolve_plot_output_path(
            args=args,
            checkpoint_paths=checkpoint_paths,
            model_type=model_variant_name,
        )
        if default_plot_output_path is not None:
            print(f"Plot output path: {default_plot_output_path}", flush=True)
    if args.table_txt_enabled:
        default_table_txt_path = default_eval_table_txt_path(
            model_variant_name=model_variant_name,
            val_sequences=args.val_sequences,
            base_dir=args.table_output_base_dir,
        )
        print(f"Table txt path: {default_table_txt_path}", flush=True)

    if group_plot_best_only_mode:
        selection_results = []
        selection_iou_mode = selection_iou_mode_for_group_plot(args)
        print(
            f"Selection pass IoU mode: {selection_iou_mode} "
            "(custom/coco/nuscenes disabled)."
        )
        for epoch, checkpoint_path in tqdm.tqdm(
            checkpoint_paths,
            desc="Checkpoint selection",
            ncols=120,
        ):
            result = evaluate_checkpoint_result(
                model=model,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                validation_loader=validation_loader,
                device=device,
                model_type=model_type,
                args=args,
                official_eval_iou_mode=selection_iou_mode,
                official_detection_metrics_enabled=False,
                custom_iou_range_eval_enabled=False,
                coco_style_eval_enabled=False,
                nuscenes_style_eval_enabled=False,
                loss_eval_enabled=False,
            )
            selection_results.append(result)
            write_evaluation_tensorboard_result(
                evaluation_tensorboard_writer,
                result,
            )
            print_checkpoint_metrics(int(epoch), result)

        if len(selection_results) > 0 and args.terminal_epoch_table_enabled:
            print_epoch_ap03_table(selection_results)
        if len(selection_results) > 0 and default_table_txt_path is not None:
            table_metadata = {
                "model_type": model_variant_name,
                "train_sequences": source_metadata["train_sequences"],
                "val_sequences": args.val_sequences,
                "eval_scope": args.eval_scope,
                "eval_coordinate_mode": args.eval_coordinate_mode,
                "effective_eval_coordinate_mode": args.effective_eval_coordinate_mode,
                "box_coordinate_mode": args.box_coordinate_mode,
                "include_bus_as_target": args.include_bus_as_target,
                "checkpoint_include_bus_as_target": source_metadata["include_bus_as_target"],
                "gt_object_ignore_override_path": source_metadata["gt_object_ignore_override_path"],
                "train_control_split_enabled": source_metadata["train_control_split_enabled"],
                "ap_score_thresh": args.ap_score_thresh,
                "score_thresh": args.score_thresh,
                "group_checkpoint_plot_best_only": True,
                **split_statistics_metadata,
            }
            saved_table_path = save_eval_table_txt(
                selection_results,
                default_table_txt_path,
                metadata=table_metadata,
            )
            print(f"Saved evaluation table txt: {saved_table_path}")

        for metric_key, selection_tag in group_plot_selection_specs(args):
            best_metric_result = select_best_result_by_metric(
                selection_results,
                metric_key,
            )
            if best_metric_result is None:
                continue
            print(
                f"{selection_tag}:",
                f"epoch={best_metric_result['epoch']}",
                f"{metric_key}={official_ap_text(best_metric_result.get(metric_key))}",
            )

        selected_entries = build_group_plot_selection_entries(
            selection_results,
            args,
        )
        for entry in selected_entries:
            selection_label = ", ".join(entry["selection_tags"])
            selected_result = entry["result"]
            print(
                "Running full export evaluation for selected checkpoint:",
                f"epoch={selected_result['epoch']}",
                f"selection={selection_label}",
            )
            full_result = evaluate_checkpoint_result(
                model=model,
                checkpoint_path=selected_result["checkpoint_path"],
                epoch=selected_result["epoch"],
                validation_loader=validation_loader,
                device=device,
                model_type=model_type,
                args=args,
            )
            entry["result"] = full_result
            write_evaluation_tensorboard_result(
                evaluation_tensorboard_writer,
                full_result,
                namespace="evaluation_selected",
            )
            print_checkpoint_metrics(int(full_result["epoch"]), full_result)

        save_group_best_only_plot_exports(
            selected_entries=selected_entries,
            checkpoint_paths=checkpoint_paths,
            model_type=model_variant_name,
            args=args,
            plot_metadata=plot_metadata,
        )
        update_domain_comparison_outputs(
            results=selection_results,
            args=args,
            model_type=model_type,
            model_variant_name=model_variant_name,
            source_metadata=source_metadata,
        )
        evaluation_tensorboard_writer.close()
        return

    results = []
    for epoch, checkpoint_path in tqdm.tqdm(
        checkpoint_paths,
        desc="Checkpoints",
        ncols=120,
    ):
        result = evaluate_checkpoint_result(
            model=model,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            validation_loader=validation_loader,
            device=device,
            model_type=model_type,
            args=args,
        )
        results.append(result)
        write_evaluation_tensorboard_result(
            evaluation_tensorboard_writer,
            result,
        )
        print_checkpoint_metrics(int(epoch), result)

    if len(results) > 0:
        if args.terminal_epoch_table_enabled and len(results) > 1:
            print_epoch_ap03_table(results)
        best_result = max(
            results,
            key=result_main_metric_value,
        )
        best_metric_key = result_main_metric_key(best_result)
        print(
            "best_epoch:",
            f"epoch={best_result['epoch']}",
            f"{best_metric_key}="
            f"{official_ap_text(result_main_metric_value(best_result))}",
        )
        if default_table_txt_path is not None:
            table_metadata = {
                "model_type": model_variant_name,
                "train_sequences": source_metadata["train_sequences"],
                "val_sequences": args.val_sequences,
                "eval_scope": args.eval_scope,
                "eval_coordinate_mode": args.eval_coordinate_mode,
                "effective_eval_coordinate_mode": args.effective_eval_coordinate_mode,
                "box_coordinate_mode": args.box_coordinate_mode,
                "include_bus_as_target": args.include_bus_as_target,
                "checkpoint_include_bus_as_target": source_metadata["include_bus_as_target"],
                "gt_object_ignore_override_path": source_metadata["gt_object_ignore_override_path"],
                "train_control_split_enabled": source_metadata["train_control_split_enabled"],
                "ap_score_thresh": args.ap_score_thresh,
                "score_thresh": args.score_thresh,
                **split_statistics_metadata,
            }
            saved_table_path = save_eval_table_txt(
                results,
                default_table_txt_path,
                metadata=table_metadata,
            )
            print(f"Saved evaluation table txt: {saved_table_path}")

    if default_plot_output_path is not None:
        output_dir = os.path.dirname(default_plot_output_path)
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
        save_evaluation_plot(
            results,
            default_plot_output_path,
            plot_metadata=plot_metadata,
        )
        yaml_output_path = resolve_yaml_output_path(default_plot_output_path)
        save_evaluation_yaml(results, yaml_output_path, plot_metadata=plot_metadata)
        print(f"Saved evaluation plot: {default_plot_output_path}")
        print(f"Saved evaluation YAML: {yaml_output_path}")

    update_domain_comparison_outputs(
        results=results,
        args=args,
        model_type=model_type,
        model_variant_name=model_variant_name,
        source_metadata=source_metadata,
    )
    evaluation_tensorboard_writer.close()


if __name__ == "__main__":
    main()
