"""Evaluation configuration, argument parsing, and runtime helpers."""

import argparse

import numpy as np
import torch

from configs.data import CARTESIAN_GT_ROOT
from data.coordinates import SCOPE_CHOICES
from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    require_cartesian_data,
)
from eval.custom_iou_range import DEFAULT_CUSTOM_IOU_THRESHOLDS
from eval.distance_quartiles import normalize_distance_quartile_bins
from models import MODEL_TYPES
from training.torch_load import load_torch_checkpoint

from configs.evaluation import EVAL_CONFIG

_EVAL_CFG_MISSING = object()
HEATMAP_SCORE_MODES = ("peak_times_local_mean", "peak_only")
OFFICIAL_CLASS_TOKEN_BY_DATASET_NAME = {
    "Sedan": "sed",
    "Bus or Truck": "bus",
}


__all__ = [
    'eval_cfg_value',
    'should_inherit_from_checkpoint',
    'load_torch_checkpoint',
    'parse_gpu_ids',
    'parse_cuda_choice',
    'select_evaluation_device',
    'parse_args',
    'normalize_bool_flag',
    'normalize_float_thresholds',
    'apply_standalone_evaluation_coordinate_mode',
    'resolve_official_eval_class_name_map'
]


def eval_cfg_value(key):
    return EVAL_CONFIG.get(key, _EVAL_CFG_MISSING)


def should_inherit_from_checkpoint(key):
    value = eval_cfg_value(key)
    if value is _EVAL_CFG_MISSING or value is None:
        return True
    # For these two workflow selectors, "auto" explicitly means that the
    # checkpoint is authoritative.  This lets configs/evaluation.py stay fully
    # automatic without requiring the user to edit it per checkpoint.
    if key in {"box_coordinate_mode", "loss_mode"}:
        return str(value).strip().lower() == "auto"
    return False


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
        "checkpoint_root": None,
        "epoch_step": 1,
        "start_epoch": None,
        "end_epoch": None,
        "batch_size": 100,
        "split_mode": "kradar_file",
        "split_dir": "data/manifests/kradar",
        "train_sequences": None,
        "val_sequences": None,
        "eval_val_sequences": None,
        "eval_frame_manifest_path": None,
        "eval_gt_object_ignore_override_path": None,
        "eval_report_path": None,
        "seed": 42,
        "num_workers": 0,
        "limit_samples": None,
        "eval_scope": None,
        "box_coordinate_mode": None,
        "cartesian_gt_root": CARTESIAN_GT_ROOT,
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
        "distance_quartile_eval_enabled": False,
        "distance_quartile_bins": None,
        "nuscenes_style_eval_enabled": False,
        "official_detection_metrics_enabled": True,
        "group_checkpoint_plot_best_only": False,
        "ap_score_thresh": 0.01,
        "score_thresh": 0.3,
        "plot_output": None,
    }
    eval_config = dict(EVAL_CONFIG)
    cfg_defaults.update(eval_config)

    parser = argparse.ArgumentParser(
        description="Run official K-Radar KITTI-style evaluation."
    )
    parser.add_argument("--checkpoint-root", default=cfg_defaults["checkpoint_root"])
    parser.add_argument("--epoch-step", type=int, default=cfg_defaults["epoch_step"])
    parser.add_argument("--start-epoch", type=int, default=cfg_defaults["start_epoch"])
    parser.add_argument("--end-epoch", type=int, default=cfg_defaults["end_epoch"])
    parser.add_argument("--batch-size", type=int, default=cfg_defaults["batch_size"])
    parser.add_argument(
        "--split-mode",
        default=cfg_defaults["split_mode"],
        choices=["kradar_file", "sequence"],
    )
    parser.add_argument("--split-dir", default=cfg_defaults["split_dir"])
    parser.add_argument("--train-sequences", default=cfg_defaults["train_sequences"])
    parser.add_argument("--val-sequences", default=cfg_defaults["val_sequences"])
    parser.add_argument(
        "--eval-val-sequences",
        default=cfg_defaults["eval_val_sequences"],
        help=(
            "Authoritative standalone-evaluation sequences. When supplied, "
            "these replace checkpoint val_sequences after checkpoint defaults."
        ),
    )
    parser.add_argument(
        "--eval-frame-manifest-path",
        default=cfg_defaults["eval_frame_manifest_path"],
        help="Strict sequence,frame.txt manifest for validation-only evaluation.",
    )
    parser.add_argument(
        "--eval-gt-object-ignore-override-path",
        default=cfg_defaults["eval_gt_object_ignore_override_path"],
        help=(
            "Evaluation-only object ignore JSON. It never changes checkpoint "
            "training metadata or the legacy training dataset."
        ),
    )
    parser.add_argument(
        "--eval-report-path",
        default=cfg_defaults["eval_report_path"],
        help="Exact evaluation TXT report path; implies --table-txt-enabled.",
    )
    parser.add_argument("--seed", type=int, default=cfg_defaults["seed"])
    parser.add_argument("--num-workers", type=int, default=cfg_defaults["num_workers"])
    parser.add_argument("--limit-samples", type=int, default=cfg_defaults["limit_samples"])
    parser.add_argument("--eval-scope", default=cfg_defaults["eval_scope"], choices=SCOPE_CHOICES)
    parser.add_argument(
        "--box-coordinate-mode",
        default=cfg_defaults["box_coordinate_mode"],
        choices=["auto", BOX_COORDINATE_CARTESIAN],
    )
    parser.add_argument(
        "--cartesian-gt-root",
        default=cfg_defaults["cartesian_gt_root"],
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
        choices=["auto", "cuda", "gpu", "cpu", "axis_aligned"],
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
        "--distance-quartile-eval-enabled",
        default=cfg_defaults["distance_quartile_eval_enabled"],
        help=(
            "Derive four half-open GT-distance quartiles from the evaluation "
            "set and report official BEV/3D AP@0.3 for each quartile."
        ),
    )
    parser.add_argument(
        "--distance-quartile-bins",
        default=cfg_defaults["distance_quartile_bins"],
        help=(
            "Optional fixed target-domain quartile boundaries as JSON/list, "
            "evaluation report/JSON path, or 0-a,a-b,b-c,c-inf."
        ),
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
    # Evaluation always keeps object_label=-1 rows.  This is intentionally
    # not exposed as an eval_cfg or command-line switch.
    args.ignore_object_label_minus_one = False
    if args.start_epoch is not None:
        args.start_epoch = int(args.start_epoch)
    if args.end_epoch is not None:
        args.end_epoch = int(args.end_epoch)
    if args.start_epoch is not None and args.start_epoch <= 0:
        raise ValueError(
            f"start_epoch must be greater than 0, got {args.start_epoch}"
        )
    if args.end_epoch is not None and args.end_epoch <= 0:
        raise ValueError(
            f"end_epoch must be greater than 0, got {args.end_epoch}"
        )
    if (
        args.start_epoch is not None
        and args.end_epoch is not None
        and args.start_epoch > args.end_epoch
    ):
        raise ValueError(
            "start_epoch must be less than or equal to end_epoch, got "
            f"{args.start_epoch}>{args.end_epoch}"
        )
    args.ap_score_thresh = float(args.ap_score_thresh)
    args.score_thresh = float(args.score_thresh)
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
    args.distance_quartile_eval_enabled = normalize_bool_flag(
        args.distance_quartile_eval_enabled,
        name="distance_quartile_eval_enabled",
    )
    args.distance_quartile_bins = normalize_distance_quartile_bins(
        args.distance_quartile_bins
    )
    if (
        args.distance_quartile_bins is not None
        and not args.distance_quartile_eval_enabled
    ):
        raise ValueError(
            "distance_quartile_bins requires "
            "distance_quartile_eval_enabled=true."
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
    # Keep the regular metric set permanently enabled.  The old command-line
    # AP@0.3-only and terminal epoch-table switches were removed.
    args.official_ap03_only = False
    args.group_checkpoint_plot_best_only = normalize_bool_flag(
        args.group_checkpoint_plot_best_only,
        name="group_checkpoint_plot_best_only",
    )
    args.eval_ignore_suppress_enabled = normalize_bool_flag(
        args.eval_ignore_suppress_enabled,
        name="eval_ignore_suppress_enabled",
    )
    args.table_txt_enabled = normalize_bool_flag(
        args.table_txt_enabled,
        name="table_txt_enabled",
    )
    for path_name in (
        "eval_frame_manifest_path",
        "eval_gt_object_ignore_override_path",
        "eval_report_path",
    ):
        value = getattr(args, path_name)
        if value is not None:
            value = str(value).strip()
            if value == "":
                value = None
        setattr(args, path_name, value)
    if args.eval_report_path is not None:
        args.table_txt_enabled = True
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


def apply_standalone_evaluation_coordinate_mode(args):
    args.box_coordinate_mode = require_cartesian_data(args.box_coordinate_mode)
    # Keep established output metadata fields while deriving them directly
    # from the only supported evaluation geometry.
    args.eval_coordinate_mode = BOX_COORDINATE_CARTESIAN
    args.effective_eval_coordinate_mode = BOX_COORDINATE_CARTESIAN
    args.official_eval_enabled = True
    args.evaluation_primary_geometry = BOX_COORDINATE_CARTESIAN
    args.official_geometry_source = "direct"
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
