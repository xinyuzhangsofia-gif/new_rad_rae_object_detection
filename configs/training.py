"""Stable model/training defaults and compatibility configuration exports."""

from configs.data import CARTESIAN_GT_ROOT, CHECKPOINT_BASE_DIR, LOG_BASE_DIR
from configs.domain_shift import DOMAIN_SHIFT_CONFIG, EXPERIMENT_QUEUE_CONFIG
from configs.historical_overrides import HISTORICAL_EXPERIMENT_QUEUE_OVERRIDES
from configs.resume import build_resume_config
from configs.runtime import (
    EXPERIMENT_QUEUE_RUNTIME_CONFIG,
    TRAIN_RUNTIME_CONFIG,
)
from data.coordinates import SCOPE_FULL, SCOPE_NARROW


# Edit stable model/training settings here, then run: python train.py
# The imported sections below keep the historical flat-dictionary API used by
# training, queue workers, checkpoints, and existing scripts.
TRAIN_CONFIG = {
    # Cartesian GT only. RAD/RAE radar tensors remain the model inputs.
    "box_coordinate_mode": "cartesian",
    # radenet, centerpoint without normalization, or auto (legacy selection).
    "loss_mode": "centerpoint",
    "cartesian_gt_root": CARTESIAN_GT_ROOT,

    # Model and optimization.
    "epochs": 50,
    "batch_size": 64,
    "lr": 5e-5,
    "max_detections": 64,
    "heatmap_radius": 3,
    "centerpoint_gwd_loss_weight": 2.0,
    "quality_loss_weight": 0.25,
    "init_from_checkpoint": "",

    # Target classes and ignored GT regions.
    "include_bus_as_target": True,
    "ignore_object_label_minus_one": False,
    "ignore_out_of_scope_gt": True,
    "gt_object_ignore_override_path": None,
    "ignore_mask_margin": 1.0,
    "ignore_mask_expand_ratio": 1.5,
    "ignore_class_names": (
        "Pedestrian",
        "Pedestrian Group",
        "Bicycle",
        "Bicycle Group",
        "Motorcycle",
    ),

    # Training-time evaluation and checkpoint selection.
    "eval_train": False,
    "training_eval_enabled": True,
    "best_metric_key": "auto",
    "official_eval_enabled": None,
    "official_eval_version": "revised",
    "official_eval_iou_backend": "gpu",
    "official_eval_iou_mode": "easy",
    "official_detection_metrics_enabled": False,
    "ap_score_thresh": 0.01,
    "score_thresh": 0.3,
    "train_scope": SCOPE_FULL,

    # Ordinary train/validation split.
    "split_mode": "kradar_file",
    "split_dir": "split",
    "train_sequences": None,
    "val_sequences": None,

    # Ordinary run and checkpoint behavior.
    "seed": 42,
    "limit_samples": None,
    "checkpoint_epoch_step": 1,
    "checkpoint_base_dir": CHECKPOINT_BASE_DIR,
    "checkpoint_layout": "legacy",
    "checkpoint_filename_style": "compact",
    "log_base_dir": LOG_BASE_DIR,
    "post_training_eval_enabled": False,
    "model_type": "model7",
    "model7_decoder_hidden_channels": "64",

    # Compatibility aggregation. Edit these settings in their named modules.
    **DOMAIN_SHIFT_CONFIG,
    **EXPERIMENT_QUEUE_CONFIG,
    **TRAIN_RUNTIME_CONFIG,
    **EXPERIMENT_QUEUE_RUNTIME_CONFIG,
    **HISTORICAL_EXPERIMENT_QUEUE_OVERRIDES,
}


# Backward-compatible import used by train_resume.py and external scripts.
# Resume-only choices live in configs/resume.py and reuse all training defaults.
RESUME_CONFIG = build_resume_config(TRAIN_CONFIG)


__all__ = ["TRAIN_CONFIG", "RESUME_CONFIG", "SCOPE_FULL", "SCOPE_NARROW"]
