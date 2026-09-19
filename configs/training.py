"""Stable model/training defaults and composed configuration exports."""

from configs.data import CARTESIAN_GT_ROOT, CHECKPOINT_BASE_DIR, LOG_BASE_DIR
from configs.domain_shift import DOMAIN_SHIFT_CONFIG, EXPERIMENT_QUEUE_CONFIG
from configs.resume import build_resume_config
from configs.runtime import (
    EXPERIMENT_QUEUE_RUNTIME_CONFIG,
    TRAIN_RUNTIME_CONFIG,
)
from data.coordinates import SCOPE_FULL, SCOPE_NARROW


# Edit stable model/training settings here, then run: python train.py
# The imported sections below compose the settings used by training and queue
# workers into one runtime dictionary.
TRAIN_CONFIG = {
    # Cartesian GT only. RAD/RAE radar tensors remain the model inputs.
    "box_coordinate_mode": "cartesian",
    # radenet, centerpoint without normalization, or auto (model-compatible selection).
    "loss_mode": "radenet",
    "cartesian_gt_root": CARTESIAN_GT_ROOT,

    # Model and optimization.
    "epochs": 30,
    "batch_size": 8,
    "lr": 5e-5,
    "max_detections": 64,
    "heatmap_radius": 3,
    "centerpoint_gwd_loss_weight": 2.0,
    "quality_loss_weight": 0.25,

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

    # Training-time evaluation.
    "training_eval_enabled": True,
    "training_eval_train_set_enabled": False,
    "training_eval_best_metric_key": "auto",
    "training_eval_official_version": "revised",
    "training_eval_iou_backend": "gpu",
    "training_eval_iou_mode": "easy",
    "training_eval_detection_metrics_enabled": False,
    "training_eval_ap_score_thresh": 0.01,
    "training_eval_score_thresh": 0.3,
    "training_eval_coco_style_enabled": False,
    "training_eval_nuscenes_style_enabled": False,
    "train_scope": SCOPE_FULL,

    # Post-training automatic evaluation.
    "post_training_eval_enabled": False,
    # post_training_eval_min_free_memory_mb comes from configs/runtime.py.

    # Standalone evaluation lives exclusively in configs/evaluation.py.

    # Ordinary train/validation split.
    "split_mode": "kradar_file",
    "split_dir": "data/manifests/kradar",
    "train_sequences": None,
    "val_sequences": None,

    # Ordinary run and checkpoint behavior.
    "seed": 42,
    "limit_samples": None,
    "checkpoint_epoch_step": 1,
    "checkpoint_base_dir": CHECKPOINT_BASE_DIR,
    "checkpoint_filename_style": "compact",
    "log_base_dir": LOG_BASE_DIR,
    "model_type": "model13",
    "model7_decoder_hidden_channels": "64",

    # Composed training, experiment, and runtime settings.
    **DOMAIN_SHIFT_CONFIG,
    **EXPERIMENT_QUEUE_CONFIG,
    **TRAIN_RUNTIME_CONFIG,
    **EXPERIMENT_QUEUE_RUNTIME_CONFIG,
}


# Backward-compatible import used by train_resume.py and external scripts.
# Resume-only choices live in configs/resume.py and reuse all training defaults.
RESUME_CONFIG = build_resume_config(TRAIN_CONFIG)


__all__ = ["TRAIN_CONFIG", "RESUME_CONFIG", "SCOPE_FULL", "SCOPE_NARROW"]
