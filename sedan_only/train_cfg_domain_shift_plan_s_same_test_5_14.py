from cfg_model import SCOPE_FULL


TRAIN_CONFIG = {
    "epochs": 30,
    "batch_size": 64,
    "lr": 1e-4,
    "max_detections": 64,
    "heatmap_radius": 3,
    "centerpoint_giou_loss_weight": 2.0,
    "quality_loss_weight": 0.25,
    "ignore_mask_margin": 1.0,
    "ignore_mask_expand_ratio": 1.5,
    "ignore_class_names": (
        "Bus or Truck",
        "Pedestrian",
        "Pedestrian Group",
        "Bicycle",
        "Bicycle Group",
        "Motorcycle",
    ),
    "init_from_checkpoint": "",

    "eval_train": False,
    "training_eval_enabled": False,
    "best_metric_key": "auto",
    "official_eval_enabled": True,
    "official_eval_version": "revised",
    "official_eval_iou_backend": "cpu",
    "official_eval_iou_mode": "easy",
    "official_detection_metrics_enabled": True,
    "detection_score_thresh": 0.3,

    "train_ratio": 0.7,
    "train_scope": SCOPE_FULL,
    "split_mode": "file",
    "split_dir": "split/domain_shift_plan_s_same_test_5_14",
    "train_sequences": (1, 6, 15, 18, 20),
    "val_sequences": (5, 14),

    "seed": 42,
    "num_workers": 0,
    "limit_samples": None,
    "checkpoint_epoch_step": 1,
    "checkpoint_base_dir": "checkpoints",
    "log_base_dir": "runs",
    "gpu_ids": "0,1,2",
    "model_type": "model7",
}


RESUME_CONFIG = {
    **TRAIN_CONFIG,
    "resume_checkpoint": "",
    "initial_best_checkpoint": None,
    "start_epoch": None,
    "end_epoch": 60,
    "load_optimizer": True,
}
