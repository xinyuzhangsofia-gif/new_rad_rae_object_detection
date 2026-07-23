from cfg_model import SCOPE_FULL, SCOPE_NARROW


# Edit this file, then run:
#   python train.py
TRAIN_CONFIG = {
    # Coordinate/annotation workflow. Change only this line to choose the way:
    #   "polar"     -> Polar GT, Polar R-A box regression, axis-aligned Polar BEV AP
    #   "cartesian" -> Cartesian GT + metric Cartesian box regression;
    #                  choose RADE-Net or CenterPoint with loss_mode.
    "box_coordinate_mode": "cartesian",     # polar order cartesian
    # Loss selector for compatible workflows:
    #   "radenet"    -> official RADE-Net heatmap + GWD + Smooth-L1 loss,
    #                   including the original detached-mean normalization
    #   "centerpoint"-> Gaussian-Focal heatmap + Smooth-L1 + GWD;
    #                   no RADE-Net detached-mean normalization
    #   "auto"       -> select from model/coordinate mode (legacy behavior)
    "loss_mode": "centerpoint",    #radenet, centerpoint without normalization
    "cartesian_gt_root": (
        "/home/local/xinyu/K-Radar-GT-cartesian-radar-v2"
    ),
    "polar_gt_root": (
        "/home/local/xinyu/K-Radar-GT-Polar-v2.9"
    ),

    # Model and optimization
    "epochs": 50,
    "batch_size": 32,
    "lr": 5e-5,   #5e-5
    "max_detections": 64,
    "heatmap_radius": 3,
    "centerpoint_gwd_loss_weight": 2.0,  # metric-space BEV Gaussian Wasserstein loss
    "quality_loss_weight": 0.25,  # only model6 has the separate quality head; inactive for model7
    "init_from_checkpoint": "",           # optional init checkpoint; 2-class cls heads can be adapted when include_bus_as_target=False

    # Target classes and ignored GT regions
    "include_bus_as_target": False,         # True -> 2-class Sedan+Bus; False -> Sedan-only and Bus becomes ignore
    "ignore_object_label_minus_one": True, # False -> keep -1 labels; True -> remove them from Polar/Cartesian GT
    "gt_object_ignore_override_path": None,  # optional; if None, auto-use split_dir/object_ignore_override.json when present
    "ignore_mask_margin": 1.0,           # ignored region margin added after box scaling
    "ignore_mask_expand_ratio": 1.5,     # ignored region = GT ignore box size * ratio + margin
    # These classes are always ignored. Bus is added/removed automatically by include_bus_as_target.
    "ignore_class_names": (
        "Pedestrian",
        "Pedestrian Group",
        "Bicycle",
        "Bicycle Group",
        "Motorcycle",
    ),

    # Training-time evaluation and checkpoint selection
    "eval_train": False,
    "training_eval_enabled": False,        # False -> skip detection evaluation and do not select a best checkpoint
    "best_metric_key": "auto",             # only used when training_eval_enabled=True
    # Set automatically from box_coordinate_mode; do not edit this separately.
    "official_eval_enabled": None,
    "official_eval_version": "revised",    # revised or legacy
    # The flags below only take effect when training_eval_enabled is True.
    # For training-time validation, use cpu to avoid GPU OOM caused by
    # numba/CUDA rotated-IoU competing with the model training memory.
    # Standalone evaluation.py can still use cuda from eval_cfg.py.
    "official_eval_iou_backend": "cpu",    # auto, cuda, or cpu
    "official_eval_iou_mode": "all",       # easy=0.3, mod=0.5, hard=0.7, all=0.5+0.3
    "polar_iou_thresholds": (0.3, 0.5),
    "ap_score_thresh": 0.01,               # only used when training_eval_enabled=True; boxes below this are dropped before AP/mAP
    "score_thresh": 0.3,                   # only used when training_eval_enabled=True; for TP/FP/FN/Precision/Recall/F1
    "train_scope": SCOPE_FULL,             # SCOPE_FULL or SCOPE_NARROW

    # Base train/validation split
    "train_ratio": 0.7,
    "split_mode": "sequence",              # "sequence", "random", or "file"
    "split_dir": "split/domain_shift_plan_s_same_test_5_14",  # used when split_mode == "file"
    "train_sequences": (1,5,6,14,15,18,20),  # used when split_mode == "sequence"  # 3, 9, 11, 12
    "val_sequences": (10,),

    # Automatic controlled training sequences
    "train_control_split_enabled": False,  # True -> generate and use a new controlled split for controled_sequences
    "controled_sequences": (),  # must be a subset of train_sequences when control is enabled
    "reference_sequences": (),  # one reference is reused for all controlled sequences, or use one per controlled sequence
    "controlled_split_base_dir": "split",  # generated control directories are created below this directory
    "control_window_position": "last",  # "first" or "last" continuous source window
    "control_ridx_bins": (
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 90.0),
        (90.0, 120.0),
        (120.0, 144.0),
    ),
    "control_num_trials": 300,  # random trials used to get the closest empty-frame rate
    "train_control_split_dir": None,  # filled automatically when control is enabled

    # Runtime, output, and model choice
    "seed": 42,
    "num_workers": 0,
    "limit_samples": None,
    "checkpoint_epoch_step": 1,
    "checkpoint_base_dir": "checkpoints",
    "log_base_dir": "runs",
    "gpu_ids": "0,1,2",
    "model_type": "model7",               # remains model7 in both coordinate modes
}


# Edit this block, then run:
#   python train_resume.py
RESUME_CONFIG = {
    **TRAIN_CONFIG,
    "resume_checkpoint": "",              # checkpoint path to resume from
    "initial_best_checkpoint": None,       # optional previous global best checkpoint
    "start_epoch": None,                   # None means checkpoint epoch + 1
    "end_epoch": 150,                      # final epoch number for resumed training
    "load_optimizer": True,                # resume optimizer state if checkpoint has it
}
