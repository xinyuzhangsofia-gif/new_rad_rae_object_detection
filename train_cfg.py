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
    "epochs": 30,
    "batch_size": 32,
    "lr": 5e-5,   #5e-5
    "max_detections": 64,
    "heatmap_radius": 3,
    "centerpoint_gwd_loss_weight": 2.0,  # metric-space BEV Gaussian Wasserstein loss
    "quality_loss_weight": 0.25,  # only model6 has the separate quality head; inactive for model7
    "init_from_checkpoint": "",           # optional init checkpoint; 2-class cls heads can be adapted when include_bus_as_target=False

    # Target classes and ignored GT regions
    "include_bus_as_target": False,         # True -> 2-class Sedan+Bus; False -> Sedan-only and Bus becomes ignore
    "ignore_object_label_minus_one": False, # Keep object_label=-1 in Polar/Cartesian GT
    "ignore_out_of_scope_gt": True,         # Ignore GT boxes outside the configured radar RAE/scope during training
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
    "split_dir": "split",  # used when split_mode == "file"

    # Ordered domain-shift experiment queue. When enabled, one top-level
    # train.py reads this table and dispatches source/target branch tasks in
    # table order; the worker settings below allow both branches to overlap.
    "experiment_queue_enabled": True,
    "experiment_sheet_path": "experiments/rain_experiments.txt",
    "experiment_queue_branches": ("source", "target"),
    # Completed branches are skipped. Missing AP values are trained/evaluated
    # and written back into the same CSV, including TD and the average row.
    "experiment_queue_skip_completed_branches": True,
    "experiment_queue_update_sheet_results": True,
    "experiment_queue_require_full_table": True,
    "experiment_results_base_dir": "evaluation_plots",
    # Asynchronous table scheduler. Two independent training subprocesses keep
    # the current three-GPU DataParallel configuration. Completed checkpoints
    # enter a separate two-worker evaluation queue, so the next training task
    # does not wait for evaluation. Only the parent process writes the table.
    "experiment_queue_train_workers": 2,
    "experiment_queue_eval_workers": 2,
    "experiment_queue_gpu_strategy": "shared_dynamic",
    "experiment_queue_train_gpu_slots": (
        "0,1,2",
        "0,1,2",
    ),
    "experiment_queue_eval_gpu_pool": "0,1,2",
    "experiment_queue_eval_min_free_memory_mb": 4000,
    # A temporary reservation prevents two evaluations launched in the same
    # polling cycle from both assuming that the same free memory is available.
    "experiment_queue_eval_reservation_memory_mb": 3500,
    "experiment_queue_poll_seconds": 1.0,

    # Domain-shift experiment design.
    #   source -> shared + source train, evaluated on target_test
    #   target -> shared + target train, evaluated on the same target_test
    # train_sequences/val_sequences are derived automatically from this block.
    "domain_shift_train_branch": "source",  # "source" or "target"
    "shared_train_sequences": (9, 12),
    "source_train_sequences": (4, 3, 20, 14),
    "target_train_sequences": (46, 47, 55, 58),
    "target_test_sequences": (54, 56),
    # The target-test weather and checkpoints/<weather>/ directory are inferred
    # automatically from target_test_sequences using this table.
    "sequence_information_path": "sequence_information.csv",
    # Optional chronological truncation, including shared sequences.
    # Empty means use each selected sequence in full.
    "train_sequence_half_selection": {},
    "train_sequence_half_ratio": 0.5,

    # Automatic controlled training sequences. In domain-shift mode the source
    # and reference sequences are derived from source_train_sequences and
    # target_train_sequences, so they cannot drift away from the experiment.
    # In an experiment queue, True controls only the s并根据每一行的对应关系生成不同控制目录；Target分支仍自动保持不控制。Group2、Group3和当前Group4则需要重新训练。ource branch; the target
    # branch is automatically kept unfiltered.
    "train_control_split_enabled": True,
    "controlled_split_base_dir": "split",  # generated control directories are created below this directory
    "control_window_position": "last",  # "first" or "last" continuous source window
    # Physical radar-center range bins used by the optional controlled split.
    # In Cartesian mode, r_m = sqrt(x**2 + y**2 + z**2) is computed from the
    # Cartesian GT center.  The intervals are left-closed and right-open.
    "control_range_m_bins": (
        (0.0, 20.0),
        (20.0, 40.0),
        (40.0, 60.0),
        (60.0, 80.0),
        (80.0, 120.0),
    ),
    "control_num_trials": 300,  # random trials for secondary distribution tie-breaking
    # Primary control: if the source total is within this ratio of the
    # reference total, keep every bbox and ignore distance-bin differences.
    "control_total_bbox_tolerance_ratio": 0.05,
    "train_control_split_dir": None,  # filled automatically when control is enabled

    # Runtime, output, and model choice
    "seed": 42,
    "num_workers": 0,
    "limit_samples": None,
    "checkpoint_epoch_step": 1,
    "checkpoint_base_dir": "checkpoints",
    "checkpoint_layout": "weather_train_test",
    "checkpoint_filename_style": "compact",
    "log_base_dir": "runs",
    "gpu_ids": "0,1,2",
    # After a successful final epoch, evaluate this run's checkpoint directory.
    # The allowed GPU with the most free memory is selected automatically.
    "post_training_eval_enabled": True,
    "post_training_eval_min_free_memory_mb": 4096,
    "model_type": "model7",               # remains model7 in both coordinate modes
    # model7 decoder width: choose 64 or 128.  "auto" keeps the historical
    # default (Polar=64, Cartesian=128).  The FPN width remains 128.
    "model7_decoder_hidden_channels": "64",
}


# Edit this block, then run:
#   python train_resume.py
RESUME_CONFIG = {
    **TRAIN_CONFIG,
    "resume_checkpoint": (
        "checkpoints/heavy_snow/"
        "0729_train_seq9_12_14-15_18_20_test_seq46-47/"
        "0729_epoch_012.pth"
    ),
    "initial_best_checkpoint": None,       # optional previous global best checkpoint
    "start_epoch": None,                   # None means checkpoint epoch + 1
    "end_epoch": 30,                       # final epoch number for resumed training
    "load_optimizer": True,                # resume optimizer state if checkpoint has it
    # Continue this interrupted run in its existing checkpoint/TensorBoard
    # directories. Existing epoch files are never overwritten.
    "resume_save_in_checkpoint_dir": True,
    "resume_tensorboard_log_dir": (
        "runs/object_detection/"
        "20260729_152024_152755__model7_sedan_only__"
        "seq9_12_14-15_18_20_46-47"
    ),
    # Restore the exact data selection used by the interrupted run.
    "shared_train_sequences": (9, 12),
    "source_train_sequences": (14, 15, 18, 20),
    "target_train_sequences": (57, 56, 54, 55),
    "target_test_sequences": (46, 47),
    "train_sequence_half_selection": {},
    "train_control_split_enabled": True,
    "train_control_split_dir": (
        "split/controled_seq14_ref57__seq15_ref56__"
        "seq18_ref54__seq20_ref55"
    ),
}
