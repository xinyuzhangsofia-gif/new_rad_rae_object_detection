from cfg_model import SCOPE_FULL, SCOPE_NARROW


# Edit this file, then run:
#   python train.py
TRAIN_CONFIG = {
    "epochs": 40,
    "batch_size": 30,
    "lr": 1e-4,
    "max_detections": 64,
    "heatmap_radius": 3,
    "centerpoint_giou_loss_weight": 2.0,
    "quality_loss_weight": 0.25,
    "score_thresh": 0.5,
    "eval_iou_thresh": 0.3,
    "eval_train": False,
    "best_metric_key": "auto",             # auto -> official_bev_mAP_0.3 when official eval is on, else mAP
    "official_eval_enabled": True,         # run official K-Radar evaluator during validation
    "official_eval_version": "revised",    # revised or legacy
    "official_eval_iou_backend": "auto",   # auto, cuda, or cpu
    "official_eval_iou_mode": "all",       # easy=0.3, mod=0.5, hard=0.7, all=all three
    "official_eval_include_empty_gt_frames": False,
    "train_ratio": 0.7,
    "train_scope": SCOPE_NARROW,             # SCOPE_FULL or SCOPE_NARROW

    "split_mode": "sequence",              # "sequence", "random", or "file"
    "train_sequences": (1,4,5,6,11,14,20),  # used when split_mode == "sequence"
    "val_sequences": (3,18),
    "seed": 42,
    "num_workers": 0,
    "limit_samples": None,
    "checkpoint_epoch_step": 10,
    "checkpoint_base_dir": "checkpoints",
    "log_base_dir": "runs",
    "gpu_ids": "0,1,2",
    "model_type": "model15",               # model1 ... model15
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
