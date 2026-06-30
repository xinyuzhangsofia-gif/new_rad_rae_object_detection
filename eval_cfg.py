# Edit this file, then run:
#   python evaluation.py
# Split / sequence / scope-related settings are inherited from the checkpoint
# unless you explicitly override them here.
EVAL_CONFIG = {
    # Checkpoint path, or a directory that contains multiple epoch checkpoints.
    "checkpoint_root": (
        "checkpoints/object_detection/"
        "20260625_002607_379659__model_8__seq1_4-6_11_14_20_3_18/"
        "20260625_045800_mAP_35p3469_model_8_global_best_epoch_037_seq1-11.pth"
    ),

    # Used only when checkpoint_root is a directory.
    "epoch_step": 1,

    # Official K-Radar KITTI-style metric settings.
    "official_eval_version": "revised",
    "official_eval_iou_mode": "all",   # easy=0.3, mod=0.5, hard=0.7, all=0.7/0.5/0.3
    "official_eval_iou_backend": "cuda",  # auto, cuda, cpu
    "custom_iou_range_eval_enabled": True,  # custom AP averaged over a user-defined IoU list
    "custom_iou_thresholds": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "coco_style_eval_enabled": True,  # COCO-style AP over IoU=0.50:0.05:0.95
    "nuscenes_style_eval_enabled": True,  # nuScenes-style AP over center-distance thresholds
    "detection_score_thresh": 0.3,  # used for TP / FP / FN / Precision / Recall / F1 only

    # Runtime settings.
    "batch_size": 100,
    "num_workers": 0,
    "gpu_ids": "0,1,2",
    "cuda": "cuda:0",
    "model_type": "auto",

    # Keep None to inherit the checkpoint train_scope.
    "eval_scope": None,

    # Use "yes" to auto-save a png result table under evaluation_plots/png_photos/,
    # or set an explicit path.
    "plot_output": "yes",
}
